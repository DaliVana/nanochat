"""
Triton kernels for Mamba-3 SSD (Structured State Space Duality).

Forward kernel: fused within-chunk computation for Two-SSD with MIMO. The decay
matrix L is NEVER materialized — computed tile-by-tile in registers. The R-loop
(MIMO rank) is fused inside the kernel. RoPE rotation is applied inline from
pre-computed cos/sin, avoiding both HBM round-trips for pre-rotated B/C and
expensive trig (SFU) ops inside the kernel. All dot products use BF16 tensor
cores with FP32 accumulators (same pattern as Flash Attention) for 2x throughput
over TF32, while RoPE rotation remains in FP32 for precision.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Forward kernel
# =============================================================================

@triton.jit
def _mamba3_chunk_scan_fwd_kernel(
    # Input tensors
    Bg_ptr, Bb_ptr,            # (batch, nc, L, ngroups, R, dstate)
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim) native dtype
    dA_cumsum_ptr,             # (batch, nc, L, nheads) float32
    C_ptr,                     # (batch, nc, L, ngroups, dstate)
    # Pre-computed cos/sin for RoPE: (batch, nc, L, half_dstate) float32
    cos_ang_ptr, sin_ang_ptr,      # cos/sin of cum_angles (for Bg and C)
    cos_angs_ptr, sin_angs_ptr,    # cos/sin of cum_angles_shifted (for Bb)
    out_ptr,                   # (batch, nc, L, nheads, headdim) float32
    # Strides for Bg/Bb (6D): batch, chunk, seqlen, group, rank, dstate
    stride_Bg_batch, stride_Bg_chunk, stride_Bg_seqlen, stride_Bg_group, stride_Bg_rank, stride_Bg_dstate,
    stride_Bb_batch, stride_Bb_chunk, stride_Bb_seqlen, stride_Bb_group, stride_Bb_rank, stride_Bb_dstate,
    # Strides for x_dt_g/x_dt_b (6D): batch, chunk, seqlen, head, rank, headdim
    stride_xg_batch, stride_xg_chunk, stride_xg_seqlen, stride_xg_head, stride_xg_rank, stride_xg_hdim,
    stride_xb_batch, stride_xb_chunk, stride_xb_seqlen, stride_xb_head, stride_xb_rank, stride_xb_hdim,
    # Strides for dA_cumsum (4D): batch, chunk, seqlen, head
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # Strides for C (5D): batch, chunk, seqlen, group, dstate
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # Strides for cos/sin (4D): batch, chunk, seqlen, half_dstate
    # cos_ang and sin_ang share strides; cos_angs and sin_angs share strides
    stride_ang_batch, stride_ang_chunk, stride_ang_seqlen, stride_ang_hdstate,
    stride_angs_batch, stride_angs_chunk, stride_angs_seqlen, stride_angs_hdstate,
    # Strides for out (5D): batch, chunk, seqlen, head, headdim
    stride_out_batch, stride_out_chunk, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # Dimensions (all tl.constexpr)
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, ngroups: tl.constexpr, nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr, mimo_rank: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO and fused RoPE.

    RoPE is applied inline: raw (unrotated) B and C are passed in, and rotation
    is computed from pre-computed cos/sin in registers. Trig (SFU) ops are done
    outside the kernel to avoid expensive per-tile cos/sin in the hot loop.

    With dstate=128 and BLOCK_DSTATE=64, the two C/B tiles map exactly to the
    two halves for RoPE: tile 0 = first half (x1), tile 1 = second half (x2).
    Rotation: rot_0 = x1*cos - x2*sin, rot_1 = x1*sin + x2*cos.

    y_intra[m,h,p] = scale * sum_{r} sum_{k<=m} (
        C_rot[m,g(h)] @ Bg_rot[k,g(h),r]^T * L[m,k,h] * x_dt_g[k,h,r,p]
      + C_rot[m,g(h)] @ Bb_rot[k,g(h),r]^T * L[m,k,h] * x_dt_b[k,h,r,p]
    )

    L[m,k,h] = exp(min(dA_cs[m,h] - dA_cs[k,h], 0)) for k<=m, 0 otherwise.
    L is computed tile-by-tile in registers, never stored to global memory.
    The R-loop is compile-time unrolled via tl.constexpr mimo_rank.
    """
    # Grid: (cdiv(L, BM) * cdiv(hdim, BN), batch * nchunks, nheads)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    pid_m = pid_mn // tl.cdiv(headdim, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(headdim, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    group_idx = pid_h // nheads_per_group

    # half_dstate = BLOCK_DSTATE when dstate = 2*BLOCK_DSTATE (our target config)
    half_dstate: tl.constexpr = dstate // 2

    # Base offsets for this (batch, chunk)
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_Bg = pid_b * stride_Bg_batch + pid_c * stride_Bg_chunk + group_idx * stride_Bg_group
    off_bc_Bb = pid_b * stride_Bb_batch + pid_c * stride_Bb_chunk + group_idx * stride_Bb_group
    off_bc_xg = pid_b * stride_xg_batch + pid_c * stride_xg_chunk
    off_bc_xb = pid_b * stride_xb_batch + pid_c * stride_xb_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk + group_idx * stride_C_group
    off_bc_ang = pid_b * stride_ang_batch + pid_c * stride_ang_chunk
    off_bc_angs = pid_b * stride_angs_batch + pid_c * stride_angs_chunk

    # Load dA_cumsum for rows m
    dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + pid_h * stride_dA_head,
                       mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    # Output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # === PRE-LOAD C tiles and apply RoPE rotation in registers ===
    # C is indexed by (m, group, d) — no k or r dependency.
    # With d_state=128, BLOCK_DSTATE=64: tile 0 = first half, tile 1 = second half.
    # RoPE rotation is applied to the cached C tiles using cum_angles.
    C_TILES: tl.constexpr = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE

    # d-state offset arrays for the two halves
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate

    # C and x_dt are LOADED as float32 for numerical stability (BF16 truncation
    # caused loss spikes in training). Bg/Bb are loaded in native dtype (BF16).
    # RoPE rotation is done in FP32 for precision. The dot products then cast
    # to BF16 to engage tensor cores (2x over TF32), with FP32 accumulators.
    c_half_0 = tl.load(
        C_ptr + off_bc_C
        + offs_m[:, None] * stride_C_seqlen
        + c_d0_offs[None, :] * stride_C_dstate,
        mask=(offs_m[:, None] < chunk_size) & c_d0_mask[None, :],
        other=0.0).to(tl.float32)

    c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
    c_d1_mask = c_d1_offs < dstate
    c_half_1 = tl.load(
        C_ptr + off_bc_C
        + offs_m[:, None] * stride_C_seqlen
        + c_d1_offs[None, :] * stride_C_dstate,
        mask=(offs_m[:, None] < chunk_size) & c_d1_mask[None, :],
        other=0.0).to(tl.float32)

    # Load pre-computed cos/sin for m-positions: (BLOCK_M, BLOCK_DSTATE) float32
    ang_d_offs = tl.arange(0, BLOCK_DSTATE)
    cs_m_mask = (offs_m[:, None] < chunk_size) & (ang_d_offs[None, :] < half_dstate)
    cos_m = tl.load(
        cos_ang_ptr + off_bc_ang
        + offs_m[:, None] * stride_ang_seqlen
        + ang_d_offs[None, :] * stride_ang_hdstate,
        mask=cs_m_mask, other=0.0)
    sin_m = tl.load(
        sin_ang_ptr + off_bc_ang
        + offs_m[:, None] * stride_ang_seqlen
        + ang_d_offs[None, :] * stride_ang_hdstate,
        mask=cs_m_mask, other=0.0)

    # Apply RoPE to C tiles: rot_0 = half_0*cos - half_1*sin, rot_1 = half_0*sin + half_1*cos
    c_tile_0 = c_half_0 * cos_m - c_half_1 * sin_m
    c_tile_1 = c_half_0 * sin_m + c_half_1 * cos_m

    # --- Intra-chunk: tile-by-tile over k positions ---
    # Causal cutoff: skip k-blocks where all k > max(m) in this tile.
    # pid_m is runtime, so k_max is non-constexpr (Triton uses while-loop).
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)
    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # === Decay computed in registers, NEVER stored to global memory ===
        dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                           mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))

        # Causal mask: only k <= m contributes
        causal = offs_m[:, None] >= k_offs[None, :]
        decay = tl.where(causal, decay, 0.0)  # (BLOCK_M, BLOCK_K)

        # Load pre-computed cos/sin for k-positions (shared across all R ranks)
        # Bg uses cos/sin from cum_angles[k], Bb uses cos/sin from cum_angles_shifted[k]
        cs_k_mask = k_mask[:, None] & (ang_d_offs[None, :] < half_dstate)
        cos_k = tl.load(
            cos_ang_ptr + off_bc_ang
            + k_offs[:, None] * stride_ang_seqlen
            + ang_d_offs[None, :] * stride_ang_hdstate,
            mask=cs_k_mask, other=0.0)
        sin_k = tl.load(
            sin_ang_ptr + off_bc_ang
            + k_offs[:, None] * stride_ang_seqlen
            + ang_d_offs[None, :] * stride_ang_hdstate,
            mask=cs_k_mask, other=0.0)

        cs_ks_mask = k_mask[:, None] & (ang_d_offs[None, :] < half_dstate)
        cos_k_s = tl.load(
            cos_angs_ptr + off_bc_angs
            + k_offs[:, None] * stride_angs_seqlen
            + ang_d_offs[None, :] * stride_angs_hdstate,
            mask=cs_ks_mask, other=0.0)
        sin_k_s = tl.load(
            sin_angs_ptr + off_bc_angs
            + k_offs[:, None] * stride_angs_seqlen
            + ang_d_offs[None, :] * stride_angs_hdstate,
            mask=cs_ks_mask, other=0.0)

        # C tiles (rotated) pre-loaded before k and r loops.
        # Each rank reuses the cached C tiles, avoiding redundant global loads.
        # With R=4, this eliminates 75% of C loads vs loading per (r, d) iteration.
        for r in range(mimo_rank):
            # Load both halves of Bg simultaneously for RoPE rotation
            bg_half0 = tl.load(
                Bg_ptr + off_bc_Bg
                + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank
                + c_d0_offs[None, :] * stride_Bg_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0).to(tl.float32)
            bg_half1 = tl.load(
                Bg_ptr + off_bc_Bg
                + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank
                + c_d1_offs[None, :] * stride_Bg_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                other=0.0).to(tl.float32)

            # Rotate Bg with cum_angles[k]
            bg_rot_0 = bg_half0 * cos_k - bg_half1 * sin_k
            bg_rot_1 = bg_half0 * sin_k + bg_half1 * cos_k

            # Load both halves of Bb simultaneously for RoPE rotation
            bb_half0 = tl.load(
                Bb_ptr + off_bc_Bb
                + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank
                + c_d0_offs[None, :] * stride_Bb_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0).to(tl.float32)
            bb_half1 = tl.load(
                Bb_ptr + off_bc_Bb
                + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank
                + c_d1_offs[None, :] * stride_Bb_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                other=0.0).to(tl.float32)

            # Rotate Bb with cum_angles_shifted[k] (position t-1 angles)
            bb_rot_0 = bb_half0 * cos_k_s - bb_half1 * sin_k_s
            bb_rot_1 = bb_half0 * sin_k_s + bb_half1 * cos_k_s

            # Dot products: C_rot @ B_rot^T = sum over both halves
            # BF16 tensor cores with FP32 accumulation (2x throughput over TF32).
            # Cast to BF16 BEFORE tl.trans to avoid Triton compilation issues.
            cb_g = tl.dot(c_tile_0.to(tl.bfloat16), tl.trans(bg_rot_0.to(tl.bfloat16)))
            cb_g = tl.dot(c_tile_1.to(tl.bfloat16), tl.trans(bg_rot_1.to(tl.bfloat16)), acc=cb_g)
            cb_b = tl.dot(c_tile_0.to(tl.bfloat16), tl.trans(bb_rot_0.to(tl.bfloat16)))
            cb_b = tl.dot(c_tile_1.to(tl.bfloat16), tl.trans(bb_rot_1.to(tl.bfloat16)), acc=cb_b)

            # Apply decay and scale: (BLOCK_M, BLOCK_K)
            scores_g = cb_g * decay * scale
            scores_b = cb_b * decay * scale

            # x_dt_g[k, h, r, p]: (BLOCK_K, BLOCK_N)
            xg_vals = tl.load(
                x_dt_g_ptr + off_bc_xg
                + k_offs[:, None] * stride_xg_seqlen
                + pid_h * stride_xg_head
                + r * stride_xg_rank
                + offs_n[None, :] * stride_xg_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0)

            # x_dt_b[k, h, r, p]: (BLOCK_K, BLOCK_N)
            xb_vals = tl.load(
                x_dt_b_ptr + off_bc_xb
                + k_offs[:, None] * stride_xb_seqlen
                + pid_h * stride_xb_head
                + r * stride_xb_rank
                + offs_n[None, :] * stride_xb_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0)

            # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            # BF16 tensor cores with FP32 accumulation via acc= parameter.
            # Same pattern as Flash Attention: BF16 products, FP32 running sum.
            acc = tl.dot(scores_g.to(tl.bfloat16), xg_vals.to(tl.bfloat16), acc=acc)
            acc = tl.dot(scores_b.to(tl.bfloat16), xb_vals.to(tl.bfloat16), acc=acc)

    # Store output
    off_bc_out = pid_b * stride_out_batch + pid_c * stride_out_chunk
    tl.store(out_ptr + off_bc_out
             + offs_m[:, None] * stride_out_seqlen
             + pid_h * stride_out_head
             + offs_n[None, :] * stride_out_hdim,
             acc,
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, dstate, num_stages=1):
    """Estimate peak SRAM usage in bytes for the mamba3 kernel with fused RoPE.

    Separates fixed allocations from loop-body loads that Triton
    double-buffers when num_stages > 1.
    """
    # Fixed: not affected by pipeline staging
    fixed = (BLOCK_M * BLOCK_N * 4             # acc (float32)
            + BLOCK_M * BLOCK_K * 4            # decay (float32)
            + BLOCK_M * dstate * 4             # c_rot tiles (float32, 2 halves)
            + 2 * BLOCK_M * BLOCK_DSTATE * 4   # cos_m, sin_m (float32)
            )
    # Staged: loop-body global loads, multiplied by num_stages
    # B loads are float32 (both halves needed simultaneously for RoPE rotation)
    staged = (4 * BLOCK_K * BLOCK_DSTATE * 4   # bg_half0, bg_half1, bb_half0, bb_half1 (float32)
            + 2 * BLOCK_K * BLOCK_N * 4        # xg, xb (float32)
            + 4 * BLOCK_K * BLOCK_DSTATE * 4   # cos_k, sin_k, cos_k_s, sin_k_s (float32)
            )
    return fixed + staged * num_stages


def _select_block_sizes(L, headdim, dstate, device=None):
    """Select block sizes and num_stages for the mamba3 kernel with fused RoPE.

    Fused RoPE requires exactly 2 d-state tiles mapping to the two halves.
    BLOCK_DSTATE = half_dstate ensures the two tiles split at the midpoint.

    Queries the actual device shared-memory limit (falls back to a
    conservative 228 KB if unavailable) and tries block-size / num_stages
    combinations in priority order until one fits.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, num_stages).
    """
    assert dstate % 2 == 0, f"Fused RoPE requires even dstate, got {dstate}"
    half_dstate = dstate // 2
    BLOCK_DSTATE = max(16, min(64, triton.next_power_of_2(half_dstate)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(L)))

    # Query actual device shared memory limit
    if device is not None:
        props = torch.cuda.get_device_properties(device)
        # Try opt-in limit first (Triton can request extended SRAM on Hopper+),
        # then fall back to the base per-block limit.
        sram_limit = getattr(props, 'max_shared_memory_per_block_optin', 0)
        if not sram_limit:
            sram_limit = getattr(props, 'shared_memory_per_block', 0)
        if not sram_limit:
            sram_limit = 228 * 1024
    else:
        sram_limit = 228 * 1024  # Conservative H100 default

    # Priority: BLOCK_M=128 is most impactful (covers full chunk),
    # BLOCK_N=128 is secondary (covers full headdim).
    BM_max = max(16, min(128, triton.next_power_of_2(L)))
    BN_max = max(16, min(128, triton.next_power_of_2(headdim)))
    BM_64 = max(16, min(64, triton.next_power_of_2(L)))
    BN_64 = max(16, min(64, triton.next_power_of_2(headdim)))

    block_candidates = [
        (BM_max, BN_max),    # Best: full chunk + full headdim
        (BM_max, BN_64),     # Good: full chunk, split headdim
        (BM_64, BN_max),     # OK: split chunk, full headdim
        (BM_64, BN_64),      # Conservative fallback
    ]
    # Prefer num_stages=2 (better latency hiding), fall back to 1
    for num_stages in (2, 1):
        for BLOCK_M, BLOCK_N in block_candidates:
            est = _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE,
                                 dstate, num_stages)
            if est <= sram_limit:
                return BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, num_stages

    return BM_64, BN_64, BLOCK_K, BLOCK_DSTATE, 1


def mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                          cos_ang, sin_ang, cos_angs, sin_angs,
                          scale, chunk_size, mimo_rank):
    """
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO and fused RoPE.

    Decay matrix L is never materialized. R-loop is fused inside the kernel.
    RoPE rotation is applied inline from pre-computed cos/sin (no trig in kernel).

    Args:
        Bg:        (batch, nchunks, chunk_size, ngroups, R, dstate)
        Bb:        (batch, nchunks, chunk_size, ngroups, R, dstate)
        x_dt_g:    (batch, nchunks, chunk_size, nheads, R, headdim) float32
        x_dt_b:    (batch, nchunks, chunk_size, nheads, R, headdim) float32
        dA_cumsum: (batch, nchunks, chunk_size, nheads) float32
        C:         (batch, nchunks, chunk_size, ngroups, dstate) float32
        cos_ang:   (batch, nchunks, chunk_size, half_dstate) float32 — cos(cum_angles)
        sin_ang:   (batch, nchunks, chunk_size, half_dstate) float32 — sin(cum_angles)
        cos_angs:  (batch, nchunks, chunk_size, half_dstate) float32 — cos(cum_angles_shifted)
        sin_angs:  (batch, nchunks, chunk_size, half_dstate) float32 — sin(cum_angles_shifted)
        scale:     float
        chunk_size: int
        mimo_rank: int
    Returns:
        y_intra: (batch, nchunks, chunk_size, nheads, headdim) float32
    """
    batch, nchunks, L, nheads, R, headdim = x_dt_g.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups

    y_intra = torch.empty(batch, nchunks, L, nheads, headdim,
                           device=x_dt_g.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, num_stages = _select_block_sizes(
        L, headdim, dstate, device=x_dt_g.device)
    assert dstate == 2 * BLOCK_DSTATE, (
        f"Fused RoPE requires dstate={dstate} == 2*BLOCK_DSTATE={2*BLOCK_DSTATE}. "
        f"The two d-state tiles must map to the two halves for RoPE rotation."
    )
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(headdim, BLOCK_N),
            batch * nchunks, nheads)

    _mamba3_chunk_scan_fwd_kernel[grid](
        Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
        cos_ang, sin_ang, cos_angs, sin_angs, y_intra,
        Bg.stride(0), Bg.stride(1), Bg.stride(2), Bg.stride(3), Bg.stride(4), Bg.stride(5),
        Bb.stride(0), Bb.stride(1), Bb.stride(2), Bb.stride(3), Bb.stride(4), Bb.stride(5),
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2), x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2), x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        cos_ang.stride(0), cos_ang.stride(1), cos_ang.stride(2), cos_ang.stride(3),
        cos_angs.stride(0), cos_angs.stride(1), cos_angs.stride(2), cos_angs.stride(3),
        y_intra.stride(0), y_intra.stride(1), y_intra.stride(2), y_intra.stride(3), y_intra.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, mimo_rank=R, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8,
        num_stages=num_stages,
    )
    return y_intra
