"""
Triton kernels for Mamba-3 SSD (Structured State Space Duality).

Fused within-chunk kernel for Two-SSD with MIMO. The decay matrix L is NEVER
materialized — computed tile-by-tile in registers. The R-loop (MIMO rank) is
fused inside the kernel.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Mamba-3: Fused within-chunk kernel (Two-SSD with MIMO)
# =============================================================================
# Replaces the PyTorch within-chunk computation in Mamba3Layer._two_ssd_forward.
# The decay matrix L is NEVER materialized — computed tile-by-tile in registers.
# The R-loop (MIMO rank) is fused inside the kernel.

@triton.jit
def _mamba3_chunk_scan_fwd_kernel(
    # Input tensors
    Bg_ptr, Bb_ptr,            # (batch, nc, L, ngroups, R, dstate)
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim) native dtype
    dA_cumsum_ptr,             # (batch, nc, L, nheads) float32
    C_ptr,                     # (batch, nc, L, ngroups, dstate)
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
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO.

    y_intra[m,h,p] = scale * sum_{r} sum_{k<=m} (
        C[m,g(h)] @ Bg[k,g(h),r]^T * L[m,k,h] * x_dt_g[k,h,r,p]
      + C[m,g(h)] @ Bb[k,g(h),r]^T * L[m,k,h] * x_dt_b[k,h,r,p]
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

    # Base offsets for this (batch, chunk)
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_Bg = pid_b * stride_Bg_batch + pid_c * stride_Bg_chunk + group_idx * stride_Bg_group
    off_bc_Bb = pid_b * stride_Bb_batch + pid_c * stride_Bb_chunk + group_idx * stride_Bb_group
    off_bc_xg = pid_b * stride_xg_batch + pid_c * stride_xg_chunk
    off_bc_xb = pid_b * stride_xb_batch + pid_c * stride_xb_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk + group_idx * stride_C_group

    # Load dA_cumsum for rows m
    dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + pid_h * stride_dA_head,
                       mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    # Output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # === PRE-LOAD C tiles into registers (independent of k and r) ===
    # C is indexed by (m, group, d) — no k or r dependency.
    # With d_state=128, BLOCK_DSTATE=64: 2 cached tiles; d_state=64: 1 tile.
    # Avoids reloading C in every (k, r, d) iteration.
    # We store tiles as a flat register "array" indexed by d_start // BLOCK_DSTATE.
    # For typical configs (dstate <= 2*BLOCK_DSTATE), we unroll manually.
    C_TILES: tl.constexpr = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE

    # Tile 0 (always present)
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate
    # C and x_dt are float32 for numerical stability (BF16 truncation caused loss
    # spikes in training). Bg/Bb remain in native dtype (BF16).
    c_tile_0 = tl.load(
        C_ptr + off_bc_C
        + offs_m[:, None] * stride_C_seqlen
        + c_d0_offs[None, :] * stride_C_dstate,
        mask=(offs_m[:, None] < chunk_size) & c_d0_mask[None, :],
        other=0.0)

    # Tile 1 (when dstate > BLOCK_DSTATE, e.g. dstate=128)
    if C_TILES > 1:
        c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
        c_d1_mask = c_d1_offs < dstate
        c_tile_1 = tl.load(
            C_ptr + off_bc_C
            + offs_m[:, None] * stride_C_seqlen
            + c_d1_offs[None, :] * stride_C_dstate,
            mask=(offs_m[:, None] < chunk_size) & c_d1_mask[None, :],
            other=0.0)

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

        # C tiles pre-loaded into registers before the k and r loops.
        # Each rank reuses the cached C tiles, avoiding redundant global loads.
        # With R=4, this eliminates 75% of C loads vs loading per (r, d) iteration.
        for r in range(mimo_rank):
            cb_g = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            cb_b = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            # d-tile 0: use cached c_tile_0
            # Bg/Bb stay in native dtype (matching C) for BF16 tensor core dot products.
            bg_vals = tl.load(
                Bg_ptr + off_bc_Bg
                + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank
                + c_d0_offs[None, :] * stride_Bg_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            bb_vals = tl.load(
                Bb_ptr + off_bc_Bb
                + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank
                + c_d0_offs[None, :] * stride_Bb_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            cb_g += tl.dot(c_tile_0, tl.trans(bg_vals))
            cb_b += tl.dot(c_tile_0, tl.trans(bb_vals))

            # d-tile 1 (when dstate > BLOCK_DSTATE)
            if C_TILES > 1:
                bg_vals = tl.load(
                    Bg_ptr + off_bc_Bg
                    + k_offs[:, None] * stride_Bg_seqlen
                    + r * stride_Bg_rank
                    + c_d1_offs[None, :] * stride_Bg_dstate,
                    mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                    other=0.0)
                bb_vals = tl.load(
                    Bb_ptr + off_bc_Bb
                    + k_offs[:, None] * stride_Bb_seqlen
                    + r * stride_Bb_rank
                    + c_d1_offs[None, :] * stride_Bb_dstate,
                    mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                    other=0.0)
                cb_g += tl.dot(c_tile_1, tl.trans(bg_vals))
                cb_b += tl.dot(c_tile_1, tl.trans(bb_vals))

            # Apply decay and scale: (BLOCK_M, BLOCK_K)
            scores_g = cb_g * decay * scale
            scores_b = cb_b * decay * scale

            # x_dt_g[k, h, r, p]: (BLOCK_K, BLOCK_N) — native dtype for BF16 tensor core dots
            xg_vals = tl.load(
                x_dt_g_ptr + off_bc_xg
                + k_offs[:, None] * stride_xg_seqlen
                + pid_h * stride_xg_head
                + r * stride_xg_rank
                + offs_n[None, :] * stride_xg_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0)

            # x_dt_b[k, h, r, p]: (BLOCK_K, BLOCK_N) — native dtype for BF16 tensor core dots
            xb_vals = tl.load(
                x_dt_b_ptr + off_bc_xb
                + k_offs[:, None] * stride_xb_seqlen
                + pid_h * stride_xb_head
                + r * stride_xb_rank
                + offs_n[None, :] * stride_xb_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0)

            # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            # scores and x_dt are both float32 — FP32 dot for numerical stability.
            acc += tl.dot(scores_g.to(xg_vals.dtype), xg_vals)
            acc += tl.dot(scores_b.to(xb_vals.dtype), xb_vals)

    # Store output
    off_bc_out = pid_b * stride_out_batch + pid_c * stride_out_chunk
    tl.store(out_ptr + off_bc_out
             + offs_m[:, None] * stride_out_seqlen
             + pid_h * stride_out_head
             + offs_n[None, :] * stride_out_hdim,
             acc,
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, dstate, num_stages=1):
    """Estimate peak SRAM usage in bytes for the mamba3 kernel.

    Separates fixed allocations from loop-body loads that Triton
    double-buffers when num_stages > 1.
    """
    # Fixed: not affected by pipeline staging
    fixed = (BLOCK_M * BLOCK_N * 4             # acc (float32)
            + 2 * BLOCK_M * BLOCK_K * 4       # cb_g, cb_b (float32)
            + BLOCK_M * BLOCK_K * 4            # decay (float32)
            + BLOCK_M * dstate * 4             # c_cache (float32)
            )
    # Staged: loop-body global loads, multiplied by num_stages
    staged = (4 * BLOCK_K * BLOCK_DSTATE * 2   # bg, bb x2 d-tiles (BF16)
            + 2 * BLOCK_K * BLOCK_N * 4        # xg, xb (float32)
            )
    return fixed + staged * num_stages


def _select_block_sizes(L, headdim, dstate, device=None):
    """Select block sizes and num_stages for the mamba3 kernel.

    Queries the actual device shared-memory limit (falls back to a
    conservative 228 KB if unavailable) and tries block-size / num_stages
    combinations in priority order until one fits.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, num_stages).
    """
    BLOCK_DSTATE = max(16, min(64, triton.next_power_of_2(dstate)))
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


def mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C, scale, chunk_size, mimo_rank):
    """
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO.

    Decay matrix L is never materialized. R-loop is fused inside the kernel.

    Args:
        Bg:         (batch, nchunks, chunk_size, ngroups, R, dstate)
        Bb:         (batch, nchunks, chunk_size, ngroups, R, dstate)
        x_dt_g:     (batch, nchunks, chunk_size, nheads, R, headdim) float32
        x_dt_b:     (batch, nchunks, chunk_size, nheads, R, headdim) float32
        dA_cumsum:  (batch, nchunks, chunk_size, nheads) float32
        C:          (batch, nchunks, chunk_size, ngroups, dstate)
        scale:      float
        chunk_size: int
        mimo_rank:  int
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
    assert dstate <= 2 * BLOCK_DSTATE, (
        f"dstate={dstate} exceeds 2*BLOCK_DSTATE={2*BLOCK_DSTATE}. "
        f"C tile pre-loading supports at most 2 tiles. "
        f"Increase BLOCK_DSTATE or extend kernel for more tiles."
    )
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(headdim, BLOCK_N),
            batch * nchunks, nheads)

    _mamba3_chunk_scan_fwd_kernel[grid](
        Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C, y_intra,
        Bg.stride(0), Bg.stride(1), Bg.stride(2), Bg.stride(3), Bg.stride(4), Bg.stride(5),
        Bb.stride(0), Bb.stride(1), Bb.stride(2), Bb.stride(3), Bb.stride(4), Bb.stride(5),
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2), x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2), x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        y_intra.stride(0), y_intra.stride(1), y_intra.stride(2), y_intra.stride(3), y_intra.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, mimo_rank=R, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8,
        num_stages=num_stages,
    )
    return y_intra


# =============================================================================
# Backward kernels
# =============================================================================

@triton.jit
def _mamba3_chunk_scan_bwd_dx_kernel(
    # Saved from forward
    Bg_ptr, Bb_ptr,            # (batch, nc, L, ngroups, R, dstate)
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim)
    dA_cumsum_ptr,             # (batch, nc, L, nheads)
    C_ptr,                     # (batch, nc, L, ngroups, dstate)
    # Upstream gradient
    dy_ptr,                    # (batch, nc, L, nheads, headdim)
    # Outputs
    dx_dt_g_ptr, dx_dt_b_ptr, # (batch, nc, L, nheads, R, headdim)
    dBg_ptr, dBb_ptr,         # (batch, nc, L, ngroups, R, dstate) — atomic add
    ddA_k_ptr,                # (batch, nc, L, nheads) — k-side contribution
    # Strides Bg/Bb
    stride_Bg_batch, stride_Bg_chunk, stride_Bg_seqlen, stride_Bg_group, stride_Bg_rank, stride_Bg_dstate,
    stride_Bb_batch, stride_Bb_chunk, stride_Bb_seqlen, stride_Bb_group, stride_Bb_rank, stride_Bb_dstate,
    # Strides x_dt
    stride_xg_batch, stride_xg_chunk, stride_xg_seqlen, stride_xg_head, stride_xg_rank, stride_xg_hdim,
    stride_xb_batch, stride_xb_chunk, stride_xb_seqlen, stride_xb_head, stride_xb_rank, stride_xb_hdim,
    # Strides dA_cumsum
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # Strides C
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # Strides dy (same layout as y_intra: batch, nc, L, nheads, headdim)
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    # Strides dx_dt output (same layout as x_dt input)
    stride_dxg_batch, stride_dxg_chunk, stride_dxg_seqlen, stride_dxg_head, stride_dxg_rank, stride_dxg_hdim,
    stride_dxb_batch, stride_dxb_chunk, stride_dxb_seqlen, stride_dxb_head, stride_dxb_rank, stride_dxb_hdim,
    # Strides dBg/dBb output (same layout as Bg/Bb)
    stride_dBg_batch, stride_dBg_chunk, stride_dBg_seqlen, stride_dBg_group, stride_dBg_rank, stride_dBg_dstate,
    stride_dBb_batch, stride_dBb_chunk, stride_dBb_seqlen, stride_dBb_group, stride_dBb_rank, stride_dBb_dstate,
    # Strides ddA output
    stride_ddA_batch, stride_ddA_chunk, stride_ddA_seqlen, stride_ddA_head,
    # Dimensions
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, ngroups: tl.constexpr, nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr, mimo_rank: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """
    Backward kernel 2: computes dx_dt_g, dx_dt_b, dBg, dBb, ddA_cumsum (k-side).

    Transposed forward: for each k-block, loop over m >= k.
    dx_dt_g[k,h,r,p] = Σ_{m>=k} scores_g[m,k,h,r] * dy[m,h,p]
    dBg[k,g,r,d] += Σ_{m>=k} dcb_g[m,k] * C[m,g,d]  (atomic across heads in group)
    ddA_k[k,h] = -Σ_{m>=k} dL * L_active
    """
    # Grid: (cdiv(L, BLOCK_K) * cdiv(headdim, BLOCK_N), batch * nchunks, nheads)
    pid_kn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    N_hdim_blocks: tl.constexpr = (headdim + BLOCK_N - 1) // BLOCK_N
    pid_k = pid_kn // N_hdim_blocks
    pid_n = pid_kn % N_hdim_blocks

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    group_idx = pid_h // nheads_per_group

    # Base offsets
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_Bg = pid_b * stride_Bg_batch + pid_c * stride_Bg_chunk + group_idx * stride_Bg_group
    off_bc_Bb = pid_b * stride_Bb_batch + pid_c * stride_Bb_chunk + group_idx * stride_Bb_group
    off_bc_xg = pid_b * stride_xg_batch + pid_c * stride_xg_chunk
    off_bc_xb = pid_b * stride_xb_batch + pid_c * stride_xb_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk + group_idx * stride_C_group
    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk

    # Load dA_cumsum for k positions (persistent across m-loop)
    dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + offs_k * stride_dA_seqlen + pid_h * stride_dA_head,
                       mask=offs_k < chunk_size, other=0.0).to(tl.float32)

    C_TILES: tl.constexpr = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate
    if C_TILES > 1:
        c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
        c_d1_mask = c_d1_offs < dstate

    # Pre-load Bg/Bb[k] tiles (persistent — k is fixed, only m varies)
    # We load per-rank in the r-loop, but the k-indexed data is reused across m.

    # ddA_k accumulator: persists across ranks (small — BLOCK_K vector)
    ddA_k_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # --- Accumulators (per rank, written after m-loop) ---
    # We loop over ranks serially to save registers.
    for r in range(mimo_rank):
        dx_g_acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        dx_b_acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        # Pre-load Bg[k,g,r,:] and Bb[k,g,r,:] for this rank (k-indexed, persistent)
        bg_k_0 = tl.load(
            Bg_ptr + off_bc_Bg + offs_k[:, None] * stride_Bg_seqlen
            + r * stride_Bg_rank + c_d0_offs[None, :] * stride_Bg_dstate,
            mask=(offs_k[:, None] < chunk_size) & c_d0_mask[None, :], other=0.0)
        bb_k_0 = tl.load(
            Bb_ptr + off_bc_Bb + offs_k[:, None] * stride_Bb_seqlen
            + r * stride_Bb_rank + c_d0_offs[None, :] * stride_Bb_dstate,
            mask=(offs_k[:, None] < chunk_size) & c_d0_mask[None, :], other=0.0)
        if C_TILES > 1:
            bg_k_1 = tl.load(
                Bg_ptr + off_bc_Bg + offs_k[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank + c_d1_offs[None, :] * stride_Bg_dstate,
                mask=(offs_k[:, None] < chunk_size) & c_d1_mask[None, :], other=0.0)
            bb_k_1 = tl.load(
                Bb_ptr + off_bc_Bb + offs_k[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank + c_d1_offs[None, :] * stride_Bb_dstate,
                mask=(offs_k[:, None] < chunk_size) & c_d1_mask[None, :], other=0.0)

        # Inner loop: m from k-block start to chunk_size (m >= k)
        for m_start in range(pid_k * BLOCK_K, chunk_size, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_offs < chunk_size

            # Decay: L[m,k,h] = exp(min(dA_cs[m] - dA_cs[k], 0)) * (m >= k)
            dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + m_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                               mask=m_mask, other=0.0).to(tl.float32)
            diff = dA_cs_m[:, None] - dA_cs_k[None, :]   # (BLOCK_M, BLOCK_K)
            decay = tl.exp(tl.minimum(diff, 0.0))
            causal = m_offs[:, None] >= offs_k[None, :]
            decay = tl.where(causal, decay, 0.0)

            # Load C[m,g,:] and compute cb = C[m] @ Bg[k]^T -> (BLOCK_M, BLOCK_K)
            c_m_0 = tl.load(
                C_ptr + off_bc_C + m_offs[:, None] * stride_C_seqlen + c_d0_offs[None, :] * stride_C_dstate,
                mask=(m_offs[:, None] < chunk_size) & c_d0_mask[None, :], other=0.0)
            cb_g = tl.dot(c_m_0, tl.trans(bg_k_0))
            cb_b = tl.dot(c_m_0, tl.trans(bb_k_0))
            if C_TILES > 1:
                c_m_1 = tl.load(
                    C_ptr + off_bc_C + m_offs[:, None] * stride_C_seqlen + c_d1_offs[None, :] * stride_C_dstate,
                    mask=(m_offs[:, None] < chunk_size) & c_d1_mask[None, :], other=0.0)
                cb_g += tl.dot(c_m_1, tl.trans(bg_k_1))
                cb_b += tl.dot(c_m_1, tl.trans(bb_k_1))

            scores_g = cb_g * decay * scale   # (BLOCK_M, BLOCK_K)
            scores_b = cb_b * decay * scale

            # Load dy[m,h,:] -> (BLOCK_M, BLOCK_N)
            dy_vals = tl.load(
                dy_ptr + off_bc_dy + m_offs[:, None] * stride_dy_seqlen
                + pid_h * stride_dy_head + offs_n[None, :] * stride_dy_hdim,
                mask=m_mask[:, None] & (offs_n[None, :] < headdim), other=0.0)

            # dx_dt_g[k,h,r,p] += scores_g^T @ dy  -> (BLOCK_K, BLOCK_N)
            dx_g_acc += tl.dot(tl.trans(scores_g), dy_vals)
            dx_b_acc += tl.dot(tl.trans(scores_b), dy_vals)

            # --- dBg/dBb/ddA: only from pid_n==0 (needs full headdim reduction) ---
            if pid_n == 0:
                # Full dscores: Σ_p dy[m,h,p] * x_dt[k,h,r,p] over ALL headdim
                dscores_g = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
                dscores_b = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
                for hdim_start in range(0, headdim, BLOCK_N):
                    h_offs = hdim_start + tl.arange(0, BLOCK_N)
                    h_mask = h_offs < headdim
                    dy_full = tl.load(
                        dy_ptr + off_bc_dy + m_offs[:, None] * stride_dy_seqlen
                        + pid_h * stride_dy_head + h_offs[None, :] * stride_dy_hdim,
                        mask=m_mask[:, None] & h_mask[None, :], other=0.0)
                    xg_full = tl.load(
                        x_dt_g_ptr + off_bc_xg + offs_k[:, None] * stride_xg_seqlen
                        + pid_h * stride_xg_head + r * stride_xg_rank + h_offs[None, :] * stride_xg_hdim,
                        mask=(offs_k[:, None] < chunk_size) & h_mask[None, :], other=0.0)
                    xb_full = tl.load(
                        x_dt_b_ptr + off_bc_xb + offs_k[:, None] * stride_xb_seqlen
                        + pid_h * stride_xb_head + r * stride_xb_rank + h_offs[None, :] * stride_xb_hdim,
                        mask=(offs_k[:, None] < chunk_size) & h_mask[None, :], other=0.0)
                    dscores_g += tl.dot(dy_full, tl.trans(xg_full))
                    dscores_b += tl.dot(dy_full, tl.trans(xb_full))

                # dcb = dscores * L * scale
                dcb_g = dscores_g * decay * scale
                dcb_b = dscores_b * decay * scale

                # dBg[k,g,r,d] += dcb_g^T @ C[m,g,d]  (atomic across heads)
                off_dBg = (pid_b * stride_dBg_batch + pid_c * stride_dBg_chunk
                           + group_idx * stride_dBg_group + r * stride_dBg_rank)
                off_dBb = (pid_b * stride_dBb_batch + pid_c * stride_dBb_chunk
                           + group_idx * stride_dBb_group + r * stride_dBb_rank)
                dBg_c0 = tl.dot(tl.trans(dcb_g), c_m_0)
                tl.atomic_add(
                    dBg_ptr + off_dBg + offs_k[:, None] * stride_dBg_seqlen + c_d0_offs[None, :] * stride_dBg_dstate,
                    dBg_c0, mask=(offs_k[:, None] < chunk_size) & c_d0_mask[None, :])
                dBb_c0 = tl.dot(tl.trans(dcb_b), c_m_0)
                tl.atomic_add(
                    dBb_ptr + off_dBb + offs_k[:, None] * stride_dBb_seqlen + c_d0_offs[None, :] * stride_dBb_dstate,
                    dBb_c0, mask=(offs_k[:, None] < chunk_size) & c_d0_mask[None, :])
                if C_TILES > 1:
                    dBg_c1 = tl.dot(tl.trans(dcb_g), c_m_1)
                    tl.atomic_add(
                        dBg_ptr + off_dBg + offs_k[:, None] * stride_dBg_seqlen + c_d1_offs[None, :] * stride_dBg_dstate,
                        dBg_c1, mask=(offs_k[:, None] < chunk_size) & c_d1_mask[None, :])
                    dBb_c1 = tl.dot(tl.trans(dcb_b), c_m_1)
                    tl.atomic_add(
                        dBb_ptr + off_dBb + offs_k[:, None] * stride_dBb_seqlen + c_d1_offs[None, :] * stride_dBb_dstate,
                        dBb_c1, mask=(offs_k[:, None] < chunk_size) & c_d1_mask[None, :])

                # ddA_cumsum k-side: ddA_k[k] -= Σ_m dL * L_active
                L_active = tl.where(diff < 0.0, decay, 0.0)
                dL_r = scale * (dscores_g * cb_g + dscores_b * cb_b)
                ddA_k_acc -= tl.sum(dL_r * L_active, axis=0)

        # Store dx_dt_g[k,h,r,:] and dx_dt_b[k,h,r,:]
        off_dxg = (pid_b * stride_dxg_batch + pid_c * stride_dxg_chunk
                   + pid_h * stride_dxg_head + r * stride_dxg_rank)
        tl.store(dx_dt_g_ptr + off_dxg + offs_k[:, None] * stride_dxg_seqlen + offs_n[None, :] * stride_dxg_hdim,
                 dx_g_acc,
                 mask=(offs_k[:, None] < chunk_size) & (offs_n[None, :] < headdim))
        off_dxb = (pid_b * stride_dxb_batch + pid_c * stride_dxb_chunk
                   + pid_h * stride_dxb_head + r * stride_dxb_rank)
        tl.store(dx_dt_b_ptr + off_dxb + offs_k[:, None] * stride_dxb_seqlen + offs_n[None, :] * stride_dxb_hdim,
                 dx_b_acc,
                 mask=(offs_k[:, None] < chunk_size) & (offs_n[None, :] < headdim))

    # Store ddA_k once after all ranks (only on first headdim block)
    if pid_n == 0:
        tl.store(ddA_k_ptr + off_bc_dA + offs_k * stride_ddA_seqlen + pid_h * stride_ddA_head,
                 ddA_k_acc, mask=offs_k < chunk_size)


@triton.jit
def _mamba3_chunk_scan_bwd_dC_kernel(
    # Saved from forward
    Bg_ptr, Bb_ptr,            # (batch, nc, L, ngroups, R, dstate)
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim)
    dA_cumsum_ptr,             # (batch, nc, L, nheads)
    C_ptr,                     # (batch, nc, L, ngroups, dstate)
    # Upstream gradient
    dy_ptr,                    # (batch, nc, L, nheads, headdim)
    # Outputs
    dC_ptr,                    # (batch, nc, L, ngroups, dstate)
    ddA_m_ptr,                 # (batch, nc, L, nheads) — m-side, zero-init, atomic add
    # Strides Bg/Bb
    stride_Bg_batch, stride_Bg_chunk, stride_Bg_seqlen, stride_Bg_group, stride_Bg_rank, stride_Bg_dstate,
    stride_Bb_batch, stride_Bb_chunk, stride_Bb_seqlen, stride_Bb_group, stride_Bb_rank, stride_Bb_dstate,
    # Strides x_dt
    stride_xg_batch, stride_xg_chunk, stride_xg_seqlen, stride_xg_head, stride_xg_rank, stride_xg_hdim,
    stride_xb_batch, stride_xb_chunk, stride_xb_seqlen, stride_xb_head, stride_xb_rank, stride_xb_hdim,
    # Strides dA_cumsum
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # Strides C
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # Strides dy
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    # Strides dC output (same layout as C)
    stride_dC_batch, stride_dC_chunk, stride_dC_seqlen, stride_dC_group, stride_dC_dstate,
    # Strides ddA_m output (same layout as dA_cumsum)
    stride_ddAm_batch, stride_ddAm_chunk, stride_ddAm_seqlen, stride_ddAm_head,
    # Dimensions
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, ngroups: tl.constexpr, nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr, mimo_rank: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """
    Backward kernel 1: computes dC and ddA_cumsum (m-side).

    Grid: (cdiv(L, BLOCK_M), batch * nchunks, ngroups)
    For each m-block, loops over k <= m, ranks, and heads within group.

    dC[m,g,d] = Σ_{k≤m} Σ_r (dcb_g[m,k,g,r] · Bg[k,g,r,d] + dcb_b · Bb)
    where dcb_g[m,k,g,r] = Σ_{h∈g} dscores_g[m,k,h,r] * L[m,k,h] * scale

    ddA_m[m,h] = +Σ_{k≤m} Σ_r dL_r * L_active   (m-side of ddA_cumsum)
    """
    pid_m = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_g = tl.program_id(2)

    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < chunk_size

    # Base offsets for this (batch, chunk, group)
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_Bg = pid_b * stride_Bg_batch + pid_c * stride_Bg_chunk + pid_g * stride_Bg_group
    off_bc_Bb = pid_b * stride_Bb_batch + pid_c * stride_Bb_chunk + pid_g * stride_Bb_group
    off_bc_xg = pid_b * stride_xg_batch + pid_c * stride_xg_chunk
    off_bc_xb = pid_b * stride_xb_batch + pid_c * stride_xb_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk + pid_g * stride_C_group
    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk

    # Pre-load C[m, g, :] tiles (persistent — m is fixed for this program)
    C_TILES: tl.constexpr = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate
    c_m_0 = tl.load(
        C_ptr + off_bc_C + offs_m[:, None] * stride_C_seqlen + c_d0_offs[None, :] * stride_C_dstate,
        mask=m_mask[:, None] & c_d0_mask[None, :], other=0.0)
    if C_TILES > 1:
        c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
        c_d1_mask = c_d1_offs < dstate
        c_m_1 = tl.load(
            C_ptr + off_bc_C + offs_m[:, None] * stride_C_seqlen + c_d1_offs[None, :] * stride_C_dstate,
            mask=m_mask[:, None] & c_d1_mask[None, :], other=0.0)

    # dC accumulators (one per dstate tile)
    dC_acc_0 = tl.zeros((BLOCK_M, BLOCK_DSTATE), dtype=tl.float32)
    if C_TILES > 1:
        dC_acc_1 = tl.zeros((BLOCK_M, BLOCK_DSTATE), dtype=tl.float32)

    # k-loop: k ≤ m (causal, same direction as forward)
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)
    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size
        causal = offs_m[:, None] >= k_offs[None, :]  # (BLOCK_M, BLOCK_K)

        for r in range(mimo_rank):
            # Load Bg[k,g,r,:] and Bb[k,g,r,:] for this rank
            bg_k_0 = tl.load(
                Bg_ptr + off_bc_Bg + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank + c_d0_offs[None, :] * stride_Bg_dstate,
                mask=k_mask[:, None] & c_d0_mask[None, :], other=0.0)
            bb_k_0 = tl.load(
                Bb_ptr + off_bc_Bb + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank + c_d0_offs[None, :] * stride_Bb_dstate,
                mask=k_mask[:, None] & c_d0_mask[None, :], other=0.0)
            if C_TILES > 1:
                bg_k_1 = tl.load(
                    Bg_ptr + off_bc_Bg + k_offs[:, None] * stride_Bg_seqlen
                    + r * stride_Bg_rank + c_d1_offs[None, :] * stride_Bg_dstate,
                    mask=k_mask[:, None] & c_d1_mask[None, :], other=0.0)
                bb_k_1 = tl.load(
                    Bb_ptr + off_bc_Bb + k_offs[:, None] * stride_Bb_seqlen
                    + r * stride_Bb_rank + c_d1_offs[None, :] * stride_Bb_dstate,
                    mask=k_mask[:, None] & c_d1_mask[None, :], other=0.0)

            # cb = C[m,g] @ B[k,g,r]^T -> (BLOCK_M, BLOCK_K)
            cb_g = tl.dot(c_m_0, tl.trans(bg_k_0))
            cb_b = tl.dot(c_m_0, tl.trans(bb_k_0))
            if C_TILES > 1:
                cb_g += tl.dot(c_m_1, tl.trans(bg_k_1))
                cb_b += tl.dot(c_m_1, tl.trans(bb_k_1))

            # dcb = Σ_{h∈g} dscores * decay * scale, accumulated over heads in group
            dcb_g = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            dcb_b = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for h_idx in range(nheads_per_group):
                h = pid_g * nheads_per_group + h_idx

                # Decay: L[m,k,h] = exp(min(dA_cs_m[h] - dA_cs_k[h], 0)) * (m >= k)
                dA_cs_m_h = tl.load(
                    dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + h * stride_dA_head,
                    mask=m_mask, other=0.0).to(tl.float32)
                dA_cs_k_h = tl.load(
                    dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + h * stride_dA_head,
                    mask=k_mask, other=0.0).to(tl.float32)
                diff = dA_cs_m_h[:, None] - dA_cs_k_h[None, :]
                decay = tl.exp(tl.minimum(diff, 0.0))
                decay = tl.where(causal, decay, 0.0)

                # dscores[m,k,h,r] = Σ_p dy[m,h,p] * x_dt[k,h,r,p] (full headdim reduction)
                dscores_g = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
                dscores_b = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
                for hdim_start in range(0, headdim, BLOCK_N):
                    hd_offs = hdim_start + tl.arange(0, BLOCK_N)
                    hd_mask = hd_offs < headdim
                    dy_h = tl.load(
                        dy_ptr + off_bc_dy + offs_m[:, None] * stride_dy_seqlen
                        + h * stride_dy_head + hd_offs[None, :] * stride_dy_hdim,
                        mask=m_mask[:, None] & hd_mask[None, :], other=0.0)
                    xg_h = tl.load(
                        x_dt_g_ptr + off_bc_xg + k_offs[:, None] * stride_xg_seqlen
                        + h * stride_xg_head + r * stride_xg_rank + hd_offs[None, :] * stride_xg_hdim,
                        mask=k_mask[:, None] & hd_mask[None, :], other=0.0)
                    xb_h = tl.load(
                        x_dt_b_ptr + off_bc_xb + k_offs[:, None] * stride_xb_seqlen
                        + h * stride_xb_head + r * stride_xb_rank + hd_offs[None, :] * stride_xb_hdim,
                        mask=k_mask[:, None] & hd_mask[None, :], other=0.0)
                    dscores_g += tl.dot(dy_h, tl.trans(xg_h))
                    dscores_b += tl.dot(dy_h, tl.trans(xb_h))

                # Accumulate dcb across heads in group
                dcb_g += dscores_g * decay * scale
                dcb_b += dscores_b * decay * scale

                # ddA_m[m,h]: dL_r * L_active, summed over k dim — local accumulation
                L_active = tl.where(diff < 0.0, decay, 0.0)
                dL_r = scale * (dscores_g * cb_g + dscores_b * cb_b)
                ddA_m_contrib = tl.sum(dL_r * L_active, axis=1)  # (BLOCK_M,)
                # No cross-program contention: each (pid_m, pid_g) writes to
                # distinct (m, h) positions. Use load-add-store instead of atomic.
                prev_ddA = tl.load(
                    ddA_m_ptr + pid_b * stride_ddAm_batch + pid_c * stride_ddAm_chunk
                    + offs_m * stride_ddAm_seqlen + h * stride_ddAm_head,
                    mask=m_mask, other=0.0)
                tl.store(
                    ddA_m_ptr + pid_b * stride_ddAm_batch + pid_c * stride_ddAm_chunk
                    + offs_m * stride_ddAm_seqlen + h * stride_ddAm_head,
                    prev_ddA + ddA_m_contrib, mask=m_mask)

            # dC += dcb_g @ Bg_k + dcb_b @ Bb_k  ->  (BLOCK_M, BLOCK_DSTATE)
            dC_acc_0 += tl.dot(dcb_g, bg_k_0) + tl.dot(dcb_b, bb_k_0)
            if C_TILES > 1:
                dC_acc_1 += tl.dot(dcb_g, bg_k_1) + tl.dot(dcb_b, bb_k_1)

    # Store dC[m, g, :]
    off_dC = pid_b * stride_dC_batch + pid_c * stride_dC_chunk + pid_g * stride_dC_group
    tl.store(dC_ptr + off_dC + offs_m[:, None] * stride_dC_seqlen + c_d0_offs[None, :] * stride_dC_dstate,
             dC_acc_0, mask=m_mask[:, None] & c_d0_mask[None, :])
    if C_TILES > 1:
        tl.store(dC_ptr + off_dC + offs_m[:, None] * stride_dC_seqlen + c_d1_offs[None, :] * stride_dC_dstate,
                 dC_acc_1, mask=m_mask[:, None] & c_d1_mask[None, :])


# =============================================================================
# Backward: SRAM estimation and block size selection
# =============================================================================

def _sram_estimate_bwd_dx(BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE, dstate):
    """Estimate peak SRAM for Kernel 2 (dx/dB/ddA_k)."""
    C_TILES = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE
    # Persistent per rank: Bg/Bb k-tiles (BF16)
    bg_bb = 2 * C_TILES * BLOCK_K * BLOCK_DSTATE * 2
    # Accumulators: dx_g, dx_b
    dx_acc = 2 * BLOCK_K * BLOCK_N * 4
    # Per m-iteration: decay, diff, cb_g/cb_b, C tiles, dy
    decay = BLOCK_M * BLOCK_K * 4
    diff = BLOCK_M * BLOCK_K * 4
    cb = 2 * BLOCK_M * BLOCK_K * 4
    c_tiles = BLOCK_M * dstate * 4
    dy = BLOCK_M * BLOCK_N * 4
    # pid_n==0 path (worst case): dscores, dcb, xg/xb headdim tiles
    dscores = 2 * BLOCK_M * BLOCK_K * 4
    dcb = 2 * BLOCK_M * BLOCK_K * 4
    xgxb = 2 * BLOCK_K * BLOCK_N * 4
    return bg_bb + dx_acc + decay + diff + cb + c_tiles + dy + dscores + dcb + xgxb


def _sram_estimate_bwd_dC(BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE, dstate):
    """Estimate peak SRAM for Kernel 1 (dC/ddA_m)."""
    C_TILES = (dstate + BLOCK_DSTATE - 1) // BLOCK_DSTATE
    # Persistent: C[m] tiles, dC accumulators
    c_tiles = BLOCK_M * dstate * 4
    dC_acc = BLOCK_M * dstate * 4
    # Per k-iteration: Bg/Bb k-tiles (BF16)
    bg_bb = 2 * C_TILES * BLOCK_K * BLOCK_DSTATE * 2
    # Per (k,r) iteration: cb_g/cb_b, dcb_g/dcb_b, decay
    cb = 2 * BLOCK_M * BLOCK_K * 4
    dcb = 2 * BLOCK_M * BLOCK_K * 4
    decay = BLOCK_M * BLOCK_K * 4
    # Per (k,r,h) iteration: dscores, diff, L_active, dy tile, xg/xb tiles
    dscores = 2 * BLOCK_M * BLOCK_K * 4
    diff = BLOCK_M * BLOCK_K * 4
    L_active = BLOCK_M * BLOCK_K * 4
    dy = BLOCK_M * BLOCK_N * 4
    xgxb = 2 * BLOCK_K * BLOCK_N * 4
    return c_tiles + dC_acc + bg_bb + cb + dcb + decay + dscores + diff + L_active + dy + xgxb


def _select_block_sizes_bwd(L, headdim, dstate, device=None):
    """Select block sizes for backward kernels.

    More conservative than forward due to higher register pressure.
    Returns (BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE).
    Always uses num_stages=1.
    """
    BLOCK_DSTATE = max(16, min(64, triton.next_power_of_2(dstate)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(headdim)))

    if device is not None:
        props = torch.cuda.get_device_properties(device)
        sram_limit = getattr(props, 'max_shared_memory_per_block_optin', 0)
        if not sram_limit:
            sram_limit = getattr(props, 'shared_memory_per_block', 0)
        if not sram_limit:
            sram_limit = 228 * 1024
    else:
        sram_limit = 228 * 1024

    # Priority order: (BLOCK_M=64, BLOCK_K=32) > (32, 32) > (32, 16)
    candidates = [
        (64, 32),
        (32, 32),
        (32, 16),
    ]
    for BLOCK_M, BLOCK_K in candidates:
        est_dx = _sram_estimate_bwd_dx(BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE, dstate)
        est_dC = _sram_estimate_bwd_dC(BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE, dstate)
        if max(est_dx, est_dC) <= sram_limit:
            return BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE

    return 32, 16, BLOCK_N, BLOCK_DSTATE


def mamba3_chunk_scan_bwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C, dy, scale, chunk_size, mimo_rank):
    """
    Backward pass for mamba3_chunk_scan_fwd using two fused Triton kernels.

    Args:
        Bg, Bb:       (batch, nchunks, chunk_size, ngroups, R, dstate)
        x_dt_g, x_dt_b: (batch, nchunks, chunk_size, nheads, R, headdim)
        dA_cumsum:    (batch, nchunks, chunk_size, nheads)
        C:            (batch, nchunks, chunk_size, ngroups, dstate)
        dy:           (batch, nchunks, chunk_size, nheads, headdim)
        scale:        float
        chunk_size:   int
        mimo_rank:    int
    Returns:
        dx_dt_g, dx_dt_b: same shape as x_dt_g, x_dt_b
        dBg, dBb:          same shape as Bg, Bb
        dC:                same shape as C
        ddA_cumsum:        same shape as dA_cumsum
    """
    batch, nchunks, L, nheads, R, headdim = x_dt_g.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups
    device = x_dt_g.device

    BLOCK_M, BLOCK_K, BLOCK_N, BLOCK_DSTATE = _select_block_sizes_bwd(
        L, headdim, dstate, device=device)
    assert dstate <= 2 * BLOCK_DSTATE

    # Allocate outputs
    dx_dt_g = torch.empty_like(x_dt_g)
    dx_dt_b = torch.empty_like(x_dt_b)
    dBg = torch.zeros_like(Bg, dtype=torch.float32)  # atomic add target
    dBb = torch.zeros_like(Bb, dtype=torch.float32)
    dC = torch.empty(batch, nchunks, L, ngroups, dstate, device=device, dtype=torch.float32)
    ddA_m = torch.zeros(batch, nchunks, L, nheads, device=device, dtype=torch.float32)
    ddA_k = torch.empty(batch, nchunks, L, nheads, device=device, dtype=torch.float32)

    # --- Kernel 2: dx_dt + dBg/dBb + ddA_k ---
    grid_dx = (triton.cdiv(L, BLOCK_K) * triton.cdiv(headdim, BLOCK_N),
               batch * nchunks, nheads)
    _mamba3_chunk_scan_bwd_dx_kernel[grid_dx](
        Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
        dy,
        dx_dt_g, dx_dt_b, dBg, dBb, ddA_k,
        # Bg strides
        Bg.stride(0), Bg.stride(1), Bg.stride(2), Bg.stride(3), Bg.stride(4), Bg.stride(5),
        # Bb strides
        Bb.stride(0), Bb.stride(1), Bb.stride(2), Bb.stride(3), Bb.stride(4), Bb.stride(5),
        # x_dt_g strides
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2), x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        # x_dt_b strides
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2), x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        # dA_cumsum strides
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        # C strides
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        # dy strides
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        # dx_dt_g output strides
        dx_dt_g.stride(0), dx_dt_g.stride(1), dx_dt_g.stride(2), dx_dt_g.stride(3), dx_dt_g.stride(4), dx_dt_g.stride(5),
        # dx_dt_b output strides
        dx_dt_b.stride(0), dx_dt_b.stride(1), dx_dt_b.stride(2), dx_dt_b.stride(3), dx_dt_b.stride(4), dx_dt_b.stride(5),
        # dBg output strides
        dBg.stride(0), dBg.stride(1), dBg.stride(2), dBg.stride(3), dBg.stride(4), dBg.stride(5),
        # dBb output strides
        dBb.stride(0), dBb.stride(1), dBb.stride(2), dBb.stride(3), dBb.stride(4), dBb.stride(5),
        # ddA_k output strides
        ddA_k.stride(0), ddA_k.stride(1), ddA_k.stride(2), ddA_k.stride(3),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, mimo_rank=R, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8, num_stages=1,
    )

    # --- Kernel 1: dC + ddA_m ---
    grid_dC = (triton.cdiv(L, BLOCK_M), batch * nchunks, ngroups)
    _mamba3_chunk_scan_bwd_dC_kernel[grid_dC](
        Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
        dy,
        dC, ddA_m,
        # Bg strides
        Bg.stride(0), Bg.stride(1), Bg.stride(2), Bg.stride(3), Bg.stride(4), Bg.stride(5),
        # Bb strides
        Bb.stride(0), Bb.stride(1), Bb.stride(2), Bb.stride(3), Bb.stride(4), Bb.stride(5),
        # x_dt_g strides (input, for dscores)
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2), x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        # x_dt_b strides
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2), x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        # dA_cumsum strides
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        # C strides
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        # dy strides
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        # dC output strides
        dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
        # ddA_m output strides
        ddA_m.stride(0), ddA_m.stride(1), ddA_m.stride(2), ddA_m.stride(3),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, mimo_rank=R, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8, num_stages=1,
    )

    # Combine ddA_cumsum from both kernels
    ddA_cumsum = ddA_m + ddA_k

    return dx_dt_g, dx_dt_b, dBg, dBb, dC, ddA_cumsum
