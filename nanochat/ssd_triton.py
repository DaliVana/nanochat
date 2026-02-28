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
    C_TILES: tl.constexpr = tl.cdiv(dstate, BLOCK_DSTATE)

    # Tile 0 (always present)
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate
    # C tiles stay in native dtype (BF16 on H100) for BF16 tensor core dot products.
    # The CB accumulator (cb_g, cb_b) is float32, so dot results are promoted on +=.
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
            # Cast scores to x_dt dtype: enables BF16 tensor cores when x_dt is BF16.
            # H100 BF16 HMMA: 1,979 TFLOPS vs 989 TFLOPS for FP32.
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


def _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, dstate):
    """Estimate peak SRAM usage in bytes for the mamba3 kernel."""
    return (BLOCK_M * BLOCK_N * 4             # acc (float32)
            + 2 * BLOCK_M * BLOCK_K * 4      # cb_g, cb_b (float32)
            + BLOCK_M * BLOCK_K * 4           # decay (float32)
            + BLOCK_M * dstate * 2            # c_cache (BF16)
            + 4 * BLOCK_K * BLOCK_DSTATE * 2  # bg, bb x2 tiles (BF16)
            + 2 * BLOCK_K * BLOCK_N * 2       # xg, xb (BF16)
            )


def _select_block_sizes(L, headdim, dstate):
    """Select block sizes for the mamba3 kernel.

    On H100 (228 KB SRAM), larger blocks improve utilization:
    - BLOCK_M=128 with chunk_size=128 covers the entire chunk in one tile
    - BLOCK_N=128 with headdim=128 eliminates N-dimension grid splits
    - BLOCK_DSTATE=64 is always sufficient (dstate inner loop is small)

    Tries candidates in priority order (most SRAM → least) until one fits.
    """
    BLOCK_DSTATE = max(16, min(64, triton.next_power_of_2(dstate)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(L)))
    H100_SRAM = 228 * 1024  # 228 KB

    # Priority: BLOCK_M=128 is most impactful (covers full chunk),
    # BLOCK_N=128 is secondary (covers full headdim).
    BM_max = max(16, min(128, triton.next_power_of_2(L)))
    BN_max = max(16, min(128, triton.next_power_of_2(headdim)))
    BM_64 = max(16, min(64, triton.next_power_of_2(L)))
    BN_64 = max(16, min(64, triton.next_power_of_2(headdim)))

    candidates = [
        (BM_max, BN_max),    # Best: full chunk + full headdim
        (BM_max, BN_64),     # Good: full chunk, split headdim
        (BM_64, BN_max),     # OK: split chunk, full headdim
        (BM_64, BN_64),      # Conservative fallback
    ]
    for BLOCK_M, BLOCK_N in candidates:
        if _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, dstate) <= H100_SRAM:
            return BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE

    return BM_64, BN_64, BLOCK_K, BLOCK_DSTATE


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

    BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE = _select_block_sizes(L, headdim, dstate)
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
        num_stages=2,
    )
    return y_intra
