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
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim) float32
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

    # --- Intra-chunk: tile-by-tile over k positions ---
    # Loop bound must be constexpr; the causal mask zeros out k > m tiles.
    for k_start in range(0, chunk_size, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # === Decay computed in registers, NEVER stored to global memory ===
        dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                           mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))

        # Causal mask: only k <= m contributes
        causal = offs_m[:, None] >= k_offs[None, :]
        decay = tl.where(causal, decay, 0.0)  # (BLOCK_M, BLOCK_K)

        # R-loop: compile-time unrolled (mimo_rank is tl.constexpr)
        for r in range(mimo_rank):
            # Compute CB_g = C[m] @ Bg[k,r]^T and CB_b = C[m] @ Bb[k,r]^T
            cb_g = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            cb_b = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for d_start in range(0, dstate, BLOCK_DSTATE):
                d_offs = d_start + tl.arange(0, BLOCK_DSTATE)
                d_mask = d_offs < dstate

                # C[m, g, d]: (BLOCK_M, BLOCK_DSTATE)
                c_vals = tl.load(
                    C_ptr + off_bc_C
                    + offs_m[:, None] * stride_C_seqlen
                    + d_offs[None, :] * stride_C_dstate,
                    mask=(offs_m[:, None] < chunk_size) & d_mask[None, :],
                    other=0.0).to(tl.float32)

                # Bg[k, g, r, d]: (BLOCK_K, BLOCK_DSTATE)
                bg_vals = tl.load(
                    Bg_ptr + off_bc_Bg
                    + k_offs[:, None] * stride_Bg_seqlen
                    + r * stride_Bg_rank
                    + d_offs[None, :] * stride_Bg_dstate,
                    mask=(k_offs[:, None] < chunk_size) & d_mask[None, :],
                    other=0.0).to(tl.float32)

                # Bb[k, g, r, d]: (BLOCK_K, BLOCK_DSTATE)
                bb_vals = tl.load(
                    Bb_ptr + off_bc_Bb
                    + k_offs[:, None] * stride_Bb_seqlen
                    + r * stride_Bb_rank
                    + d_offs[None, :] * stride_Bb_dstate,
                    mask=(k_offs[:, None] < chunk_size) & d_mask[None, :],
                    other=0.0).to(tl.float32)

                cb_g += tl.dot(c_vals, tl.trans(bg_vals))
                cb_b += tl.dot(c_vals, tl.trans(bb_vals))

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
                other=0.0).to(tl.float32)

            # x_dt_b[k, h, r, p]: (BLOCK_K, BLOCK_N)
            xb_vals = tl.load(
                x_dt_b_ptr + off_bc_xb
                + k_offs[:, None] * stride_xb_seqlen
                + pid_h * stride_xb_head
                + r * stride_xb_rank
                + offs_n[None, :] * stride_xb_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0).to(tl.float32)

            # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(scores_g, xg_vals)
            acc += tl.dot(scores_b, xb_vals)

    # Store output
    off_bc_out = pid_b * stride_out_batch + pid_c * stride_out_chunk
    tl.store(out_ptr + off_bc_out
             + offs_m[:, None] * stride_out_seqlen
             + pid_h * stride_out_head
             + offs_n[None, :] * stride_out_hdim,
             acc,
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


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

    # Minimum 16 for tl.dot compatibility on all architectures
    BLOCK_M = max(16, min(64, triton.next_power_of_2(L)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(headdim)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(L)))
    BLOCK_DSTATE = max(16, min(64, triton.next_power_of_2(dstate)))
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
