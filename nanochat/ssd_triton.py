"""
Triton kernels for Mamba-3 SSD (Structured State Space Duality).

Forward kernel: fused within-chunk computation for Two-SSD with MIMO. The decay
matrix L is NEVER materialized — computed tile-by-tile in registers. The R-loop
(MIMO rank) is fused inside the kernel. RoPE rotation is split into a single
fused pre-rotation kernel (1 launch for C+Bg+Bb) to reduce register pressure
in the main scan. Pre-rotated B tensors and x_dt values are stored in bf16 to
halve register footprint and memory traffic. CB score dots use bf16 tensor cores
(2x throughput vs TF32); score-to-value dots remain TF32 for precision on
decay-scaled scores (x_dt loaded as bf16, widened to f32 for TF32 dots).
Output is stored as bf16.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Fused RoPE pre-rotation kernel (single launch for C, Bg, Bb)
# =============================================================================

@triton.jit
def _rope_prerotate_fused_kernel(
    # 3 input tensors
    C_ptr, Bg_ptr, Bb_ptr,
    # 2 cos/sin pairs: regular (for C, Bg) and shifted (for Bb)
    cos_ptr, sin_ptr, cos_s_ptr, sin_s_ptr,
    # 3 output tensors
    C_out_ptr, Bg_out_ptr, Bb_out_ptr,
    # Strides for C: (N_pos, ngroups, dstate)
    stride_C_pos, stride_C_ch, stride_C_d,
    # Strides for B input: (N_pos, N_ch_B, dstate) — shared by Bg and Bb
    stride_B_pos, stride_B_ch, stride_B_d,
    # Strides for cos/sin: (N_pos, half_dstate) — shared by both pairs
    stride_cs_pos, stride_cs_d,
    # Strides for C output: (N_pos, ngroups, dstate)
    stride_Co_pos, stride_Co_ch, stride_Co_d,
    # Strides for B output: (N_pos, N_ch_B, dstate) — shared by Bg_rot and Bb_rot
    stride_Bo_pos, stride_Bo_ch, stride_Bo_d,
    # Dimensions
    N_pos,
    ngroups: tl.constexpr,
    N_ch_B: tl.constexpr,
    half_dstate: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused RoPE pre-rotation for C, Bg, Bb in a single kernel launch.

    Grid: (N_pos * (ngroups + 2 * N_ch_B),)
    Programs are assigned to 3 segments per position:
      [0, ngroups)                        → rotate C  with cos/sin
      [ngroups, ngroups + N_ch_B)         → rotate Bg with cos/sin
      [ngroups + N_ch_B, ngroups + 2*N_ch_B) → rotate Bb with cos_s/sin_s

    Consecutive programs within a warp stay in the same segment (minimal divergence).
    """
    seg_total: tl.constexpr = ngroups + 2 * N_ch_B
    pid = tl.program_id(0)
    pos = pid // seg_total
    local = pid % seg_total

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < half_dstate
    base_cs = pos * stride_cs_pos

    if local < ngroups:
        # --- Rotate C with cum_angles ---
        ch = local
        base_i = pos * stride_C_pos + ch * stride_C_ch
        h0 = tl.load(C_ptr + base_i + d_offs * stride_C_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        h1 = tl.load(C_ptr + base_i + (half_dstate + d_offs) * stride_C_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        cos_v = tl.load(cos_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        sin_v = tl.load(sin_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        base_o = pos * stride_Co_pos + ch * stride_Co_ch
        tl.store(C_out_ptr + base_o + d_offs * stride_Co_d,
                 h0 * cos_v - h1 * sin_v, mask=d_mask)
        tl.store(C_out_ptr + base_o + (half_dstate + d_offs) * stride_Co_d,
                 h0 * sin_v + h1 * cos_v, mask=d_mask)
    elif local < ngroups + N_ch_B:
        # --- Rotate Bg with cum_angles ---
        ch = local - ngroups
        base_i = pos * stride_B_pos + ch * stride_B_ch
        h0 = tl.load(Bg_ptr + base_i + d_offs * stride_B_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        h1 = tl.load(Bg_ptr + base_i + (half_dstate + d_offs) * stride_B_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        cos_v = tl.load(cos_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        sin_v = tl.load(sin_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        base_o = pos * stride_Bo_pos + ch * stride_Bo_ch
        tl.store(Bg_out_ptr + base_o + d_offs * stride_Bo_d,
                 (h0 * cos_v - h1 * sin_v).to(tl.bfloat16), mask=d_mask)
        tl.store(Bg_out_ptr + base_o + (half_dstate + d_offs) * stride_Bo_d,
                 (h0 * sin_v + h1 * cos_v).to(tl.bfloat16), mask=d_mask)
    else:
        # --- Rotate Bb with cum_angles_shifted ---
        ch = local - ngroups - N_ch_B
        base_i = pos * stride_B_pos + ch * stride_B_ch
        h0 = tl.load(Bb_ptr + base_i + d_offs * stride_B_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        h1 = tl.load(Bb_ptr + base_i + (half_dstate + d_offs) * stride_B_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        cos_v = tl.load(cos_s_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        sin_v = tl.load(sin_s_ptr + base_cs + d_offs * stride_cs_d,
                         mask=d_mask, other=0.0)
        base_o = pos * stride_Bo_pos + ch * stride_Bo_ch
        tl.store(Bb_out_ptr + base_o + d_offs * stride_Bo_d,
                 (h0 * cos_v - h1 * sin_v).to(tl.bfloat16), mask=d_mask)
        tl.store(Bb_out_ptr + base_o + (half_dstate + d_offs) * stride_Bo_d,
                 (h0 * sin_v + h1 * cos_v).to(tl.bfloat16), mask=d_mask)


def _prerotate_for_scan(Bg, Bb, C, cos_ang, sin_ang, cos_angs, sin_angs):
    """Pre-rotate B and C tensors with RoPE — single fused Triton kernel launch.

    Returns bf16 rotated B tensors and float32 rotated C for the scan kernel.
    """
    batch, nc, L, ngroups, R, dstate = Bg.shape
    half_dstate = dstate // 2
    BLOCK_D = triton.next_power_of_2(half_dstate)
    N_pos = batch * nc * L
    N_ch_B = ngroups * R

    cos_flat = cos_ang.contiguous().view(N_pos, half_dstate)
    sin_flat = sin_ang.contiguous().view(N_pos, half_dstate)
    cos_s_flat = cos_angs.contiguous().view(N_pos, half_dstate)
    sin_s_flat = sin_angs.contiguous().view(N_pos, half_dstate)

    C_flat = C.contiguous().view(N_pos, ngroups, dstate)
    Bg_flat = Bg.contiguous().view(N_pos, N_ch_B, dstate)
    Bb_flat = Bb.contiguous().view(N_pos, N_ch_B, dstate)

    C_rot = torch.empty_like(C_flat)
    Bg_rot = torch.empty(N_pos, N_ch_B, dstate, device=Bg.device, dtype=torch.bfloat16)
    Bb_rot = torch.empty(N_pos, N_ch_B, dstate, device=Bb.device, dtype=torch.bfloat16)

    total_programs = N_pos * (ngroups + 2 * N_ch_B)
    _rope_prerotate_fused_kernel[(total_programs,)](
        C_flat, Bg_flat, Bb_flat,
        cos_flat, sin_flat, cos_s_flat, sin_s_flat,
        C_rot, Bg_rot, Bb_rot,
        C_flat.stride(0), C_flat.stride(1), C_flat.stride(2),
        Bg_flat.stride(0), Bg_flat.stride(1), Bg_flat.stride(2),
        cos_flat.stride(0), cos_flat.stride(1),
        C_rot.stride(0), C_rot.stride(1), C_rot.stride(2),
        Bg_rot.stride(0), Bg_rot.stride(1), Bg_rot.stride(2),
        N_pos,
        ngroups=ngroups, N_ch_B=N_ch_B,
        half_dstate=half_dstate, BLOCK_D=BLOCK_D,
    )

    return (Bg_rot.view(batch, nc, L, ngroups, R, dstate),
            Bb_rot.view(batch, nc, L, ngroups, R, dstate),
            C_rot.view(batch, nc, L, ngroups, dstate))


# =============================================================================
# Forward kernel (pre-rotated inputs — no RoPE in hot loop)
# =============================================================================

@triton.autotune(
    configs=[
        # Large blocks (chunk >= 128, headdim >= 64)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=1, num_warps=4),
        # Asymmetric M/N (headdim > chunk or vice versa)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        # Medium blocks
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=1, num_warps=4),
        # Large K for large chunks (reduces K-loop iterations)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=1, num_warps=4),
        # Small K for small configs
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['chunk_size', 'headdim', 'dstate', 'mimo_rank', 'ngroups'],
)
@triton.jit
def _mamba3_chunk_scan_fwd_kernel(
    # Input tensors (pre-rotated B in bf16, C in float32)
    Bg_ptr, Bb_ptr,            # (batch, nc, L, ngroups, R, dstate) bf16
    x_dt_g_ptr, x_dt_b_ptr,   # (batch, nc, L, nheads, R, headdim) bf16
    dA_cumsum_ptr,             # (batch, nc, L, nheads) float32
    C_ptr,                     # (batch, nc, L, ngroups, dstate) float32
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

    Takes pre-rotated B (bf16) and C (float32) tensors (RoPE applied in separate kernel).
    B in bf16 halves register footprint and enables bf16 tensor cores for CB dots (2x TF32).
    Score-to-value dots remain TF32 for precision on decay-scaled scores.

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

    # Pre-load C tiles (already rotated by pre-rotation kernel, float32).
    # Cast to bf16 for CB dot products — engages bf16 tensor cores (2x TF32).
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate

    c_tile_0 = tl.load(
        C_ptr + off_bc_C
        + offs_m[:, None] * stride_C_seqlen
        + c_d0_offs[None, :] * stride_C_dstate,
        mask=(offs_m[:, None] < chunk_size) & c_d0_mask[None, :],
        other=0.0).to(tl.bfloat16)

    c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
    c_d1_mask = c_d1_offs < dstate
    c_tile_1 = tl.load(
        C_ptr + off_bc_C
        + offs_m[:, None] * stride_C_seqlen
        + c_d1_offs[None, :] * stride_C_dstate,
        mask=(offs_m[:, None] < chunk_size) & c_d1_mask[None, :],
        other=0.0).to(tl.bfloat16)

    # --- Intra-chunk: tile-by-tile over k positions ---
    # Causal cutoff: skip k-blocks where all k > max(m) in this tile.
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

        for r in range(mimo_rank):
            # Load pre-rotated Bg tiles (bf16, no RoPE needed)
            bg_tile_0 = tl.load(
                Bg_ptr + off_bc_Bg
                + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank
                + c_d0_offs[None, :] * stride_Bg_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            bg_tile_1 = tl.load(
                Bg_ptr + off_bc_Bg
                + k_offs[:, None] * stride_Bg_seqlen
                + r * stride_Bg_rank
                + c_d1_offs[None, :] * stride_Bg_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                other=0.0)

            # Load pre-rotated Bb tiles (bf16, no RoPE needed)
            bb_tile_0 = tl.load(
                Bb_ptr + off_bc_Bb
                + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank
                + c_d0_offs[None, :] * stride_Bb_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            bb_tile_1 = tl.load(
                Bb_ptr + off_bc_Bb
                + k_offs[:, None] * stride_Bb_seqlen
                + r * stride_Bb_rank
                + c_d1_offs[None, :] * stride_Bb_dstate,
                mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                other=0.0)

            # Dot products: C_rot @ B_rot^T = sum over both dstate halves
            # Both C and B are bf16 → bf16 tensor cores (2x throughput vs TF32),
            # with float32 accumulation (Triton default).
            cb_g = tl.dot(c_tile_0, tl.trans(bg_tile_0)) + tl.dot(c_tile_1, tl.trans(bg_tile_1))
            cb_b = tl.dot(c_tile_0, tl.trans(bb_tile_0)) + tl.dot(c_tile_1, tl.trans(bb_tile_1))

            # Apply decay and scale: (BLOCK_M, BLOCK_K)
            scores_g = cb_g * decay * scale
            scores_b = cb_b * decay * scale

            # x_dt_g[k, h, r, p]: (BLOCK_K, BLOCK_N) — loaded as bf16, cast to f32
            xg_vals = tl.load(
                x_dt_g_ptr + off_bc_xg
                + k_offs[:, None] * stride_xg_seqlen
                + pid_h * stride_xg_head
                + r * stride_xg_rank
                + offs_n[None, :] * stride_xg_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0).to(tl.float32)

            # x_dt_b[k, h, r, p]: (BLOCK_K, BLOCK_N) — loaded as bf16, cast to f32
            xb_vals = tl.load(
                x_dt_b_ptr + off_bc_xb
                + k_offs[:, None] * stride_xb_seqlen
                + pid_h * stride_xb_head
                + r * stride_xb_rank
                + offs_n[None, :] * stride_xb_hdim,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0).to(tl.float32)

            # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            # TF32 for these dots — scores are float32 (decay dynamic range 0->1).
            # x_dt loaded as bf16 and widened to float32 for TF32 tensor cores.
            acc += tl.dot(scores_g, xg_vals)
            acc += tl.dot(scores_b, xb_vals)

    # Store output
    off_bc_out = pid_b * stride_out_batch + pid_c * stride_out_chunk
    tl.store(out_ptr + off_bc_out
             + offs_m[:, None] * stride_out_seqlen
             + pid_h * stride_out_head
             + offs_n[None, :] * stride_out_hdim,
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def _sram_estimate(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, dstate, num_stages=1):
    """Estimate peak SRAM usage in bytes for the scan kernel.

    B tiles are bf16 (2 bytes), C tiles are bf16 (2 bytes), x_dt are bf16 (2 bytes).
    """
    # Fixed: not affected by pipeline staging
    fixed = (BLOCK_M * BLOCK_N * 4             # acc (float32)
            + BLOCK_M * BLOCK_K * 4            # decay (float32)
            + BLOCK_M * dstate * 2             # c_rot tiles (bf16, 2 halves)
            )
    # Staged: loop-body global loads, multiplied by num_stages
    staged = (4 * BLOCK_K * BLOCK_DSTATE * 2   # bg_tile0, bg_tile1, bb_tile0, bb_tile1 (bf16)
            + 2 * BLOCK_K * BLOCK_N * 2        # xg, xb (bf16, cast to f32 in registers)
            )
    return fixed + staged * num_stages


def _select_block_sizes(L, headdim, dstate, device=None):
    """Select block sizes and num_stages for the scan kernel.

    BLOCK_DSTATE = half_dstate ensures the two tiles split dstate evenly.

    Queries the actual device shared-memory limit (falls back to a
    conservative 228 KB if unavailable) and tries block-size / num_stages
    combinations in priority order until one fits.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_DSTATE, num_stages).
    """
    assert dstate % 2 == 0, f"Requires even dstate, got {dstate}"
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
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO.

    RoPE pre-rotation is applied in a single fused Triton kernel (B outputs in bf16,
    C in float32), then the main scan kernel runs with reduced register pressure
    (no cos/sin in hot loop, bf16 B tiles halve register footprint).

    Args:
        Bg:        (batch, nchunks, chunk_size, ngroups, R, dstate)
        Bb:        (batch, nchunks, chunk_size, ngroups, R, dstate)
        x_dt_g:    (batch, nchunks, chunk_size, nheads, R, headdim) bfloat16
        x_dt_b:    (batch, nchunks, chunk_size, nheads, R, headdim) bfloat16
        dA_cumsum: (batch, nchunks, chunk_size, nheads) float32
        C:         (batch, nchunks, chunk_size, ngroups, dstate) float32
        cos_ang:   (batch, nchunks, chunk_size, half_dstate) float32
        sin_ang:   (batch, nchunks, chunk_size, half_dstate) float32
        cos_angs:  (batch, nchunks, chunk_size, half_dstate) float32
        sin_angs:  (batch, nchunks, chunk_size, half_dstate) float32
        scale:     float
        chunk_size: int
        mimo_rank: int
    Returns:
        y_intra:  (batch, nchunks, chunk_size, nheads, headdim) bfloat16
        Bg_rot:   (batch, nchunks, chunk_size, ngroups, R, dstate) bfloat16
        Bb_rot:   (batch, nchunks, chunk_size, ngroups, R, dstate) bfloat16
        C_rot:    (batch, nchunks, chunk_size, ngroups, dstate) same dtype as C
    """
    batch, nchunks, L, nheads, R, headdim = x_dt_g.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups

    assert dstate % 2 == 0, f"Requires even dstate, got {dstate}"
    BLOCK_DSTATE = max(32, dstate // 2)

    # Pre-rotate B and C with RoPE (single fused Triton kernel launch)
    Bg_rot, Bb_rot, C_rot = _prerotate_for_scan(
        Bg, Bb, C, cos_ang, sin_ang, cos_angs, sin_angs)

    y_intra = torch.empty(batch, nchunks, L, nheads, headdim,
                           device=x_dt_g.device, dtype=torch.bfloat16)

    # Grid is a lambda because BLOCK_M/BLOCK_N are chosen by @triton.autotune.
    grid = lambda META: (
        triton.cdiv(L, META['BLOCK_M']) * triton.cdiv(headdim, META['BLOCK_N']),
        batch * nchunks,
        nheads,
    )

    _mamba3_chunk_scan_fwd_kernel[grid](
        Bg_rot, Bb_rot, x_dt_g, x_dt_b, dA_cumsum, C_rot,
        y_intra,
        Bg_rot.stride(0), Bg_rot.stride(1), Bg_rot.stride(2), Bg_rot.stride(3), Bg_rot.stride(4), Bg_rot.stride(5),
        Bb_rot.stride(0), Bb_rot.stride(1), Bb_rot.stride(2), Bb_rot.stride(3), Bb_rot.stride(4), Bb_rot.stride(5),
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2), x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2), x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        C_rot.stride(0), C_rot.stride(1), C_rot.stride(2), C_rot.stride(3), C_rot.stride(4),
        y_intra.stride(0), y_intra.stride(1), y_intra.stride(2), y_intra.stride(3), y_intra.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, mimo_rank=R, scale=scale,
        BLOCK_DSTATE=BLOCK_DSTATE,
    )
    return y_intra, Bg_rot, Bb_rot, C_rot


# =============================================================================
# Fused state propagation kernel (between-chunk sequential scan)
# =============================================================================

@triton.jit
def _state_propagation_fwd_kernel(
    # Inputs
    delta_h_ptr,       # (batch, n_chunks, n_heads, head_dim, d_state) float32
    chunk_decay_ptr,   # (batch, n_chunks, n_heads) float32
    # Output
    all_states_ptr,    # (batch, n_chunks, n_heads, head_dim, d_state) float32
    # Strides for delta_h (5D): batch, chunk, head, headdim, dstate
    stride_dh_b, stride_dh_c, stride_dh_h, stride_dh_p, stride_dh_n,
    # Strides for chunk_decay (3D): batch, chunk, head
    stride_cd_b, stride_cd_c, stride_cd_h,
    # Strides for all_states (5D): batch, chunk, head, headdim, dstate
    stride_as_b, stride_as_c, stride_as_h, stride_as_p, stride_as_n,
    # Dimensions
    n_chunks: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    d_state: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused sequential state propagation across chunks.

    Replaces n_chunks separate kernel launches with a single launch.
    Each program handles a (BLOCK_P, BLOCK_N) tile of the state matrix
    for one (batch, head) pair, iterating sequentially over chunks.

    Computes:
        all_states[0] = zeros
        all_states[c+1] = chunk_decay[c] * all_states[c] + delta_h[c]  for c=0..nc-2

    Grid: (cdiv(head_dim, BLOCK_P) * cdiv(d_state, BLOCK_N), batch * n_heads)
    """
    pid_tile = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decompose batch*head index
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    # Decompose tile index into (head_dim, d_state) tile coordinates
    n_tiles_n = tl.cdiv(d_state, BLOCK_N)
    pid_p = pid_tile // n_tiles_n
    pid_n = pid_tile % n_tiles_n

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tile_mask = (offs_p[:, None] < head_dim) & (offs_n[None, :] < d_state)

    # Base pointers for this (batch, head) slice
    base_dh = pid_b * stride_dh_b + pid_h * stride_dh_h
    base_cd = pid_b * stride_cd_b + pid_h * stride_cd_h
    base_as = pid_b * stride_as_b + pid_h * stride_as_h

    # Reusable offset pattern for (head_dim, d_state) tile
    tile_offsets = offs_p[:, None] * stride_dh_p + offs_n[None, :] * stride_dh_n
    tile_offsets_as = offs_p[:, None] * stride_as_p + offs_n[None, :] * stride_as_n

    # Initialize state to zeros
    state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    for c in range(n_chunks):
        # Store state BEFORE update: all_states[c] = state
        tl.store(all_states_ptr + base_as + c * stride_as_c + tile_offsets_as,
                 state, mask=tile_mask)

        # Load chunk_decay[b, c, h] — scalar per program
        decay = tl.load(chunk_decay_ptr + base_cd + c * stride_cd_c)

        # Load delta_h[b, c, h, :, :] tile
        dh = tl.load(delta_h_ptr + base_dh + c * stride_dh_c + tile_offsets,
                     mask=tile_mask, other=0.0)

        # Recurrence: state = decay * state + delta_h
        state = decay * state + dh


def state_propagation_fwd(delta_h, chunk_decay):
    """Fused state propagation for between-chunk SSD computation.

    Computes the prefix scan:
        all_states[0] = zeros
        all_states[c+1] = chunk_decay[c] * all_states[c] + delta_h[c]  for c=0..nc-2

    Returns all_states[0..nc-1] (state BEFORE each chunk's update).

    Args:
        delta_h:     (batch, n_chunks, n_heads, head_dim, d_state) float32
        chunk_decay: (batch, n_chunks, n_heads) float32
    Returns:
        all_states:  (batch, n_chunks, n_heads, head_dim, d_state) float32
    """
    batch, n_chunks, n_heads, head_dim, d_state = delta_h.shape

    all_states = torch.empty_like(delta_h)

    BLOCK_P = min(64, triton.next_power_of_2(head_dim))
    BLOCK_N = min(64, triton.next_power_of_2(d_state))
    num_warps = 2 if BLOCK_P * BLOCK_N <= 2048 else 4

    grid = (
        triton.cdiv(head_dim, BLOCK_P) * triton.cdiv(d_state, BLOCK_N),
        batch * n_heads,
    )

    _state_propagation_fwd_kernel[grid](
        delta_h, chunk_decay, all_states,
        delta_h.stride(0), delta_h.stride(1), delta_h.stride(2),
        delta_h.stride(3), delta_h.stride(4),
        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
        all_states.stride(0), all_states.stride(1), all_states.stride(2),
        all_states.stride(3), all_states.stride(4),
        n_chunks=n_chunks, n_heads=n_heads,
        head_dim=head_dim, d_state=d_state,
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return all_states


# =============================================================================
# Fused between-chunk delta_h kernel (replaces Python R-loop of einsums)
# =============================================================================

@triton.autotune(
    configs=[
        # Full tile (head_dim=64, d_state=64)
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 32}, num_stages=1, num_warps=4),
        # Larger d_state (128) or smaller tiles for higher occupancy
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 64, 'BLOCK_L': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 32, 'BLOCK_L': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 32, 'BLOCK_L': 64}, num_stages=2, num_warps=4),
        # Large L-tile (reduces loop overhead for L=256)
        triton.Config({'BLOCK_P': 64, 'BLOCK_N': 64, 'BLOCK_L': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_P': 32, 'BLOCK_N': 64, 'BLOCK_L': 128}, num_stages=1, num_warps=4),
    ],
    key=['head_dim', 'd_state', 'chunk_size', 'mimo_rank'],
)
@triton.jit
def _delta_h_fwd_kernel(
    # Inputs
    x_dt_g_ptr,        # (B, nc, L, n_heads, R, head_dim) bf16
    x_dt_b_ptr,        # (B, nc, L, n_heads, R, head_dim) bf16
    Bg_rot_ptr,        # (B, nc, L, ngroups, R, d_state)  bf16
    Bb_rot_ptr,        # (B, nc, L, ngroups, R, d_state)  bf16
    cum_log_dA_ptr,    # (B, nc, L, n_heads)               f32
    # Output
    delta_h_ptr,       # (B, nc, n_heads, head_dim, d_state) f32
    # Strides for x_dt_g (6D)
    stride_xg_b, stride_xg_c, stride_xg_l, stride_xg_h, stride_xg_r, stride_xg_p,
    # Strides for x_dt_b (6D)
    stride_xb_b, stride_xb_c, stride_xb_l, stride_xb_h, stride_xb_r, stride_xb_p,
    # Strides for Bg_rot (6D)
    stride_Bg_b, stride_Bg_c, stride_Bg_l, stride_Bg_g, stride_Bg_r, stride_Bg_n,
    # Strides for Bb_rot (6D)
    stride_Bb_b, stride_Bb_c, stride_Bb_l, stride_Bb_g, stride_Bb_r, stride_Bb_n,
    # Strides for cum_log_dA (4D)
    stride_dA_b, stride_dA_c, stride_dA_l, stride_dA_h,
    # Strides for delta_h (5D)
    stride_dh_b, stride_dh_c, stride_dh_h, stride_dh_p, stride_dh_n,
    # Dimensions
    n_heads: tl.constexpr,
    ngroups: tl.constexpr,
    hpg: tl.constexpr,
    head_dim: tl.constexpr,
    d_state: tl.constexpr,
    chunk_size: tl.constexpr,
    mimo_rank: tl.constexpr,
    # Block sizes (autotuned)
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Fused between-chunk delta_h: outer-product reduction over (L, R).

    Computes decay_to_end = exp(dA_last - dA[l]) in registers from cum_log_dA,
    avoiding a separate (B, nc, L, n_heads) allocation. The R-loop is unrolled
    at compile time. bf16 tensor core dots with f32 accumulation.

    delta_h[b,c,h,p,n] = sum_{r,l} decay_to_end[b,c,l,h] * (
        x_dt_g[b,c,l,h,r,p] * Bg_rot[b,c,l,g(h),r,n]
      + x_dt_b[b,c,l,h,r,p] * Bb_rot[b,c,l,g(h),r,n]
    )

    Grid: (cdiv(head_dim, BLOCK_P) * cdiv(d_state, BLOCK_N), batch * n_chunks * n_heads)
    """
    pid_tile = tl.program_id(0)
    pid_bch = tl.program_id(1)

    # Decompose: pid_bch = (b * n_chunks + c) * n_heads + h
    # pid_bc = b * n_chunks + c — used with stride_c since stride_b = nc * stride_c
    # for contiguous tensors.
    pid_h = pid_bch % n_heads
    pid_bc = pid_bch // n_heads

    # Decompose tile into (head_dim, d_state) coordinates
    n_tiles_n = tl.cdiv(d_state, BLOCK_N)
    pid_p = pid_tile // n_tiles_n
    pid_n = pid_tile % n_tiles_n

    group_idx = pid_h // hpg

    # Tile offsets
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = offs_p < head_dim
    n_mask = offs_n < d_state

    # Base pointers using strides (avoids needing n_chunks as constexpr)
    # pid_bc = batch_idx * n_chunks + chunk_idx
    # stride_xg_b already accounts for full batch stride
    base_xg = pid_bc * stride_xg_c + pid_h * stride_xg_h
    base_xb = pid_bc * stride_xb_c + pid_h * stride_xb_h
    base_Bg = pid_bc * stride_Bg_c + group_idx * stride_Bg_g
    base_Bb = pid_bc * stride_Bb_c + group_idx * stride_Bb_g
    base_dA = pid_bc * stride_dA_c + pid_h * stride_dA_h

    # Load cum_log_dA at L-1 (last position in chunk)
    dA_last = tl.load(cum_log_dA_ptr + base_dA + (chunk_size - 1) * stride_dA_l)

    # Accumulator
    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Reduction over L (tiled)
    for l_start in range(0, chunk_size, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < chunk_size

        # Compute decay_to_end in registers
        dA_vals = tl.load(cum_log_dA_ptr + base_dA + l_offs * stride_dA_l,
                          mask=l_mask, other=0.0)
        decay = tl.exp(dA_last - dA_vals)
        decay = tl.where(l_mask, decay, 0.0)

        # Unrolled R loop
        for r in range(mimo_rank):
            # Load x_dt_g[b, c, l, h, r, p]: (BLOCK_L, BLOCK_P)
            xg = tl.load(
                x_dt_g_ptr + base_xg
                + l_offs[:, None] * stride_xg_l
                + r * stride_xg_r
                + offs_p[None, :] * stride_xg_p,
                mask=l_mask[:, None] & p_mask[None, :],
                other=0.0)

            # Load x_dt_b[b, c, l, h, r, p]: (BLOCK_L, BLOCK_P)
            xb = tl.load(
                x_dt_b_ptr + base_xb
                + l_offs[:, None] * stride_xb_l
                + r * stride_xb_r
                + offs_p[None, :] * stride_xb_p,
                mask=l_mask[:, None] & p_mask[None, :],
                other=0.0)

            # Load Bg_rot[b, c, l, g, r, n]: (BLOCK_L, BLOCK_N)
            bg = tl.load(
                Bg_rot_ptr + base_Bg
                + l_offs[:, None] * stride_Bg_l
                + r * stride_Bg_r
                + offs_n[None, :] * stride_Bg_n,
                mask=l_mask[:, None] & n_mask[None, :],
                other=0.0)

            # Load Bb_rot[b, c, l, g, r, n]: (BLOCK_L, BLOCK_N)
            bb = tl.load(
                Bb_rot_ptr + base_Bb
                + l_offs[:, None] * stride_Bb_l
                + r * stride_Bb_r
                + offs_n[None, :] * stride_Bb_n,
                mask=l_mask[:, None] & n_mask[None, :],
                other=0.0)

            # Scale x by decay, cast to bf16 for tensor core dot
            xg_s = (xg.to(tl.float32) * decay[:, None]).to(tl.bfloat16)
            xb_s = (xb.to(tl.float32) * decay[:, None]).to(tl.bfloat16)

            # Outer product reduction:
            # (BLOCK_P, BLOCK_L) @ (BLOCK_L, BLOCK_N) -> (BLOCK_P, BLOCK_N)
            acc += tl.dot(tl.trans(xg_s), bg)
            acc += tl.dot(tl.trans(xb_s), bb)

    # Store result
    base_out = pid_bc * stride_dh_c + pid_h * stride_dh_h
    tl.store(
        delta_h_ptr + base_out
        + offs_p[:, None] * stride_dh_p
        + offs_n[None, :] * stride_dh_n,
        acc,
        mask=p_mask[:, None] & n_mask[None, :])


def delta_h_fwd(x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA):
    """Fused between-chunk delta_h computation.

    Replaces the Python R-loop of einsums with a single Triton kernel that
    computes decay_to_end in registers and unrolls the MIMO rank loop.

    Args:
        x_dt_g:     (B, nc, L, n_heads, R, head_dim) bf16
        x_dt_b:     (B, nc, L, n_heads, R, head_dim) bf16
        Bg_rot:     (B, nc, L, ngroups, R, d_state)  bf16
        Bb_rot:     (B, nc, L, ngroups, R, d_state)  bf16
        cum_log_dA: (B, nc, L, n_heads)               f32
    Returns:
        delta_h:    (B, nc, n_heads, head_dim, d_state) f32
    """
    batch, nc, L, n_heads, R, head_dim = x_dt_g.shape
    ngroups = Bg_rot.shape[3]
    d_state = Bg_rot.shape[5]
    hpg = n_heads // ngroups

    delta_h = torch.empty(batch, nc, n_heads, head_dim, d_state,
                           device=x_dt_g.device, dtype=torch.float32)

    # Flatten batch and n_chunks into one grid axis for simplicity.
    # The stride-based addressing handles the actual memory layout.
    # We pass x_dt_g pointer with batch stride folded into the chunk stride
    # by setting base = pid_bc * stride_c where pid_bc = b * nc + c.
    # This works because tensors are contiguous in (batch, nc, ...) layout,
    # so stride_b = nc * stride_c for contiguous tensors.

    grid = lambda META: (
        triton.cdiv(head_dim, META['BLOCK_P']) * triton.cdiv(d_state, META['BLOCK_N']),
        batch * nc * n_heads,
    )

    _delta_h_fwd_kernel[grid](
        x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA,
        delta_h,
        # x_dt_g strides
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2),
        x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        # x_dt_b strides
        x_dt_b.stride(0), x_dt_b.stride(1), x_dt_b.stride(2),
        x_dt_b.stride(3), x_dt_b.stride(4), x_dt_b.stride(5),
        # Bg_rot strides
        Bg_rot.stride(0), Bg_rot.stride(1), Bg_rot.stride(2),
        Bg_rot.stride(3), Bg_rot.stride(4), Bg_rot.stride(5),
        # Bb_rot strides
        Bb_rot.stride(0), Bb_rot.stride(1), Bb_rot.stride(2),
        Bb_rot.stride(3), Bb_rot.stride(4), Bb_rot.stride(5),
        # cum_log_dA strides
        cum_log_dA.stride(0), cum_log_dA.stride(1),
        cum_log_dA.stride(2), cum_log_dA.stride(3),
        # delta_h strides
        delta_h.stride(0), delta_h.stride(1), delta_h.stride(2),
        delta_h.stride(3), delta_h.stride(4),
        # Dimensions
        n_heads=n_heads, ngroups=ngroups, hpg=hpg,
        head_dim=head_dim, d_state=d_state,
        chunk_size=L, mimo_rank=R,
    )

    return delta_h
