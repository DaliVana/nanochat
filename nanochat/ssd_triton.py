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
        y_intra: (batch, nchunks, chunk_size, nheads, headdim) bfloat16
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
    return y_intra
