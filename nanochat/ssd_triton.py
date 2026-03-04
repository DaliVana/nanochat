"""
Triton kernels for Mamba-3 SSD (Structured State Space Duality).

Naming conventions:
    L — chunk_size (sequence length within a chunk)

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
    # 2 raw angle tensors (cos/sin computed in-kernel via SFU)
    angles_ptr, angles_shifted_ptr,
    # 3 output tensors
    C_out_ptr, Bg_out_ptr, Bb_out_ptr,
    # Strides for C: (N_pos, ngroups, dstate)
    stride_C_pos, stride_C_ch, stride_C_d,
    # Strides for B input: (N_pos, N_ch_B, dstate) — shared by Bg and Bb
    stride_B_pos, stride_B_ch, stride_B_d,
    # Strides for angles: (N_pos, half_dstate) — shared by both pairs
    stride_a_pos, stride_a_d,
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

    cos/sin are computed in-kernel from raw angles (SFU ops, ~4 cycles),
    eliminating 4 float32 intermediate tensors from HBM.

    Grid: (N_pos * (ngroups + 2 * N_ch_B),)
    Programs are assigned to 3 segments per position:
      [0, ngroups)                        → rotate C  with angles
      [ngroups, ngroups + N_ch_B)         → rotate Bg with angles
      [ngroups + N_ch_B, ngroups + 2*N_ch_B) → rotate Bb with angles_shifted
    """
    seg_total: tl.constexpr = ngroups + 2 * N_ch_B
    pid = tl.program_id(0)
    pos = pid // seg_total
    local = pid % seg_total

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < half_dstate

    if local < ngroups:
        # --- Rotate C with cum_angles ---
        ch = local
        base_i = pos * stride_C_pos + ch * stride_C_ch
        h0 = tl.load(C_ptr + base_i + d_offs * stride_C_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        h1 = tl.load(C_ptr + base_i + (half_dstate + d_offs) * stride_C_d,
                      mask=d_mask, other=0.0).to(tl.float32)
        angle = tl.load(angles_ptr + pos * stride_a_pos + d_offs * stride_a_d,
                        mask=d_mask, other=0.0)
        cos_v = tl.cos(angle)
        sin_v = tl.sin(angle)
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
        angle = tl.load(angles_ptr + pos * stride_a_pos + d_offs * stride_a_d,
                        mask=d_mask, other=0.0)
        cos_v = tl.cos(angle)
        sin_v = tl.sin(angle)
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
        angle = tl.load(angles_shifted_ptr + pos * stride_a_pos + d_offs * stride_a_d,
                        mask=d_mask, other=0.0)
        cos_v = tl.cos(angle)
        sin_v = tl.sin(angle)
        base_o = pos * stride_Bo_pos + ch * stride_Bo_ch
        tl.store(Bb_out_ptr + base_o + d_offs * stride_Bo_d,
                 (h0 * cos_v - h1 * sin_v).to(tl.bfloat16), mask=d_mask)
        tl.store(Bb_out_ptr + base_o + (half_dstate + d_offs) * stride_Bo_d,
                 (h0 * sin_v + h1 * cos_v).to(tl.bfloat16), mask=d_mask)


def _prerotate_for_scan(Bg, Bb, C, angles, angles_shifted):
    """Pre-rotate B and C tensors with RoPE — single fused Triton kernel launch.

    cos/sin are computed in-kernel from raw angles, eliminating 4 intermediate
    float32 tensors from HBM.

    Returns bf16 rotated B tensors and float32 rotated C for the scan kernel.
    """
    batch, nc, L, ngroups, R, dstate = Bg.shape
    half_dstate = dstate // 2
    BLOCK_D = triton.next_power_of_2(half_dstate)
    N_pos = batch * nc * L
    N_ch_B = ngroups * R

    angles_flat = angles.contiguous().view(N_pos, half_dstate)
    angles_s_flat = angles_shifted.contiguous().view(N_pos, half_dstate)

    C_flat = C.contiguous().view(N_pos, ngroups, dstate)
    Bg_flat = Bg.contiguous().view(N_pos, N_ch_B, dstate)
    Bb_flat = Bb.contiguous().view(N_pos, N_ch_B, dstate)

    C_rot = torch.empty_like(C_flat)
    Bg_rot = torch.empty(N_pos, N_ch_B, dstate, device=Bg.device, dtype=torch.bfloat16)
    Bb_rot = torch.empty(N_pos, N_ch_B, dstate, device=Bb.device, dtype=torch.bfloat16)

    total_programs = N_pos * (ngroups + 2 * N_ch_B)
    _rope_prerotate_fused_kernel[(total_programs,)](
        C_flat, Bg_flat, Bb_flat,
        angles_flat, angles_s_flat,
        C_rot, Bg_rot, Bb_rot,
        C_flat.stride(0), C_flat.stride(1), C_flat.stride(2),
        Bg_flat.stride(0), Bg_flat.stride(1), Bg_flat.stride(2),
        angles_flat.stride(0), angles_flat.stride(1),
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
    # Strides for Bg/Bb (6D): b, c, l, g, r, n
    stride_Bg_b, stride_Bg_c, stride_Bg_l, stride_Bg_g, stride_Bg_r, stride_Bg_n,
    stride_Bb_b, stride_Bb_c, stride_Bb_l, stride_Bb_g, stride_Bb_r, stride_Bb_n,
    # Strides for x_dt_g/x_dt_b (6D): b, c, l, h, r, p
    stride_xg_b, stride_xg_c, stride_xg_l, stride_xg_h, stride_xg_r, stride_xg_p,
    stride_xb_b, stride_xb_c, stride_xb_l, stride_xb_h, stride_xb_r, stride_xb_p,
    # Strides for dA_cumsum (4D): b, c, l, h
    stride_dA_b, stride_dA_c, stride_dA_l, stride_dA_h,
    # Strides for C (5D): b, c, l, g, n
    stride_C_b, stride_C_c, stride_C_l, stride_C_g, stride_C_n,
    # Strides for out (5D): b, c, l, h, p
    stride_out_b, stride_out_c, stride_out_l, stride_out_h, stride_out_p,
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
    base_dA = pid_b * stride_dA_b + pid_c * stride_dA_c
    base_Bg = pid_b * stride_Bg_b + pid_c * stride_Bg_c + group_idx * stride_Bg_g
    base_Bb = pid_b * stride_Bb_b + pid_c * stride_Bb_c + group_idx * stride_Bb_g
    base_xg = pid_b * stride_xg_b + pid_c * stride_xg_c
    base_xb = pid_b * stride_xb_b + pid_c * stride_xb_c
    base_C = pid_b * stride_C_b + pid_c * stride_C_c + group_idx * stride_C_g

    # Load dA_cumsum for rows m
    dA_cs_m = tl.load(dA_cumsum_ptr + base_dA + offs_m * stride_dA_l + pid_h * stride_dA_h,
                       mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    # Output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-load C tiles (already rotated by pre-rotation kernel, float32).
    # Cast to bf16 for CB dot products — engages bf16 tensor cores (2x TF32).
    c_d0_offs = tl.arange(0, BLOCK_DSTATE)
    c_d0_mask = c_d0_offs < dstate

    c_tile_0 = tl.load(
        C_ptr + base_C
        + offs_m[:, None] * stride_C_l
        + c_d0_offs[None, :] * stride_C_n,
        mask=(offs_m[:, None] < chunk_size) & c_d0_mask[None, :],
        other=0.0).to(tl.bfloat16)

    c_d1_offs = BLOCK_DSTATE + tl.arange(0, BLOCK_DSTATE)
    c_d1_mask = c_d1_offs < dstate
    c_tile_1 = tl.load(
        C_ptr + base_C
        + offs_m[:, None] * stride_C_l
        + c_d1_offs[None, :] * stride_C_n,
        mask=(offs_m[:, None] < chunk_size) & c_d1_mask[None, :],
        other=0.0).to(tl.bfloat16)

    # --- Intra-chunk: tile-by-tile over k positions ---
    # Causal cutoff: skip k-blocks where all k > max(m) in this tile.
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)
    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # === Decay computed in registers, NEVER stored to global memory ===
        dA_cs_k = tl.load(dA_cumsum_ptr + base_dA + k_offs * stride_dA_l + pid_h * stride_dA_h,
                           mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))

        # Causal mask: only k <= m contributes
        causal = offs_m[:, None] >= k_offs[None, :]
        decay = tl.where(causal, decay, 0.0)  # (BLOCK_M, BLOCK_K)

        for r in range(mimo_rank):
            # Load pre-rotated Bg tiles (bf16, no RoPE needed)
            bg_tile_0 = tl.load(
                Bg_ptr + base_Bg
                + k_offs[:, None] * stride_Bg_l
                + r * stride_Bg_r
                + c_d0_offs[None, :] * stride_Bg_n,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            bg_tile_1 = tl.load(
                Bg_ptr + base_Bg
                + k_offs[:, None] * stride_Bg_l
                + r * stride_Bg_r
                + c_d1_offs[None, :] * stride_Bg_n,
                mask=(k_offs[:, None] < chunk_size) & c_d1_mask[None, :],
                other=0.0)

            # Load pre-rotated Bb tiles (bf16, no RoPE needed)
            bb_tile_0 = tl.load(
                Bb_ptr + base_Bb
                + k_offs[:, None] * stride_Bb_l
                + r * stride_Bb_r
                + c_d0_offs[None, :] * stride_Bb_n,
                mask=(k_offs[:, None] < chunk_size) & c_d0_mask[None, :],
                other=0.0)
            bb_tile_1 = tl.load(
                Bb_ptr + base_Bb
                + k_offs[:, None] * stride_Bb_l
                + r * stride_Bb_r
                + c_d1_offs[None, :] * stride_Bb_n,
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
                x_dt_g_ptr + base_xg
                + k_offs[:, None] * stride_xg_l
                + pid_h * stride_xg_h
                + r * stride_xg_r
                + offs_n[None, :] * stride_xg_p,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0).to(tl.float32)

            # x_dt_b[k, h, r, p]: (BLOCK_K, BLOCK_N) — loaded as bf16, cast to f32
            xb_vals = tl.load(
                x_dt_b_ptr + base_xb
                + k_offs[:, None] * stride_xb_l
                + pid_h * stride_xb_h
                + r * stride_xb_r
                + offs_n[None, :] * stride_xb_p,
                mask=k_mask[:, None] & (offs_n[None, :] < headdim),
                other=0.0).to(tl.float32)

            # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            # TF32 for these dots — scores are float32 (decay dynamic range 0->1).
            # x_dt loaded as bf16 and widened to float32 for TF32 tensor cores.
            acc += tl.dot(scores_g, xg_vals)
            acc += tl.dot(scores_b, xb_vals)

    # Store output
    base_out = pid_b * stride_out_b + pid_c * stride_out_c
    tl.store(out_ptr + base_out
             + offs_m[:, None] * stride_out_l
             + pid_h * stride_out_h
             + offs_n[None, :] * stride_out_p,
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                          angles, angles_shifted,
                          scale, chunk_size, mimo_rank):
    """
    Fused within-chunk computation for Mamba-3 Two-SSD with MIMO.

    RoPE pre-rotation is applied in a single fused Triton kernel (cos/sin computed
    in-kernel from raw angles, B outputs in bf16, C in float32), then the main
    scan kernel runs with reduced register pressure.

    Args:
        Bg:              (batch, nchunks, chunk_size, ngroups, R, dstate)
        Bb:              (batch, nchunks, chunk_size, ngroups, R, dstate)
        x_dt_g:          (batch, nchunks, chunk_size, nheads, R, headdim) bfloat16
        x_dt_b:          (batch, nchunks, chunk_size, nheads, R, headdim) bfloat16
        dA_cumsum:       (batch, nchunks, chunk_size, nheads) float32
        C:               (batch, nchunks, chunk_size, ngroups, dstate) float32
        angles:          (batch, nchunks, chunk_size, half_dstate) float32
        angles_shifted:  (batch, nchunks, chunk_size, half_dstate) float32
        scale:           float
        chunk_size:      int
        mimo_rank:       int
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
    BLOCK_DSTATE = max(32, triton.next_power_of_2(dstate // 2))

    # Pre-rotate B and C with RoPE (single fused Triton kernel launch,
    # cos/sin computed in-kernel from raw angles)
    Bg_rot, Bb_rot, C_rot = _prerotate_for_scan(
        Bg, Bb, C, angles, angles_shifted)

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
    nchunks: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    dstate: tl.constexpr,
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

    Grid: (cdiv(head_dim, BLOCK_P) * cdiv(dstate, BLOCK_N), batch * n_heads)
    """
    pid_tile = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decompose batch*head index
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    # Decompose tile index into (head_dim, dstate) tile coordinates
    n_tiles_n = tl.cdiv(dstate, BLOCK_N)
    pid_p = pid_tile // n_tiles_n
    pid_n = pid_tile % n_tiles_n

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tile_mask = (offs_p[:, None] < head_dim) & (offs_n[None, :] < dstate)

    # Base pointers for this (batch, head) slice
    base_dh = pid_b * stride_dh_b + pid_h * stride_dh_h
    base_cd = pid_b * stride_cd_b + pid_h * stride_cd_h
    base_as = pid_b * stride_as_b + pid_h * stride_as_h

    # Reusable offset pattern for (head_dim, d_state) tile
    tile_offsets = offs_p[:, None] * stride_dh_p + offs_n[None, :] * stride_dh_n
    tile_offsets_as = offs_p[:, None] * stride_as_p + offs_n[None, :] * stride_as_n

    # Initialize state to zeros
    state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    for c in range(nchunks):
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
        nchunks=n_chunks, n_heads=n_heads,
        head_dim=head_dim, dstate=d_state,
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return all_states


# =============================================================================
# Fused inference recurrence kernel (RoPE + trapezoidal state update + output)
# =============================================================================

@triton.jit
def _trapezoidal_recurrent_fwd_kernel(
    # Inputs
    x_ptr,           # (batch, T, n_heads, R, head_dim)  bf16
    A_ptr,           # (n_heads,)                        float32
    B_ptr,           # (batch, T, ngroups, R, dstate)    bf16
    C_ptr,           # (batch, T, ngroups, dstate)       bf16
    dt_ptr,          # (batch, T, n_heads)               float32
    lam_ptr,         # (batch, T, n_heads)               float32 (post-sigmoid)
    angles_ptr,      # (batch, T, half_dstate)           float32
    # In/out state (updated in-place)
    h_ptr,           # (batch, n_heads, head_dim, dstate) float32
    prev_bx_ptr,     # (batch, n_heads, head_dim, dstate) float32
    # Output
    y_ptr,           # (batch, T, n_heads, head_dim)     float32
    # Scalar
    scale,           # float
    # Strides for x (5D): batch, T, head, R, headdim
    stride_x_b, stride_x_t, stride_x_h, stride_x_r, stride_x_p,
    # Strides for B (5D): batch, T, group, R, dstate
    stride_B_b, stride_B_t, stride_B_g, stride_B_r, stride_B_n,
    # Strides for C (4D): batch, T, group, dstate
    stride_C_b, stride_C_t, stride_C_g, stride_C_n,
    # Strides for dt (3D): batch, T, head
    stride_dt_b, stride_dt_t, stride_dt_h,
    # Strides for angles (3D): batch, T, half_dstate
    stride_a_b, stride_a_t, stride_a_d,
    # Strides for h / prev_bx (4D): batch, head, headdim, dstate
    stride_h_b, stride_h_h, stride_h_p, stride_h_n,
    # Strides for y (4D): batch, T, head, headdim
    stride_y_b, stride_y_t, stride_y_h, stride_y_p,
    # Dimensions
    T: tl.constexpr,
    n_heads: tl.constexpr,
    ngroups: tl.constexpr,
    head_dim: tl.constexpr,
    dstate: tl.constexpr,
    R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused inference recurrence: RoPE rotation + trapezoidal state update + output.

    Each program owns BLOCK_P rows of the (head_dim, dstate) state matrix for
    one (batch, head) pair. Inner loop over dstate tiles, outer loop over T.

    Computes per timestep:
        angles → cos/sin (SFU)
        B_rot, C_rot = RoPE_rotate(B, C, cos, sin)
        Bx = sum_r x[r] ⊗ B_rot[r]           (rank-R outer product)
        h  = alpha * h + beta * prev_Bx + gamma * Bx    (trapezoidal update)
        y  = scale * C_rot @ h                  (output projection)

    Grid: (cdiv(head_dim, BLOCK_P), batch * n_heads)
    """
    pid_p = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    # Group index for B/C (GQA-style)
    heads_per_group: tl.constexpr = n_heads // ngroups
    pid_g = pid_h // heads_per_group

    half_dstate: tl.constexpr = dstate // 2

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = offs_p < head_dim

    # Base pointers for this (batch, head)
    base_h = pid_b * stride_h_b + pid_h * stride_h_h
    base_x = pid_b * stride_x_b + pid_h * stride_x_h
    base_dt = pid_b * stride_dt_b + pid_h * stride_dt_h
    base_B = pid_b * stride_B_b + pid_g * stride_B_g
    base_C = pid_b * stride_C_b + pid_g * stride_C_g
    base_a = pid_b * stride_a_b
    base_y = pid_b * stride_y_b + pid_h * stride_y_h

    # Load A for this head (scalar, constant across T)
    A_h = tl.load(A_ptr + pid_h)

    for t in range(T):
        # Load per-(batch, head) scalars for this timestep
        dt_t = tl.load(dt_ptr + base_dt + t * stride_dt_t)
        lam_t = tl.load(lam_ptr + base_dt + t * stride_dt_t)

        alpha = tl.exp(A_h * dt_t)
        beta = (1.0 - lam_t) * dt_t * alpha
        gamma = lam_t * dt_t

        y_accum = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for n_tile in range(tl.cdiv(dstate, BLOCK_N)):
            offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < dstate
            tile_mask = p_mask[:, None] & n_mask[None, :]

            # ── RoPE rotation setup ──
            pair_offs = (offs_n + half_dstate) % dstate
            angle_offs = offs_n % half_dstate
            angle_mask = angle_offs < half_dstate  # always true if dstate is power-of-2

            angles = tl.load(
                angles_ptr + base_a + t * stride_a_t + angle_offs * stride_a_d,
                mask=angle_mask, other=0.0)
            cos_v = tl.cos(angles)
            sin_v = tl.sin(angles)
            is_first = offs_n < half_dstate
            sign = tl.where(is_first, -1.0, 1.0)

            # ── Rotate C and cast to float32 ──
            c_raw = tl.load(
                C_ptr + base_C + t * stride_C_t + offs_n * stride_C_n,
                mask=n_mask, other=0.0).to(tl.float32)
            c_pair = tl.load(
                C_ptr + base_C + t * stride_C_t + pair_offs * stride_C_n,
                mask=n_mask, other=0.0).to(tl.float32)
            c_rot = c_raw * cos_v + sign * c_pair * sin_v  # (BLOCK_N,)

            # ── Rank-R outer product with rotated B ──
            bx_tile = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
            for r in range(R):
                # Load x[b, t, h, r, offs_p]
                x_r = tl.load(
                    x_ptr + base_x + t * stride_x_t + r * stride_x_r
                    + offs_p * stride_x_p,
                    mask=p_mask, other=0.0).to(tl.float32)  # (BLOCK_P,)

                # Load and rotate B[b, t, g, r, offs_n]
                b_raw = tl.load(
                    B_ptr + base_B + t * stride_B_t + r * stride_B_r
                    + offs_n * stride_B_n,
                    mask=n_mask, other=0.0).to(tl.float32)
                b_pair = tl.load(
                    B_ptr + base_B + t * stride_B_t + r * stride_B_r
                    + pair_offs * stride_B_n,
                    mask=n_mask, other=0.0).to(tl.float32)
                b_rot = b_raw * cos_v + sign * b_pair * sin_v  # (BLOCK_N,)

                bx_tile += x_r[:, None] * b_rot[None, :]

            # ── Load state tiles ──
            h_offsets = offs_p[:, None] * stride_h_p + offs_n[None, :] * stride_h_n
            h_tile = tl.load(h_ptr + base_h + h_offsets, mask=tile_mask, other=0.0)
            prev_bx_tile = tl.load(
                prev_bx_ptr + base_h + h_offsets, mask=tile_mask, other=0.0)

            # ── Trapezoidal state update ──
            h_tile = alpha * h_tile + beta * prev_bx_tile + gamma * bx_tile

            # ── Store updated state ──
            tl.store(h_ptr + base_h + h_offsets, h_tile, mask=tile_mask)
            tl.store(prev_bx_ptr + base_h + h_offsets, bx_tile, mask=tile_mask)

            # ── Accumulate output: y += C_rot @ h (reduction over dstate) ──
            y_accum += tl.sum(c_rot[None, :] * h_tile, axis=1)

        # Store output for this timestep
        tl.store(y_ptr + base_y + t * stride_y_t + offs_p * stride_y_p,
                 y_accum * scale, mask=p_mask)


def trapezoidal_recurrent_fwd(x, A, B, C, dt, lam, cum_angles, ssm_state, scale,
                               ngroups, n_heads):
    """Fused inference recurrence: RoPE + trapezoidal state update + output.

    Args:
        x:          (batch, T, n_heads, R, head_dim) bf16
        A:          (n_heads,) float32
        B:          (batch, T, ngroups, R, dstate) bf16 — raw (unrotated)
        C:          (batch, T, ngroups, dstate) bf16 — raw (unrotated)
        dt:         (batch, T, n_heads) float32
        lam:        (batch, T, n_heads) float32 — already sigmoid'd
        cum_angles: (batch, T, dstate//2) float32
        ssm_state:  dict with 'h' and 'prev_Bx', each (batch, n_heads, head_dim, dstate) float32
        scale:      float scalar
        ngroups:    int
        n_heads:    int
    Returns:
        y:         (batch, T, n_heads, head_dim) float32
        ssm_state: updated dict (h and prev_Bx modified in-place)
    """
    batch, T, _, R, head_dim = x.shape
    dstate = B.shape[-1]

    h = ssm_state['h']           # (batch, n_heads, head_dim, dstate) float32
    prev_bx = ssm_state['prev_Bx']  # same shape

    y = torch.empty(batch, T, n_heads, head_dim, device=x.device, dtype=torch.float32)

    BLOCK_P = min(64, triton.next_power_of_2(head_dim))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    num_warps = 2 if BLOCK_P * BLOCK_N <= 2048 else 4

    grid = (triton.cdiv(head_dim, BLOCK_P), batch * n_heads)

    _trapezoidal_recurrent_fwd_kernel[grid](
        x, A, B, C, dt, lam, cum_angles,
        h, prev_bx,
        y,
        scale,
        # x strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        # B strides
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        # C strides
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        # dt strides (shared layout with lam)
        dt.stride(0), dt.stride(1), dt.stride(2),
        # angles strides
        cum_angles.stride(0), cum_angles.stride(1), cum_angles.stride(2),
        # h strides (shared with prev_bx)
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        # y strides
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        # dimensions
        T=T, n_heads=n_heads, ngroups=ngroups,
        head_dim=head_dim, dstate=dstate, R=R,
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return y, {'h': h, 'prev_Bx': prev_bx}


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
    key=['head_dim', 'dstate', 'chunk_size', 'mimo_rank'],
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
    nheads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nchunks: tl.constexpr,
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

    Grid: (cdiv(head_dim, BLOCK_P) * cdiv(dstate, BLOCK_N), batch * nchunks, n_heads)
    """
    pid_tile = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    # Decompose tile into (head_dim, dstate) coordinates
    n_tiles_n = tl.cdiv(dstate, BLOCK_N)
    pid_p = pid_tile // n_tiles_n
    pid_n = pid_tile % n_tiles_n

    group_idx = pid_h // nheads_per_group

    # Tile offsets
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = offs_p < head_dim
    n_mask = offs_n < dstate

    # Base pointers
    base_xg = pid_b * stride_xg_b + pid_c * stride_xg_c + pid_h * stride_xg_h
    base_xb = pid_b * stride_xb_b + pid_c * stride_xb_c + pid_h * stride_xb_h
    base_Bg = pid_b * stride_Bg_b + pid_c * stride_Bg_c + group_idx * stride_Bg_g
    base_Bb = pid_b * stride_Bb_b + pid_c * stride_Bb_c + group_idx * stride_Bb_g
    base_dA = pid_b * stride_dA_b + pid_c * stride_dA_c + pid_h * stride_dA_h

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
    base_out = pid_b * stride_dh_b + pid_c * stride_dh_c + pid_h * stride_dh_h
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
    nheads_per_group = n_heads // ngroups

    delta_h = torch.empty(batch, nc, n_heads, head_dim, d_state,
                           device=x_dt_g.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(head_dim, META['BLOCK_P']) * triton.cdiv(d_state, META['BLOCK_N']),
        batch * nc,
        n_heads,
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
        n_heads=n_heads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        head_dim=head_dim, dstate=d_state,
        chunk_size=L, nchunks=nc, mimo_rank=R,
    )

    return delta_h


# =============================================================================
# Fused inter-chunk output kernel (matmul + decay scaling + y_intra addition)
# =============================================================================

@triton.autotune(
    configs=[
        # Full L coverage (L=128 in one tile)
        triton.Config({'BLOCK_L': 128, 'BLOCK_P': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_L': 128, 'BLOCK_P': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_L': 128, 'BLOCK_P': 64, 'BLOCK_N': 128}, num_stages=1, num_warps=8),
        # Balanced tiles
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 64, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
        # Smaller tiles for higher occupancy
        triton.Config({'BLOCK_L': 32, 'BLOCK_P': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_L': 64, 'BLOCK_P': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    ],
    key=['chunk_size', 'head_dim', 'dstate'],
)
@triton.jit
def _inter_chunk_output_fwd_kernel(
    # Inputs
    C_rot_ptr,         # (B, nc, L, ngroups, d_state) float32
    all_states_ptr,    # (B, nc, n_heads, head_dim, d_state) float32
    cum_log_dA_ptr,    # (B, nc, L, n_heads) float32
    y_intra_ptr,       # (B, nc, L, n_heads, head_dim) bf16
    # Output
    y_ptr,             # (B, nc, L, n_heads, head_dim) bf16
    # Strides for C_rot (5D): batch, chunk, seqlen, group, dstate
    stride_C_b, stride_C_c, stride_C_l, stride_C_g, stride_C_n,
    # Strides for all_states (5D): batch, chunk, head, headdim, dstate
    stride_s_b, stride_s_c, stride_s_h, stride_s_p, stride_s_n,
    # Strides for cum_log_dA (4D): batch, chunk, seqlen, head
    stride_dA_b, stride_dA_c, stride_dA_l, stride_dA_h,
    # Strides for y_intra (5D): batch, chunk, seqlen, head, headdim
    stride_yi_b, stride_yi_c, stride_yi_l, stride_yi_h, stride_yi_p,
    # Strides for y (5D): batch, chunk, seqlen, head, headdim
    stride_y_b, stride_y_c, stride_y_l, stride_y_h, stride_y_p,
    # Dimensions
    chunk_size: tl.constexpr,
    head_dim: tl.constexpr,
    dstate: tl.constexpr,
    n_heads: tl.constexpr,
    ngroups: tl.constexpr,
    nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr,
    scale: tl.constexpr,
    # Block sizes (autotuned)
    BLOCK_L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused inter-chunk output: C_rot @ state matmul + decay scaling + y_intra add.

    Eliminates the intermediate state_contribution tensor by fusing:
      1. einsum(C_rot, all_states)  →  matmul reduction over dstate
      2. * scale * exp(cum_log_dA)  →  pointwise epilogue
      3. + y_intra                  →  addition in registers

    y[b,c,i,h,p] = y_intra[b,c,i,h,p]
                  + scale * exp(cum_log_dA[b,c,i,h])
                    * sum_n(C_rot[b,c,i,g(h),n] * state[b,c,h,p,n])

    Grid: (cdiv(L, BLOCK_L) * cdiv(head_dim, BLOCK_P), batch * nchunks, n_heads)
    """
    pid_tile = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Decompose batch * nchunks
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    # Decompose tile into (L, head_dim) coordinates
    n_tiles_p = tl.cdiv(head_dim, BLOCK_P)
    pid_l = pid_tile // n_tiles_p
    pid_p = pid_tile % n_tiles_p

    group_idx = pid_h // nheads_per_group

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    l_mask = offs_l < chunk_size
    p_mask = offs_p < head_dim

    # Base pointers for this (batch, chunk, head/group)
    base_C = pid_b * stride_C_b + pid_c * stride_C_c + group_idx * stride_C_g
    base_s = pid_b * stride_s_b + pid_c * stride_s_c + pid_h * stride_s_h
    base_dA = pid_b * stride_dA_b + pid_c * stride_dA_c + pid_h * stride_dA_h
    base_yi = pid_b * stride_yi_b + pid_c * stride_yi_c + pid_h * stride_yi_h
    base_y = pid_b * stride_y_b + pid_c * stride_y_c + pid_h * stride_y_h

    # Accumulator: (BLOCK_L, BLOCK_P) for C_rot @ state^T
    acc = tl.zeros((BLOCK_L, BLOCK_P), dtype=tl.float32)

    # Reduction over dstate
    for n_start in range(0, dstate, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < dstate

        # C_rot[b, c, l, g(h), n]: (BLOCK_L, BLOCK_N) float32
        c_tile = tl.load(
            C_rot_ptr + base_C
            + offs_l[:, None] * stride_C_l
            + offs_n[None, :] * stride_C_n,
            mask=l_mask[:, None] & n_mask[None, :],
            other=0.0)

        # state[b, c, h, p, n]: (BLOCK_P, BLOCK_N) float32
        s_tile = tl.load(
            all_states_ptr + base_s
            + offs_p[:, None] * stride_s_p
            + offs_n[None, :] * stride_s_n,
            mask=p_mask[:, None] & n_mask[None, :],
            other=0.0)

        # (BLOCK_L, BLOCK_N) @ (BLOCK_N, BLOCK_P) -> (BLOCK_L, BLOCK_P)
        acc += tl.dot(c_tile, tl.trans(s_tile))

    # Epilogue: scale by decay, add y_intra
    dA_vals = tl.load(
        cum_log_dA_ptr + base_dA + offs_l * stride_dA_l,
        mask=l_mask, other=0.0)
    decay_scale = tl.exp(dA_vals) * scale  # (BLOCK_L,)
    acc = acc * decay_scale[:, None]

    # Load y_intra (bf16 → f32) and add
    yi = tl.load(
        y_intra_ptr + base_yi
        + offs_l[:, None] * stride_yi_l
        + offs_p[None, :] * stride_yi_p,
        mask=l_mask[:, None] & p_mask[None, :],
        other=0.0).to(tl.float32)
    acc = acc + yi

    # Store output as bf16
    tl.store(
        y_ptr + base_y
        + offs_l[:, None] * stride_y_l
        + offs_p[None, :] * stride_y_p,
        acc.to(tl.bfloat16),
        mask=l_mask[:, None] & p_mask[None, :])


def inter_chunk_output_fwd(C_rot, all_states, cum_log_dA, y_intra, scale, nheads_per_group):
    """Fused inter-chunk output: einsum + decay scaling + y_intra addition.

    Replaces the 3-step PyTorch computation (cuBLAS einsum → pointwise decay
    multiply → y_intra addition) with a single Triton kernel that never
    materializes the intermediate state_contribution tensor.

    Args:
        C_rot:             (B, nc, L, ngroups, d_state) float32
        all_states:        (B, nc, n_heads, head_dim, d_state) float32
        cum_log_dA:        (B, nc, L, n_heads) float32
        y_intra:           (B, nc, L, n_heads, head_dim) bf16
        scale:             float
        nheads_per_group:  int (n_heads // ngroups)
    Returns:
        y: (B, nc, L, n_heads, head_dim) bf16
    """
    batch, nc, L, ngroups, d_state = C_rot.shape
    n_heads = cum_log_dA.shape[-1]
    head_dim = y_intra.shape[-1]

    y = torch.empty_like(y_intra)

    grid = lambda META: (
        triton.cdiv(L, META['BLOCK_L']) * triton.cdiv(head_dim, META['BLOCK_P']),
        batch * nc,
        n_heads,
    )

    _inter_chunk_output_fwd_kernel[grid](
        C_rot, all_states, cum_log_dA, y_intra, y,
        # C_rot strides
        C_rot.stride(0), C_rot.stride(1), C_rot.stride(2),
        C_rot.stride(3), C_rot.stride(4),
        # all_states strides
        all_states.stride(0), all_states.stride(1), all_states.stride(2),
        all_states.stride(3), all_states.stride(4),
        # cum_log_dA strides
        cum_log_dA.stride(0), cum_log_dA.stride(1),
        cum_log_dA.stride(2), cum_log_dA.stride(3),
        # y_intra strides
        y_intra.stride(0), y_intra.stride(1), y_intra.stride(2),
        y_intra.stride(3), y_intra.stride(4),
        # y strides
        y.stride(0), y.stride(1), y.stride(2),
        y.stride(3), y.stride(4),
        # Dimensions
        chunk_size=L, head_dim=head_dim, dstate=d_state,
        n_heads=n_heads, ngroups=ngroups, nheads_per_group=nheads_per_group, nchunks=nc,
        scale=scale,
    )

    return y


# =============================================================================
# Fused SSD prep kernels (replaces ~20 PyTorch ops in _two_ssd_forward_triton)
# =============================================================================

@triton.jit
def _ssd_prep_xdt_kernel(
    # Inputs (flat T-padded layout)
    x_heads_ptr,       # (B, T_pad, nh, R, hd)
    dt_ptr,            # (B, T_pad, nh)
    lambda_raw_ptr,    # (B, T_pad, nh)
    A_ptr,             # (nh,)
    # Outputs (chunked layout)
    x_dt_g_ptr,        # (B, nc, L, nh, R, hd) bf16
    x_dt_b_ptr,        # (B, nc, L, nh, R, hd) bf16
    # Strides for x_heads (5D): b, t, h, r, p
    stride_x_b, stride_x_t, stride_x_h, stride_x_r, stride_x_p,
    # Strides for dt (3D): b, t, h
    stride_dt_b, stride_dt_t, stride_dt_h,
    # Strides for lambda_raw (3D): b, t, h
    stride_lam_b, stride_lam_t, stride_lam_h,
    # Stride for A (1D)
    stride_A,
    # Strides for x_dt_g/x_dt_b output (6D): b, c, l, h, r, p
    stride_xo_b, stride_xo_c, stride_xo_l, stride_xo_h, stride_xo_r, stride_xo_p,
    # Dimensions
    T_padded,
    chunk_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    mimo_rank: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Fused x_dt_g and x_dt_b computation with lambda gating.

    Computes:
      lam = sigmoid(lambda_raw[b,t,h])
      alpha = exp(A[h] * dt[b,t,h])
      x_dt_g[b,t,h,r,p] = x[b,t,h,r,p] * lam * dt[b,t,h]
      x_dt_b[b,t,h,r,p] = x[b,t-1,h,r,p] * (1-lam) * alpha * dt[b,t,h]  (zero if t==0)

    Grid: (B * T_padded, nh * R)
    """
    pid_bt = tl.program_id(0)   # flattened (batch, time) index
    pid_hr = tl.program_id(1)   # flattened (head, rank) index

    b = pid_bt // T_padded
    t = pid_bt % T_padded
    h = pid_hr // mimo_rank
    r = pid_hr % mimo_rank

    # Chunk coordinates for output
    c = t // chunk_size
    l = t % chunk_size

    # Load scalars
    dt_val = tl.load(dt_ptr + b * stride_dt_b + t * stride_dt_t + h * stride_dt_h).to(tl.float32)
    lam_raw = tl.load(lambda_raw_ptr + b * stride_lam_b + t * stride_lam_t + h * stride_lam_h).to(tl.float32)
    A_val = tl.load(A_ptr + h * stride_A).to(tl.float32)

    lam = tl.sigmoid(lam_raw)
    alpha = tl.exp(A_val * dt_val)
    coeff_beta = (1.0 - lam) * alpha

    # Tile over head_dim
    p_offs = tl.arange(0, BLOCK_P)
    p_mask = p_offs < head_dim

    # Load x[b, t, h, r, p]
    base_x_cur = b * stride_x_b + t * stride_x_t + h * stride_x_h + r * stride_x_r
    x_cur = tl.load(x_heads_ptr + base_x_cur + p_offs * stride_x_p,
                     mask=p_mask, other=0.0).to(tl.float32)

    # x_dt_g = x * lam * dt
    x_dt_g_val = (x_cur * lam * dt_val).to(tl.bfloat16)

    # Load x[b, t-1, h, r, p] for beta path (zero if t==0)
    # Use max(t-1, 0) for the address to avoid negative offsets; mask ensures
    # the load returns 0.0 when t==0 regardless.
    t_prev = tl.maximum(t - 1, 0)
    x_prev = tl.load(x_heads_ptr + b * stride_x_b + t_prev * stride_x_t
                     + h * stride_x_h + r * stride_x_r + p_offs * stride_x_p,
                     mask=p_mask & (t > 0), other=0.0).to(tl.float32)

    # x_dt_b = x_prev * (1-lam) * alpha * dt
    x_dt_b_val = (x_prev * coeff_beta * dt_val).to(tl.bfloat16)

    # Store to chunked output layout
    base_out = b * stride_xo_b + c * stride_xo_c + l * stride_xo_l + h * stride_xo_h + r * stride_xo_r
    tl.store(x_dt_g_ptr + base_out + p_offs * stride_xo_p, x_dt_g_val, mask=p_mask)
    tl.store(x_dt_b_ptr + base_out + p_offs * stride_xo_p, x_dt_b_val, mask=p_mask)


@triton.jit
def _ssd_prep_decay_kernel(
    # Inputs
    dt_ptr,             # (B, T_pad, nh)
    A_ptr,              # (nh,)
    # Outputs
    cum_log_dA_ptr,     # (B, nc, L, nh) f32
    chunk_decay_ptr,    # (B, nc, nh) f32
    # Stride for A (1D)
    stride_A,
    # Strides for dt (3D): b, t, h
    stride_dt_b, stride_dt_t, stride_dt_h,
    # Strides for cum_log_dA (4D): b, c, l, h
    stride_dA_b, stride_dA_c, stride_dA_l, stride_dA_h,
    # Strides for chunk_decay (3D): b, c, h
    stride_cd_b, stride_cd_c, stride_cd_h,
    # Dimensions
    chunk_size: tl.constexpr,
    nchunks: tl.constexpr,
):
    """Sequential cumsum of log(dA) within each chunk, plus chunk_decay side-output.

    Computes:
      cum_log_dA[b,c,l,h] = sum_{i=0}^{l} A[h] * dt[b, c*L+i, h]
      chunk_decay[b,c,h] = exp(cum_log_dA[b,c,L-1,h])

    Grid: (B * nc, nh)
    """
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_bc // nchunks
    c = pid_bc % nchunks

    A_val = tl.load(A_ptr + pid_h * stride_A).to(tl.float32)

    base_dt = b * stride_dt_b + pid_h * stride_dt_h
    base_dA = b * stride_dA_b + c * stride_dA_c + pid_h * stride_dA_h

    running_sum = 0.0
    for l in range(chunk_size):
        t = c * chunk_size + l
        dt_val = tl.load(dt_ptr + base_dt + t * stride_dt_t).to(tl.float32)
        running_sum += A_val * dt_val
        tl.store(cum_log_dA_ptr + base_dA + l * stride_dA_l, running_sum)

    # chunk_decay = exp(cum_log_dA at last position)
    tl.store(chunk_decay_ptr + b * stride_cd_b + c * stride_cd_c + pid_h * stride_cd_h,
             tl.exp(running_sum))


@triton.jit
def _ssd_shift_kernel(
    # Inputs (flat T-padded layout)
    B_raw_ptr,          # (B, T_pad, ng, R, ds)
    # Outputs (chunked layout)
    Bb_ptr,             # (B, nc, L, ng, R, ds)
    # Strides for B_raw (5D): b, t, g, r, d
    stride_B_b, stride_B_t, stride_B_g, stride_B_r, stride_B_d,
    # Strides for Bb output (6D): b, c, l, g, r, d
    stride_Bb_b, stride_Bb_c, stride_Bb_l, stride_Bb_g, stride_Bb_r, stride_Bb_d,
    # Dimensions
    T_padded,
    chunk_size: tl.constexpr,
    ngroups: tl.constexpr,
    mimo_rank: tl.constexpr,
    dstate: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy-with-shift: B_raw[t-1] -> Bb[t].

    Grid: (B * T_padded, ngroups * R)
    """
    pid_bt = tl.program_id(0)
    pid_ch = tl.program_id(1)

    b = pid_bt // T_padded
    t = pid_bt % T_padded
    c = t // chunk_size
    l = t % chunk_size

    g = pid_ch // mimo_rank
    r = pid_ch % mimo_rank

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < dstate

    if t > 0:
        base_in = b * stride_B_b + (t - 1) * stride_B_t + g * stride_B_g + r * stride_B_r
        vals = tl.load(B_raw_ptr + base_in + d_offs * stride_B_d,
                      mask=d_mask, other=0.0)
    else:
        vals = tl.zeros((BLOCK_D,), dtype=tl.float32)

    base_out = b * stride_Bb_b + c * stride_Bb_c + l * stride_Bb_l + g * stride_Bb_g + r * stride_Bb_r
    tl.store(Bb_ptr + base_out + d_offs * stride_Bb_d, vals, mask=d_mask)


# =============================================================================
# Fused RoPE angle computation kernel
# =============================================================================
# Replaces _compute_rope_angles (3 PyTorch ops: mean, multiply, cumsum) +
# the angle-shift sub-kernel in ssd_prep_fwd with a single Triton kernel.
# Produces both angles_chunked and angles_shifted in chunked layout directly.


@triton.jit
def _compute_rope_angles_kernel(
    # Inputs
    dt_ptr,              # (B, T_padded, n_heads)
    theta_raw_ptr,       # (B, T_padded, half_dstate)
    # Outputs
    angles_chunked_ptr,  # (B, nc, L, half_dstate) float32
    angles_shifted_ptr,  # (B, nc, L, half_dstate) float32
    # dt strides (3D): b, t, h
    stride_dt_b, stride_dt_t, stride_dt_h,
    # theta_raw strides (3D): b, t, d
    stride_th_b, stride_th_t, stride_th_d,
    # output strides (4D): b, c, l, d — same layout for both outputs
    stride_out_b, stride_out_c, stride_out_l, stride_out_d,
    # Dimensions
    T_original,
    half_dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    n_heads: tl.constexpr,
    T_padded: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused cumulative RoPE angle computation with shift.

    Each program handles one (batch, d_block) slice, scanning T sequentially.
    Produces:
        angles_chunked[b,c,l,d] = -cumsum(dt_mean * theta_raw)[t]  (current)
        angles_shifted[b,c,l,d] = -cumsum(dt_mean * theta_raw)[t-1] (previous, 0 at t=0)
    Padding positions (t >= T_original) are zero.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < half_dstate
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < n_heads

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(T_padded):
        c = t // chunk_size
        l = t % chunk_size
        out_base = pid_b * stride_out_b + c * stride_out_c + l * stride_out_l

        if t < T_original:
            # Load dt[b, t, 0:n_heads] and compute mean
            dt_base = pid_b * stride_dt_b + t * stride_dt_t
            dt_vals = tl.load(dt_ptr + dt_base + h_offs * stride_dt_h,
                              mask=h_mask, other=0.0)
            dt_mean = tl.sum(dt_vals, axis=0) / n_heads

            # Load theta_raw[b, t, d_offs]
            th_base = pid_b * stride_th_b + t * stride_th_t
            theta = tl.load(theta_raw_ptr + th_base + d_offs * stride_th_d,
                            mask=d_mask, other=0.0)

            raw_angle = dt_mean * theta

            # Store shifted angle (previous accumulator, negated)
            tl.store(angles_shifted_ptr + out_base + d_offs * stride_out_d,
                     -acc, mask=d_mask)

            # Update accumulator
            acc += raw_angle

            # Store current angle (updated accumulator, negated)
            tl.store(angles_chunked_ptr + out_base + d_offs * stride_out_d,
                     -acc, mask=d_mask)
        else:
            # Padding: store zeros
            tl.store(angles_shifted_ptr + out_base + d_offs * stride_out_d,
                     tl.zeros((BLOCK_D,), dtype=tl.float32), mask=d_mask)
            tl.store(angles_chunked_ptr + out_base + d_offs * stride_out_d,
                     tl.zeros((BLOCK_D,), dtype=tl.float32), mask=d_mask)


def compute_rope_angles_fwd(dt, theta_raw, T_original, chunk_size):
    """Fused RoPE angle computation: dt_mean, multiply, cumsum, shift — one kernel.

    Args:
        dt:          (B, T_padded, n_heads) float32 — already padded
        theta_raw:   (B, T_padded, half_dstate) — already padded
        T_original:  int — original T before padding
        chunk_size:  int
    Returns:
        angles_chunked:  (B, nc, L, half_dstate) float32
        angles_shifted:  (B, nc, L, half_dstate) float32
    """
    batch, T_padded, n_heads = dt.shape
    half_dstate = theta_raw.shape[2]
    nc = T_padded // chunk_size
    L = chunk_size

    BLOCK_D = triton.next_power_of_2(half_dstate)
    BLOCK_H = triton.next_power_of_2(n_heads)

    angles_chunked = torch.zeros(batch, nc, L, half_dstate,
                                  device=dt.device, dtype=torch.float32)
    angles_shifted = torch.zeros(batch, nc, L, half_dstate,
                                  device=dt.device, dtype=torch.float32)

    grid = (batch, triton.cdiv(half_dstate, BLOCK_D))
    _compute_rope_angles_kernel[grid](
        dt, theta_raw,
        angles_chunked, angles_shifted,
        dt.stride(0), dt.stride(1), dt.stride(2),
        theta_raw.stride(0), theta_raw.stride(1), theta_raw.stride(2),
        angles_chunked.stride(0), angles_chunked.stride(1),
        angles_chunked.stride(2), angles_chunked.stride(3),
        T_original=T_original,
        half_dstate=half_dstate, chunk_size=L, n_heads=n_heads,
        T_padded=T_padded,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H,
    )
    return angles_chunked, angles_shifted


def ssd_prep_fwd(x_heads, dt, lambda_raw, A, B_raw, chunk_size):
    """Fused SSD preprocessing: lambda gating, x*dt, B shift, cumsum, chunk_decay.

    Replaces ~20 PyTorch ops with 3 Triton sub-kernels.
    RoPE angles are handled separately by compute_rope_angles_fwd.

    Args:
        x_heads:    (B, T_pad, nh, R, hd) -- already padded to chunk boundary
        dt:         (B, T_pad, nh)
        lambda_raw: (B, T_pad, nh)
        A:          (nh,)
        B_raw:      (B, T_pad, ng, R, ds)
        chunk_size: int
    Returns:
        x_dt_g:           (B, nc, L, nh, R, hd) bf16
        x_dt_b:           (B, nc, L, nh, R, hd) bf16
        Bb:               (B, nc, L, ng, R, ds)
        cum_log_dA:       (B, nc, L, nh) f32
        chunk_decay:      (B, nc, nh) f32
    """
    batch, T_pad, n_heads, R, head_dim = x_heads.shape
    ngroups = B_raw.shape[2]
    d_state = B_raw.shape[4]
    nc = T_pad // chunk_size
    L = chunk_size

    # Allocate outputs
    x_dt_g = torch.empty(batch, nc, L, n_heads, R, head_dim,
                          device=x_heads.device, dtype=torch.bfloat16)
    x_dt_b = torch.empty_like(x_dt_g)
    Bb = torch.empty(batch, nc, L, ngroups, R, d_state,
                      device=B_raw.device, dtype=B_raw.dtype)
    cum_log_dA = torch.empty(batch, nc, L, n_heads,
                              device=dt.device, dtype=torch.float32)
    chunk_decay = torch.empty(batch, nc, n_heads,
                               device=dt.device, dtype=torch.float32)

    # Sub-kernel A: x_dt_g and x_dt_b
    BLOCK_P = min(128, triton.next_power_of_2(head_dim))
    _ssd_prep_xdt_kernel[(batch * T_pad, n_heads * R)](
        x_heads, dt, lambda_raw, A,
        x_dt_g, x_dt_b,
        x_heads.stride(0), x_heads.stride(1), x_heads.stride(2),
        x_heads.stride(3), x_heads.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2),
        lambda_raw.stride(0), lambda_raw.stride(1), lambda_raw.stride(2),
        A.stride(0),
        x_dt_g.stride(0), x_dt_g.stride(1), x_dt_g.stride(2),
        x_dt_g.stride(3), x_dt_g.stride(4), x_dt_g.stride(5),
        T_padded=T_pad,
        chunk_size=L, n_heads=n_heads,
        head_dim=head_dim, mimo_rank=R,
        BLOCK_P=BLOCK_P,
    )

    # Sub-kernel B: cum_log_dA and chunk_decay
    _ssd_prep_decay_kernel[(batch * nc, n_heads)](
        dt, A,
        cum_log_dA, chunk_decay,
        A.stride(0),
        dt.stride(0), dt.stride(1), dt.stride(2),
        cum_log_dA.stride(0), cum_log_dA.stride(1),
        cum_log_dA.stride(2), cum_log_dA.stride(3),
        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
        chunk_size=L, nchunks=nc,
    )

    # Sub-kernel C: Bb (B_raw shift)
    BLOCK_D = triton.next_power_of_2(d_state)
    _ssd_shift_kernel[(batch * T_pad, ngroups * R)](
        B_raw,
        Bb,
        B_raw.stride(0), B_raw.stride(1), B_raw.stride(2),
        B_raw.stride(3), B_raw.stride(4),
        Bb.stride(0), Bb.stride(1), Bb.stride(2),
        Bb.stride(3), Bb.stride(4), Bb.stride(5),
        T_padded=T_pad,
        chunk_size=L, ngroups=ngroups, mimo_rank=R,
        dstate=d_state,
        BLOCK_D=BLOCK_D,
    )

    return x_dt_g, x_dt_b, Bb, cum_log_dA, chunk_decay


# =============================================================================
# Custom ops for torch.compile compatibility (forward + fake only)
# =============================================================================
# These wrap the Triton kernel launchers above with torch.library.custom_op
# so torch.compile can trace through them (opaque forward, PyTorch backward).
# Backward registration is done in mamba3.py where the PyTorch reference
# implementations live.


# ---- Within-chunk (mamba3_intra) ----

@torch.library.custom_op("nanochat::mamba3_intra_fwd", mutates_args=())
def _mamba3_intra_fwd_op(
    Bg: torch.Tensor, Bb: torch.Tensor,
    x_dt_g: torch.Tensor, x_dt_b: torch.Tensor,
    dA_cumsum: torch.Tensor, C: torch.Tensor,
    angles: torch.Tensor, angles_shifted: torch.Tensor,
    scale: float, chunk_size: int, mimo_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                 angles, angles_shifted,
                                 scale, chunk_size, mimo_rank)

@_mamba3_intra_fwd_op.register_fake
def _mamba3_intra_fwd_fake(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                            angles, angles_shifted,
                            scale, chunk_size, mimo_rank):
    batch, nchunks, L, nheads, R, headdim = x_dt_g.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    return (
        x_dt_g.new_empty(batch, nchunks, L, nheads, headdim, dtype=torch.bfloat16),
        Bg.new_empty(batch, nchunks, L, ngroups, R, dstate, dtype=torch.bfloat16),
        Bb.new_empty(batch, nchunks, L, ngroups, R, dstate, dtype=torch.bfloat16),
        C.new_empty(batch, nchunks, L, ngroups, dstate),
    )


# ---- Inter-chunk output ----

@torch.library.custom_op("nanochat::inter_chunk_output_fwd", mutates_args=())
def _inter_chunk_output_fwd_op(
    C_rot: torch.Tensor,
    all_states: torch.Tensor,
    cum_log_dA: torch.Tensor,
    y_intra: torch.Tensor,
    scale: float,
    nheads_per_group: int,
) -> torch.Tensor:
    return inter_chunk_output_fwd(
        C_rot, all_states, cum_log_dA, y_intra, scale, nheads_per_group)

@_inter_chunk_output_fwd_op.register_fake
def _inter_chunk_output_fwd_fake(
        C_rot, all_states, cum_log_dA, y_intra, scale, nheads_per_group):
    return y_intra.new_empty(y_intra.shape, dtype=y_intra.dtype)


# ---- Delta_h ----

@torch.library.custom_op("nanochat::delta_h_fwd", mutates_args=())
def _delta_h_fwd_op(
    x_dt_g: torch.Tensor,
    x_dt_b: torch.Tensor,
    Bg_rot: torch.Tensor,
    Bb_rot: torch.Tensor,
    cum_log_dA: torch.Tensor,
) -> torch.Tensor:
    return delta_h_fwd(x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA)

@_delta_h_fwd_op.register_fake
def _delta_h_fwd_fake(x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA):
    batch, nc, L, n_heads, R, head_dim = x_dt_g.shape
    d_state = Bg_rot.shape[5]
    return x_dt_g.new_empty(batch, nc, n_heads, head_dim, d_state,
                            dtype=torch.float32)


# ---- State propagation ----

@torch.library.custom_op("nanochat::state_propagation_fwd", mutates_args=())
def _state_propagation_fwd_op(
    delta_h: torch.Tensor,
    chunk_decay: torch.Tensor,
) -> torch.Tensor:
    return state_propagation_fwd(delta_h, chunk_decay)

@_state_propagation_fwd_op.register_fake
def _state_propagation_fwd_fake(delta_h, chunk_decay):
    return delta_h.new_empty(delta_h.shape, dtype=torch.float32)


# ---- RoPE angle computation ----

@torch.library.custom_op("nanochat::compute_rope_angles_fwd", mutates_args=())
def _compute_rope_angles_fwd_op(
    dt: torch.Tensor,
    theta_raw: torch.Tensor,
    T_original: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_rope_angles_fwd(dt, theta_raw, T_original, chunk_size)

@_compute_rope_angles_fwd_op.register_fake
def _compute_rope_angles_fwd_fake(dt, theta_raw, T_original, chunk_size):
    batch, T_padded, _ = dt.shape
    half_dstate = theta_raw.shape[2]
    nc = T_padded // chunk_size
    L = chunk_size
    return (
        dt.new_empty(batch, nc, L, half_dstate, dtype=torch.float32),
        dt.new_empty(batch, nc, L, half_dstate, dtype=torch.float32),
    )


# ---- SSD prep ----

@torch.library.custom_op("nanochat::ssd_prep_fwd", mutates_args=())
def _ssd_prep_fwd_op(
    x_heads: torch.Tensor,
    dt: torch.Tensor,
    lambda_raw: torch.Tensor,
    A: torch.Tensor,
    B_raw: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    return ssd_prep_fwd(x_heads, dt, lambda_raw, A, B_raw, chunk_size)

@_ssd_prep_fwd_op.register_fake
def _ssd_prep_fwd_fake(x_heads, dt, lambda_raw, A, B_raw, chunk_size):
    B, T_pad, nh, R, hd = x_heads.shape
    ng = B_raw.shape[2]
    ds = B_raw.shape[4]
    nc = T_pad // chunk_size
    L = chunk_size
    return (
        x_heads.new_empty(B, nc, L, nh, R, hd, dtype=torch.bfloat16),
        x_heads.new_empty(B, nc, L, nh, R, hd, dtype=torch.bfloat16),
        B_raw.new_empty(B, nc, L, ng, R, ds),
        dt.new_empty(B, nc, L, nh, dtype=torch.float32),
        dt.new_empty(B, nc, nh, dtype=torch.float32),
    )
