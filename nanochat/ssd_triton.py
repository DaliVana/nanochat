"""
Triton kernels for Mamba-2 SSD (Structured State Space Duality).

Following the official Mamba-2 implementation, we decompose the chunked SSD into
5 kernel launches:
  1. chunk_cumsum    — cumsum(A * dt) within each chunk
  2. bmm_chunk       — CB = C @ B^T per chunk (in grouped space)
  3. chunk_state     — per-chunk state delta: delta_h = sum_l exp(dA_last - dA_l) * dt_l * x_l * B_l^T
  4. state_passing   — sequential scan propagating states across chunks
  5. chunk_scan      — fused intra-chunk output: y = (CB * L) @ (x*dt) + C @ prev_states
                       L (decay matrix) is NEVER materialized — computed tile-by-tile in registers.

References:
- Official: https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton
- Mamba-2 paper: https://arxiv.org/abs/2405.21060
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Kernel 1: chunk_cumsum — compute dA_cumsum = cumsum(A * dt) per chunk
# =============================================================================
# This is simple enough to stay in PyTorch (torch.cumsum is well-optimized).

def chunk_cumsum(dt, A, chunk_size):
    """
    Args:
        dt: (batch, seqlen, nheads)
        A:  (nheads,)
    Returns:
        dA_cumsum: (batch, nchunks, chunk_size, nheads) float32
        dt_chunked: (batch, nchunks, chunk_size, nheads) float32
    """
    B, T, H = dt.shape
    nchunks = T // chunk_size
    dt_chunked = dt.view(B, nchunks, chunk_size, H)
    dA = (A * dt_chunked).float()
    dA_cumsum = torch.cumsum(dA, dim=2)
    return dA_cumsum, dt_chunked


# =============================================================================
# Kernel 2: bmm_chunk — CB = C @ B^T per chunk, in grouped space
# =============================================================================

@triton.jit
def _bmm_chunk_fwd_kernel(
    C_ptr, B_ptr, CB_ptr,
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_CB_batch, stride_CB_chunk, stride_CB_group, stride_CB_m, stride_CB_n,
    chunk_size: tl.constexpr, dstate: tl.constexpr, ngroups: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(L,BM)*cdiv(L,BN), batch, nchunks*ngroups)
    pid_mn = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_cg = tl.program_id(2)
    pid_c = pid_cg // ngroups
    pid_g = pid_cg % ngroups

    pid_m = pid_mn // tl.cdiv(chunk_size, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(chunk_size, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C_base = C_ptr + pid_b * stride_C_batch + pid_c * stride_C_chunk + pid_g * stride_C_group
    B_base = B_ptr + pid_b * stride_B_batch + pid_c * stride_B_chunk + pid_g * stride_B_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, dstate, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < dstate
        c = tl.load(C_base + offs_m[:, None] * stride_C_seqlen + k_offs[None, :] * stride_C_dstate,
                     mask=(offs_m[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(B_base + offs_n[:, None] * stride_B_seqlen + k_offs[None, :] * stride_B_dstate,
                     mask=(offs_n[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(c, tl.trans(b))

    CB_base = CB_ptr + pid_b * stride_CB_batch + pid_c * stride_CB_chunk + pid_g * stride_CB_group
    tl.store(CB_base + offs_m[:, None] * stride_CB_m + offs_n[None, :] * stride_CB_n,
             acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


def bmm_chunk_fwd(C, B, chunk_size):
    """
    Args:
        C: (batch, nchunks, chunk_size, ngroups, dstate)
        B: (batch, nchunks, chunk_size, ngroups, dstate)
    Returns:
        CB: (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    """
    batch, nchunks, L, ngroups, dstate = C.shape
    CB = torch.empty(batch, nchunks, ngroups, L, L, device=C.device, dtype=torch.float32)

    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(L))
    BLOCK_K = min(64, triton.next_power_of_2(dstate))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(L, BLOCK_N), batch, nchunks * ngroups)

    _bmm_chunk_fwd_kernel[grid](
        C, B, CB,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
        chunk_size=L, dstate=dstate, ngroups=ngroups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return CB


# =============================================================================
# Kernel 3: chunk_state — compute per-chunk state deltas
# =============================================================================

@triton.jit
def _chunk_state_fwd_kernel(
    B_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, states_ptr,
    # Strides: all tensors are 5D (batch, nchunks, ...) — NOT flattened
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_st_batch, stride_st_chunk, stride_st_head, stride_st_hdim, stride_st_dstate,
    # Dimensions
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, ngroups: tl.constexpr, nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """delta_h[b,c,h,p,n] = sum_l exp(dA_last - dA_l) * dt_l * x[l,h,p] * B[l,g(h),n]"""
    # Grid: (cdiv(headdim,BM)*cdiv(dstate,BN), batch*nchunks, nheads)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks

    pid_m = pid_mn // tl.cdiv(dstate, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(dstate, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # headdim
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate
    group_idx = pid_h // nheads_per_group

    # Base offset for this (batch, chunk)
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk
    off_bc_x = pid_b * stride_x_batch + pid_c * stride_x_chunk
    off_bc_B = pid_b * stride_B_batch + pid_c * stride_B_chunk

    # dA_last = dA_cumsum[b, c, chunk_size-1, h]
    dA_last = tl.load(dA_cumsum_ptr + off_bc_dA + (chunk_size - 1) * stride_dA_seqlen + pid_h * stride_dA_head).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, chunk_size, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # decay_to_end = exp(dA_last - dA_cumsum[l])
        dA_vals = tl.load(dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                          mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(dA_last - dA_vals)

        # dt[l, h]
        dt_vals = tl.load(dt_ptr + off_bc_dt + k_offs * stride_dt_seqlen + pid_h * stride_dt_head,
                          mask=k_mask, other=0.0).to(tl.float32)
        weight = decay * dt_vals  # (BLOCK_K,)

        # x[l, h, p]: (BLOCK_K, BLOCK_M)
        x_vals = tl.load(x_ptr + off_bc_x + k_offs[:, None] * stride_x_seqlen + pid_h * stride_x_head + offs_m[None, :] * stride_x_hdim,
                         mask=k_mask[:, None] & (offs_m[None, :] < headdim), other=0.0).to(tl.float32)
        x_vals = x_vals * weight[:, None]

        # B[l, g, n]: (BLOCK_K, BLOCK_N)
        b_vals = tl.load(B_ptr + off_bc_B + k_offs[:, None] * stride_B_seqlen + group_idx * stride_B_group + offs_n[None, :] * stride_B_dstate,
                         mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)

        acc += tl.dot(tl.trans(x_vals), b_vals)

    # Store states[b, c, h, p, n]
    off_st = states_ptr + pid_b * stride_st_batch + pid_c * stride_st_chunk + pid_h * stride_st_head
    tl.store(off_st + offs_m[:, None] * stride_st_hdim + offs_n[None, :] * stride_st_dstate,
             acc, mask=(offs_m[:, None] < headdim) & (offs_n[None, :] < dstate))


def chunk_state_fwd(B, x, dt, dA_cumsum):
    """
    Args:
        B:          (batch, nchunks, chunk_size, ngroups, dstate)
        x:          (batch, nchunks, chunk_size, nheads, headdim)
        dt:         (batch, nchunks, chunk_size, nheads)
        dA_cumsum:  (batch, nchunks, chunk_size, nheads)
    Returns:
        states: (batch, nchunks, nheads, headdim, dstate) float32
    """
    batch, nchunks, chunk_size, ngroups, dstate = B.shape
    nheads = x.shape[3]
    headdim = x.shape[4]
    nheads_per_group = nheads // ngroups

    states = torch.empty(batch, nchunks, nheads, headdim, dstate, device=x.device, dtype=torch.float32)

    BLOCK_M = min(64, triton.next_power_of_2(headdim))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    BLOCK_K = min(64, triton.next_power_of_2(chunk_size))
    grid = (triton.cdiv(headdim, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch * nchunks, nheads)

    _chunk_state_fwd_kernel[grid](
        B, x, dt, dA_cumsum, states,
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        chunk_size=chunk_size, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return states


# =============================================================================
# Kernel 4: state_passing — sequential scan across chunks
# =============================================================================

@triton.jit
def _state_passing_fwd_kernel(
    states_ptr, out_ptr, dA_last_ptr,
    stride_st_batch, stride_st_chunk, stride_st_head, stride_st_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_batch, stride_dA_chunk, stride_dA_head,
    dim: tl.constexpr, nchunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential scan: out[c] stores state entering chunk c; recurrence h = decay*h + delta."""
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d_mask = offs_d < dim

    running = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c in range(nchunks):
        # Store state entering chunk c (before adding chunk c's delta)
        tl.store(out_ptr + pid_b * stride_out_batch + c * stride_out_chunk + pid_h * stride_out_head + offs_d * stride_out_dim,
                 running, mask=d_mask)

        # Load delta and decay for chunk c
        delta = tl.load(states_ptr + pid_b * stride_st_batch + c * stride_st_chunk + pid_h * stride_st_head + offs_d * stride_st_dim,
                        mask=d_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.load(dA_last_ptr + pid_b * stride_dA_batch + c * stride_dA_chunk + pid_h * stride_dA_head).to(tl.float32))

        running = decay * running + delta


def state_passing_fwd(states, dA_cumsum_last):
    """
    Args:
        states:         (batch, nchunks, nheads, headdim, dstate) — per-chunk deltas
        dA_cumsum_last: (batch, nchunks, nheads) — decay per chunk
    Returns:
        out: (batch, nchunks, nheads, headdim, dstate) — propagated states entering each chunk
    """
    batch, nchunks, nheads, headdim, dstate = states.shape
    dim = headdim * dstate

    states_flat = states.reshape(batch, nchunks, nheads, dim).contiguous()
    out_flat = torch.empty_like(states_flat)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(dim))
    grid = (triton.cdiv(dim, BLOCK_SIZE), batch, nheads)

    _state_passing_fwd_kernel[grid](
        states_flat, out_flat, dA_cumsum_last,
        states_flat.stride(0), states_flat.stride(1), states_flat.stride(2), states_flat.stride(3),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2), out_flat.stride(3),
        dA_cumsum_last.stride(0), dA_cumsum_last.stride(1), dA_cumsum_last.stride(2),
        dim=dim, nchunks=nchunks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.view(batch, nchunks, nheads, headdim, dstate)


# =============================================================================
# Kernel 5: chunk_scan — the main fused kernel
# =============================================================================
# L (decay matrix) is NEVER materialized — computed tile-by-tile from
# dA_cumsum vectors and immediately fused into CB.

@triton.jit
def _chunk_scan_fwd_kernel(
    CB_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, C_ptr, prev_states_ptr, out_ptr,
    # CB: (batch, nchunks, ngroups, chunk_size, chunk_size)
    stride_CB_batch, stride_CB_chunk, stride_CB_group, stride_CB_m, stride_CB_n,
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # dt: (batch, nchunks, chunk_size, nheads)
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    # dA: (batch, nchunks, chunk_size, nheads)
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # C: (batch, nchunks, chunk_size, ngroups, dstate)
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # prev_states: (batch, nchunks, nheads, headdim, dstate)
    stride_ps_batch, stride_ps_chunk, stride_ps_head, stride_ps_hdim, stride_ps_dstate,
    # out: (batch, nchunks, chunk_size, nheads, headdim)
    stride_out_batch, stride_out_chunk, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # Dimensions
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, ngroups: tl.constexpr, nheads_per_group: tl.constexpr,
    nchunks: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """
    y[m,h,:] = scale * sum_k L[m,k,h] * CB[m,k,g(h)] * dt[k,h] * x[k,h,:]
             + scale * exp(dA_cs[m,h]) * C[m,g(h),:] @ prev_states[h,:,:]
    L[m,k,h] = exp(min(dA_cs[m,h] - dA_cs[k,h], 0)) for k<=m, 0 otherwise.
    L is computed tile-by-tile in registers, never stored to global memory.
    """
    # Grid: (cdiv(L,BM)*cdiv(hdim,BN), batch*nchunks, nheads)
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
    off_bc_CB = pid_b * stride_CB_batch + pid_c * stride_CB_chunk + group_idx * stride_CB_group
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk
    off_bc_x = pid_b * stride_x_batch + pid_c * stride_x_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk
    off_bc_ps = pid_b * stride_ps_batch + pid_c * stride_ps_chunk

    # Load dA_cumsum for rows m
    dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + pid_h * stride_dA_head,
                       mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    # --- Inter-chunk: scale * exp(dA_cs_m) * C[m,:] @ prev_states[h,:,:] ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d_start in range(0, dstate, BLOCK_DSTATE):
        d_offs = d_start + tl.arange(0, BLOCK_DSTATE)
        d_mask = d_offs < dstate

        c_vals = tl.load(C_ptr + off_bc_C + offs_m[:, None] * stride_C_seqlen + group_idx * stride_C_group + d_offs[None, :] * stride_C_dstate,
                         mask=(offs_m[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)
        ps_vals = tl.load(prev_states_ptr + off_bc_ps + pid_h * stride_ps_head + offs_n[:, None] * stride_ps_hdim + d_offs[None, :] * stride_ps_dstate,
                          mask=(offs_n[:, None] < headdim) & d_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(c_vals, tl.trans(ps_vals))

    acc = acc * (tl.exp(dA_cs_m)[:, None] * scale)

    # --- Intra-chunk: scale * sum_k L[m,k] * CB[m,k] * dt[k] * x[k] ---
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)
    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # CB[m, k]
        cb = tl.load(CB_ptr + off_bc_CB + offs_m[:, None] * stride_CB_m + k_offs[None, :] * stride_CB_n,
                     mask=(offs_m[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)

        # === L computed tile-by-tile, NEVER stored to global memory ===
        dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                          mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))

        # Causal mask
        cb = tl.where(offs_m[:, None] >= k_offs[None, :], cb, 0.0)
        cb = cb * decay * scale

        # dt[k]
        dt_vals = tl.load(dt_ptr + off_bc_dt + k_offs * stride_dt_seqlen + pid_h * stride_dt_head,
                          mask=k_mask, other=0.0).to(tl.float32)
        cb = cb * dt_vals[None, :]

        # x[k, h, n]
        x_vals = tl.load(x_ptr + off_bc_x + k_offs[:, None] * stride_x_seqlen + pid_h * stride_x_head + offs_n[None, :] * stride_x_hdim,
                         mask=k_mask[:, None] & (offs_n[None, :] < headdim), other=0.0)

        acc += tl.dot(cb.to(x_vals.dtype), x_vals).to(tl.float32)

    # Store
    off_bc_out = pid_b * stride_out_batch + pid_c * stride_out_chunk
    tl.store(out_ptr + off_bc_out + offs_m[:, None] * stride_out_seqlen + pid_h * stride_out_head + offs_n[None, :] * stride_out_hdim,
             acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def chunk_scan_fwd(CB, x, dt, dA_cumsum, C, prev_states, scale, chunk_size):
    """
    Args:
        CB:          (batch, nchunks, ngroups, chunk_size, chunk_size)
        x:           (batch, nchunks, chunk_size, nheads, headdim)
        dt:          (batch, nchunks, chunk_size, nheads)
        dA_cumsum:   (batch, nchunks, chunk_size, nheads)
        C:           (batch, nchunks, chunk_size, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        scale:       float
        chunk_size:  int
    Returns:
        out: (batch, nchunks, chunk_size, nheads, headdim) same dtype as x
    """
    batch, nchunks, L, nheads, headdim = x.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups

    out = torch.empty(batch, nchunks, L, nheads, headdim, device=x.device, dtype=x.dtype)

    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(headdim))
    BLOCK_K = min(64, triton.next_power_of_2(L))
    BLOCK_DSTATE = min(64, triton.next_power_of_2(dstate))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(headdim, BLOCK_N), batch * nchunks, nheads)

    _chunk_scan_fwd_kernel[grid](
        CB, x, dt, dA_cumsum, C, prev_states, out,
        CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        nchunks=nchunks, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_DSTATE=BLOCK_DSTATE,
    )
    return out


# =============================================================================
# Combined forward: orchestrates all 5 steps
# =============================================================================

def ssd_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, scale):
    """
    Full SSD forward using Triton kernels. L is never materialized.

    Args:
        x:  (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A:  (nheads,)
        B:  (batch, seqlen, ngroups, dstate)
        C:  (batch, seqlen, ngroups, dstate)
        chunk_size: int
        scale: float (d_state ** -0.5)
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    dstate = B.shape[3]
    nchunks = seqlen // chunk_size
    L = chunk_size

    # Reshape into chunks
    x_c = x.view(batch, nchunks, L, nheads, headdim)
    B_c = B.view(batch, nchunks, L, ngroups, dstate)
    C_c = C.view(batch, nchunks, L, ngroups, dstate)

    # Step 1: cumsum
    dA_cumsum, dt_c = chunk_cumsum(dt, A, chunk_size)

    # Step 2: CB = C @ B^T (in grouped space)
    CB = bmm_chunk_fwd(C_c, B_c, chunk_size)

    # Step 3: per-chunk state deltas
    states = chunk_state_fwd(B_c, x_c, dt_c, dA_cumsum)

    # Step 4: propagate states across chunks
    dA_last = dA_cumsum[:, :, -1, :]  # (batch, nchunks, nheads)
    prev_states = state_passing_fwd(states, dA_last)

    # Step 5: fused chunk scan (L computed tile-by-tile, never materialized)
    y = chunk_scan_fwd(CB, x_c, dt_c, dA_cumsum, C_c, prev_states, scale, chunk_size)

    return y.view(batch, seqlen, nheads, headdim)
