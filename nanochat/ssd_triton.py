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
    B_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, C_ptr, prev_states_ptr, out_ptr,
    # B: (batch, nchunks, chunk_size, ngroups, dstate)
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
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
    y[m,h,:] = scale * sum_k L[m,k,h] * (C[m,g(h),:] @ B[k,g(h),:]) * dt[k,h] * x[k,h,:]
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
    off_bc_B = pid_b * stride_B_batch + pid_c * stride_B_chunk + group_idx * stride_B_group
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

    # --- Intra-chunk: scale * sum_k L[m,k] * (C[m]@B[k]^T) * dt[k] * x[k] ---
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)
    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size

        # Compute CB[m, k] = C[m] @ B[k]^T on the fly
        cb = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for d_start in range(0, dstate, BLOCK_DSTATE):
            d_offs = d_start + tl.arange(0, BLOCK_DSTATE)
            d_mask = d_offs < dstate
            
            c_vals2 = tl.load(C_ptr + off_bc_C + offs_m[:, None] * stride_C_seqlen + group_idx * stride_C_group + d_offs[None, :] * stride_C_dstate,
                             mask=(offs_m[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)
            b_vals = tl.load(B_ptr + off_bc_B + k_offs[:, None] * stride_B_seqlen + d_offs[None, :] * stride_B_dstate,
                             mask=(k_offs[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)
            cb += tl.dot(c_vals2, tl.trans(b_vals))

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


def chunk_scan_fwd(B, x, dt, dA_cumsum, C, prev_states, scale, chunk_size):
    """
    Args:
        B:           (batch, nchunks, chunk_size, ngroups, dstate)
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
        B, x, dt, dA_cumsum, C, prev_states, out,
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
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
        num_warps=8,
        num_stages=4,
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

    # Step 2: per-chunk state deltas
    states = chunk_state_fwd(B_c, x_c, dt_c, dA_cumsum)

    # Step 4: propagate states across chunks
    dA_last = dA_cumsum[:, :, -1, :]  # (batch, nchunks, nheads)
    prev_states = state_passing_fwd(states, dA_last)

    # Step 4: fused chunk scan (L computed tile-by-tile, never materialized)
    y = chunk_scan_fwd(B_c, x_c, dt_c, dA_cumsum, C_c, prev_states, scale, chunk_size)

    return y.view(batch, seqlen, nheads, headdim)


# =============================================================================
# BACKWARD KERNELS
# =============================================================================

# -----------------------------------------------------------------------------
# Backward Kernel 5a: chunk_scan_bwd_dstates
# d_prev_states[h,p,n] = scale * sum_m exp(dA_cs[m,h]) * C[m,g(h),n] * dy[m,h,p]
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_scan_bwd_dstates_kernel(
    dy_ptr, C_ptr, dA_cumsum_ptr, d_prev_states_ptr,
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dps_batch, stride_dps_chunk, stride_dps_head, stride_dps_hdim, stride_dps_dstate,
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads_per_group: tl.constexpr, nchunks: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(headdim,BM)*cdiv(dstate,BN), batch*nchunks, nheads)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_m = pid_mn // tl.cdiv(dstate, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(dstate, BLOCK_N)
    offs_p = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # headdim
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate
    group_idx = pid_h // nheads_per_group

    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, chunk_size, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < chunk_size
        # exp(dA_cs[m, h])
        dA_vals = tl.load(dA_cumsum_ptr + off_bc_dA + k_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                          mask=k_mask, other=0.0).to(tl.float32)
        weight = tl.exp(dA_vals) * scale  # (BLOCK_K,)
        # dy[m, h, p]: (BLOCK_K, BLOCK_M)
        dy_vals = tl.load(dy_ptr + off_bc_dy + k_offs[:, None] * stride_dy_seqlen + pid_h * stride_dy_head + offs_p[None, :] * stride_dy_hdim,
                          mask=k_mask[:, None] & (offs_p[None, :] < headdim), other=0.0).to(tl.float32)
        dy_vals = dy_vals * weight[:, None]
        # C[m, g, n]: (BLOCK_K, BLOCK_N)
        c_vals = tl.load(C_ptr + off_bc_C + k_offs[:, None] * stride_C_seqlen + group_idx * stride_C_group + offs_n[None, :] * stride_C_dstate,
                         mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        acc += tl.dot(tl.trans(dy_vals), c_vals)

    off_dps = d_prev_states_ptr + pid_b * stride_dps_batch + pid_c * stride_dps_chunk + pid_h * stride_dps_head
    tl.store(off_dps + offs_p[:, None] * stride_dps_hdim + offs_n[None, :] * stride_dps_dstate,
             acc, mask=(offs_p[:, None] < headdim) & (offs_n[None, :] < dstate))


def chunk_scan_bwd_dstates(dy, C, dA_cumsum, prev_states, scale):
    """
    Args:
        dy:         (batch, nchunks, chunk_size, nheads, headdim)
        C:          (batch, nchunks, chunk_size, ngroups, dstate)
        dA_cumsum:  (batch, nchunks, chunk_size, nheads)
        prev_states: unused (kept for API consistency)
        scale:      float
    Returns:
        d_prev_states: (batch, nchunks, nheads, headdim, dstate) float32
    """
    batch, nchunks, L, nheads, headdim = dy.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups
    d_prev_states = torch.empty(batch, nchunks, nheads, headdim, dstate, device=dy.device, dtype=torch.float32)
    BLOCK_M = min(64, triton.next_power_of_2(headdim))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    BLOCK_K = min(64, triton.next_power_of_2(L))
    grid = (triton.cdiv(headdim, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch * nchunks, nheads)
    _chunk_scan_bwd_dstates_kernel[grid](
        dy, C, dA_cumsum, d_prev_states,
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        d_prev_states.stride(0), d_prev_states.stride(1), d_prev_states.stride(2), d_prev_states.stride(3), d_prev_states.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads_per_group=nheads_per_group, nchunks=nchunks, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return d_prev_states


# -----------------------------------------------------------------------------
# Backward Kernel 5b: chunk_scan_bwd_dC
# dC[m,g,n] = sum_{h in g} scale * exp(dA_cs[m,h]) * sum_p dy[m,h,p] * prev_states[h,p,n]
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_scan_bwd_dC_kernel(
    dy_ptr, prev_states_ptr, dA_cumsum_ptr, dC_ptr,
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    stride_ps_batch, stride_ps_chunk, stride_ps_head, stride_ps_hdim, stride_ps_dstate,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dC_batch, stride_dC_chunk, stride_dC_seqlen, stride_dC_group, stride_dC_dstate,
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, nheads_per_group: tl.constexpr, nchunks: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(L,BM)*cdiv(dstate,BN), batch*nchunks, ngroups)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_g = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_m = pid_mn // tl.cdiv(dstate, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(dstate, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # chunk positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate

    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk
    off_bc_ps = pid_b * stride_ps_batch + pid_c * stride_ps_chunk
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Sum over heads in this group
    for h_offset in range(nheads_per_group):
        h = pid_g * nheads_per_group + h_offset
        # Load exp(dA_cs[m, h]) for these positions
        dA_vals = tl.load(dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + h * stride_dA_head,
                          mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        weight = tl.exp(dA_vals) * scale  # (BLOCK_M,)
        # Sum over headdim: dy[m,h,:] @ prev_states[h,:,n]
        for k_start in range(0, headdim, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < headdim
            dy_vals = tl.load(dy_ptr + off_bc_dy + offs_m[:, None] * stride_dy_seqlen + h * stride_dy_head + k_offs[None, :] * stride_dy_hdim,
                              mask=(offs_m[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)
            dy_vals = dy_vals * weight[:, None]
            ps_vals = tl.load(prev_states_ptr + off_bc_ps + h * stride_ps_head + k_offs[:, None] * stride_ps_hdim + offs_n[None, :] * stride_ps_dstate,
                              mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            acc += tl.dot(dy_vals, ps_vals)

    off_dC = dC_ptr + pid_b * stride_dC_batch + pid_c * stride_dC_chunk + pid_g * stride_dC_group
    tl.store(off_dC + offs_m[:, None] * stride_dC_seqlen + offs_n[None, :] * stride_dC_dstate,
             acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < dstate))


def chunk_scan_bwd_dC(dy, prev_states, dA_cumsum, ngroups, scale):
    """
    Args:
        dy:          (batch, nchunks, chunk_size, nheads, headdim)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        dA_cumsum:   (batch, nchunks, chunk_size, nheads)
        ngroups:     int
        scale:       float
    Returns:
        dC: (batch, nchunks, chunk_size, ngroups, dstate) float32
    """
    batch, nchunks, L, nheads, headdim = dy.shape
    dstate = prev_states.shape[4]
    return _chunk_scan_bwd_dC_impl(dy, prev_states, dA_cumsum, scale, ngroups, dstate)


def _chunk_scan_bwd_dC_impl(dy, prev_states, dA_cumsum, scale, ngroups, dstate_unused=None):
    batch, nchunks, L, nheads, headdim = dy.shape
    dstate = prev_states.shape[4]
    nheads_per_group = nheads // ngroups
    dC = torch.empty(batch, nchunks, L, ngroups, dstate, device=dy.device, dtype=torch.float32)
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    BLOCK_K = min(64, triton.next_power_of_2(headdim))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch * nchunks, ngroups)
    _chunk_scan_bwd_dC_kernel[grid](
        dy, prev_states, dA_cumsum, dC,
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, nheads_per_group=nheads_per_group, nchunks=nchunks, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return dC


# -----------------------------------------------------------------------------
# Backward Kernel 5c: chunk_scan_bwd_dCB
# dCB[m,k,g] = sum_{h in g} scale * L[m,k,h] * dt[k,h] * sum_p(dy[m,h,p]*x[k,h,p])
# L computed tile-by-tile, never materialized.
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_scan_bwd_dCB_kernel(
    dy_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, dCB_ptr,
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dCB_batch, stride_dCB_chunk, stride_dCB_group, stride_dCB_m, stride_dCB_n,
    chunk_size: tl.constexpr, headdim: tl.constexpr,
    nheads: tl.constexpr, nheads_per_group: tl.constexpr, nchunks: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(L,BM)*cdiv(L,BN), batch*nchunks, ngroups)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_g = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_m = pid_mn // tl.cdiv(chunk_size, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(chunk_size, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk
    off_bc_x = pid_b * stride_x_batch + pid_c * stride_x_chunk
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h_offset in range(nheads_per_group):
        h = pid_g * nheads_per_group + h_offset
        # L[m,k,h] = exp(min(dA_cs[m]-dA_cs[k], 0)) with causal mask
        dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + offs_m * stride_dA_seqlen + h * stride_dA_head,
                          mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + offs_k * stride_dA_seqlen + h * stride_dA_head,
                          mask=offs_k < chunk_size, other=0.0).to(tl.float32)
        L_tile = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        L_tile = tl.where(offs_m[:, None] >= offs_k[None, :], L_tile, 0.0)
        # dt[k,h]
        dt_k = tl.load(dt_ptr + off_bc_dt + offs_k * stride_dt_seqlen + h * stride_dt_head,
                        mask=offs_k < chunk_size, other=0.0).to(tl.float32)
        # D[m,k] = sum_p dy[m,h,p]*x[k,h,p] — dot product over headdim
        D = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for p_start in range(0, headdim, BLOCK_K):
            p_offs = p_start + tl.arange(0, BLOCK_K)
            p_mask = p_offs < headdim
            dy_vals = tl.load(dy_ptr + off_bc_dy + offs_m[:, None] * stride_dy_seqlen + h * stride_dy_head + p_offs[None, :] * stride_dy_hdim,
                              mask=(offs_m[:, None] < chunk_size) & p_mask[None, :], other=0.0).to(tl.float32)
            x_vals = tl.load(x_ptr + off_bc_x + offs_k[:, None] * stride_x_seqlen + h * stride_x_head + p_offs[None, :] * stride_x_hdim,
                             mask=(offs_k[:, None] < chunk_size) & p_mask[None, :], other=0.0).to(tl.float32)
            D += tl.dot(dy_vals, tl.trans(x_vals))
        acc += scale * L_tile * dt_k[None, :] * D

    off_dCB = dCB_ptr + pid_b * stride_dCB_batch + pid_c * stride_dCB_chunk + pid_g * stride_dCB_group
    tl.store(off_dCB + offs_m[:, None] * stride_dCB_m + offs_k[None, :] * stride_dCB_n,
             acc, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size))


def chunk_scan_bwd_dCB(dy, x, dt, dA_cumsum, ngroups, scale):
    """
    Args:
        dy:         (batch, nchunks, chunk_size, nheads, headdim)
        x:          (batch, nchunks, chunk_size, nheads, headdim)
        dt:         (batch, nchunks, chunk_size, nheads)
        dA_cumsum:  (batch, nchunks, chunk_size, nheads)
        ngroups:    int
        scale:      float
    Returns:
        dCB: (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    """
    batch, nchunks, L, nheads, headdim = dy.shape
    nheads_per_group = nheads // ngroups
    dCB = torch.empty(batch, nchunks, ngroups, L, L, device=dy.device, dtype=torch.float32)
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(L))
    BLOCK_K = min(64, triton.next_power_of_2(headdim))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(L, BLOCK_N), batch * nchunks, ngroups)
    _chunk_scan_bwd_dCB_kernel[grid](
        dy, x, dt, dA_cumsum, dCB,
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        dCB.stride(0), dCB.stride(1), dCB.stride(2), dCB.stride(3), dCB.stride(4),
        chunk_size=L, headdim=headdim,
        nheads=nheads, nheads_per_group=nheads_per_group, nchunks=nchunks, scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return dCB


# -----------------------------------------------------------------------------
# Backward Kernel 5d: chunk_scan_bwd_dx
# dx[k,h,p] = scale * dt[k,h] * sum_{m>=k} L[m,k,h] * CB[m,k,g(h)] * dy[m,h,p]
# Transpose of forward intra-chunk. L tile-by-tile, never materialized.
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_scan_bwd_dx_kernel(
    dy_ptr, C_ptr, B_ptr, dt_ptr, dA_cumsum_ptr, dx_ptr,
    stride_dy_batch, stride_dy_chunk, stride_dy_seqlen, stride_dy_head, stride_dy_hdim,
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dx_batch, stride_dx_chunk, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads_per_group: tl.constexpr, nchunks: tl.constexpr, scale: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """
    Grid: (cdiv(L,BK)*cdiv(headdim,BN), batch*nchunks, nheads)
    For each (k-block, p-block, head): iterate over m >= k_block_start.
    """
    pid_kn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_k = pid_kn // tl.cdiv(headdim, BLOCK_N)
    pid_n = pid_kn % tl.cdiv(headdim, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # headdim
    group_idx = pid_h // nheads_per_group

    off_bc_dy = pid_b * stride_dy_batch + pid_c * stride_dy_chunk
    off_bc_C = pid_b * stride_C_batch + pid_c * stride_C_chunk + group_idx * stride_C_group
    off_bc_B = pid_b * stride_B_batch + pid_c * stride_B_chunk + group_idx * stride_B_group
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk

    # Load dA_cs[k, h] once
    dA_cs_k = tl.load(dA_cumsum_ptr + off_bc_dA + offs_k * stride_dA_seqlen + pid_h * stride_dA_head,
                       mask=offs_k < chunk_size, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    # Iterate over m-blocks: m must be >= k for causal mask
    for m_start in range(pid_k * BLOCK_K, chunk_size, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < chunk_size
        # L[m,k,h] = exp(min(dA_cs[m]-dA_cs[k], 0)) with causal mask m>=k
        dA_cs_m = tl.load(dA_cumsum_ptr + off_bc_dA + m_offs * stride_dA_seqlen + pid_h * stride_dA_head,
                          mask=m_mask, other=0.0).to(tl.float32)
        L_tile = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))  # (BM, BK)
        L_tile = tl.where(m_offs[:, None] >= offs_k[None, :], L_tile, 0.0)
        # cb[m, k, g] = C[m] @ B[k]^T on the fly
        cb = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for d_start in range(0, dstate, BLOCK_DSTATE):
            d_offs = d_start + tl.arange(0, BLOCK_DSTATE)
            d_mask = d_offs < dstate
            
            c_vals2 = tl.load(C_ptr + off_bc_C + m_offs[:, None] * stride_C_seqlen + d_offs[None, :] * stride_C_dstate,
                             mask=(m_offs[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)
            b_vals = tl.load(B_ptr + off_bc_B + offs_k[:, None] * stride_B_seqlen + d_offs[None, :] * stride_B_dstate,
                             mask=(offs_k[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)
            cb += tl.dot(c_vals2, tl.trans(b_vals))

        # score[m,k] = scale * L[m,k] * CB[m,k,g]
        score = scale * L_tile * cb  # (BM, BK)
        # dy[m, h, p]: (BM, BN)
        dy_vals = tl.load(dy_ptr + off_bc_dy + m_offs[:, None] * stride_dy_seqlen + pid_h * stride_dy_head + offs_n[None, :] * stride_dy_hdim,
                          mask=m_mask[:, None] & (offs_n[None, :] < headdim), other=0.0)
        # dx[k,p] += score^T @ dy = (BK,BM) @ (BM,BN) = (BK,BN)
        acc += tl.dot(tl.trans(score.to(dy_vals.dtype)), dy_vals).to(tl.float32)

    # Multiply by dt[k,h]
    dt_k = tl.load(dt_ptr + off_bc_dt + offs_k * stride_dt_seqlen + pid_h * stride_dt_head,
                    mask=offs_k < chunk_size, other=0.0).to(tl.float32)
    acc = acc * dt_k[:, None]

    off_dx = dx_ptr + pid_b * stride_dx_batch + pid_c * stride_dx_chunk
    tl.store(off_dx + offs_k[:, None] * stride_dx_seqlen + pid_h * stride_dx_head + offs_n[None, :] * stride_dx_hdim,
             acc.to(dx_ptr.dtype.element_ty),
             mask=(offs_k[:, None] < chunk_size) & (offs_n[None, :] < headdim))


def chunk_scan_bwd_dx(dy, C, B, dt, dA_cumsum, scale):
    """
    Args:
        dy:         (batch, nchunks, chunk_size, nheads, headdim)
        C:          (batch, nchunks, chunk_size, ngroups, dstate)
        B:          (batch, nchunks, chunk_size, ngroups, dstate)
        dt:         (batch, nchunks, chunk_size, nheads)
        dA_cumsum:  (batch, nchunks, chunk_size, nheads)
        scale:      float
    Returns:
        dx: (batch, nchunks, chunk_size, nheads, headdim) same dtype as dy
    """
    batch, nchunks, L, nheads, headdim = dy.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups
    dx = torch.empty_like(dy)
    BLOCK_K = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(headdim))
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_DSTATE = min(64, triton.next_power_of_2(dstate))
    grid = (triton.cdiv(L, BLOCK_K) * triton.cdiv(headdim, BLOCK_N), batch * nchunks, nheads)
    _chunk_scan_bwd_dx_kernel[grid](
        dy, C, B, dt, dA_cumsum, dx,
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3), dy.stride(4),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3), dx.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads_per_group=nheads_per_group, nchunks=nchunks, scale=scale,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M, BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8,
        num_stages=4,
    )
    return dx


# -----------------------------------------------------------------------------
# Backward Kernel 3a: chunk_state_bwd_dx
# dx[l,h,p] += sum_n exp(dA_last - dA_l) * dt_l * B[l,g(h),n] * d_states[h,p,n]
# Also computes ddt contribution from chunk_state.
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_state_bwd_dx_kernel(
    d_states_ptr, B_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, dx_ptr, ddt_ptr,
    stride_ds_batch, stride_ds_chunk, stride_ds_head, stride_ds_hdim, stride_ds_dstate,
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dx_batch, stride_dx_chunk, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_seqlen, stride_ddt_head,
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads_per_group: tl.constexpr, nchunks: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Grid: (cdiv(chunk_size,BM)*cdiv(headdim,BN), batch*nchunks, nheads)
    dx[l,h,p] = weight[l] * sum_n B[l,g,n] * d_states[h,p,n]
    ddt[l,h] = weight[l] / dt[l,h] * sum_p x[l,h,p] * dx[l,h,p] / weight[l]
             = sum_n sum_p exp(dA_last-dA_l) * x[l,h,p] * B[l,g,n] * d_states[h,p,n]
    """
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_m = pid_mn // tl.cdiv(headdim, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(headdim, BLOCK_N)
    offs_l = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    group_idx = pid_h // nheads_per_group

    off_bc_B = pid_b * stride_B_batch + pid_c * stride_B_chunk
    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk
    off_ds = d_states_ptr + pid_b * stride_ds_batch + pid_c * stride_ds_chunk + pid_h * stride_ds_head

    # weight[l] = exp(dA_last - dA_cumsum[l]) * dt[l]
    dA_last = tl.load(dA_cumsum_ptr + off_bc_dA + (chunk_size - 1) * stride_dA_seqlen + pid_h * stride_dA_head).to(tl.float32)
    dA_vals = tl.load(dA_cumsum_ptr + off_bc_dA + offs_l * stride_dA_seqlen + pid_h * stride_dA_head,
                      mask=offs_l < chunk_size, other=0.0).to(tl.float32)
    decay = tl.exp(dA_last - dA_vals)
    dt_vals = tl.load(dt_ptr + off_bc_dt + offs_l * stride_dt_seqlen + pid_h * stride_dt_head,
                      mask=offs_l < chunk_size, other=0.0).to(tl.float32)
    weight = decay * dt_vals

    # dx[l,p] = weight[l] * sum_n B[l,g,n] * d_states[h,p,n]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, dstate, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < dstate
        b_vals = tl.load(B_ptr + off_bc_B + offs_l[:, None] * stride_B_seqlen + group_idx * stride_B_group + k_offs[None, :] * stride_B_dstate,
                         mask=(offs_l[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)
        ds_vals = tl.load(off_ds + offs_p[:, None] * stride_ds_hdim + k_offs[None, :] * stride_ds_dstate,
                          mask=(offs_p[:, None] < headdim) & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(b_vals, tl.trans(ds_vals))
    unweighted_acc = acc
    acc = acc * weight[:, None]

    off_dx = dx_ptr + pid_b * stride_dx_batch + pid_c * stride_dx_chunk
    tl.store(off_dx + offs_l[:, None] * stride_dx_seqlen + pid_h * stride_dx_head + offs_p[None, :] * stride_dx_hdim,
             acc.to(dx_ptr.dtype.element_ty),
             mask=(offs_l[:, None] < chunk_size) & (offs_p[None, :] < headdim))

    # ddt[l,h] = decay[l] * sum_p x[l,h,p] * (sum_n B[l,g,n]*d_states[h,p,n])
    # = decay[l] * sum_p x[l,p] * unweighted_acc[l,p]
    off_bc_x = pid_b * stride_x_batch + pid_c * stride_x_chunk
    x_vals = tl.load(x_ptr + off_bc_x + offs_l[:, None] * stride_x_seqlen + pid_h * stride_x_head + offs_p[None, :] * stride_x_hdim,
                     mask=(offs_l[:, None] < chunk_size) & (offs_p[None, :] < headdim), other=0.0).to(tl.float32)
    
    ddt_partial = tl.sum(x_vals * unweighted_acc, axis=1) * decay
    off_ddt = ddt_ptr + pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk
    tl.atomic_add(off_ddt + offs_l * stride_ddt_seqlen + pid_h * stride_ddt_head,
                  ddt_partial, mask=offs_l < chunk_size)


def chunk_state_bwd_dx(d_states, B, x, dt, dA_cumsum):
    """
    Returns:
        dx:  (batch, nchunks, chunk_size, nheads, headdim)
        ddt: (batch, nchunks, chunk_size, nheads) float32
    """
    batch, nchunks, nheads, headdim, dstate = d_states.shape
    L = B.shape[2]
    ngroups = B.shape[3]
    nheads_per_group = nheads // ngroups
    dx = torch.empty(batch, nchunks, L, nheads, headdim, device=x.device, dtype=x.dtype)
    ddt = torch.zeros(batch, nchunks, L, nheads, device=x.device, dtype=torch.float32)
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(headdim))
    BLOCK_K = min(64, triton.next_power_of_2(dstate))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(headdim, BLOCK_N), batch * nchunks, nheads)
    _chunk_state_bwd_dx_kernel[grid](
        d_states, B, x, dt, dA_cumsum, dx, ddt,
        d_states.stride(0), d_states.stride(1), d_states.stride(2), d_states.stride(3), d_states.stride(4),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3), dx.stride(4),
        ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads_per_group=nheads_per_group, nchunks=nchunks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return dx, ddt


# -----------------------------------------------------------------------------
# Backward Kernel 3b: chunk_state_bwd_dB
# dB[l,g,n] = sum_{h in g} exp(dA_last - dA_l) * dt_l * sum_p x[l,h,p] * d_states[h,p,n]
# Same structure as chunk_state_fwd but transposed: x and d_states swap roles vs B and states.
# -----------------------------------------------------------------------------

@triton.jit
def _chunk_state_bwd_dB_kernel(
    d_states_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, dB_ptr,
    stride_ds_batch, stride_ds_chunk, stride_ds_head, stride_ds_hdim, stride_ds_dstate,
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    stride_dB_batch, stride_dB_chunk, stride_dB_seqlen, stride_dB_group, stride_dB_dstate,
    chunk_size: tl.constexpr, headdim: tl.constexpr, dstate: tl.constexpr,
    nheads: tl.constexpr, nheads_per_group: tl.constexpr, nchunks: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(chunk_size,BM)*cdiv(dstate,BN), batch*nchunks, ngroups)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_g = tl.program_id(2)
    pid_b = pid_bc // nchunks
    pid_c = pid_bc % nchunks
    pid_m = pid_mn // tl.cdiv(dstate, BLOCK_N)
    pid_n = pid_mn % tl.cdiv(dstate, BLOCK_N)
    offs_l = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    off_bc_dA = pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    off_bc_dt = pid_b * stride_dt_batch + pid_c * stride_dt_chunk
    off_bc_x = pid_b * stride_x_batch + pid_c * stride_x_chunk
    off_ds = d_states_ptr + pid_b * stride_ds_batch + pid_c * stride_ds_chunk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h_offset in range(nheads_per_group):
        h = pid_g * nheads_per_group + h_offset
        # weight[l] = exp(dA_last - dA_l) * dt[l,h]
        dA_last = tl.load(dA_cumsum_ptr + off_bc_dA + (chunk_size - 1) * stride_dA_seqlen + h * stride_dA_head).to(tl.float32)
        dA_vals = tl.load(dA_cumsum_ptr + off_bc_dA + offs_l * stride_dA_seqlen + h * stride_dA_head,
                          mask=offs_l < chunk_size, other=0.0).to(tl.float32)
        dt_vals = tl.load(dt_ptr + off_bc_dt + offs_l * stride_dt_seqlen + h * stride_dt_head,
                          mask=offs_l < chunk_size, other=0.0).to(tl.float32)
        weight = tl.exp(dA_last - dA_vals) * dt_vals
        # sum_p x[l,h,p] * d_states[h,p,n]
        for k_start in range(0, headdim, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < headdim
            x_vals = tl.load(x_ptr + off_bc_x + offs_l[:, None] * stride_x_seqlen + h * stride_x_head + k_offs[None, :] * stride_x_hdim,
                             mask=(offs_l[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)
            x_vals = x_vals * weight[:, None]
            ds_vals = tl.load(off_ds + h * stride_ds_head + k_offs[:, None] * stride_ds_hdim + offs_n[None, :] * stride_ds_dstate,
                              mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            acc += tl.dot(x_vals, ds_vals)

    off_dB = dB_ptr + pid_b * stride_dB_batch + pid_c * stride_dB_chunk + pid_g * stride_dB_group
    tl.store(off_dB + offs_l[:, None] * stride_dB_seqlen + offs_n[None, :] * stride_dB_dstate,
             acc, mask=(offs_l[:, None] < chunk_size) & (offs_n[None, :] < dstate))


def chunk_state_bwd_dB(d_states, x, dt, dA_cumsum, ngroups):
    batch, nchunks, nheads, headdim, dstate = d_states.shape
    L = x.shape[2]
    return _chunk_state_bwd_dB_impl(d_states, x, dt, dA_cumsum, ngroups)


def _chunk_state_bwd_dB_impl(d_states, x, dt, dA_cumsum, ngroups):
    batch, nchunks, nheads, headdim, dstate = d_states.shape
    L = x.shape[2]
    nheads_per_group = nheads // ngroups
    dB = torch.empty(batch, nchunks, L, ngroups, dstate, device=x.device, dtype=torch.float32)
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    BLOCK_K = min(64, triton.next_power_of_2(headdim))
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch * nchunks, ngroups)
    _chunk_state_bwd_dB_kernel[grid](
        d_states, x, dt, dA_cumsum, dB,
        d_states.stride(0), d_states.stride(1), d_states.stride(2), d_states.stride(3), d_states.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
        dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3), dB.stride(4),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, nheads_per_group=nheads_per_group, nchunks=nchunks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return dB


# =============================================================================
# PyTorch backward helpers
# =============================================================================

def state_passing_bwd(d_prev_states, prev_states, dA_cumsum_last):
    """
    Reverse scan: backward through state_passing_fwd.
    Forward: prev_states[c] = accumulated state entering chunk c.
             After storing, running = exp(dA_last[c]) * running + states[c].
    Backward:
        d_states[c-1] = adj[c]  (gradient of state delta for chunk c-1)
        d_dA_last[c-1] = (adj[c] * prev_states[c-1]).sum() * exp(dA_last[c-1])
        adj[c-1] += adj[c] * exp(dA_last[c-1])

    Args:
        d_prev_states:  (batch, nchunks, nheads, headdim, dstate)
        prev_states:    (batch, nchunks, nheads, headdim, dstate)
        dA_cumsum_last: (batch, nchunks, nheads)
    Returns:
        d_states:  (batch, nchunks, nheads, headdim, dstate) — gradient for chunk_state deltas
        d_dA_last: (batch, nchunks, nheads) — gradient for dA_cumsum[:,:,-1,:]
    """
    batch, nchunks, nheads, headdim, dstate = d_prev_states.shape
    d_states = torch.zeros_like(d_prev_states)
    d_dA_last = torch.zeros_like(dA_cumsum_last)
    adj = d_prev_states.clone()

    for c in range(nchunks - 1, 0, -1):
        d_states[:, c - 1] = adj[:, c]
        decay = torch.exp(dA_cumsum_last[:, c - 1])  # (batch, nheads)
        # d_dA_last[c-1] = sum over (headdim, dstate) of adj[c] * prev_states[c-1] * decay
        d_dA_last[:, c - 1] = (adj[:, c] * prev_states[:, c - 1]).sum(dim=(-2, -1)) * decay
        adj[:, c - 1] = adj[:, c - 1] + adj[:, c] * decay[:, :, None, None]

    return d_states, d_dA_last


def bmm_chunk_bwd(dCB, C, B):
    """
    Backward through CB = C @ B^T (per chunk, in grouped space).
    dC[m,g,n] = sum_k dCB[m,k,g] * B[k,g,n]
    dB[k,g,n] = sum_m dCB[m,k,g] * C[m,g,n]  (equivalently dCB^T @ C)

    Args:
        dCB: (batch, nchunks, ngroups, chunk_size, chunk_size) float32
        C:   (batch, nchunks, chunk_size, ngroups, dstate)
        B:   (batch, nchunks, chunk_size, ngroups, dstate)
    Returns:
        dC: (batch, nchunks, chunk_size, ngroups, dstate) float32
        dB: (batch, nchunks, chunk_size, ngroups, dstate) float32
    """
    batch, nchunks, ngroups, L, _ = dCB.shape
    dstate = C.shape[4]
    # dCB: (b, nc, ng, L, L), C: (b, nc, L, ng, ds), B: (b, nc, L, ng, ds)
    # dC[b,nc,m,g,n] = sum_k dCB[b,nc,g,m,k] * B[b,nc,k,g,n]
    # Reshape for einsum
    dC = torch.einsum('bcgmk, bckgn -> bcmgn', dCB.float(), B.float())
    dB = torch.einsum('bcgmk, bcmgn -> bckgn', dCB.float(), C.float())
    return dC, dB


def chunk_cumsum_bwd(d_dA_cumsum, A, dt_chunked):
    """
    Backward through dA_cumsum = cumsum(A * dt, dim=2).
    d(A*dt) = reverse_cumsum(d_dA_cumsum)
    ddt += A * d(A*dt)
    dA = sum(dt * d(A*dt))

    Args:
        d_dA_cumsum: (batch, nchunks, chunk_size, nheads) float32
        A:           (nheads,) float32
        dt_chunked:  (batch, nchunks, chunk_size, nheads) float32
    Returns:
        ddt: (batch, nchunks, chunk_size, nheads) float32
        dA:  (nheads,) float32
    """
    d_Adt = d_dA_cumsum.flip(2).cumsum(2).flip(2)  # reverse cumsum
    ddt = A * d_Adt
    dA = (dt_chunked.float() * d_Adt).sum(dim=(0, 1, 2))
    return ddt, dA


def chunk_scan_bwd_ddt_ddA(dy, x, dt, B, dA_cumsum, C, prev_states, ngroups, scale):
    """
    Compute ddt and d_dA_cumsum from both intra-chunk and inter-chunk.
    Processes one chunk at a time to avoid materializing full L.

    Args:
        dy:          (batch, nchunks, L, nheads, headdim)
        x:           (batch, nchunks, L, nheads, headdim)
        dt:          (batch, nchunks, L, nheads)
        B:           (batch, nchunks, L, ngroups, dstate)
        dA_cumsum:   (batch, nchunks, L, nheads)
        C:           (batch, nchunks, L, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        ngroups:     int
        scale:       float
    Returns:
        ddt:         (batch, nchunks, L, nheads) float32
        d_dA_cumsum: (batch, nchunks, L, nheads) float32
    """
    batch, nchunks, L, nheads, headdim = dy.shape
    nheads_per_group = nheads // ngroups
    dstate = C.shape[4]

    ddt_total = torch.zeros(batch, nchunks, L, nheads, device=dy.device, dtype=torch.float32)
    d_dA_total = torch.zeros(batch, nchunks, L, nheads, device=dy.device, dtype=torch.float32)

    for c in range(nchunks):
        dy_c = dy[:, c].float()   # (B, L, H, hd)
        x_c = x[:, c].float()
        dt_c = dt[:, c].float()   # (B, L, H)
        dA_c = dA_cumsum[:, c].float()  # (B, L, H)

        # --- Inter-chunk d_dA contribution ---
        # d_dA_inter[m,h] = scale * exp(dA_cs[m,h]) * sum_p(dy[m,h,p] * sum_n(C[m,g,n]*ps[h,p,n]))
        C_c = C[:, c].float()     # (B, L, ng, ds)
        ps_c = prev_states[:, c].float()  # (B, H, hd, ds)
        # Expand C for per-head: (B, L, H, ds) via repeat_interleave
        if ngroups < nheads:
            C_exp = C_c.repeat_interleave(nheads_per_group, dim=2)
        else:
            C_exp = C_c
        # y_inter_partial = sum_n C[m,h,n] * ps[h,:,n] -> (B, L, H, hd)
        y_inter_partial = torch.einsum('blhn, bhpn -> blhp', C_exp, ps_c)
        d_dA_inter = (dy_c * y_inter_partial).sum(dim=-1) * scale * torch.exp(dA_c)
        d_dA_total[:, c] += d_dA_inter

        # --- Intra-chunk ddt and d_dA contributions ---
        # Process per-head: materialize L for this chunk
        # L[m,k,h] = exp(min(dA_cs[m]-dA_cs[k], 0)) with causal mask
        log_L = dA_c.unsqueeze(2) - dA_c.unsqueeze(1)  # (B, L, L, H)
        causal = torch.arange(L, device=dy.device)
        causal_mask = causal.unsqueeze(1) >= causal.unsqueeze(0)  # (L, L) lower-triangular
        log_L_clamped = torch.clamp(log_L, max=0.0)
        L_mat = torch.exp(log_L_clamped) * causal_mask.unsqueeze(0).unsqueeze(-1)  # (B, L, L, H)

        # D[m,k,h] = sum_p dy[m,h,p]*x[k,h,p]
        D = torch.einsum('bmhp, bkhp -> bmkh', dy_c, x_c)  # (B, L, L, H)

        # Expand local chunk CB for per-head
        B_c = B[:, c].float()
        CB_c = torch.einsum('blgd,bkgd->bglk', C_c, B_c)  # (B, ng, L, L)
        if ngroups < nheads:
            CB_exp = CB_c.repeat_interleave(nheads_per_group, dim=1)  # (B, H, L, L)
        else:
            CB_exp = CB_c
        CB_mk = CB_exp.permute(0, 2, 3, 1)  # (B, L, L, H) — CB[m,k,h]

        # upstream[m,k,h] = scale * L[m,k,h] * CB[m,k,g(h)] * dt[k,h] * D[m,k,h]
        # ddt[k,h] = sum_m upstream[m,k,h] / dt[k,h] * dt[k,h] = sum_m scale*L*CB*D summed over m
        ddt_c = (scale * L_mat * CB_mk * D).sum(dim=1)  # sum over m -> (B, L, H)
        ddt_total[:, c] += ddt_c

        # d_dA from L gradient
        active = log_L < 0  # positions where clamping to 0 was NOT active, i.e. actual decay
        # The full gradient: dL/d(dA_cs[m]) = L * active, dL/d(dA_cs[k]) = -L * active
        L_grad = scale * CB_mk * dt_c.unsqueeze(1) * D * L_mat  # (B, L, L, H) - the upstream * L
        L_grad_active = L_grad * active
        d_dA_total[:, c] += L_grad_active.sum(dim=2)    # sum over k (row contribution for m)
        d_dA_total[:, c] -= L_grad_active.sum(dim=1)    # sum over m (col contribution for k)

    return ddt_total, d_dA_total


# =============================================================================
# Combined backward: orchestrates all backward steps
# =============================================================================

def ssd_chunk_scan_combined_bwd(dy, x, dt, A, B, C, chunk_size, scale):
    """
    Full SSD backward using Triton kernels where possible. L is never materialized
    in the main backward kernels (only in the ddt/ddA helper, one chunk at a time).

    Args:
        dy: (batch, seqlen, nheads, headdim) — gradient from upstream
        x:  (batch, seqlen, nheads, headdim) — saved from forward
        dt: (batch, seqlen, nheads) — saved from forward
        A:  (nheads,) — saved from forward
        B:  (batch, seqlen, ngroups, dstate) — saved from forward
        C:  (batch, seqlen, ngroups, dstate) — saved from forward
        chunk_size: int
        scale: float
    Returns:
        dx, dA, dB, dC, ddt — gradients for x, A, B, C, dt
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    dstate = B.shape[3]
    nchunks = seqlen // chunk_size
    L = chunk_size

    # Reshape into chunks
    x_c = x.view(batch, nchunks, L, nheads, headdim)
    dy_c = dy.view(batch, nchunks, L, nheads, headdim)
    B_c = B.view(batch, nchunks, L, ngroups, dstate)
    C_c = C.view(batch, nchunks, L, ngroups, dstate)

    # Recompute cheap forward intermediates
    dA_cumsum, dt_c = chunk_cumsum(dt, A, chunk_size)
    states = chunk_state_fwd(B_c, x_c, dt_c, dA_cumsum)
    dA_last = dA_cumsum[:, :, -1, :]
    prev_states = state_passing_fwd(states, dA_last)

    # Step 5': chunk_scan backward (Triton)
    d_prev_states = chunk_scan_bwd_dstates(dy_c, C_c, dA_cumsum, prev_states, scale)
    dC = _chunk_scan_bwd_dC_impl(dy_c, prev_states, dA_cumsum, scale, ngroups)
    dCB = chunk_scan_bwd_dCB(dy_c, x_c, dt_c, dA_cumsum, ngroups, scale)
    dx = chunk_scan_bwd_dx(dy_c, C_c, B_c, dt_c, dA_cumsum, scale)

    # Step 4': state_passing backward (PyTorch)
    d_states, d_dA_last = state_passing_bwd(d_prev_states, prev_states, dA_last)

    # Step 3': chunk_state backward (Triton)
    dx2, ddt_state = chunk_state_bwd_dx(d_states, B_c, x_c, dt_c, dA_cumsum)
    dB = _chunk_state_bwd_dB_impl(d_states, x_c, dt_c, dA_cumsum, ngroups)

    # Step 2': bmm_chunk backward (PyTorch)
    dC2, dB2 = bmm_chunk_bwd(dCB, C_c, B_c)

    # Compute ddt and d_dA_cumsum from intra+inter chunk (PyTorch, per-chunk)
    ddt_scan, d_dA_cumsum = chunk_scan_bwd_ddt_ddA(
        dy_c, x_c, dt_c, B_c, dA_cumsum, C_c, prev_states, ngroups, scale
    )

    # Add d_dA_last contribution from state_passing backward
    # d_dA_last affects dA_cumsum at the last position of each chunk
    d_dA_cumsum[:, :, -1, :] += d_dA_last

    # Add d_dA_cumsum contribution from chunk_state backward
    # chunk_state uses exp(dA_last - dA_cumsum[l]) * dt_l
    # d_dA_cumsum[l,h] from chunk_state: -decay * (same as fwd weight) * gradient
    # This is accounted for in the ddt_state computation via the dt gradient chain

    # Step 1': chunk_cumsum backward (PyTorch)
    ddt_cumsum, dA = chunk_cumsum_bwd(d_dA_cumsum, A, dt_c)

    # Combine gradients
    dx_total = (dx.float() + dx2.float()).to(x.dtype).view(batch, seqlen, nheads, headdim)
    ddt_total = (ddt_scan + ddt_state + ddt_cumsum).view(batch, seqlen, nheads)
    dB_total = (dB.float() + dB2).view(batch, seqlen, ngroups, dstate).to(B.dtype)
    dC_total = (dC + dC2).view(batch, seqlen, ngroups, dstate).to(C.dtype)

    return dx_total, dA, dB_total, dC_total, ddt_total


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

        # R-loop: compile-time unrolled
        for r in tl.static_range(mimo_rank):
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
            acc += tl.dot(scores_g.to(tl.float32), xg_vals)
            acc += tl.dot(scores_b.to(tl.float32), xb_vals)

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

    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(headdim))
    BLOCK_K = min(64, triton.next_power_of_2(L))
    BLOCK_DSTATE = min(64, triton.next_power_of_2(dstate))
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
        num_stages=4,
    )
    return y_intra
