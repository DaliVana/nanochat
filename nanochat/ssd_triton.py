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
# Kept as a Python function for clarity.

def chunk_cumsum(dt, A, chunk_size):
    """
    Compute cumulative sum of A * dt within each chunk.

    Args:
        dt: (batch, seqlen, nheads) — discretization timestep (already softplus'd + clamped)
        A:  (nheads,) — diagonal state decay (negative values)
    Returns:
        dA_cumsum: (batch, nchunks, chunk_size, nheads) float32
        dt_chunked: (batch, nchunks, chunk_size, nheads) float32
    """
    B, T, H = dt.shape
    assert T % chunk_size == 0, f"T ({T}) must be divisible by chunk_size ({chunk_size})"
    nchunks = T // chunk_size
    dt_chunked = dt.view(B, nchunks, chunk_size, H)
    dA = (A * dt_chunked).float()
    dA_cumsum = torch.cumsum(dA, dim=2)  # (B, nc, L, H)
    return dA_cumsum, dt_chunked


# =============================================================================
# Kernel 2: bmm_chunk — CB = C @ B^T per chunk, in grouped space
# =============================================================================

@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers
    C_ptr, B_ptr, CB_ptr,
    # Strides for C: (batch, nchunks, chunk_size, ngroups, dstate)
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # Strides for B: same layout
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    # Strides for CB: (batch, nchunks, ngroups, chunk_size, chunk_size)
    stride_CB_batch, stride_CB_chunk, stride_CB_group, stride_CB_m, stride_CB_n,
    # Dimensions
    chunk_size: tl.constexpr,
    dstate: tl.constexpr,
    ngroups: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute CB = C @ B^T for each (batch, chunk, group)."""
    # Grid: (cdiv(chunk_size, BLOCK_M) * cdiv(chunk_size, BLOCK_N), batch, nchunks * ngroups)
    pid_mn = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_cg = tl.program_id(2)
    pid_c = pid_cg // ngroups
    pid_g = pid_cg % ngroups

    num_n_blocks = tl.cdiv(chunk_size, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers
    C_base = C_ptr + pid_b * stride_C_batch + pid_c * stride_C_chunk + pid_g * stride_C_group
    B_base = B_ptr + pid_b * stride_B_batch + pid_c * stride_B_chunk + pid_g * stride_B_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, dstate, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < dstate

        # C[m, k]: shape (BLOCK_M, BLOCK_K)
        C_ptrs = C_base + offs_m[:, None] * stride_C_seqlen + k_offs[None, :] * stride_C_dstate
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)

        # B[n, k]: shape (BLOCK_N, BLOCK_K)
        B_ptrs = B_base + offs_n[:, None] * stride_B_seqlen + k_offs[None, :] * stride_B_dstate
        b = tl.load(B_ptrs, mask=(offs_n[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)

        # CB[m, n] += C[m, k] @ B[n, k]^T
        acc += tl.dot(c, tl.trans(b))

    # Store CB
    CB_base = CB_ptr + pid_b * stride_CB_batch + pid_c * stride_CB_chunk + pid_g * stride_CB_group
    CB_ptrs = CB_base + offs_m[:, None] * stride_CB_m + offs_n[None, :] * stride_CB_n
    mask = (offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size)
    tl.store(CB_ptrs, acc, mask=mask)


def bmm_chunk_fwd(C, B, chunk_size):
    """
    Compute CB = C @ B^T for each chunk, in grouped space.

    Args:
        C: (batch, nchunks, chunk_size, ngroups, dstate)
        B: (batch, nchunks, chunk_size, ngroups, dstate)
    Returns:
        CB: (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    """
    batch, nchunks, L, ngroups, dstate = C.shape
    assert L == chunk_size

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
    # Pointers
    B_ptr, x_ptr, dt_ptr, dA_cumsum_ptr, states_ptr,
    # Strides for B: (batch, nchunks, chunk_size, ngroups, dstate)
    stride_B_batch, stride_B_chunk, stride_B_seqlen, stride_B_group, stride_B_dstate,
    # Strides for x: (batch, nchunks, chunk_size, nheads, headdim)
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # Strides for dt: (batch, nchunks, chunk_size, nheads)
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    # Strides for dA_cumsum: (batch, nchunks, chunk_size, nheads)
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # Strides for states: (batch, nchunks, nheads, headdim, dstate)
    stride_st_batch, stride_st_chunk, stride_st_head, stride_st_hdim, stride_st_dstate,
    # Dimensions
    chunk_size: tl.constexpr,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads: tl.constexpr,
    ngroups: tl.constexpr,
    nheads_per_group: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,  # headdim tile
    BLOCK_N: tl.constexpr,  # dstate tile
    BLOCK_K: tl.constexpr,  # chunk_size tile
):
    """
    Compute delta_h[b,c,h,p,n] = sum_l exp(dA_last - dA_l) * dt_l * x[l,h,p] * B[l,g(h),n]
    """
    # Grid: (cdiv(headdim, BLOCK_M) * cdiv(dstate, BLOCK_N), batch * nchunks, nheads)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    num_n_blocks = tl.cdiv(dstate, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    batch_idx = pid_bc  # We'll use batch * nchunks as axis 1
    # Decompose batch_idx into batch and chunk
    # (passed as batch * nchunks in grid)
    nchunks_val = tl.load(dA_cumsum_ptr - dA_cumsum_ptr + tl.cast(0, tl.int64))  # dummy to get nchunks from grid
    # Actually, we pass batch*nchunks as grid dim and decompose here
    # But we can't easily get nchunks. Let's restructure.
    # Better: use pid_b and pid_c from separate grid dims is cleaner but limited.
    # We'll pass nchunks and decompose.

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # headdim offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # dstate offsets
    offs_k = tl.arange(0, BLOCK_K)                     # chunk_size offsets

    # Group index for this head
    group_idx = pid_h // nheads_per_group

    # Load dA_cumsum for last position in chunk (for decay_to_end)
    # dA_cumsum: (batch, nchunks, chunk_size, nheads)
    dA_last_ptr = dA_cumsum_ptr + pid_bc * stride_dA_chunk + (chunk_size - 1) * stride_dA_seqlen + pid_h * stride_dA_head
    dA_last = tl.load(dA_last_ptr).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, chunk_size, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < chunk_size

        # decay_to_end[l] = exp(dA_last - dA_cumsum[l])
        dA_ptrs = dA_cumsum_ptr + pid_bc * stride_dA_chunk + k_offs * stride_dA_seqlen + pid_h * stride_dA_head
        dA_vals = tl.load(dA_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        decay = tl.exp(dA_last - dA_vals)  # (BLOCK_K,)

        # dt[l]
        dt_ptrs = dt_ptr + pid_bc * stride_dt_chunk + k_offs * stride_dt_seqlen + pid_h * stride_dt_head
        dt_vals = tl.load(dt_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Combined weight: decay * dt
        weight = decay * dt_vals  # (BLOCK_K,)

        # x[l, h, p]: shape (BLOCK_K, BLOCK_M)
        x_ptrs = x_ptr + pid_bc * stride_x_chunk + k_offs[:, None] * stride_x_seqlen + pid_h * stride_x_head + offs_m[None, :] * stride_x_hdim
        x_vals = tl.load(x_ptrs, mask=k_mask[:, None] & (offs_m[None, :] < headdim), other=0.0).to(tl.float32)
        x_vals = x_vals * weight[:, None]  # (BLOCK_K, BLOCK_M)

        # B[l, g, n]: shape (BLOCK_K, BLOCK_N)
        B_ptrs = B_ptr + pid_bc * stride_B_chunk + k_offs[:, None] * stride_B_seqlen + group_idx * stride_B_group + offs_n[None, :] * stride_B_dstate
        b_vals = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)

        # acc += x_vals^T @ b_vals = (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(tl.trans(x_vals), b_vals)

    # Store states[b, c, h, p, n]
    st_ptrs = states_ptr + pid_bc * stride_st_chunk + pid_h * stride_st_head + offs_m[:, None] * stride_st_hdim + offs_n[None, :] * stride_st_dstate
    mask = (offs_m[:, None] < headdim) & (offs_n[None, :] < dstate)
    tl.store(st_ptrs, acc, mask=mask)


def chunk_state_fwd(B, x, dt, dA_cumsum):
    """
    Compute per-chunk state deltas.

    Args:
        B:          (batch, nchunks, chunk_size, ngroups, dstate)
        x:          (batch, nchunks, chunk_size, nheads, headdim)
        dt:         (batch, nchunks, chunk_size, nheads)
        dA_cumsum:  (batch, nchunks, chunk_size, nheads)
    Returns:
        states: (batch, nchunks, nheads, headdim, dstate) float32
    """
    batch_nchunks = B.shape[0] * B.shape[1]
    nchunks = B.shape[1]
    chunk_size = B.shape[2]
    ngroups = B.shape[3]
    dstate = B.shape[4]
    nheads = x.shape[3]
    headdim = x.shape[4]
    nheads_per_group = nheads // ngroups

    # Flatten batch and nchunks for grid
    B_flat = B.reshape(batch_nchunks, chunk_size, ngroups, dstate)
    x_flat = x.reshape(batch_nchunks, chunk_size, nheads, headdim)
    dt_flat = dt.reshape(batch_nchunks, chunk_size, nheads)
    dA_flat = dA_cumsum.reshape(batch_nchunks, chunk_size, nheads)

    states = torch.empty(batch_nchunks, nheads, headdim, dstate, device=x.device, dtype=torch.float32)

    BLOCK_M = min(64, triton.next_power_of_2(headdim))
    BLOCK_N = min(64, triton.next_power_of_2(dstate))
    BLOCK_K = min(64, triton.next_power_of_2(chunk_size))

    grid = (triton.cdiv(headdim, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch_nchunks, nheads)

    _chunk_state_fwd_kernel[grid](
        B_flat, x_flat, dt_flat, dA_flat, states,
        # B strides (4D after flatten)
        B_flat.stride(0), 0, B_flat.stride(1), B_flat.stride(2), B_flat.stride(3),
        # x strides
        x_flat.stride(0), 0, x_flat.stride(1), x_flat.stride(2), x_flat.stride(3),
        # dt strides
        dt_flat.stride(0), 0, dt_flat.stride(1), dt_flat.stride(2),
        # dA strides
        dA_flat.stride(0), 0, dA_flat.stride(1), dA_flat.stride(2),
        # states strides
        states.stride(0), 0, states.stride(1), states.stride(2), states.stride(3),
        chunk_size=chunk_size, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return states.view(B.shape[0], nchunks, nheads, headdim, dstate)


# =============================================================================
# Kernel 4: state_passing — sequential scan across chunks
# =============================================================================
# The sequential scan is inherently serial across chunks. A Triton kernel
# parallelizes over the (headdim * dstate) state dimensions, but loops over chunks.

@triton.jit
def _state_passing_fwd_kernel(
    # Pointers
    states_ptr,      # (batch, nchunks, nheads, dim) — input deltas (flattened hdim*dstate)
    out_ptr,         # (batch, nchunks, nheads, dim) — output propagated states
    dA_last_ptr,     # (batch, nchunks, nheads) — decay per chunk (dA_cumsum[:,:,-1,:])
    # Strides
    stride_st_batch, stride_st_chunk, stride_st_head, stride_st_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_batch, stride_dA_chunk, stride_dA_head,
    # Dimensions
    dim: tl.constexpr,
    nchunks: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential scan: out[c] = decay[c] * out[c-1] + states[c], storing intermediate states."""
    pid_d = tl.program_id(0)  # dim tile
    pid_b = tl.program_id(1)  # batch
    pid_h = tl.program_id(2)  # head

    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d_mask = offs_d < dim

    # Initialize running state to zero
    running = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c in range(nchunks):
        # Store propagated state BEFORE adding current chunk's delta
        # (this is the state entering chunk c, to be used by chunk c's output)
        out_ptrs = out_ptr + pid_b * stride_out_batch + c * stride_out_chunk + pid_h * stride_out_head + offs_d * stride_out_dim
        tl.store(out_ptrs, running, mask=d_mask)

        # Load chunk delta and decay
        st_ptrs = states_ptr + pid_b * stride_st_batch + c * stride_st_chunk + pid_h * stride_st_head + offs_d * stride_st_dim
        delta = tl.load(st_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        dA_ptr = dA_last_ptr + pid_b * stride_dA_batch + c * stride_dA_chunk + pid_h * stride_dA_head
        decay = tl.exp(tl.load(dA_ptr).to(tl.float32))

        # Recurrence: h_{c+1} = decay * h_c + delta_c
        running = decay * running + delta


def state_passing_fwd(states, dA_cumsum_last):
    """
    Sequential scan propagating states across chunks.

    Args:
        states:         (batch, nchunks, nheads, headdim, dstate) — per-chunk deltas
        dA_cumsum_last: (batch, nchunks, nheads) — decay factor per chunk (dA_cumsum[:,:,-1,:])
    Returns:
        out: (batch, nchunks, nheads, headdim, dstate) — propagated states entering each chunk
    """
    batch, nchunks, nheads, headdim, dstate = states.shape
    dim = headdim * dstate

    # Flatten state dims for the scan kernel
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
# This is the critical kernel: computes y = (CB * L) @ (x * dt) + C @ prev_states
# The L (decay) matrix is NEVER materialized — it's computed tile-by-tile from
# dA_cumsum vectors and immediately fused into CB.

@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers
    CB_ptr,           # (batch, nchunks, ngroups, chunk_size, chunk_size)
    x_ptr,            # (batch, nchunks, chunk_size, nheads, headdim)
    dt_ptr,           # (batch, nchunks, chunk_size, nheads)
    dA_cumsum_ptr,    # (batch, nchunks, chunk_size, nheads)
    C_ptr,            # (batch, nchunks, chunk_size, ngroups, dstate)
    prev_states_ptr,  # (batch, nchunks, nheads, headdim, dstate)
    out_ptr,          # (batch, nchunks, chunk_size, nheads, headdim)
    # CB strides
    stride_CB_batch, stride_CB_chunk, stride_CB_group, stride_CB_m, stride_CB_n,
    # x strides
    stride_x_batch, stride_x_chunk, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # dt strides
    stride_dt_batch, stride_dt_chunk, stride_dt_seqlen, stride_dt_head,
    # dA strides
    stride_dA_batch, stride_dA_chunk, stride_dA_seqlen, stride_dA_head,
    # C strides
    stride_C_batch, stride_C_chunk, stride_C_seqlen, stride_C_group, stride_C_dstate,
    # prev_states strides
    stride_ps_batch, stride_ps_chunk, stride_ps_head, stride_ps_hdim, stride_ps_dstate,
    # out strides
    stride_out_batch, stride_out_chunk, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # Dimensions
    chunk_size: tl.constexpr,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads: tl.constexpr,
    ngroups: tl.constexpr,
    nheads_per_group: tl.constexpr,
    scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,   # chunk_size rows (output positions)
    BLOCK_N: tl.constexpr,   # headdim cols
    BLOCK_K: tl.constexpr,   # chunk_size inner loop
    BLOCK_DSTATE: tl.constexpr,
):
    """
    Fused intra-chunk + inter-chunk scan.

    For each output position m:
      y[m, h, :] = scale * sum_k L[m,k,h] * CB[m,k,g(h)] * dt[k,h] * x[k,h,:]   (intra)
                 + scale * exp(dA_cumsum[m,h]) * C[m,g(h),:] @ prev_states[h,:,:]  (inter)

    L[m,k,h] = exp(min(dA_cumsum[m,h] - dA_cumsum[k,h], 0)) for k <= m, else 0.
    L is computed tile-by-tile in registers and never stored.
    """
    # Grid: (cdiv(chunk_size, BLOCK_M) * cdiv(headdim, BLOCK_N), batch * nchunks, nheads)
    pid_mn = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_h = tl.program_id(2)

    num_n_blocks = tl.cdiv(headdim, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output positions in chunk
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # headdim offsets
    offs_k = tl.arange(0, BLOCK_K)                     # inner loop offsets

    group_idx = pid_h // nheads_per_group

    # Load dA_cumsum for output rows m: (BLOCK_M,)
    dA_m_ptrs = dA_cumsum_ptr + pid_bc * stride_dA_chunk + offs_m * stride_dA_seqlen + pid_h * stride_dA_head
    dA_cs_m = tl.load(dA_m_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    # --- Part 1: Inter-chunk contribution: scale * exp(dA_cs_m) * C[m,:] @ prev_states ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # C: (BLOCK_M, BLOCK_DSTATE) for this group
    # prev_states: (BLOCK_N, BLOCK_DSTATE) transposed for matmul — actually (headdim, dstate)
    # We accumulate: acc = C @ prev_states^T (after suitable tiling over dstate)
    for d_start in range(0, dstate, BLOCK_DSTATE):
        d_offs = d_start + tl.arange(0, BLOCK_DSTATE)
        d_mask = d_offs < dstate

        # C[m, g, d]: (BLOCK_M, BLOCK_DSTATE)
        C_ptrs = C_ptr + pid_bc * stride_C_chunk + offs_m[:, None] * stride_C_seqlen + group_idx * stride_C_group + d_offs[None, :] * stride_C_dstate
        c_vals = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size) & d_mask[None, :], other=0.0).to(tl.float32)

        # prev_states[h, n, d]: (BLOCK_N, BLOCK_DSTATE)
        ps_ptrs = prev_states_ptr + pid_bc * stride_ps_chunk + pid_h * stride_ps_head + offs_n[:, None] * stride_ps_hdim + d_offs[None, :] * stride_ps_dstate
        ps_vals = tl.load(ps_ptrs, mask=(offs_n[:, None] < headdim) & d_mask[None, :], other=0.0).to(tl.float32)

        # acc += C @ ps^T: (BLOCK_M, BLOCK_DSTATE) @ (BLOCK_DSTATE, BLOCK_N)
        acc += tl.dot(c_vals, tl.trans(ps_vals))

    # Scale by exp(dA_cs_m) * scale
    acc = acc * (tl.exp(dA_cs_m)[:, None] * scale)

    # --- Part 2: Intra-chunk contribution ---
    # y[m] += scale * sum_k L[m,k] * CB[m,k] * dt[k] * x[k]
    # L[m,k] = exp(min(dA_cs_m - dA_cs_k, 0)) for k <= m, else 0

    # Causal limit: only process k <= max(offs_m)
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, chunk_size)

    for k_start in range(0, k_max, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < chunk_size

        # Load CB[m, k] for this group: (BLOCK_M, BLOCK_K)
        CB_ptrs = CB_ptr + pid_bc * stride_CB_chunk + group_idx * stride_CB_group + offs_m[:, None] * stride_CB_m + k_offs[None, :] * stride_CB_n
        cb = tl.load(CB_ptrs, mask=(offs_m[:, None] < chunk_size) & k_mask[None, :], other=0.0).to(tl.float32)

        # Load dA_cumsum for k positions: (BLOCK_K,)
        dA_k_ptrs = dA_cumsum_ptr + pid_bc * stride_dA_chunk + k_offs * stride_dA_seqlen + pid_h * stride_dA_head
        dA_cs_k = tl.load(dA_k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # === THE KEY: L computed tile-by-tile, never stored to global memory ===
        # L[m,k] = exp(min(dA_cs_m - dA_cs_k, 0))
        decay = tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))

        # Causal mask: zero out k > m
        causal_mask = offs_m[:, None] >= k_offs[None, :]
        cb = tl.where(causal_mask, cb, 0.0)

        # Fuse: cb = CB * L * scale
        cb = cb * decay * scale

        # Load dt[k]: (BLOCK_K,)
        dt_ptrs = dt_ptr + pid_bc * stride_dt_chunk + k_offs * stride_dt_seqlen + pid_h * stride_dt_head
        dt_vals = tl.load(dt_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        cb = cb * dt_vals[None, :]

        # Load x[k, h, n]: (BLOCK_K, BLOCK_N)
        x_ptrs = x_ptr + pid_bc * stride_x_chunk + k_offs[:, None] * stride_x_seqlen + pid_h * stride_x_head + offs_n[None, :] * stride_x_hdim
        x_vals = tl.load(x_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < headdim), other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(cb.to(x_vals.dtype), x_vals).to(tl.float32)

    # Store output
    out_ptrs = out_ptr + pid_bc * stride_out_chunk + offs_m[:, None] * stride_out_seqlen + pid_h * stride_out_head + offs_n[None, :] * stride_out_hdim
    mask = (offs_m[:, None] < chunk_size) & (offs_n[None, :] < headdim)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def chunk_scan_fwd(CB, x, dt, dA_cumsum, C, prev_states, scale, chunk_size):
    """
    Fused chunk scan: y = (CB * L) @ (x * dt) + C @ prev_states.
    L is computed tile-by-tile, never materialized.

    Args:
        CB:          (batch, nchunks, ngroups, chunk_size, chunk_size)
        x:           (batch, nchunks, chunk_size, nheads, headdim)
        dt:          (batch, nchunks, chunk_size, nheads)
        dA_cumsum:   (batch, nchunks, chunk_size, nheads)
        C:           (batch, nchunks, chunk_size, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        scale:       float scalar (d_state ** -0.5)
        chunk_size:  int
    Returns:
        out: (batch, nchunks, chunk_size, nheads, headdim) same dtype as x
    """
    batch, nchunks, L, nheads, headdim = x.shape
    ngroups = C.shape[3]
    dstate = C.shape[4]
    nheads_per_group = nheads // ngroups
    batch_nchunks = batch * nchunks

    # Flatten batch and nchunks
    CB_flat = CB.reshape(batch_nchunks, ngroups, L, L)
    x_flat = x.reshape(batch_nchunks, L, nheads, headdim)
    dt_flat = dt.reshape(batch_nchunks, L, nheads)
    dA_flat = dA_cumsum.reshape(batch_nchunks, L, nheads)
    C_flat = C.reshape(batch_nchunks, L, ngroups, dstate)
    ps_flat = prev_states.reshape(batch_nchunks, nheads, headdim, dstate)

    out = torch.empty(batch_nchunks, L, nheads, headdim, device=x.device, dtype=x.dtype)

    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(headdim))
    BLOCK_K = min(64, triton.next_power_of_2(L))
    BLOCK_DSTATE = min(64, triton.next_power_of_2(dstate))

    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(headdim, BLOCK_N), batch_nchunks, nheads)

    _chunk_scan_fwd_kernel[grid](
        CB_flat, x_flat, dt_flat, dA_flat, C_flat, ps_flat, out,
        # CB strides (4D: batch_nchunks, ngroups, L, L)
        CB_flat.stride(0), 0, CB_flat.stride(1), CB_flat.stride(2), CB_flat.stride(3),
        # x strides
        x_flat.stride(0), 0, x_flat.stride(1), x_flat.stride(2), x_flat.stride(3),
        # dt strides
        dt_flat.stride(0), 0, dt_flat.stride(1), dt_flat.stride(2),
        # dA strides
        dA_flat.stride(0), 0, dA_flat.stride(1), dA_flat.stride(2),
        # C strides
        C_flat.stride(0), 0, C_flat.stride(1), C_flat.stride(2), C_flat.stride(3),
        # prev_states strides
        ps_flat.stride(0), 0, ps_flat.stride(1), ps_flat.stride(2), ps_flat.stride(3),
        # out strides
        out.stride(0), 0, out.stride(1), out.stride(2), out.stride(3),
        chunk_size=L, headdim=headdim, dstate=dstate,
        nheads=nheads, ngroups=ngroups, nheads_per_group=nheads_per_group,
        scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_DSTATE=BLOCK_DSTATE,
    )

    return out.view(batch, nchunks, L, nheads, headdim)


# =============================================================================
# Combined forward: orchestrates all 5 steps
# =============================================================================

def ssd_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, scale):
    """
    Full SSD forward using Triton kernels. L is never materialized.

    Args:
        x:  (batch, seqlen, nheads, headdim) — input after conv+silu
        dt: (batch, seqlen, nheads) — discretization timestep (already softplus'd + clamped)
        A:  (nheads,) — diagonal state decay (negative)
        B:  (batch, seqlen, ngroups, dstate) — input-to-state
        C:  (batch, seqlen, ngroups, dstate) — state-to-output
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
