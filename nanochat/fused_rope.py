import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    seqlen, head_dim: tl.constexpr, rot_dim: tl.constexpr,
    n_heads: tl.constexpr, n_kv_heads: tl.constexpr,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    BLOCK_SEQ: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    seq_start = pid_seq * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offs < seqlen
    
    half_rot_dim = rot_dim // 2
    dim_offs_low = tl.arange(0, half_rot_dim)
    dim_offs_high = half_rot_dim + tl.arange(0, half_rot_dim)

    # Load Cos/Sin for the current seq positions (broadcasting across batch/heads)
    cos = tl.load(cos_ptr + seq_offs[:, None] * stride_cos_seq + dim_offs_low[None, :] * stride_cos_dim, mask=seq_mask[:, None], other=1.0)
    sin = tl.load(sin_ptr + seq_offs[:, None] * stride_sin_seq + dim_offs_low[None, :] * stride_sin_dim, mask=seq_mask[:, None], other=0.0)

    # Apply RoPE to Queries and Keys
    if pid_head < n_heads:
        h = pid_head
        q_base = q_ptr + pid_batch * stride_q_batch + h * stride_q_head
        
        q1 = tl.load(q_base + seq_offs[:, None] * stride_q_seq + dim_offs_low[None, :] * stride_q_dim, mask=seq_mask[:, None])
        q2 = tl.load(q_base + seq_offs[:, None] * stride_q_seq + dim_offs_high[None, :] * stride_q_dim, mask=seq_mask[:, None])
        
        q1_out = q1 * cos - q2 * sin
        q2_out = q1 * sin + q2 * cos
        
        tl.store(q_base + seq_offs[:, None] * stride_q_seq + dim_offs_low[None, :] * stride_q_dim, q1_out, mask=seq_mask[:, None])
        tl.store(q_base + seq_offs[:, None] * stride_q_seq + dim_offs_high[None, :] * stride_q_dim, q2_out, mask=seq_mask[:, None])

    else:
        h_kv = pid_head - n_heads
        k_base = k_ptr + pid_batch * stride_k_batch + h_kv * stride_k_head
        
        k1 = tl.load(k_base + seq_offs[:, None] * stride_k_seq + dim_offs_low[None, :] * stride_k_dim, mask=seq_mask[:, None])
        k2 = tl.load(k_base + seq_offs[:, None] * stride_k_seq + dim_offs_high[None, :] * stride_k_dim, mask=seq_mask[:, None])
        
        k1_out = k1 * cos - k2 * sin
        k2_out = k1 * sin + k2 * cos
        
        tl.store(k_base + seq_offs[:, None] * stride_k_seq + dim_offs_low[None, :] * stride_k_dim, k1_out, mask=seq_mask[:, None])
        tl.store(k_base + seq_offs[:, None] * stride_k_seq + dim_offs_high[None, :] * stride_k_dim, k2_out, mask=seq_mask[:, None])


def apply_rotary_emb_triton(q, k, cos, sin):
    """
    Applies Rotary Position Embedding in-place to Queries and Keys.
    Args:
        q: (batch, seq, n_heads, head_dim)
        k: (batch, seq, n_kv_heads, head_dim)
        cos: (batch, seq, 1, head_dim // 2) or (1, seq, 1, head_dim // 2)
        sin: (batch, seq, 1, head_dim // 2) or (1, seq, 1, head_dim // 2)
    Returns:
        q, k (both modified in place)
    """
    batch, seqlen, n_heads, head_dim = q.shape
    _, _, n_kv_heads, _ = k.shape
    rot_dim = cos.shape[-1] * 2

    BLOCK_SEQ = min(128, triton.next_power_of_2(seqlen))
    grid = (triton.cdiv(seqlen, BLOCK_SEQ), batch, n_heads + n_kv_heads)

    _rotary_embedding_kernel[grid](
        q, k, cos, sin,
        seqlen, head_dim, rot_dim,
        n_heads, n_kv_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(1), cos.stride(3),
        sin.stride(1), sin.stride(3),
        BLOCK_SEQ=BLOCK_SEQ,
    )
    return q, k


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr, y_ptr,
    stride_x_batch, stride_y_batch,
    seqlen: tl.constexpr, d_model: tl.constexpr, eps: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    pid_b = tl.program_id(0)
    
    x_base = x_ptr + pid_b * stride_x_batch
    y_base = y_ptr + pid_b * stride_y_batch
    
    offs_d = tl.arange(0, BLOCK_DIM)
    mask = offs_d < d_model
    
    # Load all dimensions for the token
    x = tl.load(x_base + offs_d, mask=mask, other=0.0).to(tl.float32)
    
    # Compute variance
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / d_model
    
    # Rsqrt and normalize
    rsqrt = tl.math.rsqrt(variance + eps)
    y = x * rsqrt
    
    tl.store(y_base + offs_d, y.to(x_ptr.dtype.element_ty), mask=mask)

def fused_rms_norm_triton(x, eps=1e-5):
    """
    Applies RMSNorm fused over the last dimension.
    Args:
        x: (..., d_model)
    Returns:
        y: (..., d_model)
    """
    # Flatten everything except the final dimension
    d_model = x.shape[-1]
    x_flat = x.contiguous().view(-1, d_model)
    out = torch.empty_like(x_flat)
    
    batch_tokens = x_flat.shape[0]
    BLOCK_DIM = triton.next_power_of_2(d_model)
    
    grid = (batch_tokens,)
    
    _fused_rms_norm_kernel[grid](
        x_flat, out,
        x_flat.stride(0), out.stride(0),
        seqlen=batch_tokens, d_model=d_model, eps=eps,
        BLOCK_DIM=BLOCK_DIM
    )
    
    return out.view_as(x)
