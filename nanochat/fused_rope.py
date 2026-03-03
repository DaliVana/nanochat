import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    seqlen,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    GQA_RATIO: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, HALF_ROT_DIM: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)  # iterates over Q heads only

    seq_start = pid_seq * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offs < seqlen

    dim_offs_low = tl.arange(0, HALF_ROT_DIM)
    dim_offs_high = HALF_ROT_DIM + tl.arange(0, HALF_ROT_DIM)
    seq_dim_low = seq_offs[:, None] * stride_q_seq + dim_offs_low[None, :] * stride_q_dim
    seq_dim_high = seq_offs[:, None] * stride_q_seq + dim_offs_high[None, :] * stride_q_dim

    # Load cos/sin once — shared by Q and K
    cos = tl.load(cos_ptr + seq_offs[:, None] * stride_cos_seq + dim_offs_low[None, :] * stride_cos_dim, mask=seq_mask[:, None], other=1.0)
    sin = tl.load(sin_ptr + seq_offs[:, None] * stride_sin_seq + dim_offs_low[None, :] * stride_sin_dim, mask=seq_mask[:, None], other=0.0)

    # Apply RoPE to Q head
    q_base = q_ptr + pid_batch * stride_q_batch + pid_head * stride_q_head
    q1 = tl.load(q_base + seq_dim_low, mask=seq_mask[:, None])
    q2 = tl.load(q_base + seq_dim_high, mask=seq_mask[:, None])
    tl.store(q_base + seq_dim_low, q1 * cos + q2 * sin, mask=seq_mask[:, None])
    tl.store(q_base + seq_dim_high, q1 * (-sin) + q2 * cos, mask=seq_mask[:, None])

    # Apply RoPE to corresponding K head (once per GQA group)
    if pid_head % GQA_RATIO == 0:
        h_kv = pid_head // GQA_RATIO
        k_base = k_ptr + pid_batch * stride_k_batch + h_kv * stride_k_head
        k_seq_dim_low = seq_offs[:, None] * stride_k_seq + dim_offs_low[None, :] * stride_k_dim
        k_seq_dim_high = seq_offs[:, None] * stride_k_seq + dim_offs_high[None, :] * stride_k_dim
        k1 = tl.load(k_base + k_seq_dim_low, mask=seq_mask[:, None])
        k2 = tl.load(k_base + k_seq_dim_high, mask=seq_mask[:, None])
        tl.store(k_base + k_seq_dim_low, k1 * cos + k2 * sin, mask=seq_mask[:, None])
        tl.store(k_base + k_seq_dim_high, k1 * (-sin) + k2 * cos, mask=seq_mask[:, None])


def _launch_rotary_kernel(q, k, cos, sin):
    """Launch the Triton rotary embedding kernel (in-place on q and k)."""
    batch, seqlen, n_heads, _ = q.shape
    _, _, n_kv_heads, _ = k.shape
    half_rot_dim = cos.shape[-1]
    gqa_ratio = n_heads // n_kv_heads

    BLOCK_SEQ = min(128, triton.next_power_of_2(seqlen))
    num_warps = 2 if BLOCK_SEQ * half_rot_dim <= 2048 else 4
    grid = (triton.cdiv(seqlen, BLOCK_SEQ), batch, n_heads)

    _rotary_embedding_kernel[grid](
        q, k, cos, sin,
        seqlen,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(1), cos.stride(3),
        sin.stride(1), sin.stride(3),
        GQA_RATIO=gqa_ratio,
        BLOCK_SEQ=BLOCK_SEQ, HALF_ROT_DIM=half_rot_dim,
        num_warps=num_warps,
    )


class _ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q = q.clone()
        k = k.clone()
        _launch_rotary_kernel(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq = dq.clone()
        dk = dk.clone()
        # Inverse rotation: same kernel with negated sin
        _launch_rotary_kernel(dq, dk, cos, -sin)
        return dq, dk, None, None


def apply_rotary_emb_triton(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to Queries and Keys with correct autograd support.
    Args:
        q: (batch, seq, n_heads, head_dim)
        k: (batch, seq, n_kv_heads, head_dim)
        cos: (batch, seq, 1, head_dim // 2) or (1, seq, 1, head_dim // 2)
        sin: (batch, seq, 1, head_dim // 2) or (1, seq, 1, head_dim // 2)
    Returns:
        q, k (new tensors with rotary embeddings applied)
    """
    return _ApplyRotaryEmb.apply(q, k, cos, sin)


@triton.jit
def _rotary_norm_fwd_kernel(
    q_in_ptr, q_out_ptr, k_in_ptr, k_out_ptr,
    cos_ptr, sin_ptr,
    q_rsqrt_ptr, k_rsqrt_ptr,
    seqlen,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    stride_qr_batch, stride_qr_seq, stride_qr_head,
    stride_kr_batch, stride_kr_seq, stride_kr_head,
    GQA_RATIO: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, HALF_ROT_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)

    seq_start = pid_seq * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offs < seqlen

    dim_offs_low = tl.arange(0, HALF_ROT_DIM)
    dim_offs_high = HALF_ROT_DIM + tl.arange(0, HALF_ROT_DIM)
    seq_dim_q_low = seq_offs[:, None] * stride_q_seq + dim_offs_low[None, :] * stride_q_dim
    seq_dim_q_high = seq_offs[:, None] * stride_q_seq + dim_offs_high[None, :] * stride_q_dim

    # Load cos/sin once — shared by Q and K
    cos = tl.load(cos_ptr + seq_offs[:, None] * stride_cos_seq + dim_offs_low[None, :] * stride_cos_dim,
                  mask=seq_mask[:, None], other=1.0)
    sin = tl.load(sin_ptr + seq_offs[:, None] * stride_sin_seq + dim_offs_low[None, :] * stride_sin_dim,
                  mask=seq_mask[:, None], other=0.0)

    # --- Q head: RoPE + RMSNorm ---
    q_in_base = q_in_ptr + pid_batch * stride_q_batch + pid_head * stride_q_head
    q1 = tl.load(q_in_base + seq_dim_q_low, mask=seq_mask[:, None])
    q2 = tl.load(q_in_base + seq_dim_q_high, mask=seq_mask[:, None])

    # Rotate
    y1 = q1 * cos + q2 * sin
    y2 = q1 * (-sin) + q2 * cos

    # RMSNorm in f32
    y1_f = y1.to(tl.float32)
    y2_f = y2.to(tl.float32)
    head_dim = 2 * HALF_ROT_DIM
    var = (tl.sum(y1_f * y1_f, axis=1) + tl.sum(y2_f * y2_f, axis=1)) / head_dim
    rs = tl.rsqrt(var + EPS)
    y1_n = (y1_f * rs[:, None]).to(q_in_ptr.dtype.element_ty)
    y2_n = (y2_f * rs[:, None]).to(q_in_ptr.dtype.element_ty)

    # Store normalized output
    q_out_base = q_out_ptr + pid_batch * stride_q_batch + pid_head * stride_q_head
    tl.store(q_out_base + seq_dim_q_low, y1_n, mask=seq_mask[:, None])
    tl.store(q_out_base + seq_dim_q_high, y2_n, mask=seq_mask[:, None])

    # Store rsqrt scalar
    qr_base = q_rsqrt_ptr + pid_batch * stride_qr_batch + pid_head * stride_qr_head
    tl.store(qr_base + seq_offs * stride_qr_seq, rs, mask=seq_mask)

    # --- K head: RoPE + RMSNorm (once per GQA group) ---
    if pid_head % GQA_RATIO == 0:
        h_kv = pid_head // GQA_RATIO
        k_in_base = k_in_ptr + pid_batch * stride_k_batch + h_kv * stride_k_head
        seq_dim_k_low = seq_offs[:, None] * stride_k_seq + dim_offs_low[None, :] * stride_k_dim
        seq_dim_k_high = seq_offs[:, None] * stride_k_seq + dim_offs_high[None, :] * stride_k_dim

        k1 = tl.load(k_in_base + seq_dim_k_low, mask=seq_mask[:, None])
        k2 = tl.load(k_in_base + seq_dim_k_high, mask=seq_mask[:, None])

        ky1 = k1 * cos + k2 * sin
        ky2 = k1 * (-sin) + k2 * cos

        ky1_f = ky1.to(tl.float32)
        ky2_f = ky2.to(tl.float32)
        k_var = (tl.sum(ky1_f * ky1_f, axis=1) + tl.sum(ky2_f * ky2_f, axis=1)) / head_dim
        k_rs = tl.rsqrt(k_var + EPS)
        ky1_n = (ky1_f * k_rs[:, None]).to(k_in_ptr.dtype.element_ty)
        ky2_n = (ky2_f * k_rs[:, None]).to(k_in_ptr.dtype.element_ty)

        k_out_base = k_out_ptr + pid_batch * stride_k_batch + h_kv * stride_k_head
        tl.store(k_out_base + seq_dim_k_low, ky1_n, mask=seq_mask[:, None])
        tl.store(k_out_base + seq_dim_k_high, ky2_n, mask=seq_mask[:, None])

        kr_base = k_rsqrt_ptr + pid_batch * stride_kr_batch + h_kv * stride_kr_head
        tl.store(kr_base + seq_offs * stride_kr_seq, k_rs, mask=seq_mask)


@triton.jit
def _rotary_norm_bwd_kernel(
    dq_ptr, dk_ptr,
    q_out_ptr, k_out_ptr,
    cos_ptr, sin_ptr,
    q_rsqrt_ptr, k_rsqrt_ptr,
    dq_in_ptr, dk_in_ptr,
    seqlen,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    stride_qr_batch, stride_qr_seq, stride_qr_head,
    stride_kr_batch, stride_kr_seq, stride_kr_head,
    GQA_RATIO: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, HALF_ROT_DIM: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)

    seq_start = pid_seq * BLOCK_SEQ
    seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offs < seqlen

    dim_offs_low = tl.arange(0, HALF_ROT_DIM)
    dim_offs_high = HALF_ROT_DIM + tl.arange(0, HALF_ROT_DIM)
    seq_dim_q_low = seq_offs[:, None] * stride_q_seq + dim_offs_low[None, :] * stride_q_dim
    seq_dim_q_high = seq_offs[:, None] * stride_q_seq + dim_offs_high[None, :] * stride_q_dim

    # Load cos/sin (bf16 → f32)
    cos_f = tl.load(cos_ptr + seq_offs[:, None] * stride_cos_seq + dim_offs_low[None, :] * stride_cos_dim,
                    mask=seq_mask[:, None], other=1.0).to(tl.float32)
    sin_f = tl.load(sin_ptr + seq_offs[:, None] * stride_sin_seq + dim_offs_low[None, :] * stride_sin_dim,
                    mask=seq_mask[:, None], other=0.0).to(tl.float32)

    head_dim = 2 * HALF_ROT_DIM

    # --- Q head backward ---
    q_off = pid_batch * stride_q_batch + pid_head * stride_q_head

    dq_low = tl.load(dq_ptr + q_off + seq_dim_q_low, mask=seq_mask[:, None]).to(tl.float32)
    dq_high = tl.load(dq_ptr + q_off + seq_dim_q_high, mask=seq_mask[:, None]).to(tl.float32)
    q_low = tl.load(q_out_ptr + q_off + seq_dim_q_low, mask=seq_mask[:, None]).to(tl.float32)
    q_high = tl.load(q_out_ptr + q_off + seq_dim_q_high, mask=seq_mask[:, None]).to(tl.float32)

    qr_base = q_rsqrt_ptr + pid_batch * stride_qr_batch + pid_head * stride_qr_head
    rs = tl.load(qr_base + seq_offs * stride_qr_seq, mask=seq_mask)

    # RMSNorm backward: dx = s * (dy - y * dot(y, dy) / d)
    dot = tl.sum(q_low * dq_low, axis=1) + tl.sum(q_high * dq_high, axis=1)
    drot_low = rs[:, None] * (dq_low - q_low * (dot[:, None] / head_dim))
    drot_high = rs[:, None] * (dq_high - q_high * (dot[:, None] / head_dim))

    # Inverse rotation: R^T = [[cos, -sin], [sin, cos]]
    dx_low = drot_low * cos_f - drot_high * sin_f
    dx_high = drot_low * sin_f + drot_high * cos_f

    tl.store(dq_in_ptr + q_off + seq_dim_q_low, dx_low.to(dq_ptr.dtype.element_ty), mask=seq_mask[:, None])
    tl.store(dq_in_ptr + q_off + seq_dim_q_high, dx_high.to(dq_ptr.dtype.element_ty), mask=seq_mask[:, None])

    # --- K head backward (once per GQA group) ---
    if pid_head % GQA_RATIO == 0:
        h_kv = pid_head // GQA_RATIO
        k_off = pid_batch * stride_k_batch + h_kv * stride_k_head
        seq_dim_k_low = seq_offs[:, None] * stride_k_seq + dim_offs_low[None, :] * stride_k_dim
        seq_dim_k_high = seq_offs[:, None] * stride_k_seq + dim_offs_high[None, :] * stride_k_dim

        dk_low = tl.load(dk_ptr + k_off + seq_dim_k_low, mask=seq_mask[:, None]).to(tl.float32)
        dk_high = tl.load(dk_ptr + k_off + seq_dim_k_high, mask=seq_mask[:, None]).to(tl.float32)
        k_low = tl.load(k_out_ptr + k_off + seq_dim_k_low, mask=seq_mask[:, None]).to(tl.float32)
        k_high = tl.load(k_out_ptr + k_off + seq_dim_k_high, mask=seq_mask[:, None]).to(tl.float32)

        kr_base = k_rsqrt_ptr + pid_batch * stride_kr_batch + h_kv * stride_kr_head
        k_rs = tl.load(kr_base + seq_offs * stride_kr_seq, mask=seq_mask)

        k_dot = tl.sum(k_low * dk_low, axis=1) + tl.sum(k_high * dk_high, axis=1)
        k_drot_low = k_rs[:, None] * (dk_low - k_low * (k_dot[:, None] / head_dim))
        k_drot_high = k_rs[:, None] * (dk_high - k_high * (k_dot[:, None] / head_dim))

        k_dx_low = k_drot_low * cos_f - k_drot_high * sin_f
        k_dx_high = k_drot_low * sin_f + k_drot_high * cos_f

        tl.store(dk_in_ptr + k_off + seq_dim_k_low, k_dx_low.to(dk_ptr.dtype.element_ty), mask=seq_mask[:, None])
        tl.store(dk_in_ptr + k_off + seq_dim_k_high, k_dx_high.to(dk_ptr.dtype.element_ty), mask=seq_mask[:, None])


def _launch_rotary_norm_fwd(q, k, cos, sin, q_out, k_out, q_rsqrt, k_rsqrt):
    batch, seqlen, n_heads, _ = q.shape
    _, _, n_kv_heads, _ = k.shape
    half_rot_dim = cos.shape[-1]
    gqa_ratio = n_heads // n_kv_heads

    BLOCK_SEQ = min(128, triton.next_power_of_2(seqlen))
    num_warps = 2 if BLOCK_SEQ * half_rot_dim <= 2048 else 4
    grid = (triton.cdiv(seqlen, BLOCK_SEQ), batch, n_heads)

    _rotary_norm_fwd_kernel[grid](
        q, q_out, k, k_out,
        cos, sin,
        q_rsqrt, k_rsqrt,
        seqlen,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(1), cos.stride(3),
        sin.stride(1), sin.stride(3),
        q_rsqrt.stride(0), q_rsqrt.stride(1), q_rsqrt.stride(2),
        k_rsqrt.stride(0), k_rsqrt.stride(1), k_rsqrt.stride(2),
        GQA_RATIO=gqa_ratio,
        BLOCK_SEQ=BLOCK_SEQ, HALF_ROT_DIM=half_rot_dim,
        EPS=1e-5,
        num_warps=num_warps,
    )


def _launch_rotary_norm_bwd(dq, dk, q_out, k_out, cos, sin, q_rsqrt, k_rsqrt, dq_in, dk_in):
    batch, seqlen, n_heads, _ = dq.shape
    _, _, n_kv_heads, _ = dk.shape
    half_rot_dim = cos.shape[-1]
    gqa_ratio = n_heads // n_kv_heads

    BLOCK_SEQ = min(128, triton.next_power_of_2(seqlen))
    num_warps = 2 if BLOCK_SEQ * half_rot_dim <= 2048 else 4
    grid = (triton.cdiv(seqlen, BLOCK_SEQ), batch, n_heads)

    _rotary_norm_bwd_kernel[grid](
        dq, dk,
        q_out, k_out,
        cos, sin,
        q_rsqrt, k_rsqrt,
        dq_in, dk_in,
        seqlen,
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        cos.stride(1), cos.stride(3),
        sin.stride(1), sin.stride(3),
        q_rsqrt.stride(0), q_rsqrt.stride(1), q_rsqrt.stride(2),
        k_rsqrt.stride(0), k_rsqrt.stride(1), k_rsqrt.stride(2),
        GQA_RATIO=gqa_ratio,
        BLOCK_SEQ=BLOCK_SEQ, HALF_ROT_DIM=half_rot_dim,
        num_warps=num_warps,
    )


class _ApplyRotaryNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        batch, seqlen, n_heads, _ = q.shape
        _, _, n_kv_heads, _ = k.shape
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        q_rsqrt = torch.empty(batch, seqlen, n_heads, device=q.device, dtype=torch.float32)
        k_rsqrt = torch.empty(batch, seqlen, n_kv_heads, device=k.device, dtype=torch.float32)
        _launch_rotary_norm_fwd(q, k, cos, sin, q_out, k_out, q_rsqrt, k_rsqrt)
        ctx.save_for_backward(q_out, k_out, cos, sin, q_rsqrt, k_rsqrt)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        q_out, k_out, cos, sin, q_rsqrt, k_rsqrt = ctx.saved_tensors
        dq_in = torch.empty_like(dq)
        dk_in = torch.empty_like(dk)
        _launch_rotary_norm_bwd(dq, dk, q_out, k_out, cos, sin, q_rsqrt, k_rsqrt, dq_in, dk_in)
        return dq_in, dk_in, None, None


def apply_rotary_norm_triton(q, k, cos, sin):
    """
    Fused Rotary Position Embedding + RMSNorm for Queries and Keys.
    Saves two memory round-trips by computing RMSNorm immediately after rotation
    while values are still in registers.
    Args:
        q: (batch, seq, n_heads, head_dim)
        k: (batch, seq, n_kv_heads, head_dim)
        cos: (1, seq, 1, head_dim // 2)
        sin: (1, seq, 1, head_dim // 2)
    Returns:
        q, k with rotary embeddings applied and RMSNorm'd
    """
    return _ApplyRotaryNorm.apply(q, k, cos, sin)


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
    rsqrt = tl.rsqrt(variance + eps)
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
