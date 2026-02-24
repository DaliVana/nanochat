"""
Fused Softcap Cross Entropy Kernel
Replaces: `loss = F.cross_entropy(softcap * tanh(x @ weight.T / softcap), targets)`

This avoids materializing the full [B, T, V] logits tensor in global memory by computing
the chunked matmul and the cross_entropy reduction on-the-fly in SRAM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # Portable tanh: works across all Triton versions
    exp2x = tl.exp(2.0 * x)
    return (exp2x - 1.0) / (exp2x + 1.0)


@triton.jit
def _fused_cross_entropy_fwd_kernel(
    X_ptr, W_ptr, targets_ptr,
    loss_ptr, lse_ptr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_loss, stride_lse,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    softcap: tl.constexpr, ignore_index: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(M, BLOCK_M), 1, 1) - we handle all N within the block to do the logsumexp
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Load targets early so we can mask effectively
    targets = tl.load(targets_ptr + offs_m, mask=m_mask, other=ignore_index)
    valid_mask = (targets != ignore_index) & m_mask

    # We need to compute logsumexp and the logit of the target token.
    # To do this safely over N dimension (which might be 130k+), we loop over N blocks.
    # We keep running max (m) and running sum (l)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Store the target logit when we find it
    target_logits = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Compute logits for this block of N: X[m, k] @ W[n, k].T
        logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.bfloat16)
            # W is of shape (N, K)
            w = tl.load(W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                        mask=n_mask[None, :] & k_mask[:, None], other=0.0).to(tl.bfloat16)

            logits += tl.dot(x, w).to(tl.float32)

        # Apply Softcap
        if softcap > 0.0:
            logits = softcap * _tanh(logits / softcap)

        # Update running max and sum for logsumexp
        # Mask out out-of-bounds N
        logits = tl.where(n_mask[None, :], logits, float("-inf"))
        
        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Scaling factor for previous sum
        alpha = tl.exp(m_i - m_new)
        
        # Sum of exp of current block
        p_ij = tl.exp(logits - m_new[:, None])
        # Mask padded elements before summing
        p_ij = tl.where(n_mask[None, :], p_ij, 0.0)
        l_ij = tl.sum(p_ij, axis=1)
        
        l_i = l_i * alpha + l_ij
        m_i = m_new
        
        # Extract target logits
        # where target in range(n_start, n_start + BLOCK_N)
        is_target_in_block = (targets >= n_start) & (targets < n_start + BLOCK_N) & valid_mask
        
        # Grab the specific logit for the target (offset within the block)
        target_indices = targets - n_start
        
        # we can use tl.arange to create a mask for gather equivalent
        col_indices = tl.arange(0, BLOCK_N)
        target_mask = is_target_in_block[:, None] & (col_indices[None, :] == target_indices[:, None])
        
        # Extract the correctly masked logit
        target_logit_block_vals = tl.sum(tl.where(target_mask, logits, 0.0), axis=1)
        target_logits += target_logit_block_vals

    # Finalize logsumexp
    lse = m_i + tl.log(l_i)
    
    # Loss = lse - target_logit
    loss = lse - target_logits
    
    # Apply ignore_index mask (zero loss for ignored tokens so reduction ignores them)
    loss = tl.where(valid_mask, loss, 0.0)
    
    # Store results
    tl.store(loss_ptr + offs_m * stride_loss, loss, mask=m_mask)
    tl.store(lse_ptr + offs_m * stride_lse, lse, mask=m_mask)


@triton.jit
def _fused_cross_entropy_bwd_kernel(
    X_ptr, W_ptr, targets_ptr, lse_ptr, grad_out_ptr,
    dX_ptr, dW_ptr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_dxm, stride_dxk,
    stride_dwn, stride_dwk,
    stride_lse, stride_grad_out,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    softcap: tl.constexpr, ignore_index: tl.constexpr,
    inv_normalizer: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Backward pass needs to compute:
    # d(Loss) / d(logits) = softmax(logits) - 1(target)
    # Since softcap: dp = (softmax(logits) - 1(target)) * (1 - tanh^2(x@W^T / softcap))
    # dX = dp @ W
    # dW = dp^T @ X
    
    # We tile across M and N to compute dX and dW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    targets = tl.load(targets_ptr + offs_m, mask=m_mask, other=ignore_index)
    lse = tl.load(lse_ptr + offs_m * stride_lse, mask=m_mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offs_m * stride_grad_out, mask=m_mask, other=0.0).to(tl.float32)
    valid_mask = (targets != ignore_index) & m_mask
    
    # Recompute logits for this block
    logits_pre_cap = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.bfloat16)
        w = tl.load(W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                    mask=n_mask[None, :] & k_mask[:, None], other=0.0).to(tl.bfloat16)

        logits_pre_cap += tl.dot(x, w).to(tl.float32)

    # Re-apply softcap
    if softcap > 0.0:
        tanh_val = _tanh(logits_pre_cap / softcap)
        logits = softcap * tanh_val
        # derivative of softcap * tanh(x / softcap) is 1 - tanh^2(x / softcap)
        d_softcap = 1.0 - tanh_val * tanh_val
    else:
        logits = logits_pre_cap
        d_softcap = 1.0

    # dp = softmax(logits) - 1_target
    probs = tl.exp(logits - lse[:, None])
    
    # indicator for targets
    is_target = (targets[:, None] == offs_n[None, :]) & valid_mask[:, None]
    
    dp = probs - tl.where(is_target, 1.0, 0.0)
    
    # apply mask for ignore_index
    dp = tl.where(valid_mask[:, None], dp, 0.0)
    
    # chain rule through softcap
    dp = dp * d_softcap
    
    # scale by loss normalizer and incoming gradient
    dp = dp * grad_out[:, None] * inv_normalizer
    
    # Cast dp to bf16 for tensor-core dot products
    dp_bf16 = dp.to(tl.bfloat16)

    # Accumulate dX
    # dX[m, k] += dp[m, n] @ W[n, k]
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        w_trans = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                          mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.bfloat16)

        dx_val = tl.dot(dp_bf16, w_trans).to(tl.float32)

        # Use atomic add because grid is over N as well
        tl.atomic_add(dX_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk,
                      dx_val, mask=m_mask[:, None] & k_mask[None, :])

    # Accumulate dW
    # dW[n, k] += dp[m, n]^T @ X[m, k]
    dp_trans = tl.trans(dp_bf16)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.bfloat16)

        dw_val = tl.dot(dp_trans, x_val).to(tl.float32)

        # Use atomic add because grid is over M as well
        tl.atomic_add(dW_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
                      dw_val, mask=n_mask[:, None] & k_mask[None, :])


class FastCrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, softcap=0.0, ignore_index=-1, reduction='mean'):
        # Flatten x and targets
        orig_shape = x.shape
        M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        K = x.shape[-1]
        N = weight.shape[0]
        
        x_flat = x.view(M, K)
        targets_flat = targets.view(M)
        
        loss = torch.empty(M, dtype=torch.float32, device=x.device)
        lse = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # Forward tuning params
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = min(64, triton.next_power_of_2(K))
        
        grid = (triton.cdiv(M, BLOCK_M), 1, 1)
        
        _fused_cross_entropy_fwd_kernel[grid](
            x_flat, weight, targets_flat,
            loss, lse,
            x_flat.stride(0), x_flat.stride(1),
            weight.stride(0), weight.stride(1),
            loss.stride(0), lse.stride(0),
            M=M, N=N, K=K,
            softcap=softcap, ignore_index=ignore_index,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )
        
        ctx.save_for_backward(x_flat, weight, targets_flat, lse)
        ctx.softcap = softcap
        ctx.ignore_index = ignore_index
        ctx.M = M
        ctx.N = N
        ctx.K = K
        ctx.orig_shape = orig_shape
        
        # Calculate normalizer
        valid_tokens = (targets_flat != ignore_index).sum()
        if reduction == 'mean':
            inv_normalizer = 1.0 / valid_tokens.clamp(min=1).float().item()
            out_loss = loss.sum() * inv_normalizer
        elif reduction == 'sum':
            inv_normalizer = 1.0
            out_loss = loss.sum()
        else:
            inv_normalizer = 1.0
            out_loss = loss.view(orig_shape[:-1])
            
        ctx.inv_normalizer = inv_normalizer
        return out_loss
        
    @staticmethod
    def backward(ctx, grad_output):
        x_flat, weight, targets_flat, lse = ctx.saved_tensors
        
        # Setup output gradients exactly like inputs
        dx = torch.zeros_like(x_flat)
        dw = torch.zeros_like(weight)
        
        if grad_output.numel() == 1:
            grad_out_flat = grad_output.expand(ctx.M)
        else:
            grad_out_flat = grad_output.contiguous().view(ctx.M)
        
        # Multiply normalizer
        inv_normalizer = ctx.inv_normalizer
        
        # Backward tuning params
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = min(64, triton.next_power_of_2(ctx.K))
        
        grid = (triton.cdiv(ctx.M, BLOCK_M), triton.cdiv(ctx.N, BLOCK_N), 1)
        
        _fused_cross_entropy_bwd_kernel[grid](
            x_flat, weight, targets_flat, lse, grad_out_flat,
            dx, dw,
            x_flat.stride(0), x_flat.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            dw.stride(0), dw.stride(1),
            lse.stride(0), grad_out_flat.stride(0),
            M=ctx.M, N=ctx.N, K=ctx.K,
            softcap=ctx.softcap, ignore_index=ctx.ignore_index,
            inv_normalizer=inv_normalizer,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3
        )
        
        return dx.view(ctx.orig_shape) if len(ctx.needs_input_grad) > 0 else dx, dw, None, None, None, None

def fused_cross_entropy(x, weight, targets, softcap=0.0, ignore_index=-1, reduction='mean'):
    return FastCrossEntropyLossFn.apply(x, weight, targets, softcap, ignore_index, reduction)
