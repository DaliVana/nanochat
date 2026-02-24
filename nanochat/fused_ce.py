"""
Chunked Softcap Cross Entropy
Replaces: `loss = F.cross_entropy(softcap * tanh(x @ weight.T / softcap), targets)`

Uses cuBLAS for matmuls in chunks along the vocab dimension, avoiding
materializing the full [B*T, V] logits tensor while retaining GEMM speed.
"""

import torch

# Vocab chunk size: trades peak memory for kernel launch overhead.
# 16384 keeps per-chunk logits at ~1GB (float32) while limiting loop iterations.
_CHUNK_V = 16384


class ChunkedCrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, softcap=0.0, ignore_index=-1, reduction='mean'):
        orig_shape = x.shape
        M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        K = x.shape[-1]
        N = weight.shape[0]

        # Ensure matching dtypes for matmuls (weight may be fp32, x may be bf16)
        compute_dtype = x.dtype
        x_flat = x.reshape(M, K)
        w = weight.to(compute_dtype) if weight.dtype != compute_dtype else weight
        targets_flat = targets.reshape(M)
        valid_mask = targets_flat != ignore_index

        # Online logsumexp accumulators
        max_logit = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)
        sum_exp = torch.zeros(M, device=x.device, dtype=torch.float32)
        target_logits = torch.zeros(M, device=x.device, dtype=torch.float32)
        rows = torch.arange(M, device=x.device)

        for n0 in range(0, N, _CHUNK_V):
            n1 = min(n0 + _CHUNK_V, N)
            chunk_n = n1 - n0

            # cuBLAS GEMM: [M, K] @ [chunk, K]^T -> [M, chunk]
            logits = (x_flat @ w[n0:n1].T).float()
            if softcap > 0.0:
                logits = softcap * torch.tanh(logits / softcap)

            # Online logsumexp
            chunk_max = logits.max(dim=1).values
            new_max = torch.maximum(max_logit, chunk_max)
            sum_exp = sum_exp * torch.exp(max_logit - new_max) + torch.exp(logits - new_max.unsqueeze(1)).sum(dim=1)
            max_logit = new_max

            # Extract target logits — no .any() CUDA sync, always execute masked
            in_chunk = valid_mask & (targets_flat >= n0) & (targets_flat < n1)
            local_idx = (targets_flat - n0).clamp(0, chunk_n - 1)
            target_logits += torch.where(in_chunk, logits[rows, local_idx], target_logits.new_zeros(()))

        lse = max_logit + torch.log(sum_exp)
        loss = torch.where(valid_mask, lse - target_logits, lse.new_zeros(()))

        # Normalizer (kept as tensor — no .item() CUDA sync)
        valid_count = valid_mask.sum().clamp(min=1).float()
        if reduction == 'mean':
            inv_normalizer = 1.0 / valid_count
            out_loss = loss.sum() * inv_normalizer
        elif reduction == 'sum':
            inv_normalizer = loss.new_ones(())
            out_loss = loss.sum()
        else:
            inv_normalizer = loss.new_ones(())
            out_loss = loss.view(orig_shape[:-1])

        ctx.save_for_backward(x_flat, w, targets_flat, lse)
        ctx.softcap = softcap
        ctx.ignore_index = ignore_index
        ctx.inv_normalizer = inv_normalizer
        ctx.M, ctx.N, ctx.K = M, N, K
        ctx.orig_shape = orig_shape
        return out_loss

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w, targets_flat, lse = ctx.saved_tensors
        M, N = ctx.M, ctx.N
        valid_mask = targets_flat != ctx.ignore_index

        # Pre-compute per-token grad scale
        if grad_output.numel() == 1:
            g = grad_output * ctx.inv_normalizer  # scalar tensor
        else:
            g = grad_output.reshape(M) * ctx.inv_normalizer  # [M]

        dx = torch.zeros_like(x_flat)
        dw = torch.empty_like(w)
        rows = torch.arange(M, device=x_flat.device)

        for n0 in range(0, N, _CHUNK_V):
            n1 = min(n0 + _CHUNK_V, N)
            chunk_n = n1 - n0

            # Recompute logits (cuBLAS)
            logits = (x_flat @ w[n0:n1].T).float()

            if ctx.softcap > 0.0:
                t = torch.tanh(logits / ctx.softcap)
                logits_capped = ctx.softcap * t
                dsoftcap = 1.0 - t * t
            else:
                logits_capped = logits
                dsoftcap = 1.0

            # dp = softmax - one_hot(target), no .any() CUDA sync
            dp = torch.exp(logits_capped - lse.unsqueeze(1))
            in_chunk = valid_mask & (targets_flat >= n0) & (targets_flat < n1)
            local_idx = (targets_flat - n0).clamp(0, chunk_n - 1)
            dp[rows, local_idx] -= in_chunk.float()

            # Zero out ignored tokens, apply softcap derivative
            dp *= valid_mask.unsqueeze(1)
            dp *= dsoftcap

            # Apply grad scaling
            if g.numel() == 1:
                dp *= g
            else:
                dp *= g.unsqueeze(1)

            dp = dp.to(x_flat.dtype)

            # cuBLAS GEMMs for parameter gradients
            dx.addmm_(dp, w[n0:n1])               # [M, K] += [M, chunk] @ [chunk, K]
            torch.mm(dp.T, x_flat, out=dw[n0:n1]) # [chunk, K] = [chunk, M] @ [M, K]

        return dx.reshape(ctx.orig_shape), dw, None, None, None, None


def fused_cross_entropy(x, weight, targets, softcap=0.0, ignore_index=-1, reduction='mean'):
    return ChunkedCrossEntropyLossFn.apply(x, weight, targets, softcap, ignore_index, reduction)
