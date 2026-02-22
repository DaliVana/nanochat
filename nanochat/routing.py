"""
Routing utilities for sparse model variants (MoD, MoH, MoE).

Provides:
- TopKRouter: Learned routing with top-k selection (used by MoH, MoD)
- SigmoidRouter: Sigmoid-gated top-K router with aux-loss-free bias balancing (used by MoE, MoVaE)
- Load balancing losses for even distribution
- Routing statistics for monitoring
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class SigmoidRouter(nn.Module):
    """Sigmoid-gated top-K router with auxiliary-loss-free load balancing (DeepSeekV3).

    Each token independently picks K experts via sigmoid scoring. Expert selection
    is biased to encourage balanced load, but the bias does NOT affect gating weights
    (only selection), so the training loss remains pure language modeling.

    Args:
        dim: Input feature dimension
        num_experts: Total number of routable experts
        top_k: Number of experts selected per token
    """

    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        # Auxiliary-loss-free load balancing (DeepSeekV3)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('tokens_per_expert_counter', torch.zeros(num_experts))

    def forward(self, x):
        """
        Args:
            x: (T, dim) flattened token representations
        Returns:
            top_scores:             (T, top_k)    routing weights for selected experts
            selected_experts:       (T, top_k)    which experts each token chose
            num_tokens_per_expert:  (num_experts,) how many tokens each expert received
        """
        scores = self.gate(x)                       # (T, num_experts)
        scores = torch.sigmoid(scores.float())      # values in (0, 1)
        # Bias affects expert SELECTION but not gating weights (DeepSeekV3)
        biased_scores = scores + self.expert_bias
        _, selected_experts = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)
        top_scores = scores.gather(dim=-1, index=selected_experts)
        num_tokens_per_expert = torch.histc(
            selected_experts.float().view(-1),
            bins=self.num_experts, min=0, max=self.num_experts,
        )
        # Accumulate token counts for load balancing updates
        self.tokens_per_expert_counter += num_tokens_per_expert
        return top_scores, selected_experts, num_tokens_per_expert

    def update_expert_bias(self, coeff=1e-3):
        """Auxiliary-loss-free bias update (DeepSeekV3). Call before optimizer.step()."""
        counts = self.tokens_per_expert_counter
        # Sync token counts across GPUs if distributed
        if dist.is_initialized():
            dist.all_reduce(counts)
        if counts.sum() == 0:
            return
        mean_count = counts.mean()
        # Nudge underloaded experts up, overloaded experts down
        self.expert_bias += coeff * torch.sign(mean_count - counts)
        self.expert_bias -= self.expert_bias.mean()  # center to prevent drift
        self.tokens_per_expert_counter.zero_()


class TopKRouter(nn.Module):
    """
    Learned router that selects top-k items from n options.

    Args:
        input_dim: Dimension of input features
        num_options: Number of options to route between (layers, heads, or experts)

    Forward:
        x: Input tensor (B, T, input_dim)
        Returns: routing scores (B, T, num_options)
    """
    def __init__(self, input_dim, num_options):
        super().__init__()
        self.router = nn.Linear(input_dim, num_options, bias=False)

    def forward(self, x):
        """Compute routing scores for each option."""
        return self.router(x)  # (B, T, num_options)


def top_k_routing(scores, k, return_mask=False):
    """
    Select top-k options based on routing scores.
    Uses straight-through estimator for gradients during training.

    Args:
        scores: Routing scores (B, T, N) where N is number of options
        k: Number of options to select (int or float ratio)
        return_mask: If True, return binary mask instead of indices

    Returns:
        If return_mask=False: indices of top-k options (B, T, k)
        If return_mask=True: binary mask (B, T, N) with 1 for selected, 0 for others
    """
    B, T, N = scores.shape

    # Handle k as ratio (float) or absolute count (int)
    if isinstance(k, float):
        k = max(1, int(k * N))

    # Get top-k indices
    _, indices = torch.topk(scores, k, dim=-1)  # (B, T, k)

    if return_mask:
        # Create binary mask
        mask = torch.zeros_like(scores)  # (B, T, N)
        mask.scatter_(-1, indices, 1.0)

        # Straight-through estimator: use mask in forward, scores.softmax() in backward
        # Only compute softmax if we actually need gradients (training mode)
        if scores.requires_grad and torch.is_grad_enabled():
            probs = F.softmax(scores, dim=-1)
            mask = mask + probs - probs.detach()  # Gradient flows through probs

        return mask
    else:
        return indices


def compute_load_balance_loss(routing_probs, num_options, weight=0.01):
    """
    Compute load balancing loss to encourage even distribution across options.

    Minimizes the variance in option usage across the batch.

    Args:
        routing_probs: Routing probabilities or binary masks (B, T, num_options)
        num_options: Total number of options
        weight: Loss weight coefficient

    Returns:
        Scalar loss value
    """
    # Average probability per option across batch and sequence
    mean_probs = routing_probs.mean(dim=[0, 1])  # (num_options,)

    # Target: uniform distribution
    target = 1.0 / num_options

    # Loss: mean squared error from uniform distribution
    loss = weight * ((mean_probs - target) ** 2).sum()

    return loss


def compute_routing_stats(routing_decisions):
    """
    Compute statistics about routing decisions for logging.

    Args:
        routing_decisions: Binary mask (B, T, N) or indices (B, T, k)

    Returns:
        Dict with statistics:
        - mean_usage: Average usage per option (as fraction)
        - std_usage: Standard deviation of usage
        - min_usage: Minimum usage
        - max_usage: Maximum usage
        - entropy: Routing entropy (higher = more balanced)
    """
    if routing_decisions.ndim == 3:
        # Binary mask: (B, T, N)
        usage = routing_decisions.float().mean(dim=[0, 1])  # (N,)
    else:
        # Indices: (B, T, k) - convert to usage counts
        B, T, k = routing_decisions.shape
        num_options = routing_decisions.max().item() + 1
        usage = torch.zeros(num_options, device=routing_decisions.device)
        for i in range(num_options):
            usage[i] = (routing_decisions == i).float().sum() / (B * T)

    # Compute statistics
    stats = {
        'mean_usage': usage.mean().item(),
        'std_usage': usage.std().item(),
        'min_usage': usage.min().item(),
        'max_usage': usage.max().item(),
    }

    # Compute entropy (clip to avoid log(0))
    usage_clipped = torch.clamp(usage, min=1e-10)
    entropy = -(usage_clipped * torch.log(usage_clipped)).sum().item()
    stats['entropy'] = entropy

    return stats


def threshold_routing(scores, threshold=0.5, target_ratio=None, target_weight=0.01):
    """
    Threshold-based routing: each option independently decides whether to activate.

    Each option gets a sigmoid probability. During forward pass, hard threshold at `threshold`.
    Gradients flow through sigmoid via straight-through estimator.

    Args:
        scores: Routing scores (B, T, N) where N is number of options
        threshold: Activation threshold (default 0.5)
        target_ratio: If set, adds a regularization loss to push average activation
                      toward this ratio. Returns (mask, sparsity_loss) tuple.
        target_weight: Weight for the sparsity regularization loss

    Returns:
        If target_ratio is None: binary mask (B, T, N)
        If target_ratio is set: (binary mask, sparsity_loss) tuple
    """
    # Sigmoid gives independent per-option probabilities
    probs = torch.sigmoid(scores)  # (B, T, N)

    # Hard threshold in forward, sigmoid gradients in backward (straight-through)
    hard_mask = (probs >= threshold).float()
    mask = hard_mask + probs - probs.detach()  # STE

    if target_ratio is not None:
        # Encourage average activation to match target_ratio
        mean_activation = probs.mean()
        sparsity_loss = target_weight * (mean_activation - target_ratio) ** 2
        return mask, sparsity_loss

    return mask


class StraightThroughTopK(torch.autograd.Function):
    """
    Top-k selection with straight-through estimator for gradients.
    Forward: Hard top-k selection (binary mask)
    Backward: Gradient flows through softmax probabilities
    """
    @staticmethod
    def forward(ctx, scores, k):
        """
        Args:
            scores: (B, T, N) routing scores
            k: number of options to select

        Returns:
            mask: (B, T, N) binary mask with 1 for selected, 0 for others
        """
        B, T, N = scores.shape

        # Handle k as ratio or count
        if isinstance(k, float):
            k = max(1, int(k * N))

        # Get top-k indices
        _, indices = torch.topk(scores, k, dim=-1)

        # Create binary mask
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, indices, 1.0)

        # Save softmax for backward
        probs = F.softmax(scores, dim=-1)
        ctx.save_for_backward(probs)

        return mask

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient flows through softmax probabilities."""
        probs, = ctx.saved_tensors
        return grad_output * probs, None  # None for k (no gradient)
