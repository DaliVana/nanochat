"""
MoVE (Mixture of Value Embeddings) CPU Offloading.

This module provides CPU offloading for MoVE embeddings during training and inference.
MoVE's sparse access pattern (only batch tokens needed) and one-time lookup (mixed into
KV cache) make it ideal for offloading—unlike weight matrices that are accessed every
forward pass.

Key components:
- MoVEOffloadedEmbedding: Training-time embedding with CPU storage and async prefetch
- MoVEInferenceEmbedding: Inference-time embedding with LRU cache
- MoVECPUAdamW: CPU-side AdamW optimizer for sparse embedding updates

Usage:
    # During training
    embedding = MoVEOffloadedEmbedding(vocab_size, num_slots, ve_dim, device='cuda')
    optimizer = MoVECPUAdamW(embedding, lr=0.2)

    # In forward pass
    ve = embedding(token_ids)  # Fetches from CPU
    embedding.prefetch(next_batch_ids)  # Async prefetch next batch

    # After backward
    optimizer.step()

    # During inference
    embedding = MoVEInferenceEmbedding(vocab_size, num_slots, ve_dim, device='cuda')
    ve = embedding(token_ids)  # Uses LRU cache
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from collections import OrderedDict


class MoVEOffloadedEmbedding(nn.Module):
    """
    MoVE embedding with CPU offloading for training.

    Features:
    - Embedding weights stored on CPU with pinned memory
    - Async CUDA streams for prefetching next batch
    - Double-buffered transfers to overlap with compute
    - Sparse gradient accumulation on CPU

    The embedding is NOT a standard nn.Embedding because we need fine-grained
    control over CPU-GPU transfers and gradient handling.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,  # = num_slots * ve_dim
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gpu_device = device
        self.dtype = dtype

        # Main embedding weights on CPU with pinned memory for fast transfers
        # Note: pin_memory only works with CUDA, not MPS
        use_pin_memory = device != 'cpu' and torch.cuda.is_available()
        self.register_buffer(
            'weight',
            torch.zeros(num_embeddings, embedding_dim, dtype=dtype, pin_memory=use_pin_memory),
            persistent=True
        )

        # Gradient accumulator on CPU (float32 for precision)
        self.register_buffer(
            'grad_accumulator',
            torch.zeros(num_embeddings, embedding_dim, dtype=torch.float32, pin_memory=use_pin_memory),
            persistent=False
        )

        # Async transfer streams
        if device != 'cpu' and torch.cuda.is_available():
            self.fetch_stream = torch.cuda.Stream()
            self.grad_stream = torch.cuda.Stream()
        else:
            self.fetch_stream = None
            self.grad_stream = None

        # Double buffer for prefetching
        self._prefetch_buffer: Optional[Tensor] = None
        self._prefetch_ids: Optional[Tensor] = None
        self._prefetch_inverse: Optional[Tensor] = None

        # State for backward pass
        self._last_unique_ids: Optional[Tensor] = None
        self._last_inverse: Optional[Tensor] = None

    def init_weights(self, std: float = 0.02):
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def prefetch(self, token_ids: Tensor) -> None:
        """
        Prefetch embeddings for next batch asynchronously.
        Call this during current forward pass to overlap transfer with compute.

        Args:
            token_ids: (B, T) tensor of token indices (can be on GPU)
        """
        if self.fetch_stream is None:
            return  # No async on CPU

        # Get unique tokens on CPU
        unique_ids, inverse = token_ids.flatten().unique(return_inverse=True)
        unique_ids_cpu = unique_ids.cpu()

        with torch.cuda.stream(self.fetch_stream):
            # Fetch subset from CPU embedding
            subset = self.weight[unique_ids_cpu].to(self.gpu_device, non_blocking=True)
            self._prefetch_buffer = subset
            self._prefetch_ids = unique_ids_cpu
            self._prefetch_inverse = inverse

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Fetch embeddings for current batch.

        Args:
            token_ids: (B, T) tensor of token indices on GPU

        Returns:
            (B, T, embedding_dim) tensor of embeddings
        """
        B, T = token_ids.shape
        unique_ids, inverse = token_ids.flatten().unique(return_inverse=True)
        unique_ids_cpu = unique_ids.cpu()

        # Check if we have prefetched data matching this batch
        if (self._prefetch_buffer is not None and
            self._prefetch_ids is not None and
            unique_ids_cpu.shape == self._prefetch_ids.shape and
            torch.equal(unique_ids_cpu, self._prefetch_ids)):
            # Use prefetched buffer
            if self.fetch_stream is not None:
                self.fetch_stream.synchronize()
            subset_embeds = self._prefetch_buffer
            inverse = self._prefetch_inverse
        else:
            # Fetch synchronously (first batch or cache miss)
            subset_embeds = self.weight[unique_ids_cpu].to(self.gpu_device)

        # Clear prefetch state
        self._prefetch_buffer = None
        self._prefetch_ids = None
        self._prefetch_inverse = None

        # Store for backward pass
        self._last_unique_ids = unique_ids_cpu
        self._last_inverse = inverse

        # Index back to full batch shape
        full_embeds = subset_embeds[inverse]  # (B*T, embedding_dim)
        full_embeds = full_embeds.view(B, T, self.embedding_dim)

        return full_embeds

    def accumulate_grad(self, grad: Tensor) -> None:
        """
        Accumulate gradients back to CPU.
        Call this after backward pass with the gradient w.r.t. the embedding output.

        Args:
            grad: (B, T, embedding_dim) gradient tensor
        """
        if self._last_unique_ids is None or self._last_inverse is None:
            raise RuntimeError("accumulate_grad called without prior forward pass")

        B, T, _ = grad.shape
        grad_flat = grad.view(-1, self.embedding_dim)  # (B*T, embedding_dim)

        # Scatter-add gradients for unique tokens
        num_unique = len(self._last_unique_ids)
        unique_grad = torch.zeros(
            num_unique, self.embedding_dim,
            device=grad.device, dtype=torch.float32
        )
        unique_grad.index_add_(0, self._last_inverse, grad_flat.float())

        # Transfer to CPU and accumulate
        if self.grad_stream is not None:
            with torch.cuda.stream(self.grad_stream):
                cpu_grad = unique_grad.cpu()
                self.grad_stream.synchronize()
                self.grad_accumulator[self._last_unique_ids] += cpu_grad
        else:
            cpu_grad = unique_grad.cpu()
            self.grad_accumulator[self._last_unique_ids] += cpu_grad

    def get_and_clear_grad(self) -> Tuple[Tensor, Tensor]:
        """
        Get accumulated gradients and clear buffer.
        Call this before optimizer step.

        Returns:
            (grad, mask) where grad is the full gradient tensor and mask indicates
            which rows have non-zero gradients.
        """
        if self.grad_stream is not None:
            self.grad_stream.synchronize()

        # Find non-zero rows (rows that received gradients)
        grad_norms = self.grad_accumulator.abs().sum(dim=1)
        mask = grad_norms > 0

        grad = self.grad_accumulator.clone()
        self.grad_accumulator.zero_()

        return grad, mask

    def state_dict(self, *args, **kwargs):
        """Override to ensure weight is saved."""
        state = super().state_dict(*args, **kwargs)
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Override to handle weight loading."""
        super().load_state_dict(state_dict, *args, **kwargs)


class MoVECPUAdamW:
    """
    AdamW optimizer running on CPU for MoVE embeddings.

    Features:
    - Sparse updates: only updates rows that received gradients
    - Optimizer state kept on CPU (saves GPU memory)
    - Supports learning rate scheduling via set_lr()

    This is a standalone optimizer, not nn.Module, to avoid
    registering optimizer state as model parameters.
    """

    def __init__(
        self,
        embedding: MoVEOffloadedEmbedding,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.8, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        self.embedding = embedding
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Optimizer state on CPU
        self.m = torch.zeros_like(embedding.weight, dtype=torch.float32)  # First moment
        self.v = torch.zeros_like(embedding.weight, dtype=torch.float32)  # Second moment
        self.step_count = 0

    def set_lr(self, lr: float) -> None:
        """Update learning rate (for scheduling)."""
        self.lr = lr

    @torch.no_grad()
    def step(self) -> None:
        """Perform optimization step on CPU."""
        self.step_count += 1
        grad, mask = self.embedding.get_and_clear_grad()

        if mask.sum() == 0:
            return  # No gradients to apply

        # Bias correction factors
        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count

        # Update moments only for active rows (sparse update)
        self.m[mask] = self.beta1 * self.m[mask] + (1 - self.beta1) * grad[mask]
        self.v[mask] = self.beta2 * self.v[mask] + (1 - self.beta2) * grad[mask].square()

        # Compute bias-corrected estimates
        m_hat = self.m[mask] / bc1
        v_hat = self.v[mask] / bc2

        # Compute update
        update = m_hat / (v_hat.sqrt() + self.eps)

        # Apply weight decay (decoupled)
        if self.weight_decay > 0:
            update = update + self.weight_decay * self.embedding.weight[mask].float()

        # Apply update
        self.embedding.weight[mask] = (
            self.embedding.weight[mask].float() - self.lr * update
        ).to(self.embedding.weight.dtype)

    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            'step_count': self.step_count,
            'm': self.m,
            'v': self.v,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.step_count = state['step_count']
        self.m = state['m']
        self.v = state['v']
        self.lr = state.get('lr', self.lr)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)


class MoVEInferenceEmbedding(nn.Module):
    """
    MoVE embedding for inference with CPU offload and LRU cache.

    Features:
    - Embedding weights on CPU (optionally memory-mapped for very large banks)
    - LRU cache on GPU for frequently accessed tokens
    - Speculative prefetching for likely next tokens

    The LRU cache significantly reduces transfers for common tokens
    (articles, punctuation, frequent words).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,  # = num_slots * ve_dim
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        cache_size: int = 1024,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gpu_device = device
        self.dtype = dtype
        self.cache_size = cache_size

        # Main embedding weights on CPU with pinned memory
        # Note: pin_memory only works with CUDA, not MPS
        use_pin_memory = device != 'cpu' and torch.cuda.is_available()
        self.register_buffer(
            'weight',
            torch.zeros(num_embeddings, embedding_dim, dtype=dtype, pin_memory=use_pin_memory),
            persistent=True
        )

        # LRU cache: OrderedDict maps token_id -> GPU tensor
        self._cache: OrderedDict[int, Tensor] = OrderedDict()

        # Prefetch stream
        if device != 'cpu' and torch.cuda.is_available():
            self._fetch_stream = torch.cuda.Stream()
        else:
            self._fetch_stream = None
        self._prefetch_buffer: Optional[Tensor] = None
        self._prefetch_id: Optional[int] = None

    def load_weights(self, weight: Tensor) -> None:
        """Load weights from another tensor (e.g., from checkpoint)."""
        self.weight.copy_(weight)

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self._cache.clear()

    def _get_from_cache(self, token_id: int) -> Optional[Tensor]:
        """Get embedding from cache, updating LRU order."""
        if token_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(token_id)
            return self._cache[token_id]
        return None

    def _add_to_cache(self, token_id: int, embed: Tensor) -> None:
        """Add embedding to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Evict least recently used (first item)
            self._cache.popitem(last=False)
        self._cache[token_id] = embed

    def prefetch_token(self, token_id: int) -> None:
        """Prefetch a single token's embedding asynchronously."""
        if token_id in self._cache:
            return  # Already cached

        if self._fetch_stream is not None:
            with torch.cuda.stream(self._fetch_stream):
                embed = self.weight[token_id].to(self.gpu_device, non_blocking=True)
                self._prefetch_buffer = embed
                self._prefetch_id = token_id

    def forward_single(self, token_id: int) -> Tensor:
        """
        Get embedding for a single token (autoregressive generation).

        Args:
            token_id: Single token index

        Returns:
            (embedding_dim,) tensor
        """
        # Check prefetch buffer first
        if self._prefetch_id == token_id and self._prefetch_buffer is not None:
            if self._fetch_stream is not None:
                self._fetch_stream.synchronize()
            embed = self._prefetch_buffer
            self._prefetch_buffer = None
            self._prefetch_id = None
            self._add_to_cache(token_id, embed)
            return embed

        # Check cache
        cached = self._get_from_cache(token_id)
        if cached is not None:
            return cached

        # Fetch from CPU
        embed = self.weight[token_id].to(self.gpu_device)
        self._add_to_cache(token_id, embed)
        return embed

    def forward_batch(self, token_ids: Tensor) -> Tensor:
        """
        Get embeddings for a batch (prompt processing).

        Args:
            token_ids: (B, T) tensor

        Returns:
            (B, T, embedding_dim) tensor
        """
        B, T = token_ids.shape
        unique_ids, inverse = token_ids.flatten().unique(return_inverse=True)
        unique_ids_list = unique_ids.cpu().tolist()

        # Collect embeddings, using cache where possible
        embeds_list = []
        for tid in unique_ids_list:
            cached = self._get_from_cache(tid)
            if cached is not None:
                embeds_list.append(cached)
            else:
                embed = self.weight[tid].to(self.gpu_device)
                self._add_to_cache(tid, embed)
                embeds_list.append(embed)

        unique_embeds = torch.stack(embeds_list)  # (num_unique, embedding_dim)
        full_embeds = unique_embeds[inverse]  # (B*T, embedding_dim)

        return full_embeds.view(B, T, self.embedding_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Unified forward for both batch and single token."""
        if token_ids.numel() == 1:
            return self.forward_single(token_ids.item()).unsqueeze(0).unsqueeze(0)
        return self.forward_batch(token_ids.view(token_ids.shape[0], -1))


def create_offloaded_embedding(
    num_embeddings: int,
    num_slots: int,
    ve_dim: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
    training: bool = True,
    cache_size: int = 1024,
) -> nn.Module:
    """
    Factory function to create the appropriate MoVE embedding.

    Args:
        num_embeddings: Vocabulary size
        num_slots: Number of memory slots (M in paper)
        ve_dim: Per-slot embedding dimension
        device: Target GPU device
        dtype: Embedding dtype
        training: If True, return training embedding; if False, inference embedding
        cache_size: LRU cache size for inference

    Returns:
        MoVEOffloadedEmbedding or MoVEInferenceEmbedding
    """
    embedding_dim = num_slots * ve_dim

    if training:
        return MoVEOffloadedEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
    else:
        return MoVEInferenceEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            cache_size=cache_size,
        )
