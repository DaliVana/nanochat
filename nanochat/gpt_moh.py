"""
GPT model with Mixture of Heads (MoH) - Head-selective attention.

Based on nanochat/gpt.py with modifications for dynamic head selection.
Tokens are routed to use only top-k most relevant attention heads per layer.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.routing import TopKRouter, top_k_routing, compute_load_balance_loss, compute_routing_stats


@dataclass
class GPTConfigMoH:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # MoH-specific parameters
    moh_active_heads_ratio: float = 0.5  # Fraction of heads to activate per layer
    protect_first_layer: bool = False  # If True, first layer uses all heads (no routing)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttentionMoH(nn.Module):
    """
    Attention with Mixture of Heads: tokens route to a subset of attention heads.

    Key changes from base:
    - Adds head_router to select which heads to use per token
    - Uses Flash Attention (or SDPA fallback) for all heads, then masks output
    - Load balancing loss encourages even head utilization

    Performance notes:
    - Uses flash_attn for fast computation (with SDPA fallback on Mac/CPU)
    - Masking applied AFTER attention (yes, computes unused heads, but FA is fast)
    - Much faster than pre-masking Q,K,V due to optimized kernels
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.moh_active_heads_ratio = config.moh_active_heads_ratio
        self.protected = config.protect_first_layer and layer_idx == 0
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

        # MoH-specific: Head router
        self.head_router = TopKRouter(self.n_embd, self.n_head)

        # Store head statistics for logging
        self.head_stats = {}

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Protected layer: skip routing, use all heads
        if self.protected:
            head_mask = None
        else:
            # MoH: Compute head routing scores
            routing_scores = self.head_router(x)  # (B, T, n_head)
            # Select top-k heads per token
            head_mask = top_k_routing(routing_scores, k=self.moh_active_heads_ratio, return_mask=True)  # (B, T, n_head)

        # Project to Q, K, V for all heads
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply rotary embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Compute attention for all heads using Flash Attention (or SDPA fallback)
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Apply head mask to zero out non-selected heads (skip if protected)
        # This is applied AFTER attention computation (faster than pre-masking)
        if head_mask is not None:
            head_mask_expanded = head_mask.unsqueeze(-1)  # (B, T, n_head, 1)
            y = y * head_mask_expanded

        # Re-assemble and project
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, head_mask  # Return mask for load balancing

class MLP(nn.Module):
    """Same as base GPT - no changes for MoH."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class BlockMoH(nn.Module):
    """Block with MoH attention that returns head masks."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttentionMoH(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        attn_out, head_mask = self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, head_mask


class GPTMoH(nn.Module):
    """
    GPT with Mixture of Heads: tokens route to a subset of attention heads.

    Key additions:
    - head_router in each attention layer: predicts which heads are relevant for each token
    - Executes all heads but masks outputs (top-k heads contribute, others are zeroed)
    - Load balancing loss encourages even head utilization across layers
    """
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([BlockMoH(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Track routing statistics
        self.routing_stats = {}

    @torch.no_grad()
    def init_weights(self):
        """Initialize model weights (same as base GPT + head routers)."""
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # Initialize head router to near-zero (start with uniform routing)
            torch.nn.init.normal_(block.attn.head_router.router.weight, mean=0.0, std=0.01)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Estimate FLOPs per token.
        With MoH, we still compute Q,K,V for all heads, so the savings are primarily in the attention computation itself,
        which is a smaller fraction of total FLOPs. We can approximate as slightly reduced vs base.
        """
        # Base FLOPs calculation (similar to base GPT)
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        router_numel = sum(sum(p.numel() for p in block.attn.head_router.parameters()) for block in self.transformer.h)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() + router_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Attention FLOPs (scale by active heads ratio)
        active_ratio = self.config.moh_active_heads_ratio
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq * active_ratio  # Scale by active heads

        # Matmul FLOPs (not reduced - we still compute Q,K,V for all heads)
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops

        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters() if p.requires_grad)
        router = sum(sum(p.numel() for p in block.attn.head_router.parameters()) for block in self.transformer.h)
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices - router,  # Exclude router from matrices
            'router': router,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, router_lr=0.001):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Group parameters
        # Collect all non-router params from transformer blocks
        matrix_params = []
        router_params = []
        for block in self.transformer.h:
            # Attention matrices (excluding router)
            matrix_params.extend([block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.attn.c_proj.weight])
            if block.attn.ve_gate is not None:
                matrix_params.append(block.attn.ve_gate.weight)
            # MLP matrices
            matrix_params.extend([block.mlp.c_fc.weight, block.mlp.c_proj.weight])
            # Head router
            router_params.extend(block.attn.head_router.parameters())

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=router_params, lr=router_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),  # MoH routers
        ]

        # Muon groups
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Rotary embeddings
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x

        # Collect head masks for load balancing
        head_masks = []

        # Execute transformer blocks
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, head_mask = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            head_masks.append(head_mask)

        x = norm(x)

        # LM head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Compute routing statistics (aggregate across layers) - disabled for speed
        # if self.training and head_masks:
        #     # Stack all head masks and compute aggregate stats
        #     all_head_masks = torch.stack(head_masks, dim=0)  # (n_layer, B, T, n_head)
        #     avg_head_mask = all_head_masks.mean(dim=[0, 1, 2])  # Average over layers, batch, time
        #     self.routing_stats = {
        #         'mean_head_usage': avg_head_mask.mean().item(),
        #         'std_head_usage': avg_head_mask.std().item(),
        #         'min_head_usage': avg_head_mask.min().item(),
        #         'max_head_usage': avg_head_mask.max().item(),
        #     }

        if targets is not None:
            # Training: compute loss + load balancing loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Load balancing loss: encourage even head utilization across all layers
            # Reduced weight from 0.01 to 0.001 for speed (or set to 0.0 to disable)
            balance_weight = 0.001
            if balance_weight > 0:
                balance_loss = 0
                active_masks = [m for m in head_masks if m is not None]
                for head_mask in active_masks:
                    balance_loss += compute_load_balance_loss(head_mask, num_options=self.config.n_head, weight=balance_weight)
                if active_masks:
                    balance_loss = balance_loss / len(active_masks)  # Average over routed layers
                total_loss = ce_loss + balance_loss
            else:
                balance_loss = torch.tensor(0.0, device=ce_loss.device)
                total_loss = ce_loss

            # Store individual losses for logging (only if scalar)
            if loss_reduction == 'mean':
                self.routing_stats['ce_loss'] = ce_loss.item()
                self.routing_stats['balance_loss'] = balance_loss.item() if isinstance(balance_loss, torch.Tensor) else 0.0

            return total_loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Simple autoregressive generation (same as base GPT)."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
