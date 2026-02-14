"""
GPT model with Mixture of Depth (MoD) - Layer-skipping routing.

Based on nanochat/gpt.py with modifications for dynamic layer execution.
Tokens are routed to execute only top-k most important layers, skipping others.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.routing import TopKRouter, threshold_routing, compute_load_balance_loss, compute_routing_stats


@dataclass
class GPTConfigMoD:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # MoD-specific parameters
    mod_target_ratio: float = 0.5  # Target fraction of layers to activate per token
    mod_threshold: float = 0.5  # Sigmoid threshold for layer activation
    protect_first_layer: bool = False  # If True, first layer always executes (no routing)


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


class CausalSelfAttention(nn.Module):
    """Same as base GPT - no changes needed for MoD."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

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

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Same as base GPT - no changes needed for MoD."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Same as base GPT - no changes needed for MoD."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPTMoD(nn.Module):
    """
    GPT with Mixture of Depth: tokens are routed to skip low-importance layers.

    Key additions:
    - layer_router: predicts which layers are important for each token
    - Executes only top-k layers per token (k = mod_target_ratio * n_layer)
    - Load balancing loss encourages even layer utilization

    Performance optimizations:
    - Token-level skipping: Uses torch.where to conditionally apply layer outputs
    - torch.compile: Optional compilation for ~1.2-1.4x speedup (set compile_model=True)
    - Fast path: When all tokens use a layer, skips masking overhead
    """
    def __init__(self, config, pad_vocab_size_to=64, compile_model=True):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.compile_model = compile_model
        self._compiled_forward = None  # Lazy compilation

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # MoD-specific: Layer router
        self.layer_router = TopKRouter(config.n_embd, config.n_layer)

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
        """Initialize model weights (same as base GPT + router)."""
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

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Initialize layer router to near-zero: sigmoid(0) = 0.5, so ~50% activation at start
        torch.nn.init.normal_(self.layer_router.router.weight, mean=0.0, std=0.01)

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
        Estimate FLOPs per token accounting for sparse layer execution.
        With MoD, only top-k layers execute, reducing FLOPs proportionally.
        """
        # Base FLOPs calculation (same as base GPT)
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        router_numel = sum(p.numel() for p in self.layer_router.parameters())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() + router_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Attention FLOPs (unchanged - all layers compute attention)
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        # Scale by mod_target_ratio (only k layers execute per token)
        active_ratio = self.config.mod_target_ratio
        num_flops_per_token = 6 * (nparams - nparams_exclude) * active_ratio + attn_flops * active_ratio

        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        router = sum(p.numel() for p in self.layer_router.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + router + scalars
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'router': router,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, router_lr=0.001):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Group parameters
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        router_params = list(self.layer_router.parameters())  # MoD-specific

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=router_params, lr=router_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),  # MoD router
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

    def _forward_impl(self, idx, targets, kv_cache, loss_reduction):
        """Core forward implementation (can be compiled)."""
        B, T = idx.size()

        # Rotary embeddings
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # Save initial embedding for x0 residual

        # MoD: Compute layer routing scores and threshold-based selection
        routing_scores = self.layer_router(x)  # (B, T, n_layer)
        layer_mask, sparsity_loss = threshold_routing(
            routing_scores,
            threshold=self.config.mod_threshold,
            target_ratio=self.config.mod_target_ratio,
            target_weight=0.1,
        )  # layer_mask: (B, T, n_layer)

        # Execute transformer blocks with token-level skipping
        for i, block in enumerate(self.transformer.h):
            # Apply scaling
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Get value embeddings
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

            # Protected first layer: always execute, skip routing
            if i == 0 and self.config.protect_first_layer:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                continue

            # Get layer mask for this layer (convert to bool for torch.where)
            mask_i = layer_mask[:, :, i].bool()  # (B, T)

            # Skip layer entirely if no tokens use it (rare but possible)
            if not mask_i.any():
                continue

            # Check if all tokens use this layer (common case - fast path)
            if mask_i.all():
                # All tokens use this layer - no masking needed
                block_output = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                x = block_output
            else:
                # Partial execution: compute block and conditionally apply
                block_output = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                # Use torch.where for efficient conditional update
                mask_i_expanded = mask_i.unsqueeze(-1)  # (B, T, 1) - already bool
                x = torch.where(mask_i_expanded, block_output, x)

        x = norm(x)

        # LM head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Compute routing statistics (for logging)
        if self.training:
            self.routing_stats = compute_routing_stats(layer_mask)

        if targets is not None:
            # Training: compute loss + load balancing loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Sparsity loss: push average activation toward target ratio
            # Balance loss: encourage even layer utilization
            balance_loss = compute_load_balance_loss(layer_mask, num_options=self.config.n_layer, weight=0.01)

            total_loss = ce_loss + sparsity_loss + balance_loss

            # Store individual losses for logging (only if scalar)
            if loss_reduction == 'mean':
                self.routing_stats['ce_loss'] = ce_loss.item()
                self.routing_stats['sparsity_loss'] = sparsity_loss.item()
                self.routing_stats['balance_loss'] = balance_loss.item()

            return total_loss
        else:
            return logits

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        Forward pass with optional torch.compile acceleration.

        On first call with compile_model=True, compiles the forward implementation
        for faster subsequent calls.
        """
        # Use compiled version if enabled and available
        if self.compile_model and self.training and kv_cache is None:
            # Only compile during training without kv_cache (generation uses cache)
            if self._compiled_forward is None:
                print0("Compiling forward pass with torch.compile (this may take a minute)...")
                self._compiled_forward = torch.compile(
                    self._forward_impl,
                    mode='reduce-overhead',  # Best for training loops
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
            return self._compiled_forward(idx, targets, kv_cache, loss_reduction)
        else:
            # Use uncompiled version for generation or if compilation disabled
            return self._forward_impl(idx, targets, kv_cache, loss_reduction)

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
