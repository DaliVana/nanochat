"""
GPT model with Mixture of Value and Experts (MoVaE).

Based on nanochat/gpt_moe.py. Unifies MLP experts and Value Embedding (VE) experts
in a single MoE layer with a shared router. VE experts are nn.Embedding lookups
indexed by token ID, shared across all layers. Current value embeddings in attention
are removed entirely.

Uses DeepSeekV3-style sigmoid routing with auxiliary-loss-free bias balancing.
MLP expert weights stored as 3D tensors for zero-copy einsum computation.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.routing import SigmoidRouter


@dataclass
class GPTConfigMoVaE:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # MoE parameters (for MLP experts)
    moe_num_experts: int = 4  # Number of MLP expert FFNs per layer
    moe_experts_per_tok: int = 2  # Number of active experts per token (from combined pool)
    moe_expert_hidden_ratio: float = 1.0  # Expert hidden dim = ratio * (4 * n_embd)
    # MoVaE-specific parameters
    movae_num_ve_experts: int = -1  # Number of shared VE experts (-1 = n_layer // 2)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Attention without value embeddings (VE is handled in MoVaE layer)."""
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

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

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


class VEExpert(nn.Module):
    """Value Embedding expert: embedding lookup by token ID."""
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)

    def forward(self, token_ids):
        return self.embed(token_ids)


class MoVaELayer(nn.Module):
    """
    Mixture of Value and Experts layer: routes tokens to MLP experts and/or VE experts.

    MLP experts are per-layer FFNs stored as 3D weight tensors. VE experts are shared
    nn.Embedding lookups. A unified sigmoid router selects top-k from the combined pool.
    Uses auxiliary-loss-free bias balancing (DeepSeekV3).
    """
    def __init__(self, config, num_ve_experts):
        super().__init__()
        self.num_mlp_experts = config.moe_num_experts
        self.num_ve_experts = num_ve_experts
        self.num_total_experts = self.num_mlp_experts + self.num_ve_experts
        self.experts_per_tok = config.moe_experts_per_tok
        self.n_embd = config.n_embd

        # Per-layer MLP experts as 3D tensors (no stacking at forward time)
        self.hidden_dim = round(4 * config.n_embd * config.moe_expert_hidden_ratio / 128) * 128
        self.w_up = nn.Parameter(torch.empty(self.num_mlp_experts, self.hidden_dim, config.n_embd))
        self.w_down = nn.Parameter(torch.empty(self.num_mlp_experts, config.n_embd, self.hidden_dim))

        # Unified sigmoid router over all experts (MLP + VE) with bias balancing
        self.router = SigmoidRouter(config.n_embd, self.num_total_experts, self.experts_per_tok)

    def forward(self, x, token_ids, ve_experts):
        """
        x: (B, T, C) hidden states
        token_ids: (B, T) token IDs for VE expert lookup
        ve_experts: list of VEExpert modules (shared, passed from model)
        """
        B, T, C = x.size()
        N = B * T
        k = self.experts_per_tok
        E_mlp = self.num_mlp_experts
        E_ve = self.num_ve_experts
        E_total = self.num_total_experts

        # === Unified sigmoid routing with bias balancing ===
        x_flat = x.reshape(N, C)
        top_k_weights, top_k_indices, _ = self.router(x_flat)  # (N, k), (N, k), (E_total,)

        # Build sparse weight matrix: (N, E_total) with only top-k entries nonzero
        sparse_weights = torch.zeros(N, E_total, device=x.device, dtype=x.dtype)
        sparse_weights.scatter_(1, top_k_indices, top_k_weights.to(x.dtype))

        # === MLP path (einsum: compute ALL experts, weight by routing) ===
        # At E=4, einsum is faster than permute-compute-unpermute (no sorting overhead)
        hidden = torch.einsum('nc,ehc->enh', x_flat, self.w_up)        # (E_mlp, N, H)
        hidden = F.relu(hidden).square()
        mlp_out = torch.einsum('enh,ech->enc', hidden, self.w_down)    # (E_mlp, N, C)

        mlp_routing = sparse_weights[:, :E_mlp]                         # (N, E_mlp)
        mlp_contribution = torch.einsum('enc,ne->nc', mlp_out, mlp_routing)

        # === VE path (einsum: lookup ALL VE embeddings, weight by routing) ===
        token_ids_flat = token_ids.reshape(N)                            # (N,)
        ve_outputs = torch.stack([ve(token_ids_flat) for ve in ve_experts])  # (E_ve, N, C)

        ve_routing = sparse_weights[:, E_mlp:]                           # (N, E_ve)
        ve_contribution = torch.einsum('enc,ne->nc', ve_outputs, ve_routing)

        output = (mlp_contribution + ve_contribution).to(x.dtype)
        output = output.view(B, T, C)

        return output


class Block(nn.Module):
    """Standard block with MLP (used for protected first layer)."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache, token_ids, ve_experts):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        h = self.c_fc(norm(x))
        h = F.relu(h).square()
        x = x + self.c_proj(h)
        return x


class BlockMoVaE(nn.Module):
    """Block with MoVaE FFN (unified MLP + VE experts)."""
    def __init__(self, config, layer_idx, num_ve_experts):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.movae = MoVaELayer(config, num_ve_experts)

    def forward(self, x, cos_sin, window_size, kv_cache, token_ids, ve_experts):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.movae(norm(x), token_ids, ve_experts)
        return x


class GPTMoVaE(nn.Module):
    """
    GPT with Mixture of Value and Experts: MLP experts + shared VE experts
    with unified routing. Value embeddings are removed from attention.
    """
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Resolve movae_num_ve_experts default
        if config.movae_num_ve_experts == -1:
            config.movae_num_ve_experts = config.n_layer // 2

        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # Shared VE experts (model-level, referenced by all MoVaE layers)
        self.ve_experts = nn.ModuleList([
            VEExpert(padded_vocab_size, config.n_embd)
            for _ in range(config.movae_num_ve_experts)
        ])

        def make_block(layer_idx):
            return BlockMoVaE(config, layer_idx, config.movae_num_ve_experts)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([make_block(layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize model weights."""
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # Attention
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

            if hasattr(block, 'movae'):
                # MoVaE: 3D expert params
                torch.nn.init.uniform_(block.movae.w_up, -s, s)
                torch.nn.init.zeros_(block.movae.w_down)
                # Router gate
                torch.nn.init.uniform_(block.movae.router.gate.weight, -s, s)
                # Bias balancing buffers
                block.movae.router.expert_bias.zero_()
                block.movae.router.tokens_per_expert_counter.zero_()
            else:
                # Standard MLP (protected first layer)
                torch.nn.init.uniform_(block.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # VE expert embeddings
        for ve_expert in self.ve_experts:
            torch.nn.init.uniform_(ve_expert.embed.weight, -s, s)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve_expert in self.ve_experts:
                ve_expert.embed.to(dtype=torch.bfloat16)

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
        """Estimate FLOPs per token. VE experts contribute zero FLOPs (embedding lookup)."""
        attn_params = 0
        for block in self.transformer.h:
            attn_params += sum(p.numel() for p in block.attn.parameters())

        # MLP expert params scaled by expected active ratio
        effective_expert_params = 0
        for block in self.transformer.h:
            if hasattr(block, 'movae'):
                layer = block.movae
                mlp_params = layer.w_up.numel() + layer.w_down.numel()
                # Expected MLP experts selected = experts_per_tok * E_mlp / E_total
                expected_mlp_per_tok = layer.experts_per_tok * layer.num_mlp_experts / layer.num_total_experts
                effective_expert_params += mlp_params * expected_mlp_per_tok / layer.num_mlp_experts
            else:
                # Protected layer: standard MLP, always fully active
                effective_expert_params += sum(p.numel() for n, p in block.named_parameters() if 'attn' not in n)

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        num_flops_per_token = 6 * (attn_params + effective_expert_params) + attn_flops

        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        ve_experts_numel = sum(p.numel() for ve in self.ve_experts for p in ve.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        attn_params = 0
        expert_params = 0
        router_params = 0
        for block in self.transformer.h:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            if hasattr(block, 'movae'):
                expert_params += block.movae.w_up.numel() + block.movae.w_down.numel()
                router_params += sum(p.numel() for p in block.movae.router.parameters())
            else:
                expert_params += sum(p.numel() for n, p in block.named_parameters() if 'attn' not in n)

        transformer_matrices = attn_params + expert_params

        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + ve_experts_numel + lm_head + transformer_matrices + router_params + scalars

        return {
            'wte': wte,
            'value_embeds': ve_experts_numel,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'router': router_params,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = []
        for block in self.transformer.h:
            # Attention matrices
            matrix_params.extend([block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.attn.c_proj.weight])
            # MoVaE expert 3D params + router gate
            if hasattr(block, 'movae'):
                matrix_params.extend([block.movae.w_up, block.movae.w_down])
                matrix_params.append(block.movae.router.gate.weight)
            else:
                matrix_params.extend([block.c_fc.weight, block.c_proj.weight])

        ve_expert_params = [ve.embed.weight for ve in self.ve_experts]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ve_expert_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups (attention + MLP expert + router matrices, grouped by shape)
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

    def update_moe_balancing(self, coeff=1e-3):
        """Update expert routing bias for load balancing. Call before optimizer.step()."""
        for block in self.transformer.h:
            if hasattr(block, 'movae'):
                block.movae.router.update_expert_bias(coeff)

    def get_moe_stats(self):
        """Collect MoE routing statistics for logging. Call BEFORE update_moe_balancing."""
        all_counts = []
        all_biases = []
        for block in self.transformer.h:
            if hasattr(block, 'movae'):
                router = block.movae.router
                all_counts.append(router.tokens_per_expert_counter)
                all_biases.append(router.expert_bias)
        if not all_counts:
            return {}
        counts = torch.stack(all_counts).float()
        biases = torch.stack(all_biases).float()
        counts_mean = counts.mean(dim=-1).clamp(min=1)
        counts_std = counts.std(dim=-1)
        load_imbalance = (counts_std / counts_mean).mean().item()
        return {
            "moe/load_imbalance": load_imbalance,
            "moe/expert_bias_std": biases.std().item(),
            "moe/expert_bias_max": biases.abs().max().item(),
        }

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

        # Execute transformer blocks
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache, token_ids=idx, ve_experts=self.ve_experts)

        x = norm(x)

        # LM head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Simple autoregressive generation."""
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
