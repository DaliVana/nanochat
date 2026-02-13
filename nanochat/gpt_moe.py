"""
GPT model with Mixture of Experts (MoE) - Expert FFN layers.

Based on nanochat/gpt.py with modifications for expert routing in FFN.
Tokens are routed to a subset of expert FFNs, with top-k selection.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.routing import TopKRouter, compute_load_balance_loss, compute_routing_stats


@dataclass
class GPTConfigMoE:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # MoE-specific parameters
    moe_num_experts: int = 4  # Number of expert FFNs per layer
    moe_experts_per_tok: int = 2  # Number of active experts per token
    moe_expert_hidden_ratio: float = 1.0  # Expert hidden dim = ratio * (4 * n_embd). Set to 0.25 for param parity with baseline
    protect_first_layer: bool = False  # If True, first layer uses standard MLP (no expert routing)


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
    """Same as base GPT - no changes for MoE."""
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
    """Standard FFN (used for protected first layer)."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Expert(nn.Module):
    """Single FFN expert with configurable hidden dimension."""
    def __init__(self, config):
        super().__init__()
        # Scale expert capacity by moe_expert_hidden_ratio
        # Default ratio=1.0 gives 4*n_embd (same as baseline MLP)
        # ratio=0.25 gives n_embd (1/4 capacity, for param parity with baseline when using 4 experts)
        self.hidden_dim = int(4 * config.n_embd * config.moe_expert_hidden_ratio)
        self.c_fc = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer: routes tokens to top-k expert FFNs.

    Architecture:
    - N parallel expert FFNs (same as base MLP)
    - Router: selects top-k experts per token
    - Weighted combination of expert outputs
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.experts_per_tok = config.moe_experts_per_tok
        self.n_embd = config.n_embd

        # Create N expert FFNs
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

        # Router: predicts expert scores for each token
        self.router = TopKRouter(config.n_embd, self.num_experts)

        # Track expert statistics
        self.expert_stats = {}

    def forward(self, x):
        B, T, C = x.size()
        N = B * T
        k = self.experts_per_tok
        E = self.num_experts
        H = self.experts[0].hidden_dim

        # Route: compute scores and select top-k experts per token
        routing_scores = self.router(x)                                    # (B, T, E)
        routing_weights = F.softmax(routing_scores, dim=-1)                # (B, T, E)
        top_k_weights, top_k_indices = torch.topk(routing_weights, k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # === Permute → Padded Batched GEMM → Unpermute ===
        # Zero CPU syncs, zero Python loops over experts.
        # 1. Sort tokens by expert assignment
        # 2. Scatter into (E, capacity, C) padded tensor — all on GPU
        # 3. Single bmm through all experts in parallel
        # 4. Gather results back and scatter_add to original positions

        # Flatten: each token appears k times
        flat_expert_ids = top_k_indices.reshape(-1)                        # (N*k,)
        flat_weights = top_k_weights.reshape(-1)                           # (N*k,)
        flat_token_ids = torch.arange(N, device=x.device).repeat_interleave(k)

        # Sort by expert
        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Gather tokens in expert-sorted order
        x_flat = x.reshape(N, C)
        permuted_x = x_flat[sorted_token_ids]                             # (N*k, C)

        # Compute position-within-expert for each sorted token (all on GPU)
        tokens_per_expert = torch.bincount(flat_expert_ids, minlength=E)   # (E,) GPU
        offsets = torch.zeros(E, device=x.device, dtype=torch.long)
        offsets[1:] = tokens_per_expert[:-1].cumsum(0)
        pos_in_expert = torch.arange(N * k, device=x.device) - offsets[sorted_expert_ids]

        # Fixed capacity: with load balancing, experts are ~balanced.
        # 2x headroom handles any imbalance without dropping tokens.
        capacity = 2 * N * k // E

        # Scatter sorted tokens into padded (E, capacity, C) batch
        flat_batch_idx = sorted_expert_ids * capacity + pos_in_expert
        # Clamp to capacity (tokens beyond capacity are dropped — rare with 2x headroom)
        valid = pos_in_expert < capacity
        flat_batch_idx = flat_batch_idx[valid]
        permuted_x_valid = permuted_x[valid]
        sorted_weights_valid = sorted_weights[valid]
        sorted_token_ids_valid = sorted_token_ids[valid]

        batched_x = torch.zeros(E * capacity, C, device=x.device, dtype=x.dtype)
        batched_x.scatter_(0, flat_batch_idx.unsqueeze(-1).expand(-1, C), permuted_x_valid)

        # Batched GEMM: all experts in parallel, single kernel launch per matmul
        fc_w = torch.stack([e.c_fc.weight for e in self.experts])          # (E, H, C)
        proj_w = torch.stack([e.c_proj.weight for e in self.experts])      # (E, C, H)

        batched_x = batched_x.view(E, capacity, C)
        hidden = torch.bmm(batched_x, fc_w.transpose(1, 2))               # (E, capacity, H)
        hidden = F.relu(hidden).square()
        batched_out = torch.bmm(hidden, proj_w.transpose(1, 2))           # (E, capacity, C)

        # Gather results for valid positions, weight by routing scores
        permuted_output = batched_out.view(E * capacity, C)[flat_batch_idx]
        permuted_output = permuted_output * sorted_weights_valid.unsqueeze(-1)

        # Unpermute: scatter back to original token positions
        permuted_output = permuted_output.to(x.dtype)
        output = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        output.scatter_add_(0, sorted_token_ids_valid.unsqueeze(-1).expand_as(permuted_output), permuted_output)
        output = output.view(B, T, C)

        # Expert statistics
        if self.training:
            self.expert_stats = {
                'mean_utilization': (tokens_per_expert > 0).float().mean(),
            }

        return output, routing_weights


class Block(nn.Module):
    """Standard block with MLP (used for protected first layer)."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x, None  # None routing_weights for interface compatibility


class BlockMoE(nn.Module):
    """Block with MoE FFN that returns routing weights."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.moe = MoELayer(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        moe_out, routing_weights = self.moe(norm(x))
        x = x + moe_out
        return x, routing_weights


class GPTMoE(nn.Module):
    """
    GPT with Mixture of Experts: FFN layers replaced with expert routing.

    Key additions:
    - MoELayer replaces MLP in each block
    - Router predicts expert scores for each token
    - Top-k expert selection (default: 2 out of 4 experts)
    - Load balancing loss encourages even expert utilization
    """
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        def make_block(layer_idx):
            if config.protect_first_layer and layer_idx == 0:
                return Block(config, layer_idx)
            return BlockMoE(config, layer_idx)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([make_block(layer_idx) for layer_idx in range(config.n_layer)]),
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
        """Initialize model weights (same as base GPT + experts + routers)."""
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

            # MLP or MoE experts
            if hasattr(block, 'moe'):
                for expert in block.moe.experts:
                    torch.nn.init.uniform_(expert.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(expert.c_proj.weight)
                # Router (near-zero init for uniform routing at start)
                torch.nn.init.normal_(block.moe.router.router.weight, mean=0.0, std=0.01)
            else:
                # Standard MLP (protected first layer)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

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
        Estimate FLOPs per token accounting for MoE.
        With MoE, we have more parameters but only k/n experts are active per token.
        """
        # Count parameters
        # Attention params (unchanged)
        attn_params = 0
        for block in self.transformer.h:
            attn_params += sum(p.numel() for n, p in block.attn.named_parameters() if 've_gate' not in n)

        # Expert params (multiply by active ratio) + protected layer MLP params
        active_expert_ratio = self.config.moe_experts_per_tok / self.config.moe_num_experts
        effective_expert_params = 0
        router_numel = 0
        for block in self.transformer.h:
            if hasattr(block, 'moe'):
                layer_expert_params = sum(p.numel() for p in block.moe.experts.parameters())
                effective_expert_params += layer_expert_params * active_expert_ratio
                router_numel += sum(p.numel() for p in block.moe.router.parameters())
            else:
                # Protected layer: standard MLP, always fully active
                effective_expert_params += sum(p.numel() for p in block.mlp.parameters())

        # Embedding params (exclude from matmul FLOPs)
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() + router_numel)

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Attention FLOPs
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
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        # Count attention and expert params
        attn_params = 0
        expert_params = 0
        router_params = 0
        for block in self.transformer.h:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            if hasattr(block, 'moe'):
                expert_params += sum(p.numel() for p in block.moe.experts.parameters())
                router_params += sum(p.numel() for p in block.moe.router.parameters())
            else:
                expert_params += sum(p.numel() for p in block.mlp.parameters())

        # Combine attention and expert params for consistency with other models
        transformer_matrices = attn_params + expert_params

        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + router_params + scalars

        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'router': router_params,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, router_lr=0.001):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Group parameters
        # Vectorized einsum implementation gives gradients to all experts every step,
        # so Muon works for expert params (no sparse gradient issue)
        matrix_params = []
        router_params = []
        for block in self.transformer.h:
            # Attention matrices
            matrix_params.extend([block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.attn.c_proj.weight])
            if block.attn.ve_gate is not None:
                matrix_params.append(block.attn.ve_gate.weight)
            # Expert matrices or standard MLP (protected first layer)
            if hasattr(block, 'moe'):
                for expert in block.moe.experts:
                    matrix_params.extend([expert.c_fc.weight, expert.c_proj.weight])
                router_params.extend(block.moe.router.parameters())
            else:
                matrix_params.extend([block.mlp.c_fc.weight, block.mlp.c_proj.weight])

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
            dict(kind='adamw', params=router_params, lr=router_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups (attention + expert matrices, grouped by shape)
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

        # Collect routing weights for load balancing
        routing_weights_list = []

        # Execute transformer blocks
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, routing_weights = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            routing_weights_list.append(routing_weights)

        x = norm(x)

        # LM head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Compute expert statistics (aggregate across layers)
        if self.training:
            all_expert_utils = [block.moe.expert_stats['mean_utilization'] for block in self.transformer.h if hasattr(block, 'moe') and block.moe.expert_stats]
            if all_expert_utils:
                mean_util = sum(all_expert_utils) / len(all_expert_utils)
                self.routing_stats = {
                    'mean_expert_utilization': mean_util.item() if isinstance(mean_util, torch.Tensor) else mean_util,
                }

        if targets is not None:
            # Training: compute loss + load balancing loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Load balancing loss: encourage even expert utilization (skip protected layers)
            balance_loss = 0
            active_routing = [rw for rw in routing_weights_list if rw is not None]
            for routing_weights in active_routing:
                balance_loss += compute_load_balance_loss(routing_weights, num_options=self.config.moe_num_experts, weight=0.01)
            if active_routing:
                balance_loss = balance_loss / len(active_routing)

            total_loss = ce_loss + balance_loss

            # Store individual losses for logging (only if scalar)
            if loss_reduction == 'mean':
                self.routing_stats['ce_loss'] = ce_loss.item()
                self.routing_stats['balance_loss'] = balance_loss.item()

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
