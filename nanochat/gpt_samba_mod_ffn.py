"""
Samba + MoD-FFN: Mamba-2 + Sliding Window Attention with per-layer MLP routing.

Based on gpt_samba.py. Attention and Mamba layers always run on all tokens.
MLP is conditionally executed per-token via a per-layer router (decided from the
current hidden state after attn/mamba). Gather/scatter on selected tokens gives
real tok/sec improvement since MLP is token-independent.

The router is a cheap dot-product (n_embd -> 1) per layer. Sigmoid + hard threshold
with straight-through estimator for gradients. Sparsity loss pushes mean activation
toward target_ratio. Balance loss encourages even MLP usage across layers.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.mamba2 import Mamba2Layer
from nanochat.routing import compute_load_balance_loss, compute_routing_stats

from nanochat.fused_rope import apply_rotary_emb_triton


@dataclass
class GPTConfigSambaMoDFFN:
    sequence_len: int = 32768
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6       # number of query heads (attention layers)
    n_kv_head: int = 6    # number of key/value heads (attention layers, GQA)
    n_embd: int = 768
    # Sliding window size for all attention layers (Mamba handles long-range)
    sliding_window: int = 1024
    # Layer pattern: M=Mamba-2, A=Attention, tiled across layers
    layer_pattern: str = "MA"
    # Mamba-2 parameters
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 256
    # MoD-FFN parameters
    mod_ffn_target_ratio: float = 0.5   # target fraction of tokens that run MLP
    mod_ffn_threshold: float = 0.5      # sigmoid threshold for MLP activation


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return False #layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@torch.compiler.disable
def _flash_attn_kvcache_eager(q, k_cache, v_cache, k=None, v=None,
                               cache_seqlens=None, window_size=(-1, -1)):
    """Wrapper to prevent torch.compile from tracing into FA3's KV cache op."""
    return flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
        causal=True, window_size=window_size,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.sliding_window = config.sliding_window
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        if kv_cache is None:
            cos, sin = cos_sin
            apply_rotary_emb_triton(q, k, cos, sin)
            q, k = norm(q), norm(k)
            y = flash_attn.flash_attn_func(q, k, v, causal=True,
                                           window_size=(self.sliding_window, 0))
        else:
            cos, sin = cos_sin
            apply_rotary_emb_triton(q, k, cos, sin)
            q, k = norm(q), norm(k)

            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = _flash_attn_kvcache_eager(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                window_size=(self.sliding_window, 0),
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class AttentionBlock(nn.Module):
    """Transformer block: CausalSelfAttention + MLP, pre-norm residual."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class MambaBlock(nn.Module):
    """Mamba-2 block: Mamba2Layer + MLP, pre-norm residual."""
    def __init__(self, config):
        super().__init__()
        self.mamba = Mamba2Layer(
            d_model=config.n_embd,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
        )
        self.mlp = MLP(config)

    def forward(self, x):
        mamba_out, _, _ = self.mamba(norm(x))
        x = x + mamba_out
        x = x + self.mlp(norm(x))
        return x


def _compute_layer_types(config):
    """Compute per-layer type from layer_pattern string. Returns list of 'M' or 'A'."""
    pattern = config.layer_pattern.upper()
    assert all(c in "MA" for c in pattern), f"Invalid layer_pattern: {pattern}. Use only M and A."
    assert any(c == 'A' for c in pattern), "layer_pattern must contain at least one A (attention) layer"
    return [pattern[i % len(pattern)] for i in range(config.n_layer)]


@torch.compiler.disable
def _gated_mlp(mlp, x, mask_i, training):
    """
    Run MLP only on tokens selected by mask_i, using gather/scatter.

    Args:
        mlp: MLP module
        x: (B, T, C) input tensor
        mask_i: (B, T) float mask with STE gradients (~0 or ~1)
        training: bool
    Returns:
        x with MLP residual applied only to selected tokens
    """
    B, T, C = x.shape
    mask_bool = (mask_i > 0.5)

    # Inference fast paths (T=1 during generation)
    if not training:
        if mask_bool.all():
            return x + mlp(norm(x))
        if not mask_bool.any():
            return x

    # Gather active tokens
    flat_mask = mask_bool.view(-1)
    indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # (K,)

    if indices.numel() == 0:
        return x

    x_selected = x.view(-1, C)[indices]       # (K, C) — K ≈ target_ratio * B*T
    mlp_out = mlp(norm(x_selected))            # MLP on fewer tokens → faster

    # Scatter back
    out_flat = torch.zeros(B * T, C, device=x.device, dtype=x.dtype)
    out_flat[indices] = mlp_out
    # Multiply by STE mask for gradient flow to the router
    x = x + out_flat.view(B, T, C) * mask_i.unsqueeze(-1)
    return x


class GPTSambaMoDFFN(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: this __init__ runs in meta device context.
        All data initialization happens in init_weights().
        """
        super().__init__()
        self.config = config

        # Determine layer types
        self.layer_types = _compute_layer_types(config)

        # Pad vocab
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # Build heterogeneous block list
        blocks = []
        for layer_idx in range(config.n_layer):
            if self.layer_types[layer_idx] == 'A':
                blocks.append(AttentionBlock(config, layer_idx))
            else:
                blocks.append(MambaBlock(config))

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Value embeddings: only on attention layers that pass has_ve()
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if self.layer_types[i] == 'A' and has_ve(i, config.n_layer)
        })

        # Per-layer MLP routers: cheap dot-product from current hidden state
        self.mlp_routers = nn.ModuleList([
            nn.Linear(config.n_embd, 1, bias=False) for _ in range(config.n_layer)
        ])

        # Rotary embeddings (only used by attention layers, but precompute for all positions)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Gradient checkpointing: recompute forward during backward to save memory
        self.gradient_checkpointing = False

        # Routing statistics for logging
        self.routing_stats = {}

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights."""
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        for i, block in enumerate(self.transformer.h):
            if self.layer_types[i] == 'A':
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                mamba = block.mamba
                torch.nn.init.uniform_(mamba.in_proj.weight, -s, s)
                torch.nn.init.kaiming_uniform_(mamba.conv1d.weight, a=math.sqrt(5))
                mamba.A_log.copy_(torch.log(torch.linspace(0.001, mamba.n_heads, mamba.n_heads, dtype=torch.float32, device=mamba.A_log.device)))
                torch.nn.init.uniform_(mamba.dt_bias, -4.0, -2.0)
                torch.nn.init.zeros_(mamba.out_proj.weight)
                mamba._causal_mask.copy_(torch.triu(torch.ones(mamba.chunk_size, mamba.chunk_size, dtype=torch.bool, device=mamba._causal_mask.device), diagonal=1))
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE gates
        for i, block in enumerate(self.transformer.h):
            if self.layer_types[i] == 'A' and block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # MLP routers: near-zero so sigmoid(~0) ≈ 0.5, ~50% activation at start
        for router in self.mlp_routers:
            torch.nn.init.normal_(router.weight, mean=0.0, std=0.01)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # bf16 embeddings on CUDA
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

    def get_device(self):
        return self.transformer.wte.weight.device

    def _full_layer_forward(self, i, block, x, x0, idx, cos_sin, kv_cache):
        """Single layer forward: scaling + attn/mamba + routed MLP. Used for gradient checkpointing."""
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

        # Attn or Mamba sub-layer (always on all tokens)
        if self.layer_types[i] == 'A':
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = x + block.attn(norm(x), ve, cos_sin, kv_cache)
        else:
            mamba_out, _, _ = block.mamba(norm(x))
            x = x + mamba_out

        # Per-layer MLP routing from current hidden state
        score = self.mlp_routers[i](x).squeeze(-1)  # (B, T)
        prob = torch.sigmoid(score)
        mask_hard = (prob >= self.config.mod_ffn_threshold).float()
        mask_i = mask_hard + prob - prob.detach()  # STE
        sparsity_loss_i = 0.1 * (prob.mean() - self.config.mod_ffn_target_ratio) ** 2

        # Gated MLP with gather/scatter
        x = _gated_mlp(block.mlp, x, mask_i, self.training)

        return x, mask_i, sparsity_loss_i

    def estimate_flops(self):
        """
        Return estimated FLOPs per token (forward + backward).
        MLP FLOPs scaled by mod_ffn_target_ratio, everything else full.
        """
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        router_numel = sum(p.numel() for p in self.mlp_routers.parameters())

        mamba_scalar_numel = 0
        for name, p in self.named_parameters():
            if 'A_log' in name or 'dt_bias' in name:
                mamba_scalar_numel += p.numel()

        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          mamba_scalar_numel + router_numel)

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Attention FLOPs (sliding window — full, not reduced by MoD)
        W = self.config.sliding_window
        n_attn = sum(1 for lt in self.layer_types if lt == 'A')
        effective_seq = min(W, t)
        attn_flops = n_attn * 12 * h * q * effective_seq

        # SSD FLOPs for Mamba layers (full, not reduced by MoD)
        d_inner = int(self.config.mamba_expand * self.config.n_embd)
        d_state = self.config.mamba_d_state
        ngroups = self.config.mamba_ngroups
        chunk_size = self.config.mamba_chunk_size
        n_mamba = sum(1 for lt in self.layer_types if lt == 'M')
        ssd_flops = n_mamba * (
            6 * chunk_size * ngroups * d_state
            + 6 * chunk_size * d_inner
            + 12 * d_inner * d_state
        )

        # Split matmul params into MLP vs non-MLP
        n_embd = self.config.n_embd
        mlp_params_per_layer = n_embd * (4 * n_embd) + (4 * n_embd) * n_embd
        total_mlp_params = self.config.n_layer * mlp_params_per_layer
        non_mlp_matmul_params = (nparams - nparams_exclude) - total_mlp_params

        # Non-MLP matmuls at full rate, MLP at target_ratio
        active_ratio = self.config.mod_ffn_target_ratio
        num_flops_per_token = (6 * non_mlp_matmul_params +
                              6 * total_mlp_params * active_ratio +
                              attn_flops + ssd_flops)
        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        router = sum(p.numel() for p in self.mlp_routers.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + router + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"

        # Per-layer-type breakdown
        mamba_params = 0
        attn_params = 0
        for i, block in enumerate(self.transformer.h):
            block_params = sum(p.numel() for p in block.parameters())
            if self.layer_types[i] == 'M':
                mamba_params += block_params
            else:
                attn_params += block_params

        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'attn_blocks': attn_params,
            'mamba_blocks': mamba_params,
            'router': router,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate Mamba 1D params (A_log, dt_bias) and conv1d from matrix params
        mamba_scalar_params = []
        mamba_conv_params = []
        matrix_params = []
        for name, p in self.transformer.h.named_parameters():
            if 'A_log' in name or 'dt_bias' in name:
                mamba_scalar_params.append(p)
            elif 'conv1d' in name:
                mamba_conv_params.append(p)
            else:
                matrix_params.append(p)

        router_params = list(self.mlp_routers.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Verify all params accounted for
        all_params = len(list(self.parameters()))
        grouped = (len(matrix_params) + len(mamba_scalar_params) + len(mamba_conv_params) +
                   len(router_params) + len(embedding_params) +
                   len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        assert all_params == grouped, f"Parameter count mismatch: {all_params} != {grouped}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mamba_scalar_params, lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mamba_conv_params, lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=router_params, lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
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

    def get_mod_stats(self):
        """Collect routing statistics for logging."""
        return self.routing_stats

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

        if kv_cache is None:
            cos_sin = self.cos[:, :T], self.sin[:, :T]
        else:
            T0 = kv_cache.get_pos()
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x

        total_sparsity_loss = 0.0
        all_masks = []

        for i, block in enumerate(self.transformer.h):
            if self.gradient_checkpointing and self.training:
                x, mask_i, sparsity_loss_i = checkpoint(
                    self._full_layer_forward, i, block, x, x0, idx, cos_sin, kv_cache,
                    use_reentrant=False,
                )
            else:
                x, mask_i, sparsity_loss_i = self._full_layer_forward(
                    i, block, x, x0, idx, cos_sin, kv_cache,
                )
            total_sparsity_loss = total_sparsity_loss + sparsity_loss_i
            all_masks.append(mask_i.detach())

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Balance loss: encourage even MLP usage across layers
            mlp_mask = torch.stack(all_masks, dim=-1)  # (B, T, n_layer)
            balance_loss = compute_load_balance_loss(mlp_mask, num_options=self.config.n_layer, weight=0.01)

            total_loss = ce_loss + total_sparsity_loss + balance_loss

            if self.training:
                self.routing_stats = compute_routing_stats(mlp_mask)
                if loss_reduction == 'mean':
                    self.routing_stats['ce_loss'] = ce_loss.item()
                    self.routing_stats['sparsity_loss'] = total_sparsity_loss.item() if isinstance(total_sparsity_loss, torch.Tensor) else total_sparsity_loss
                    self.routing_stats['balance_loss'] = balance_loss.item()

            return total_loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive streaming inference (no cache, full recompute)."""
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
