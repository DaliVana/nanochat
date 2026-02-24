"""
Samba model variant: Mamba-2 (SSD) + Sliding Window Attention.

Inspired by Microsoft's Samba architecture, this variant interleaves Mamba-2 layers
with sliding window attention layers. Mamba layers handle long-range dependencies via
compressed recurrent state; attention layers (with RoPE + sliding window) handle precise
local retrieval.

Layer composition is configurable via `layer_pattern` (e.g. "MA" = alternating,
"MMA" = two Mamba per one Attention).

Notable features (inherited from base gpt.py):
- rotary embeddings on attention layers
- QK norm on attention layers
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- no learnable params in rmsnorm, no bias in linear layers
- GQA support, Flash Attention 3 integration
- sliding window attention on attention layers
- value embeddings (ResFormer-style) on attention layers
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.mamba2 import Mamba2Layer
from nanochat.fused_ce import fused_cross_entropy
from nanochat.fused_rope import apply_rotary_emb_triton


@dataclass
class GPTConfigSamba:
    sequence_len: int = 32768
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6       # number of query heads (attention layers)
    n_kv_head: int = 6    # number of key/value heads (attention layers, GQA)
    n_embd: int = 768
    # Sliding window size for all attention layers (Mamba handles long-range)
    sliding_window: int = 1024
    # Layer pattern: M=Mamba-2, A=Attention, tiled across layers
    # Examples: "MA"=alternating, "MMA"=two Mamba per one Attn, "MMMA"=three Mamba per one Attn
    layer_pattern: str = "MA"
    # Mamba-2 parameters
    mamba_d_state: int = 64     # SSM state dimension
    mamba_d_conv: int = 4       # causal convolution kernel size
    mamba_expand: int = 2       # expansion factor for inner dim
    mamba_ngroups: int = 1      # B/C group count (1=max sharing, n_heads=per-head like original)
    mamba_chunk_size: int = 256 # chunk size for SSD algorithm


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
            # Training: sliding window attention with global RoPE
            cos, sin = cos_sin
            # Fused Triton kernel applies this in-place
            apply_rotary_emb_triton(q, k, cos, sin)
            q, k = norm(q), norm(k)
            y = flash_attn.flash_attn_func(q, k, v, causal=True,
                                           window_size=(self.sliding_window, 0))
        else:
            # Inference: use KV cache with sliding window
            cos, sin = cos_sin  # global positions for inference
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


class GPTSamba(nn.Module):
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

        # Rotary embeddings (only used by attention layers, but precompute for all positions)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Gradient checkpointing: recompute forward during backward to save memory
        self.gradient_checkpointing = False

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
                # Attention block: same as base gpt.py
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                # Mamba block
                mamba = block.mamba
                torch.nn.init.uniform_(mamba.in_proj.weight, -s, s)
                torch.nn.init.kaiming_uniform_(mamba.conv1d.weight, a=math.sqrt(5))
                # A_log: log(1..n_heads) for diverse timescales
                mamba.A_log.copy_(torch.log(torch.linspace(0.001, mamba.n_heads, mamba.n_heads, dtype=torch.float32, device=mamba.A_log.device)))
                # dt_bias: softplus(-4..-2) gives ~0.02-0.13
                torch.nn.init.uniform_(mamba.dt_bias, -4.0, -2.0)
                torch.nn.init.zeros_(mamba.out_proj.weight)
                # Re-initialize causal mask buffer (was garbage after to_empty from meta device)
                mamba._causal_mask.copy_(torch.triu(torch.ones(mamba.chunk_size, mamba.chunk_size, dtype=torch.bool, device=mamba._causal_mask.device), diagonal=1))
                # MLP in Mamba block: same as base
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

    def estimate_flops(self):
        """
        Return estimated FLOPs per token (forward + backward).
        Attention layers: 6 * matmul_params + 12*h*q*effective_seq
        Mamba layers: 6 * matmul_params + 12 * d_inner * d_state (SSD state-space ops)
        """
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())

        # Exclude non-matmul params
        mamba_scalar_numel = 0
        for name, p in self.named_parameters():
            if 'A_log' in name or 'dt_bias' in name:
                mamba_scalar_numel += p.numel()

        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          mamba_scalar_numel)

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Attention FLOPs (sliding window — each attention layer sees at most W tokens)
        W = self.config.sliding_window
        n_attn = sum(1 for lt in self.layer_types if lt == 'A')
        effective_seq = min(W, t)
        attn_flops = n_attn * 12 * h * q * effective_seq

        # SSD FLOPs for Mamba layers
        d_inner = int(self.config.mamba_expand * self.config.n_embd)
        d_state = self.config.mamba_d_state
        n_mamba = sum(1 for lt in self.layer_types if lt == 'M')
        ssd_flops = n_mamba * 12 * d_inner * d_state

        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops + ssd_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
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
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate Mamba 1D params (A_log, dt_bias) from matrix params
        mamba_scalar_params = []
        matrix_params = []
        for name, p in self.transformer.h.named_parameters():
            if 'A_log' in name or 'dt_bias' in name:
                mamba_scalar_params.append(p)
            else:
                matrix_params.append(p)

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Verify all params accounted for
        all_params = len(list(self.parameters()))
        grouped = (len(matrix_params) + len(mamba_scalar_params) + len(embedding_params) +
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

        if kv_cache is None:
            # Training: global RoPE positions (0..T-1) for sliding window attention
            cos_sin = self.cos[:, :T], self.sin[:, :T]
        else:
            # Inference: global RoPE positions for KV cache
            T0 = kv_cache.get_pos()
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.layer_types[i] == 'A':
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, ve, cos_sin, kv_cache, use_reentrant=False)
                else:
                    x = block(x, ve, cos_sin, kv_cache)
            else:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        x = norm(x)

        if targets is not None and self.training:
            # Fused Cross Entropy avoids materializing the massive [B, T, V] tensor
            return fused_cross_entropy(x, self.lm_head.weight, targets, softcap=15.0, ignore_index=-1, reduction=loss_reduction)

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
