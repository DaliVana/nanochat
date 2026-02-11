"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.move_offload import MoVEOffloadedEmbedding, MoVECPUAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of attention heads
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Mixture of Depths (MoD) configuration
    # mod_capacity=1.0 disables MoD (all tokens processed), <1.0 enables routing
    mod_capacity: float = 1.0                # Fraction of tokens processed (1.0 = disabled)
    mod_fixed_layers_start: int = 5          # First N layers always full capacity
    mod_fixed_layers_end: int = 1            # Last N layers always full capacity
    # Multi-Head Latent Attention (MLA) parameters
    mla_d_c: int = 0                # KV compression dimension (0 = auto-scale: n_embd // 4)
    mla_d_c1: int = 0               # Q compression dimension (0 = same as mla_d_c)
    mla_d_rope: int = 0             # Decoupled RoPE dimension (0 = auto-scale: head_dim // 2)
    # Mixture of Value Embeddings (MoVE) - replaces per-layer LaVE with a global shared bank
    use_move: bool = False          # Enable MoVE (False = per-layer LaVE)
    move_slots: int = 0             # Number of memory slots M (0 = auto: n_layer)
    # MoVE CPU Offloading - keeps value embeddings on CPU, fetches per batch
    move_offload_cpu: bool = False  # Enable CPU offloading for MoVE embeddings
    move_prefetch: bool = True      # Async prefetch next batch (only with offload)
    move_cache_size: int = 1024     # LRU cache size for inference (only with offload)


def get_mla_dims(config):
    """Compute MLA dimensions with automatic scaling based on model size."""
    head_dim = config.n_embd // config.n_head

    if config.mla_d_c > 0:
        d_c = config.mla_d_c
    else:
        # Auto-scale: d_c = n_embd // 4 (moderate ~4x compression)
        d_c = config.n_embd // 4

    if config.mla_d_c1 > 0:
        d_c1 = config.mla_d_c1
    else:
        # Default: same as d_c for query compression
        d_c1 = d_c

    if config.mla_d_rope > 0:
        d_rope = config.mla_d_rope
    else:
        # Auto-scale: d_rope = head_dim // 2 (half for position, half for content)
        d_rope = head_dim // 2

    # Ensure d_rope < head_dim (need room for non-positional part)
    assert d_rope < head_dim, f"d_rope ({d_rope}) must be less than head_dim ({head_dim})"

    return d_c, d_c1, d_rope


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def is_mod_layer(layer_idx: int, n_layer: int, config: 'GPTConfig') -> bool:
    """Returns True if layer uses MoD routing (alternating in middle layers).

    Protected layers (always full capacity):
    - First `mod_fixed_layers_start` layers
    - Last `mod_fixed_layers_end` layers

    MoD layers (capacity-limited):
    - Alternating layers in the middle range

    MoD is disabled when mod_capacity >= 1.0.
    """
    if config.mod_capacity >= 1.0:
        return False
    if layer_idx < config.mod_fixed_layers_start:
        return False
    if layer_idx >= n_layer - config.mod_fixed_layers_end:
        return False
    # Alternating pattern in routable range
    # routable_idx = layer_idx - config.mod_fixed_layers_start
    # return routable_idx % 2 == 0
    return True

def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary position embeddings.

    Args:
        x: (B, T, H, D) tensor to rotate
        cos, sin: (B, T, D//2) tensors - will be broadcast to match x
    """
    # Add head dim for broadcasting: (B, T, D//2) -> (B, T, 1, D//2)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    x1, x2 = x.chunk(2, dim=-1)  # split last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.

    Key differences from standard attention:
    - KV are compressed through a low-rank bottleneck (d_c dimension)
    - Q is optionally compressed through a separate bottleneck (d_c1 dimension)
    - RoPE is decoupled: position-independent part goes through compression,
      position-aware part (d_rope) is computed separately and concatenated
    - All heads share the same compressed KV latent
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.head_dim = config.n_embd // config.n_head

        # Get MLA dimensions
        d_c, d_c1, d_rope = get_mla_dims(config)
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rope = d_rope
        self.d_nope = self.head_dim - d_rope  # non-positional dim per head

        assert self.d_nope > 0, f"d_rope ({d_rope}) must be less than head_dim ({self.head_dim})"

        # Fused down-projection: project x to [c_kv, c_q, k_rope] in one matmul
        # This reduces 3 separate matmuls to 1, improving memory bandwidth utilization
        q_intermediate = self.d_c1 if self.d_c1 > 0 else self.d_c
        self.q_intermediate = q_intermediate
        self.w_down = nn.Linear(config.n_embd, self.d_c + q_intermediate + self.d_rope, bias=False)

        # Up-projections (separate because they have different input dimensions in unfused version)
        # w_ukv: d_c -> n_head * (d_nope + head_dim) for K_nope and V
        self.w_ukv = nn.Linear(self.d_c, self.n_head * (self.d_nope + self.head_dim), bias=False)
        # w_uq: q_intermediate -> n_head * head_dim for Q
        self.w_uq = nn.Linear(q_intermediate, self.n_head * self.head_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        # Value embedding gating
        self.use_move = config.use_move
        if config.use_move:
            self.move_slots = config.move_slots if config.move_slots > 0 else config.n_layer
            # MoVE: full hidden dim -> n_head * (M+1) gates
            self.ve_gate = nn.Linear(config.n_embd, self.n_head * (self.move_slots + 1), bias=False)
        else:
            self.move_slots = 0
            self.ve_gate_channels = 32
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos, sin, window_size, kv_cache=None):
        """
        MLA forward pass optimized for speed and memory:
        - Fused down-projection (3 matmuls -> 1)
        - No conditional branches
        - Minimal tensor reshaping
        """
        B, T, C = x.size()

        # 1. Fused down-projection: x -> [c_kv, c_q, k_rope_raw] in single matmul
        down = self.w_down(x)  # (B, T, d_c + q_intermediate + d_rope)
        c_kv, c_q, k_rope_raw = down.split([self.d_c, self.q_intermediate, self.d_rope], dim=-1)

        # 2. Up-project KV: c_kv -> [K_nope, V]
        kv = self.w_ukv(c_kv).view(B, T, self.n_head, self.d_nope + self.head_dim)
        k_nope, v = kv.split([self.d_nope, self.head_dim], dim=-1)

        # 3. Apply RoPE to k_rope (single head, then broadcast)
        k_rope = apply_rotary_emb(k_rope_raw.view(B, T, 1, self.d_rope), cos, sin)

        # 4. Value embedding mixing
        if self.use_move:
            # MoVE: multi-slot gating with gated standard V path
            M = self.move_slots
            ve = ve.view(B, T, M, self.n_head, self.head_dim)  # (B, T, M, H, D)
            gates = 2 * torch.sigmoid(self.ve_gate(x))  # (B, T, H*(M+1))
            gates = gates.view(B, T, self.n_head, M + 1)  # (B, T, H, M+1)
            v = gates[..., 0:1] * v + torch.einsum('bthm,btmhd->bthd', gates[..., 1:], ve)
        else:
            # LaVE: single per-layer value embedding, additive
            ve = ve.view(B, T, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x.narrow(-1, 0, self.ve_gate_channels)))
            v = v + gate.unsqueeze(-1) * ve

        # 5. Up-project Q: c_q -> Q, then split into nope/rope parts
        q = self.w_uq(c_q).view(B, T, self.n_head, self.head_dim)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        q_rope = apply_rotary_emb(q_rope, cos, sin)

        # 6. Assemble final Q and K
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_head, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 7. QK norm
        q, k = norm(q), norm(k)

        # 8. Attention
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache
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

        # 11. Re-assemble and project
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


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos, sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos, sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x

    def forward_mod(self, x, ve, cos, sin, window_size, capacity):
        """MoD forward using norm-based routing (parameter-free).

        Uses gather/scatter instead of advanced indexing for better torch.compile compatibility.
        """
        B, T, C = x.size()

        # Norm-based routing: tokens with larger L2 norm are more "important"
        routing_scores = x.norm(dim=-1)  # (B, T)

        # Top-k selection (sorted for causal order)
        _, top_indices = torch.topk(routing_scores, capacity, dim=-1)
        top_indices = torch.sort(top_indices, dim=-1)[0]  # (B, k)

        # Gather selected tokens (more compile-friendly than advanced indexing)
        indices_x = top_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, k, C)
        x_selected = torch.gather(x, 1, indices_x)  # (B, k, C)
        if ve is not None:
            indices_ve = top_indices.unsqueeze(-1).expand(-1, -1, ve.size(-1))  # (B, k, ve_dim)
            ve_selected = torch.gather(ve, 1, indices_ve)
        else:
            ve_selected = None

        # Gather cos/sin for selected positions (cos/sin are already (B, T, rope_dim))
        rope_dim = cos.size(-1)
        indices_rope = top_indices.unsqueeze(-1).expand(-1, -1, rope_dim)  # (B, k, rope_dim)
        cos_selected = torch.gather(cos, 1, indices_rope)  # (B, k, rope_dim)
        sin_selected = torch.gather(sin, 1, indices_rope)

        # Attention on selected tokens
        attn_out = self.attn(norm(x_selected), ve_selected, cos_selected, sin_selected, window_size)

        # MLP
        mlp_input = x_selected + attn_out
        mlp_out = self.mlp(norm(mlp_input))
        block_out = mlp_input + mlp_out

        # Scatter back (more compile-friendly than indexed assignment)
        output = x.scatter(1, indices_x, block_out)

        return output


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        ve_dim = config.n_head * head_dim
        self.ve_dim = ve_dim
        self.move_offload_cpu = config.use_move and config.move_offload_cpu
        if config.use_move:
            # MoVE: single global bank shared across all layers
            move_slots = config.move_slots if config.move_slots > 0 else config.n_layer
            if config.move_offload_cpu:
                # CPU offloaded MoVE: initialized later in init_weights()
                # We create a placeholder that will be replaced
                self.value_embeds = None
                self._ve_offload_params = (padded_vocab_size, move_slots * ve_dim)
            else:
                self.value_embeds = nn.Embedding(padded_vocab_size, move_slots * ve_dim)
        else:
            # LaVE: per-layer value embeddings
            self.value_embeds = nn.ModuleList([nn.Embedding(padded_vocab_size, ve_dim) for _ in range(config.n_layer)])
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        # MLA: RoPE is applied only to the d_rope portion of each head
        _, _, d_rope = get_mla_dims(config)
        rope_dim = d_rope
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rope_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            Standard attention:
                attn.c_q:        uniform, std=1/sqrt(n_embd)
                attn.c_k:        uniform, std=1/sqrt(n_embd)
                attn.c_v:        uniform, std=1/sqrt(n_embd)
                attn.c_proj:     zeros
            MLA attention:
                attn.w_dkv:      uniform, std=1/sqrt(n_embd)
                attn.w_uk:       uniform, std=1/sqrt(d_c)
                attn.w_uv:       uniform, std=1/sqrt(d_c)
                attn.w_kr:       uniform, std=1/sqrt(n_embd)
                attn.w_dq:       uniform, std=1/sqrt(n_embd) (if exists)
                attn.w_uq:       uniform, std=1/sqrt(d_c1 or n_embd)
                attn.w_qr:       uniform, std=1/sqrt(d_c1 or n_embd)
                attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

        # MLA attention initialization
        d_c, d_c1, _ = get_mla_dims(self.config)
        s_dc = 3**0.5 * d_c**-0.5
        q_intermediate = d_c1 if d_c1 > 0 else d_c
        s_q_int = 3**0.5 * q_intermediate**-0.5
        for block in self.transformer.h:
            # Fused down-projection (all from n_embd)
            torch.nn.init.uniform_(block.attn.w_down.weight, -s, s)
            # Up-projections
            torch.nn.init.uniform_(block.attn.w_ukv.weight, -s_dc, s_dc)  # from d_c
            torch.nn.init.uniform_(block.attn.w_uq.weight, -s_q_int, s_q_int)  # from q_intermediate
            # Output projection
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        if self.config.use_move:
            if self.move_offload_cpu:
                # Create offloaded embedding now that we're out of meta device context
                vocab_size, embedding_dim = self._ve_offload_params
                device = self.transformer.wte.weight.device
                self.value_embeds = MoVEOffloadedEmbedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    device=str(device) if device.type != 'meta' else 'cuda',
                    dtype=torch.bfloat16,
                )
                # Initialize weights (on CPU)
                torch.nn.init.uniform_(self.value_embeds.weight, -s, s)
            else:
                torch.nn.init.uniform_(self.value_embeds.weight, -s, s)
        else:
            for ve in self.value_embeds:
                torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        _, _, d_rope = get_mla_dims(self.config)
        rope_dim = d_rope
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rope_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            if self.config.use_move:
                if not self.move_offload_cpu:
                    # Only cast if not offloaded (offloaded is already bf16)
                    self.value_embeds.to(dtype=torch.bfloat16)
            else:
                for ve in self.value_embeds:
                    ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        # Return flat (seq_len, rope_dim) - apply_rotary_emb handles broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        if self.config.use_move:
            value_embeds_numel = self.value_embeds.weight.numel()
        else:
            value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        if self.config.use_move:
            value_embeds = self.value_embeds.weight.numel()
        else:
            value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        # When MoVE is offloaded, value_embeds is a buffer (not Parameter), so exclude from check
        params_in_model = sum(p.numel() for p in self.parameters())
        if self.move_offload_cpu:
            assert total == params_in_model + value_embeds, "Parameter count mismatch"
        else:
            assert total == params_in_model, "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Handle value embeddings based on offload setting
        if self.move_offload_cpu:
            # Offloaded: create separate CPU optimizer (returned separately)
            value_embeds_params = []  # Don't include in GPU optimizer
        else:
            value_embeds_params = list(self.value_embeds.parameters())

        # Verify parameter count (adjust for offloaded case)
        expected_params = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        actual_params = len(list(self.parameters()))
        if not self.move_offload_cpu:
            assert actual_params == expected_params, f"Parameter count mismatch: {actual_params} != {expected_params}"

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        # Only add value_embeds to GPU optimizer if not offloaded
        if not self.move_offload_cpu:
            param_groups.append(
                dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
            )

        param_groups.extend([
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ])
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

        # Create CPU optimizer for offloaded MoVE embeddings
        if self.move_offload_cpu:
            cpu_optimizer = MoVECPUAdamW(
                self.value_embeds,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            )
            return optimizer, cpu_optimizer

        return optimizer

    def prefetch_value_embeds(self, next_idx: torch.Tensor) -> None:
        """
        Prefetch value embeddings for next batch (only when MoVE CPU offload is enabled).
        Call this during current forward pass to overlap transfer with compute.

        Args:
            next_idx: (B, T) tensor of next batch's token indices
        """
        if self.move_offload_cpu and self.config.move_prefetch:
            self.value_embeds.prefetch(next_idx)

    def accumulate_ve_grad(self) -> None:
        """
        Accumulate gradients for offloaded MoVE embeddings.
        Call this after backward pass when using CPU offload.
        """
        if self.move_offload_cpu and hasattr(self, '_last_ve_global') and self._last_ve_global is not None:
            if self._last_ve_global.grad is not None:
                self.value_embeds.accumulate_grad(self._last_ve_global.grad)
            self._last_ve_global = None

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(0), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(0)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        # Slice and expand to (B, T, rope_dim) for unified interface
        cos = self.cos[T0:T0+T].unsqueeze(0).expand(B, -1, -1)
        sin = self.sin[T0:T0+T].unsqueeze(0).expand(B, -1, -1)

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual

        # MoD: enabled during training only (not with KV cache), disabled when capacity >= 1.0
        # Log-scale capacity: full capacity at short contexts, mod_capacity fraction at max context
        # This matches compute savings to where attention is expensive (long contexts)
        mod_active = self.config.mod_capacity < 1.0 and kv_cache is None
        if mod_active:
            max_T = self.config.sequence_len
            log_ratio = math.log(T) / math.log(max_T) if T > 1 else 0.0
            capacity_ratio = 1.0 - (log_ratio * (1.0 - self.config.mod_capacity))
            capacity = max(1, int(T * capacity_ratio))
        else:
            capacity = T

        # MoVE: lookup global bank once; LaVE: lookup per-layer in loop
        if self.config.use_move:
            ve_global = self.value_embeds(idx)  # (B, T, M * ve_dim) - shared across all layers
            # Register backward hook for gradient accumulation if using CPU offload
            if self.move_offload_cpu and self.training and ve_global.requires_grad:
                def ve_grad_hook(grad):
                    self.value_embeds.accumulate_grad(grad)
                    return grad
                ve_global.register_hook(ve_grad_hook)

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = ve_global if self.config.use_move else self.value_embeds[i](idx)

            if mod_active and is_mod_layer(i, self.config.n_layer, self.config):
                # MoD layer with norm-based routing
                x = block.forward_mod(x, ve, cos, sin, self.window_sizes[i], capacity)
            else:
                # Standard layer
                x = block(x, ve, cos, sin, self.window_sizes[i], kv_cache)

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
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
