# Mixture of Depths (MoD) — Implementation Documentation

## Overview

Mixture of Depths (MoD) is a conditional computation technique that reduces FLOPs by selectively skipping tokens at certain transformer layers. Instead of processing every token through every layer, a router decides which tokens are "important" enough to be processed. Unselected tokens pass through the layer unchanged via the residual connection.

**Reference paper:** [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2402.01000) (Raposo et al., 2024)

## Configuration

MoD is configured through three fields in `GPTConfig` (`nanochat/gpt.py`):

| Parameter | Default | Description |
|---|---|---|
| `mod_capacity` | `1.0` | Fraction of tokens processed at maximum sequence length. `1.0` = MoD disabled (all tokens processed). Typical value: `0.125` (12.5%). |
| `mod_fixed_layers_start` | `5` | Number of layers at the **beginning** of the model that always run at full capacity (no routing). |
| `mod_fixed_layers_end` | `1` | Number of layers at the **end** of the model that always run at full capacity (no routing). |

### CLI Usage

Enable MoD via the `--mod-capacity` flag in `scripts/base_train.py`:

```bash
python -m scripts.base_train --depth=24 --mod-capacity=0.125 ...
```

## Architecture

### Which Layers Use MoD?

Determined by `is_mod_layer()` in `nanochat/gpt.py`:

```
Layer index:  0  1  2  3  4 │ 5  6  7  8 ... N-3  N-2 │ N-1
              ─────────────   ──────────────────────────   ───
              Fixed (start=5) │  MoD routing layers       │ Fixed (end=1)
              Full capacity   │  Capacity-limited          │ Full capacity
```

- **First `mod_fixed_layers_start` layers** (default 5): Always process all tokens. These early layers build foundational representations.
- **Last `mod_fixed_layers_end` layers** (default 1): Always process all tokens. The final layer(s) need full-sequence context for prediction.
- **Middle layers**: All use MoD routing (capacity-limited). A commented-out line suggests alternating patterns were explored but currently every middle layer is routed.

### Logarithmic Capacity Scaling

The effective capacity is **not constant** — it scales logarithmically with sequence length. Short sequences get near-full capacity; long sequences approach the configured `mod_capacity` floor. This matches the intuition that attention FLOPs are most expensive at long contexts.

Computed in `GPT.forward()`:

```python
log_ratio = math.log(T) / math.log(max_T)  # 0.0 at T=1, 1.0 at T=max_T
capacity_ratio = 1.0 - (log_ratio * (1.0 - mod_capacity))
capacity = max(1, int(T * capacity_ratio))
```

**Example** with `mod_capacity=0.125` and `max_T=2048`:

| Sequence Length (T) | log_ratio | capacity_ratio | Tokens Processed |
|---|---|---|---|
| 1 | 0.0 | 1.0 | 1 (all) |
| 64 | 0.546 | 0.522 | 33 of 64 (52%) |
| 256 | 0.727 | 0.364 | 93 of 256 (36%) |
| 1024 | 0.909 | 0.205 | 210 of 1024 (21%) |
| 2048 | 1.0 | 0.125 | 256 of 2048 (12.5%) |

### Routing Strategy: Parameter-Free Norm-Based Routing

The implementation uses **L2 norm of the residual stream** as the routing signal — no learned router parameters. Tokens with larger L2 norms are considered more "important" and are selected for processing.

Implemented in `Block.forward_mod()`:

```python
routing_scores = x.norm(dim=-1)                          # (B, T) — L2 norm per token
_, top_indices = torch.topk(routing_scores, capacity, dim=-1)  # select top-k
top_indices = torch.sort(top_indices, dim=-1)[0]               # restore causal order
```

**Key design choice:** The selected indices are **sorted** after top-k selection. This preserves causal ordering, which is essential for the attention mechanism to work correctly.

### Token Selection and Processing

The full `forward_mod` flow in `Block`:

1. **Score:** Compute L2 norm of each token's residual representation → `routing_scores (B, T)`.
2. **Select:** Pick the top-`capacity` tokens by score → `top_indices (B, k)`.
3. **Sort:** Re-sort selected indices to preserve causal (left-to-right) order.
4. **Gather:** Extract selected tokens, value embeddings, and RoPE (cos/sin) tensors using `torch.gather`.
5. **Process:** Run normal Block computation (attention + MLP) on the `(B, k, C)` subset.
6. **Scatter:** Write processed tokens back to their original positions using `x.scatter()`.

Unselected tokens are **not modified** — they retain their value from the residual stream (identity/skip connection).

```
Input x: [t0, t1, t2, t3, t4, t5, t6, t7]   (B, T, C)
                ↓ norm-based routing (capacity=3)
Selected:       [t1, t4, t6]                   (B, k, C)  — gathered
                ↓ attention + MLP
Processed:      [t1', t4', t6']                 (B, k, C)
                ↓ scatter back
Output:   [t0, t1', t2, t3, t4', t5, t6', t7]  (B, T, C)
```

### Gather/Scatter vs Advanced Indexing

The implementation deliberately uses `torch.gather` and `torch.scatter` instead of advanced indexing (`x[:, indices]`). This is noted as being more compatible with `torch.compile`.

### Training vs Inference Behavior

- **Training:** MoD is active when `mod_capacity < 1.0`. Tokens are routed per the mechanism above.
- **Inference (with KV cache):** MoD is **disabled**. All tokens are processed at every layer. This is because:
  - Autoregressive generation processes one token at a time — there is no meaningful batch of tokens to route.
  - KV cache management assumes all layers see all tokens.

```python
mod_active = self.config.mod_capacity < 1.0 and kv_cache is None
```

### Interaction with Other Components

- **RoPE (Rotary Embeddings):** When tokens are gathered, their corresponding cos/sin RoPE values are also gathered, preserving correct positional encoding for the subset.
- **Value Embeddings (ResFormer):** Value embeddings are gathered alongside input tokens, so MoD layers still benefit from value embedding injection.
- **Sliding Window Attention:** MoD layers respect the per-layer window size configuration. The `window_size` parameter is passed through to attention on the selected subset.
- **Residual Lambdas / x0 Blending:** Applied *before* the MoD routing in the main loop, so all tokens (including unselected ones) get the residual scaling and x0 blending.
- **FLOPs Estimation:** The current `estimate_flops()` method does **not** account for MoD savings — it reports theoretical FLOPs assuming all tokens are processed at every layer.

## Backward Compatibility

The checkpoint manager (`nanochat/checkpoint_manager.py`) patches missing MoD config keys in old checkpoints:

```python
if "mod_capacity" not in model_config_kwargs:
    model_config_kwargs["mod_capacity"] = 1.0        # disabled by default
if "mod_fixed_layers_start" not in model_config_kwargs:
    model_config_kwargs["mod_fixed_layers_start"] = 5
if "mod_fixed_layers_end" not in model_config_kwargs:
    model_config_kwargs["mod_fixed_layers_end"] = 1
```

This ensures pre-MoD checkpoints load correctly with MoD disabled.

## Typical Usage

From [runs/speedrun.sh](../runs/speedrun.sh) (24-layer GPU speedrun):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --mod-capacity=0.125 ...
```

From [runs/runcpu.sh](../runs/runcpu.sh) (8-layer CPU training):
```bash
python -m scripts.base_train \
    --depth=8 --mod-capacity=0.125 ...
```

Both use `0.125` (12.5%) capacity, meaning at maximum sequence length only ~12.5% of tokens are processed through MoD layers.

## Summary of Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Router type | Parameter-free (L2 norm) | Simplicity, no extra learnable parameters, no auxiliary losses needed |
| Capacity scaling | Logarithmic with seq length | Matches quadratic attention cost — saves more FLOPs where they matter most |
| Protected layers | First 5 + Last 1 | Early layers build representations; final layer needs full context for prediction |
| MoD layer pattern | Every middle layer | Simpler than alternating; commented-out alternating code suggests this was explored |
| Indexing | gather/scatter | Better `torch.compile` compatibility than advanced indexing |
| Inference | MoD disabled | Autoregressive generation is token-by-token; routing not meaningful |
