# MoVE: Mixture of Value Embeddings

**Paper:** [arXiv:2601.22887](https://arxiv.org/abs/2601.22887)
**Authors:** Yangyan Li (Ant Group)
**Venue:** ICML 2026 submission

## Core Idea

MoVE decouples **parametric memory** from **computational cost** in Transformers. Instead of scaling model depth/width (which linearly increases FLOPs), MoVE introduces a global bank of learnable value embeddings shared across all attention layers. Each token dynamically mixes relevant embeddings from this bank into its value projection via a soft gating mechanism.

The key insight: the Value stream in attention carries semantic content (per mechanistic interpretability research), so augmenting it with a large bank of "concept vectors" lets the model store more factual/visual knowledge without deepening the reasoning engine.

## Architecture

### Value Embedding Bank
- Global tensor `E` of shape `(vocab_size, M, d)` where `M` = number of embedding slots per token
- Shared across ALL attention layers (not per-layer)
- Indexed by token IDs (sparse lookup, not dense computation)

### Routing and Gating
- Per-head router: projection `W_G` from hidden state to `H*(M+1)` logits
- Scaled sigmoid activation: `g = 2 * sigmoid(z)` mapping to range (0, 2)
  - At init (z=0), gate = 1.0 (identity/neutral)
  - Can amplify (up to 2.0) or suppress (down to 0.0)

### Value Mixing
```
V_S^(h) = g_{t,0}^(h) * V^(h) + sum_{i=1..M} g_{t,i}^(h) * M_{t,i}^(h)
```
- First term (index 0): gated standard value projection (critical for performance)
- Remaining terms: gated retrieved memory embeddings
- The mixed value is what goes into the KV cache (overhead only at generation time)

### Compute Overhead
- Only from routing projection W_G: ~1.8% overhead for typical configs (d=2048, H=16, M=32)
- Parameters accessed via sparse indexing (embedding lookup), not dense matmul

## Key Results

### Text Generation (nanochat framework, FineWeb-Edu)

| Model | Standard BPB | MoVE Best BPB | Gain |
|-------|-------------|---------------|------|
| D12 (186M) | 0.838 | 0.797 (x8) | 0.041 |
| D20 (561M) | 0.763 | 0.739 (x4) | 0.024 |
| D32 (1.88B) | 0.693 | 0.677 (x2) | 0.016 |

### MoVE vs LaVE (layer-wise value embeddings)
- MoVE x1 matches or beats LaVE x2 (half the memory slots, same or better performance)
- LaVE saturates quickly (tied to layer count); MoVE scales further (x4, x8, x32)
- Global shared memory enables "gradient highways" - all layers contribute gradients to the same parameters

### MLA Integration
- MoVE extends to Multi-Head Latent Attention by injecting memory into compressed latent space
- Preserves MLA's KV compression efficiency
- MoVE-x32 on MLA-D12: 0.0136 BPB improvement with ~312M extra params
- LaVE actually degrades on MLA when scaled (x1 -> x2), while MoVE keeps improving

### Image Generation (LlamaGen, ImageNet)
- GPT-B: FID 6.53 -> 5.62 (MoVE x1)
- GPT-L: FID 3.47 -> 3.10 (MoVE x1); LaVE actually hurt performance at this scale (3.77)

### Ablation
Two key components, both contribute:
1. **Global shared memory** (primary driver): MoVE > LaVE even when both have same gating
2. **Gated standard path (g_0)**: especially effective at higher memory densities

## Comparison with Existing Approaches

| Approach | Memory Scope | Scaling | Routing |
|----------|-------------|---------|---------|
| Persistent Memory | Per-layer, static | Tied to depth | Standard attention |
| LaVE (modded-nanoGPT) | Per-layer, dynamic | Tied to depth | Sigmoid gate |
| SVFormer/ResFormer | Shared, static | Fixed | None (shared projection) |
| MoE | Block-level | Sparse experts | Hard top-k |
| **MoVE** | **Global, dynamic** | **Independent of depth** | **Soft per-head gates** |

## Limitations
- Parameter efficiency per added param is lower than dense scaling (lots of params for modest gains)
- Memory bandwidth cost (HBM fetches) not negligible even though FLOPs are minimal
- Open question: can semantically related tokens share embedding banks for compactness?
- Not yet combined with MoE (hypothesized to be complementary)

## Relevance to nanochat

### Current Implementation
nanochat already has **layer-wise value embeddings** (LaVE), which is exactly the baseline compared against in this paper:
- `gpt.py:403` - `self.value_embeds = nn.ModuleList([nn.Embedding(padded_vocab_size, ve_dim) for _ in range(config.n_layer)])`
- Every layer has its own independent embedding table
- Gated via `ve_gate` using first 32 channels: `gate = 2 * sigmoid(ve_gate(x[:, :, :32]))`
- Added residually: `v = v + gate * ve`

The paper explicitly uses nanochat as its text generation framework, so results directly apply.

### What MoVE Would Change in nanochat

1. **Replace per-layer embeddings with a single global bank:**
   - Current: `nn.ModuleList([nn.Embedding(V, ve_dim) for _ in range(n_layer)])` (one per layer)
   - MoVE: Single `nn.Embedding(V, M * ve_dim)` shared across all layers (M = number of slots)

2. **Upgrade gating mechanism:**
   - Current: Gate uses only 32 channels, produces per-head scalar, no gate on standard V
   - MoVE: Full hidden dim -> `H*(M+1)` logits, per-head per-slot gates, gates the standard V too

3. **Memory savings at scale:**
   - Current D32: 32 separate embedding tables
   - MoVE D32 x1: 1 shared table with 16 slots (same param budget as LaVE x1, better performance)

### Concrete Opportunities

1. **Drop-in upgrade**: Since nanochat is the exact framework used in the paper, the architectural change is well-validated. MoVE x1 matches LaVE x2 performance, so switching to MoVE with M=L/2 slots should improve results with no extra params.

2. **MLA synergy**: nanochat already supports MLA (`use_mla=True`). The paper shows MoVE + MLA is particularly effective because:
   - MLA's compressed latent space benefits from external memory injection
   - LaVE actually degrades in MLA at higher scales, while MoVE keeps improving

3. **Super-dense memory regime**: MoVE enables scaling to x4, x8, x32 without adding layers - useful when compute budget (FLOPs) is fixed but memory (HBM) is available.

4. **Gate the standard V path**: Even without full MoVE, the ablation shows that adding a gate on the standard value projection (g_0) improves the existing LaVE setup, especially at x2. This is a minimal code change.

5. **Full hidden dim for gating**: The paper notes nanochat currently uses only 32 channels for gating (for efficiency), but their LaVE baseline uses full hidden dim. The comparison shows full-dim gating is better for expressivity.

### Implementation Considerations
- The global embedding bank is large: `vocab_size * M * d` parameters. For D20 with M=10 and vocab=65536, that's ~838M params (accessed sparsely).
- Training: shared bank creates "gradient highways" - faster convergence since all layers update the same params.
- Inference: value embeddings are looked up once per token and mixed into V before KV cache storage. No repeated cost during attention.
- The scaled sigmoid initialization (gates start at 1.0) is important for training stability.
