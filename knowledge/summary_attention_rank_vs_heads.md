# Quality over Quantity in Attention Layers: When Adding More Heads Hurts

**Authors**: Noah Amsel, Gilad Yehudai, Joan Bruna (NYU / Flatiron Institute)
**Venue**: ICLR 2025
**Source**: https://proceedings.iclr.cc/paper_files/paper/2025/file/9c537882044c8b5352c363e840872ddb-Paper-Conference.pdf

## Core Thesis

The nearly universal practice of setting `H = d/r` (number of heads = embedding dimension / rank per head) in transformers is not theoretically justified and can **hurt** performance. The paper proves that **rank** (the dimension of Q/K projections per head) is the primary factor determining an attention layer's representational capacity — not the total number of parameters or number of heads. Increasing heads while decreasing rank (keeping total params constant) can severely degrade accuracy.

## Key Results

### 1. Full-Rank Attention is Fundamentally More Expressive (Theorem 2)
- **Target function**: Nearest-neighbor search on the unit sphere — a natural, simple function analogous to semantic retrieval.
- A **single full-rank attention head** (rank r = d) can represent this function exactly with O(d²) parameters.
- **Low-rank attention** (rank r << d) cannot approximate it unless the number of heads H grows exponentially:
  - High-accuracy regime: H must be at least exp(Ω(d − r))
  - High-dimensional regime: H must be at least poly(d/r)

### 2. Exponential Separation (Theorem 3)
- A combined target using polynomially many biased nearest-neighbor functions can be approximated by poly-many full-rank heads, but requires **exp(Ω(d − r))** rank-r heads.
- This is analogous to the classical depth-width separation in neural networks (2-layer vs 3-layer).

### 3. Depth Can Partially Compensate (Section 6)
- Two-layer transformers with rank-1 heads and polynomially many heads can approximate the nearest-neighbor target for **short sequences** (N=2), via a "majority voting" strategy.
- But this approach introduces unfavorable dependence on sequence length N.
- **Conjecture 6**: For arbitrary N, no fixed-size low-rank multi-layer transformer can approximate the target for all sequence lengths — full-rank attention is necessary regardless of depth.

### 4. Experimental Validation (Section 7)
- Standard multi-layer transformer encoders (with MLPs, skip connections, normalization) trained on nearest-neighbor and in-context learning tasks.
- **Full-rank models (H=1, r=d) consistently outperform models with H=d/r heads**, even with fewer total parameters.
- With L=1 layer: full-rank is necessary and sufficient; even d²/2 parameters at rank d/2 fail.
- With L>1 layers: trade-off is more favorable, but full-rank still significantly outperforms.
- For the Garg et al. (2022) in-context linear regression task: simply reducing H while maintaining H=d/r scaling yields significantly better accuracy without changing parameter count.

## Why This Matters

The standard scaling `H = d/r` is deeply embedded in transformer practice — it's even hard-coded into PyTorch and xFormers libraries. The original motivation (Vaswani et al. 2017) was simply to match parameter count with the single-head case, not because it was optimal. This paper shows it can be actively harmful.

## Connection to nanochat

### Current nanochat Configuration

In `scripts/base_train.py`, nanochat sets:
```python
num_heads = model_dim // args.head_dim  # head_dim defaults to 128
num_kv_heads = num_heads  # 1:1 GQA ratio (no grouping)
```

For the speedrun config (depth=28, aspect_ratio=64):
- `model_dim = 28 * 64 = 1792`, nudged to nearest multiple of 128 → 1792
- `num_heads = 1792 / 128 = 14`
- `head_dim = 128` → rank r = 128

This means nanochat uses the standard `H = d/r` scaling that the paper questions. With d=1792 and r=128, we have H=14 heads.

### What We Might Try

1. **Fewer heads, higher rank per head**: The paper's clearest recommendation. For example, with d=1792:
   - Current: H=14, r=128 → 14 × 128 = 1792 Q/K params per layer
   - Alternative: H=7, r=256 → 7 × 256 = 1792 Q/K params per layer (same total)
   - Alternative: H=4, r=448 → 4 × 448 = 1792 Q/K params per layer
   - Or even H=1, r=1792 (full-rank, single head)

2. **Asymmetric Q/K rank vs number of heads**: The paper focuses on rank of Q/K matrices. We could increase Q/K rank while keeping the V rank the same (since the V output rank is tied to the output projection).

3. **Relevance to MLA**: nanochat's MLA (Multi-Head Latent Attention) compresses KV through a low-rank bottleneck (`d_c = n_embd // 3`). The paper's warnings about low-rank attention are **directly relevant** — MLA's compression may harm the representational capacity of the attention layer. The paper suggests that the compression ratio in MLA should be carefully tuned, and that more aggressive compression (lower d_c) could have outsized negative effects.

4. **Relevance to GQA**: Group-Query Attention effectively reduces the rank of the K/V projections by sharing them across head groups. The paper suggests that GQA's parameter savings may come at a representational cost, especially for tasks requiring nearest-neighbor-like retrieval (which is core to what attention does).

5. **Practical experiment**: A simple experiment would be to run the speedrun with `--head-dim=256` (halving the number of heads to 7 while doubling rank per head) and compare eval metrics. The paper predicts this should improve or at least maintain quality while the total parameter count stays roughly the same.

### Caveats

- The paper's theoretical analysis is primarily about **shallow** transformers. Depth can partially compensate for low rank.
- nanochat uses 28 layers, so depth-based compensation may mitigate the low-rank issues.
- The target functions studied (nearest-neighbor) are specific; real language modeling may not require the same expressive capacity from individual attention layers.
- QK normalization (which nanochat uses) may interact with the rank in ways the paper doesn't study.
- The paper studies encoders, not autoregressive decoders with causal masking.

### Key Takeaway

When exploring the head count / rank trade-off in nanochat, consider that the current `H = d/r` default is not sacred. The paper provides strong theoretical and empirical evidence that **fewer heads with higher rank can be strictly better** for the same parameter budget. This is a cheap hyperparameter experiment that could yield free quality gains.
