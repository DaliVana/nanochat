# MHA2MLA: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs

**Paper:** arXiv:2502.14837v2
**Authors:** Tao Ji et al. (Fudan University, ECNU, Shanghai AI Lab, Hikvision, Pengcheng Laboratory)
**Code:** https://github.com/JT-Ushio/MHA2MLA

## Summary

MHA2MLA is the first data-efficient fine-tuning method for transitioning pre-trained LLMs from Multi-Head Attention (MHA) or Grouped-Query Attention (GQA) to Multi-Head Latent Attention (MLA), the architecture used by DeepSeek. The key insight is that MLA's KV cache compression can be retrofitted onto existing models without retraining from scratch.

### Key Results
- **KV cache reduction**: Up to 92.19% reduction in KV cache size for Llama2-7B with only 1% performance drop on LongBench
- **Data efficiency**: Requires only 0.6%-1% of pretraining tokens to recover performance
- **Compatibility**: Works with both MHA and GQA architectures; can be combined with KV quantization for up to 96.87% total cache reduction

## Core Techniques

### 1. Partial-RoPE (Converting Full-RoPE to Partial-RoPE)

MLA fundamentally separates position-aware (RoPE) and position-agnostic (NoPE) components:
- **Position-aware**: Keeps RoPE on a small subset of dimensions (`d_rope`)
- **Position-agnostic**: Remaining dimensions are compressed into a latent space

The paper proposes 4 strategies for selecting which dimensions retain RoPE:

| Strategy | Description | Performance |
|----------|-------------|-------------|
| `S_high` | Retain highest-frequency subspaces (k=0, 1, ...) | Good |
| `S_low` | Retain lowest-frequency subspaces | Poor (-5.25% on 135M) |
| `S_uniform` | Uniform spacing across frequencies | Very Good |
| `S_2-norm` | Retain by attention score contribution (head-specific) | Best |

**Key finding**: High-frequency subspaces are critical; low-frequency retention causes severe degradation. The 2-norm strategy adaptively selects dimensions that contribute most to attention scores using the Cauchy-Schwarz inequality bound:

```
|<q[2k,2k+1], k[2k,2k+1]>| <= ||q[2k,2k+1]|| * ||k[2k,2k+1]||
```

The optimal configuration is `r = d_h/16` (retain 1/16 of dimensions for RoPE).

### 2. Low-Rank Approximation via Joint SVD

After identifying NoPE dimensions, the key and value projections for these dimensions are jointly compressed:

**SVD_joint** (recommended):
```
[W_k_nope, W_v] = U_kv * Sigma_kv * V_kv^T

W_dkv = U_kv[:, :d_kv] * sqrt(Sigma_kv[:d_kv, :d_kv])
W_uk = sqrt(Sigma_kv) * V_kv[:d_kv, :-d_v]
W_uv = sqrt(Sigma_kv) * V_kv[:d_kv, d_v:]
```

This preserves cross-parameter dependencies between K and V, outperforming separate SVD (SVD_split) by ~1% on average.

### 3. Inference Optimization

The MLA architecture enables matrix merging during inference:
```
q_nope * k_nope^T = x_i * (W_q_nope * W_uk^T) * c_kv^T
```

The term `(W_q_nope * W_uk^T)` can be pre-merged, and only the compressed `c_kv` is stored in the KV cache.

## Experimental Results

### KV Cache Reduction vs Performance

| Model | d_kv | Cache Reduction | CommonSense Acc | LongBench Score |
|-------|------|-----------------|-----------------|-----------------|
| Llama2-7B | 64 | -81.25% | -0.27% | - |
| Llama2-7B | 32 | -87.50% | -0.37% | -2.3% |
| Llama2-7B | 16 | -93.75% | -0.60% | - |
| Llama2-7B + Int4_HQQ | 64 | -92.19% | - | -1.0% |
| Llama2-7B + Int4_HQQ | 16 | -96.87% | - | -2.4% |

### Scaling Law Finding
Larger models experience less performance degradation during MLA transition:
- 135M: -2.24% at 18.75% cache
- 7B: -0.30% at 18.75% cache
- 13B: -0.23% at 18.75% cache

## Relation to nanochat

nanochat already implements MLA in `nanochat/gpt.py` (`MLACausalSelfAttention` class, lines 154-275). The implementation includes:

- **KV compression**: `w_dkv` (down-project), `w_uk`/`w_uv` (up-project)
- **Q compression**: Optional via `w_dq`, `w_uq`, `w_qr`
- **Decoupled RoPE**: Separate `w_kr` for positional keys
- **Configurable dimensions**: `mla_d_c`, `mla_d_c1`, `mla_d_rope`

### Key Differences from MHA2MLA Paper

1. **nanochat trains MLA from scratch**: The GPT model initializes with MLA architecture rather than converting from MHA post-training.

2. **No SVD initialization**: nanochat uses random uniform initialization for MLA weights rather than SVD-based initialization from pretrained MHA weights.

3. **Q compression is optional**: nanochat's implementation allows disabling Q compression (`d_c1 = 0`), whereas MHA2MLA focuses on KV compression.

4. **RoPE dimension selection**: nanochat uses a fixed split (first `d_nope` dims for content, last `d_rope` for position) rather than the contribution-aware selection from MHA2MLA.

### Potential Improvements for nanochat

1. **2-Norm RoPE Selection**: Instead of a fixed split, implement head-wise 2-norm analysis to identify which dimensions contribute most to attention scores. This could be done as a post-training analysis or during training.

2. **SVD-Based Checkpoint Conversion**: For users with pretrained MHA models, implement a conversion script that:
   - Computes joint SVD of `W_k` and `W_v` matrices
   - Initializes `w_dkv`, `w_uk`, `w_uv` from the SVD factors
   - Allows fine-tuning with much less data than retraining

3. **Latent KV Cache for Inference**: The current MLA inference stores full K/V in cache (line 259-267). Implementing true latent caching would:
   - Store only `c_kv` (compressed latent) and `k_rope` in cache
   - Reduce inference memory proportional to compression ratio
   - Require custom attention kernel or pre-merging W matrices

4. **Partial-RoPE Experiments**: nanochat could experiment with different `d_rope` values:
   - Current default: `head_dim // 2`
   - Paper recommends: `head_dim / 8` (for 87.5%+ compression)
   - Trade-off: lower `d_rope` = more compression but harder recovery

5. **Combined Quantization**: Since MHA2MLA shows MLA is compatible with KV quantization, nanochat could explore Int4 quantization of the latent cache for additional memory savings.

### Code Reference Points

| Component | nanochat location | MHA2MLA equivalent |
|-----------|-------------------|-------------------|
| KV compression | `gpt.py:184-186` (w_dkv, w_uk, w_uv) | W_dkv, W_uk, W_uv from joint SVD |
| Decoupled RoPE | `gpt.py:189` (w_kr) | Partial-RoPE with 2-norm selection |
| Q compression | `gpt.py:192-200` (w_dq, w_uq, w_qr) | Not focus of paper (keep original W_q) |
| RoPE application | `gpt.py:242-244` | Same, applied only to d_rope dims |
| Inference | `gpt.py:255-270` | Could use matrix merging + latent cache |

## Key Takeaways

1. **MLA can be retrofitted**: Existing MHA/GQA models can be converted to MLA with minimal fine-tuning data, not just trained from scratch.

2. **High-frequency RoPE is critical**: When selecting partial-RoPE dimensions, prioritize high-frequency subspaces or use 2-norm contribution analysis.

3. **Joint SVD preserves knowledge**: Jointly factorizing K and V preserves their interdependencies better than separate factorization.

4. **Larger models convert better**: The MHA2MLA transition shows a scaling law - larger models lose less performance at equivalent compression ratios.

5. **MLA + quantization stacks**: The two compression techniques are orthogonal and can be combined for nearly 97% KV cache reduction.
