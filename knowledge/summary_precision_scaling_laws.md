# Scaling Laws for Precision

**Paper:** Kumar et al., "Scaling Laws for Precision" (arXiv:2411.04330, Nov 2024)
**Authors:** Tanishq Kumar (Harvard), Zachary Ankner (MIT/Databricks), Benjamin F. Spector (Stanford), Blake Bordelon (Harvard), Niklas Muennighoff (Stanford), Mansheej Paul (Databricks), Cengiz Pehlevan (Harvard), Christopher Re (Stanford), Aditi Raghunathan (CMU)

## Overview

This paper develops "precision-aware" scaling laws that extend Chinchilla scaling to account for the precision (bit-width) used during training and inference. The key insight is that training in lower precision reduces a model's **effective parameter count** (`N_eff`), and this relationship is predictable.

## Key Findings

### 1. Post-Training Quantization (PTQ) Degrades More with Overtraining

**Critical Finding:** Overtrained models (high D/N ratio) are MORE sensitive to post-training quantization.

The degradation from PTQ follows:
```
δ_PTQ(N, D, P_post) = C_T * (D^γ_D / N^γ_N) * exp(-P_post / γ_post)
```

Where:
- `D` = training tokens
- `N` = parameters
- `P_post` = post-training weight precision (bits)
- `C_T, γ_D, γ_N, γ_post` = fitted constants

**Implication:** For highly overtrained models (D/N >> 20), there exists a critical data size beyond which **additional pretraining data is actively harmful** if the model will be quantized after training. The model compresses more information into weights during overtraining, making perturbations from quantization more damaging.

### 2. Effective Parameter Count During Training

Training with quantized weights/activations/KV-cache reduces the model's effective capacity. The effects are **multiplicative and independent**:

```
N_eff(P_w, P_a, P_kv) = N * (1 - exp(-P_w/γ_w)) * (1 - exp(-P_a/γ_a)) * (1 - exp(-P_kv/γ_kv))
```

Where:
- `P_w, P_a, P_kv` = precision of weights, activations, KV-cache during training
- `γ_w, γ_a, γ_kv` = sensitivity constants (activations are most sensitive)

The loss then follows a modified Chinchilla form:
```
L(N, D, P) = A * N_eff^(-α) + B * D^(-β) + E
```

### 3. Compute-Optimal Precision

When jointly optimizing N, D, and P:
- **Compute-optimal precision is independent of compute budget** (~7-8 bits with their fits)
- 16-bit training has unnecessary bits
- Sub-4-bit training requires disproportionately larger models to maintain loss scaling

**When model size N is constrained** (e.g., training a model family):
- Compute-optimal precision scales as `P* ∝ log(D/N)`
- Larger overtrained models should use higher precision

### 4. Unified Scaling Law

The complete unified form:
```
L(N, D, P_train, P_post) = A*N_eff^(-α) + B*D^(-β) + E + δ_PTQ(N_eff, D, P_train, P_post)
```

With PTQ degradation accounting for training precision:
```
δ_PTQ = C_T * exp(-P_post/γ_post) * (D^γ_D / N_eff^γ_N) * ∏[1 - exp(-C_x*(P_x - P_post))]
```

**Two competing effects:**
1. **Robustification:** Training in low precision makes weights robust to quantization noise
2. **Overtraining effect:** Lower N_eff increases degradation sensitivity

The robustification effect dominates, so **models trained in lower precision degrade LESS when post-train quantized**.

## Experimental Setup

- 465 pretraining runs
- Model sizes: 30M, 60M, 110M, 220M parameters
- Data: 1.5B to 26B tokens (D/N up to ~10^3)
- Precision sweep: 3-16 bits for weights, activations, KV-cache
- Architecture: OLMo-style Transformer++ on Dolma V1.7
- Integer quantization for fitting (validated on floating-point)

## Practical Implications

1. **Don't over-train if you'll PTQ:** At D/N >> 1000 with aggressive quantization, more training data can hurt inference performance

2. **7-8 bit training may be optimal:** 16-bit wastes compute, sub-4-bit requires unreasonable model size increases

3. **If training at low precision:** Increase parameters, decrease data (relative to Chinchilla-optimal)

4. **For model families at different sizes:** Larger models should be trained at higher precision if heavily overtrained

5. **Quantization-aware training helps PTQ:** Models trained with weight quantization are more robust to inference-time quantization

## Relevance to nanochat

### Current State in nanochat

nanochat currently uses:
- **BF16 training** on CUDA (`gpt.py:506`, `gpt.py:523`)
- **Float32 fallback** on CPU/MPS (`engine.py:181`)
- Embeddings cast to BF16 (`gpt.py:504-508`)
- Muon optimizer operates in BF16 (`optim.py:111`)

The speedrun trains to ~4 hours on 8xH100, and the codebase doesn't currently implement:
- Low-precision training (below BF16)
- Post-training quantization
- Quantization-aware training

### Potential Experiments for nanochat

1. **PTQ Sensitivity Study:**
   - After training d20 (561M) or d32 (2.2B) models, measure degradation when quantizing to 8-bit, 4-bit
   - Track how degradation changes with overtraining level (D/N)
   - nanochat's speedrun trains at ~D/N = 125 (70B tokens / 561M params), well within Chinchilla territory

2. **Compute-Optimal Precision Search:**
   - Train models at FP8 with adjusted model sizes
   - Compare loss at matched compute budgets
   - The paper suggests potential gains at 7-8 bit training

3. **Quantized Training Integration:**
   - Implement integer/FP8 quantization for forward pass
   - Key places to modify:
     - `gpt.py`: Add quantization hooks in `CausalSelfAttention` and `MLP`
     - `optim.py`: Already uses BF16 for Muon, could extend to lower precision
   - Focus on weights first (QAT), then activations/KV-cache

4. **Scaling Law Validation:**
   - Fit the paper's functional forms on nanochat's existing training runs
   - Validate whether `N_eff = N * (1 - exp(-P/γ))^3` holds for nanochat's architecture

5. **Inference Optimization:**
   - Implement GPTQ/AWQ/RTN for `engine.py`
   - Especially valuable for chat deployment with KV-cache quantization
   - Current `KVCache` class (`engine.py:93-103`) stores in full precision

### Key Code Locations

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Model precision | `gpt.py` | 504-523 | BF16 casting |
| KV Cache | `engine.py` | 93-103, 175-203 | Could add quantization |
| Optimizer | `optim.py` | 111, 160 | Already BF16-stable |
| Inference | `engine.py` | 313 | Autocast context |

### Specific Suggestions

1. **Low-hanging fruit:** Add PTQ evaluation script
   ```python
   # In scripts/chat_eval.py or new scripts/ptq_eval.py
   # Quantize trained model weights with RTN/GPTQ
   # Measure loss degradation vs full precision
   ```

2. **FP8 Training Experiment:**
   - Modify `gpt.py` to use `torch.float8_e4m3fn` for forward pass
   - Requires H100+ (native FP8 support)
   - Compare loss at matched compute

3. **Track D/N Sensitivity:**
   - Log post-quantization degradation at checkpoints
   - Plot δ_PTQ vs D/N to validate the paper's finding

### Caveats

- Paper's fits are on integer quantization; floating-point may differ slightly
- Architecture-specific: nanochat uses RoPE, RMSNorm, relu^2 - sensitivities may vary
- The paper uses OLMo-style models on Dolma, similar but not identical to nanochat
- Fitted constants (γ_w, γ_a, γ_kv) will need to be re-fit for nanochat's architecture

## Summary

This paper provides a principled framework for understanding precision/compute tradeoffs in LLM training. For nanochat, the most actionable insights are:

1. **If deploying quantized:** Monitor overtraining level carefully
2. **For maximum compute efficiency:** Consider FP8 training (7-8 bits)
3. **For inference:** Implement PTQ with awareness that overtrained models degrade more
4. **For research:** Validate these scaling laws on nanochat's architecture and training setup

The `N_eff` framework elegantly unifies precision effects into existing Chinchilla scaling, making it straightforward to reason about precision as "another dimension of scale" alongside parameters and data.
