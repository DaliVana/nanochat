# Paper Summary: Scaling Laws with Vocabulary - Larger Models Deserve Larger Vocabularies

**Paper:** arXiv:2407.13623v1
**Authors:** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, et al.
**Affiliations:** University of Hong Kong, Sea AI Lab, Contextual AI, Stanford University, Ohio State University

---

## Key Findings

### 1. Vocabulary Size Matters for Scaling

The paper demonstrates that **vocabulary size is a critical but overlooked component of scaling laws**. Prior work (Kaplan et al., Chinchilla) focused on model parameters and training data, but vocabulary parameters follow their own power-law scaling relationship.

### 2. Optimal Vocabulary Parameters Scale with FLOPs

The relationship between non-vocabulary parameters (N_nv) and optimal vocabulary parameters (N_v^opt) follows:

```
N_v^opt ∝ N_nv^γ  where γ ≈ 0.83 < 1
```

This means:
- Vocabulary parameters should scale **slower** than non-vocabulary parameters
- But they are still **critical** and should not be ignored
- Optimal vocabulary size: `N_v = k * C^0.42` (where C = FLOPs budget)

### 3. Current LLMs Are Under-Allocated

Most existing LLMs use insufficient vocabulary sizes:

| Model | Current V | Predicted Optimal V |
|-------|-----------|---------------------|
| Llama2-7B | 32K | ~62-67K |
| Llama2-70B | 32K | ~216K (7x larger!) |
| 3B params | 32K | ~39-43K |
| 7B params | 32K | ~60-67K |
| 13B params | 32K | ~81-91K |
| 70B params | 32K | ~212-231K |

### 4. Three Approaches to Predict Optimal Vocabulary

1. **IsoFLOPs Analysis:** Train models at fixed FLOPs with varying vocabulary sizes, fit power laws
2. **Derivative-based Estimation:** Minimize FLOPs w.r.t vocabulary size via calculus
3. **Parametric Loss Formula:** Modified Chinchilla loss function incorporating vocabulary:
   ```
   L_u = -E + A1/N_nv^α1 + A2/N_v^α2 + B/D^β
   ```

### 5. Undertraining vs Overtraining Effects

- **Undertraining (scarce data):** Optimal vocab is **smaller** (prevents overfitting on rare tokens)
- **Compute-optimal:** Follow the scaling laws
- **Overtraining (excessive data):** Optimal vocab is **larger** (better utilization of extra data)

### 6. Vocabulary-Insensitive Loss

The paper uses unigram-normalized loss L_u for fair comparison across vocabulary sizes:
```
L_u = -1/T * Σ log(p(w_i | context) / p(w_i))
```
This correlates well with bits-per-character (BPC) and downstream performance.

### 7. Token-Character Compression Ratio

They model the compression ratio as:
```
f(V) = a*log²(V) + b*log(V) + c
```
With fitted parameters: a=0.0064, b=-0.1581, c=1.2047

---

## Implications for nanochat

### Current nanochat Configuration

Looking at the nanochat codebase:

1. **Default vocab size:** 32,768 (2^15) - set in `tok_train.py` and `gpt.py`
2. **Speedrun model (d20):** 561M params with 32K vocab
3. **Run1000 model (d32):** ~1.88B params with 65,536 vocab (2^16)

### Analysis Using Paper's Predictions

**For the speedrun model (561M total params):**
- Estimated N_nv ≈ 500M (excluding embeddings)
- Using Approach 1 power law: N_v^opt ≈ 0.20 * C^0.42
- With typical embedding dim of ~768-1024, optimal V ≈ 24K-32K
- **Current 32K is approximately optimal** for this size

**For the run1000 model (~1.88B total params):**
- Estimated N_nv ≈ 1.75B
- From Table 1 in paper: for N_nv = 1.13B, optimal V ≈ 39-43K
- Extrapolating to 1.75B: optimal V ≈ 50-60K
- **Current 65K (2^16) is slightly higher than optimal**, but close

### Actionable Recommendations for nanochat

#### 1. Consider Dynamic Vocabulary Sizing

Add a utility function to compute optimal vocabulary size given model size:

```python
def compute_optimal_vocab_size(n_nv, d_model):
    """
    Compute optimal vocabulary size based on scaling laws paper.

    N_v^opt = 0.20 * C^0.42, and N_nv = 0.08 * C^0.50
    => N_v = 0.20 * (N_nv / 0.08)^(0.42/0.50) = 0.20 * (N_nv / 0.08)^0.84

    Then V = N_v / d_model
    """
    gamma = 0.84  # from paper's Approach 1
    n_v_opt = 0.20 * (n_nv / 0.08) ** gamma
    v_opt = n_v_opt / d_model

    # Round to nearest power of 2 or multiple of 128 for efficiency
    return max(4096, int(v_opt))
```

#### 2. Vocabulary Scaling Table for nanochat

Based on the paper's findings, recommended vocabulary sizes for different nanochat model depths:

| Depth | Approx N_nv | d_model | Recommended V |
|-------|-------------|---------|---------------|
| 12 | 100M | 768 | 16K-24K |
| 16 | 200M | 768 | 24K-32K |
| 20 | 500M | 1024 | 32K-40K |
| 24 | 800M | 1536 | 40K-48K |
| 28 | 1.2B | 1536 | 48K-56K |
| 32 | 1.8B | 2048 | 56K-72K |

#### 3. Overtraining Considerations

nanochat uses a 20:1 token-to-parameter ratio (Chinchilla optimal). However, if you want to train longer (like Llama2 at 2T tokens for 7B params):

- **Increase vocabulary size** when overtraining
- For 2x overtraining: increase V by ~20%
- For 4x overtraining: increase V by ~35%

#### 4. Data-Constrained Scenarios

If training data is limited:
- **Decrease vocabulary size** to prevent rare token underfitting
- For 0.5x of optimal data: decrease V by ~20%

#### 5. Tokenizer Training Implications

The paper trains tokenizers on varying amounts of data with varying vocab sizes. Key insight:
- Larger vocab → better compression ratio → more "characters per token"
- But need sufficient training data to learn rare token embeddings properly

Current nanochat trains tokenizer on 2B chars (speedrun) or 4B chars (run1000).
This is reasonable for vocab sizes of 32K-64K.

#### 6. Experiment Ideas

1. **Vocab scaling experiment:** Train same model (e.g., d20) with V=16K, 24K, 32K, 48K, 64K under same FLOPs budget, measure BPC and downstream performance

2. **Measure token embedding utilization:** Track average L2 norm of embeddings for rare vs common tokens to detect underfitting

3. **Implement unigram-normalized loss:** Add the paper's L_u metric to nanochat's evaluation for fair comparisons across vocab sizes:
   ```python
   def unigram_normalized_loss(logits, targets, token_frequencies):
       lm_loss = F.cross_entropy(logits, targets, reduction='none')
       unigram_log_probs = torch.log(token_frequencies[targets])
       return (lm_loss - unigram_log_probs).mean()
   ```

---

## Key Equations Reference

### FLOPs Calculation
```
C ≈ 6 * N * D = 6 * (N_nv + V*d) * H * f(V)
```
Where:
- C = FLOPs
- N = total params
- D = training tokens
- H = training characters
- f(V) = compression ratio (tokens per character)

### Optimal Allocations (Approach 1)
```
N_nv = 0.08 * C^0.50
N_v = 0.20 * C^0.42
H = 6.42 * C^0.50
```

### Loss Formula (Approach 3)
```
L_u = -E + A1/N_nv^α1 + A2/N_v^α2 + B/D^β
```
Fitted params: A1=1.831, A2=0.196, B=2.124, E=5.533, α1=β=0.447, α2=0.671

---

## Conclusion

The paper makes a compelling case that vocabulary size deserves more attention in LLM scaling. For nanochat:

1. **Current defaults are reasonable** but could be optimized
2. **Smaller models benefit more** from vocabulary optimization (proportionally)
3. **Add vocabulary size to hyperparameter tuning** for new model configurations
4. **Consider overtraining/undertraining effects** on optimal vocab size

The key insight is that vocabulary scaling follows `N_v ∝ N_nv^0.84` - a slower growth rate than non-vocabulary parameters, but still important enough to impact performance significantly (2-3% on downstream tasks).
