# Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs

**Paper**: Bergsma et al. (2025), Cerebras Systems
**arXiv**: 2502.15938v2
**Key Result**: Linear decay-to-zero (D2Z) consistently outperforms cosine decay to 10% at compute-optimal and over-trained TPP settings

---

## Executive Summary

This paper presents a large-scale empirical study showing that **linear decay-to-zero (D2Z)** is the optimal learning rate schedule for LLM training with AdamW. The key insight is that at compute-optimal tokens-per-parameter (TPP ~20) and especially at higher TPP (over-training), decaying the LR all the way to zero significantly outperforms the common practice of decaying to 10% of peak LR ("10x decay").

The paper also provides a novel theoretical interpretation: AdamW weights can be viewed as an **exponential moving average (EMA) of weight updates**, where the LR schedule controls the combination coefficients. This perspective explains why linear D2Z optimally balances:
1. **Bias reduction** (early training): Moving away from random initialization
2. **Variance reduction** (late training): Averaging over gradient noise

---

## Key Findings

### 1. Linear D2Z is Optimal

At 20 TPP (compute-optimal) and higher:
- Linear D2Z achieves **0.77% lower loss** than linear 10x decay at 610M scale
- Linear D2Z achieves **~0.1% lower loss** than cosine D2Z (consistent across all LRs)
- Benefits **increase with TPP**: at 200 TPP, D2Z is 2.8% better than 10x decay

**Critical insight**: A 610M model trained for 80 TPP with D2Z achieves **lower loss** than the same model trained for 200 TPP with 10x decay - **60% compute savings**!

### 2. Why 10x Decay Has Persisted

The paper explains why D2Z hasn't been widely adopted:

1. **Low-TPP bias**: At very low TPP (< 4), 10x decay actually performs better - but there's no practical reason to train LLMs at such low TPP
2. **Coupled hyperparameters**: Optimal peak LR differs between schedules; if you tune LR for 10x and test D2Z at that LR, D2Z may appear worse
3. **Training stability**: D2Z benefits from higher peak LRs, but unstable models may not tolerate them (NanoGPT needed fp32 to reach optimal D2Z LR)

### 3. Hyperparameter Stability

D2Z provides much **more stable optimal hyperparameters**:
- Optimal peak LR barely shifts as TPP increases (vs. significant shift for constant/10x)
- Optimal peak LR is less sensitive to batch size changes
- Optimal peak LR is less sensitive to weight decay changes
- Loss curves are "flatter" - suboptimal LRs hurt less

This is crucial for hyperparameter transfer with muP.

### 4. EMA Interpretation of AdamW

The paper's key theoretical contribution: AdamW weights at step t are a **convex combination** of all prior weight updates:

```
theta_t = sum_{i=1}^t c_{t,i} * x_i
```

where coefficients c_{t,i} depend on the full LR schedule (not just instantaneous LR). Key insights:

- **Constant LR**: Coefficients decay exponentially backward - recent updates dominate
- **Decaying LR**: Coefficients flatten out - more updates contribute evenly
- **Linear D2Z**: Optimal balance - early updates aren't over-weighted, late updates aren't under-weighted
- **Cosine D2Z**: Slightly worse because it approaches zero too quickly, under-weighting final updates

### 5. Bias vs Variance Trade-off

The bias/variance decomposition from SGD theory:
```
E[L(theta_t) - L(theta_*)] <= (1 - eta*mu)^t ||theta_0 - theta_*||^2 + eta * sigma^2
                              ^^^^^^^^^^ BIAS ^^^^^^^^^^^^^^^^^^   ^^^ VARIANCE
```

- **Bias** (initial conditions): Decreases exponentially with steps, depends on average LR
- **Variance** (gradient noise): Increases with TPP, reduced by averaging more updates

At higher TPP:
- Bias is naturally handled by more steps
- Variance becomes the bottleneck
- D2Z's flatter coefficients average over more updates, reducing variance

### 6. Practical Implications

| Finding | Implication |
|---------|-------------|
| D2Z better at 20+ TPP | Use D2Z for all production LLM training |
| Benefits grow with TPP | Over-trained models (Llama-2 at 286 TPP) wasted compute |
| Stable hyperparameters | Easier muP transfer, less tuning needed |
| Less overfitting to final batches | Data ordering matters less with D2Z |
| Higher optimal peak LR | May need better training stability (QK-norm, etc.) |

### 7. WSD Comparison

The paper also shows D2Z outperforms Warmup-Stable-Decay (WSD):
- WSD is 0.84% worse than D2Z at 80 TPP
- WSD's constant phase is "not truly schedule-free" - optimal constant LR still depends on TPP
- The cooldown phase of WSD essentially admits that D2Z is needed at the end

---

## Connection to nanochat

### Current nanochat LR Schedule (base_train.py:250-260)

```python
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)  # default 0.0
    warmdown_iters = round(args.warmdown_ratio * num_iterations)  # default 0.5
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac  # default 0.0
```

**This IS linear D2Z by default!** With `warmup_ratio=0.0`, `warmdown_ratio=0.5`, `final_lr_frac=0.0`:
- No warmup
- Constant LR for first 50% of training
- Linear decay to zero for last 50% of training

This is similar to WSD with 50% cooldown. According to the paper, nanochat could benefit from:

### Potential Improvements

1. **More warmup**: The paper uses 10% warmup as standard practice, noting it may reduce LR sensitivity. Current nanochat default is 0% warmup.

2. **Less constant phase, more decay**: The paper suggests linear decay over the full training (not just 50%) might be optimal. Consider testing:
   - `warmup_ratio=0.1, warmdown_ratio=0.9` (10% warmup, then 90% linear decay)
   - Or even just removing the constant phase entirely

3. **Shorter warmdown might suffice**: At high TPP, even 22.5% cooldown was effective in the paper. The 50% may be more than needed.

4. **Peak LR tuning**: D2Z allows higher peak LRs. If training is stable, could try increasing peak LRs.

5. **Weight decay schedule**: nanochat already has a weight decay schedule that linearly decays to zero (line 269-270):
   ```python
   def get_weight_decay(it):
       return weight_decay_scaled * (1 - it / num_iterations)
   ```
   This is interesting - the paper suggests weight decay primarily matters for variance reduction, which aligns with decaying it as training progresses.

### Specific Recommendations for nanochat

1. **Experiment with full linear decay**: Change defaults to:
   ```python
   parser.add_argument("--warmup-ratio", type=float, default=0.1)  # 10% warmup
   parser.add_argument("--warmdown-ratio", type=float, default=0.9)  # 90% linear decay
   ```
   This would give true linear D2Z after warmup.

2. **Consider the bias/variance trade-off**: At nanochat's default 10.5 TPP ratio, D2Z should be beneficial but not as dramatically as at 20+ TPP. For higher TPP training (e.g., for speculative decoding draft models), D2Z becomes even more important.

3. **Training stability**: If increasing peak LRs causes instability, ensure:
   - QK normalization is enabled (nanochat has this)
   - Consider fp32 for numerical precision if needed
   - Longer warmup can help stability

4. **Monitor for overfitting patterns**: The paper shows D2Z reduces overfitting to late training batches. Could verify this with nanochat.

---

## Key Equations

### EMA Coefficient Formula
```
c_{t,i} = (prod_{j=i+1}^{t} (1 - alpha_j)) * alpha_i
```
where `alpha_j = eta_j * lambda` (LR times weight decay)

### Bias Coefficient Approximation
```
c_{t,1} ≈ (1 - alpha_bar)^{t-1}
```
Bias decreases exponentially in absolute steps, not fraction of training.

### Optimal Schedule Properties
- Linear D2Z: coefficients are roughly trapezoidal, giving balanced contribution
- Cosine D2Z: coefficients drop too fast at end, under-weighting final updates
- 10x decay: coefficients are too spiky, over-weighting recent updates

---

## Experimental Details

- Models: 111M to 1.7B parameters
- Dataset: SlimPajama
- Optimizer: AdamW with weight decay = 0.1
- Parameterization: muP (maximal update parameterization)
- Warmup: 10% of total steps
- Hardware: Cerebras CS-3

---

## Citation

```bibtex
@article{bergsma2025straight,
  title={Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs},
  author={Bergsma, Shane and Dey, Nolan and Gosal, Gurpreet and Gray, Gavia and Soboleva, Daria and Hestness, Joel},
  journal={arXiv preprint arXiv:2502.15938},
  year={2025}
}
```
