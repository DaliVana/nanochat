# Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations

**Paper**: Hägele et al. (2024), EPFL & Hugging Face
**arXiv**: 2405.18392v3
**Code**: https://github.com/epfml/schedules-and-scaling/

---

## Executive Summary

This paper argues that LLM training research has been **unnecessarily complex** due to reliance on the cosine learning rate schedule. The authors demonstrate that **constant learning rate + cooldown** achieves equivalent performance while providing significant advantages for flexible training and scaling experiments.

---

## Key Findings

### 1. Cosine Schedule Limitations

The cosine schedule requires knowing the exact training duration in advance:
- Optimal loss is achieved **only** when the cosine cycle length matches training duration
- Cosine is **suboptimal during training** (underestimates model performance)
- **Continuation is problematic**: rewarming causes loss spikes and training instability

### 2. Constant LR + Cooldown Alternative

A simple alternative: constant learning rate followed by a short cooldown phase:

```
η(n) = η_max                           if warmup < n ≤ N - N_decay
η(n) = f(n, N, N_decay) · η_max        if n > N - N_decay
```

**Key advantages**:
- No need to specify training steps in advance
- Cooldown can be initiated at any checkpoint
- Enables **continual learning** without loss spikes
- Supports **flexible data mixture** changes during cooldown
- Dramatically **reduces cost of scaling experiments**

### 3. Optimal Cooldown Settings

- **Cooldown length**: 10-20% of total training steps is sufficient
- **Cooldown shape**: `(1 - sqrt)` function consistently outperforms linear:
  ```
  f(n, N, N_decay) = 1 - sqrt((n - (N - N_decay)) / N_decay)
  ```
- For **long training runs**, even 5% cooldown can match cosine performance
- The optimal constant LR is approximately **half** the optimal cosine max LR

### 4. Stochastic Weight Averaging (SWA)

SWA improves performance during training without additional cost:
- Averages checkpoints within fixed windows (500-2500 steps optimal)
- Provides **reliable model estimates** at any point during training
- Does **not fully match** cooldown performance, but reduces the gap significantly
- Can be used on top of any schedule (constant or cosine)

### 5. Scaling Law Implications

The paper's most impactful finding for research efficiency:

| Approach | Runs per Model Size | Compute Savings |
|----------|---------------------|-----------------|
| Cosine (traditional) | N runs from scratch | Baseline |
| Constant + Cooldown | 1 run + N cooldowns | ~50% reduction |

The authors estimate the original **Chinchilla experiments** could have been done with **less than half** the compute using this approach.

---

## Relevance to nanochat

### Current nanochat Implementation

Looking at `scripts/base_train.py` (lines 250-270), nanochat already uses a **warmup-stable-decay (WSD)** schedule:

```python
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac
```

The default settings are:
- `--warmup-ratio 0.0` (no warmup by default)
- `--warmdown-ratio 0.5` (50% of training is warmdown/cooldown)
- `--final-lr-frac 0.0` (decay to zero)

This is **already aligned** with the paper's findings, but with a **linear decay** rather than the paper's recommended `(1-sqrt)` shape.

### Potential Improvements for nanochat

#### 1. Implement (1-sqrt) Cooldown Shape

The paper finds `(1-sqrt)` consistently outperforms linear decay. This could be added as an option:

```python
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        # (1-sqrt) decay as per paper
        decay_progress = (it - (num_iterations - warmdown_iters)) / warmdown_iters
        if args.cooldown_shape == "sqrt":
            decay_factor = 1 - decay_progress ** 0.5  # (1-sqrt) shape
        else:
            decay_factor = 1 - decay_progress  # linear (current)
        return decay_factor * (1 - args.final_lr_frac) + args.final_lr_frac
```

#### 2. Reduce Default Cooldown Length

The paper suggests 10-20% cooldown is sufficient. nanochat currently uses 50% (`--warmdown-ratio 0.5`), which may be excessive. Consider:
- Default of 20% (`--warmdown-ratio 0.2`)
- Or even 10% for very long runs

#### 3. Add Stochastic Weight Averaging (SWA)

SWA could be added as a free performance boost:

```python
# In training loop, maintain running average
if step % swa_window == 0 and step > swa_start:
    if swa_model is None:
        swa_model = copy.deepcopy(orig_model)
    else:
        for swa_p, model_p in zip(swa_model.parameters(), orig_model.parameters()):
            swa_p.data.lerp_(model_p.data, 1 / swa_count)
    swa_count += 1
```

The paper recommends windows of 500-2500 steps.

#### 4. Enable Flexible Cooldown from Checkpoints

Add a mode to resume from a checkpoint and immediately start cooldown:

```bash
python -m scripts.base_train --resume-from-step 50000 --cooldown-now --cooldown-steps 5000
```

This would enable efficient scaling experiments and model evaluation at different compute budgets.

#### 5. Scaling Law Experiments

For nanochat's miniseries/scaling experiments (`runs/scaling_laws.sh`), the paper's approach could dramatically reduce compute:

**Current approach** (hypothetical):
- Train d12, d16, d20, d24 each for 3 different token counts
- = 12 full training runs

**Paper's approach**:
- Train d12, d16, d20, d24 each once to max tokens
- Apply cooldowns from checkpoints for other token counts
- = 4 training runs + 8 cooldown-only continuations
- ~50% compute reduction

---

## Takeaways for nanochat Development

1. **Current schedule is good**: nanochat's WSD schedule is aligned with the paper's recommendations. The linear decay shape could be upgraded to (1-sqrt).

2. **Consider shorter cooldown**: 50% warmdown may be excessive; 20% might suffice.

3. **SWA is free lunch**: Adding checkpoint averaging would provide better intermediate model quality at no training cost.

4. **Scaling experiments can be cheaper**: Future scaling law work for nanochat could leverage the cooldown approach for significant compute savings.

5. **Continual training is viable**: The paper validates that constant LR training naturally supports training continuation without loss spikes, which is useful for nanochat's modular training pipeline (base → mid → SFT → RL).

---

## Technical Details

### Model Architecture (paper's experiments)
- Decoder-only transformer (LLaMA-style)
- SwiGLU, RoPE, RMSNorm
- Sizes: 33M to 8B parameters
- Dataset: SlimPajama (6B tokens), FineWeb (100B-460B tokens)

### Optimizer Settings
- AdamW with β₁=0.9, β₂=0.95
- Weight decay: 0.1
- Gradient clipping: 1.0
- Warmup: 300-3000 steps

### Key Experimental Results

| Schedule | Final Loss | Notes |
|----------|-----------|-------|
| Cosine (10% final LR) | Baseline | Requires knowing duration |
| Linear cooldown (20%) | ~= Cosine | Flexible |
| (1-sqrt) cooldown (20%) | < Cosine | Best performance |
| SWA on constant LR | Gap to cooldown | Free improvement |

---

## References

- Chinchilla scaling laws: Hoffmann et al. (2022)
- Original cosine schedule: Loshchilov & Hutter (2016) SGDR
- Schedule-free optimizer: Defazio et al. (2024)
- MiniCPM (uses WSD): Hu et al. (2024)
