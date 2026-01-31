# APEX: Advantageous Parameter Expansion Training

**Paper:** "Advantageous Parameter EXpansion Training Makes Better Large Language Models"
**Authors:** Naibin Gu, Yilong Chen, Zhenyu Zhang, Peng Fu, et al. (CAS & Baidu)
**Venue:** NeurIPS 2025
**ArXiv:** 2505.24241

## Core Insight

Not all parameters in LLMs contribute equally to model performance. The paper identifies **advantageous parameters** (those with high activations during forward pass) as critical for performance, while **disadvantageous parameters** (low activations) contribute less. Key observation: stronger models have a more uniform distribution of activation magnitudes (lower std deviation), suggesting better parameter utilization.

## The APEX Method

APEX progressively **expands** the parameter space of advantageous parameters into the space of disadvantageous ones, increasing the proportion of useful parameters without changing model architecture or parameter count.

### 1. Advantage Assessment

Track activation magnitudes during forward passes:
- **MHA heads:** `A_MHA^h = ||head_h||_F^2` (Frobenius norm of head output)
- **FFN channels:** `A_FFN^c = ||f(X @ W_G[:,c])||_F^2` (post-activation norm)

Score each component using relative ranking across a dataset:
```
s_MHA^h = count(A^h in Top-K) - count(A^h in Min-K)
```

Select advantageous (`d^P`) and disadvantageous (`d^N`) parameter sets based on these scores.

### 2. Expansion Operators

Introduce learnable transformation matrices that expand advantageous parameters into disadvantageous space:

**For MHA:**
```
head_N_new = gamma_MHA(head_P, head_N)
W_O[d^N,:] = gamma_MHA(W_O[d^P,:], W_O[d^N,:])
```

**For FFN:**
```
W_U[:,d^N] = gamma_FFN(W_U[:,d^P], W_U[:,d^N])
W_G[:,d^N] = gamma_FFN(W_G[:,d^P], W_G[:,d^N])
W_D[d^N,:] = gamma_FFN(W_D[d^P,:], W_D[d^N,:])
```

**Operator implementation:**
```
W_new = W_P @ M + W_N  (where M is transformation matrix, init to zeros)
```

Uses **Monarch matrices** for efficient parameterization (O(d^2) instead of O(d^4)):
```
M = [D_ij] @ diag(R_1, ..., R_d)
```
where D, R are diagonal matrices. R initialized to zeros (for zero-init of M), D randomly initialized.

### 3. Stage-wise Training

Training is divided into multiple stages:
1. At stage start: reassess and reselect advantageous/disadvantageous parameter sets
2. During stage: train with expansion operators active, continuously track activations
3. At stage end: fuse operators back into model weights

```
W[:,d^N]_final = W[:,d^P] @ M + W[:,d^N]
```

This preserves the original architecture (plug-and-play).

## Theoretical Justification

APEX improves the **effective rank** of weight matrices. The transformation expands the column space of disadvantageous parameters to be more orthogonal to advantageous ones, increasing overall matrix rank utilization.

## Key Results

### Instruction Tuning (LLaMA2-7B on Tulu V2)
| Method | Trainable Params | Avg Score |
|--------|-----------------|-----------|
| Full-FT | 100% | 41.0 |
| LoRA | 2.4% | 40.1 |
| APEX | 2.4% | **42.6** |
| HFT | 52% | 42.9 |
| GMT | 60% | 44.0 |
| APEX | 52% | **44.0** |

**APEX matches 60% parameter training (GMT) with only 52% trainable parameters, and beats Full-FT with just 2.4% parameters.**

### Continued Pre-training (TinyLLaMA 1.1B on RedPajama)
| Method | Data | Perplexity |
|--------|------|------------|
| Vanilla | 10B tokens | 5.83 |
| APEX | 3B tokens | 5.74 |
| APEX | 10B tokens | **5.44** |

**APEX achieves lower perplexity with 33% of the training data!**

## Hyperparameters

- `K_MHA`, `K_FFN`: proportion threshold for selecting top/bottom parameters (18.75% for IT, 12.5% for CPT)
- Number of stages: 2-4 (diminishing returns after 3)
- Optimal improvement at 18.75% threshold (max +1.1% improvement)

## Relevance to nanochat

### Direct Applicability

1. **Architecture Match:** nanochat uses standard Transformer with MHA and FFN (Gated Linear Units with `relu^2`), exactly matching APEX's target components:
   - MHA: `c_q`, `c_k`, `c_v`, `c_proj` matrices with per-head structure
   - FFN: `c_fc`, `c_proj` in `MLP` class
   - MLA variant also has clear per-head structure

2. **Training Pipeline:** nanochat already has:
   - **Continued pre-training** (`base_train.py`)
   - **SFT** (`chat_sft.py`)
   - Multi-stage training capability

   APEX would fit naturally into these pipelines.

### Implementation Ideas for nanochat

1. **Activation Tracking Module:**
   Add hooks to track per-head (MHA) and per-channel (FFN) activation norms during forward pass:
   ```python
   # In CausalSelfAttention.forward():
   head_activations[h] = torch.norm(head_h, 'fro')**2

   # In MLP.forward():
   channel_activations[c] = torch.norm(F.relu(x @ c_fc.weight[:,c]).square(), 'fro')**2
   ```

2. **Expansion Operators:**
   Could be implemented as additional `nn.Linear` layers or custom Monarch matrix modules.

3. **Stage Management:**
   Add stage boundaries to training loop in `base_train.py`, fusing operators at stage transitions.

### Potential Benefits for nanochat

1. **Faster Convergence:** Paper shows 33% data efficiency improvement in continued pre-training. For nanochat's speedrun (~4h on 8xH100), this could mean similar quality with less compute.

2. **Better Fine-tuning:** APEX beats full-parameter tuning with 52% parameters. For `chat_sft.py`, this could improve quality without increasing cost.

3. **Synergy with Existing Features:**
   - **Value Embeddings (ResFormer):** nanochat already uses per-head gating for value embeddings. APEX's head-level advantage assessment aligns well with this.
   - **Per-layer scalars:** `resid_lambdas` and `x0_lambdas` could potentially be incorporated into advantage assessment.

### Considerations

1. **MLA Compatibility:** nanochat's MLA attention compresses KV through shared latent space. APEX's per-head expansion would need adaptation - perhaps expand at the compressed latent level instead.

2. **Overhead:** APEX adds minimal compute (+0.7% training time, 1.01x FLOPs) - acceptable for nanochat's efficiency focus.

3. **Stage Selection:** Paper suggests 3 stages is optimal (diminishing returns beyond). For nanochat's continued pre-training with 10B tokens per stage, this aligns well.

## Key Takeaways

1. **Parameter advantage is measurable** via activation magnitude tracking
2. **Strong models have more uniform activation distributions** (lower std dev)
3. **Expanding advantageous to disadvantageous parameter space** improves utilization
4. **Monarch matrices** provide efficient parameterization for expansion operators
5. **Stage-wise training** with periodic re-assessment is key
6. **33% data efficiency** in pre-training, **52% parameter efficiency** in fine-tuning

## Citation

```bibtex
@inproceedings{gu2025apex,
  title={Advantageous Parameter Expansion Training Makes Better Large Language Models},
  author={Gu, Naibin and Chen, Yilong and Zhang, Zhenyu and Fu, Peng and Lin, Zheng and Wang, Shuohuan and Sun, Yu and Wu, Hua and Wang, Weiping and Wang, Haifeng},
  booktitle={NeurIPS},
  year={2025}
}
```
