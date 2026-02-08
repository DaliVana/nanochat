# Cut Your Losses in Large-Vocabulary Language Models

**Paper**: [arXiv 2411.09009](https://arxiv.org/abs/2411.09009) (ICLR 2025)
**Authors**: Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, Philipp Krahenbuhl (Apple)
**Code**: [github.com/apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)

## Problem

As LLM vocabularies grow, the cross-entropy loss layer becomes the dominant memory consumer during training. The logit matrix has shape `(N_tokens, |V|)` and for models like Gemma 2 (2B) with `|V|=256,128`, a batch of 8192 tokens produces a 24 GB logit matrix in fp32. This single layer can account for 40-89% of total training memory, dwarfing the model parameters, optimizer state, and activations combined.

## Core Idea: Cut Cross-Entropy (CCE)

The cross-entropy loss decomposes into two terms:

```
ell_i = C_{x_i}^T * E_i  -  log(sum_j exp(C_j^T * E_i))
```

- **Term 1** (indexed matmul): Only needs the logit for the *correct* token `x_i` -- a dot product between the embedding `E_i` and the classifier column `C_{x_i}`. Output is a scalar per token.
- **Term 2** (log-sum-exp): A reduction over all vocabulary entries. Also produces a scalar per token.

**Key insight**: Neither term requires materializing the full `|V| x N` logit matrix in GPU global memory. Both can be computed blockwise in on-chip SRAM using custom Triton kernels, analogous to how FlashAttention avoids materializing the `N x N` attention matrix.

## Three Custom Kernels

### 1. Indexed Matrix Multiplication (Forward)
Fuses the index lookup `C_{x_i}` with the dot product `C_{x_i}^T * E_i`. Loads blocks of `E` and the corresponding indexed columns of `C` into SRAM, computes dot products, writes only the scalar results to global memory. Zero GPU memory allocation.

### 2. Linear-Log-Sum-Exp (Forward)
Uses the same blocking strategy as standard matrix multiplication. Each CUDA block computes a `V_B x N_B` block of logits in SRAM, then reduces to a local log-sum-exp. Results are combined across blocks using a spin-lock atomic log-add-exp operation. Output: a vector of N scalars (`LSE`).

### 3. Linear-Log-Sum-Exp (Backward)
The gradient requires the softmax matrix `S = exp(C^T * E - LSE)`, which is also `|V| x N` and cannot be materialized. CCE recomputes the logits blockwise in SRAM (reusing the forward's matmul pattern), computes `S` on-the-fly, and accumulates gradients `dE` and `dC` directly.

Two critical optimizations for the backward pass:

- **Gradient Filtering**: The softmax distribution is extremely sparse -- empirically <0.02% of entries are non-negligible (above `2^{-12}` in bf16). For any block where all softmax values fall below this threshold, the entire gradient computation for that block is skipped. This yields a ~3.5x speedup with no loss of precision.

- **Vocabulary Sorting**: Tokens are reordered by their average logit (computed during the forward pass) so that "important" tokens cluster into contiguous blocks. This improves block-level sparsity, making gradient filtering more effective. Requires ~1 MB temporary buffer.

## Results

**Memory** (Gemma 2 2B, 8192 tokens, |V|=256K, A100):
| Method | Loss+Gradient Memory |
|--------|---------------------|
| Baseline (PyTorch) | 28,000 MB |
| torch.compile | 16,000 MB |
| Torch Tune (8 chunks) | 9,631 MB |
| Liger Kernels | 1,475 MB |
| **CCE** | **1,165 MB** |
| Lower bound | 1,161 MB |

CCE is within 4 MB of the theoretical lower bound (the gradient buffers themselves).

**Speed**: CCE computes Loss+Gradient in 145ms vs 143ms for torch.compile (1.4% slower) and 208ms for baseline (30% faster). Loss-only is faster than all baselines.

**Convergence**: Fine-tuning loss curves are indistinguishable from torch.compile across Gemma 2, Phi 3.5 Mini, Qwen 2.5 7B, and Mistral NeMo.

### Pretraining Considerations

For pretraining (vs fine-tuning), two modifications are needed (variant called `CCE-Kahan-FullC`):
1. **Disable gradient filtering on dC**: Without this, tokens with little training data support receive no gradient updates, harming validation perplexity.
2. **Kahan summation**: The bf16 accumulation in global memory loses precision that matters for pretraining. Kahan summation recovers this. Doubles the gradient memory (to ~2.3 GB for Gemma 2B) but still far below baseline.

With these modifications, pretraining validation perplexity matches torch.compile exactly.

### Removing Ignored Tokens

A practical optimization applicable to all methods: tokens with `ignore_index` (padding, system prompts, user input) can be filtered *before* logit computation rather than after. This gives up to 3x speedup for all methods. CCE already has minimal memory so the memory savings are smaller, but the speed improvement is significant.

### Scaling Properties

CCE's memory advantage is most dramatic when `|V|/D` is large. As hidden dimension grows relative to vocab size, the speed advantage diminishes but memory savings remain substantial. For Phi 3.5 Mini (`|V|=32K, D=3072`), CCE is ~50% slower than torch.compile but uses 8.5x less memory.

## Relevance to nanochat

### Current State in nanochat

In `nanochat/gpt.py:698-708`, the forward pass materializes the full logit matrix:

```python
logits = self.lm_head(x)  # (B, T, padded_vocab_size) <- "very big tensor, large amount of memory"
logits = logits[..., :self.config.vocab_size]
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)  # logit softcapping
return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

There is even a TODO comment: `# TODO experiment with chunked cross-entropy?`

### Impact Assessment

nanochat uses `vocab_size=32,768` by default. At this vocabulary size, the logit matrix for a batch of 8192 tokens is `8192 * 32768 * 4 bytes = 1 GB` in fp32. This is substantial but not as extreme as the Gemma 2 case (256K vocab). The memory savings would be:
- Baseline: ~1 GB for logits alone
- CCE: ~0.2 MB for logits (essentially free)

For nanochat's d20 model (561M params, `n_embd=1024`), the `|V|/D` ratio is ~32, which is favorable for CCE. For the d32 model (2.2B params, `n_embd=2048`), the ratio is ~16, still reasonable.

### Integration Considerations

1. **Logit softcapping**: nanochat uses `softcap = 15` with `tanh` squashing. The CCE paper's backward pass breakdown explicitly includes softcap gradient computation and shows it benefits from the fused kernel (17ms -> 4.7ms for Gemma 2B). The Apple `ml-cross-entropy` library supports softcapping natively.

2. **Drop-in replacement**: The library provides `cce_loss = LinearCrossEntropyLoss()` that takes `(embeddings, classifier_weights, targets)` instead of `(logits, targets)`. This requires restructuring the forward pass to not compute logits explicitly when training. The `lm_head` linear layer would not be called; instead, its weight matrix would be passed directly to CCE.

3. **Pretraining vs fine-tuning**: For nanochat's pretraining (`base_train.py`), the `CCE-Kahan-FullC` variant should be used. For SFT (`chat_sft.py`) and RL (`chat_rl.py`), standard CCE is sufficient.

4. **Ignored token filtering**: nanochat already uses `ignore_index=-1` for padding/non-target tokens. The paper shows that filtering these before logit computation gives large speedups. This is orthogonal to CCE and could be applied independently.

5. **Practical benefit**: The main benefit for nanochat would be enabling larger batch sizes on the same hardware. On 8xH100, freeing ~1 GB per GPU from the logit matrix could allow increasing `--device-batch-size` or `--total-batch-size`, potentially improving training throughput and stability.

### What to Try

- **Quick win**: Install `pip install cut-cross-entropy` (or from `apple/ml-cross-entropy`) and replace the loss computation in `gpt.py:forward()`. This requires passing `self.lm_head.weight` and the normalized embeddings `x` directly to the CCE function instead of computing logits first.
- **Benchmark**: Compare memory usage and training speed with and without CCE on the speedrun configuration. The `|V|=32768` vocab size should show moderate memory savings and comparable speed.
- **Scaling**: If nanochat experiments with larger vocabularies (the paper cites evidence that even 256K may benefit from expansion), CCE becomes increasingly critical.
- **Token filtering**: Independently implement pre-filtering of ignored tokens before the lm_head projection. This is a simple change that benefits all methods.
