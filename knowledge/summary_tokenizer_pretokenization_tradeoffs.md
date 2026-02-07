# Getting the Most Out of Your Tokenizer for Pre-Training and Domain Adaptation

**Paper:** Dagan, Synnaeve, Roziere (University of Edinburgh / Meta AI), ICML 2024
**ArXiv:** [2402.01035v2](https://arxiv.org/abs/2402.01035v2)

---

## TL;DR

Tokenizer design (vocab size, pre-tokenization regex, training data mix) significantly impacts compression, inference speed, context size, and memory — but has surprisingly little impact on downstream task performance, as long as you avoid extreme choices (like no pre-tokenization at all). Swapping tokenizers during fine-tuning is viable with >50B tokens of adaptation. The GPT-4 pre-tokenization regex is recommended as the best compression-performance tradeoff.

---

## Key Findings

### 1. Three Levers for Compression

The paper identifies three main ways to increase in-domain tokenizer compression:
- **Training data**: Train the tokenizer on in-domain data (tokenizers should be trained on the data mix they will see during training/inference)
- **Pre-tokenization regex**: The regex pattern that splits text before BPE determines an upper bound on token composition
- **Vocabulary size**: Larger vocab = better compression, but with diminishing returns and memory costs

### 2. Pre-tokenization Regex Matters Most for Performance

The paper compares four pre-tokenization schemes:
- **Llama** (GPT-2 style): The baseline, with English-specific contractions
- **GPT-4**: Extended GPT-2 regex with 3-digit max, better multilingual support
- **Punct**: A custom regex that more aggressively separates syntax from semantics (e.g., `.append` becomes `.` + `append`)
- **Identity**: No pre-tokenization at all (maximum compression, worst performance)

Key results:
- **Identity tokenizer**: Gets 30% better code compression than Llama but has severely degraded downstream performance. Without token healing, it essentially cannot generate valid code on HumanEval.
- **GPT-4 regex**: Best compression-performance balance. Similar downstream performance to Llama but 19-25% better code compression.
- **Punct**: Sometimes slightly better Pass@1, but the advantage disappears at 7B scale. Sacrifices ~5% compression vs GPT-4.
- **Recommendation**: Use GPT-4 regex — the extra 5% compression over Punct is free.

### 3. Vocabulary Size Has No Significant Impact on Performance

Testing 32k, 64k, 128k, 256k with GPT-4 regex on 1.5B model:
- Pearson correlation between vocab size and HumanEval Pass@1: r = -0.13, p = 0.87
- **Conclusion**: Vocab size does not significantly affect task performance at these scales.

### 4. Optimal Vocabulary Size Depends on Model Size

- **Inference-optimal vocab size** grows with model size (larger models benefit more from compression gains since forward pass is expensive)
- **Memory-optimal vocab size** depends on batch size and sequence length. For short sequences/small batches, keep vocab small. For long sequences/large batches, larger vocab saves KV cache memory.

### 5. Tokenizer Switching During Fine-Tuning

- Pre-trained model weights are still leveraged even after tokenizer change (fine-tuned models with new tokenizer > trained from scratch with new tokenizer)
- **Fast Vocabulary Transfer (FVT)** helps: initializing new embeddings from old ones gives noticeable improvement
- **Tokenizer extension** (adding domain-specific tokens to existing tokenizer) gives small gains over complete replacement
- **Freezing non-embedding weights** during adaptation is ineffective
- **Critical threshold**: ~50B tokens of fine-tuning needed for the model to fully recover from tokenizer change

### 6. Token Healing

Token healing (backtracking by N tokens at prompt boundaries and constraining first decoded token) is critical for tokenizers with loose boundaries (Identity), but has negligible effect on well-structured tokenizers (GPT-4, Punct, Llama). The paper introduces an N-step backtrack version vs. the original single-step.

### 7. BPE Dropout

Randomly dropping merge entries during training makes models more robust to tokenization variations and prompt boundary effects.

### 8. Renyi Entropy

Higher Renyi entropy (more even token distribution) correlates *negatively* with code generation performance (Pearson r=-0.78 on HumanEval, r=-0.90 on MBPP). The Identity tokenizer has highest Renyi entropy but worst performance. This contradicts findings from MT tasks.

---

## Relevance to nanochat

### Current nanochat tokenizer design

nanochat already follows many of the paper's recommendations:
- Uses a custom BPE tokenizer trained with HuggingFace tokenizers / rustbpe + tiktoken
- Uses a GPT-4-style pre-tokenization regex (evolved beyond the paper's GPT-4 regex)
- Default vocab size is 32,768 (32K)
- Trains on 2B characters from the FineWeb-Edu dataset
- No normalization (the paper also recommends against normalization for reversibility)

### nanochat's SPLIT_PATTERN vs. the paper's findings

nanochat's current regex (`tokenizer.py:116`) is a significantly evolved version of the GPT-4 regex:

```
[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*|0[xXbBoO][\p{N}a-fA-F]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Key differences from the paper's GPT-4 regex:
1. **Unicode marks (\p{M})**: nanochat includes combining marks for proper Abugida script support (Tamil, Hindi, Thai, etc.). The paper's tokenizers don't address this.
2. **Language-agnostic contractions**: nanochat uses `(?:'[\p{L}\p{M}]+)*` instead of GPT-4's explicit English suffix list (`'s|'t|'re|'ve|'ll|'d|'m`). This is a superset that also handles French/Italian elision.
3. **Hex/binary/octal literals**: nanochat has a dedicated rule `0[xXbBoO][\p{N}a-fA-F]+` that keeps hex literals intact. The paper doesn't mention this.
4. **Digit grouping**: nanochat uses `\p{N}{1,3}` — the paper tested {1,3} and notes it costs 1110 vocab slots but helps with IPs, dates, HTTP codes.

### Actionable insights for nanochat

1. **Vocab size experimentation**: The paper shows vocab size has negligible impact on performance. nanochat uses 32K; the paper suggests that for the d20 model (561M params), this is reasonable, but for the d32 model (2.2B params), a larger vocab (64K-100K) could improve compression and inference speed at no performance cost. Consider adding a `--vocab-size` parameter to `tok_train.py` (it already has one, defaulting to 32768).

2. **Compression evaluation**: nanochat's `tok_eval.py` already computes bytes-per-token ratios across domains (news, Korean, code, math, science). This aligns well with the paper's NSL metric. Could add NSL computation against GPT-4's tokenizer as a standardized comparison.

3. **Training data mix for tokenizer**: The paper finds that tokenizers should be trained on the expected data distribution. nanochat trains only on FineWeb-Edu (general English text). If the goal is also code generation, mixing in code data would improve code compression. The paper uses 70% code / 30% English for their code-focused tokenizers.

4. **Token healing**: nanochat doesn't appear to implement token healing in its inference engine. For well-structured tokenizers (like nanochat's), the paper shows this is not critical, but it could still help at prompt boundaries during chat inference.

5. **BPE dropout during training**: Not currently implemented in nanochat. Could be a simple way to make the model more robust to tokenization edge cases, especially useful during SFT and chat fine-tuning.

6. **FVT for tokenizer changes**: If nanochat users want to switch tokenizers mid-training (e.g., adding domain-specific tokens for code), the paper validates that FVT + 50B tokens of continued training is sufficient. This is relevant for the midtraining stage (`mid_train.py`).

7. **The Punct finding**: nanochat's regex does *not* adopt the Punct-style separation of `.` from method names (i.e., `.append` could be a single token). The paper shows this doesn't help at scale (7B), so nanochat's choice to keep GPT-4-style here is validated.

---

## Summary Table: Paper's Pre-tokenization Regex Recommendations

| Regex | Code Compression (NSL) | HumanEval Pass@1 | Recommendation |
|-------|----------------------|------------------|----------------|
| Llama | 1.00 (baseline) | 20.5% (1.5B) | Reasonable default |
| GPT-4 | 0.81 (19% better) | 20.5% (1.5B) | **Recommended** — best balance |
| Punct | 0.86 (14% better) | 21.1% (1.5B) | Slightly better perf, worse compression |
| Identity | 0.69 (31% better) | 18.8% (1.5B) | Avoid — too fragile |

---

## Citation

```bibtex
@inproceedings{dagan2024tokenizer,
  title={Getting the most out of your tokenizer for pre-training and domain adaptation},
  author={Dagan, Gautier and Synnaeve, Gabriel and Rozi{\`e}re, Baptiste},
  booktitle={ICML},
  year={2024}
}
```
