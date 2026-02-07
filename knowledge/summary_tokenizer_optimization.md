# Getting the Most Out of Your Tokenizer for Pre-Training and Domain Adaptation

**Paper:** Dagan, Synnaeve, Rozière (University of Edinburgh / Meta AI), ICML 2024
**ArXiv:** 2402.01035

---

## Core Contribution

This paper provides a comprehensive study of how tokenizer design choices — vocabulary size, pre-tokenization regex, and training data — affect downstream LLM performance, compression, inference speed, and memory usage. The key finding is that tokenizers are an underexplored lever for improving LLM efficiency, and that a pre-trained model's tokenizer can be swapped during fine-tuning with minimal performance loss if trained on enough data (>50B tokens).

---

## Key Findings

### 1. Three Levers for Tokenizer Compression

- **Training data distribution**: Tokenizers should be trained on data resembling their deployment domain. Training on 70% code + 30% English maximized code compression.
- **Pre-tokenization regex**: The regex pattern that splits text before BPE significantly impacts both compression and downstream performance. The paper compares:
  - **Llama** (GPT-2 style): Baseline, single-digit splitting
  - **GPT-4**: More refined, limits digits to 3, allows `.append`-style tokens
  - **Punct**: Stronger syntax/semantics separation (splits on `.` before identifiers), slightly better Pass@1 at 1.5B but no advantage at 7B
  - **Identity**: No pre-tokenization at all — maximum compression (~30% fewer tokens than Llama on code) but significantly degraded downstream performance
- **Vocabulary size**: Larger vocab → better compression with diminishing returns. Crucially, **vocabulary size (32k–256k) has no statistically significant impact on downstream code generation performance**.

### 2. Tokenizer Switching During Fine-Tuning

- Pre-trained model weights are still leveraged after a tokenizer change — training from scratch with the same tokenizer is worse than switching tokenizers on a pre-trained model.
- After ~50B tokens of fine-tuning, a model with a swapped tokenizer recovers to match or exceed the original tokenizer's performance.
- Fast Vocabulary Transfer (FVT) helps initialize new embedding weights and provides noticeable improvement over random initialization.
- Freezing non-embedding weights during tokenizer adaptation is **ineffective** — full fine-tuning works better.

### 3. Optimal Vocabulary Size

- **Inference-optimal**: Larger models benefit from larger vocabularies (compression savings outweigh per-step cost increase). For 7B+ models, vocabularies of 80k+ are beneficial.
- **Memory-optimal**: Depends on sequence length and batch size. For short sequences (1000 tokens) and batch=1, small vocabs are better. For long sequences or large batches, larger vocabs save memory via reduced KV cache.

### 4. Token Healing

- Token healing (backtracking at prompt boundaries to fix tokenization misalignment) is critical for tokenizers without strong word boundaries (Identity tokenizer drops from ~19% to ~0.1% HumanEval Pass@1 without it). For well-structured tokenizers (GPT-4, Punct), the effect is negligible.

### 5. BPE Dropout

- Randomly dropping merge rules during training creates more robust tokenizers that handle prompt boundary issues better. Referenced but not deeply ablated.

### 6. Compression Comparisons

Using Normalized Sequence Length (NSL) against Llama as baseline:
- GPT-4 (100k): 0.75 on code (25% fewer tokens)
- Their GPT-4-regex (100k): 0.74 on code
- Identity (100k): 0.59 on code (41% fewer tokens, but kills performance)
- Punct (100k): 0.81 on code
- Key insight: GPT-4 regex at 32k already compresses 19% better than Llama on code

---

## Recommendations from the Paper

1. **Use GPT-4 pre-tokenization regex** — best balance of compression and performance
2. **Match tokenizer training data to deployment domain** (e.g., 70% code for code models)
3. **Don't be afraid to increase vocabulary size** — it doesn't hurt performance and improves compression
4. **Tokenizer switching is viable** if fine-tuning for 50B+ tokens with FVT initialization
5. **Avoid Identity (no pre-tokenization)** unless you only care about Pass@100 / large context
6. **Use token healing** especially with aggressive tokenizers

---

## Relevance to nanochat

### Current nanochat tokenizer setup

nanochat uses a GPT-4-style BPE tokenizer (`nanochat/tokenizer.py`) with:
- **Vocab size**: 32,768 (2^15) — on the smaller end
- **Pre-tokenization regex**: Modified GPT-4 pattern with `\p{N}{1,2}` instead of `\p{N}{1,3}` (max 2 digits per number token instead of 3)
- **Training data**: 2B characters from pretraining corpus via `parquets_iter_batched`
- **No normalization** (consistent with paper's recommendation)
- **Implementation**: rustbpe for training + tiktoken for inference
- **Evaluation**: `tok_eval.py` compares bytes-per-token against GPT-2 and GPT-4 tokenizers

### Potential improvements inspired by this paper

1. **Digit splitting choice is validated**: nanochat's decision to use `\p{N}{1,2}` (max 2-digit tokens) instead of GPT-4's `\p{N}{1,3}` is reasonable for a 32K vocab, though the paper doesn't deeply ablate this specific choice. The paper does note that GPT-4's 3-digit limit already prevents wasteful long-number tokens. nanochat goes even further.

2. **Vocabulary size could be increased**: The paper shows vocab size (32k→256k) doesn't hurt code performance and improves compression. For nanochat's speedrun (561M params), the embedding/unembedding overhead is more significant (~5% of params at 32k, ~10% at 64k for a 561M model), so the tradeoff is less clear. For the 2.2B run1000 model, going to 64k would be worth testing — the paper shows the parameter overhead becomes negligible for larger models.

3. **Training data for tokenizer**: nanochat trains its tokenizer on its pretraining corpus. The paper confirms this is correct practice — tokenizer training data should match deployment data. If nanochat were specialized for code, training tokenizer on 70% code would help.

4. **Token healing for inference**: nanochat's inference engine (`engine.py`) does not appear to implement token healing. For the GPT-4-style tokenizer nanochat uses, the paper shows this has negligible impact, so it's fine to skip. But if nanochat ever experimented with more aggressive tokenizers, token healing would become critical.

5. **Tokenizer evaluation could be extended**: nanochat's `tok_eval.py` measures bytes-per-token ratio. The paper introduces Normalized Sequence Length (NSL) as a relative metric. Adding NSL computation and evaluating on domain-specific subsets (code, English, multilingual) could provide more actionable insights.

6. **BPE dropout for robustness**: The paper references BPE dropout as a technique to make models more robust to tokenization variations. This could be implemented as a training-time augmentation in nanochat's dataloader — randomly dropping merge rules to create alternative tokenizations of the same text.

7. **Tokenizer switching for domain adaptation**: If nanochat's base model is later fine-tuned for specific domains (e.g., heavy code focus), the paper shows the tokenizer can be swapped with FVT initialization and 50B+ tokens of training. This is relevant for the mid-training and SFT stages.

### What nanochat already does right

- Uses GPT-4-style pre-tokenization (the paper's top recommendation)
- No normalization (paper recommends reversible tokenization)
- Trains tokenizer on deployment data distribution
- Uses byte-level fallback (essential for handling any input)
- Reports bytes-per-token metrics in evaluation
