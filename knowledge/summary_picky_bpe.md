# Picky BPE: Efficient Vocabulary Refinement During Tokenizer Training

**Paper:** Chizhov, Arnett, Korotkova, Yamshchikov (2024)
**Link:** https://arxiv.org/abs/2409.04599
**Code:** https://github.com/pchizhov/picky_bpe

## Problem

Standard BPE creates "intermediate" or "junk" tokens — tokens that exist only because they were stepping stones during the merge process, but are rarely used independently during tokenization. For example, to produce the token `Kentucky`, BPE must first create `entucky`, which clutters the vocabulary. These intermediate tokens:

1. **Waste vocabulary slots** — each token adds embedding parameters that could be used for more meaningful tokens.
2. **Become under-trained** — low-frequency tokens have poorly learned embeddings (low L2 norm), leading to hallucinations and exploitable "glitch tokens."
3. **Cannot be easily removed post-hoc** — post-training trimming reduces final vocabulary size unpredictably and worsens compression.

## Core Idea: Intersection over Self (IoS)

When merging tokens `x1` and `x2` into `x1+x2`, check whether either constituent is "intermediate" using:

```
IoS(x1 | x1, x2) = pair_frequency(x1, x2) / token_frequency(x1)
```

If `IoS >= threshold T` (e.g., 0.9), it means token `x1` almost always appears as part of the pair `(x1, x2)` — it's an intermediate token and can be safely removed from the vocabulary. The freed slot is then filled by the next merge, maintaining a fixed final vocabulary size.

## Algorithm

**Training step** (modified BPE):
1. Find most frequent pair `(x1, x2)`, merge into `x3 = x1+x2`, add `x3` to vocabulary.
2. Check `IoS(x1 | x1, x2)` — if `>= T`, remove `x1` from vocabulary.
3. Check `IoS(x2 | x1, x2)` — if `>= T`, remove `x2` from vocabulary.
4. Record merge and removal events in chronological order.

**Tokenization (inference):**
- Split input into symbols, then replay events (merges and removals) in training chronological order. This is critical — naive approaches (tokenize with vanilla BPE then split removed tokens) break the event order and worsen compression.

**Key properties:**
- **Universal threshold**: `T` is relative (0.6–0.9 recommended), independent of corpus/vocab size. No dataset-specific heuristics needed.
- **Second chances**: A removed token can be re-merged later if its frequency justifies it (e.g., `he` removed when `the` is formed, then re-merged as a standalone word).
- **Fixed vocab size**: Unlike post-trimming, the final vocabulary is exactly the desired size.

## Results

**Machine translation (EN-DE, DE-ET, UK-ET):**
- With vocab size 8192: Picky BPE matches or exceeds vanilla BPE on BLEU/COMET across all language pairs. Threshold 0.7 gives the best EN-DE COMET (0.434 vs 0.431 baseline).
- With larger vocabs (16K–65K): Generally comparable, with COMET improvements in most settings.

**Under-trained tokens:**
- Tokens removed by Picky BPE are predominantly low-frequency tokens with low embedding L2 norms (candidates for under-training).
- Tokens added in their place are higher-frequency with higher L2 norms — better utilized by the model.

**Compression:**
- No loss in compression (unlike post-training trimming methods). At T=0.6, compression actually *improves* by ~1% (fewer tokens needed to represent the same text).

**Token quality:**
- Removed tokens are often incomplete word fragments (`_Chicag`, `roprietary`, `omenclature`).
- Added tokens are more often word-initial and complete words (`_renovated`, `_overcoat`, `_cognition`).
- Mean token length increases slightly (e.g., 5.38 → 5.50 chars at T=0.6 for vocab 8192).

**Comparison with Unigram (SentencePiece):**
- Unigram has longer tokens and more word-initial tokens, but drastically worse compression (~14% more tokens needed).
- Picky BPE occupies a sweet spot: better token quality than vanilla BPE without Unigram's compression penalty.

## Relevance to nanochat

### Current tokenizer setup
nanochat uses a standard BPE tokenizer trained via `rustbpe` with `tiktoken` for inference (`nanochat/tokenizer.py`, `scripts/tok_train.py`). The default vocab size is 32,768. Training uses 2B characters from the pretraining corpus.

### How Picky BPE could apply

1. **Direct implementation in rustbpe**: The Picky BPE modification is a small addition to the BPE training loop — after each merge, compute IoS for both constituents and optionally remove them. The `rustbpe` library would need to be modified (it's a Rust BPE implementation). The event-order-based tokenization at inference would also need to be supported by `tiktoken` or an alternative.

2. **Practical challenge — tiktoken compatibility**: nanochat uses `tiktoken` for inference, which expects a standard BPE merge table (`mergeable_ranks`). Picky BPE's event-order tokenization (interleaving merges and removals) is fundamentally different from standard BPE inference. Two options:
   - **Option A**: Modify the inference path to use Picky BPE's event-order tokenization instead of tiktoken. This loses tiktoken's speed advantage.
   - **Option B**: After Picky BPE training, extract only the final vocabulary and use greedy or standard BPE inference. The paper notes this is suboptimal but may be acceptable at high thresholds (T=0.9 where few tokens are removed).

3. **Expected impact at nanochat scale**: nanochat trains on ~100B tokens with a 32K vocabulary. The paper's experiments used much smaller datasets, and the authors specifically note that under-trained token issues scale with the ratio of vocab size to training data. At nanochat's scale, the benefit may be modest but still worthwhile:
   - Eliminating ~2-7% of junk tokens (at T=0.9, ~677 tokens removed out of 32K; at T=0.7, ~1970 tokens).
   - Freeing those slots for meaningful word-initial tokens and complete words.
   - Potentially reducing glitch-token-induced hallucinations.

4. **Simpler alternative — post-training analysis**: Even without implementing Picky BPE, the IoS metric could be used as a diagnostic tool to analyze the current nanochat tokenizer. Computing IoS for all tokens would identify intermediate junk tokens in the existing vocabulary, providing insight into vocabulary efficiency.

5. **Interaction with the split pattern**: nanochat uses a carefully designed pre-tokenization regex (`SPLIT_PATTERN` in `tokenizer.py`) that already improves token quality for multilingual text and code. Picky BPE would complement this by cleaning up intermediate tokens that the pre-tokenization pattern can't prevent (since they arise from the BPE merge order, not pre-tokenization boundaries).

### Recommendation

The IoS-based vocabulary refinement is a clean, principled improvement to BPE with no downside on compression and potential upside on token quality. For nanochat:

- **Low-effort**: Use IoS as a diagnostic metric on the existing tokenizer to quantify junk tokens.
- **Medium-effort**: Implement Picky BPE in rustbpe training, but use Option B (standard BPE inference with the cleaned vocabulary) for tiktoken compatibility. This sacrifices the event-order guarantee but keeps the improved vocabulary.
- **High-effort**: Full Picky BPE with event-order tokenization, replacing tiktoken with a custom inference engine. Maximum correctness but significant engineering cost.

The recommended threshold is **T=0.9** (conservative, removes ~2% of tokens) or **T=0.8** (moderate, removes ~4%) based on the paper's results showing these are the safest options with consistent improvement.
