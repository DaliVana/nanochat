# Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training

**Paper:** arXiv 2508.15390v2 (NeurIPS 2025)
**Authors:** Woojin Chung, Jeonghoon Kim (KAIST / NAVER Cloud)
**Code:** https://github.com/Chung-Kim/vocab-imbalance

## Core Thesis

Larger vocabularies improve language model performance primarily by **reducing the Kolmogorov complexity of tokenized text** and **sharpening token-frequency imbalance**, which lets the model focus optimization capacity on the ~2,500 most frequent words. These frequent words dominate ~75% of tokens in both training data and downstream benchmarks.

## Key Findings

### 1. Larger Vocabularies Reduce Tokenized Text Complexity

Using an upper bound on Kolmogorov complexity (`K(X^N) ~ N * H(p)` where N is token count, H(p) is unigram Shannon entropy), the authors show that increasing vocab from 24K to 196K reduces the complexity of a 46B-byte FineWeb-Edu corpus:

| Vocab Size | K(X^N) | NCR (Normalized Compression Ratio) |
|------------|--------|------|
| 24K | 10.74B | 0.234 |
| 49K | 10.43B | 0.227 |
| 98K | 10.23B | 0.223 |
| 196K | 10.16B | 0.221 |

Lower complexity means the tokenized text has simpler patterns for the model to learn.

### 2. Segmentation Saturates Early, Imbalance Keeps Growing

Beyond ~24K vocab, all common words are already single tokens. Further vocabulary growth doesn't improve segmentation efficiency but **amplifies token-frequency imbalance** (measured by Jensen-Shannon divergence from uniform). The top 2,500 words cover ~74-76% of tokens in both FineWeb-Edu and OpenWebText.

### 3. Loss Decomposition: Frequent Words Drive Global CE Reduction

Expanding vocabulary from 24K to 196K:
- **Frequent word loss (top 2,500):** drops by ~0.1 nats
- **Rare word loss (bottom 20,000):** rises from ~11.2 to ~13.4 nats
- **Global cross-entropy:** drops from ~3.179 to ~3.136 nats

The gains on frequent words outweigh the degradation on rare words because frequent words dominate ~75% of the total loss.

### 4. Frequent-Word Overlap Explains Downstream Transfer

The most frequent 2,500 words in FineWeb-Edu cover 72-78% of tokens in downstream benchmarks (ARC, HellaSwag, SciQ, PIQA). Reducing frequent-word loss during pretraining directly translates to better downstream accuracy.

### 5. Model Scaling Replicates Vocabulary Scaling Benefits

Increasing model parameters (Pythia 160M -> 1B -> 6.9B) achieves the same effect: it primarily reduces loss on frequent words. Unlike vocab scaling, model scaling does NOT increase rare-token loss -- it benefits both, but disproportionately helps frequent tokens.

### 6. SuperBPE: Reducing Complexity Without More Imbalance

SuperBPE (cross-word-boundary merges after threshold t) achieves lower tokenized text complexity than standard BPE at equal vocab size, while avoiding additional frequency imbalance. The SuperBPE variant with lowest complexity had the best downstream performance.

### 7. Effects Are Robust

- Hold across dataset quality levels (FineWeb-Edu vs OpenWebText)
- Hold across model scales (85M, 450M)
- Hold across data scales (10B, 30B tokens)
- Hold with both tied and untied embeddings
- Consistent across multiple learning rates

## Experimental Setup

- **Model:** 85M non-embedding params, pre-LN Transformer, 12 layers, 12 heads
- **Optimizer:** AdamW (beta1=0.9, beta2=0.95, eps=1e-8), LR=6e-4, cosine decay, 350M-token warmup
- **Data:** ~40B characters (~7.5B tokens for 49K vocab) from FineWeb-Edu and OpenWebText
- **BPE:** Standard BPE tokenizer, vocabulary sizes: 24K, 49K, 98K, 196K
- **All experiments:** 5 random seeds

## Relevance to nanochat

### Direct Connections

1. **nanochat's current vocab_size = 32,768 (32K):** This falls in the "already saturated" range for segmentation efficiency. According to this paper, all common words are single tokens by 24K. nanochat's 32K vocab is already well past the segmentation saturation point, meaning the benefit is coming from frequency imbalance exploitation, not from better segmentation.

2. **Vocabulary size as a tuning knob:** The paper suggests that nanochat could experiment with larger vocabularies (49K, 64K, 98K) as a "free" improvement lever. The tradeoff: larger vocab = larger embedding tables (more params, more memory), but better compression and lower cross-entropy. For nanochat's small models (~561M for d20, ~2.2B for d32), the embedding table size increase matters more proportionally.

3. **Pre-tokenization pattern matters:** nanochat already has a sophisticated SPLIT_PATTERN (GPT-4 style with Unicode mark support). The paper's discussion of SuperBPE (cross-word-boundary merges) suggests that relaxing pre-tokenization rules after a certain vocab threshold could further reduce tokenized text complexity without additional frequency imbalance. This is a concrete experiment nanochat could try.

4. **Untied embeddings are fine:** nanochat uses untied embeddings (`gpt.py` line 7: "untied weights for token embedding and lm_head"). The paper confirms tied vs untied shows nearly identical behavior for the frequency-imbalance effect, so nanochat's design choice doesn't interact negatively with these dynamics.

5. **Bits-per-byte is the right metric:** nanochat already computes token_bytes for BPB evaluation (`tok_train.py` lines 72-91), which is invariant to vocab size. The paper reinforces that raw cross-entropy comparisons across different vocab sizes are misleading -- nanochat's BPB approach is the correct way to compare.

### Things to Try in nanochat

1. **Vocab size sweep:** Train tokenizers at 24K, 32K (current), 49K, 64K, 98K and compare downstream performance. The paper predicts monotonic improvement, but nanochat's smaller model size may hit diminishing returns earlier due to embedding table overhead.

2. **SuperBPE-style training:** Modify `tok_train.py` to use a two-stage BPE: standard BPE up to some threshold (e.g., 24K), then disable whitespace pre-tokenization for remaining merges. This could reduce tokenized text complexity without the frequency imbalance penalty on rare tokens.

3. **Loss decomposition analysis:** Add a word-level loss decomposition evaluation script to measure how nanochat's loss distributes across frequent vs rare tokens. This would help diagnose whether larger vocab sizes are actually helping the same way the paper predicts.

4. **Compression ratio as a proxy metric:** The paper shows Kolmogorov complexity (approximated by `N * H(p)`) correlates with downstream performance. nanochat's `tok_eval.py` already measures compression ratios; adding the `N * H(p)` metric would give a quick, training-free signal for tokenizer quality.

5. **Current branch work (tokenizer branch):** The `tok_train_finetranslations.py` on the current branch trains on multilingual data. The paper's finding that frequent words overlap ~75% across datasets suggests that a tokenizer trained on multilingual data should still exhibit the same frequency-imbalance dynamics. However, the multilingual data will have more diverse frequent words, potentially spreading the optimization benefit across more languages -- which is exactly what the current branch seems to be exploring.

### Key Insight for nanochat's Scale

The paper's most actionable insight for nanochat: at nanochat's model sizes (85M-2.2B non-embedding), **model scaling provides strictly better benefits than vocab scaling** because it reduces frequent-word loss WITHOUT increasing rare-token loss. Vocab scaling always comes with a rare-token tax. So for a fixed compute budget, investing in more model parameters is likely better than a larger vocabulary. However, vocab scaling is essentially "free" compute -- it only costs embedding memory -- so doing both is optimal if memory allows.
