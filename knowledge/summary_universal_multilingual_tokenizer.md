# One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual Tokenizers

**Paper**: arXiv 2506.10766
**Authors**: Diana Abagyan, Alejandro R. Salamanca, et al. (Cohere / Cohere Labs)
**Core question**: Can training with a massively multilingual ("universal") tokenizer from the start of pretraining improve a model's ability to adapt to new languages later — even if those languages aren't in the pretraining data?

## Key Concepts

**Language Plasticity**: The capability of an LLM to quickly adapt to new language distributions post-training. The paper focuses on tokenizer design as a cheap intervention during pretraining that dramatically improves downstream plasticity.

**Universal vs. Cluster tokenizers**:
- **Universal**: A single BPE tokenizer trained on data from all 62 languages (diverse scripts and families).
- **Cluster**: BPE tokenizers trained only on the "primary" languages of a geographic cluster (European, Asian, ME-Indic).
- Both use the same BPE algorithm, same GPT-4o pretokenization regex, trained with the HuggingFace `tokenizers` library with `min_frequency=5`, on 50GB of sampled data.

## Main Findings

### 1. Universal tokenizer does NOT hurt pretraining performance
- On primary languages, Universal vs. Cluster differs by at most ~1% accuracy across Belebele, M-MMLU, and English benchmarks.
- The trend holds throughout the entire pretraining run, suggesting it would also hold in longer training.

### 2. Massive plasticity gains for expanded (new) languages
- **Continued pretraining** (primary + expanded data): Universal achieves +18.9% average win rate on expanded languages across all three clusters, with near-zero degradation on primary languages (+0.3%).
- **Targeted SFT** (expanded languages only): Universal achieves +14.6% average improvement over Cluster.
- **Fully unseen languages** (not in tokenizer OR pretraining): Universal still outperforms Cluster by up to +5% win rate on 7 extremely under-resourced languages (Afrikaans, Kazakh, Belarusian, Cantonese, Nepali, Armenian, Sinhala).

### 3. 8x faster adaptation
- Universal reaches the same expanded-language performance in 300 steps that Cluster achieves at 2500 steps — meaning 8x fewer samples needed (150K vs 1.3M).

### 4. Better than post-hoc vocabulary adaptation (CVA)
- Replacing the Cluster tokenizer with Universal after pretraining (CVA) and re-initializing new token embeddings is significantly worse than using Universal from the start.
- CVA with random init: -35.2% win rate vs Universal.
- CVA with mean init: -7% win rate vs Universal.
- **Takeaway**: It's much more effective to invest in Universal tokenizer upfront than to retrofit later.

### 5. Vocabulary size matters
- Universal tokenizer needs a large vocabulary (250K tokens) to match Cluster performance on primary languages.
- At 100K or 175K, the Universal tokenizer underperforms Cluster because the vocab budget is too tight to cover many scripts well.
- Cluster tokenizers are relatively stable across vocab sizes since they only need to cover a smaller set of scripts.

### 6. Even 0% expanded data in pretraining helps
- Universal tokenizer with 0% expanded language data in pretraining still achieves +12.8% win rate over Cluster on expanded languages.
- Adding just 5% expanded data increases this to +19.8% with no primary language degradation.

## Tokenizer Training Details

- **Algorithm**: BPE (HuggingFace `tokenizers` library)
- **Pretokenization**: GPT-4o regex (`gpt4-o200k`)
- **Vocab size**: 250K tokens (main experiments)
- **Training data**: 50GB sampled from pretraining mixture
- **Language weighting**: NOT uniform. Uses a principled scheme combining:
  1. Natural data distribution weights per language
  2. Language bucketing by script + language family
  - Within each bucket, uniform weighting across languages
  - Formula: `w_i = (w_i^data * w_i^bucket) / sum(w_n^data * w_n^bucket)`
  - English fixed at 30% proportion
- This weighting outperforms uniform weighting by +2.2% on average (Belebele, Euro cluster)

## Model/Training Setup

- 3.3B parameter decoder-only Transformer
- Parallel Attention Blocks, GQA, SwiGLU, RoPE
- 100B tokens pretraining (25K steps), batch size 512, seq len 8192
- Peak LR: 2e-2, cosine schedule with 2500 step warmup
- Continued pretraining: 10.5B additional tokens, constant LR 1e-4
- Trained on H100s using JAX-based FAX framework

## Relevance to nanochat

### Current nanochat tokenizer setup

nanochat trains a BPE tokenizer via `scripts/tok_train.py`:
- Uses `RustBPETokenizer` (rustbpe + tiktoken)
- Default vocab size: 32,768
- Trained on FineWeb-Edu (English-dominated)
- Uses the GPT-4-style split pattern with `\p{M}` support for combining marks
- The `tokenizer` branch shows active work on multilingual tokenizer training (`tok_train_finetranslations.py`) using translated data

### What nanochat could learn from this paper

1. **Vocabulary size budget**: The paper shows that universal tokenizers need ~250K vocab to avoid degrading primary language performance. nanochat uses 32K. If nanochat wanted to support multilingual, it would likely need to scale vocab significantly (at least 64K-128K for a modest set of languages, proportionally). However, for a small model (561M-2.2B params), the embedding table cost of 250K vocab would be proportionally much larger — this is a key tradeoff nanochat would need to navigate.

2. **Language weighting scheme**: The paper's language bucketing approach (grouping languages by script + family, then balancing with data availability) is directly applicable to `tok_train_finetranslations.py`. Currently that script iterates through all parquet files uniformly. Implementing the weighted sampling from Equation 1 would likely improve compression ratios and downstream performance for the multilingual tokenizer.

3. **Universal tokenizer from the start**: The paper's strongest message is: don't retrofit multilingual support later — bake it in from tokenizer training. If nanochat ever targets multilingual, the tokenizer should be trained on multilingual data from the start, even if pretraining is English-dominated.

4. **Expanded language data in pretraining**: Even 5% of training data from additional languages (reallocated from English) gives massive plasticity gains. For nanochat's ~100B token speedrun, 5% would be ~5B tokens of multilingual data — feasible if the data exists.

5. **Evaluation approach**: The paper uses LLM-as-a-Judge win rates for multilingual evaluation. nanochat's `tok_eval.py` already includes cross-lingual compression ratio metrics and Gini coefficient — these align with the paper's compression analysis. The paper confirms that compression ratio correlates with downstream performance.

6. **Pretokenization regex**: Both nanochat and this paper use the GPT-4o regex for pretokenization. nanochat's custom `SPLIT_PATTERN` already includes `\p{M}` for combining marks (better for Abugida scripts) — this is an improvement over the stock GPT-4o regex and aligns with the paper's multilingual goals.

7. **min_frequency parameter**: The paper uses `min_frequency=5` in BPE training to control minimum merge frequency. nanochat's `HuggingFaceTokenizer` uses `min_frequency=0`. The paper doesn't directly compare these, but filtering rare merges could help with multilingual vocab quality by avoiding noise tokens.

### Practical experiment ideas for nanochat

- **A/B test**: Train two tokenizers on the finetranslations data — one with uniform language weighting, one with the paper's script-family bucketing scheme — and compare compression ratios using `tok_eval.py`.
- **Vocab scaling**: Try training the multilingual tokenizer at 32K, 48K, and 64K vocab sizes to find the sweet spot where multilingual coverage doesn't degrade English compression too much for nanochat's model sizes.
- **5% multilingual pretraining**: If using a multilingual tokenizer, include 5% multilingual data in the pretraining mix (reallocated from English) to enable the plasticity benefits the paper demonstrates.
