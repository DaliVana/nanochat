# Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization

**Paper:** arXiv 2508.04796v1
**Authors:** Negar Foroutan, Clara Meister, Debjit Paul, Joel Niklaus, Sina Ahmadi, Antoine Bosselut, Rico Sennrich (EPFL, ETH Zurich, Niklaus.ai, University of Zurich)
**Code:** https://github.com/swiss-ai/parity-aware-bpe

---

## Problem

Standard BPE tokenizers maximize a global frequency-based objective over the entire training corpus. In multilingual settings, this inevitably favors high-resource languages: they get longer, more compressive tokens, while low-resource languages are left with shorter, fragmented tokens. This means:

1. **More tokens per text** for low-resource languages, increasing compute cost and latency.
2. **Worse morphological alignment** -- tokens break words at arbitrary byte boundaries rather than morpheme boundaries.
3. **Economic inequity** -- API services charging per-token impose a hidden "token tax" on speakers of underrepresented languages.

## Core Idea: Parity-Aware BPE

A simple modification to the BPE merge selection rule. Instead of always picking the globally most frequent bigram pair, Parity-Aware BPE uses a **max-min** criterion:

1. At each merge step k, identify the language with the **worst compression rate** so far.
2. Compute bigram pair frequencies **only from that language's portion** of the corpus.
3. Select the most frequent pair from that subset as the next merge.
4. Apply the merge **across all languages** (this is crucial -- it distinguishes the approach from simply concatenating monolingual tokenizers).

### Formal Objective

Classical BPE: `merges* = argmax_merges CR(D; tok_merges)`

Parity-Aware BPE: `merges* = argmax_merges min_l CR(l; tok_merges)`

Where CR(l; tok) is the compression rate for language l.

## Algorithm Variants

### Hybrid Parity-Aware BPE
- First K merges use the global (classical) BPE objective.
- Next J merges use the parity-aware objective.
- Allows practitioners to budget between global compression and fairness.
- Useful when some data (e.g., code) has no parallel corpus or no language label.

### Moving-Window Balancing
- Tracks the W most recently selected languages.
- Prevents the algorithm from getting "stuck" on a single language if its compression plateaus.
- A language is excluded if it was selected more than alpha * W / |L| times in the window.
- Default: W=100, alpha=2.

## Cross-lingual Compression Rate Comparison

A key subtlety: comparing CR across languages requires careful normalization. Different scripts have different bytes-per-character (e.g., ASCII vs UTF-8 CJK). The paper recommends using a **parallel corpus** (e.g., FLORES+) for computing per-language compression rates, so that normalization is by content rather than script. The parallel dev corpus can be small and separate from the training data.

## Key Results

### Intrinsic Metrics (128k vocab, 30 languages, unbalanced data)
- **Gini coefficient** (fairness): drops from 0.064 (Classical BPE) to 0.011 (Parity-Aware BPE) -- an 83% reduction in cross-lingual inequality.
- **Global compression rate**: virtually identical to Classical BPE.
- **Fertility**: reduced (fewer tokens per word on average).
- **MorphScore**: improved (better alignment with morpheme boundaries).
- **Vocabulary utilization**: more uniform across languages, especially benefiting low- and medium-resource languages.

### Extrinsic Metrics (3B param LLaMA models, 100B tokens, FineWeb2)
- Median per-language accuracy change: **+0.19 percentage points** (14 languages improved, 6 declined).
- Models trained with parity-aware tokenizers show **much more uniform perplexity** across languages.
- No statistically significant degradation on any benchmark.
- Evaluated on 13 multilingual benchmarks (Belebele, mTruthfulQA, PAWS-X, XNLI, XStoryCloze, XWinogrande, MMMLU, INCLUDE, etc.).

### Complexity
- Only O(|L|) overhead per merge step for recomputing per-language compression rates.
- Same asymptotic complexity as classical BPE.
- Drop-in replacement: no changes to the tokenization function itself, only to the learning phase.

## Relevance to nanochat

### Current nanochat tokenizer setup

nanochat uses a standard BPE tokenizer (`nanochat/tokenizer.py`) trained via `rustbpe` on ~2B characters from FineWeb-Edu, with a default vocab size of 32,768. The training is done in `scripts/tok_train.py` using `RustBPETokenizer.train_from_iterator()`. Evaluation (`scripts/tok_eval.py`) compares compression ratios against GPT-2 and GPT-4 tokenizers on English, Korean, code, math, and science text.

### How Parity-Aware BPE could apply

1. **nanochat is currently English-focused**: The training data (`parquets_iter_batched`) streams from FineWeb-Edu, which is predominantly English. The paper's approach is most relevant if nanochat is extended to multilingual training data, which the existing `tok_eval.py` already tests for (Korean text evaluation).

2. **Implementation in rustbpe**: The parity-aware modification is to the BPE *learning* algorithm, not the tokenization function. If one wanted to implement this in nanochat:
   - The `rustbpe.Tokenizer.train_from_iterator()` call in `tok_train.py` would need to be modified (or a new method added to rustbpe) that accepts language labels alongside text.
   - At each merge step, instead of computing global bigram counts, the algorithm would:
     a. Compute per-language compression on a dev set.
     b. Pick the worst-compressed language.
     c. Compute bigram counts only from that language's documents.
   - The merge is still applied globally to all documents.

3. **Hybrid approach is practical**: For nanochat's use case (primarily English + code + some multilingual), the hybrid variant makes the most sense: use classical BPE for the first ~half of merges (capturing common English/code patterns), then switch to parity-aware for the remainder (giving underrepresented languages a fairer share of the vocabulary).

4. **Small parallel dev set suffices**: The paper shows that even a small sentence-aligned development set (like FLORES+) is sufficient for the parity decision. This keeps the data requirements manageable.

5. **Evaluation improvements for tok_eval.py**: The paper introduces several useful metrics that nanochat's `tok_eval.py` could adopt:
   - **Gini coefficient** for measuring cross-lingual fairness.
   - **Fertility** (tokens per word).
   - **MorphScore** for morphological alignment.
   - **Vocabulary utilization** per language.
   - Currently `tok_eval.py` only measures bytes/token compression ratio.

6. **Vocab size consideration**: nanochat uses 32K vocab (small by modern standards). The paper tested 128K and 256K. At smaller vocab sizes, the parity-fairness trade-off might be more pronounced since there are fewer merge slots to allocate.

### Concrete experiment ideas

- **Multilingual tokenizer training**: If nanochat's training data is expanded to include multilingual FineWeb2 data, implement parity-aware BPE to ensure fair compression across languages.
- **Compression parity evaluation**: Add per-language compression rate analysis to `tok_eval.py` using a parallel corpus like FLORES+, plus Gini coefficient computation.
- **Hybrid BPE experiment**: Train a hybrid tokenizer (first 50% classical, second 50% parity-aware) and compare downstream model quality on multilingual benchmarks vs. the current English-only tokenizer.
- **Script-aware normalization**: When comparing compression across scripts, use document-level normalization on parallel text rather than raw byte counts to avoid bias from different bytes-per-character across writing systems.

## Key Takeaways

1. Parity-aware BPE is a simple, elegant modification: change *which language's statistics* drive the merge selection at each step.
2. The fairness gains are large (83% Gini reduction) while the compression/performance cost is negligible.
3. The approach is a **drop-in replacement** for classical BPE -- no architectural changes needed.
4. The hybrid variant provides a practical knob for balancing global compression vs. cross-lingual fairness.
5. For nanochat, this is most relevant when extending to multilingual training, and the evaluation metrics from the paper would be valuable additions to `tok_eval.py` regardless.
