# Egalitarian Language Representation in Language Models: It All Begins with Tokenizers

**Paper**: arXiv:2409.11501
**Authors**: Menan Velayuthan, Kengatharaiyer Sarveswaran (University of Jaffna, Sri Lanka)
**Focus**: Pre-tokenization's outsized impact on tokenizer quality for complex-script languages; Grapheme Pair Encoding (GPE)

---

## Core Thesis

The paper argues that **pre-tokenization** (the regex-based splitting step before BPE) matters far more than the choice of tokenization algorithm (BPE vs Unigram vs WordPiece) for achieving fair representation of complex-script languages like Tamil, Sinhala, and Hindi.

## Key Findings

### 1. Pre-tokenization is the Binding Constraint

The GPT-2/GPT-4/Llama 3 style regex pre-tokenizers split text into pre-tokens before BPE training. These pre-tokens define an upper bound on the longest token that can ever be learned. For complex scripts (abugida writing systems), the regex patterns used by GPT-2/4/Llama 3 split words at byte boundaries that fall in the middle of multi-codepoint grapheme clusters, creating artificially short pre-tokens and capping compression.

**Quantitative results (CRmax = maximum achievable compression ratio):**

| Tokenizer | English | Tamil | Sinhala | Hindi |
|-----------|---------|-------|---------|-------|
| GPT-2     | 5.26x   | 1.36x | 1.55x   | 1.56x |
| GPT-4     | 5.23x   | 2.13x | 2.16x   | 2.04x |
| Llama 3   | 5.23x   | 2.13x | 2.16x   | 2.04x |
| FLAN-T5   | 6.06x   | 9.21x | 6.34x   | 5.13x |
| mT5/NLLB  | 6.06x   | 9.21x | 6.34x   | 5.13x |

GPT-2's pre-tokenizer can never exceed 1.36x compression for Tamil even with unlimited training data. FLAN-T5/Gemma 2 (which use simpler whitespace-based pre-tokenization or SentencePiece) achieve much better compression on complex scripts.

### 2. Algorithm Choice is Secondary

When training BPE, Unigram, and WordPiece on 150k Tamil samples with the same pre-tokenizer:
- With GPT-2 pre-tokenization: all algorithms achieve ~1.36x CR (nearly identical, poor)
- With whitespace pre-tokenization: all algorithms achieve ~4.12-4.32x CR (nearly identical, good)

The pre-tokenizer dominates; the algorithm is secondary.

### 3. Tokenization Parity (TP)

TP measures how many tokens language A requires relative to language B. GPT-2's Tamil TP is 4.54x (i.e., Tamil requires 4.54x as many tokens as English for the same content), which directly translates to 4.54x the context window requirement and 4.54x the compute cost.

### 4. Grapheme Pair Encoding (GPE)

The paper proposes GPE: modify BPE to use **graphemes** (human-perceived characters, which may span multiple Unicode codepoints) as the atomic units instead of bytes. Algorithm:

1. Extract all unique graphemes from the training corpus
2. Initialize vocabulary with these graphemes (instead of 256 bytes)
3. Run standard BPE merging on grapheme sequences

Results on Tamil: GPE achieves 4.36x CR vs standard BPE's 4.32x. The improvement is marginal (+0.04) because with proper pre-tokenization, BPE already works reasonably well. The main benefit is representational correctness — tokens align with linguistic character boundaries.

### 5. Byte-level vs Grapheme-level Tokenizers

Comparing character-level approaches (no learned merges):
- ByT5 (UTF-8 bytes): Very poor for complex scripts (0.37x CR for Tamil)
- CANINE (UTF-32 codepoints): Near 1.0x for all languages
- Grapheme-based: Best performance (1.55x for Tamil) — compresses because multiple codepoints collapse into single graphemes

---

## Relevance to nanochat

### Current nanochat Tokenizer Design

nanochat uses a GPT-4-style BPE tokenizer (`nanochat/tokenizer.py`) with:
- **Pre-tokenizer regex** (SPLIT_PATTERN): `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`
- **Byte-level fallback**: `pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)` after the regex split
- **Training**: 2B characters from FineWeb-Edu, vocab size 32768
- **Implementation**: `rustbpe` for training, `tiktoken` for inference

### Implications and Possible Actions

1. **nanochat's pre-tokenizer is in the GPT-4/Llama 3 family.** Per this paper, the `\p{L}+` pattern in the regex handles complex scripts better than GPT-2's byte-oriented approach (GPT-2 matched `[a-zA-Z]` style patterns), but it still operates at the Unicode codepoint level, not at the grapheme level. For abugida scripts, the `\p{L}+` match will correctly group letter sequences, but the subsequent `ByteLevel` pre-tokenizer step decomposes everything into bytes, which fragments multi-codepoint graphemes.

2. **If nanochat ever targets multilingual support**, the key insight is: changing the pre-tokenization regex would have a much bigger impact than switching tokenization algorithms. Specifically:
   - Using whitespace-only or SentencePiece-style pre-tokenization would dramatically improve compression for non-Latin scripts
   - The current regex is heavily optimized for English (contractions like `'s`, `'ll`, `'ve`, `'re`)

3. **Grapheme-aware tokenization** could be integrated into nanochat by:
   - Adding a grapheme extraction step (using Python's `grapheme` library or `regex` library with `\X` pattern) before byte-level encoding
   - Initializing the BPE vocabulary with grapheme clusters instead of raw bytes
   - This would require modifications to both `rustbpe` (the Rust training library) and the pre-tokenizer pipeline in `tokenizer.py`

4. **For the current English-focused training**, this paper's findings suggest nanochat's tokenizer is already well-suited — the GPT-4-style pre-tokenizer achieves excellent compression on English (5.23x). The paper validates that nanochat's approach is a sensible default for English-centric models.

5. **Evaluation inspiration**: nanochat's `tok_eval.py` already tests compression on Korean text alongside English. The paper's metrics (Compression Ratio and Tokenization Parity) could be added as formal evaluation metrics, especially if multilingual support becomes a goal.

6. **The vocab size trade-off**: nanochat uses 32K vocab, and the paper trains with 5K vocab for experiments. The paper's finding that pre-tokenization dominates algorithm choice should hold at any vocab size, but the GPE benefit (graphemes as atomic units) could be more pronounced at smaller vocab sizes where byte-level tokens waste more capacity.

### Key Takeaway for nanochat

If nanochat were to support non-English languages in the future, the single most impactful change would be **modifying the pre-tokenization regex** (SPLIT_PATTERN in `tokenizer.py`), not changing the BPE algorithm. The current GPT-4-style regex is adequate for Latin-script languages but would need to be replaced with something more language-agnostic (like SentencePiece-style or simple whitespace splitting) for true multilingual parity.
