"""
Test tokenizer fairness and information-theoretic metrics. Example run:

python -m pytest tests/test_tokenizer_metrics.py -v
"""

import math

# The metric functions are defined in scripts/tok_eval.py which runs at module level
# (loads tokenizers, downloads data on import). We duplicate the pure functions here
# for isolated testing — they are small (~10 lines each) and have no dependencies.

def compute_gini(values):
    """
    Gini coefficient for a list of non-negative values.
    0 = perfect equality, 1 = maximum inequality.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    weighted_sum = sum((n + 1 - i) * c for i, c in enumerate(sorted_vals, 1))
    return (1 / n) * (n + 1 - 2 * weighted_sum / total)

def compute_fertility(text, num_tokens):
    """
    Fertility: average number of tokens per whitespace-delimited word.
    Falls back to tokens per character for CJK-like text.
    """
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return 0.0
    avg_word_len = len(text) / num_words
    if avg_word_len > 10:
        num_chars = len(text.replace(" ", "").replace("\n", "").replace("\r", ""))
        return num_tokens / max(num_chars, 1)
    return num_tokens / num_words

def compute_vocab_utilization(token_ids, vocab_size):
    """
    Vocabulary utilization: fraction of the tokenizer's vocabulary actually used.
    """
    unique_tokens = len(set(token_ids))
    return unique_tokens / vocab_size


# =============================================================================
# Tests
# =============================================================================

class TestGini:

    def test_perfect_equality(self):
        """All equal values should give Gini = 0."""
        assert compute_gini([1, 1, 1, 1, 1]) == 0.0
        assert compute_gini([42, 42, 42]) == 0.0

    def test_high_inequality(self):
        """One value dominates, rest are near-zero."""
        gini = compute_gini([0.001, 0.001, 0.001, 1000.0])
        assert gini > 0.7

    def test_single_value(self):
        """Single value should return 0 (no inequality possible)."""
        assert compute_gini([42.0]) == 0.0

    def test_empty(self):
        """Empty list should return 0."""
        assert compute_gini([]) == 0.0

    def test_known_value(self):
        """Known Gini for [1, 2, 3, 4, 5] is 4/15 ~ 0.2667."""
        gini = compute_gini([1, 2, 3, 4, 5])
        assert abs(gini - 4/15) < 0.001

    def test_two_equal(self):
        """Two equal values."""
        assert compute_gini([5, 5]) == 0.0

    def test_two_unequal(self):
        """Two values: [1, 3]. Gini = 1/4 = 0.25."""
        gini = compute_gini([1, 3])
        assert abs(gini - 0.25) < 0.001

    def test_order_invariant(self):
        """Gini should not depend on input order."""
        assert compute_gini([1, 5, 3, 2, 4]) == compute_gini([1, 2, 3, 4, 5])

    def test_all_zeros(self):
        """All zeros should return 0 (avoid division by zero)."""
        assert compute_gini([0, 0, 0]) == 0.0


class TestFertility:

    def test_one_token_per_word(self):
        """If tokens == words, fertility should be 1.0."""
        assert compute_fertility("hello world foo", 3) == 1.0

    def test_subword_splitting(self):
        """More tokens than words means fertility > 1."""
        assert compute_fertility("hello world", 6) == 3.0

    def test_empty_text(self):
        """Empty text returns 0."""
        assert compute_fertility("", 0) == 0.0

    def test_single_word(self):
        """Single word with multiple tokens (short enough to stay in word mode)."""
        assert compute_fertility("hello", 3) == 3.0

    def test_cjk_fallback(self):
        """CJK text (no spaces between words) should use character-based fertility."""
        # 10 CJK characters, no spaces -> avg "word" length is 30 bytes >> 10
        cjk = "光合成は地球上で最も重要"
        fertility = compute_fertility(cjk, 10)
        # Should be tokens/characters, not tokens/words (would be 10/1 = 10)
        num_chars = len(cjk)
        assert abs(fertility - 10 / num_chars) < 0.01

    def test_cjk_with_spaces(self):
        """CJK text with some spaces still triggers character fallback if avg word len > 10."""
        # Each "word" here is very long (> 10 chars avg)
        text = "光合成は地球上で最も重要な生化学的プロセス 二酸化炭素と水をグルコースと酸素に変換"
        words = text.split()
        avg_word_len = len(text) / len(words)
        fertility = compute_fertility(text, 20)
        if avg_word_len > 10:
            num_chars = len(text.replace(" ", "").replace("\n", ""))
            assert abs(fertility - 20 / num_chars) < 0.01


class TestVocabUtilization:

    def test_full_utilization(self):
        """All vocab tokens used exactly once."""
        assert compute_vocab_utilization([0, 1, 2, 3], 4) == 1.0

    def test_partial_utilization(self):
        """Half of vocab used."""
        assert compute_vocab_utilization([0, 0, 1, 1], 4) == 0.5

    def test_single_token_repeated(self):
        """Only one unique token used."""
        assert compute_vocab_utilization([7, 7, 7, 7], 100) == 0.01

    def test_empty_input(self):
        """No tokens should give 0 utilization."""
        assert compute_vocab_utilization([], 100) == 0.0

    def test_all_unique(self):
        """All tokens are unique."""
        ids = list(range(50))
        assert compute_vocab_utilization(ids, 100) == 0.5

    def test_larger_than_vocab(self):
        """More unique tokens than vocab size (shouldn't happen in practice but test the math)."""
        # If somehow we got 10 unique IDs with vocab_size=5, util > 1.0
        assert compute_vocab_utilization([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5) == 2.0


# =============================================================================
# Information-theoretic metric functions (duplicated from scripts/tok_eval.py)
# =============================================================================

from collections import Counter

def compute_kgram_entropies(token_ids, max_k=5):
    """k-gram Shannon entropies H_k for k=1..max_k."""
    n = len(token_ids)
    if n == 0:
        return {'raw': [0.0] * max_k, 'conditional': [0.0] * max_k}
    raw_entropies = []
    for k in range(1, max_k + 1):
        if n < k:
            raw_entropies.append(0.0)
            continue
        counts = Counter()
        for i in range(n - k + 1):
            gram = tuple(token_ids[i:i + k])
            counts[gram] += 1
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        raw_entropies.append(entropy)
    conditional = [raw_entropies[0]]
    for k in range(1, max_k):
        conditional.append(raw_entropies[k] - raw_entropies[k - 1])
    return {'raw': raw_entropies, 'conditional': conditional}

def compute_capacity_utilization(token_ids, vocab_size):
    """Capacity utilization η = H_1 / log2(vocab_size)."""
    if vocab_size <= 1 or len(token_ids) == 0:
        return 0.0
    log2_v = math.log2(vocab_size)
    counts = Counter(token_ids)
    total = sum(counts.values())
    h1 = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h1 / log2_v

def compute_renyi_utilization(token_ids, vocab_size, alpha=2):
    """Rényi utilization η_α = H_α / log2(vocab_size)."""
    if vocab_size <= 1 or len(token_ids) == 0 or alpha == 1:
        return 0.0
    log2_v = math.log2(vocab_size)
    counts = Counter(token_ids)
    total = sum(counts.values())
    sum_p_alpha = sum((c / total) ** alpha for c in counts.values())
    if sum_p_alpha == 0:
        return 0.0
    h_alpha = (1 / (1 - alpha)) * math.log2(sum_p_alpha)
    return h_alpha / log2_v

def compute_ios_stats(token_ids, threshold=0.9):
    """IoS analysis for identifying junk BPE tokens."""
    if len(token_ids) < 2:
        return []
    token_freq = Counter(token_ids)
    pair_freq = Counter()
    for i in range(len(token_ids) - 1):
        pair_freq[(token_ids[i], token_ids[i + 1])] += 1
    token_max_ios = {}
    for (a, b), pf in pair_freq.items():
        ios_a = pf / token_freq[a]
        if ios_a >= threshold:
            if a not in token_max_ios or ios_a > token_max_ios[a][0]:
                token_max_ios[a] = (ios_a, (a, b))
        ios_b = pf / token_freq[b]
        if ios_b >= threshold:
            if b not in token_max_ios or ios_b > token_max_ios[b][0]:
                token_max_ios[b] = (ios_b, (a, b))
    results = [
        (tok_id, ios_score, pair)
        for tok_id, (ios_score, pair) in token_max_ios.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# Tests for information-theoretic metrics
# =============================================================================

class TestKgramEntropies:

    def test_uniform_unigram(self):
        """Uniform distribution over K tokens -> H_1 = log2(K)."""
        # 4 distinct tokens each appearing once: H_1 = log2(4) = 2.0
        ids = [0, 1, 2, 3]
        result = compute_kgram_entropies(ids, max_k=1)
        assert abs(result['raw'][0] - 2.0) < 0.001

    def test_single_token_zero_entropy(self):
        """All same token -> H_1 = 0."""
        ids = [7, 7, 7, 7, 7]
        result = compute_kgram_entropies(ids, max_k=3)
        assert abs(result['raw'][0]) < 0.001
        assert abs(result['raw'][1]) < 0.001
        assert abs(result['raw'][2]) < 0.001

    def test_conditional_h1_equals_raw_h1(self):
        """H_{1|0} = H_1 - H_0 = H_1 (since H_0 = 0)."""
        ids = [0, 1, 2, 3, 0, 1, 2, 3]
        result = compute_kgram_entropies(ids, max_k=3)
        assert abs(result['conditional'][0] - result['raw'][0]) < 0.001

    def test_deterministic_bigram_chain(self):
        """Deterministic chain: 0->1->0->1->... has H_{2|1} = 0."""
        # After token 0 always comes 1, after 1 always comes 0
        ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        result = compute_kgram_entropies(ids, max_k=2)
        # H_1 = log2(2) = 1.0 (two tokens equally frequent)
        assert abs(result['raw'][0] - 1.0) < 0.001
        # H_2 should also be ~1.0 (only two bigrams: (0,1) and (1,0))
        # H_{2|1} = H_2 - H_1 ≈ 0 (small boundary effect from 9 bigrams vs 10 unigrams)
        assert abs(result['conditional'][1]) < 0.02

    def test_conditional_decreases(self):
        """Conditional entropies should generally be non-increasing."""
        # Repeating pattern: 0 1 2 0 1 2 ...
        ids = [0, 1, 2] * 50
        result = compute_kgram_entropies(ids, max_k=5)
        # Each conditional should be <= the previous (approximately)
        for k in range(1, 5):
            assert result['conditional'][k] <= result['conditional'][k - 1] + 0.01

    def test_empty_input(self):
        """Empty token list returns zeros."""
        result = compute_kgram_entropies([], max_k=3)
        assert result['raw'] == [0.0, 0.0, 0.0]
        assert result['conditional'] == [0.0, 0.0, 0.0]

    def test_short_sequence(self):
        """Sequence shorter than k should return 0 for that k."""
        ids = [1, 2]
        result = compute_kgram_entropies(ids, max_k=5)
        # k=1: two distinct tokens -> H_1 = log2(2) = 1.0
        assert result['raw'][0] > 0
        # k=2: only one bigram (1,2) -> H_2 = 0 (single event, p=1)
        assert result['raw'][1] == 0.0
        # k=3,4,5: not enough tokens
        assert result['raw'][2] == 0.0
        assert result['raw'][3] == 0.0
        assert result['raw'][4] == 0.0


class TestCapacityUtilization:

    def test_uniform_distribution(self):
        """Uniform over entire vocab -> η = 1.0."""
        ids = list(range(8))
        assert abs(compute_capacity_utilization(ids, 8) - 1.0) < 0.001

    def test_single_token(self):
        """Single repeated token -> η = 0."""
        ids = [42] * 100
        assert compute_capacity_utilization(ids, 1000) == 0.0

    def test_empty(self):
        """Empty input -> 0."""
        assert compute_capacity_utilization([], 100) == 0.0

    def test_partial_usage(self):
        """Using half the vocab uniformly."""
        ids = list(range(4))  # 4 tokens out of 16
        eta = compute_capacity_utilization(ids, 16)
        # H_1 = log2(4) = 2.0, log2(16) = 4.0, η = 0.5
        assert abs(eta - 0.5) < 0.001

    def test_between_zero_and_one(self):
        """Typical case: η should be in (0, 1)."""
        ids = [0, 0, 0, 1, 1, 2]
        eta = compute_capacity_utilization(ids, 32)
        assert 0 < eta < 1


class TestRenyiUtilization:

    def test_uniform_distribution(self):
        """Uniform over entire vocab -> η_2 = 1.0."""
        ids = list(range(8))
        assert abs(compute_renyi_utilization(ids, 8, alpha=2) - 1.0) < 0.001

    def test_single_token(self):
        """Single repeated token -> η_2 = 0."""
        ids = [42] * 100
        assert compute_renyi_utilization(ids, 1000, alpha=2) == 0.0

    def test_empty(self):
        """Empty input -> 0."""
        assert compute_renyi_utilization([], 100) == 0.0

    def test_less_than_or_equal_capacity(self):
        """Rényi η_2 <= Shannon η for non-uniform distributions."""
        ids = [0, 0, 0, 0, 1, 2, 3, 4]
        eta_shannon = compute_capacity_utilization(ids, 32)
        eta_renyi = compute_renyi_utilization(ids, 32, alpha=2)
        assert eta_renyi <= eta_shannon + 0.001

    def test_alpha_one_returns_zero(self):
        """Alpha=1 is a degenerate case, returns 0."""
        ids = [0, 1, 2, 3]
        assert compute_renyi_utilization(ids, 4, alpha=1) == 0.0


class TestIosStats:

    def test_always_paired_token(self):
        """Token that only ever appears in one pair gets IoS = 1.0."""
        # Token 99 only appears before token 100
        ids = [0, 1, 99, 100, 2, 3, 99, 100, 4, 5, 99, 100]
        results = compute_ios_stats(ids, threshold=0.9)
        junk_ids = [r[0] for r in results]
        assert 99 in junk_ids
        # Token 99's IoS should be 1.0 (always followed by 100)
        for tok_id, ios, pair in results:
            if tok_id == 99:
                assert abs(ios - 1.0) < 0.001
                assert pair == (99, 100)

    def test_no_junk_in_diverse_stream(self):
        """Tokens appearing in many different pairs -> no junk (IoS is low)."""
        # Each token (0-4) appears with many different neighbors
        import random
        random.seed(42)
        ids = [random.randint(0, 4) for _ in range(500)]
        results = compute_ios_stats(ids, threshold=0.9)
        # With 5 tokens and 500 samples, each token appears ~100 times
        # in ~5 different pair contexts, so IoS should be well below 0.9
        assert len(results) == 0

    def test_empty(self):
        """Empty / single token returns empty."""
        assert compute_ios_stats([], threshold=0.9) == []
        assert compute_ios_stats([1], threshold=0.9) == []

    def test_threshold_filtering(self):
        """Tokens below threshold are excluded."""
        # Token 5 appears 10 times, but only 8 times before token 6
        ids = []
        for _ in range(8):
            ids.extend([5, 6])
        ids.extend([5, 7, 5, 8])  # 2 times token 5 not followed by 6
        # IoS(5 | 5,6) = 8/10 = 0.8
        results_high = compute_ios_stats(ids, threshold=0.9)
        results_low = compute_ios_stats(ids, threshold=0.7)
        assert all(r[0] != 5 for r in results_high)  # 0.8 < 0.9
        assert any(r[0] == 5 for r in results_low)    # 0.8 >= 0.7

    def test_sorted_by_ios_descending(self):
        """Results should be sorted by IoS score descending."""
        # Token 10 always paired (IoS=1.0), token 20 mostly paired (IoS~0.9)
        ids = []
        for _ in range(10):
            ids.extend([10, 11])  # IoS(10) = 1.0
        for _ in range(9):
            ids.extend([20, 21])  # 9 times before 21
        ids.extend([20, 22])  # 1 time before 22 -> IoS(20) = 0.9
        results = compute_ios_stats(ids, threshold=0.9)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]
