"""
Test tokenizer fairness metrics. Example run:

python -m pytest tests/test_tokenizer_metrics.py -v
"""

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
