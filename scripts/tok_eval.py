"""
Evaluate compression ratio and cross-lingual fairness of the tokenizer.

Metrics:
- Compression ratio (bytes per token) across domains and languages
- Gini coefficient (cross-lingual fairness of token costs)
- Fertility (tokens per word / character)
- Vocabulary utilization (fraction of vocab used per language)
- k-gram entropies H_1..H_5 (how much local structure the tokenizer absorbs)
- Capacity utilization η = H_1/log2(V) (how efficiently the vocab is used)
- Rényi utilization η_2 (whether frequency mass concentrates on few tokens)
- Intersection-over-Self (IoS) junk token analysis (intermediate BPE fragments)

See: Parity-Aware BPE (arXiv 2508.04796) for fairness metric definitions.
See: Information-Theoretic Tokenizers (arXiv 2601.09039) for entropy metrics.
See: Picky BPE (arXiv 2409.04599) for IoS metric.
"""

import math
from collections import Counter

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Fairness metric functions
# These are pure functions with no side effects, suitable for unit testing.

def compute_gini(values):
    """
    Gini coefficient for a list of non-negative values.
    0 = perfect equality, 1 = maximum inequality.

    Formula (from Parity-Aware BPE, arXiv 2508.04796):
    Given sorted values c_1 <= c_2 <= ... <= c_n:
    Gini = (1/n)(n + 1 - 2 * sum((n+1-i)*c_i) / sum(c_i))
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
    For CJK and other scripts without word boundaries (detected by heuristic:
    avg whitespace-delimited "word" > 10 chars), falls back to tokens per character.
    """
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return 0.0
    avg_word_len = len(text) / num_words
    if avg_word_len > 10:
        # Likely CJK or similar; use character count instead
        num_chars = len(text.replace(" ", "").replace("\n", "").replace("\r", ""))
        return num_tokens / max(num_chars, 1)
    return num_tokens / num_words

def compute_vocab_utilization(token_ids, vocab_size):
    """
    Vocabulary utilization: fraction of the tokenizer's vocabulary actually used.
    |unique tokens in corpus| / vocab_size.
    """
    unique_tokens = len(set(token_ids))
    return unique_tokens / vocab_size

# -----------------------------------------------------------------------------
# Information-theoretic metric functions
# See: arXiv 2601.09039 (Information-Theoretic Perspective on LLM Tokenizers)
# See: arXiv 2409.04599 (Picky BPE) for IoS metric.

def compute_kgram_entropies(token_ids, max_k=5):
    """
    Compute k-gram Shannon entropies H_k for k=1..max_k on a token stream.

    Returns dict with:
      'raw': [H_1, H_2, ..., H_max_k]  — joint entropy of k-grams in bits
      'conditional': [H_1, H_{2|1}, ..., H_{k|k-1}]  — conditional entropies

    H_k = -sum(p(gram) * log2(p(gram))) over all observed k-grams.
    H_{k|k-1} = H_k - H_{k-1} measures residual uncertainty given (k-1) context.

    For a good tokenizer, conditional entropies H_{k|k-1} drop toward zero
    as k grows — the tokenizer absorbs short-range regularity from the text.
    """
    n = len(token_ids)
    if n == 0:
        return {'raw': [0.0] * max_k, 'conditional': [0.0] * max_k}

    raw_entropies = []
    for k in range(1, max_k + 1):
        if n < k:
            raw_entropies.append(0.0)
            continue
        # Count k-gram frequencies
        counts = Counter()
        for i in range(n - k + 1):
            gram = tuple(token_ids[i:i + k])
            counts[gram] += 1
        total = sum(counts.values())
        # Shannon entropy H_k = -sum(p * log2(p))
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        raw_entropies.append(entropy)

    # Conditional entropies: H_{k|k-1} = H_k - H_{k-1}, with H_0 = 0
    conditional = [raw_entropies[0]]  # H_{1|0} = H_1
    for k in range(1, max_k):
        conditional.append(raw_entropies[k] - raw_entropies[k - 1])

    return {'raw': raw_entropies, 'conditional': conditional}

def compute_capacity_utilization(token_ids, vocab_size):
    """
    Capacity utilization η = H_1 / log2(vocab_size).

    Measures what fraction of the tokenizer's channel capacity is actually used.
    BPE/WordPiece typically plateau at η ≈ 0.75-0.77 on English (arXiv 2601.09039).
    η = 1.0 means perfectly uniform token usage; η → 0 means a few tokens dominate.
    """
    if vocab_size <= 1 or len(token_ids) == 0:
        return 0.0
    log2_v = math.log2(vocab_size)
    # Compute H_1 (unigram entropy)
    counts = Counter(token_ids)
    total = sum(counts.values())
    h1 = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h1 / log2_v

def compute_renyi_utilization(token_ids, vocab_size, alpha=2):
    """
    Rényi utilization η_α = H_α / log2(vocab_size).

    H_α = 1/(1-α) * log2(sum(p_i^α))  (Rényi entropy of order α).

    When η_2 declines even as Shannon η increases, it indicates growing probability
    mass concentration among a few very frequent tokens while rare tokens proliferate
    in the tail (arXiv 2601.09039, Section 6).
    """
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
    """
    Intersection-over-Self (IoS) analysis for identifying junk BPE tokens.

    For each consecutive bigram (a, b) in the token stream:
      IoS(a | a,b) = pair_freq(a,b) / token_freq(a)
      IoS(b | a,b) = pair_freq(a,b) / token_freq(b)

    A token with IoS >= threshold almost always appears as part of one specific
    pair — it's likely an intermediate BPE artifact (e.g., 'entucky' in 'Kentucky').

    Returns list of (token_id, ios_score, dominant_pair) sorted by IoS descending.
    Only tokens with IoS >= threshold are included.

    See: Picky BPE (arXiv 2409.04599).
    """
    if len(token_ids) < 2:
        return []

    # Count individual token frequencies and bigram pair frequencies
    token_freq = Counter(token_ids)
    pair_freq = Counter()
    for i in range(len(token_ids) - 1):
        pair_freq[(token_ids[i], token_ids[i + 1])] += 1

    # For each token, find its max IoS across all pairs it participates in
    # token -> (max_ios, dominant_pair)
    token_max_ios = {}
    for (a, b), pf in pair_freq.items():
        # IoS for left constituent
        ios_a = pf / token_freq[a]
        if ios_a >= threshold:
            if a not in token_max_ios or ios_a > token_max_ios[a][0]:
                token_max_ios[a] = (ios_a, (a, b))
        # IoS for right constituent
        ios_b = pf / token_freq[b]
        if ios_b >= threshold:
            if b not in token_max_ios or ios_b > token_max_ios[b][0]:
                token_max_ios[b] = (ios_b, (a, b))

    # Sort by IoS descending
    results = [
        (tok_id, ios_score, pair)
        for tok_id, (ios_score, pair) in token_max_ios.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# -----------------------------------------------------------------------------
# Text samples: natural languages (for cross-lingual fairness metrics)

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# Chinese text (Simplified, educational/scientific)
chinese_text = r"""
光合作用是地球上最重要的生物化学过程之一。绿色植物利用太阳光的能量，将二氧化碳和水转化为葡萄糖和氧气。这个过程发生在植物细胞的叶绿体中，特别是在类囊体膜上。光合作用分为两个主要阶段：光反应和暗反应。在光反应阶段，叶绿素吸收光能，分解水分子，释放氧气，并产生能量载体。在暗反应阶段，也称为卡尔文循环，植物利用之前产生的能量将二氧化碳固定为有机物。光合作用不仅为植物本身提供营养，还为地球上几乎所有的生命形式提供了食物和氧气的来源。全球每年通过光合作用固定的碳约为一千亿吨，这对维持大气中的碳氧平衡至关重要。气候变化和环境污染正在影响全球光合作用的效率，科学家们正在研究如何提高作物的光合效率以应对未来的粮食安全挑战。
""".strip()

# Arabic text (Modern Standard Arabic, educational/scientific)
arabic_text = r"""
يعد التمثيل الضوئي من أهم العمليات الكيميائية الحيوية على سطح الأرض. تستخدم النباتات الخضراء طاقة ضوء الشمس لتحويل ثاني أكسيد الكربون والماء إلى جلوكوز وأكسجين. تحدث هذه العملية في البلاستيدات الخضراء داخل خلايا النبات، وتحديداً على أغشية الثايلاكويد. ينقسم التمثيل الضوئي إلى مرحلتين رئيسيتين: التفاعلات الضوئية والتفاعلات اللاضوئية. في مرحلة التفاعلات الضوئية، يمتص الكلوروفيل الطاقة الضوئية ويحلل جزيئات الماء ويطلق الأكسجين وينتج حاملات الطاقة. في مرحلة التفاعلات اللاضوئية، المعروفة أيضاً بدورة كالفن، تستخدم النباتات الطاقة المنتجة سابقاً لتثبيت ثاني أكسيد الكربون في مواد عضوية. لا يوفر التمثيل الضوئي الغذاء للنباتات فحسب، بل يوفر أيضاً الغذاء والأكسجين لجميع أشكال الحياة على الأرض تقريباً. يثبت التمثيل الضوئي عالمياً حوالي مائة مليار طن من الكربون سنوياً.
""".strip()

# Hindi text (Devanagari, educational/scientific)
hindi_text = r"""
प्रकाश संश्लेषण पृथ्वी पर सबसे महत्वपूर्ण जैव रासायनिक प्रक्रियाओं में से एक है। हरे पौधे सूर्य के प्रकाश की ऊर्जा का उपयोग करके कार्बन डाइऑक्साइड और पानी को ग्लूकोज और ऑक्सीजन में बदलते हैं। यह प्रक्रिया पौधों की कोशिकाओं में हरितलवक में होती है, विशेष रूप से थाइलेकॉइड झिल्ली पर। प्रकाश संश्लेषण दो मुख्य चरणों में विभाजित होता है: प्रकाश अभिक्रियाएँ और अंधेरी अभिक्रियाएँ। प्रकाश अभिक्रियाओं के चरण में, क्लोरोफिल प्रकाश ऊर्जा को अवशोषित करता है, पानी के अणुओं को तोड़ता है, ऑक्सीजन मुक्त करता है और ऊर्जा वाहक उत्पन्न करता है। अंधेरी अभिक्रियाओं के चरण में, जिसे केल्विन चक्र भी कहा जाता है, पौधे पहले उत्पन्न ऊर्जा का उपयोग करके कार्बन डाइऑक्साइड को कार्बनिक पदार्थों में स्थिर करते हैं। प्रकाश संश्लेषण न केवल पौधों को पोषण प्रदान करता है बल्कि पृथ्वी पर लगभग सभी जीवन रूपों के लिए भोजन और ऑक्सीजन का स्रोत भी है।
""".strip()

# Spanish text (educational/scientific)
spanish_text = r"""
La fotosíntesis es uno de los procesos bioquímicos más importantes de la Tierra. Las plantas verdes utilizan la energía de la luz solar para convertir el dióxido de carbono y el agua en glucosa y oxígeno. Este proceso ocurre en los cloroplastos de las células vegetales, específicamente en las membranas tilacoides. La fotosíntesis se divide en dos etapas principales: las reacciones luminosas y las reacciones oscuras. En la etapa de reacciones luminosas, la clorofila absorbe la energía lumínica, descompone las moléculas de agua, libera oxígeno y produce portadores de energía. En la etapa de reacciones oscuras, también conocida como el ciclo de Calvin, las plantas utilizan la energía producida anteriormente para fijar el dióxido de carbono en compuestos orgánicos. La fotosíntesis no solo proporciona nutrientes a las propias plantas, sino que también es la fuente de alimento y oxígeno para casi todas las formas de vida en la Tierra. Cada año, la fotosíntesis global fija aproximadamente cien mil millones de toneladas de carbono, lo cual es fundamental para mantener el equilibrio de carbono y oxígeno en la atmósfera.
""".strip()

# Japanese text (mixed script: Kanji + Hiragana + Katakana, educational/scientific)
japanese_text = r"""
光合成は地球上で最も重要な生化学的プロセスの一つです。緑色植物は太陽光のエネルギーを利用して、二酸化炭素と水をグルコースと酸素に変換します。このプロセスは植物細胞の葉緑体で、特にチラコイド膜上で起こります。光合成は二つの主要な段階に分けられます。明反応と暗反応です。明反応の段階では、クロロフィルが光エネルギーを吸収し、水分子を分解して酸素を放出し、エネルギー運搬体を生成します。暗反応の段階は、カルビン回路とも呼ばれ、植物は以前に生成されたエネルギーを使って二酸化炭素を有機物に固定します。光合成は植物自体に栄養を提供するだけでなく、地球上のほぼすべての生命体に食物と酸素の源を提供しています。世界全体で毎年光合成によって固定される炭素は約千億トンであり、これは大気中の炭素と酸素のバランスを維持するために極めて重要です。気候変動や環境汚染は光合成の効率に影響を与えており、科学者たちは将来の食料安全保障に対応するため作物の光合成効率を向上させる方法を研究しています。
""".strip()

# -----------------------------------------------------------------------------
# Text samples: domain-specific (not used for cross-lingual fairness metrics)

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()

# Natural language samples (used for cross-lingual fairness metrics)
language_texts = [
    ("english",  news_text),
    ("korean",   korean_text),
    ("chinese",  chinese_text),
    ("arabic",   arabic_text),
    ("hindi",    hindi_text),
    ("spanish",  spanish_text),
    ("japanese", japanese_text),
]

# Domain-specific samples (not used for cross-lingual fairness)
domain_texts = [
    ("code",    code_text),
    ("math",    math_text),
    ("science", science_text),
]

# The tokenizer was trained on data from earlier shards, so it has seen this data
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

# Combined list for compression ratio comparison (preserves existing behavior)
all_text = language_texts + domain_texts
all_text.append(("fwe-train", train_text))
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio,
            'token_ids': encoded,
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# ---------------------------------------------------------------------------
# Fairness metrics (cross-lingual)
# See: Parity-Aware BPE (arXiv 2508.04796) for metric definitions.
# Note: with only 7 languages these are directional — useful for comparing
# tokenizers against each other, not as absolute fairness measurements.
# ---------------------------------------------------------------------------
YELLOW = '\033[93m'
lang_names = [name for name, _ in language_texts]
lang_text_map = dict(language_texts)

print(f"\n\nCross-lingual Fairness Metrics")
print("=" * 95)
print("Cost = tokens/byte (lower = more efficient). Gini over per-language costs.")
print(f"Languages: {', '.join(lang_names)}")

for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    results = tokenizer_results[tokenizer_name]
    vocab_size = vocab_sizes[tokenizer_name]

    print(f"\n--- {tokenizer_name.upper()} (vocab: {vocab_size:,}) ---")
    print(f"{'Language':<12} {'Tokens':<8} {'Bytes':<8} {'Tok/Byte':<10} "
          f"{'Fertility':<12} {'Vocab Util':<12}")
    print("-" * 72)

    per_lang_costs = []
    fertilities = []
    vocab_utils = []

    for lang_name in lang_names:
        data = results[lang_name]
        text = lang_text_map[lang_name]

        tok_per_byte = data['tokens'] / data['bytes']
        fertility = compute_fertility(text, data['tokens'])
        vocab_util = compute_vocab_utilization(data['token_ids'], vocab_size)

        per_lang_costs.append(tok_per_byte)
        fertilities.append(fertility)
        vocab_utils.append(vocab_util)

        print(f"{lang_name:<12} {data['tokens']:<8} {data['bytes']:<8} "
              f"{tok_per_byte:<10.4f} {fertility:<12.2f} {vocab_util:<12.4f}")

    gini = compute_gini(per_lang_costs)
    mean_fertility = sum(fertilities) / len(fertilities)
    mean_vocab_util = sum(vocab_utils) / len(vocab_utils)

    # Color the Gini: green if < 0.05, yellow if < 0.10, red if >= 0.10
    gini_color = GREEN if gini < 0.05 else (YELLOW if gini < 0.10 else RED)

    print("-" * 72)
    print(f"{'Gini coefficient:':<34} {gini_color}{gini:.4f}{RESET}")
    print(f"{'Mean fertility:':<34} {mean_fertility:.2f}")
    print(f"{'Mean vocab utilization:':<34} {mean_vocab_util:.4f}")

# ---------------------------------------------------------------------------
# Information-theoretic metrics (on fwe-train data)
# See: arXiv 2601.09039 (Information-Theoretic Perspective on LLM Tokenizers)
# ---------------------------------------------------------------------------
CYAN = '\033[96m'
BOLD = '\033[1m'

print(f"\n\nInformation-Theoretic Metrics (on fwe-train)")
print("=" * 95)
print("H_k = joint k-gram entropy (bits). H_{k|k-1} = conditional entropy (bits).")
print("Good tokenizers: H_{k|k-1} drops toward 0 as k grows (short-range structure absorbed).")

for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    results = tokenizer_results[tokenizer_name]
    vocab_size = vocab_sizes[tokenizer_name]
    fwe_ids = results['fwe-train']['token_ids']

    entropies = compute_kgram_entropies(fwe_ids, max_k=5)
    cap_util = compute_capacity_utilization(fwe_ids, vocab_size)
    renyi_util = compute_renyi_utilization(fwe_ids, vocab_size, alpha=2)

    print(f"\n--- {tokenizer_name.upper()} (vocab: {vocab_size:,}) ---")
    print(f"{'k':<4} {'H_k (raw)':<14} {'H_{k|k-1} (cond)':<18}")
    print("-" * 36)
    for k in range(5):
        raw_h = entropies['raw'][k]
        cond_h = entropies['conditional'][k]
        print(f"{k+1:<4} {raw_h:<14.4f} {cond_h:<18.4f}")
    print("-" * 36)
    print(f"{'Capacity util η:':<34} {CYAN}{cap_util:.4f}{RESET}")
    print(f"{'Rényi util η₂:':<34} {CYAN}{renyi_util:.4f}{RESET}")

# ---------------------------------------------------------------------------
# IoS junk token analysis (on ~100M chars of train data, ours tokenizer only)
# See: Picky BPE (arXiv 2409.04599) for IoS metric definition.
# ---------------------------------------------------------------------------
IOS_TARGET_CHARS = 100_000_000
threshold=0.8

print(f"\n\nIoS Junk Token Analysis (ours tokenizer, ~{IOS_TARGET_CHARS//1_000_000}M chars, threshold={threshold})")
print("=" * 95)
print(f"Tokens with IoS >= {threshold} almost always appear as part of one specific pair — likely junk.")
# Collect ~10M characters from training shards for statistically robust IoS
ours_tokenizer = get_tokenizer()
ios_texts = []
ios_char_count = 0
for batch in parquets_iter_batched(split="train"):
    for doc in batch:
        ios_texts.append(doc)
        ios_char_count += len(doc)
    if ios_char_count >= IOS_TARGET_CHARS:
        break
ios_corpus = "\n".join(ios_texts)
ios_token_ids = ours_tokenizer.encode(ios_corpus)
print(f"Corpus: {ios_char_count:,} chars, {len(ios_token_ids):,} tokens")
ios_results = compute_ios_stats(ios_token_ids, threshold=threshold)

print(f"\nJunk tokens found: {BOLD}{len(ios_results)}{RESET} / {vocab_sizes['ours']:,} vocab")
if ios_results:
    print(f"\n{'Token ID':<10} {'IoS':<8} {'Token (hex)':<24} {'Token (text)':<20} {'Dominant Pair'}")
    print("-" * 90)
    for tok_id, ios_score, (pair_a, pair_b) in ios_results[:20]:
        # Decode token bytes for display
        try:
            tok_bytes = ours_tokenizer.decode([tok_id]).encode('utf-8')
            tok_hex = tok_bytes.hex()
            tok_text = ours_tokenizer.decode([tok_id])
            # Sanitize non-printable characters for display
            tok_display = repr(tok_text)[1:-1]  # strip outer quotes from repr
        except Exception:
            tok_hex = "???"
            tok_display = "???"
        try:
            pair_a_text = repr(ours_tokenizer.decode([pair_a]))[1:-1]
            pair_b_text = repr(ours_tokenizer.decode([pair_b]))[1:-1]
        except Exception:
            pair_a_text = str(pair_a)
            pair_b_text = str(pair_b)
        print(f"{tok_id:<10} {ios_score:<8.4f} {tok_hex:<24} {tok_display:<20} "
              f"[{pair_a_text}] + [{pair_b_text}]")
    if len(ios_results) > 20:
        print(f"  ... and {len(ios_results) - 20} more")

# Log to report
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")

# Fairness metrics
lines.append("### Cross-lingual Fairness Metrics")
lines.append("")
lines.append("Cost = tokens/byte (lower = more efficient). Gini over per-language costs.")
lines.append("")
for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    results = tokenizer_results[tokenizer_name]
    vocab_size = vocab_sizes[tokenizer_name]
    lines.append(f"#### {tokenizer_name.upper()} (vocab: {vocab_size:,})")
    lines.append("")
    lines.append("| Language | Tokens | Bytes | Tok/Byte | Fertility | Vocab Util |")
    lines.append("|----------|--------|-------|----------|-----------|------------|")
    per_lang_costs = []
    fertilities = []
    vocab_utils = []
    for lang_name in lang_names:
        data = results[lang_name]
        text = lang_text_map[lang_name]
        tok_per_byte = data['tokens'] / data['bytes']
        fertility = compute_fertility(text, data['tokens'])
        vocab_util = compute_vocab_utilization(data['token_ids'], vocab_size)
        per_lang_costs.append(tok_per_byte)
        fertilities.append(fertility)
        vocab_utils.append(vocab_util)
        lines.append(f"| {lang_name} | {data['tokens']} | {data['bytes']} | "
                     f"{tok_per_byte:.4f} | {fertility:.2f} | {vocab_util:.4f} |")
    gini = compute_gini(per_lang_costs)
    mean_fertility = sum(fertilities) / len(fertilities)
    mean_vocab_util = sum(vocab_utils) / len(vocab_utils)
    lines.append("")
    lines.append(f"**Gini**: {gini:.4f} | **Mean fertility**: {mean_fertility:.2f} | **Mean vocab util**: {mean_vocab_util:.4f}")
    lines.append("")

# Information-theoretic metrics
lines.append("### Information-Theoretic Metrics (fwe-train)")
lines.append("")
lines.append("H_k = joint k-gram entropy (bits). H_{k|k-1} = conditional entropy (bits).")
lines.append("")
for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    vocab_size = vocab_sizes[tokenizer_name]
    fwe_ids = tokenizer_results[tokenizer_name]['fwe-train']['token_ids']
    entropies = compute_kgram_entropies(fwe_ids, max_k=5)
    cap_util = compute_capacity_utilization(fwe_ids, vocab_size)
    renyi_util = compute_renyi_utilization(fwe_ids, vocab_size, alpha=2)
    lines.append(f"#### {tokenizer_name.upper()} (vocab: {vocab_size:,})")
    lines.append("")
    lines.append("| k | H_k (raw) | H_{k|k-1} (cond) |")
    lines.append("|---|-----------|-------------------|")
    for k in range(5):
        lines.append(f"| {k+1} | {entropies['raw'][k]:.4f} | {entropies['conditional'][k]:.4f} |")
    lines.append("")
    lines.append(f"**Capacity util η**: {cap_util:.4f} | **Rényi util η₂**: {renyi_util:.4f}")
    lines.append("")

# IoS analysis
lines.append("### IoS Junk Token Analysis (ours, ~10M chars, threshold=0.9)")
lines.append("")
lines.append(f"Junk tokens found: **{len(ios_results)}** / {vocab_sizes['ours']:,} vocab")
lines.append("")
if ios_results:
    lines.append("| Token ID | IoS | Token (text) | Dominant Pair |")
    lines.append("|----------|-----|--------------|---------------|")
    for tok_id, ios_score, (pair_a, pair_b) in ios_results[:20]:
        try:
            tok_display = repr(ours_tokenizer.decode([tok_id]))[1:-1]
        except Exception:
            tok_display = "???"
        try:
            pair_a_text = repr(ours_tokenizer.decode([pair_a]))[1:-1]
            pair_b_text = repr(ours_tokenizer.decode([pair_b]))[1:-1]
        except Exception:
            pair_a_text = str(pair_a)
            pair_b_text = str(pair_b)
        lines.append(f"| {tok_id} | {ios_score:.4f} | {tok_display} | [{pair_a_text}] + [{pair_b_text}] |")
    if len(ios_results) > 20:
        lines.append(f"| ... | | | {len(ios_results) - 20} more |")
    lines.append("")

report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
