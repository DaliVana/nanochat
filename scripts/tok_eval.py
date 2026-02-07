"""
Evaluate compression ratio and cross-lingual fairness of the tokenizer.

Metrics:
- Compression ratio (bytes per token) across domains and languages
- Gini coefficient (cross-lingual fairness of token costs)
- Fertility (tokens per word / character)
- Vocabulary utilization (fraction of vocab used per language)

See: Parity-Aware BPE (arXiv 2508.04796) for metric definitions.
"""

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

report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
