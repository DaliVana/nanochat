# A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning

**Paper:** arxiv 2507.08267v1
**Authors:** Hiroshi Yoshihara (Aillis Inc./UTokyo), Taiki Yamaguchi (Rist Inc.), Yuichi Inoue (Sakana AI)
**Venue:** ICML 2025
**Code:** https://github.com/analokmaus/kaggle-aimo2-fast-math-r1

## Summary

This paper proposes a two-stage training recipe that combines extended Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to build mathematical reasoning LLMs that are both accurate and token-efficient.

### Key Insight

**SFT and RL play complementary, not competing, roles:**
- **SFT (Stage 1):** Pushes model accuracy to its limits through extended training
- **GRPO (Stage 2):** Dramatically improves token efficiency while preserving peak performance

This is a shift from prior work that often viewed accuracy and efficiency as a direct trade-off to be optimized jointly.

## Method Details

### Stage 1: Extended Supervised Fine-Tuning

**Dataset Construction:**
- ~7,900 high-difficulty math problem-solution pairs from:
  - OpenR1 Math: ~3,000 examples where R1 solution traces > 12,800 tokens and accuracy > 50%
  - OpenR1 Math: ~3,000 examples where accuracy was 50-75%
  - openr1 hard: ~2,500 problems that DeepSeek-R1-Distill-Qwen-32B couldn't solve in 4 attempts
  - Light-R1-SFT second-stage data
- For duplicates: selected the **shortest correct generation**

**Training Configuration:**
- 8x H200 GPUs
- Full-parameter SFT (not LoRA/QLoRA)
- 10 epochs (this is crucial - not 1-3 epochs)
- Batch size: 1 per device, gradient accumulation: 8
- Max sequence length: 24,000 tokens
- Learning rate: 1e-5 with cosine scheduler
- Packing enabled
- System prompt: "Please reason step by step, and put your final answer within \boxed{{}}"

**Critical Finding:** Training for only 1 epoch increases token length but drops accuracy sharply. Extended training (10 epochs) is essential for performance breakthroughs, even though early epochs may show temporary dips.

### Stage 2: GRPO for Token Efficiency

**Dataset:** Same Light-R1-SFT second-stage data used in SFT

**Reward Function (3 components):**
1. **Format Reward:** Binary (+1 or 0) based on matching regex pattern `r"^.*?oxed\{(.*?)\}.*?</think>.*?$"`
2. **Cosine Similarity Reward:** Measures similarity between generated trace and reference trace
   - Correct answers: scaled 0.1 to 1.0 (penalizes longer correct traces)
   - Incorrect answers: scaled -1.0 to -0.1 (penalizes shorter incorrect traces)
   - Max trace length: 30,000 tokens
3. **Length Penalty:** Explicitly penalizes verbose solutions

**Training Configuration:**
- num_generations: 8 (rollouts per example)
- beta: 0.04
- Batch size: 2 per device, gradient accumulation: 8
- 50 steps only
- Max completion length: 16,384
- Learning rate: 4e-6 with cosine scheduler

## Results

### AIME 2024/2025 Performance (14B model)

| Method | AIME 2024 Acc | AIME 2024 Tokens | AIME 2025 Acc | AIME 2025 Tokens |
|--------|---------------|------------------|---------------|------------------|
| Original | 63.3% | 9,590 | 46.7% | 10,602 |
| Original + RL | 60.9% | 7,255 | 41.8% | 8,246 |
| + SFT (10 epochs) | 65.2% | 10,268 | 49.7% | 11,264 |
| **+ SFT + RL** | **66.0%** | **7,932** | **49.2%** | **9,066** |

Key observations:
- RL alone on original model: improves efficiency but **drops accuracy**
- SFT alone: improves accuracy but increases token count
- SFT + RL: improves accuracy AND efficiency

### MATH-500 Performance (14B model)

| Method | Accuracy | Tokens |
|--------|----------|--------|
| Original | 86.4% | 2,556 |
| + SFT (10 epochs) | 88.1% | 2,969 |
| + SFT + RL | **91.2%** | **2,084** |

### AIMO Competition

Achieved 29/50 on public set (4th place) and 28/50 on private set (8th place) out of 2,212 teams.

## Relevance to nanochat

The nanochat project already implements both SFT (`scripts/chat_sft.py`) and RL (`scripts/chat_rl.py`), making this paper directly applicable.

### Current nanochat SFT Implementation

**Similarities:**
- Uses `TaskMixture` for combining datasets
- Supports multi-epoch training via `--num-epochs`
- Uses Muon optimizer for matrix parameters, AdamW for embeddings

**Key Differences from Paper's Recipe:**

1. **Epoch Count:** nanochat defaults to `--num-epochs=1`. The paper shows 10 epochs is crucial for math reasoning.

2. **Dataset Composition:** nanochat's SFT mixture includes:
   - ARC-Easy/Challenge (~3.4K)
   - GSM8K (~8K)
   - SmolTalk (~10K)
   - Identity conversations (~1K)
   - Spelling tasks (~600)

   The paper uses exclusively high-difficulty math problems with long reasoning traces.

3. **Sequence Length:** nanochat doesn't explicitly set max sequence length. Paper uses 24,000 tokens.

4. **Data Selection Strategy:** Paper specifically selects:
   - Problems where model accuracy is 50-75% (not too easy, not too hard)
   - The shortest correct solution for each problem

   This "curriculum" approach isn't currently in nanochat.

### Current nanochat RL Implementation

**Similarities:**
- Uses GRPO-style policy gradient (simplified REINFORCE)
- Operates on GSM8K
- Uses advantage normalization (subtracting mean)

**Key Differences from Paper's Recipe:**

1. **Reward Function:** nanochat uses just the task's `reward()` method (typically binary correctness). Paper uses:
   - Format reward
   - Cosine similarity reward (nuanced accuracy signal)
   - Explicit length penalty

2. **Training Duration:** Paper trains for only 50 steps with much larger batch (8 rollouts per example).

3. **No Trust Region/KL:** Both drop KL regularization, but paper explicitly mentions this design choice.

### Concrete Improvements for nanochat

Based on this paper, the following changes could improve nanochat's math reasoning:

#### 1. Extended SFT Training
```python
# In scripts/chat_sft.py, consider:
parser.add_argument("--num-epochs", type=int, default=10, ...)  # Was 1
```

#### 2. Add Length Penalty to RL Rewards
```python
# In scripts/chat_rl.py, modify reward calculation:
def calculate_reward(conversation, generated_text, sequence_length, max_length=16384):
    base_reward = train_task.reward(conversation, generated_text)
    # Add length penalty as in the paper
    length_penalty = sequence_length / max_length  # 0 to 1
    return base_reward - 0.1 * length_penalty  # tune coefficient
```

#### 3. Add Cosine Similarity Reward
The paper's cosine reward provides a continuous signal based on similarity to reference solutions. This could be implemented using sentence embeddings:
```python
# Pseudocode for cosine reward
def cosine_reward(generated_trace, reference_trace, is_correct):
    similarity = cosine_similarity(embed(generated_trace), embed(reference_trace))
    if is_correct:
        return 0.1 + 0.9 * similarity  # 0.1 to 1.0
    else:
        return -1.0 + 0.9 * similarity  # -1.0 to -0.1
```

#### 4. Curate Difficulty-Appropriate Data
Create a "hard math" dataset by:
- Running the current model on math problems
- Keeping problems where pass rate is 25-75%
- Selecting shortest correct solutions

#### 5. Shorter RL Phase
The paper shows that only 50 RL steps are needed after extended SFT. The current nanochat RL trains for a full epoch over GSM8K.

### Task Implementation Considerations

For nanochat tasks (`tasks/gsm8k.py`, etc.), consider adding:

1. **Reference solution storage** for cosine similarity rewards
2. **Format checking** to ensure boxed answer format
3. **Length tracking** in the reward calculation

### Architectural Notes

The paper uses DeepSeek-R1-Distill-Qwen models (1.5B, 7B, 14B). nanochat's GPT architecture (`nanochat/gpt.py`) is different but the training recipe principles should transfer:
- Extended SFT helps regardless of architecture
- Token efficiency through RL is model-agnostic
- The complementary nature of SFT→RL is a training methodology insight

## Key Takeaways for nanochat

1. **Don't stop SFT early.** 10 epochs of SFT on hard problems is better than 1-3 epochs, even if early epochs show dips.

2. **RL is for efficiency, not accuracy.** After good SFT, RL's main role is reducing token count while preserving performance.

3. **Order matters.** SFT first (for accuracy ceiling), then RL (for efficiency). Not the reverse.

4. **Length penalties work.** Adding explicit length penalty to RL rewards successfully shortens generations without hurting accuracy.

5. **Hard data matters.** Focus SFT on problems the model finds challenging (50-75% accuracy range), not easy ones.

6. **Short RL is sufficient.** After proper SFT, only 50 RL steps can achieve the efficiency gains.
