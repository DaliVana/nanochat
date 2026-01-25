# CLAUDE.md - AI Assistant Guide for nanochat

This document provides guidance for AI assistants working with the nanochat codebase.

## Project Overview

**nanochat** is a full-stack ChatGPT clone by Andrej Karpathy, designed to train end-to-end on a modest budget (~$100-$1000). The project covers:

- Tokenization (BPE)
- Pretraining (base model)
- Midtraining (conversation tuning)
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning (RL)
- Inference (CLI and WebUI)

**Philosophy**: Minimal, hackable, dependency-lite. Not a configurable framework but a cohesive "strong baseline" codebase.

## Directory Structure

```
nanochat/
├── nanochat/           # Core module
│   ├── gpt.py          # GPT Transformer model (rotary embeddings, QK norm, GQA, Flash Attention 3)
│   ├── tokenizer.py    # BPE tokenizer (GPT-4 style)
│   ├── dataloader.py   # Distributed dataloaders
│   ├── dataset.py      # Parquet streaming for pretraining
│   ├── engine.py       # Inference engine with KV cache
│   ├── muon.py         # Muon optimizer for matrix parameters
│   ├── adamw.py        # AdamW optimizer for embeddings
│   ├── checkpoint_manager.py  # Save/load checkpoints
│   ├── common.py       # Utilities (DDP, logging, device detection)
│   ├── core_eval.py    # CORE metric evaluation
│   ├── execution.py    # Python REPL tool for LLM
│   └── ui.html         # Embedded ChatGPT-like web UI
├── scripts/            # Entry point scripts
│   ├── base_train.py   # Pretrain base model
│   ├── base_eval.py    # Evaluate CORE metrics
│   ├── base_loss.py    # Evaluate BPB, sample
│   ├── mid_train.py    # Midtraining on conversation tokens
│   ├── chat_sft.py     # Supervised fine-tuning
│   ├── chat_rl.py      # Reinforcement learning
│   ├── chat_eval.py    # Evaluate chat model
│   ├── chat_cli.py     # CLI chat interface
│   ├── chat_web.py     # FastAPI web server
│   ├── tok_train.py    # Train tokenizer
│   └── tok_eval.py     # Evaluate tokenizer
├── tasks/              # Evaluation benchmarks
│   ├── common.py       # Base Task class, TaskMixture, TaskSequence
│   ├── mmlu.py         # MMLU benchmark
│   ├── arc.py          # ARC (science questions)
│   ├── gsm8k.py        # Grade School Math
│   ├── humaneval.py    # Python coding tasks
│   ├── spellingbee.py  # Letter counting/spelling
│   ├── smoltalk.py     # SmolTalk conversations
│   └── customjson.py   # Custom JSONL conversations
├── runs/               # Complete pipeline shell scripts
│   ├── speedrun.sh     # ~$100, ~4 hours, d20 (561M params)
│   ├── run1000.sh      # ~$1000, ~42 hours, d32 (2.2B params)
│   ├── miniseries.sh   # Miniseries experiments
│   ├── scaling_laws.sh # Scaling experiments
│   └── runcpu.sh       # CPU/MPS demo
├── tests/              # Pytest tests
├── dev/                # Development utilities
└── pyproject.toml      # Project config, dependencies
```

## Quick Commands

```bash
# Full training pipeline (~4 hours on 8XH100)
bash runs/speedrun.sh

# Run in screen for long runs
screen -L -Logfile speedrun.log -S speedrun bash runs/speedrun.sh

# CLI chat
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI
python -m scripts.chat_web  # Opens at localhost:8000

# Tests
python -m pytest tests/ -v -s
python -m pytest tests/ -m "not slow"  # Skip slow tests

# Tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

# Dataset download
python -m nanochat.dataset -n 8  # Download 8 shards

# Report
python -m nanochat.report generate
```

## Code Conventions

### Distributed Training (DDP)

All code assumes potential DDP usage with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

Key patterns:
- Use `print0(s)` for rank-0 only printing
- Use `get_dist_info()` to get DDP rank/world size
- Use `compute_init(device_type)` and `compute_cleanup()` for setup/teardown
- Master process (rank 0) handles checkpointing and reporting

### Logging

Use the `logging` module with per-file loggers:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Processing...")
```

Colored ANSI output is configured via `ColoredFormatter` in `common.py`.

### Argument Parsing

Scripts use `argparse` with common patterns:
- `--device-type`: cuda|cpu|mps (or auto-detect)
- `--run`: wandb run name ("dummy" disables logging)
- `--device-batch-size`: Batch size per GPU
- `--total-batch-size`: Total batch (gradient accumulation auto-calculated)
- `--eval-every`, `--sample-every`, `--save-every`: Periodic actions

### Model Configuration

Use `@dataclass` for type-safe configs:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    ...
```

### Special Tokens

Defined in `nanochat/tokenizer.py`:

```python
SPECIAL_TOKENS = [
    "<|bos|>",            # Beginning of sequence
    "<|user_start|>",     # User message start
    "<|user_end|>",
    "<|assistant_start|>", # Assistant message start
    "<|assistant_end|>",
    "<|python_start|>",   # Python tool invocation
    "<|python_end|>",
    "<|output_start|>",   # Tool output
    "<|output_end|>",
]
```

### Tasks/Datasets

Tasks inherit from base `Task` class:

```python
class MyTask(Task):
    @property
    def eval_type(self):
        return 'generative'  # or 'categorical'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        return {"messages": [...]}

    def evaluate(self, problem, completion):
        return True/False  # or float score
```

Use `TaskMixture` for SFT training on multiple datasets.
Use `TaskSequence` for curriculum learning.

### Conversation Format

Tasks return conversations as dictionaries:

```python
{
    "messages": [
        {"role": "user", "content": "Question text"},
        {"role": "assistant", "content": "Answer text"}
    ]
}
```

### Memory Management

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

If OOM, reduce `--device-batch-size` (32 -> 16 -> 8 -> 4 -> 2 -> 1).

## Model Architecture

Key features in `nanochat/gpt.py`:
- Rotary embeddings (no positional embeddings)
- QK normalization for attention stability
- RMSNorm without learnable parameters
- No bias in linear layers
- `relu^2` activation in MLP
- Group-Query Attention (GQA) support
- Flash Attention 3 on Hopper+ GPUs, PyTorch SDPA fallback
- Sliding window attention via `window_pattern` (e.g., "SSSL")
- Value embedding (ResFormer style) on alternating layers

## Training Pipeline Flow

1. **Tokenizer Training** (`tok_train.py`) - Train BPE on 2B characters
2. **Pretraining** (`base_train.py`) - Train on large corpus
3. **Base Evaluation** (`base_loss.py`, `base_eval.py`) - BPB, CORE metrics
4. **Midtraining** (`mid_train.py`) - Conversation tokens, identity data
5. **SFT** (`chat_sft.py`) - Supervised fine-tuning on task mixtures
6. **RL** (`chat_rl.py`) - Optional reinforcement learning
7. **Evaluation** (`chat_eval.py`) - Benchmark evaluation
8. **Report** (`nanochat.report`) - Generate metrics table

## Optimization

Two optimizers used:
- **Muon** (`muon.py`): For matrix parameters, uses Polar Express orthogonalization
- **AdamW** (`adamw.py`): For embedding/unembedding layers

Learning rates are separate for different parameter groups.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v -s

# Skip slow tests
python -m pytest tests/ -m "not slow"
```

Tests use mocks for isolated testing of components like the inference engine.

## Environment

- **Python**: 3.10+
- **Package Manager**: `uv` (fast Python package manager)
- **Primary Target**: 8XH100 GPU node
- **Also Supported**: A100, single GPU, CPU, MPS (with limitations)

## Contributing Guidelines

From README:
- No giant configuration objects or if-then-else monsters
- Keep code minimal, readable, maximally-forkable
- Disclose any substantial LLM contributions in PRs
- Run tests before submitting

## Common Issues

1. **OOM**: Reduce `--device-batch-size`
2. **Missing shards**: Run `python -m nanochat.dataset -n <num_shards>`
3. **No CUDA**: Scripts auto-detect and fallback to CPU/MPS
4. **wandb errors**: Use `--run dummy` to disable

## Key Files to Read First

1. `nanochat/gpt.py` - Model architecture
2. `nanochat/common.py` - Utilities and patterns
3. `scripts/base_train.py` - Training loop example
4. `runs/speedrun.sh` - Full pipeline example
5. `tasks/common.py` - Task abstraction
