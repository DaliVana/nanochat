#!/bin/bash

# Tokenizer vocabulary size and split pattern sweep.
# Trains tokenizers at 16K, 32K, 64K, 128K, 200K, 256K vocab sizes
# for two different SPLIT_PATTERN variants, then runs d12 pretraining
# for each. Results are logged to wandb project "nanochat" with
# descriptive run names like "tok-patA-16k", "tok-patB-64k", etc.
#
# Usage:
#   bash runs/tokenizer_sweep.sh
#   # or in a screen session:
#   screen -L -Logfile runs/tokenizer_sweep.log -S toksweep bash runs/tokenizer_sweep.sh

set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Configuration -----------------------------------------------------------

VOCAB_SIZES=(16384 32768 65536 131072 200000 262144)
VOCAB_LABELS=("16k"  "32k"  "64k"  "128k"  "200k"   "256k")

# Pattern A: current nanochat pattern (number grouping {1,2})
PATTERN_A="'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

# Pattern B: user-provided alternative pattern
PATTERN_B="[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*|0[xXbBoO][\p{N}a-fA-F]+|\p{N}{1,1}| ?[^\s\p{L}\p{N}]++[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

DEPTH=16

# --- Helper functions --------------------------------------------------------

run_experiment() {
    local pattern_name="$1"
    local split_pattern="$2"
    local vocab_size="$3"
    local vocab_label="$4"

    local run_name="tok-${pattern_name}-D${DEPTH}-${vocab_label}"

    echo ""
    echo "=============================================================================="
    echo "  EXPERIMENT: ${run_name}"
    echo "  Pattern: ${pattern_name} | Vocab size: ${vocab_size} (${vocab_label})"
    echo "=============================================================================="
    echo ""

    # 1) Train tokenizer with the given split pattern
    echo ">>> Training tokenizer (vocab_size=${vocab_size})..."
    uv run python -m scripts.tok_train --vocab-size="${vocab_size}" --split-pattern="${split_pattern}"

    # 2) Evaluate tokenizer
    echo ">>> Evaluating tokenizer..."
    uv run python -m scripts.tok_eval || true

    # 3) Run pretraining
    echo ">>> Starting pretraining (d${DEPTH})..."
    uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
        --depth="${DEPTH}" \
        --run="${run_name}" \
        --model-tag="${run_name}" \
        --target-flops=5e18 \
        --device-batch-size=32

    echo ">>> Experiment ${run_name} complete."
}

# --- Ensure data is downloaded ------------------------------------------------

echo ">>> Ensuring dataset is downloaded..."
uv run python -m nanochat.dataset -n 170

# --- Run all experiments ------------------------------------------------------

# Pattern A sweep
# for i in "${!VOCAB_SIZES[@]}"; do
#     run_experiment "patA" "$PATTERN_A" "${VOCAB_SIZES[$i]}" "${VOCAB_LABELS[$i]}"
# done

# Pattern B sweep
for i in "${!VOCAB_SIZES[@]}"; do
    run_experiment "patB" "$PATTERN_B" "${VOCAB_SIZES[$i]}" "${VOCAB_LABELS[$i]}"
done

echo ""
echo "=============================================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=============================================================================="
