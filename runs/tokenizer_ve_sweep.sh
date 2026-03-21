#!/bin/bash

# Two additional tokenizer + value-embedding experiments:
#   1) PATTERN_B, 16K vocab, value embedding in EVERY layer  (ve_every=1)
#   2) PATTERN_B, 64K vocab, value embedding every 4th layer (ve_every=4)
#
# Usage:
#   bash runs/tokenizer_ve_sweep.sh
#   # or in a screen session:
#   screen -L -Logfile runs/tokenizer_ve_sweep.log -S tokve bash runs/tokenizer_ve_sweep.sh

set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Configuration -----------------------------------------------------------

# Pattern B: user-provided alternative pattern
PATTERN_B="[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*|0[xXbBoO][\p{N}a-fA-F]+|\p{N}{1,1}| ?[^\s\p{L}\p{N}]++[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

DEPTH=12

# --- Helper functions --------------------------------------------------------

run_experiment() {
    local run_name="$1"
    local split_pattern="$2"
    local vocab_size="$3"
    local ve_every="$4"

    echo ""
    echo "=============================================================================="
    echo "  EXPERIMENT: ${run_name}"
    echo "  Vocab size: ${vocab_size} | ve_every: ${ve_every}"
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
        --device-batch-size=16 \
        --ve-every="${ve_every}"

    echo ">>> Experiment ${run_name} complete."
}

# --- Ensure data is downloaded ------------------------------------------------

echo ">>> Ensuring dataset is downloaded..."
uv run python -m nanochat.dataset -n 170

# --- Run experiments ----------------------------------------------------------

# 1) Pattern B, 16K vocab, value embedding in every layer
run_experiment "tok-patB-16k-ve1" "$PATTERN_B" 16384 1

# 2) Pattern B, 64K vocab, value embedding every 4th layer
run_experiment "tok-patB-64k-ve4" "$PATTERN_B" 65536 4

echo ""
echo "=============================================================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=============================================================================="
