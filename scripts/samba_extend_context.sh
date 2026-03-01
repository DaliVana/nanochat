#!/bin/bash
# Continue-train Samba for context extension: 4096 -> 32768
# Loads pretrained weights from compare_samba_ma_d16_V3 checkpoint.
# Uses progressive schedule to gradually extend context length.
# Filters training data to documents with 4096+ tokens.
#
# Usage:
#   bash scripts/samba_extend_context.sh
#
# For distributed training:
#   Replace "uv run python" with "torchrun --nproc_per_node=8"

BASE_DIR=$(python -c "from nanochat.common import get_base_dir; print(get_base_dir())")
CHECKPOINT_DIR="$BASE_DIR/samba_checkpoints/compare_samba_ma_d16_V3"

torchrun --standalone --nproc_per_node=8 -m scripts.samba_train --  \
    --run=samba_ma_d16_V3_ctx32k \
    --model-tag=samba_ma_d16_V3_ctx32k \
    --depth=16 \
    --max-seq-len=32768 \
    --sliding-window=512 \
    --mamba-d-state=128 \
    --mamba-expand=2 \
    --mimo-rank=4 \
    --total-batch-size=524288 \
    --seq-len-schedule="4096,8192,16384,32768" \
    --batch-size-schedule="8,4,2,1" \
    --target-param-data-ratio=16 \
    --continue-from="$CHECKPOINT_DIR" \
    --min-doc-tokens=4096 \
    --mamba-matrix-lr=0.00025 \
    --mamba-scalar-lr=0.00625 \
    --warmup-ratio=0.05 \
    --warmdown-ratio=0.3 \
    --gradient-checkpointing \
    --eval-every=50 \
    --eval-tokens=524288 \
    --core-metric-every=1000 \
    --sample-every=500 \
    --save-every=-1
