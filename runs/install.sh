#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 100
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d16_V2 --model-tag=compare_samba_ma_d16 --depth=16 --max-seq-len=1024 --device-batch-size=16 --total-batch-size=65536 --num-iterations=1000 --eval-every=50 --eval-tokens=65536 --core-metric-every=1000 --sample-every=100 --save-every=-1 --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d20_V2 --model-tag=compare_samba_ma_d20 --depth=20 --max-seq-len=1024 --device-batch-size=8 --total-batch-size=65536 --num-iterations=100 --eval-every=50 --eval-tokens=65536 --core-metric-every=100 --sample-every=100 --save-every=-1 --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d24_V2 --model-tag=compare_samba_ma_d24 --depth=24 --max-seq-len=2048 --device-batch-size=4 --total-batch-size=131072 --eval-every=50 --eval-tokens=131072 --core-metric-every=1000 --sample-every=500 --save-every=-1 --mamba-d-state=64 --sliding-window=512 --num-iterations=10000

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d32_V2 --model-tag=compare_samba_ma_d32 --depth=32 --max-seq-len=1024 --device-batch-size=4 --total-batch-size=65536--num-iterations=100 --eval-every=50 --eval-tokens=65536 --core-metric-every=100 --sample-every=100 --save-every=-1 --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d16_V2 --model-tag=compare_samba_ma_d16 --depth=16 --max-seq-len=1024 --device-batch-size=16 --total-batch-size=131072 --num-iterations=1000 --eval-every=50 --eval-tokens=131072 --core-metric-every=1000 --sample-every=100 --save-every=-1 --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d24_V2 --model-tag=compare_samba_ma_d24 --depth=24 --max-seq-len=1024 --device-batch-size=8 --total-batch-size=65536--num-iterations=100 --eval-every=50 --eval-tokens=65536 --core-metric-every=100 --sample-every=100 --save-every=-1  --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d16_V2 --model-tag=compare_samba_ma_d16 --depth=16 --max-seq-len=1024 --device-batch-size=8 --total-batch-size=65536 --num-iterations=100 --eval-every=50 --eval-tokens=65536 --core-metric-every=100 --sample-every=100 --save-every=-1 --layer-pattern=MMMA --mamba-d-state=64 --sliding-window=1024

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d24_V2 --model-tag=compare_samba_ma_d24 --depth=24 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=131072 --eval-every=50 --eval-tokens=65536 --core-metric-every=1000 --sample-every=500 --save-every=-1 --mamba-d-state=64 --sliding-window=512 --num-iterations=100

torchrun --standalone --nproc_per_node=2 -m scripts.samba_train -- --run=compare_samba_ma_d24_V2 --model-tag=compare_samba_ma_d24 --depth=32 --max-seq-len=2048 --device-batch-size=4 --total-batch-size=131072 --eval-every=50 --eval-tokens=65536 --core-metric-every=1000 --sample-every=500 --save-every=-1 --mamba-d-state=64 --sliding-window=512 --num-iterations=100