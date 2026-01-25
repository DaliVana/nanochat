"""
Model Surgery: Convert checkpoints between different model configurations.

This script allows you to create new checkpoints from existing ones with different
parameters like max_seq_len, vocab_size, num_heads, n_kv_head, n_embd, and n_layer.

For example, to expand a d8 model to d10:
    python -m scripts.surgery --source base_checkpoints/d8 --step 10000 \
        --target base_checkpoints/d10_from_d8 --depth 10

Operations supported:
- sequence_len change: No weight changes needed (rotary embeddings are recomputed)
- vocab_size change: Expand/shrink embeddings (new tokens get random init)
- n_layer (depth) change: Add/remove layers (new layers get random init)
- n_head change: Resize query projections
- n_kv_head change: Resize key/value projections and gates
- n_embd (model_dim) change: Resize all linear layers

Usage:
    python -m scripts.surgery --source <checkpoint_dir> --step <step> \
        --target <output_dir> [--depth N] [--model-dim N] [--vocab-size N] ...
"""
import os
import json
import argparse
import logging
from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch
import torch.nn as nn

from nanochat.common import setup_default_logging, get_base_dir
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint, find_last_step
from nanochat.gpt import GPT, GPTConfig, has_ve

setup_default_logging()
logger = logging.getLogger(__name__)


def compute_model_config(depth: int, aspect_ratio: int = 64, head_dim: int = 128,
                         vocab_size: int = 32768, sequence_len: int = 2048,
                         window_pattern: str = "SSSL") -> GPTConfig:
    """Compute model config from depth (like d8, d10, d20)."""
    num_layers = depth
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    num_kv_heads = num_heads  # default 1:1 GQA ratio
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int:
    """Pad vocab size to nearest multiple of pad_to for efficiency."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def resize_embedding(old_weight: torch.Tensor, new_vocab_size: int, new_embd_dim: int,
                     init_std: float = 1.0) -> torch.Tensor:
    """Resize embedding matrix [vocab_size, embd_dim] -> [new_vocab_size, new_embd_dim]."""
    old_vocab, old_embd = old_weight.shape
    new_weight = torch.empty(new_vocab_size, new_embd_dim, dtype=old_weight.dtype)
    # Initialize with random values
    torch.nn.init.normal_(new_weight, mean=0.0, std=init_std)
    # Copy over existing weights
    copy_vocab = min(old_vocab, new_vocab_size)
    copy_embd = min(old_embd, new_embd_dim)
    new_weight[:copy_vocab, :copy_embd] = old_weight[:copy_vocab, :copy_embd]
    return new_weight


def resize_linear(old_weight: torch.Tensor, new_out: int, new_in: int,
                  init_zeros: bool = False, init_uniform_scale: Optional[float] = None) -> torch.Tensor:
    """Resize linear weight matrix [out_features, in_features] -> [new_out, new_in]."""
    old_out, old_in = old_weight.shape
    new_weight = torch.zeros(new_out, new_in, dtype=old_weight.dtype)
    if init_uniform_scale is not None:
        torch.nn.init.uniform_(new_weight, -init_uniform_scale, init_uniform_scale)
    elif not init_zeros:
        # Default: use same distribution as source (estimate from weight)
        std = old_weight.std().item()
        torch.nn.init.normal_(new_weight, mean=0.0, std=std)
    # Copy over existing weights
    copy_out = min(old_out, new_out)
    copy_in = min(old_in, new_in)
    new_weight[:copy_out, :copy_in] = old_weight[:copy_out, :copy_in]
    return new_weight


def convert_checkpoint(src_config: GPTConfig, src_state: dict, tgt_config: GPTConfig) -> dict:
    """
    Convert model state_dict from source config to target config.

    Returns new state_dict compatible with target config.
    """
    new_state = {}

    src_n_embd = src_config.n_embd
    tgt_n_embd = tgt_config.n_embd
    src_n_layer = src_config.n_layer
    tgt_n_layer = tgt_config.n_layer
    src_n_head = src_config.n_head
    tgt_n_head = tgt_config.n_head
    src_n_kv_head = src_config.n_kv_head
    tgt_n_kv_head = tgt_config.n_kv_head

    src_head_dim = src_n_embd // src_n_head
    tgt_head_dim = tgt_n_embd // tgt_n_head

    src_padded_vocab = pad_vocab_size(src_config.vocab_size)
    tgt_padded_vocab = pad_vocab_size(tgt_config.vocab_size)

    # Uniform init scale: sqrt(3) * n_embd^-0.5
    tgt_init_scale = 3**0.5 * tgt_n_embd**-0.5

    logger.info(f"Converting: n_embd {src_n_embd} -> {tgt_n_embd}")
    logger.info(f"Converting: n_layer {src_n_layer} -> {tgt_n_layer}")
    logger.info(f"Converting: n_head {src_n_head} -> {tgt_n_head}")
    logger.info(f"Converting: n_kv_head {src_n_kv_head} -> {tgt_n_kv_head}")
    logger.info(f"Converting: vocab_size {src_config.vocab_size} -> {tgt_config.vocab_size}")
    logger.info(f"Converting: padded_vocab {src_padded_vocab} -> {tgt_padded_vocab}")

    # Token embedding: [padded_vocab_size, n_embd]
    wte_key = "transformer.wte.weight"
    new_state[wte_key] = resize_embedding(
        src_state[wte_key], tgt_padded_vocab, tgt_n_embd, init_std=1.0
    )
    logger.info(f"  {wte_key}: {src_state[wte_key].shape} -> {new_state[wte_key].shape}")

    # LM head: [padded_vocab_size, n_embd]
    lm_key = "lm_head.weight"
    new_state[lm_key] = resize_embedding(
        src_state[lm_key], tgt_padded_vocab, tgt_n_embd, init_std=0.001
    )
    logger.info(f"  {lm_key}: {src_state[lm_key].shape} -> {new_state[lm_key].shape}")

    # Per-layer scalars
    # resid_lambdas: [n_layer]
    old_resid = src_state.get("resid_lambdas", torch.ones(src_n_layer))
    new_resid = torch.ones(tgt_n_layer, dtype=old_resid.dtype)
    copy_layers = min(src_n_layer, tgt_n_layer)
    new_resid[:copy_layers] = old_resid[:copy_layers]
    new_state["resid_lambdas"] = new_resid
    logger.info(f"  resid_lambdas: [{src_n_layer}] -> [{tgt_n_layer}]")

    # x0_lambdas: [n_layer]
    old_x0 = src_state.get("x0_lambdas", torch.zeros(src_n_layer))
    new_x0 = torch.zeros(tgt_n_layer, dtype=old_x0.dtype)
    new_x0[:copy_layers] = old_x0[:copy_layers]
    new_state["x0_lambdas"] = new_x0
    logger.info(f"  x0_lambdas: [{src_n_layer}] -> [{tgt_n_layer}]")

    # Transformer blocks
    for tgt_layer_idx in range(tgt_n_layer):
        src_layer_idx = tgt_layer_idx if tgt_layer_idx < src_n_layer else None
        prefix = f"transformer.h.{tgt_layer_idx}"

        # Attention layers
        # c_q: [n_head * head_dim, n_embd]
        c_q_key = f"{prefix}.attn.c_q.weight"
        if src_layer_idx is not None:
            src_c_q = src_state[f"transformer.h.{src_layer_idx}.attn.c_q.weight"]
            new_state[c_q_key] = resize_linear(
                src_c_q, tgt_n_head * tgt_head_dim, tgt_n_embd,
                init_uniform_scale=tgt_init_scale
            )
        else:
            # New layer: random init
            new_state[c_q_key] = torch.empty(tgt_n_head * tgt_head_dim, tgt_n_embd)
            torch.nn.init.uniform_(new_state[c_q_key], -tgt_init_scale, tgt_init_scale)

        # c_k: [n_kv_head * head_dim, n_embd]
        c_k_key = f"{prefix}.attn.c_k.weight"
        if src_layer_idx is not None:
            src_c_k = src_state[f"transformer.h.{src_layer_idx}.attn.c_k.weight"]
            new_state[c_k_key] = resize_linear(
                src_c_k, tgt_n_kv_head * tgt_head_dim, tgt_n_embd,
                init_uniform_scale=tgt_init_scale
            )
        else:
            new_state[c_k_key] = torch.empty(tgt_n_kv_head * tgt_head_dim, tgt_n_embd)
            torch.nn.init.uniform_(new_state[c_k_key], -tgt_init_scale, tgt_init_scale)

        # c_v: [n_kv_head * head_dim, n_embd]
        c_v_key = f"{prefix}.attn.c_v.weight"
        if src_layer_idx is not None:
            src_c_v = src_state[f"transformer.h.{src_layer_idx}.attn.c_v.weight"]
            new_state[c_v_key] = resize_linear(
                src_c_v, tgt_n_kv_head * tgt_head_dim, tgt_n_embd,
                init_uniform_scale=tgt_init_scale
            )
        else:
            new_state[c_v_key] = torch.empty(tgt_n_kv_head * tgt_head_dim, tgt_n_embd)
            torch.nn.init.uniform_(new_state[c_v_key], -tgt_init_scale, tgt_init_scale)

        # c_proj: [n_embd, n_embd] (initialized to zeros)
        c_proj_key = f"{prefix}.attn.c_proj.weight"
        if src_layer_idx is not None:
            src_c_proj = src_state[f"transformer.h.{src_layer_idx}.attn.c_proj.weight"]
            new_state[c_proj_key] = resize_linear(
                src_c_proj, tgt_n_embd, tgt_n_embd, init_zeros=True
            )
        else:
            new_state[c_proj_key] = torch.zeros(tgt_n_embd, tgt_n_embd)

        # ve_gate: [32, n_kv_head] - only for layers with value embeddings
        tgt_has_ve = has_ve(tgt_layer_idx, tgt_n_layer)
        if tgt_has_ve:
            ve_gate_key = f"{prefix}.attn.ve_gate.weight"
            if src_layer_idx is not None and has_ve(src_layer_idx, src_n_layer):
                src_ve_gate = src_state[f"transformer.h.{src_layer_idx}.attn.ve_gate.weight"]
                new_state[ve_gate_key] = resize_linear(
                    src_ve_gate, 32, tgt_n_kv_head, init_zeros=True
                )
            else:
                # Initialize to zeros (neutral gate)
                new_state[ve_gate_key] = torch.zeros(32, tgt_n_kv_head)

        # MLP layers
        # c_fc: [4 * n_embd, n_embd]
        c_fc_key = f"{prefix}.mlp.c_fc.weight"
        if src_layer_idx is not None:
            src_c_fc = src_state[f"transformer.h.{src_layer_idx}.mlp.c_fc.weight"]
            new_state[c_fc_key] = resize_linear(
                src_c_fc, 4 * tgt_n_embd, tgt_n_embd,
                init_uniform_scale=tgt_init_scale
            )
        else:
            new_state[c_fc_key] = torch.empty(4 * tgt_n_embd, tgt_n_embd)
            torch.nn.init.uniform_(new_state[c_fc_key], -tgt_init_scale, tgt_init_scale)

        # c_proj (MLP): [n_embd, 4 * n_embd] (initialized to zeros)
        mlp_proj_key = f"{prefix}.mlp.c_proj.weight"
        if src_layer_idx is not None:
            src_mlp_proj = src_state[f"transformer.h.{src_layer_idx}.mlp.c_proj.weight"]
            new_state[mlp_proj_key] = resize_linear(
                src_mlp_proj, tgt_n_embd, 4 * tgt_n_embd, init_zeros=True
            )
        else:
            new_state[mlp_proj_key] = torch.zeros(tgt_n_embd, 4 * tgt_n_embd)

        if src_layer_idx is not None:
            logger.info(f"  Layer {tgt_layer_idx}: copied from source layer {src_layer_idx}")
        else:
            logger.info(f"  Layer {tgt_layer_idx}: initialized randomly (new layer)")

    # Value embeddings: [padded_vocab_size, n_kv_head * head_dim]
    tgt_kv_dim = tgt_n_kv_head * tgt_head_dim
    src_kv_dim = src_n_kv_head * src_head_dim
    for tgt_layer_idx in range(tgt_n_layer):
        if has_ve(tgt_layer_idx, tgt_n_layer):
            ve_key = f"value_embeds.{tgt_layer_idx}.weight"
            src_layer_idx = tgt_layer_idx if tgt_layer_idx < src_n_layer else None

            if src_layer_idx is not None and has_ve(src_layer_idx, src_n_layer):
                src_ve_key = f"value_embeds.{src_layer_idx}.weight"
                src_ve = src_state[src_ve_key]
                # Resize: [src_padded_vocab, src_kv_dim] -> [tgt_padded_vocab, tgt_kv_dim]
                new_ve = torch.empty(tgt_padded_vocab, tgt_kv_dim, dtype=src_ve.dtype)
                torch.nn.init.uniform_(new_ve, -tgt_init_scale, tgt_init_scale)
                copy_vocab = min(src_padded_vocab, tgt_padded_vocab)
                copy_kv = min(src_kv_dim, tgt_kv_dim)
                new_ve[:copy_vocab, :copy_kv] = src_ve[:copy_vocab, :copy_kv]
                new_state[ve_key] = new_ve
                logger.info(f"  {ve_key}: {src_ve.shape} -> {new_ve.shape}")
            else:
                # Random init for new value embedding
                new_state[ve_key] = torch.empty(tgt_padded_vocab, tgt_kv_dim)
                torch.nn.init.uniform_(new_state[ve_key], -tgt_init_scale, tgt_init_scale)
                logger.info(f"  {ve_key}: initialized randomly (new layer)")

    return new_state


def main():
    parser = argparse.ArgumentParser(description="Model surgery: convert checkpoints between configs")

    # Source checkpoint
    parser.add_argument("--source", type=str, required=True,
                        help="Source checkpoint directory (e.g., base_checkpoints/d8)")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to convert (default: latest)")

    # Target checkpoint
    parser.add_argument("--target", type=str, required=True,
                        help="Target checkpoint directory for output")
    parser.add_argument("--target-step", type=int, default=0,
                        help="Step number for output checkpoint (default: 0)")

    # Target model configuration - specify either depth OR individual params
    parser.add_argument("--depth", type=int, default=None,
                        help="Target depth (computes n_embd, n_head, etc. automatically)")
    parser.add_argument("--aspect-ratio", type=int, default=64,
                        help="Aspect ratio for computing model_dim (default: 64)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Target head dimension (default: 128)")

    # Individual overrides (used when not using --depth, or to override depth-computed values)
    parser.add_argument("--n-layer", type=int, default=None,
                        help="Target number of layers")
    parser.add_argument("--n-embd", type=int, default=None,
                        help="Target embedding dimension")
    parser.add_argument("--n-head", type=int, default=None,
                        help="Target number of attention heads")
    parser.add_argument("--n-kv-head", type=int, default=None,
                        help="Target number of KV heads (default: same as n_head)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Target vocabulary size")
    parser.add_argument("--sequence-len", type=int, default=None,
                        help="Target sequence length")
    parser.add_argument("--window-pattern", type=str, default=None,
                        help="Target sliding window pattern (e.g., 'SSSL')")

    # Options
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for loading/saving (default: cpu)")

    args = parser.parse_args()

    # Load source checkpoint
    source_dir = args.source
    if not os.path.isabs(source_dir):
        source_dir = os.path.join(get_base_dir(), source_dir)

    step = args.step
    if step is None:
        step = find_last_step(source_dir)
        logger.info(f"No step specified, using latest: {step}")

    logger.info(f"Loading source checkpoint from {source_dir} step {step}")
    model_data, _, meta_data = load_checkpoint(source_dir, step, args.device, load_optimizer=False)

    # Strip _orig_mod. prefix (torch.compile artifact)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Get source config
    src_config_dict = meta_data["model_config"]
    # Patch missing config keys for backward compatibility
    if "window_pattern" not in src_config_dict:
        src_config_dict["window_pattern"] = "L"
    src_config = GPTConfig(**src_config_dict)
    logger.info(f"Source config: {src_config}")

    # Build target config
    if args.depth is not None:
        # Compute config from depth
        tgt_config = compute_model_config(
            depth=args.depth,
            aspect_ratio=args.aspect_ratio,
            head_dim=args.head_dim,
            vocab_size=args.vocab_size or src_config.vocab_size,
            sequence_len=args.sequence_len or src_config.sequence_len,
            window_pattern=args.window_pattern or src_config.window_pattern,
        )
    else:
        # Start from source config and apply overrides
        tgt_config = GPTConfig(
            sequence_len=args.sequence_len or src_config.sequence_len,
            vocab_size=args.vocab_size or src_config.vocab_size,
            n_layer=args.n_layer or src_config.n_layer,
            n_head=args.n_head or src_config.n_head,
            n_kv_head=args.n_kv_head or args.n_head or src_config.n_kv_head,
            n_embd=args.n_embd or src_config.n_embd,
            window_pattern=args.window_pattern or src_config.window_pattern,
        )

    # Apply additional overrides even when using --depth
    if args.depth is not None:
        if args.n_layer is not None:
            tgt_config.n_layer = args.n_layer
        if args.n_embd is not None:
            tgt_config.n_embd = args.n_embd
        if args.n_head is not None:
            tgt_config.n_head = args.n_head
        if args.n_kv_head is not None:
            tgt_config.n_kv_head = args.n_kv_head

    logger.info(f"Target config: {tgt_config}")

    # Validate target config
    assert tgt_config.n_embd % tgt_config.n_head == 0, \
        f"n_embd ({tgt_config.n_embd}) must be divisible by n_head ({tgt_config.n_head})"
    assert tgt_config.n_head % tgt_config.n_kv_head == 0, \
        f"n_head ({tgt_config.n_head}) must be divisible by n_kv_head ({tgt_config.n_kv_head})"

    # Convert checkpoint
    logger.info("Converting checkpoint...")
    new_model_data = convert_checkpoint(src_config, model_data, tgt_config)

    # Prepare metadata
    new_meta_data = {
        "model_config": asdict(tgt_config),
        "surgery_source": {
            "checkpoint_dir": args.source,
            "step": step,
            "config": asdict(src_config),
        },
        "step": args.target_step,
    }

    # Save target checkpoint
    target_dir = args.target
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(get_base_dir(), target_dir)

    logger.info(f"Saving target checkpoint to {target_dir} step {args.target_step}")
    save_checkpoint(target_dir, args.target_step, new_model_data, None, new_meta_data, rank=0)

    # Print summary
    src_params = sum(p.numel() for p in model_data.values())
    tgt_params = sum(p.numel() for p in new_model_data.values())
    logger.info(f"Conversion complete!")
    logger.info(f"  Source parameters: {src_params:,}")
    logger.info(f"  Target parameters: {tgt_params:,}")
    logger.info(f"  Parameter ratio: {tgt_params / src_params:.2f}x")


if __name__ == "__main__":
    main()
