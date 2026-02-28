"""
Train Samba model (Mamba-2 + Sliding Window Attention). From root directory of the project, run as:

python -m scripts.samba_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.samba_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.samba_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.gpt_samba import GPTSamba as GPT, GPTConfigSamba as GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_o200k_harmony_tokenizer, compute_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain Samba model (Mamba-2 + Sliding Window Attention)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the model (total number of layers)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=32768, help="max context length")
parser.add_argument("--sliding-window", type=int, default=1024, help="sliding window size for all attention layers (Mamba handles long-range)")
# Samba-specific
parser.add_argument("--layer-pattern", type=str, default="MA", help="layer pattern: M=Mamba-2, A=Attention (e.g. 'MA', 'MMA', 'MMMA')")
parser.add_argument("--mamba-d-state", type=int, default=64, help="Mamba-2 SSM state dimension")
parser.add_argument("--mamba-d-conv", type=int, default=4, help="Mamba-2 causal convolution kernel size")
parser.add_argument("--mamba-expand", type=int, default=1, help="Mamba-2 expansion factor for inner dim")
parser.add_argument("--mamba-ngroups", type=int, default=1, help="Mamba-2 B/C group count (1=max sharing, -1=per-head)")
parser.add_argument("--mamba-chunk-size", type=int, default=128, help="Mamba-2 chunk size for SSD algorithm")
parser.add_argument("--mamba-version", type=int, default=2, choices=[2, 3], help="Mamba version: 2=Mamba-2 (SSD), 3=Mamba-3 (trapezoidal SSD + RoPE)")
parser.add_argument("--mimo-rank", type=int, default=1, help="MIMO rank for Mamba-3 (1=SISO, 2-4=rank-R shared-state MIMO, higher=more FLOPs/byte)")
parser.add_argument("--gradient-checkpointing", action="store_true", help="recompute forward during backward to save memory (slower but fits longer contexts)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=1, help="per-device batch size (sequences per micro-batch). With 32k context, 1 is usually optimal.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--mamba-matrix-lr", type=float, default=-1, help="learning rate for Mamba matrix parameters (Adam). -1 = use --unembedding-lr")
parser.add_argument("--mamba-scalar-lr", type=float, default=-1, help="learning rate for Mamba scalar parameters (A_log, dt_bias, norm biases). -1 = use scalar-lr * 0.1")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Progressive schedule
parser.add_argument("--seq-len-schedule", type=str, default=None, help="comma-separated seq lengths for progressive training (e.g. '1024,2048,4096,8192,16384,32768')")
parser.add_argument("--batch-size-schedule", type=str, default=None, help="comma-separated batch sizes for progressive training (e.g. '32,16,8,4,2,1')")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging

# Parse progressive schedule
schedule_phases = None
if args.seq_len_schedule is not None or args.batch_size_schedule is not None:
    assert args.seq_len_schedule is not None and args.batch_size_schedule is not None, \
        "--seq-len-schedule and --batch-size-schedule must both be provided"
    _seq_lens = [int(x) for x in args.seq_len_schedule.split(",")]
    _batch_sizes = [int(x) for x in args.batch_size_schedule.split(",")]
    assert len(_seq_lens) == len(_batch_sizes), \
        f"Schedule length mismatch: {len(_seq_lens)} seq_lens vs {len(_batch_sizes)} batch_sizes"
    _products = [b * t for b, t in zip(_batch_sizes, _seq_lens)]
    assert len(set(_products)) == 1, \
        f"B*T product must be constant across all phases, got: {_products}"
    schedule_phases = list(zip(_batch_sizes, _seq_lens))
    # Override to max seq len so model config handles all lengths
    args.max_seq_len = max(_seq_lens)
    args.device_batch_size = _batch_sizes[0]
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    print0(f"NOTE: Attention uses sliding window (W={args.sliding_window}).")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_o200k_harmony_tokenizer()
token_bytes = compute_token_bytes(tokenizer, device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    # Resolve ngroups=-1 to per-head (n_heads = d_inner // d_state)
    d_inner = int(args.mamba_expand * model_dim)
    mamba_n_heads = d_inner // args.mamba_d_state
    mamba_ngroups = mamba_n_heads if args.mamba_ngroups == -1 else args.mamba_ngroups
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        sliding_window=args.sliding_window,
        layer_pattern=args.layer_pattern,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_ngroups=mamba_ngroups,
        mamba_chunk_size=args.mamba_chunk_size,
        mamba_version=args.mamba_version,
        mimo_rank=args.mimo_rank,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
print0(f"Layer types: {''.join(model.layer_types)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
if args.gradient_checkpointing:
    model.gradient_checkpointing = True
    print0("Gradient checkpointing enabled (recompute forward during backward to save memory)")

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"samba_d{args.depth}" # e.g. samba_d12
checkpoint_dir = os.path.join(base_dir, "samba_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
    import torch.nn as nn

    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model
torch._dynamo.config.capture_scalar_outputs = True
# Whole-model compile: SSD forward/backward are registered as custom ops so
# torch.compile treats them as opaque nodes and fuses surrounding ops. Whole-model
# (vs per-block) eliminates Python overhead at block boundaries and enables cross-block
# fusion. dynamic=False because training shapes are fixed within each phase.
# Skip compilation on MPS — Metal shader compiler can't handle large-vocab kernels.
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)


# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer
mamba_matrix_lr = args.mamba_matrix_lr if args.mamba_matrix_lr > 0 else args.unembedding_lr
mamba_scalar_lr = args.mamba_scalar_lr * batch_lr_scale if args.mamba_scalar_lr > 0 else None
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    mamba_matrix_lr=mamba_matrix_lr * batch_lr_scale,
    mamba_scalar_lr=mamba_scalar_lr,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders

def create_train_loader(batch_size, seq_len, resume_state=None):
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, batch_size, seq_len, split="train", device=device, resume_state_dict=resume_state)

def create_val_loader(batch_size, seq_len):
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, seq_len, split="val", device=device)

def get_phase(step):
    """Return (phase_idx, device_batch_size, max_seq_len) for the given step."""
    if schedule_phases is None:
        return 0, args.device_batch_size, args.max_seq_len
    for i in range(len(phase_start_steps) - 1, -1, -1):
        if step >= phase_start_steps[i]:
            return i, schedule_phases[i][0], schedule_phases[i][1]
    return 0, schedule_phases[0][0], schedule_phases[0][1]

# -----------------------------------------------------------------------------
# Calculate training iterations and set up schedulers

assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Compute phase boundary steps (needs num_iterations)
if schedule_phases is not None:
    num_phases = len(schedule_phases)
    phase_start_steps = [round(i * num_iterations / num_phases) for i in range(num_phases)]
    print0(f"Progressive schedule: {num_phases} phases over {num_iterations} iterations")
    for i, (b, t) in enumerate(schedule_phases):
        end = phase_start_steps[i + 1] if i + 1 < num_phases else num_iterations
        print0(f"  Phase {i}: steps {phase_start_steps[i]}-{end}, B={b}, T={t}")
else:
    phase_start_steps = [0]

def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Training loop

if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Initialize current phase and dataloader
current_phase_idx, current_batch_size, current_seq_len = get_phase(step)
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = create_train_loader(current_batch_size, current_seq_len, dataloader_resume_state_dict)
build_val_loader = lambda b=current_batch_size, t=current_seq_len: create_val_loader(b, t)
x, y, dataloader_state_dict = next(train_loader)

# B*T is constant across all phases (or just the single fixed config)
tokens_per_fwdbwd = current_batch_size * current_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {current_batch_size} x {current_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    # Check for phase transition
    new_phase_idx, new_batch_size, new_seq_len = get_phase(step)
    if new_phase_idx != current_phase_idx:
        print0(f"{'=' * 80}")
        print0(f"PHASE TRANSITION: Phase {current_phase_idx} -> {new_phase_idx} | B: {current_batch_size} -> {new_batch_size}, T: {current_seq_len} -> {new_seq_len}")
        print0(f"{'=' * 80}")
        current_phase_idx = new_phase_idx
        current_batch_size = new_batch_size
        current_seq_len = new_seq_len
        # Recreate dataloader with new B, T (pass state for data continuity)
        del train_loader
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
        train_loader = create_train_loader(current_batch_size, current_seq_len, dataloader_state_dict)
        build_val_loader = lambda b=current_batch_size, t=current_seq_len: create_val_loader(b, t)
        x, y, dataloader_state_dict = next(train_loader)
        wandb_run.log({"step": step, "schedule/phase": current_phase_idx, "schedule/batch_size": current_batch_size, "schedule/seq_len": current_seq_len})

    last_step = step == num_iterations

    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        if device_type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        eval_batch_size = max(1, current_batch_size // 4)
        val_loader = create_val_loader(eval_batch_size, current_seq_len)
        eval_steps = args.eval_tokens // (eval_batch_size * current_seq_len * ddp_world_size)
        with disable_fp8(orig_model), autocast_ctx:
            val_bpb = evaluate_bpb(orig_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|startoftext|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": current_batch_size,
                "max_seq_len": current_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    phase_str = f" | phase: {current_phase_idx} (B={current_batch_size},T={current_seq_len})" if schedule_phases is not None else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch}{phase_str} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if schedule_phases is not None:
            log_data["schedule/phase"] = current_phase_idx
            log_data["schedule/batch_size"] = current_batch_size
            log_data["schedule/seq_len"] = current_seq_len
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Samba model training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish()
compute_cleanup()
