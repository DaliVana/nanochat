"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch
import tempfile
import shutil

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")
    # MoD defaults (disabled for backward compatibility, capacity=1.0 means disabled)
    if "mod_capacity" not in model_config_kwargs:
        model_config_kwargs["mod_capacity"] = 1.0  # disabled by default
    if "mod_fixed_layers_start" not in model_config_kwargs:
        model_config_kwargs["mod_fixed_layers_start"] = 5
    if "mod_fixed_layers_end" not in model_config_kwargs:
        model_config_kwargs["mod_fixed_layers_end"] = 1

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def save_checkpoint_to_wandb(wandb_run, checkpoint_dir, step, world_size=1, artifact_name="model-checkpoint"):
    """
    Save checkpoint files to wandb as an artifact.
    This should be called after save_checkpoint() has written files to disk.
    
    Args:
        wandb_run: Active wandb run object (or DummyWandb)
        checkpoint_dir: Directory where checkpoint files are saved
        step: Training step number
        world_size: Number of distributed training processes (for optimizer sharding)
        artifact_name: Base name for the wandb artifact
    """
    if not HAS_WANDB:
        log0("wandb not available, skipping checkpoint upload")
        return
    
    # Check if this is a real wandb run
    if not hasattr(wandb_run, 'log_artifact'):
        # It's a DummyWandb, skip upload
        return
    
    try:
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "step": step,
                "world_size": world_size,
            }
        )
        
        # Add model and metadata files
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        
        if os.path.exists(model_path):
            artifact.add_file(model_path, name=f"model_{step:06d}.pt")
        if os.path.exists(meta_path):
            artifact.add_file(meta_path, name=f"meta_{step:06d}.json")
        
        # Add optimizer files for all ranks
        for rank in range(world_size):
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
            if os.path.exists(optimizer_path):
                artifact.add_file(optimizer_path, name=f"optim_{step:06d}_rank{rank}.pt")
        
        # Log the artifact
        wandb_run.log_artifact(artifact)
        log0(f"Uploaded checkpoint to wandb artifact: {artifact_name}:v{step}")
    
    except Exception as e:
        log0(f"Failed to upload checkpoint to wandb: {e}")


def cleanup_old_checkpoints(checkpoint_dir, current_step, keep_last_n=1, rank=0):
    """
    Remove old checkpoint files, keeping only the last N checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        current_step: The step that was just saved
        keep_last_n: Number of recent checkpoints to keep (default: 1)
        rank: Process rank for distributed training
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    try:
        # Find all checkpoint steps
        model_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
        if not model_files:
            return
        
        # Extract step numbers
        steps = []
        for f in model_files:
            match = re.search(r"model_(\d+)\.pt$", os.path.basename(f))
            if match:
                steps.append(int(match.group(1)))
        
        # Sort steps and determine which to delete
        steps.sort()
        if len(steps) <= keep_last_n:
            return  # Not enough checkpoints to clean up
        
        steps_to_delete = steps[:-keep_last_n]  # Keep only last N
        
        # Delete old checkpoint files
        for step in steps_to_delete:
            # Delete model file (only on rank 0)
            if rank == 0:
                model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
                if os.path.exists(model_path):
                    os.remove(model_path)
                    log0(f"Deleted old checkpoint: {model_path}")
                
                # Delete metadata file
                meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            
            # Delete optimizer file (each rank deletes its own)
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
                logger.info(f"Deleted old optimizer checkpoint: {optimizer_path}")
    
    except Exception as e:
        log0(f"Warning: Failed to cleanup old checkpoints: {e}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def load_checkpoint_from_wandb(wandb_artifact_path, device, load_optimizer=False, rank=0, cache_dir=None):
    """
    Download and load checkpoint from a wandb artifact.
    
    Args:
        wandb_artifact_path: Full wandb artifact path (e.g., 'entity/project/artifact_name:version' or 'artifact_name:latest')
        device: Device to load tensors onto
        load_optimizer: Whether to load optimizer state
        rank: Rank for distributed training (for optimizer sharding)
        cache_dir: Optional directory to cache downloaded artifacts (default: temp dir)
    
    Returns:
        tuple: (model_data, optimizer_data, meta_data)
    """
    if not HAS_WANDB:
        raise ImportError("wandb is required to load checkpoints from wandb artifacts")
    
    log0(f"Downloading checkpoint from wandb artifact: {wandb_artifact_path}")
    
    # Use wandb API to download artifact
    api = wandb.Api()
    artifact = api.artifact(wandb_artifact_path)
    
    # Download to cache directory
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="wandb_checkpoint_")
    
    artifact_dir = artifact.download(root=cache_dir)
    log0(f"Downloaded artifact to: {artifact_dir}")
    
    # Extract step from artifact metadata or filename
    step = artifact.metadata.get('step')
    if step is None:
        # Try to infer from files in the artifact
        model_files = glob.glob(os.path.join(artifact_dir, "model_*.pt"))
        if model_files:
            step = int(os.path.basename(model_files[0]).split("_")[-1].split(".")[0])
        else:
            raise ValueError(f"Could not determine step from artifact {wandb_artifact_path}")
    
    # Load the model state
    model_path = os.path.join(artifact_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(artifact_dir, f"optim_{step:06d}_rank{rank}.pt")
        if os.path.exists(optimizer_path):
            optimizer_data = torch.load(optimizer_path, map_location=device)
        else:
            log0(f"Warning: Optimizer file not found: {optimizer_path}")
    
    # Load the metadata
    meta_path = os.path.join(artifact_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
