"""
Analyze a trained model checkpoint to identify undertrained tokens.

Computes multiple signals:
  A. Embedding norm analysis (wte + lm_head norms vs initialization)
  B. Cosine similarity to mean embedding
  C. KNN self-similarity (average cosine sim to K nearest neighbors)
  D. Per-token loss on validation data (optional, needs data)
  E. Token frequency in validation data (from D)

Usage:
  uv run python -m scripts.tok_analysis --source base --eval-tokens 0 --top-k 50
  uv run python -m scripts.tok_analysis --source base --eval-tokens 1000000 --output analysis.json
"""

import os
import json
import math
import argparse
import torch
import torch.nn.functional as F

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0, setup_default_logging, get_base_dir
from nanochat.checkpoint_manager import load_checkpoint, find_largest_model, find_last_step
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes, HuggingFaceTokenizer

setup_default_logging()

# -----------------------------------------------------------------------------
# Helpers

def safe_token_repr(tokenizer, token_id):
    """Return a safe printable representation of a token."""
    if tokenizer is None:
        return f"[{token_id}]"
    try:
        s = tokenizer.decode([token_id])
        if not s.isprintable() or not s.strip():
            return repr(s)
        return s
    except Exception:
        return f"[{token_id}]"


def normalize_01(x):
    """Normalize tensor to [0, 1] range, handling NaN."""
    valid = ~torch.isnan(x)
    if valid.sum() < 2:
        return torch.zeros_like(x)
    x_min = x[valid].min()
    x_max = x[valid].max()
    if x_max == x_min:
        return torch.zeros_like(x)
    result = (x - x_min) / (x_max - x_min)
    result[~valid] = float('nan')
    return result

# -----------------------------------------------------------------------------
# Analysis A: Embedding norm analysis

def analyze_embedding_norms(model, vocab_size, n_embd):
    """Compute L2 norms of wte and lm_head rows, and distance from initialization."""
    wte_weight = model.transformer.wte.weight[:vocab_size].float()
    lm_head_weight = model.lm_head.weight[:vocab_size].float()

    wte_norms = torch.norm(wte_weight, dim=1)
    lm_head_norms = torch.norm(lm_head_weight, dim=1)

    # Expected norms at initialization
    # wte: Normal(0, 0.8) => expected L2 norm ≈ 0.8 * sqrt(n_embd)
    # lm_head: Normal(0, 0.001) => expected L2 norm ≈ 0.001 * sqrt(n_embd)
    wte_init_norm = 0.8 * math.sqrt(n_embd)
    lm_head_init_norm = 0.001 * math.sqrt(n_embd)

    wte_dist_from_init = (wte_norms - wte_init_norm).abs()
    lm_head_dist_from_init = (lm_head_norms - lm_head_init_norm).abs()

    print0(f"  wte norms:     mean={wte_norms.mean():.3f}, std={wte_norms.std():.3f}, "
           f"init_expected={wte_init_norm:.3f}")
    print0(f"  lm_head norms: mean={lm_head_norms.mean():.3f}, std={lm_head_norms.std():.3f}, "
           f"init_expected={lm_head_init_norm:.3f}")

    return {
        'wte_norms': wte_norms,
        'lm_head_norms': lm_head_norms,
        'wte_dist_from_init': wte_dist_from_init,
        'lm_head_dist_from_init': lm_head_dist_from_init,
    }

# -----------------------------------------------------------------------------
# Analysis B: Cosine similarity to mean embedding

def analyze_cosine_to_mean(model, vocab_size):
    """Compute cosine similarity of each token embedding to the mean embedding."""
    wte_weight = model.transformer.wte.weight[:vocab_size].float()
    wte_mean = wte_weight.mean(dim=0, keepdim=True)
    cos_sim = F.cosine_similarity(wte_weight, wte_mean.expand_as(wte_weight), dim=1)

    print0(f"  cos_sim to mean: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}, "
           f"max={cos_sim.max():.4f}, min={cos_sim.min():.4f}")

    return cos_sim

# -----------------------------------------------------------------------------
# Analysis C: KNN self-similarity

def analyze_knn_similarity(model, vocab_size, K=10, chunk_size=4096):
    """Compute average cosine similarity to K nearest neighbors for each token."""
    wte_weight = model.transformer.wte.weight[:vocab_size].float()
    wte_normed = F.normalize(wte_weight, dim=1)  # (V, n_embd)

    avg_knn_sim = torch.zeros(vocab_size)

    # Process in chunks to avoid OOM on CPU
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = wte_normed[start:end]  # (chunk_size, n_embd)
        sim = chunk @ wte_normed.T     # (chunk_size, V)

        # Zero out self-similarity
        for i in range(end - start):
            sim[i, start + i] = -1.0

        # Top-K similarities (excluding self)
        topk_vals, _ = sim.topk(K, dim=1)
        avg_knn_sim[start:end] = topk_vals.mean(dim=1)

        if start % (chunk_size * 4) == 0 and start > 0:
            print0(f"    KNN progress: {start}/{vocab_size}")

    print0(f"  avg KNN sim (K={K}): mean={avg_knn_sim.mean():.4f}, std={avg_knn_sim.std():.4f}, "
           f"max={avg_knn_sim.max():.4f}, min={avg_knn_sim.min():.4f}")

    return avg_knn_sim

# -----------------------------------------------------------------------------
# Analysis D+E: Per-token loss and frequency

@torch.no_grad()
def analyze_per_token_loss(model, tokenizer, vocab_size, device, eval_tokens, sequence_len):
    """Run validation data through the model and compute per-token loss and frequency."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    device_batch_size = 4 if device.type == 'cpu' else 16
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_batch_size, sequence_len, "val", device=device
    )

    token_loss_sum = torch.zeros(vocab_size, device=device)
    token_count = torch.zeros(vocab_size, dtype=torch.long, device=device)
    tokens_seen = 0

    for x, y in loader:
        loss = model(x, y, loss_reduction='none')  # (B*T,)
        targets = y.view(-1)
        valid = targets >= 0

        valid_targets = targets[valid]
        valid_loss = loss[valid]

        # Accumulate per-token statistics
        token_loss_sum.scatter_add_(0, valid_targets, valid_loss.float())
        token_count.scatter_add_(0, valid_targets, torch.ones_like(valid_targets, dtype=torch.long))

        tokens_seen += valid.sum().item()
        if tokens_seen % 200_000 < device_batch_size * sequence_len:
            print0(f"    Per-token loss progress: {tokens_seen:,}/{eval_tokens:,} tokens")
        if tokens_seen >= eval_tokens:
            break

    print0(f"    Total tokens evaluated: {tokens_seen:,}")

    # Compute per-token mean loss
    has_data = token_count > 0
    token_mean_loss = torch.full((vocab_size,), float('nan'), device=device)
    token_mean_loss[has_data] = token_loss_sum[has_data] / token_count[has_data].float()

    n_seen = has_data.sum().item()
    n_unseen = (~has_data).sum().item()
    print0(f"  Tokens with data: {n_seen:,} | Tokens unseen: {n_unseen:,}")
    if n_seen > 0:
        print0(f"  Per-token mean loss: mean={token_mean_loss[has_data].mean():.4f}, "
               f"std={token_mean_loss[has_data].std():.4f}")

    return token_mean_loss.cpu(), token_count.cpu()

# -----------------------------------------------------------------------------
# Composite scoring

def compute_composite_score(signals, has_loss_data):
    """Combine normalized signals into a composite undertrained score (higher = more undertrained)."""
    # Invert signals so higher = more undertrained
    # wte_dist_from_init: small distance = undertrained → invert
    score_wte = 1.0 - normalize_01(signals['wte_dist_from_init'])
    # lm_head norms: small norm = undertrained → invert
    score_lm = 1.0 - normalize_01(signals['lm_head_norms'])
    # cos_sim to mean: high sim = undertrained → keep as-is
    score_cos = normalize_01(signals['cos_sim_to_mean'])
    # KNN sim: high sim = undertrained → keep as-is
    score_knn = normalize_01(signals['avg_knn_sim'])

    components = [score_wte, score_lm, score_cos, score_knn]
    weights = [1.0, 1.0, 1.0, 1.0]

    if has_loss_data:
        # token mean loss: high loss = undertrained → keep as-is
        score_loss = normalize_01(signals['token_mean_loss'])
        # token count: low count = undertrained → invert
        score_freq = 1.0 - normalize_01(torch.log1p(signals['token_count'].float()))

        # For tokens with no data, set data-based scores to 1.0 (maximally undertrained)
        no_data = signals['token_count'] == 0
        score_loss[no_data] = 1.0
        score_freq[no_data] = 1.0
        # Clear NaN from score_loss
        score_loss = torch.nan_to_num(score_loss, nan=1.0)

        components.extend([score_loss, score_freq])
        weights.extend([1.0, 1.0])

    # Weighted average
    total_weight = sum(weights)
    composite = sum(w * c for w, c in zip(weights, components)) / total_weight

    return composite

# -----------------------------------------------------------------------------
# Output formatting

def print_token_table(title, token_ids, signals, tokenizer, special_token_ids, has_loss_data):
    """Print a formatted table of tokens with their signal values."""
    print0(f"\n{'=' * 100}")
    print0(f"  {title}")
    print0(f"{'=' * 100}")

    header = f"{'Rank':>4}  {'ID':>6}  {'Token':<30}  {'Score':>7}  {'WTE':>8}  {'LM':>8}  {'CosSim':>7}  {'KNN':>7}"
    if has_loss_data:
        header += f"  {'Loss':>8}  {'Count':>8}"
    header += f"  {'Special':>7}"
    print0(header)
    print0("-" * len(header))

    for rank, tid in enumerate(token_ids, 1):
        tid = tid.item() if isinstance(tid, torch.Tensor) else tid
        token_str = safe_token_repr(tokenizer, tid)
        # Truncate long tokens for display
        if len(token_str) > 28:
            token_str = token_str[:25] + "..."
        is_special = "  *" if tid in special_token_ids else ""

        line = (f"{rank:>4}  {tid:>6}  {token_str:<30}  "
                f"{signals['composite'][tid]:>7.4f}  "
                f"{signals['wte_norms'][tid]:>8.3f}  "
                f"{signals['lm_head_norms'][tid]:>8.4f}  "
                f"{signals['cos_sim_to_mean'][tid]:>7.4f}  "
                f"{signals['avg_knn_sim'][tid]:>7.4f}")
        if has_loss_data:
            loss_val = signals['token_mean_loss'][tid]
            count_val = signals['token_count'][tid].item()
            loss_str = f"{loss_val:>8.4f}" if not math.isnan(loss_val) else "     N/A"
            line += f"  {loss_str}  {count_val:>8}"
        line += f"  {is_special:>7}"
        print0(line)


def save_json_output(output_path, signals, tokenizer, meta, vocab_size, has_loss_data):
    """Save detailed per-token results to a JSON file."""
    tokens = []
    for tid in range(vocab_size):
        entry = {
            'token_id': tid,
            'token_str': safe_token_repr(tokenizer, tid),
            'composite_score': round(signals['composite'][tid].item(), 6),
            'wte_norm': round(signals['wte_norms'][tid].item(), 4),
            'lm_head_norm': round(signals['lm_head_norms'][tid].item(), 6),
            'cos_sim_to_mean': round(signals['cos_sim_to_mean'][tid].item(), 4),
            'avg_knn_sim': round(signals['avg_knn_sim'][tid].item(), 4),
        }
        if has_loss_data:
            loss_val = signals['token_mean_loss'][tid].item()
            entry['mean_loss'] = round(loss_val, 4) if not math.isnan(loss_val) else None
            entry['count'] = signals['token_count'][tid].item()
        tokens.append(entry)

    # Sort by composite score descending (most undertrained first)
    tokens.sort(key=lambda x: x['composite_score'], reverse=True)

    result = {
        'meta': {
            'source': meta.get('_source', 'base'),
            'step': meta.get('step', None),
            'vocab_size': vocab_size,
            'n_embd': meta['model_config']['n_embd'],
            'has_loss_data': has_loss_data,
        },
        'tokens': tokens,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print0(f"\nSaved detailed results to {output_path}")

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Analyze trained model for undertrained tokens")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"])
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-tokens", type=int, default=1_000_000,
                        help="Tokens of validation data for per-token loss. 0 to skip.")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--knn-k", type=int, default=10, help="K for KNN self-similarity")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to a HuggingFace tokenizer JSON file (overrides default tokenizer)")
    args = parser.parse_args()

    # Init
    device_type = autodetect_device_type() if args.device == "" else args.device
    _, _, _, _, device = compute_init(device_type)

    # Resolve checkpoint path
    model_dir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    model_tag = args.model_tag if args.model_tag else find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    step = args.step if args.step else find_last_step(checkpoint_dir)

    # Load checkpoint directly (bypasses tokenizer vocab assertion)
    print0(f"\nLoading {args.source} model: {model_tag} step {step}...")
    model_data, _, meta = load_checkpoint(checkpoint_dir, step, device)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    model_config = GPTConfig(**meta["model_config"])
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    vocab_size = meta['model_config']['vocab_size']
    n_embd = meta['model_config']['n_embd']
    sequence_len = meta['model_config'].get('sequence_len', 2048)

    # Try to load tokenizer (may not match if tokenizer was re-trained)
    tokenizer = None
    if args.tokenizer:
        # Load from explicit HuggingFace tokenizer JSON path
        from tokenizers import Tokenizer as HFTokenizer
        tok = HuggingFaceTokenizer(HFTokenizer.from_file(args.tokenizer))
        if tok.get_vocab_size() == vocab_size:
            tokenizer = tok
            print0(f"Tokenizer loaded from {args.tokenizer} (vocab matches: {vocab_size:,})")
        else:
            print0(f"WARNING: Tokenizer vocab ({tok.get_vocab_size():,}) != model vocab ({vocab_size:,}). "
                   f"Token rendering will use IDs only.")
    else:
        try:
            tok = get_tokenizer()
            if tok.get_vocab_size() == vocab_size:
                tokenizer = tok
                print0(f"Tokenizer loaded (vocab matches: {vocab_size:,})")
            else:
                print0(f"WARNING: Tokenizer vocab ({tok.get_vocab_size():,}) != model vocab ({vocab_size:,}). "
                       f"Token rendering will use IDs only.")
        except Exception as e:
            print0(f"WARNING: Could not load tokenizer: {e}. Token rendering will use IDs only.")
    if tokenizer is None and args.eval_tokens > 0:
        print0(f"  Forcing --eval-tokens 0 due to missing/mismatched tokenizer.")
        args.eval_tokens = 0

    # Identify special tokens
    special_token_ids = set()
    if tokenizer is not None:
        for name in tokenizer.get_special_tokens():
            try:
                special_token_ids.add(tokenizer.encode_special(name))
            except Exception:
                pass

    print0(f"\n{'=' * 60}")
    print0(f"  Token Analysis: {args.source} model ({model_tag}, step {step})")
    print0(f"  Vocab size: {vocab_size:,} | n_embd: {n_embd}")
    print0(f"  Special tokens: {len(special_token_ids)}")
    print0(f"  Tokenizer: {'matched' if tokenizer else 'unavailable'}")
    print0(f"{'=' * 60}")

    # --- Analysis A: Embedding norms ---
    print0(f"\n[A] Embedding norm analysis...")
    norm_results = analyze_embedding_norms(model, vocab_size, n_embd)

    # --- Analysis B: Cosine similarity to mean ---
    print0(f"\n[B] Cosine similarity to mean embedding...")
    cos_sim_to_mean = analyze_cosine_to_mean(model, vocab_size)

    # --- Analysis C: KNN self-similarity ---
    print0(f"\n[C] KNN self-similarity (K={args.knn_k})...")
    avg_knn_sim = analyze_knn_similarity(model, vocab_size, K=args.knn_k)

    # --- Analysis D+E: Per-token loss and frequency ---
    has_loss_data = args.eval_tokens > 0
    if has_loss_data:
        print0(f"\n[D+E] Per-token loss and frequency ({args.eval_tokens:,} tokens)...")
        token_mean_loss, token_count = analyze_per_token_loss(
            model, tokenizer, vocab_size, device, args.eval_tokens, sequence_len
        )
    else:
        print0(f"\n[D+E] Skipped (--eval-tokens 0)")
        token_mean_loss = torch.full((vocab_size,), float('nan'))
        token_count = torch.zeros(vocab_size, dtype=torch.long)

    # --- Composite score (all on CPU) ---
    print0(f"\n[F] Computing composite undertrained score...")
    signals = {
        'wte_norms': norm_results['wte_norms'].cpu(),
        'lm_head_norms': norm_results['lm_head_norms'].cpu(),
        'wte_dist_from_init': norm_results['wte_dist_from_init'].cpu(),
        'lm_head_dist_from_init': norm_results['lm_head_dist_from_init'].cpu(),
        'cos_sim_to_mean': cos_sim_to_mean.cpu(),
        'avg_knn_sim': avg_knn_sim.cpu(),
        'token_mean_loss': token_mean_loss.cpu(),
        'token_count': token_count.cpu(),
    }
    composite = compute_composite_score(signals, has_loss_data)
    signals['composite'] = composite

    # --- Summary statistics ---
    thresholds = [0.9, 0.8, 0.7, 0.5]
    print0(f"\n  Composite score distribution:")
    for t in thresholds:
        n = (composite >= t).sum().item()
        print0(f"    score >= {t}: {n:>6} tokens ({100*n/vocab_size:.1f}%)")

    # --- Print tables ---
    top_k = args.top_k

    # Most undertrained (excluding special tokens for a cleaner view)
    sorted_ids = composite.argsort(descending=True)
    non_special_sorted = [tid for tid in sorted_ids if tid.item() not in special_token_ids]

    print_token_table(
        f"Top-{top_k} Most UNDERTRAINED Tokens (composite score, excluding special tokens)",
        non_special_sorted[:top_k], signals, tokenizer, special_token_ids, has_loss_data
    )

    # Special tokens
    special_sorted = [tid for tid in sorted_ids if tid.item() in special_token_ids]
    if special_sorted:
        print_token_table(
            "Special Tokens",
            special_sorted, signals, tokenizer, special_token_ids, has_loss_data
        )

    # Most well-trained
    print_token_table(
        f"Top-{top_k} Most WELL-TRAINED Tokens (lowest composite score)",
        sorted_ids[-top_k:].flip(0), signals, tokenizer, special_token_ids, has_loss_data  # flip so best first
    )

    # --- Save JSON ---
    if args.output:
        meta['_source'] = args.source
        save_json_output(args.output, signals, tokenizer, meta, vocab_size, has_loss_data)

    compute_cleanup()
    print0("\nDone.")


if __name__ == "__main__":
    main()
