"""
Token Score Analysis Tool

Calculates and displays scores for every token in the vocabulary showing:
- Training Score: How much the token embedding has been trained (based on embedding weight norms)
- Activation Score: How much the token is used in predictions (based on lm_head weight norms)

Usage:
    # Analyze a trained model
    python -m scripts.tok_score --model-tag d12

    # Show top N tokens by various metrics
    python -m scripts.tok_score --model-tag d12 --top 50 --sort-by embed_norm

    # Export to CSV
    python -m scripts.tok_score --model-tag d12 --output scores.csv
"""

import os
import argparse
import json
import csv

import torch
import numpy as np

from nanochat.common import print0, get_base_dir, autodetect_device_type
from nanochat.tokenizer import SPECIAL_TOKENS
from nanochat.checkpoint_manager import load_model


def load_model_from_tag(device, model_tag=None, step=None, source="base"):
    """Load model from checkpoint."""
    try:
        model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
        return model, tokenizer, meta
    except FileNotFoundError as e:
        print0(f"Could not load model: {e}")
        return None, None, None


def compute_embedding_norms(model):
    """
    Compute per-token statistics from the token embedding layer.
    
    Returns dict with:
    - embed_norm: L2 norm of each token's embedding vector
    - embed_mean: Mean of each token's embedding values
    - embed_std: Std dev of each token's embedding values
    """
    # Get token embedding weights: (vocab_size, n_embd)
    wte = model.transformer.wte.weight.float()
    vocab_size = wte.size(0)
    
    # Compute statistics
    embed_norm = torch.norm(wte, dim=1)  # L2 norm per token
    embed_mean = wte.mean(dim=1)         # Mean per token
    embed_std = wte.std(dim=1)           # Std per token
    
    return {
        'embed_norm': embed_norm.cpu().numpy(),
        'embed_mean': embed_mean.cpu().numpy(),
        'embed_std': embed_std.cpu().numpy(),
    }


def compute_lm_head_norms(model):
    """
    Compute per-token statistics from the language model head (unembedding).
    
    Returns dict with:
    - head_norm: L2 norm of each token's output projection
    - head_mean: Mean of each token's output values
    - head_std: Std dev of each token's output values
    """
    # Get lm_head weights: (vocab_size, n_embd)
    lm_head = model.lm_head.weight.float()
    vocab_size = lm_head.size(0)
    
    # Compute statistics
    head_norm = torch.norm(lm_head, dim=1)  # L2 norm per token
    head_mean = lm_head.mean(dim=1)          # Mean per token
    head_std = lm_head.std(dim=1)            # Std per token
    
    return {
        'head_norm': head_norm.cpu().numpy(),
        'head_mean': head_mean.cpu().numpy(),
        'head_std': head_std.cpu().numpy(),
    }


def compute_value_embed_norms(model):
    """
    Compute per-token statistics from value embeddings (if present).
    
    Returns dict with aggregated stats across all value embedding layers.
    """
    if not model.value_embeds:
        return {}
    
    # Stack all value embedding weights
    all_ve_weights = []
    for layer_idx, ve in model.value_embeds.items():
        all_ve_weights.append(ve.weight.float())
    
    # Average across layers
    stacked = torch.stack(all_ve_weights, dim=0)  # (n_layers, vocab_size, kv_dim)
    avg_ve = stacked.mean(dim=0)  # (vocab_size, kv_dim)
    
    ve_norm = torch.norm(avg_ve, dim=1)
    ve_mean = avg_ve.mean(dim=1)
    ve_std = avg_ve.std(dim=1)
    
    return {
        've_norm': ve_norm.cpu().numpy(),
        've_mean': ve_mean.cpu().numpy(),
        've_std': ve_std.cpu().numpy(),
    }


def compute_combined_score(data):
    """
    Compute a combined "training intensity" score.
    
    Higher score = token has been more actively trained.
    Uses embed_norm and head_norm as primary signals.
    """
    embed_norm = data.get('embed_norm')
    head_norm = data.get('head_norm')
    
    # Normalize both to 0-1 range
    if embed_norm is not None and head_norm is not None:
        embed_normalized = (embed_norm - embed_norm.min()) / (embed_norm.max() - embed_norm.min() + 1e-10)
        head_normalized = (head_norm - head_norm.min()) / (head_norm.max() - head_norm.min() + 1e-10)
        # Combined score: geometric mean
        combined = np.sqrt(embed_normalized * head_normalized)
        return combined
    elif embed_norm is not None:
        return (embed_norm - embed_norm.min()) / (embed_norm.max() - embed_norm.min() + 1e-10)
    elif head_norm is not None:
        return (head_norm - head_norm.min()) / (head_norm.max() - head_norm.min() + 1e-10)
    else:
        return None


def format_token_display(token_str, max_len=30):
    """Format a token string for display, handling special characters."""
    # Replace problematic characters
    display = token_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    # Truncate if too long
    if len(display) > max_len:
        display = display[:max_len-3] + '...'
    return display


def analyze_tokens(model, tokenizer):
    """
    Main analysis function. Returns a list of dicts with per-token data.
    """
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Analyzing {vocab_size:,} tokens...")
    
    # Initialize data structure
    token_data = {
        'token_id': np.arange(vocab_size),
        'token_str': [],
        'is_special': [],
        'byte_length': [],
    }
    
    # Get token strings and properties
    special_tokens_set = tokenizer.get_special_tokens()
    for i in range(vocab_size):
        token_str = tokenizer.decode([i])
        token_data['token_str'].append(token_str)
        token_data['is_special'].append(token_str in special_tokens_set or token_str in SPECIAL_TOKENS)
        token_data['byte_length'].append(len(token_str.encode('utf-8', errors='replace')))
    
    token_data['is_special'] = np.array(token_data['is_special'])
    token_data['byte_length'] = np.array(token_data['byte_length'])
    
    # Compute model-based scores
    print0("Computing embedding norms...")
    embed_stats = compute_embedding_norms(model)
    token_data.update(embed_stats)
    
    print0("Computing lm_head norms...")
    head_stats = compute_lm_head_norms(model)
    token_data.update(head_stats)
    
    print0("Computing value embedding norms...")
    ve_stats = compute_value_embed_norms(model)
    token_data.update(ve_stats)
    
    # Combined score
    print0("Computing combined score...")
    token_data['combined_score'] = compute_combined_score(token_data)
    
    return token_data


def print_token_table(token_data, top_n=50, sort_by='embed_norm', ascending=False, filter_special=False):
    """Print a formatted table of token scores."""
    
    # Build list of tokens with their data
    vocab_size = len(token_data['token_str'])
    tokens = []
    
    for i in range(vocab_size):
        if filter_special and token_data['is_special'][i]:
            continue
            
        entry = {
            'id': i,
            'token': token_data['token_str'][i],
            'is_special': token_data['is_special'][i],
        }
        
        # Add available metrics
        for key in ['embed_norm', 'head_norm', 've_norm', 'combined_score']:
            if key in token_data:
                entry[key] = token_data[key][i]
        
        tokens.append(entry)
    
    # Sort
    if sort_by in token_data:
        tokens.sort(key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    
    # Print header
    print("\n" + "="*100)
    print(f"Top {top_n} tokens sorted by {sort_by} ({'ascending' if ascending else 'descending'})")
    print("="*100)
    
    # Build header
    header = f"{'Rank':>5} {'ID':>7} {'Token':>32}"
    if 'embed_norm' in token_data:
        header += f" {'Embed':>10}"
    if 'head_norm' in token_data:
        header += f" {'Head':>10}"
    if 've_norm' in token_data:
        header += f" {'ValEmb':>10}"
    if 'combined_score' in token_data:
        header += f" {'Combined':>10}"
    header += f" {'Special':>8}"
    
    print(header)
    print("-"*100)
    
    # Print rows
    for rank, entry in enumerate(tokens[:top_n], 1):
        token_display = format_token_display(entry['token'])
        row = f"{rank:>5} {entry['id']:>7} {token_display:>32}"
        
        if 'embed_norm' in token_data:
            row += f" {entry.get('embed_norm', 0):>10.4f}"
        if 'head_norm' in token_data:
            row += f" {entry.get('head_norm', 0):>10.4f}"
        if 've_norm' in token_data:
            row += f" {entry.get('ve_norm', 0):>10.4f}"
        if 'combined_score' in token_data:
            row += f" {entry.get('combined_score', 0):>10.4f}"
        row += f" {'Yes' if entry['is_special'] else 'No':>8}"
        
        print(row)
    
    print("="*100)


def print_statistics(token_data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    vocab_size = len(token_data['token_str'])
    special_count = token_data['is_special'].sum()
    
    print(f"Total tokens: {vocab_size:,}")
    print(f"Special tokens: {special_count:,}")
    print(f"Regular tokens: {vocab_size - special_count:,}")
    
    for metric in ['embed_norm', 'head_norm', 've_norm', 'combined_score']:
        if metric in token_data:
            data = token_data[metric]
            print(f"\n{metric}:")
            print(f"  Mean: {data.mean():.4f}")
            print(f"  Std:  {data.std():.4f}")
            print(f"  Min:  {data.min():.4f}")
            print(f"  Max:  {data.max():.4f}")
            
            # Count zeros and very low values
            zeros = (data == 0).sum()
            low = (data < data.mean() - 2*data.std()).sum()
            high = (data > data.mean() + 2*data.std()).sum()
            print(f"  Zeros: {zeros:,} ({zeros/vocab_size*100:.2f}%)")
            print(f"  Low (>2σ below mean): {low:,} ({low/vocab_size*100:.2f}%)")
            print(f"  High (>2σ above mean): {high:,} ({high/vocab_size*100:.2f}%)")


def export_to_csv(token_data, output_path):
    """Export token data to CSV file."""
    vocab_size = len(token_data['token_str'])
    
    # Determine columns
    columns = ['token_id', 'token_str', 'is_special', 'byte_length']
    for key in ['embed_norm', 'embed_mean', 'embed_std', 
                'head_norm', 'head_mean', 'head_std',
                've_norm', 've_mean', 've_std',
                'combined_score']:
        if key in token_data:
            columns.append(key)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        for i in range(vocab_size):
            row = []
            for col in columns:
                if col == 'token_id':
                    row.append(i)
                elif col == 'token_str':
                    row.append(token_data['token_str'][i])
                elif isinstance(token_data[col], np.ndarray):
                    row.append(token_data[col][i])
                else:
                    row.append(token_data[col][i])
            writer.writerow(row)
    
    print0(f"Exported to {output_path}")


def plot_distributions(token_data, output_path=None):
    """Generate distribution plots for token metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print0("matplotlib not installed, skipping plots")
        return
    
    metrics = ['embed_norm', 'head_norm', 've_norm', 'combined_score']
    available = [m for m in metrics if m in token_data]
    
    if not available:
        print0("No metrics available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(available):
        if i >= 4:
            break
        ax = axes[i]
        data = token_data[metric]
        
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric}')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        ax.legend()
    
    # Hide unused axes
    for i in range(len(available), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print0(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze token training and activation scores")
    
    # Model loading
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Model tag to identify the checkpoint directory')
    parser.add_argument('--model-step', type=int, default=None,
                        help='Checkpoint step to load (default: latest)')
    parser.add_argument('--source', type=str, default='base', choices=['base', 'sft', 'rl'],
                        help='Which checkpoint source to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    
    # Display options
    parser.add_argument('--top', type=int, default=50,
                        help='Number of top tokens to display')
    parser.add_argument('--sort-by', type=str, default='combined_score',
                        choices=['embed_norm', 'head_norm', 've_norm', 'combined_score'],
                        help='Metric to sort by')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort in ascending order (show lowest scores)')
    parser.add_argument('--filter-special', action='store_true',
                        help='Filter out special tokens from display')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Export results to CSV file')
    parser.add_argument('--plot', type=str, default=None,
                        help='Save distribution plots to file')
    
    # Show specific token
    parser.add_argument('--token-id', type=int, default=None,
                        help='Show details for a specific token ID')
    parser.add_argument('--search', type=str, default=None,
                        help='Search for tokens containing this string')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        args.device = autodetect_device_type()
    device = torch.device(args.device)
    print0(f"Using device: {device}")
    
    # Load model and tokenizer
    print0(f"Loading model (source={args.source}, tag={args.model_tag}, step={args.model_step})...")
    model, tokenizer, meta = load_model_from_tag(device, args.model_tag, args.model_step, args.source)
    if model is None:
        print0("Failed to load model, exiting.")
        return
    
    # Analyze tokens
    token_data = analyze_tokens(model, tokenizer)
    
    # Handle specific token lookup
    if args.token_id is not None:
        tid = args.token_id
        print(f"\n{'='*60}")
        print(f"Token ID: {tid}")
        print(f"Token string: {repr(token_data['token_str'][tid])}")
        print(f"Is special: {token_data['is_special'][tid]}")
        print(f"Byte length: {token_data['byte_length'][tid]}")
        for metric in ['embed_norm', 'head_norm', 've_norm', 'combined_score']:
            if metric in token_data:
                print(f"{metric}: {token_data[metric][tid]:.6f}")
        print(f"{'='*60}")
        return
    
    # Handle search
    if args.search is not None:
        search_str = args.search.lower()
        matches = []
        for i, token_str in enumerate(token_data['token_str']):
            if search_str in token_str.lower():
                matches.append(i)
        print(f"\nFound {len(matches)} tokens containing '{args.search}':")
        for tid in matches[:50]:
            token_str = format_token_display(token_data['token_str'][tid])
            print(f"  ID {tid:>6}: {token_str}")
        if len(matches) > 50:
            print(f"  ... and {len(matches) - 50} more")
        return
    
    # Print statistics if requested
    if args.stats:
        print_statistics(token_data)
    
    # Print table
    print_token_table(
        token_data, 
        top_n=args.top, 
        sort_by=args.sort_by,
        ascending=args.ascending,
        filter_special=args.filter_special
    )
    
    # Also show bottom tokens (least trained)
    if not args.ascending:
        print("\n\nBottom tokens (least trained/activated):")
        print_token_table(
            token_data, 
            top_n=args.top, 
            sort_by=args.sort_by,
            ascending=True,
            filter_special=args.filter_special
        )
    
    # Export if requested
    if args.output:
        export_to_csv(token_data, args.output)
    
    # Plot if requested
    if args.plot:
        plot_distributions(token_data, args.plot)


if __name__ == "__main__":
    main()
