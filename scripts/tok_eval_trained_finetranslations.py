#!/usr/bin/env python3
"""
Evaluate trained finetranslations tokenizer against GPT-2 and GPT-4.
Tests the tokenizer created by tok_train_finetranslations.py on the same dataset.

Usage:
    python scripts/tok_eval_trained_finetranslations.py --tokenizer-name tokenizer_finetranslations
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pyarrow.parquet as pq

from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir


def load_finetranslations_samples(output_dir: Path, max_chars_per_shard: int = 100_000) -> List[Tuple[str, str]]:
    """
    Load sample text from finetranslations parquet shards.
    
    Args:
        output_dir: Directory containing shard_*.parquet files
        max_chars_per_shard: Maximum characters to load per shard for evaluation
        
    Returns:
        List of (shard_name, text) tuples
    """
    samples = []
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} not found")
        print("Run dev/download_finetranslations.py first to generate the dataset")
        return samples
    
    # Get all parquet shards
    shard_files = sorted(output_dir.glob("shard_*.parquet"))
    
    if not shard_files:
        print(f"Error: No shard files found in {output_dir}")
        return samples
    
    print(f"Found {len(shard_files)} shard files in {output_dir}")
    
    for shard_file in shard_files:
        # Read parquet file
        table = pq.read_table(shard_file)
        texts = table.column("text").to_pylist()
        
        # Concatenate documents until we hit the char limit
        combined_text = ""
        for doc in texts:
            if not doc:  # Skip empty/padding docs
                continue
            if len(combined_text) + len(doc) > max_chars_per_shard:
                break
            combined_text += doc + "\n"
        
        if combined_text.strip():
            samples.append((shard_file.stem, combined_text.strip()))
            print(f"  Loaded {len(combined_text):,} chars from {shard_file.name}")
    
    return samples


def evaluate_tokenizers(samples: List[Tuple[str, str]], trained_tokenizer_name: str) -> None:
    """
    Evaluate compression ratios for different tokenizers on the samples.
    
    Args:
        samples: List of (name, text) tuples to evaluate
        trained_tokenizer_name: Name of the trained tokenizer directory
    """
    if not samples:
        print("No samples to evaluate")
        return
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    # Load trained tokenizer
    base_dir = get_base_dir()
    trained_tokenizer_path = Path(base_dir) / trained_tokenizer_name
    
    if not trained_tokenizer_path.exists():
        print(f"Error: Trained tokenizer not found at {trained_tokenizer_path}")
        print(f"Please run tok_train_finetranslations.py first with --tokenizer-name {trained_tokenizer_name}")
        sys.exit(1)
    
    print(f"\n{BLUE}Loading trained tokenizer from: {trained_tokenizer_path}{RESET}")
    
    # Initialize tokenizers
    tokenizer_configs = [
        ("GPT-2", RustBPETokenizer.from_pretrained("gpt2")),
        ("GPT-4", RustBPETokenizer.from_pretrained("cl100k_base")),
        ("Trained FT", RustBPETokenizer.from_saved(str(trained_tokenizer_path))),
    ]
    
    # Print vocab sizes
    print("\n" + "=" * 80)
    print("VOCABULARY SIZES")
    print("=" * 80)
    for name, tokenizer in tokenizer_configs:
        print(f"{name:15s}: {tokenizer.get_vocab_size():,} tokens")
    
    # Evaluate each tokenizer
    results = {}
    for tok_name, tokenizer in tokenizer_configs:
        print(f"\n{YELLOW}Evaluating {tok_name} tokenizer...{RESET}")
        results[tok_name] = {}
        
        for sample_name, text in samples:
            # Encode and verify
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text, f"Decode mismatch for {tok_name} on {sample_name}"
            
            # Calculate metrics
            encoded_bytes = text.encode('utf-8')
            ratio = len(encoded_bytes) / len(encoded)
            
            results[tok_name][sample_name] = {
                'bytes': len(encoded_bytes),
                'tokens': len(encoded),
                'ratio': ratio
            }
            print(f"  {sample_name}: {len(encoded):,} tokens, {ratio:.2f} bytes/token")
    
    # Print comparison tables
    baselines = ["GPT-2", "GPT-4"]
    
    for baseline_name in baselines:
        baseline_results = results[baseline_name]
        trained_results = results["Trained FT"]
        
        print("\n" + "=" * 110)
        print(f"COMPARISON: TRAINED FINETRANSLATIONS vs {baseline_name.upper()}")
        print("=" * 110)
        print(f"{'Sample':<25} {'Bytes':<10} {baseline_name.upper():<25} {'TRAINED FT':<25} {'Improvement':<15} {'Winner':<10}")
        print(f"{'':25} {'':10} {'Tokens':<11} {'Ratio':<13} {'Tokens':<11} {'Ratio':<13} {'%':<15}")
        print("-" * 110)
        
        total_baseline_tokens = 0
        total_trained_tokens = 0
        total_bytes = 0
        
        for sample_name, text in samples:
            baseline_data = baseline_results[sample_name]
            trained_data = trained_results[sample_name]
            
            total_baseline_tokens += baseline_data['tokens']
            total_trained_tokens += trained_data['tokens']
            total_bytes += baseline_data['bytes']
            
            # Calculate improvement (positive = trained is better)
            improvement = ((baseline_data['tokens'] - trained_data['tokens']) / baseline_data['tokens']) * 100
            
            # Determine winner (higher ratio = better compression)
            if baseline_data['ratio'] > trained_data['ratio']:
                baseline_color, trained_color = GREEN, RED
                winner = baseline_name
                improve_color = RED
            elif trained_data['ratio'] > baseline_data['ratio']:
                baseline_color, trained_color = RED, GREEN
                winner = "TRAINED"
                improve_color = GREEN
            else:
                baseline_color, trained_color = "", ""
                winner = "TIE"
                improve_color = ""
            
            print(f"{sample_name:<25} {baseline_data['bytes']:<10,} "
                  f"{baseline_color}{baseline_data['tokens']:<11,}{RESET} "
                  f"{baseline_color}{baseline_data['ratio']:<13.3f}{RESET} "
                  f"{trained_color}{trained_data['tokens']:<11,}{RESET} "
                  f"{trained_color}{trained_data['ratio']:<13.3f}{RESET} "
                  f"{improve_color}{improvement:+9.2f}%{RESET}     "
                  f"{winner:<10}")
        
        # Print totals
        print("-" * 110)
        total_baseline_ratio = total_bytes / total_baseline_tokens
        total_trained_ratio = total_bytes / total_trained_tokens
        total_improvement = ((total_baseline_tokens - total_trained_tokens) / total_baseline_tokens) * 100
        
        if total_baseline_ratio > total_trained_ratio:
            baseline_color, trained_color = GREEN, RED
            winner = baseline_name
            improve_color = RED
        elif total_trained_ratio > total_baseline_ratio:
            baseline_color, trained_color = RED, GREEN
            winner = "TRAINED"
            improve_color = GREEN
        else:
            baseline_color, trained_color = "", ""
            winner = "TIE"
            improve_color = ""
        
        print(f"{'TOTAL':<25} {total_bytes:<10,} "
              f"{baseline_color}{total_baseline_tokens:<11,}{RESET} "
              f"{baseline_color}{total_baseline_ratio:<13.3f}{RESET} "
              f"{trained_color}{total_trained_tokens:<11,}{RESET} "
              f"{trained_color}{total_trained_ratio:<13.3f}{RESET} "
              f"{improve_color}{total_improvement:+9.2f}%{RESET}     "
              f"{winner:<10}")
    
    # Print summary comparison of all three
    print("\n" + "=" * 110)
    print("SUMMARY: ALL TOKENIZERS")
    print("=" * 110)
    print(f"{'Tokenizer':<15} {'Vocab Size':<15} {'Total Tokens':<15} {'Avg Ratio':<15} {'vs GPT-2':<15} {'vs GPT-4':<15}")
    print("-" * 110)
    
    for tok_name, tokenizer in tokenizer_configs:
        tok_results = results[tok_name]
        total_bytes = sum(r['bytes'] for r in tok_results.values())
        total_tokens = sum(r['tokens'] for r in tok_results.values())
        avg_ratio = total_bytes / total_tokens
        vocab_size = tokenizer.get_vocab_size()
        
        # Calculate improvements
        gpt2_tokens = sum(results["GPT-2"][name]['tokens'] for name, _ in samples)
        gpt4_tokens = sum(results["GPT-4"][name]['tokens'] for name, _ in samples)
        
        vs_gpt2 = ((gpt2_tokens - total_tokens) / gpt2_tokens) * 100
        vs_gpt4 = ((gpt4_tokens - total_tokens) / gpt4_tokens) * 100
        
        print(f"{tok_name:<15} {vocab_size:<15,} {total_tokens:<15,} {avg_ratio:<15.3f} "
              f"{vs_gpt2:+7.2f}%{' ':7} {vs_gpt4:+7.2f}%")
    
    # Log to report
    try:
        from nanochat.report import get_report
        
        lines = []
        lines.append(f"# Tokenizer Evaluation: {trained_tokenizer_name}")
        lines.append("")
        lines.append("## Vocabulary Sizes")
        lines.append("")
        for name, tokenizer in tokenizer_configs:
            lines.append(f"- **{name}**: {tokenizer.get_vocab_size():,} tokens")
        lines.append("")
        
        for baseline_name in baselines:
            baseline_results = results[baseline_name]
            trained_results = results["Trained FT"]
            
            lines.append(f"## Comparison with {baseline_name}")
            lines.append("")
            lines.append(f"| Sample | Bytes | {baseline_name} Tokens | {baseline_name} Ratio | Trained Tokens | Trained Ratio | Improvement % |")
            lines.append("|--------|-------|----------------|---------------|----------------|---------------|---------------|")
            
            total_baseline_tokens = 0
            total_trained_tokens = 0
            total_bytes = 0
            
            for sample_name, text in samples:
                baseline_data = baseline_results[sample_name]
                trained_data = trained_results[sample_name]
                
                total_baseline_tokens += baseline_data['tokens']
                total_trained_tokens += trained_data['tokens']
                total_bytes += baseline_data['bytes']
                
                improvement = ((baseline_data['tokens'] - trained_data['tokens']) / baseline_data['tokens']) * 100
                
                lines.append(f"| {sample_name} | {baseline_data['bytes']:,} | {baseline_data['tokens']:,} | "
                           f"{baseline_data['ratio']:.3f} | {trained_data['tokens']:,} | {trained_data['ratio']:.3f} | {improvement:+.2f}% |")
            
            # Add totals row
            total_baseline_ratio = total_bytes / total_baseline_tokens
            total_trained_ratio = total_bytes / total_trained_tokens
            total_improvement = ((total_baseline_tokens - total_trained_tokens) / total_baseline_tokens) * 100
            
            lines.append(f"| **TOTAL** | **{total_bytes:,}** | **{total_baseline_tokens:,}** | "
                       f"**{total_baseline_ratio:.3f}** | **{total_trained_tokens:,}** | **{total_trained_ratio:.3f}** | **{total_improvement:+.2f}%** |")
            lines.append("")
        
        # Add summary table
        lines.append("## Summary: All Tokenizers")
        lines.append("")
        lines.append("| Tokenizer | Vocab Size | Total Tokens | Avg Ratio | vs GPT-2 | vs GPT-4 |")
        lines.append("|-----------|------------|--------------|-----------|----------|----------|")
        
        for tok_name, tokenizer in tokenizer_configs:
            tok_results = results[tok_name]
            total_bytes = sum(r['bytes'] for r in tok_results.values())
            total_tokens = sum(r['tokens'] for r in tok_results.values())
            avg_ratio = total_bytes / total_tokens
            vocab_size = tokenizer.get_vocab_size()
            
            gpt2_tokens = sum(results["GPT-2"][name]['tokens'] for name, _ in samples)
            gpt4_tokens = sum(results["GPT-4"][name]['tokens'] for name, _ in samples)
            
            vs_gpt2 = ((gpt2_tokens - total_tokens) / gpt2_tokens) * 100
            vs_gpt4 = ((gpt4_tokens - total_tokens) / gpt4_tokens) * 100
            
            lines.append(f"| {tok_name} | {vocab_size:,} | {total_tokens:,} | {avg_ratio:.3f} | {vs_gpt2:+.2f}% | {vs_gpt4:+.2f}% |")
        
        lines.append("")
        
        report_markdown = "\n".join(lines)
        get_report().log(section=f"Tokenizer evaluation - {trained_tokenizer_name}", data=[report_markdown])
        print(f"\n{GREEN}✓ Results logged to report{RESET}")
        
    except Exception as e:
        print(f"\n{YELLOW}Warning: Could not log to report: {e}{RESET}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate trained finetranslations tokenizer')
    parser.add_argument('--tokenizer-name', type=str, default='tokenizer_finetranslations',
                        help='Name of the trained tokenizer directory (default: tokenizer_finetranslations)')
    parser.add_argument('--data-dir', type=str, default='dev/finetranslations_output',
                        help='Directory containing parquet files (default: dev/finetranslations_output)')
    parser.add_argument('--max-chars-per-shard', type=int, default=100_000,
                        help='Maximum characters to load per shard (default: 100,000)')
    args = parser.parse_args()
    
    # Resolve data directory
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        script_dir = Path(__file__).parent.parent
        data_path = script_dir / args.data_dir
    
    print("=" * 80)
    print("TRAINED FINETRANSLATIONS TOKENIZER EVALUATION")
    print("=" * 80)
    print(f"\nTokenizer: {args.tokenizer_name}")
    print(f"Data directory: {data_path}")
    print(f"Max chars per shard: {args.max_chars_per_shard:,}")
    
    # Load samples from parquet shards
    print("\nLoading samples...")
    samples = load_finetranslations_samples(data_path, max_chars_per_shard=args.max_chars_per_shard)
    
    if not samples:
        print("\nNo samples loaded. Exiting.")
        sys.exit(1)
    
    total_chars = sum(len(text) for _, text in samples)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Total characters: {total_chars:,}")
    
    # Evaluate tokenizers
    evaluate_tokenizers(samples, args.tokenizer_name)
    
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
