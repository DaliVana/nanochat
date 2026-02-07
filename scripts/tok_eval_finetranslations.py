#!/usr/bin/env python3
"""
Evaluate tokenizer compression ratios on finetranslations dataset.
Tests tokenizer on real multilingual text from the downloaded finetranslations data.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer


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
        print("Run download_finetranslations.py first to generate the dataset")
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
            if len(combined_text) + len(doc) > max_chars_per_shard:
                break
            combined_text += doc + "\n"
        
        if combined_text.strip():
            samples.append((shard_file.stem, combined_text.strip()))
            print(f"  Loaded {len(combined_text):,} chars from {shard_file.name}")
    
    return samples


def evaluate_tokenizers(samples: List[Tuple[str, str]]) -> None:
    """
    Evaluate compression ratios for different tokenizers on the samples.
    
    Args:
        samples: List of (name, text) tuples to evaluate
    """
    if not samples:
        print("No samples to evaluate")
        return
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # Initialize tokenizers
    tokenizer_configs = [
        ("gpt2", RustBPETokenizer.from_pretrained("gpt2")),
        ("gpt4", RustBPETokenizer.from_pretrained("cl100k_base")),
        ("ours", get_tokenizer()),
    ]
    
    # Print vocab sizes
    print("\n" + "=" * 80)
    print("VOCABULARY SIZES")
    print("=" * 80)
    for name, tokenizer in tokenizer_configs:
        print(f"{name:10s}: {tokenizer.get_vocab_size():,} tokens")
    
    # Evaluate each tokenizer
    results = {}
    for tok_name, tokenizer in tokenizer_configs:
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
    
    # Print comparison tables
    for baseline_name in ["gpt2", "gpt4"]:
        baseline_results = results[baseline_name]
        ours_results = results["ours"]
        
        print("\n" + "=" * 100)
        print(f"COMPARISON WITH {baseline_name.upper()}")
        print("=" * 100)
        print(f"{'Sample':<25} {'Bytes':<10} {baseline_name.upper():<20} {'OURS':<20} {'Relative':<15} {'Winner':<10}")
        print(f"{'':25} {'':10} {'Tokens':<9} {'Ratio':<10} {'Tokens':<9} {'Ratio':<10} {'Diff %':<15}")
        print("-" * 100)
        
        total_baseline_tokens = 0
        total_ours_tokens = 0
        total_bytes = 0
        
        for sample_name, text in samples:
            baseline_data = baseline_results[sample_name]
            ours_data = ours_results[sample_name]
            
            total_baseline_tokens += baseline_data['tokens']
            total_ours_tokens += ours_data['tokens']
            total_bytes += baseline_data['bytes']
            
            # Calculate relative difference (positive means ours is better)
            relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
            
            # Determine winner (higher ratio = better compression)
            if baseline_data['ratio'] > ours_data['ratio']:
                baseline_color, ours_color = GREEN, RED
                winner = baseline_name.upper()
                diff_color = RED
            elif ours_data['ratio'] > baseline_data['ratio']:
                baseline_color, ours_color = RED, GREEN
                winner = "OURS"
                diff_color = GREEN
            else:
                baseline_color, ours_color = "", ""
                winner = "TIE"
                diff_color = ""
            
            print(f"{sample_name:<25} {baseline_data['bytes']:<10,} "
                  f"{baseline_color}{baseline_data['tokens']:<9,}{RESET} "
                  f"{baseline_color}{baseline_data['ratio']:<10.2f}{RESET} "
                  f"{ours_color}{ours_data['tokens']:<9,}{RESET} "
                  f"{ours_color}{ours_data['ratio']:<10.2f}{RESET} "
                  f"{diff_color}{relative_diff:+8.1f}%{RESET}      "
                  f"{winner:<10}")
        
        # Print totals
        print("-" * 100)
        total_baseline_ratio = total_bytes / total_baseline_tokens
        total_ours_ratio = total_bytes / total_ours_tokens
        total_relative_diff = ((total_baseline_tokens - total_ours_tokens) / total_baseline_tokens) * 100
        
        if total_baseline_ratio > total_ours_ratio:
            baseline_color, ours_color = GREEN, RED
            winner = baseline_name.upper()
            diff_color = RED
        elif total_ours_ratio > total_baseline_ratio:
            baseline_color, ours_color = RED, GREEN
            winner = "OURS"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            winner = "TIE"
            diff_color = ""
        
        print(f"{'TOTAL':<25} {total_bytes:<10,} "
              f"{baseline_color}{total_baseline_tokens:<9,}{RESET} "
              f"{baseline_color}{total_baseline_ratio:<10.2f}{RESET} "
              f"{ours_color}{total_ours_tokens:<9,}{RESET} "
              f"{ours_color}{total_ours_ratio:<10.2f}{RESET} "
              f"{diff_color}{total_relative_diff:+8.1f}%{RESET}      "
              f"{winner:<10}")
    
    # Log to report
    try:
        from nanochat.report import get_report
        
        lines = []
        lines.append("# Tokenizer Evaluation on FineTranslations Dataset")
        lines.append("")
        lines.append("## Vocabulary Sizes")
        lines.append("")
        for name, tokenizer in tokenizer_configs:
            lines.append(f"- **{name}**: {tokenizer.get_vocab_size():,} tokens")
        lines.append("")
        
        for baseline_name in ["gpt2", "gpt4"]:
            baseline_results = results[baseline_name]
            ours_results = results["ours"]
            
            lines.append(f"## Comparison with {baseline_name.upper()}")
            lines.append("")
            lines.append(f"| Sample | Bytes | {baseline_name.upper()} Tokens | {baseline_name.upper()} Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
            lines.append("|--------|-------|----------------|------------------|-------------|------------|-----------------|")
            
            total_baseline_tokens = 0
            total_ours_tokens = 0
            total_bytes = 0
            
            for sample_name, text in samples:
                baseline_data = baseline_results[sample_name]
                ours_data = ours_results[sample_name]
                
                total_baseline_tokens += baseline_data['tokens']
                total_ours_tokens += ours_data['tokens']
                total_bytes += baseline_data['bytes']
                
                relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
                
                lines.append(f"| {sample_name} | {baseline_data['bytes']:,} | {baseline_data['tokens']:,} | "
                           f"{baseline_data['ratio']:.2f} | {ours_data['tokens']:,} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
            
            # Add totals row
            total_baseline_ratio = total_bytes / total_baseline_tokens
            total_ours_ratio = total_bytes / total_ours_tokens
            total_relative_diff = ((total_baseline_tokens - total_ours_tokens) / total_baseline_tokens) * 100
            
            lines.append(f"| **TOTAL** | **{total_bytes:,}** | **{total_baseline_tokens:,}** | "
                       f"**{total_baseline_ratio:.2f}** | **{total_ours_tokens:,}** | **{total_ours_ratio:.2f}** | **{total_relative_diff:+.1f}%** |")
            lines.append("")
        
        report_markdown = "\n".join(lines)
        get_report().log(section="Tokenizer evaluation - FineTranslations", data=[report_markdown])
        print("\n✓ Results logged to report")
        
    except Exception as e:
        print(f"\nWarning: Could not log to report: {e}")


def main():
    """Main evaluation pipeline."""
    script_dir = Path(__file__).parent.parent / "dev"
    output_dir = script_dir / "finetranslations_output"
    
    print("=" * 80)
    print("TOKENIZER EVALUATION ON FINETRANSLATIONS DATASET")
    print("=" * 80)
    print(f"\nLoading samples from: {output_dir}")
    
    # Load samples from parquet shards
    samples = load_finetranslations_samples(output_dir, max_chars_per_shard=100_000)
    
    if not samples:
        print("\nNo samples loaded. Exiting.")
        sys.exit(1)
    
    total_chars = sum(len(text) for _, text in samples)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Total characters: {total_chars:,}")
    
    # Evaluate tokenizers
    evaluate_tokenizers(samples)
    
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
