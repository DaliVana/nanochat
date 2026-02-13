"""
Compare sparsity variants (MoD, MoH, MoE) against baseline.

Trains all variants with identical configuration and compares:
- Training loss curves
- Validation BPB
- FLOPs per forward pass
- Throughput (tokens/sec)
- Memory usage
- Routing statistics

Usage:
python -m scripts.compare_sparsity --depth=4 --max-seq-len=512 --num-iterations=100

This will train:
1. Baseline GPT (from base_train.py)
2. MoD (Mixture of Depth)
3. MoH (Mixture of Heads)
4. MoE (Mixture of Experts)

And generate a comparison report at dev/sparsity_comparison.md
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def run_variant(variant_name, script_name, args_dict, run_name, sparsity_args=None):
    """
    Run a single variant and capture metrics.

    Args:
        variant_name: Human-readable name (e.g., "Baseline", "MoD")
        script_name: Python module to run (e.g., "scripts.base_train")
        args_dict: Dictionary of arguments to pass
        run_name: WandB run name
        sparsity_args: List of command line args for sparsity settings (or None for baseline)

    Returns:
        Dict with metrics (or None if failed)
    """
    print(f"\n{'='*80}")
    print(f"Training {variant_name}")
    print(f"{'='*80}\n")

    # Build command
    cmd = ["python", "-m", script_name]
    cmd.append(f"--run={run_name}")

    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")

    # Add variant-specific args
    if sparsity_args:
        cmd.extend(sparsity_args)

    print(f"Running: {' '.join(cmd)}\n")

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"ERROR: {variant_name} training failed!")
            print(f"STDERR: {result.stderr}")
            return None

        # Parse output for metrics (basic parsing - can be improved)
        output = result.stdout
        metrics = {
            "variant": variant_name,
            "training_time": elapsed_time,
            "success": True,
            "sparsity_args": sparsity_args if sparsity_args else [],
        }

        # Extract key metrics from output
        # Look for lines like "Final validation bpb: X.XXXXXX"
        for line in output.split('\n'):
            if "validation bpb" in line.lower():
                try:
                    bpb = float(line.split(':')[-1].strip())
                    metrics["val_bpb"] = bpb
                except:
                    pass
            elif "peak memory usage" in line.lower():
                try:
                    mem_str = line.split(':')[-1].strip()
                    mem_val = float(mem_str.replace('MiB', '').replace('GiB', '').strip())
                    metrics["peak_memory_mib"] = mem_val
                except:
                    pass
            elif "tok/s" in line.lower() or "tokens/sec" in line.lower():
                try:
                    # Extract number before "tok/s" or "tokens/sec"
                    parts = line.lower().replace('tokens/sec', 'tok/s').split('tok/s')[0].strip()
                    tok_s = float(parts.split()[-1])
                    metrics["tokens_per_sec"] = tok_s
                except:
                    pass
            elif "mfu" in line.lower() and "%" in line:
                try:
                    # Extract percentage value
                    mfu_str = line.lower().split('mfu')[-1].strip()
                    mfu_val = float(mfu_str.replace('%', '').split()[0].strip())
                    metrics["mfu_percent"] = mfu_val
                except:
                    pass

        print(f"\n{variant_name} completed in {elapsed_time:.1f}s")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: {variant_name} training timed out!")
        return None
    except Exception as e:
        print(f"ERROR: {variant_name} training failed with exception: {e}")
        return None


def generate_report(results, output_path):
    """Generate markdown comparison report."""

    # Filter out failed runs
    results = [r for r in results if r is not None]

    if not results:
        print("ERROR: All variants failed, cannot generate report")
        return

    report = []
    report.append("# Sparsity Variants Comparison")
    report.append("")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary table
    report.append("## Summary")
    report.append("")
    report.append("| Variant | Training Time (s) | Val BPB | Tokens/s | MFU (%) | Peak Memory (MiB) |")
    report.append("|---------|------------------|---------|----------|---------|-------------------|")

    for r in results:
        training_time = f"{r.get('training_time', 0):.1f}"
        val_bpb = f"{r.get('val_bpb', 'N/A'):.6f}" if isinstance(r.get('val_bpb'), float) else "N/A"
        tokens_per_sec = f"{r.get('tokens_per_sec', 'N/A'):.1f}" if isinstance(r.get('tokens_per_sec'), (int, float)) else "N/A"
        mfu = f"{r.get('mfu_percent', 'N/A'):.2f}" if isinstance(r.get('mfu_percent'), (int, float)) else "N/A"
        peak_mem = f"{r.get('peak_memory_mib', 'N/A'):.1f}" if isinstance(r.get('peak_memory_mib'), float) else "N/A"

        report.append(f"| {r['variant']} | {training_time} | {val_bpb} | {tokens_per_sec} | {mfu} | {peak_mem} |")

    report.append("")

    # Detailed results
    report.append("## Detailed Results")
    report.append("")

    for r in results:
        report.append(f"### {r['variant']}")
        report.append("")
        report.append("```json")
        report.append(json.dumps(r, indent=2))
        report.append("```")
        report.append("")

    # Write report
    report_content = '\n'.join(report)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"\nComparison report saved to: {output_path}")
    print("\n" + "="*80)
    print("REPORT PREVIEW:")
    print("="*80)
    print(report_content)


def main():
    parser = argparse.ArgumentParser(description="Compare sparsity variants")

    # Model config
    parser.add_argument("--depth", type=int, default=4, help="Model depth")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Device batch size")
    parser.add_argument("--total-batch-size", type=int, default=4096, help="Total batch size")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--eval-tokens", type=int, default=16384, help="Tokens for evaluation")  # 4 eval batches (4 * 8 * 512 = 16384)

    # Comparison options
    parser.add_argument("--variants", type=str, default="all",
                       help="Comma-separated list of variants to test (baseline,mod,moh,moe) or 'all'")
    parser.add_argument("--output", type=str, default="dev/sparsity_comparison.md",
                       help="Output path for comparison report")

    args = parser.parse_args()

    # Common training arguments
    common_args = {
        "depth": args.depth,
        "max-seq-len": args.max_seq_len,
        "device-batch-size": args.device_batch_size,
        "total-batch-size": args.total_batch_size,
        "num-iterations": args.num_iterations,
        "eval-every": args.eval_every,
        "eval-tokens": args.eval_tokens,
        "core-metric-every": -1,  # Disable CORE for faster comparison
        "sample-every": -1,  # Disable sampling for faster comparison
        "save-every": -1,  # Don't save checkpoints during comparison
    }

    # Define all runs with their sparsity configurations
    # Format: (variant_name, script_name, run_suffix, sparsity_args)
    all_runs = [
        # 1. Baseline
        ("Baseline", "scripts.base_train", "baseline", None),

        # 2-4. MoD with different sparsity levels
        ("MoD (50%)", "scripts.mod_train", "mod_50", ["--mod-top-k-ratio=0.5"]),
        ("MoD (25%)", "scripts.mod_train", "mod_25", ["--mod-top-k-ratio=0.25"]),
        ("MoD (12.5%)", "scripts.mod_train", "mod_12.5", ["--mod-top-k-ratio=0.125"]),

        # 5-7. MoH with different sparsity levels
        ("MoH (50%)", "scripts.moh_train", "moh_50", ["--moh-active-heads-ratio=0.5"]),
        ("MoH (25%)", "scripts.moh_train", "moh_25", ["--moh-active-heads-ratio=0.25"]),
        ("MoH (12.5%)", "scripts.moh_train", "moh_12.5", ["--moh-active-heads-ratio=0.125"]),

        # 8-10. MoE with different expert configurations
        ("MoE (4x2)", "scripts.moe_train", "moe_4x2",
         ["--moe-num-experts=4", "--moe-experts-per-tok=2"]),
        ("MoE (8x2)", "scripts.moe_train", "moe_8x2",
         ["--moe-num-experts=8", "--moe-experts-per-tok=2", "--moe-expert-hidden-ratio=0.125"]),
        ("MoE (8x1)", "scripts.moe_train", "moe_8x1",
         ["--moe-num-experts=8", "--moe-experts-per-tok=1", "--moe-expert-hidden-ratio=0.125"]),
    ]

    # Run all configurations
    results = []

    for variant_name, script_name, run_suffix, sparsity_args in all_runs:
        run_name = f"compare_{run_suffix}_d{args.depth}"

        metrics = run_variant(variant_name, script_name, common_args, run_name, sparsity_args)
        if metrics:
            results.append(metrics)

    # Generate report
    if results:
        generate_report(results, args.output)
    else:
        print("ERROR: No variants completed successfully")


if __name__ == "__main__":
    main()
