"""
Standalone benchmark for the Mamba-3 SSD Triton forward kernel.

Measures kernel performance (latency, throughput) without the full model.
Useful for profiling kernel-level optimizations in isolation.

Usage:
    uv run python -m scripts.bench_triton_kernel                          # all profiles
    uv run python -m scripts.bench_triton_kernel --profile medium         # single profile
    uv run python -m scripts.bench_triton_kernel --sweep-chunk-size       # sweep chunk sizes
    uv run python -m scripts.bench_triton_kernel --batch=4 --seq-len=4096 # custom config
"""

import argparse
import torch
import triton

from nanochat.ssd_triton import mamba3_chunk_scan_fwd, _select_block_sizes


PROFILES = {
    "small":  dict(batch=2, seq_len=512,  d_model=256,  expand=1, d_state=64,  chunk_size=128, mimo_rank=1, ngroups=1),
    "medium-64chunk": dict(batch=2, seq_len=2048, d_model=768,  expand=1, d_state=64,  chunk_size=64, mimo_rank=2, ngroups=1),
    "medium-128chunk": dict(batch=2, seq_len=2048, d_model=768,  expand=1, d_state=64,  chunk_size=128, mimo_rank=2, ngroups=1),
    "medium-256chunk": dict(batch=2, seq_len=2048, d_model=768,  expand=1, d_state=64,  chunk_size=256, mimo_rank=2, ngroups=1),
    "large":  dict(batch=2, seq_len=8192, d_model=1280, expand=1, d_state=64,  chunk_size=256, mimo_rank=2, ngroups=1),
    "xlarge-1ngroup": dict(batch=1, seq_len=8192, d_model=2560, expand=1, d_state=128, chunk_size=256, mimo_rank=4, ngroups=1),
    "xlarge-2ngroups": dict(batch=1, seq_len=8192, d_model=2560, expand=1, d_state=128, chunk_size=256, mimo_rank=4, ngroups=2),
    "xlarge-4ngroups": dict(batch=1, seq_len=8192, d_model=2560, expand=1, d_state=128, chunk_size=256, mimo_rank=4, ngroups=4),
    "xlarge-8ngroups": dict(batch=1, seq_len=8192, d_model=2560, expand=1, d_state=128, chunk_size=256, mimo_rank=4, ngroups=8),
}


def make_inputs(batch, seq_len, d_model, expand, d_state, chunk_size, mimo_rank, ngroups, device="cuda"):
    """Generate realistic inputs matching the training path in mamba3.py."""
    d_inner = expand * d_model
    n_heads = d_inner // d_state
    headdim = d_inner // n_heads
    R = mimo_rank
    nc = seq_len // chunk_size
    L = chunk_size
    half_d = d_state // 2

    # Bg/Bb: bfloat16 (matches training dtype)
    Bg = torch.randn(batch, nc, L, ngroups, R, d_state, device=device, dtype=torch.bfloat16)
    Bb = torch.randn(batch, nc, L, ngroups, R, d_state, device=device, dtype=torch.bfloat16)

    # x_dt: float32 (explicitly cast in mamba3.py:678-679)
    x_dt_g = torch.randn(batch, nc, L, n_heads, R, headdim, device=device, dtype=torch.float32)
    x_dt_b = torch.randn(batch, nc, L, n_heads, R, headdim, device=device, dtype=torch.float32)

    # dA_cumsum: float32, monotonically non-increasing (A < 0, dt > 0)
    dA_cumsum = (-torch.rand(batch, nc, L, n_heads, device=device, dtype=torch.float32) * 0.1).cumsum(dim=2)

    # C: float32
    C = torch.randn(batch, nc, L, ngroups, d_state, device=device, dtype=torch.float32)

    # Pre-computed cos/sin: float32
    angles = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5
    angles_s = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5
    cos_ang = torch.cos(angles)
    sin_ang = torch.sin(angles)
    cos_angs = torch.cos(angles_s)
    sin_angs = torch.sin(angles_s)

    scale = (d_state * R) ** -0.5

    return (Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
            cos_ang, sin_ang, cos_angs, sin_angs, scale, L, R,
            n_heads, headdim, nc)


def compute_bytes(batch, nc, L, ngroups, R, d_state, n_heads, headdim):
    """Total bytes read + written by the kernel."""
    half_d = d_state // 2
    # Reads (bf16 for B, f32 for rest)
    b_bytes = 2 * (batch * nc * L * ngroups * R * d_state * 2)       # Bg + Bb in bf16
    xdt_bytes = 2 * (batch * nc * L * n_heads * R * headdim * 4)     # x_dt_g + x_dt_b in f32
    dA_bytes = batch * nc * L * n_heads * 4                           # dA_cumsum in f32
    c_bytes = batch * nc * L * ngroups * d_state * 4                  # C in f32
    cs_bytes = 4 * (batch * nc * L * half_d * 4)                      # 4 cos/sin tensors in f32
    # Write
    out_bytes = batch * nc * L * n_heads * headdim * 4                # output in f32

    return b_bytes + xdt_bytes + dA_bytes + c_bytes + cs_bytes + out_bytes


def bench_config(cfg, warmup=25, rep=100):
    """Benchmark a single config. Returns (latency_ms, throughput_gb_s, block_info)."""
    inputs = make_inputs(**cfg)
    Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C, cos_ang, sin_ang, cos_angs, sin_angs, scale, L, R, n_heads, headdim, nc = inputs

    d_state = cfg["d_state"]
    ngroups = cfg["ngroups"]
    batch = cfg["batch"]

    # Get block sizes
    BM, BN, BK, BD, stages = _select_block_sizes(L, headdim, d_state, device=Bg.device)
    block_info = f"BLOCK_M={BM} BLOCK_N={BN} BLOCK_K={BK} BLOCK_DSTATE={BD} stages={stages}"

    # Benchmark
    fn = lambda: mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                        cos_ang, sin_ang, cos_angs, sin_angs,
                                        scale, L, R)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    total_bytes = compute_bytes(batch, nc, L, ngroups, R, d_state, n_heads, headdim)
    gb_s = total_bytes / (ms * 1e-3) / 1e9

    input_mb = sum(t.nelement() * t.element_size() for t in [Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                                               cos_ang, sin_ang, cos_angs, sin_angs]) / 1e6
    out_mb = (batch * nc * L * n_heads * headdim * 4) / 1e6

    return ms, gb_s, block_info, input_mb, out_mb


def get_peak_bw():
    """Get approximate peak HBM bandwidth for the current GPU (GB/s)."""
    name = torch.cuda.get_device_name().lower()
    # Common GPUs and their approximate peak HBM bandwidths
    if "b300" in name:
        return 8000
    elif "b200" in name:
        return 8000
    elif "h200" in name:
        return 4800
    elif "h100" in name and "sxm" in name:
        return 3350
    elif "h100" in name:
        return 2000  # PCIe
    elif "a100" in name and "sxm" in name:
        return 2039
    elif "a100" in name:
        return 1555  # PCIe
    elif "4090" in name:
        return 1008
    elif "3090" in name:
        return 936
    elif "l40" in name:
        return 864
    return 0  # unknown


def print_result(name, cfg, ms, gb_s, block_info, input_mb, out_mb, peak_bw):
    """Print formatted benchmark result."""
    parts = [f"batch={cfg['batch']}", f"T={cfg['seq_len']}", f"d={cfg['d_model']}",
             f"d_state={cfg['d_state']}", f"chunk={cfg['chunk_size']}", f"R={cfg['mimo_rank']}"]
    desc = ", ".join(parts)
    bw_pct = f" ({gb_s / peak_bw * 100:.1f}% of peak)" if peak_bw else ""

    print(f"Config: {name} ({desc})")
    print(f"  Blocks: {block_info}")
    print(f"  Input: {input_mb:.1f} MB | Output: {out_mb:.1f} MB")
    print(f"  Latency: {ms:.3f} ms")
    print(f"  Throughput: {gb_s:.1f} GB/s{bw_pct}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba-3 SSD Triton forward kernel")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), help="Run a specific profile (default: all)")
    parser.add_argument("--sweep-chunk-size", action="store_true", help="Sweep chunk_size in {64, 128, 256}")
    parser.add_argument("--sweep-mimo-rank", action="store_true", help="Sweep mimo_rank in {1, 2, 4}")
    parser.add_argument("--sweep-d-state", action="store_true", help="Sweep d_state in {64, 128}")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    # Custom config overrides
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--expand", type=int)
    parser.add_argument("--d-state", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--mimo-rank", type=int)
    parser.add_argument("--ngroups", type=int)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    gpu_name = torch.cuda.get_device_name()
    peak_bw = get_peak_bw()
    bw_str = f" | Peak BW: {peak_bw} GB/s" if peak_bw else ""

    print(f"\nMamba-3 SSD Triton Forward Kernel Benchmark")
    print(f"GPU: {gpu_name}{bw_str}")
    sep = "=" * 60

    # Check if custom config provided
    custom_keys = ["batch", "seq_len", "d_model", "expand", "d_state", "chunk_size", "mimo_rank", "ngroups"]
    has_custom = any(getattr(args, k.replace("-", "_")) is not None for k in custom_keys)

    if has_custom:
        # Build custom config from defaults + overrides
        base = dict(PROFILES["medium"])
        for k in custom_keys:
            v = getattr(args, k.replace("-", "_"))
            if v is not None:
                base[k] = v
        # Validate seq_len divisible by chunk_size
        assert base["seq_len"] % base["chunk_size"] == 0, \
            f"seq_len ({base['seq_len']}) must be divisible by chunk_size ({base['chunk_size']})"
        d_inner = base["expand"] * base["d_model"]
        assert d_inner % base["d_state"] == 0, \
            f"expand*d_model ({d_inner}) must be divisible by d_state ({base['d_state']})"

        print(sep)
        ms, gb_s, bi, imb, omb = bench_config(base, args.warmup, args.rep)
        print_result("custom", base, ms, gb_s, bi, imb, omb, peak_bw)
        print(sep)
        return

    # Determine which profiles to run
    profiles = {args.profile: PROFILES[args.profile]} if args.profile else PROFILES

    # Any sweep active?
    any_sweep = args.sweep_chunk_size or args.sweep_mimo_rank or args.sweep_d_state

    if any_sweep:
        for pname, base_cfg in profiles.items():
            sweep_values = []
            if args.sweep_chunk_size:
                for cs in [64, 128, 256]:
                    if base_cfg["seq_len"] % cs == 0:
                        c = dict(base_cfg, chunk_size=cs)
                        sweep_values.append((f"{pname}/chunk={cs}", c))
            if args.sweep_mimo_rank:
                for r in [1, 2, 4]:
                    c = dict(base_cfg, mimo_rank=r)
                    sweep_values.append((f"{pname}/R={r}", c))
            if args.sweep_d_state:
                for ds in [64, 128]:
                    d_inner = base_cfg["expand"] * base_cfg["d_model"]
                    if d_inner % ds == 0:
                        c = dict(base_cfg, d_state=ds)
                        sweep_values.append((f"{pname}/d_state={ds}", c))

            if sweep_values:
                print(sep)
                for name, cfg in sweep_values:
                    ms, gb_s, bi, imb, omb = bench_config(cfg, args.warmup, args.rep)
                    print_result(name, cfg, ms, gb_s, bi, imb, omb, peak_bw)
                    print("-" * 60)
        print(sep)
    else:
        # Run all selected profiles
        print(sep)
        for pname, cfg in profiles.items():
            ms, gb_s, bi, imb, omb = bench_config(cfg, args.warmup, args.rep)
            print_result(pname, cfg, ms, gb_s, bi, imb, omb, peak_bw)
            print("-" * 60)
        print(sep)


if __name__ == "__main__":
    main()
