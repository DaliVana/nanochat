"""
Standalone benchmark for the Mamba-3 SSD Triton kernel and full layer.

Measures kernel and layer performance (latency, throughput, compute) without
the full model. Supports saving results to JSON and comparing against a
baseline to detect regressions.

Usage:
    uv run python -m scripts.bench_triton_kernel                                   # kernel only
    uv run python -m scripts.bench_triton_kernel --layer                           # kernel + layer
    uv run python -m scripts.bench_triton_kernel --save baseline.json              # save results
    uv run python -m scripts.bench_triton_kernel --compare baseline.json           # compare vs saved
    uv run python -m scripts.bench_triton_kernel --profile medium-128chunk --layer # single profile
"""

import argparse
import json
import torch
import triton

from nanochat.ssd_triton import mamba3_chunk_scan_fwd, _mamba3_chunk_scan_fwd_kernel


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

# Subset of profiles for layer benchmarks (full layer is slower to bench)
LAYER_PROFILES = {
    "small":  PROFILES["small"],
    "medium-128chunk": PROFILES["medium-128chunk"],
    "large":  PROFILES["large"],
}


# ── Kernel-level benchmark ──────────────────────────────────────────────────

def make_inputs(batch, seq_len, d_model, expand, d_state, chunk_size, mimo_rank, ngroups, device="cuda"):
    """Generate realistic inputs matching the training path in mamba3.py."""
    d_inner = expand * d_model
    n_heads = d_inner // d_state
    headdim = d_inner // n_heads
    R = mimo_rank
    nc = seq_len // chunk_size
    L = chunk_size
    half_d = d_state // 2

    Bg = torch.randn(batch, nc, L, ngroups, R, d_state, device=device, dtype=torch.bfloat16)
    Bb = torch.randn(batch, nc, L, ngroups, R, d_state, device=device, dtype=torch.bfloat16)
    x_dt_g = torch.randn(batch, nc, L, n_heads, R, headdim, device=device, dtype=torch.float32)
    x_dt_b = torch.randn(batch, nc, L, n_heads, R, headdim, device=device, dtype=torch.float32)
    dA_cumsum = (-torch.rand(batch, nc, L, n_heads, device=device, dtype=torch.float32) * 0.1).cumsum(dim=2)
    C = torch.randn(batch, nc, L, ngroups, d_state, device=device, dtype=torch.float32)

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
    b_bytes = 2 * (batch * nc * L * ngroups * R * d_state * 2)
    xdt_bytes = 2 * (batch * nc * L * n_heads * R * headdim * 4)
    dA_bytes = batch * nc * L * n_heads * 4
    c_bytes = batch * nc * L * ngroups * d_state * 4
    cs_bytes = 4 * (batch * nc * L * half_d * 4)
    out_bytes = batch * nc * L * n_heads * headdim * 4
    return b_bytes + xdt_bytes + dA_bytes + c_bytes + cs_bytes + out_bytes


def compute_flops(batch, nc, L, n_heads, R, d_state, headdim):
    """Total FLOPs for the kernel (dot products only, dominates compute).

    Per (batch, chunk, head, rank), the kernel computes:
      CB scores: C_rot (L, dstate) @ B_rot (L, dstate)^T -> causal (L, L)
        = L^2/2 * 2*dstate per SSD, x2 for gamma+beta = 2 * L^2 * dstate
      Final accum: scores (L, L, causal) @ x_dt (L, headdim) -> (L, headdim)
        = L^2/2 * 2*headdim per SSD, x2 for gamma+beta = 2 * L^2 * headdim
    """
    flops_per_head_rank = 2 * L * L * (d_state + headdim)
    return batch * nc * n_heads * R * flops_per_head_rank


def bench_kernel(cfg, warmup=25, rep=100):
    """Benchmark the Triton forward kernel for one config."""
    inputs = make_inputs(**cfg)
    Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C, cos_ang, sin_ang, cos_angs, sin_angs, scale, L, R, n_heads, headdim, nc = inputs

    d_state = cfg["d_state"]
    ngroups = cfg["ngroups"]
    batch = cfg["batch"]

    fn = lambda: mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                        cos_ang, sin_ang, cos_angs, sin_angs,
                                        scale, L, R)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # Query block sizes chosen by @triton.autotune (available after first launch)
    best = _mamba3_chunk_scan_fwd_kernel.best_config
    bk = best.kwargs
    block_info = (f"BLOCK_M={bk['BLOCK_M']} BLOCK_N={bk['BLOCK_N']} BLOCK_K={bk['BLOCK_K']} "
                  f"BLOCK_DSTATE={d_state // 2} stages={best.num_stages} warps={best.num_warps}")

    total_bytes = compute_bytes(batch, nc, L, ngroups, R, d_state, n_heads, headdim)
    gb_s = total_bytes / (ms * 1e-3) / 1e9
    total_flops = compute_flops(batch, nc, L, n_heads, R, d_state, headdim)
    tflops = total_flops / (ms * 1e-3) / 1e12

    input_mb = sum(t.nelement() * t.element_size() for t in [Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                                               cos_ang, sin_ang, cos_angs, sin_angs]) / 1e6
    out_mb = (batch * nc * L * n_heads * headdim * 4) / 1e6

    return dict(ms=ms, gb_s=gb_s, tflops=tflops, block_info=block_info,
                input_mb=input_mb, out_mb=out_mb)


# ── Layer-level benchmark ───────────────────────────────────────────────────

def make_layer(d_model, expand, d_state, chunk_size, mimo_rank, ngroups, device="cuda", dtype=torch.bfloat16):
    """Create a Mamba3Layer on the given device."""
    from nanochat.mamba3 import Mamba3Layer
    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                        chunk_size=chunk_size, mimo_rank=mimo_rank, ngroups=ngroups)
    layer.to(device=device, dtype=dtype)
    layer.eval()
    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)
    return layer


def bench_layer_train(cfg, warmup=10, rep=50):
    """Benchmark full layer forward + backward (training speed)."""
    layer = make_layer(cfg["d_model"], cfg["expand"], cfg["d_state"],
                       cfg["chunk_size"], cfg["mimo_rank"], cfg["ngroups"])
    layer.train()

    batch, T = cfg["batch"], cfg["seq_len"]
    x = torch.randn(batch, T, cfg["d_model"], device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward + backward
    def fwd_bwd():
        y, _, _ = layer(x, ssm_state=None)
        y.sum().backward()
        # Clear grads for next iteration (don't accumulate)
        if x.grad is not None:
            x.grad = None
        for p in layer.parameters():
            p.grad = None

    ms = triton.testing.do_bench(fwd_bwd, warmup=warmup, rep=rep)
    # Tokens/sec
    tokens_per_sec = (batch * T) / (ms * 1e-3)
    return dict(ms=ms, tokens_per_sec=tokens_per_sec)


def bench_layer_inference(cfg, warmup=25, rep=100):
    """Benchmark single-token recurrent inference step."""
    layer = make_layer(cfg["d_model"], cfg["expand"], cfg["d_state"],
                       cfg["chunk_size"], cfg["mimo_rank"], cfg["ngroups"])
    batch = cfg["batch"]
    d_state = cfg["d_state"]
    n_heads = (cfg["expand"] * cfg["d_model"]) // d_state
    headdim = (cfg["expand"] * cfg["d_model"]) // n_heads

    # Pre-allocate SSM state in float32: the recurrent loop promotes h to float32
    # because A is computed in float32 (A = -exp(A_log.float())), so alpha_t * h
    # upcasts h on the first step. Match that steady-state dtype here.
    h = torch.zeros(batch, n_heads, headdim, d_state, device="cuda", dtype=torch.float32)
    prev_Bx = torch.zeros_like(h)
    ssm_state = {'h': h, 'prev_Bx': prev_Bx}

    x = torch.randn(batch, 1, cfg["d_model"], device="cuda", dtype=torch.bfloat16)

    def step():
        with torch.no_grad():
            layer(x, ssm_state=ssm_state)

    ms = triton.testing.do_bench(step, warmup=warmup, rep=rep)
    tokens_per_sec = batch / (ms * 1e-3)
    return dict(ms=ms, tokens_per_sec=tokens_per_sec)


# ── GPU specs ───────────────────────────────────────────────────────────────

def get_gpu_specs():
    """Get approximate peak HBM bandwidth (GB/s) and TF32 dense TFLOP/s for the current GPU."""
    name = torch.cuda.get_device_name().lower()
    specs = [
        ("b300",  8000, 2250),
        ("b200",  8000, 2250),
        ("h200",  4800,  495),
        ("h100",  3350,  495, "sxm"),
        ("h100",  2000,  378),
        ("a100",  2039,  156, "sxm"),
        ("a100",  1555,  156),
        ("4090",  1008,   83),
        ("3090",   936,   36),
        ("l40",    864,   91),
    ]
    for entry in specs:
        gpu_key, bw, tflops = entry[0], entry[1], entry[2]
        variant = entry[3] if len(entry) > 3 else None
        if gpu_key in name and (variant is None or variant in name):
            return bw, tflops
    return 0, 0


# ── Display ─────────────────────────────────────────────────────────────────

def cfg_desc(cfg):
    parts = [f"batch={cfg['batch']}", f"T={cfg['seq_len']}", f"d={cfg['d_model']}",
             f"d_state={cfg['d_state']}", f"chunk={cfg['chunk_size']}", f"R={cfg['mimo_rank']}"]
    return ", ".join(parts)


def fmt_delta(cur, base):
    """Format a value with delta vs baseline. Positive = slower (red), negative = faster (green)."""
    if base is None:
        return ""
    pct = (cur - base) / base * 100
    arrow = "+" if pct >= 0 else ""
    return f"  [{arrow}{pct:.1f}% vs baseline]"


def print_kernel_result(name, cfg, r, peak_bw, peak_tflops, baseline=None):
    bw_pct = f" ({r['gb_s'] / peak_bw * 100:.1f}% of peak)" if peak_bw else ""
    compute_pct = f" ({r['tflops'] / peak_tflops * 100:.1f}% of peak)" if peak_tflops else ""
    b = baseline.get(f"kernel/{name}") if baseline else None
    lat_delta = fmt_delta(r["ms"], b["ms"] if b else None)

    print(f"  [{name}] ({cfg_desc(cfg)})")
    print(f"    Blocks: {r['block_info']}")
    print(f"    Input: {r['input_mb']:.1f} MB | Output: {r['out_mb']:.1f} MB")
    print(f"    Latency:    {r['ms']:.3f} ms{lat_delta}")
    print(f"    Throughput: {r['gb_s']:.1f} GB/s{bw_pct}")
    print(f"    Compute:    {r['tflops']:.1f} TFLOP/s{compute_pct}")


def print_layer_result(name, kind, r, baseline=None):
    b = baseline.get(f"{kind}/{name}") if baseline else None
    lat_delta = fmt_delta(r["ms"], b["ms"] if b else None)
    tok_delta = fmt_delta(r["tokens_per_sec"], b["tokens_per_sec"] if b else None)
    label = "Train fwd+bwd" if kind == "train" else "Inference step"
    print(f"    {label}: {r['ms']:.3f} ms  ({r['tokens_per_sec']:.0f} tok/s){lat_delta}{tok_delta}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba-3 SSD Triton kernel and layer")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), help="Run a specific profile (default: all)")
    parser.add_argument("--layer", action="store_true", help="Also benchmark full layer (train + inference)")
    parser.add_argument("--layer-only", action="store_true", help="Only run layer benchmarks (skip kernel)")
    parser.add_argument("--sweep-chunk-size", action="store_true", help="Sweep chunk_size in {64, 128, 256}")
    parser.add_argument("--sweep-mimo-rank", action="store_true", help="Sweep mimo_rank in {1, 2, 4}")
    parser.add_argument("--sweep-d-state", action="store_true", help="Sweep d_state in {64, 128}")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--save", type=str, metavar="FILE", help="Save results to JSON for later comparison")
    parser.add_argument("--compare", type=str, metavar="FILE", help="Compare against a saved baseline JSON")
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
    do_layer = args.layer or args.layer_only
    do_kernel = not args.layer_only

    gpu_name = torch.cuda.get_device_name()
    peak_bw, peak_tflops = get_gpu_specs()
    specs_parts = []
    if peak_bw:
        specs_parts.append(f"Peak BW: {peak_bw} GB/s")
    if peak_tflops:
        specs_parts.append(f"Peak TF32: {peak_tflops} TFLOP/s")
    specs_str = f" | {' | '.join(specs_parts)}" if specs_parts else ""

    # Load baseline for comparison
    baseline = None
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        print(f"\nComparing against: {args.compare}")

    print(f"\nMamba-3 SSD Benchmark")
    print(f"GPU: {gpu_name}{specs_str}")
    sep = "=" * 70

    results = {"gpu": gpu_name}

    # Check if custom config provided
    custom_keys = ["batch", "seq_len", "d_model", "expand", "d_state", "chunk_size", "mimo_rank", "ngroups"]
    has_custom = any(getattr(args, k.replace("-", "_")) is not None for k in custom_keys)

    if has_custom:
        base = dict(PROFILES["medium-128chunk"])
        for k in custom_keys:
            v = getattr(args, k.replace("-", "_"))
            if v is not None:
                base[k] = v
        assert base["seq_len"] % base["chunk_size"] == 0
        d_inner = base["expand"] * base["d_model"]
        assert d_inner % base["d_state"] == 0

        print(sep)
        if do_kernel:
            r = bench_kernel(base, args.warmup, args.rep)
            results[f"kernel/custom"] = r
            print_kernel_result("custom", base, r, peak_bw, peak_tflops, baseline)
        if do_layer:
            rt = bench_layer_train(base, max(1, args.warmup // 2), max(10, args.rep // 2))
            ri = bench_layer_inference(base, args.warmup, args.rep)
            results[f"train/custom"] = rt
            results[f"inference/custom"] = ri
            print_layer_result("custom", "train", rt, baseline)
            print_layer_result("custom", "inference", ri, baseline)
        print(sep)
    else:
        # Determine profiles
        if args.profile:
            kernel_profiles = {args.profile: PROFILES[args.profile]}
            layer_profiles = {args.profile: PROFILES[args.profile]}
        else:
            kernel_profiles = PROFILES
            layer_profiles = LAYER_PROFILES

        any_sweep = args.sweep_chunk_size or args.sweep_mimo_rank or args.sweep_d_state

        def collect_configs(profiles):
            """Return list of (name, cfg) including sweeps if active."""
            configs = []
            if any_sweep:
                for pname, base_cfg in profiles.items():
                    if args.sweep_chunk_size:
                        for cs in [64, 128, 256]:
                            if base_cfg["seq_len"] % cs == 0:
                                configs.append((f"{pname}/chunk={cs}", dict(base_cfg, chunk_size=cs)))
                    if args.sweep_mimo_rank:
                        for r in [1, 2, 4]:
                            configs.append((f"{pname}/R={r}", dict(base_cfg, mimo_rank=r)))
                    if args.sweep_d_state:
                        for ds in [64, 128]:
                            d_inner = base_cfg["expand"] * base_cfg["d_model"]
                            if d_inner % ds == 0:
                                configs.append((f"{pname}/d_state={ds}", dict(base_cfg, d_state=ds)))
            else:
                configs = list(profiles.items())
            return configs

        # Kernel benchmarks
        if do_kernel:
            print(sep)
            print("KERNEL (Triton forward)")
            print(sep)
            for name, cfg in collect_configs(kernel_profiles):
                r = bench_kernel(cfg, args.warmup, args.rep)
                results[f"kernel/{name}"] = r
                print_kernel_result(name, cfg, r, peak_bw, peak_tflops, baseline)
                print("-" * 70)

        # Layer benchmarks
        if do_layer:
            print(sep)
            print("LAYER (Mamba3Layer)")
            print(sep)
            for name, cfg in collect_configs(layer_profiles):
                print(f"  [{name}] ({cfg_desc(cfg)})")
                rt = bench_layer_train(cfg, max(1, args.warmup // 2), max(10, args.rep // 2))
                ri = bench_layer_inference(cfg, args.warmup, args.rep)
                results[f"train/{name}"] = rt
                results[f"inference/{name}"] = ri
                print_layer_result(name, "train", rt, baseline)
                print_layer_result(name, "inference", ri, baseline)
                print("-" * 70)

        print(sep)

    # Save results
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
