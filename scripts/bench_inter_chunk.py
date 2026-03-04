"""
Benchmark and correctness test for the fused inter-chunk output Triton kernel.

Compares the Triton kernel against the PyTorch reference (cuBLAS einsum +
separate decay scaling + y_intra addition). Measures latency, HBM traffic
reduction, and gradient correctness.

Usage:
    uv run python -m scripts.bench_inter_chunk              # correctness + benchmark
    uv run python -m scripts.bench_inter_chunk --backward   # also test backward gradients
"""

import argparse
import torch
import triton

from nanochat.ssd_triton import inter_chunk_output_fwd


def make_inputs(batch, nc, L, n_heads, head_dim, d_state, ngroups, device="cuda"):
    """Create test tensors matching the inter-chunk output computation."""
    hpg = n_heads // ngroups
    C_rot = torch.randn(batch, nc, L, ngroups, d_state, device=device, dtype=torch.float32)
    all_states = torch.randn(batch, nc, n_heads, head_dim, d_state, device=device, dtype=torch.float32)
    cum_log_dA = (-torch.rand(batch, nc, L, n_heads, device=device, dtype=torch.float32) * 0.1).cumsum(dim=2)
    y_intra = torch.randn(batch, nc, L, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    scale = (d_state * 4) ** -0.5  # typical scale
    return C_rot, all_states, cum_log_dA, y_intra, scale, hpg


def pytorch_reference(C_rot, all_states, cum_log_dA, y_intra, scale, hpg):
    """PyTorch reference: cuBLAS einsum + decay + y_intra addition."""
    C_rot_f = C_rot.float()
    ngroups = C_rot.shape[3]
    n_heads = cum_log_dA.shape[-1]
    decay_from_start = torch.exp(cum_log_dA)
    if ngroups < n_heads:
        batch, nc, L = C_rot.shape[:3]
        head_dim, d_state = all_states.shape[-2], all_states.shape[-1]
        all_states_g = all_states.view(batch, nc, ngroups, hpg, head_dim, d_state)
        state_contribution = torch.einsum(
            'bcign, bcgkpn -> bcigkp', C_rot_f, all_states_g)
        state_contribution = state_contribution.reshape(
            batch, nc, L, n_heads, head_dim)
    else:
        state_contribution = torch.einsum(
            'bcihn, bchpn -> bcihp', C_rot_f, all_states)
    y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)
    return (y_intra + y_inter).to(y_intra.dtype)


def test_correctness(cfg, tol_fwd=1e-2, tol_bwd=5e-2, test_backward=False):
    """Test forward (and optionally backward) correctness."""
    C_rot, all_states, cum_log_dA, y_intra, scale, hpg = make_inputs(**cfg)

    # Forward
    y_ref = pytorch_reference(C_rot, all_states, cum_log_dA, y_intra, scale, hpg)
    y_triton = inter_chunk_output_fwd(C_rot, all_states, cum_log_dA, y_intra, scale, hpg)

    # Compare
    max_diff = (y_ref.float() - y_triton.float()).abs().max().item()
    rel_diff = max_diff / (y_ref.float().abs().max().item() + 1e-8)
    fwd_ok = rel_diff < tol_fwd

    print(f"  Forward:  max_abs_diff={max_diff:.6f}  rel_diff={rel_diff:.6f}  "
          f"{'PASS' if fwd_ok else 'FAIL'}")

    if not test_backward:
        return fwd_ok

    # Backward test: compare gradients via custom_op
    from nanochat.mamba3 import _inter_chunk_output_fwd_op

    # PyTorch backward
    C1 = C_rot.clone().requires_grad_()
    s1 = all_states.clone().requires_grad_()
    d1 = cum_log_dA.clone().requires_grad_()
    yi1 = y_intra.clone()
    y1 = pytorch_reference(C1, s1, d1, yi1, scale, hpg)
    dy = torch.randn_like(y1)
    y1.float().backward(dy.float())

    # Triton backward (via custom_op)
    C2 = C_rot.clone().requires_grad_()
    s2 = all_states.clone().requires_grad_()
    d2 = cum_log_dA.clone().requires_grad_()
    yi2 = y_intra.clone()
    y2 = _inter_chunk_output_fwd_op(C2, s2, d2, yi2, scale, hpg)
    y2.float().backward(dy.float())

    bwd_ok = True
    for name, g_ref, g_tri in [("dC_rot", C1.grad, C2.grad),
                                ("dall_states", s1.grad, s2.grad),
                                ("dcum_log_dA", d1.grad, d2.grad)]:
        if g_ref is None or g_tri is None:
            print(f"  Backward {name}: SKIP (no grad)")
            continue
        diff = (g_ref - g_tri).abs().max().item()
        rel = diff / (g_ref.abs().max().item() + 1e-8)
        ok = rel < tol_bwd
        bwd_ok = bwd_ok and ok
        print(f"  Backward {name}: max_abs_diff={diff:.6f}  rel_diff={rel:.6f}  "
              f"{'PASS' if ok else 'FAIL'}")

    return fwd_ok and bwd_ok


def bench(cfg, warmup=25, rep=100):
    """Benchmark Triton kernel vs PyTorch reference."""
    C_rot, all_states, cum_log_dA, y_intra, scale, hpg = make_inputs(**cfg)

    fn_pytorch = lambda: pytorch_reference(C_rot, all_states, cum_log_dA, y_intra, scale, hpg)
    fn_triton = lambda: inter_chunk_output_fwd(C_rot, all_states, cum_log_dA, y_intra, scale, hpg)

    ms_pytorch = triton.testing.do_bench(fn_pytorch, warmup=warmup, rep=rep)
    ms_triton = triton.testing.do_bench(fn_triton, warmup=warmup, rep=rep)

    # HBM traffic estimates
    batch, nc, L, ngroups, d_state = C_rot.shape
    n_heads = cum_log_dA.shape[-1]
    head_dim = y_intra.shape[-1]

    # PyTorch: read C_rot + states + cum_log_dA + y_intra, write state_contribution + y_inter + y
    pytorch_bytes = (
        C_rot.nelement() * 4  # read C_rot
        + all_states.nelement() * 4  # read states
        + cum_log_dA.nelement() * 4  # read cum_log_dA
        + batch * nc * L * n_heads * head_dim * 4  # write state_contribution (f32)
        + batch * nc * L * n_heads * head_dim * 4  # read state_contribution
        + cum_log_dA.nelement() * 4  # read cum_log_dA again (for exp)
        + batch * nc * L * n_heads * head_dim * 4  # write y_inter (f32)
        + y_intra.nelement() * 2  # read y_intra
        + batch * nc * L * n_heads * head_dim * 4  # read y_inter
        + batch * nc * L * n_heads * head_dim * 2  # write y (bf16)
    )

    # Triton: read C_rot + states + cum_log_dA + y_intra, write y
    triton_bytes = (
        C_rot.nelement() * 4
        + all_states.nelement() * 4
        + cum_log_dA.nelement() * 4
        + y_intra.nelement() * 2
        + batch * nc * L * n_heads * head_dim * 2  # write y (bf16)
    )

    saved_mb = (pytorch_bytes - triton_bytes) / 1e6
    speedup = ms_pytorch / ms_triton

    print(f"  PyTorch:  {ms_pytorch:.3f} ms | "
          f"HBM traffic: {pytorch_bytes / 1e6:.1f} MB")
    print(f"  Triton:   {ms_triton:.3f} ms | "
          f"HBM traffic: {triton_bytes / 1e6:.1f} MB")
    print(f"  Speedup:  {speedup:.2f}x | "
          f"HBM saved: {saved_mb:.1f} MB ({saved_mb / (pytorch_bytes / 1e6) * 100:.0f}%)")

    return dict(ms_pytorch=ms_pytorch, ms_triton=ms_triton, speedup=speedup,
                saved_mb=saved_mb)


CONFIGS = {
    "user-config": dict(batch=8, nc=32, L=128, n_heads=18, head_dim=128,
                        d_state=128, ngroups=2),
    "ngroups=nheads": dict(batch=8, nc=32, L=128, n_heads=18, head_dim=128,
                           d_state=128, ngroups=18),
    "small": dict(batch=16, nc=4, L=128, n_heads=4, head_dim=64,
                  d_state=64, ngroups=1),
    "large-dstate64": dict(batch=8, nc=32, L=128, n_heads=40, head_dim=64,
                           d_state=64, ngroups=1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", action="store_true",
                        help="Also test backward gradient correctness")
    parser.add_argument("--profile", choices=list(CONFIGS.keys()),
                        help="Run a single config")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    configs = {args.profile: CONFIGS[args.profile]} if args.profile else CONFIGS

    for name, cfg in configs.items():
        print(f"\n[{name}] batch={cfg['batch']} nc={cfg['nc']} L={cfg['L']} "
              f"nh={cfg['n_heads']} hd={cfg['head_dim']} ds={cfg['d_state']} "
              f"ng={cfg['ngroups']}")
        print("-" * 60)

        ok = test_correctness(cfg, test_backward=args.backward)
        if not ok:
            print("  *** CORRECTNESS FAILURE — skipping benchmark ***")
            continue

        bench(cfg)
        print()


if __name__ == "__main__":
    main()
