"""
Gradient correctness test: Triton backward vs PyTorch backward for SSD.

Compares the new Triton-based backward (ssd_chunk_scan_combined_bwd) against
the reference PyTorch backward (_ssd_chunked_pytorch with autograd) to verify
gradient correctness.

Usage:
    uv run python -m scripts.test_ssd_backward
"""

import torch
import sys


def test_ssd_backward_correctness():
    from nanochat.mamba2 import _ssd_chunked_pytorch
    from nanochat.ssd_triton import ssd_chunk_scan_combined_fwd, ssd_chunk_scan_combined_bwd

    device = "cuda"
    dtype = torch.bfloat16

    # Small dimensions for testing
    batch = 2
    seqlen = 256
    nheads = 8
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 64
    scale = 1.0 / (headdim ** 0.5)

    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    A = -torch.rand(nheads, device=device, dtype=dtype) * 0.1  # small negative
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
    dt = torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.5 + 0.1
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=dtype))

    # ---- Reference: PyTorch backward via autograd ----
    x_ref = x.detach().clone().requires_grad_(True)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)
    C_ref = C.detach().clone().requires_grad_(True)
    dt_ref = dt.detach().clone().requires_grad_(True)

    y_ref = _ssd_chunked_pytorch(x_ref, A_ref, B_ref, C_ref, dt_ref, causal_mask,
                                  ngroups, scale, chunk_size)

    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)

    # ---- Triton backward ----
    # First verify forward matches
    y_triton = ssd_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, scale)
    fwd_diff = (y_triton.float() - y_ref.detach().float()).abs().max().item()
    fwd_rel = fwd_diff / (y_ref.detach().float().abs().max().item() + 1e-8)

    dx_tri, dA_tri, dB_tri, dC_tri, ddt_tri = ssd_chunk_scan_combined_bwd(
        dy, x, dt, A, B, C, chunk_size, scale
    )

    # ---- Compare gradients ----
    def compare(name, ref, tri, atol=2e-2, rtol=5e-2):
        ref_f = ref.float()
        tri_f = tri.float()
        abs_diff = (ref_f - tri_f).abs()
        max_diff = abs_diff.max().item()
        ref_max = ref_f.abs().max().item()
        rel_diff = max_diff / (ref_max + 1e-8)
        # Also compute per-element relative error for a cosine similarity check
        cos = torch.nn.functional.cosine_similarity(
            ref_f.reshape(-1).unsqueeze(0),
            tri_f.reshape(-1).unsqueeze(0)
        ).item()
        passed = (rel_diff < rtol or max_diff < atol) and cos > 0.95
        status = "PASS" if passed else "FAIL"
        print(f"  {name:8s}: max_abs={max_diff:.6f}, rel={rel_diff:.6f}, "
              f"cos_sim={cos:.6f} [{status}]")
        return passed

    print(f"\nForward match: max_abs_diff={fwd_diff:.6f}, rel={fwd_rel:.6f}")
    print(f"\nGradient comparison (Triton backward vs PyTorch autograd):")

    all_pass = True
    all_pass &= compare("dx", x_ref.grad, dx_tri)
    all_pass &= compare("dA", A_ref.grad, dA_tri, atol=5e-2, rtol=0.1)
    all_pass &= compare("dB", B_ref.grad, dB_tri)
    all_pass &= compare("dC", C_ref.grad, dC_tri)
    all_pass &= compare("ddt", dt_ref.grad, ddt_tri)

    if all_pass:
        print("\nAll gradient checks PASSED!")
    else:
        print("\nSome gradient checks FAILED!")
        sys.exit(1)


def test_ssd_backward_end_to_end():
    """Test that a full forward+backward through _SSDTritonFn works without errors."""
    from nanochat.mamba2 import _ssd_fwd_op

    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen, nheads, headdim = 2, 128, 4, 32
    ngroups, dstate, chunk_size = 2, 16, 64
    scale = 1.0 / (headdim ** 0.5)

    torch.manual_seed(123)
    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    A = (-torch.rand(nheads, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    B = (torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    dt = (torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.5 + 0.1).requires_grad_(True)
    y = _ssd_fwd_op(x, A, B, C, dt, chunk_size, scale)
    loss = y.sum()
    loss.backward()

    for name, param in [("x", x), ("A", A), ("B", B), ("C", C), ("dt", dt)]:
        assert param.grad is not None, f"No gradient for {name}"
        assert not param.grad.isnan().any(), f"NaN in gradient for {name}"
        assert not param.grad.isinf().any(), f"Inf in gradient for {name}"

    print("\nEnd-to-end _ssd_fwd_op forward+backward: PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this test"
    print("Testing SSD Triton backward correctness...")
    test_ssd_backward_correctness()
    test_ssd_backward_end_to_end()
    print("\nAll tests passed!")
