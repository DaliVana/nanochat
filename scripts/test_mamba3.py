"""
Correctness tests for Mamba-3 layer.

Tests:
1. Two-SSD chunked output matches naive trapezoidal recurrence
2. Data-dependent RoPE correctness
3. Gradient flow through all parameters
4. Output shape compatibility with Mamba-2
5. Config dispatch (mamba_version=3 creates Mamba3Layer)

Usage:
    uv run python -m scripts.test_mamba3           # CPU tests
    uv run python -m scripts.test_mamba3 --cuda     # include CUDA/Triton tests
"""

import sys
import torch
import torch.nn.functional as F


def test_trapezoidal_vs_recurrent():
    """Verify Two-SSD chunked output matches step-by-step trapezoidal recurrence."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32  # full precision for comparison

    batch, T = 2, 64
    d_model, d_state, expand, chunk_size = 64, 16, 2, 16

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                        chunk_size=chunk_size)
    layer.to(device=device, dtype=dtype)
    layer.eval()

    # Initialize with reasonable values
    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype)

    # --- Chunked (training) path ---
    with torch.no_grad():
        y_chunked, _, _ = layer(x, ssm_state=None)

    # --- Recurrent (inference) path ---
    with torch.no_grad():
        h = torch.zeros(batch, layer.n_heads, layer.head_dim, d_state, device=device, dtype=dtype)
        prev_Bx = torch.zeros_like(h)
        ssm_state = {'h': h, 'prev_Bx': prev_Bx}
        y_recurrent, _ = layer(x, ssm_state=ssm_state)[:2]

    # Compare
    max_diff = (y_chunked - y_recurrent).abs().max().item()
    rel_diff = max_diff / (y_chunked.abs().max().item() + 1e-8)
    cos_sim = F.cosine_similarity(
        y_chunked.reshape(-1).unsqueeze(0),
        y_recurrent.reshape(-1).unsqueeze(0)
    ).item()

    print(f"  Trapezoidal vs Recurrent: max_abs={max_diff:.6f}, rel={rel_diff:.6f}, cos_sim={cos_sim:.6f}")

    # The two paths won't be exactly identical because the chunked path uses SSD
    # (which includes within-chunk quadratic approximation) while recurrent is exact.
    # But they should be very close for small dt values.
    passed = cos_sim > 0.95 and rel_diff < 0.1
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_rope_correctness():
    """Verify data-dependent RoPE produces correct rotations."""
    from nanochat.mamba3 import _apply_rope_pairs

    torch.manual_seed(123)
    batch, T, ngroups, d_state = 2, 8, 1, 16
    half_d = d_state // 2

    x = torch.randn(batch, T, ngroups, d_state)
    angles = torch.randn(batch, T, 1, half_d)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Apply RoPE
    y = _apply_rope_pairs(x, cos_a, sin_a)

    # Manual verification: check first pair
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    y1_expected = x1 * cos_a - x2 * sin_a
    y2_expected = x1 * sin_a + x2 * cos_a
    y_expected = torch.cat([y1_expected, y2_expected], dim=-1)

    diff = (y - y_expected).abs().max().item()
    passed = diff < 1e-6
    print(f"  RoPE correctness: max_diff={diff:.8f} {'PASS' if passed else 'FAIL'}")

    # Verify rotation is norm-preserving
    norm_in = x.norm(dim=-1)
    norm_out = y.norm(dim=-1)
    norm_diff = (norm_in - norm_out).abs().max().item()
    norm_ok = norm_diff < 1e-5
    print(f"  RoPE norm-preserving: max_diff={norm_diff:.8f} {'PASS' if norm_ok else 'FAIL'}")

    return passed and norm_ok


def test_gradient_flow():
    """Verify gradients flow through all Mamba3Layer parameters without NaN/Inf."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    batch, T = 2, 32
    d_model, d_state, chunk_size = 64, 16, 16

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2, chunk_size=chunk_size)
    layer.to(device=device, dtype=dtype)

    # Initialize
    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype, requires_grad=True)
    y, _, _ = layer(x)
    loss = y.sum()
    loss.backward()

    all_pass = True
    for name, p in layer.named_parameters():
        has_grad = p.grad is not None
        no_nan = has_grad and not p.grad.isnan().any()
        no_inf = has_grad and not p.grad.isinf().any()
        ok = has_grad and no_nan and no_inf
        if not ok:
            print(f"  FAIL: {name} grad={'None' if not has_grad else 'has NaN/Inf'}")
            all_pass = False

    # Check input gradient too
    if x.grad is None or x.grad.isnan().any() or x.grad.isinf().any():
        print(f"  FAIL: input gradient issue")
        all_pass = False

    print(f"  Gradient flow: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_shape_compatibility():
    """Verify Mamba3Layer produces same output shape as Mamba2Layer."""
    from nanochat.mamba2 import Mamba2Layer
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    d_model, d_state, expand, chunk_size = 64, 16, 2, 16
    batch, T = 2, 32

    m2 = Mamba2Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size)
    m3 = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size)

    x = torch.randn(batch, T, d_model)

    with torch.no_grad():
        y2, _, _ = m2(x)
        y3, _, _ = m3(x)

    shape_ok = y2.shape == y3.shape == (batch, T, d_model)
    print(f"  Shape compatibility: m2={y2.shape}, m3={y3.shape} {'PASS' if shape_ok else 'FAIL'}")
    return shape_ok


def test_config_dispatch():
    """Verify GPTConfigSamba with mamba_version=3 creates Mamba3Layer blocks."""
    from nanochat.mamba3 import Mamba3Layer
    from nanochat.mamba2 import Mamba2Layer
    try:
        from nanochat.gpt_samba import GPTConfigSamba, MambaBlock
    except ImportError as e:
        print(f"  Config dispatch: SKIP ({e})")
        return True

    config_v2 = GPTConfigSamba(n_embd=64, n_layer=2, n_head=4, n_kv_head=4,
                                mamba_d_state=16, mamba_expand=2, mamba_version=2)
    config_v3 = GPTConfigSamba(n_embd=64, n_layer=2, n_head=4, n_kv_head=4,
                                mamba_d_state=16, mamba_expand=2, mamba_version=3)

    block_v2 = MambaBlock(config_v2)
    block_v3 = MambaBlock(config_v3)

    is_v2 = isinstance(block_v2.mamba, Mamba2Layer)
    is_v3 = isinstance(block_v3.mamba, Mamba3Layer)

    passed = is_v2 and is_v3
    print(f"  Config dispatch: v2={type(block_v2.mamba).__name__}, v3={type(block_v3.mamba).__name__} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def test_triton_dispatch():
    """Test that Mamba-3 correctly dispatches to Triton SSD on CUDA."""
    from nanochat.mamba3 import Mamba3Layer, _HAS_TRITON

    if not _HAS_TRITON:
        print("  Triton dispatch: SKIP (Triton not available)")
        return True

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, T = 2, 128
    d_model, d_state, chunk_size = 128, 64, 64

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2, chunk_size=chunk_size)
    layer.to(device=device, dtype=dtype)

    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype, requires_grad=True)
    y, _, _ = layer(x)
    loss = y.sum()
    loss.backward()

    has_grad = x.grad is not None and not x.grad.isnan().any()
    print(f"  Triton dispatch (CUDA): {'PASS' if has_grad else 'FAIL'}")
    return has_grad


if __name__ == "__main__":
    use_cuda = "--cuda" in sys.argv

    print("Testing Mamba-3 layer correctness...\n")

    all_pass = True

    print("1. RoPE correctness:")
    all_pass &= test_rope_correctness()

    print("\n2. Gradient flow:")
    all_pass &= test_gradient_flow()

    print("\n3. Shape compatibility:")
    all_pass &= test_shape_compatibility()

    print("\n4. Config dispatch:")
    all_pass &= test_config_dispatch()

    print("\n5. Trapezoidal vs Recurrent:")
    all_pass &= test_trapezoidal_vs_recurrent()

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA required for --cuda tests"
        print("\n6. Triton dispatch (CUDA):")
        all_pass &= test_triton_dispatch()

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
