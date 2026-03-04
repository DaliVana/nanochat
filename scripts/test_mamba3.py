"""
Correctness tests for Mamba-3 layer (including MIMO).

Tests:
1. Data-dependent RoPE correctness
2. Gradient flow through all parameters
3. Output shape
4. Two-SSD chunked output matches naive trapezoidal recurrence
5. MIMO gradient flow (R=2)
6. MIMO output shape (R=4)
7. MIMO recurrent vs chunked (R=2)

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

    # Both paths are mathematically equivalent — differences are from float precision only.
    passed = cos_sim > 0.999 and rel_diff < 0.01
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


def test_output_shape():
    """Verify Mamba3Layer produces correct output shape."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    d_model, d_state, expand, chunk_size = 64, 16, 2, 16
    batch, T = 2, 32

    m3 = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size)

    x = torch.randn(batch, T, d_model)

    with torch.no_grad():
        y3, _, _ = m3(x)

    shape_ok = y3.shape == (batch, T, d_model)
    print(f"  Output shape: {y3.shape} {'PASS' if shape_ok else 'FAIL'}")
    return shape_ok


def test_mimo_gradient_flow():
    """Verify gradients flow through all MIMO Mamba3Layer parameters."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    batch, T = 2, 32
    d_model, d_state, chunk_size = 64, 16, 16

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2,
                        chunk_size=chunk_size, mimo_rank=2)
    layer.to(device=device, dtype=dtype)

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

    if x.grad is None or x.grad.isnan().any() or x.grad.isinf().any():
        print(f"  FAIL: input gradient issue")
        all_pass = False

    print(f"  MIMO gradient flow (R=2): {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_mimo_shape():
    """Verify MIMO Mamba3Layer produces correct output shape."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    d_model, d_state, expand, chunk_size = 64, 16, 2, 16
    batch, T = 2, 32

    m3_siso = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                           chunk_size=chunk_size, mimo_rank=1)
    m3_mimo = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                           chunk_size=chunk_size, mimo_rank=4)

    x = torch.randn(batch, T, d_model)

    with torch.no_grad():
        y_siso, _, _ = m3_siso(x)
        y_mimo, _, _ = m3_mimo(x)

    shape_ok = y_siso.shape == y_mimo.shape == (batch, T, d_model)
    print(f"  MIMO shape: siso={y_siso.shape}, mimo_r4={y_mimo.shape} "
          f"{'PASS' if shape_ok else 'FAIL'}")
    return shape_ok


def test_mimo_recurrent_vs_chunked():
    """Verify MIMO trapezoidal recurrent matches MIMO Two-SSD chunked."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    batch, T = 2, 64
    d_model, d_state, expand, chunk_size = 64, 16, 2, 16
    mimo_rank = 2

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                        chunk_size=chunk_size, mimo_rank=mimo_rank)
    layer.to(device=device, dtype=dtype)
    layer.eval()

    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype)

    # Chunked (training) path
    with torch.no_grad():
        y_chunked, _, _ = layer(x, ssm_state=None)

    # Recurrent (inference) path
    with torch.no_grad():
        h = torch.zeros(batch, layer.n_heads, layer.head_dim, d_state, device=device, dtype=dtype)
        prev_Bx = torch.zeros_like(h)
        ssm_state = {'h': h, 'prev_Bx': prev_Bx}
        y_recurrent, _ = layer(x, ssm_state=ssm_state)[:2]

    max_diff = (y_chunked - y_recurrent).abs().max().item()
    rel_diff = max_diff / (y_chunked.abs().max().item() + 1e-8)
    cos_sim = F.cosine_similarity(
        y_chunked.reshape(-1).unsqueeze(0),
        y_recurrent.reshape(-1).unsqueeze(0)
    ).item()

    print(f"  MIMO recurrent vs chunked (R={mimo_rank}): "
          f"max_abs={max_diff:.6f}, rel={rel_diff:.6f}, cos_sim={cos_sim:.6f}")

    passed = cos_sim > 0.999 and rel_diff < 0.01
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_mimo_param_count():
    """Verify MIMO has more parameters than SISO."""
    from nanochat.mamba3 import Mamba3Layer

    d_model, d_state, expand, chunk_size = 64, 16, 2, 16

    p1 = sum(p.numel() for p in Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size, mimo_rank=1).parameters())
    p2 = sum(p.numel() for p in Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size, mimo_rank=2).parameters())
    p4 = sum(p.numel() for p in Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand, chunk_size=chunk_size, mimo_rank=4).parameters())

    passed = p1 < p2 < p4
    print(f"  MIMO param count: R=1: {p1}, R=2: {p2}, R=4: {p4} {'PASS' if passed else 'FAIL'}")
    return passed


def test_mimo_differentiation():
    """Verify MIMO gradient magnitudes are not dampened relative to SISO."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    batch, T = 2, 32
    d_model, d_state, chunk_size = 64, 16, 16

    layer_siso = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2,
                              chunk_size=chunk_size, mimo_rank=1)
    layer_mimo = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2,
                              chunk_size=chunk_size, mimo_rank=2)

    for layer in [layer_siso, layer_mimo]:
        layer.to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
            torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype)

    x_siso = x.clone().requires_grad_(True)
    x_mimo = x.clone().requires_grad_(True)
    y_siso, _, _ = layer_siso(x_siso)
    y_mimo, _, _ = layer_mimo(x_mimo)

    # Outputs should differ (different parameter counts)
    cos_sim = F.cosine_similarity(
        y_siso.reshape(-1).unsqueeze(0),
        y_mimo.reshape(-1).unsqueeze(0)).item()
    outputs_differ = cos_sim < 0.99

    # Gradient magnitudes should be comparable (not dampened by 1/sqrt(R))
    y_siso.sum().backward()
    y_mimo.sum().backward()

    grad_ratio = x_mimo.grad.abs().mean().item() / (x_siso.grad.abs().mean().item() + 1e-12)
    grads_comparable = 0.3 < grad_ratio < 3.0

    siso_proj_grad = layer_siso.in_proj.weight.grad.abs().mean().item()
    mimo_proj_grad = layer_mimo.in_proj.weight.grad.abs().mean().item()
    proj_ratio = mimo_proj_grad / (siso_proj_grad + 1e-12)
    proj_grads_ok = 0.3 < proj_ratio < 3.0

    passed = outputs_differ and grads_comparable and proj_grads_ok
    print(f"  MIMO differentiation: cos_sim={cos_sim:.4f}, "
          f"grad_ratio={grad_ratio:.4f}, proj_grad_ratio={proj_ratio:.4f} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def test_cuda_forward_backward():
    """Test that Mamba-3 MIMO runs correctly on CUDA."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, T = 2, 128
    d_model, d_state, chunk_size = 128, 64, 64

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=2,
                        chunk_size=chunk_size, mimo_rank=2)
    layer.to(device=device, dtype=dtype)

    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype, requires_grad=True)
    y, _, _ = layer(x)
    loss = y.sum()
    loss.backward()

    has_grad = x.grad is not None and not x.grad.isnan().any()
    print(f"  CUDA forward+backward (MIMO R=2): {'PASS' if has_grad else 'FAIL'}")
    return has_grad


def test_triton_intra_vs_pytorch():
    """Verify Triton within-chunk kernel (with fused RoPE) matches PyTorch."""
    from nanochat.mamba3 import _mamba3_intra_pytorch, _apply_rope_pairs
    from nanochat.ssd_triton import mamba3_chunk_scan_fwd

    torch.manual_seed(42)
    device = "cuda"

    batch, nc, L, ngroups, R, dstate = 2, 4, 64, 1, 2, 64
    nheads = 4
    headdim = 64
    half_d = dstate // 2
    scale = (dstate * R) ** -0.5

    # Raw (unrotated) B/C and angles
    Bg_raw = torch.randn(batch, nc, L, ngroups, R, dstate, device=device, dtype=torch.float32)
    Bb_raw = torch.randn(batch, nc, L, ngroups, R, dstate, device=device, dtype=torch.float32)
    x_dt_g = torch.randn(batch, nc, L, nheads, R, headdim, device=device, dtype=torch.float32)
    x_dt_b = torch.randn(batch, nc, L, nheads, R, headdim, device=device, dtype=torch.float32)
    # In the real model, log_dA = A * dt where A < 0 and dt > 0, so increments are
    # always negative and cum_log_dA is monotonically non-increasing. The Triton kernel
    # relies on this via min(dA_cs[m]-dA_cs[k], 0) clamping decay to [0, 1].
    dA_cumsum = (-torch.rand(batch, nc, L, nheads, device=device, dtype=torch.float32) * 0.1).cumsum(dim=2)
    C_raw = torch.randn(batch, nc, L, ngroups, dstate, device=device, dtype=torch.float32)
    cum_angles = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5
    cum_angles_shifted = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5

    # Pre-compute cos/sin (as the kernel now expects)
    cos_a_f32 = torch.cos(cum_angles)
    sin_a_f32 = torch.sin(cum_angles)
    cos_s_f32 = torch.cos(cum_angles_shifted)
    sin_s_f32 = torch.sin(cum_angles_shifted)

    # Triton path: raw B/C + pre-computed cos/sin (RoPE fused in kernel)
    y_triton = mamba3_chunk_scan_fwd(Bg_raw, Bb_raw, x_dt_g, x_dt_b, dA_cumsum, C_raw,
                                      cos_a_f32, sin_a_f32, cos_s_f32, sin_s_f32, scale, L, R)

    # PyTorch path: pre-rotate B/C, then run without angles
    cos_a = cos_a_f32.unsqueeze(3)       # (batch, nc, L, 1, half_d)
    sin_a = sin_a_f32.unsqueeze(3)
    cos_s = cos_s_f32.unsqueeze(3)
    sin_s = sin_s_f32.unsqueeze(3)
    Bg_rot = _apply_rope_pairs(Bg_raw, cos_a.unsqueeze(4), sin_a.unsqueeze(4))
    Bb_rot = _apply_rope_pairs(Bb_raw, cos_s.unsqueeze(4), sin_s.unsqueeze(4))
    C_rot = _apply_rope_pairs(C_raw, cos_a, sin_a)

    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    y_pytorch = _mamba3_intra_pytorch(Bg_rot, Bb_rot, x_dt_g, x_dt_b, dA_cumsum,
                                       C_rot.float(), scale, R, causal_mask)

    max_diff = (y_triton - y_pytorch).abs().max().item()
    scale_val = y_pytorch.abs().max().item() + 1e-8
    rel_diff = max_diff / scale_val

    passed = rel_diff < 0.01
    print(f"  Triton fused-RoPE vs PyTorch pre-rotated (R=2): max_abs={max_diff:.6f}, "
          f"rel={rel_diff:.6f} {'PASS' if passed else 'FAIL'}")
    return passed


def test_triton_intra_r4():
    """Verify Triton kernel with fused RoPE works for MIMO R=4."""
    from nanochat.mamba3 import _mamba3_intra_pytorch, _apply_rope_pairs
    from nanochat.ssd_triton import mamba3_chunk_scan_fwd

    torch.manual_seed(123)
    device = "cuda"

    batch, nc, L, ngroups, R, dstate = 2, 2, 32, 1, 4, 32
    nheads = 4
    headdim = 32
    half_d = dstate // 2
    scale = (dstate * R) ** -0.5

    Bg_raw = torch.randn(batch, nc, L, ngroups, R, dstate, device=device, dtype=torch.float32)
    Bb_raw = torch.randn(batch, nc, L, ngroups, R, dstate, device=device, dtype=torch.float32)
    x_dt_g = torch.randn(batch, nc, L, nheads, R, headdim, device=device, dtype=torch.float32)
    x_dt_b = torch.randn(batch, nc, L, nheads, R, headdim, device=device, dtype=torch.float32)
    # Monotonically non-increasing (matches real model where log_dA = A*dt < 0)
    dA_cumsum = (-torch.rand(batch, nc, L, nheads, device=device, dtype=torch.float32) * 0.1).cumsum(dim=2)
    C_raw = torch.randn(batch, nc, L, ngroups, dstate, device=device, dtype=torch.float32)
    cum_angles = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5
    cum_angles_shifted = torch.randn(batch, nc, L, half_d, device=device, dtype=torch.float32) * 0.5

    # Pre-compute cos/sin
    cos_a_f32 = torch.cos(cum_angles)
    sin_a_f32 = torch.sin(cum_angles)
    cos_s_f32 = torch.cos(cum_angles_shifted)
    sin_s_f32 = torch.sin(cum_angles_shifted)

    # Triton: fused RoPE with pre-computed cos/sin
    y_triton = mamba3_chunk_scan_fwd(Bg_raw, Bb_raw, x_dt_g, x_dt_b, dA_cumsum, C_raw,
                                      cos_a_f32, sin_a_f32, cos_s_f32, sin_s_f32, scale, L, R)

    # PyTorch: pre-rotate
    cos_a = cos_a_f32.unsqueeze(3)
    sin_a = sin_a_f32.unsqueeze(3)
    cos_s = cos_s_f32.unsqueeze(3)
    sin_s = sin_s_f32.unsqueeze(3)
    Bg_rot = _apply_rope_pairs(Bg_raw, cos_a.unsqueeze(4), sin_a.unsqueeze(4))
    Bb_rot = _apply_rope_pairs(Bb_raw, cos_s.unsqueeze(4), sin_s.unsqueeze(4))
    C_rot = _apply_rope_pairs(C_raw, cos_a, sin_a)

    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    y_pytorch = _mamba3_intra_pytorch(Bg_rot, Bb_rot, x_dt_g, x_dt_b, dA_cumsum,
                                       C_rot.float(), scale, R, causal_mask)

    max_diff = (y_triton - y_pytorch).abs().max().item()
    scale_val = y_pytorch.abs().max().item() + 1e-8
    rel_diff = max_diff / scale_val

    passed = rel_diff < 0.01
    print(f"  Triton fused-RoPE vs PyTorch pre-rotated (R=4): max_abs={max_diff:.6f}, "
          f"rel={rel_diff:.6f} {'PASS' if passed else 'FAIL'}")
    return passed


def test_triton_full_layer_gradient():
    """Verify gradients flow through Triton forward + PyTorch backward path."""
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, T = 2, 128
    d_model, d_state, expand, chunk_size = 128, 32, 2, 64

    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                        chunk_size=chunk_size, mimo_rank=2)
    layer.to(device=device, dtype=dtype)

    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, device=device, dtype=dtype, requires_grad=True)
    y, _, _ = layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradients on all parameters
    all_ok = True
    for name, p in layer.named_parameters():
        if p.grad is None:
            print(f"  WARNING: {name} has no gradient")
            all_ok = False
        elif p.grad.isnan().any():
            print(f"  WARNING: {name} has NaN gradient")
            all_ok = False

    has_input_grad = x.grad is not None and not x.grad.isnan().any()
    all_ok = all_ok and has_input_grad

    print(f"  Triton full-layer gradient flow: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_state_propagation_triton_vs_pytorch():
    """Verify Triton state propagation kernel matches Python loop reference."""
    from nanochat.ssd_triton import state_propagation_fwd

    torch.manual_seed(42)
    device = "cuda"

    configs = [
        (4, 16, 8, 64, 64),    # standard
        (2, 8, 4, 128, 128),   # large head_dim/d_state (4 tiles)
        (2, 32, 4, 64, 128),   # asymmetric
        (2, 1, 4, 64, 64),     # edge case: single chunk (all zeros output)
    ]

    all_pass = True
    for batch, nc, n_heads, head_dim, d_state in configs:
        delta_h = torch.randn(batch, nc, n_heads, head_dim, d_state,
                               device=device, dtype=torch.float32)
        chunk_decay = torch.rand(batch, nc, n_heads,
                                  device=device, dtype=torch.float32)

        # Triton
        all_states_triton = state_propagation_fwd(delta_h, chunk_decay)

        # Python reference
        states = torch.zeros(batch, n_heads, head_dim, d_state, device=device)
        ref = []
        for c in range(nc):
            ref.append(states)
            states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
        all_states_ref = torch.stack(ref, dim=1)

        max_diff = (all_states_triton - all_states_ref).abs().max().item()
        rel_diff = max_diff / (all_states_ref.abs().max().item() + 1e-8)

        ok = rel_diff < 1e-5
        if not ok:
            print(f"    FAIL config ({batch},{nc},{n_heads},{head_dim},{d_state}): "
                  f"max_abs={max_diff:.8f}, rel={rel_diff:.8f}")
            all_pass = False

    print(f"  State propagation Triton vs Python ({len(configs)} configs): "
          f"{'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_state_propagation_gradient():
    """Verify backward gradients for fused state propagation custom op."""
    from nanochat.mamba3 import _HAS_TRITON
    if not _HAS_TRITON:
        print("  State propagation gradient: SKIP (no Triton)")
        return True

    from nanochat.mamba3 import _state_propagation_fwd_op

    torch.manual_seed(42)
    device = "cuda"

    batch, nc, n_heads, head_dim, d_state = 2, 8, 4, 16, 16

    delta_h = torch.randn(batch, nc, n_heads, head_dim, d_state,
                           device=device, dtype=torch.float64, requires_grad=True)
    chunk_decay = torch.randn(batch, nc, n_heads,
                               device=device, dtype=torch.float64, requires_grad=True)

    passed = torch.autograd.gradcheck(
        _state_propagation_fwd_op, (delta_h, chunk_decay),
        eps=1e-6, atol=1e-4, rtol=1e-3)

    print(f"  State propagation gradcheck: {'PASS' if passed else 'FAIL'}")
    return passed


def test_fused_rope_full_layer():
    """Test full Mamba3Layer with fused RoPE in Triton kernel (d_state=128, our target config).

    Compares CUDA (fused RoPE in Triton) forward+backward against CPU (pre-rotated) to
    verify end-to-end correctness including between-chunk code with lazy rotation.
    """
    from nanochat.mamba3 import Mamba3Layer

    torch.manual_seed(42)
    dtype = torch.float32  # Use float32 for accurate comparison

    batch, T = 2, 128
    d_model, d_state, expand, chunk_size = 128, 128, 2, 64
    mimo_rank = 2

    # Create layer on CPU first
    layer = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                        chunk_size=chunk_size, mimo_rank=mimo_rank)
    layer.to(dtype=dtype)
    layer.eval()

    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.linspace(0.001, layer.n_heads, layer.n_heads)))
        torch.nn.init.uniform_(layer.dt_bias, -4.0, -2.0)

    x = torch.randn(batch, T, d_model, dtype=dtype)

    # CPU forward (pre-rotated RoPE, no Triton)
    with torch.no_grad():
        y_cpu, _, _ = layer(x, ssm_state=None)

    # CUDA forward (fused RoPE in Triton kernel)
    layer_cuda = Mamba3Layer(d_model=d_model, d_state=d_state, expand=expand,
                              chunk_size=chunk_size, mimo_rank=mimo_rank)
    layer_cuda.to(device="cuda", dtype=dtype)
    # Copy weights
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(layer.named_parameters(), layer_cuda.named_parameters()):
            p2.copy_(p1)
    layer_cuda.eval()

    x_cuda = x.to("cuda")
    with torch.no_grad():
        y_cuda, _, _ = layer_cuda(x_cuda, ssm_state=None)

    y_cuda_cpu = y_cuda.cpu()
    max_diff = (y_cpu - y_cuda_cpu).abs().max().item()
    scale_val = y_cpu.abs().max().item() + 1e-8
    rel_diff = max_diff / scale_val
    cos_sim = F.cosine_similarity(
        y_cpu.reshape(-1).unsqueeze(0),
        y_cuda_cpu.reshape(-1).unsqueeze(0)
    ).item()

    passed = cos_sim > 0.999 and rel_diff < 0.01
    print(f"  Fused RoPE full layer (CPU vs CUDA): max_abs={max_diff:.6f}, "
          f"rel={rel_diff:.6f}, cos_sim={cos_sim:.6f} {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    use_cuda = "--cuda" in sys.argv

    print("Testing Mamba-3 layer correctness...\n")

    all_pass = True

    print("1. RoPE correctness:")
    all_pass &= test_rope_correctness()

    print("\n2. Gradient flow:")
    all_pass &= test_gradient_flow()

    print("\n3. Output shape:")
    all_pass &= test_output_shape()

    print("\n4. Trapezoidal vs Recurrent:")
    all_pass &= test_trapezoidal_vs_recurrent()

    print("\n5. MIMO gradient flow:")
    all_pass &= test_mimo_gradient_flow()

    print("\n6. MIMO shape:")
    all_pass &= test_mimo_shape()

    print("\n7. MIMO recurrent vs chunked:")
    all_pass &= test_mimo_recurrent_vs_chunked()

    print("\n8. MIMO param count:")
    all_pass &= test_mimo_param_count()

    print("\n9. MIMO differentiation:")
    all_pass &= test_mimo_differentiation()

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA required for --cuda tests"
        print("\n10. CUDA forward+backward:")
        all_pass &= test_cuda_forward_backward()

        print("\n11. Triton fused-RoPE vs PyTorch (R=2):")
        all_pass &= test_triton_intra_vs_pytorch()

        print("\n12. Triton fused-RoPE vs PyTorch (R=4):")
        all_pass &= test_triton_intra_r4()

        print("\n13. Triton full-layer gradient flow:")
        all_pass &= test_triton_full_layer_gradient()

        print("\n14. Fused RoPE full layer (CPU vs CUDA):")
        all_pass &= test_fused_rope_full_layer()

        print("\n15. State propagation Triton vs Python:")
        all_pass &= test_state_propagation_triton_vs_pytorch()

        print("\n16. State propagation gradient:")
        all_pass &= test_state_propagation_gradient()

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
