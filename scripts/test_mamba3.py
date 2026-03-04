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

    # Triton path: raw B/C + raw angles (cos/sin computed in-kernel)
    y_triton, _, _, _ = mamba3_chunk_scan_fwd(Bg_raw, Bb_raw, x_dt_g, x_dt_b, dA_cumsum, C_raw,
                                               cum_angles, cum_angles_shifted, scale, L, R)

    # PyTorch path: pre-rotate B/C, then run without angles
    cos_a = torch.cos(cum_angles).unsqueeze(3)       # (batch, nc, L, 1, half_d)
    sin_a = torch.sin(cum_angles).unsqueeze(3)
    cos_s = torch.cos(cum_angles_shifted).unsqueeze(3)
    sin_s = torch.sin(cum_angles_shifted).unsqueeze(3)
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

    # Triton: raw angles (cos/sin computed in-kernel)
    y_triton, _, _, _ = mamba3_chunk_scan_fwd(Bg_raw, Bb_raw, x_dt_g, x_dt_b, dA_cumsum, C_raw,
                                               cum_angles, cum_angles_shifted, scale, L, R)

    # PyTorch: pre-rotate
    cos_a = torch.cos(cum_angles).unsqueeze(3)
    sin_a = torch.sin(cum_angles).unsqueeze(3)
    cos_s = torch.cos(cum_angles_shifted).unsqueeze(3)
    sin_s = torch.sin(cum_angles_shifted).unsqueeze(3)
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


def test_delta_h_triton_vs_pytorch():
    """Verify Triton delta_h kernel matches PyTorch einsum reference."""
    from nanochat.ssd_triton import delta_h_fwd

    torch.manual_seed(42)
    device = "cuda"

    configs = [
        # (B, nc, L, n_heads, ngroups, R, head_dim, d_state)
        (2, 4, 128, 4, 4, 2, 64, 64),     # ngroups == n_heads, R=2
        (2, 4, 128, 8, 2, 2, 64, 64),     # ngroups < n_heads (hpg=4)
        (2, 4, 64,  4, 1, 1, 64, 64),     # ngroups=1, R=1 (SISO)
        (2, 4, 256, 4, 4, 4, 64, 128),    # large L, R=4, d_state=128
        (1, 2, 128, 4, 2, 2, 128, 64),    # large head_dim
    ]

    all_pass = True
    for B, nc, L, nh, ng, R, hd, ds in configs:
        nheads_per_group = nh // ng
        x_dt_g = torch.randn(B, nc, L, nh, R, hd, device=device, dtype=torch.bfloat16)
        x_dt_b = torch.randn(B, nc, L, nh, R, hd, device=device, dtype=torch.bfloat16)
        Bg_rot = torch.randn(B, nc, L, ng, R, ds, device=device, dtype=torch.bfloat16)
        Bb_rot = torch.randn(B, nc, L, ng, R, ds, device=device, dtype=torch.bfloat16)
        cum_log_dA = (-torch.rand(B, nc, L, nh, device=device) * 0.1).cumsum(dim=2)

        # Triton
        delta_h_triton = delta_h_fwd(x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA)

        # Python reference
        decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)
        delta_h_ref = torch.zeros(B, nc, nh, hd, ds, device=device, dtype=torch.float32)
        if ng < nh:
            decay_g = decay_to_end.view(B, nc, L, ng, nheads_per_group)
            for r in range(R):
                xg_r = x_dt_g[:, :, :, :, r, :].float().view(B, nc, L, ng, nheads_per_group, hd)
                xb_r = x_dt_b[:, :, :, :, r, :].float().view(B, nc, L, ng, nheads_per_group, hd)
                delta_h_ref += torch.einsum('bclgk, bclgkp, bclgn -> bcgkpn',
                    decay_g, xg_r, Bg_rot[:, :, :, :, r, :].float()
                ).reshape(B, nc, nh, hd, ds)
                delta_h_ref += torch.einsum('bclgk, bclgkp, bclgn -> bcgkpn',
                    decay_g, xb_r, Bb_rot[:, :, :, :, r, :].float()
                ).reshape(B, nc, nh, hd, ds)
        else:
            for r in range(R):
                delta_h_ref += torch.einsum('bclh, bclhp, bclhn -> bchpn',
                    decay_to_end, x_dt_g[:, :, :, :, r, :].float(),
                    Bg_rot[:, :, :, :, r, :].float())
                delta_h_ref += torch.einsum('bclh, bclhp, bclhn -> bchpn',
                    decay_to_end, x_dt_b[:, :, :, :, r, :].float(),
                    Bb_rot[:, :, :, :, r, :].float())

        max_diff = (delta_h_triton - delta_h_ref).abs().max().item()
        rel_diff = max_diff / (delta_h_ref.abs().max().item() + 1e-8)
        ok = rel_diff < 0.02  # bf16 dots have limited precision
        if not ok:
            print(f"    FAIL ({B},{nc},{L},{nh},{ng},{R},{hd},{ds}): "
                  f"max_abs={max_diff:.6f}, rel={rel_diff:.6f}")
            all_pass = False

    print(f"  delta_h Triton vs Python ({len(configs)} configs): "
          f"{'PASS' if all_pass else 'FAIL'}")
    return all_pass


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


def test_ssd_prep_triton_vs_pytorch():
    """Verify fused SSD prep kernel matches PyTorch reference."""
    from nanochat.ssd_triton import ssd_prep_fwd

    torch.manual_seed(42)
    device = "cuda"

    batch, T, nheads, R, headdim = 2, 128, 4, 2, 64
    ngroups, dstate = 2, 64
    chunk_size = 64

    # Raw inputs (already padded — T is multiple of chunk_size)
    x_heads = torch.randn(batch, T, nheads, R, headdim, device=device)
    dt = torch.rand(batch, T, nheads, device=device) * 0.1 + 0.01
    lambda_raw = torch.randn(batch, T, nheads, device=device)
    A = -torch.rand(nheads, device=device) * 0.5 - 0.01  # negative
    B_raw = torch.randn(batch, T, ngroups, R, dstate, device=device)

    # Triton prep
    x_dt_g_t, x_dt_b_t, Bb_t, cum_log_dA_t, chunk_decay_t = \
        ssd_prep_fwd(x_heads, dt, lambda_raw, A, B_raw, chunk_size)

    # PyTorch reference
    nc = T // chunk_size
    L = chunk_size
    lam = torch.sigmoid(lambda_raw)
    alpha = torch.exp(A * dt)
    coeff_beta = (1 - lam) * alpha
    x_gamma = x_heads * lam.unsqueeze(-1).unsqueeze(-1)
    x_beta = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0)) * coeff_beta.unsqueeze(-1).unsqueeze(-1)
    B_shifted = F.pad(B_raw[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))

    x_g = x_gamma.view(batch, nc, L, nheads, R, headdim)
    x_b = x_beta.view(batch, nc, L, nheads, R, headdim)
    dt_ = dt.view(batch, nc, L, nheads)
    dt_broad = dt_.unsqueeze(-1).unsqueeze(-1)
    x_dt_g_ref = (x_g * dt_broad).bfloat16()
    x_dt_b_ref = (x_b * dt_broad).bfloat16()
    log_dA = (A * dt_).float()
    cum_log_dA_ref = torch.cumsum(log_dA, dim=2)
    chunk_decay_ref = torch.exp(cum_log_dA_ref[:, :, -1, :])
    Bb_ref = B_shifted.view(batch, nc, L, ngroups, R, dstate)

    all_ok = True
    for name, tri, ref in [
        ("x_dt_g", x_dt_g_t.float(), x_dt_g_ref.float()),
        ("x_dt_b", x_dt_b_t.float(), x_dt_b_ref.float()),
        ("Bb", Bb_t.float(), Bb_ref.float()),
        ("cum_log_dA", cum_log_dA_t, cum_log_dA_ref),
        ("chunk_decay", chunk_decay_t, chunk_decay_ref),
    ]:
        max_diff = (tri - ref).abs().max().item()
        scale_val = ref.abs().max().item() + 1e-8
        rel = max_diff / scale_val
        ok = rel < 0.01
        all_ok &= ok
        print(f"    {name}: max_abs={max_diff:.6f}, rel={rel:.6f} {'PASS' if ok else 'FAIL'}")

    return all_ok


def test_compute_rope_angles_triton_vs_pytorch():
    """Verify fused RoPE angle kernel matches PyTorch reference."""
    from nanochat.ssd_triton import compute_rope_angles_fwd

    torch.manual_seed(42)
    device = "cuda"

    # Test with padding needed (T=100, chunk_size=64 → T_padded=128)
    batch, T, nheads, dstate = 2, 100, 4, 64
    chunk_size = 64
    half_d = dstate // 2
    pad_len = chunk_size - (T % chunk_size)  # 28
    T_padded = T + pad_len  # 128
    nc = T_padded // chunk_size

    dt = (torch.rand(batch, T, nheads, device=device) * 0.1 + 0.01).float()
    theta_raw = torch.randn(batch, T, half_d, device=device).float()

    # Pad inputs
    dt_padded = F.pad(dt, (0, 0, 0, pad_len))
    theta_padded = F.pad(theta_raw, (0, 0, 0, pad_len))

    # Triton kernel
    angles_chunked_t, angles_shifted_t = compute_rope_angles_fwd(
        dt_padded, theta_padded, T, chunk_size)

    # PyTorch reference
    dt_mean = dt.mean(dim=-1, keepdim=True)  # (batch, T, 1)
    raw_angles = dt_mean * theta_raw
    cum_angles = -torch.cumsum(raw_angles.float(), dim=1)
    cum_angles_padded = F.pad(cum_angles, (0, 0, 0, pad_len))
    angles_chunked_ref = cum_angles_padded.view(batch, nc, chunk_size, half_d)
    cum_angles_shifted = F.pad(cum_angles_padded[:, :-1], (0, 0, 1, 0), value=0.0)
    angles_shifted_ref = cum_angles_shifted.view(batch, nc, chunk_size, half_d)

    all_ok = True
    for name, tri, ref in [
        ("angles_chunked", angles_chunked_t, angles_chunked_ref),
        ("angles_shifted", angles_shifted_t, angles_shifted_ref),
    ]:
        max_diff = (tri - ref).abs().max().item()
        scale_val = ref.abs().max().item() + 1e-8
        rel = max_diff / scale_val
        ok = rel < 0.01
        all_ok &= ok
        print(f"    {name}: max_abs={max_diff:.6f}, rel={rel:.6f} {'PASS' if ok else 'FAIL'}")

    return all_ok


def test_trapezoidal_recurrent_triton_vs_pytorch():
    """Verify fused Triton inference recurrence matches Python RoPE + recurrence."""
    from nanochat.ssd_triton import trapezoidal_recurrent_fwd
    from nanochat.mamba3 import _apply_rope_pairs

    torch.manual_seed(42)
    device = "cuda"

    # (batch, T, n_heads, head_dim, d_state, R, ngroups)
    configs = [
        (2, 1, 4, 64, 64, 1, 4),     # T=1, R=1, per-head groups
        (2, 1, 8, 64, 64, 2, 8),     # T=1, R=2, per-head groups
        (1, 4, 4, 64, 64, 2, 4),     # T=4, R=2
        (2, 1, 8, 128, 128, 2, 1),   # large dims, shared group
        (2, 1, 8, 64, 128, 4, 2),    # R=4, asymmetric, grouped
    ]

    all_pass = True
    for batch, T, n_heads, head_dim, d_state, R, ngroups in configs:
        # Generate inputs
        x = torch.randn(batch, T, n_heads, R, head_dim, device=device, dtype=torch.bfloat16)
        A = -torch.rand(n_heads, device=device, dtype=torch.float32).abs() - 0.1
        B = torch.randn(batch, T, ngroups, R, d_state, device=device, dtype=torch.bfloat16)
        C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=torch.bfloat16)
        dt = torch.rand(batch, T, n_heads, device=device, dtype=torch.float32) * 0.4 + 0.05
        lambda_raw = torch.randn(batch, T, n_heads, device=device, dtype=torch.float32)
        theta_raw = torch.randn(batch, T, d_state // 2, device=device, dtype=torch.float32) * 0.1

        h = torch.randn(batch, n_heads, head_dim, d_state, device=device, dtype=torch.float32) * 0.1
        prev_bx = torch.randn(batch, n_heads, head_dim, d_state, device=device, dtype=torch.float32) * 0.1
        scale = 1.0 / (d_state ** 0.5)

        # ── Triton path ──
        h_tri = h.clone()
        prev_bx_tri = prev_bx.clone()
        cum_angles = -torch.cumsum(
            dt.mean(dim=-1, keepdim=True) * theta_raw, dim=1).float()
        lam = torch.sigmoid(lambda_raw)
        y_tri, state_tri = trapezoidal_recurrent_fwd(
            x, A, B, C, dt, lam, cum_angles,
            {'h': h_tri, 'prev_Bx': prev_bx_tri},
            scale, ngroups, n_heads)

        # ── Python reference (RoPE + recurrence) ──
        h_ref = h.clone()
        prev_bx_ref = prev_bx.clone()
        # Apply RoPE
        cos_a = torch.cos(cum_angles).to(B.dtype).unsqueeze(2)  # (B, T, 1, half)
        sin_a = torch.sin(cum_angles).to(B.dtype).unsqueeze(2)
        C_rot = _apply_rope_pairs(C, cos_a, sin_a)
        B_rot = _apply_rope_pairs(B, cos_a.unsqueeze(3), sin_a.unsqueeze(3))

        # Expand groups
        heads_per_group = n_heads // ngroups
        if ngroups != n_heads:
            B_exp = B_rot.repeat_interleave(heads_per_group, dim=2)
            C_exp = C_rot.repeat_interleave(heads_per_group, dim=2)
        else:
            B_exp = B_rot
            C_exp = C_rot

        lam_ref = torch.sigmoid(lambda_raw)
        outputs = []
        for t in range(T):
            dt_t = dt[:, t]
            lam_t = lam_ref[:, t]
            alpha_t = torch.exp(A * dt_t)
            beta_t = (1 - lam_t) * dt_t * alpha_t
            gamma_t = lam_t * dt_t
            x_t = x[:, t]
            B_t = B_exp[:, t]
            Bx_t = torch.einsum('bhri, bhrj -> bhij', x_t.float(), B_t.float())
            h_ref = (alpha_t[:, :, None, None] * h_ref
                     + beta_t[:, :, None, None] * prev_bx_ref
                     + gamma_t[:, :, None, None] * Bx_t)
            prev_bx_ref = Bx_t
            C_t = C_exp[:, t]
            y_t = torch.einsum('bhn, bhpn -> bhp', C_t.float(), h_ref) * scale
            outputs.append(y_t)
        y_ref = torch.stack(outputs, dim=1)

        # Compare
        y_diff = (y_tri - y_ref).abs().max().item()
        y_scale = y_ref.abs().max().item() + 1e-8
        y_rel = y_diff / y_scale

        h_diff = (state_tri['h'] - h_ref).abs().max().item()
        h_scale_val = h_ref.abs().max().item() + 1e-8
        h_rel = h_diff / h_scale_val

        ok = y_rel < 1e-3 and h_rel < 1e-3
        if not ok:
            print(f"    FAIL ({batch},{T},{n_heads},{head_dim},{d_state},{R},{ngroups}): "
                  f"y_rel={y_rel:.6f}, h_rel={h_rel:.6f}")
            all_pass = False

    print(f"  Trapezoidal recurrent Triton vs Python ({len(configs)} configs): "
          f"{'PASS' if all_pass else 'FAIL'}")
    return all_pass


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

        print("\n16. delta_h Triton vs Python:")
        all_pass &= test_delta_h_triton_vs_pytorch()

        print("\n17. SSD prep Triton vs Python:")
        all_pass &= test_ssd_prep_triton_vs_pytorch()

        print("\n18. RoPE angles Triton vs Python:")
        all_pass &= test_compute_rope_angles_triton_vs_pytorch()

        print("\n19. Trapezoidal recurrent Triton vs Python:")
        all_pass &= test_trapezoidal_recurrent_triton_vs_pytorch()

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
