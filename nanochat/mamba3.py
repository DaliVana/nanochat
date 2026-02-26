"""
Mamba-3 (Trapezoidal SSD) layer for nanochat.

Improvements over Mamba-2:
1. Trapezoidal discretization (2nd-order) via Two-SSD decomposition:
   h_t = alpha_t * h_{t-1} + beta_t * B_{t-1} * x_{t-1} + gamma_t * B_t * x_t
   Implemented by calling the standard SSD kernel twice with shifted inputs.

2. Data-dependent RoPE on B and C projections:
   Cumulative rotation angles derived from input, enabling state-tracking
   capabilities (e.g. parity, modular arithmetic) that Mamba-2 cannot learn.

3. QK-normalization on B and C (RMSNorm + learnable bias):
   Stabilizes training, analogous to QK-norm in attention.

4. No conv1d: The trapezoidal rule's dependency on x_{t-1} subsumes
   the role of the short causal convolution from Mamba-2.

References:
- Mamba-3: https://openreview.net/forum?id=HwCvaJOiCj
- Mamba-2: https://arxiv.org/abs/2405.21060
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the SSD infrastructure from Mamba-2
from nanochat.mamba2 import _HAS_TRITON, _ssd_fwd_op, _ssd_chunked_pytorch


def _apply_rope_pairs(x, cos, sin):
    """Apply RoPE rotation to tensor with paired dimensions.

    x:   (..., d_state)  where d_state is even
    cos: (..., d_state//2) broadcastable to x's leading dims
    sin: (..., d_state//2) broadcastable to x's leading dims

    Splits d_state into first-half / second-half pairs and applies 2D rotation.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Mamba3Layer(nn.Module):
    """
    Mamba-3 (Trapezoidal SSD) layer.

    Architecture:
    1. Input projection: x -> (z, x_raw, B, C, dt, lambda, theta)
    2. SiLU on x_raw (no conv1d)
    3. QK-norm on B and C
    4. Data-dependent RoPE on B and C
    5. Two-SSD scan (gamma-SSD + beta-SSD)
    6. Output gate: y = rms_norm(y) * silu(z)
    7. Output projection
    """

    def __init__(self, d_model, d_state=64, expand=2, n_heads=None, ngroups=1, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        assert d_state % 2 == 0, f"d_state must be even for RoPE, got {d_state}"

        # Inner dimension (expanded)
        self.d_inner = int(expand * d_model)
        self.n_heads = n_heads if n_heads is not None else self.d_inner // d_state
        assert self.d_inner % self.n_heads == 0, f"d_inner ({self.d_inner}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_inner // self.n_heads

        # B/C group count (like GQA)
        self.ngroups = ngroups
        assert self.n_heads % self.ngroups == 0, f"n_heads ({self.n_heads}) must be divisible by ngroups ({self.ngroups})"

        # Input projection: z + x + B + C + dt + lambda + theta
        self.bc_dim = self.ngroups * self.d_state
        d_proj = (2 * self.d_inner          # z + x
                  + 2 * self.bc_dim         # B + C
                  + 2 * self.n_heads        # dt + lambda
                  + self.d_state // 2)      # theta (RoPE frequencies)
        self.in_proj = nn.Linear(d_model, d_proj, bias=False)

        # No conv1d in Mamba-3 (trapezoidal rule subsumes it)

        # Learnable log(A) parameter (per-head, always negative after exp)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))

        # dt bias (per-head, added to dt before softplus)
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))

        # QK-normalization: learnable per-group bias for B and C (init ones = identity)
        self.B_norm_bias = nn.Parameter(torch.ones(self.ngroups, self.d_state))
        self.C_norm_bias = nn.Parameter(torch.ones(self.ngroups, self.d_state))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._ssd_scale = d_state ** -0.5
        self.register_buffer(
            '_causal_mask',
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _expand_groups(self, x):
        """Expand grouped B/C from (*, ngroups, d_state) to (*, n_heads, d_state)."""
        if self.ngroups == self.n_heads:
            return x
        return x.repeat_interleave(self.n_heads // self.ngroups, dim=-2)

    def forward(self, x, ssm_state=None):
        """
        Args:
            x: (B, T, d_model) input tensor
            ssm_state: None for training, dict for inference
        Returns:
            y: (B, T, d_model) output tensor
            new_ssm_state: updated state dict (None during training)
            None: placeholder for conv_state compatibility
        """
        B, T, _ = x.shape

        # 1. Input projection
        proj = self.in_proj(x)  # (B, T, d_proj)
        z, x_raw, B_mat, C_mat, dt_raw, lambda_raw, theta_raw = self._split_projection(proj)

        # 2. SiLU activation (replaces conv1d + SiLU from Mamba-2)
        x_raw = F.silu(x_raw)

        # 3. QK-normalization on B and C
        B_mat = F.rms_norm(B_mat, (self.d_state,)) * self.B_norm_bias
        C_mat = F.rms_norm(C_mat, (self.d_state,)) * self.C_norm_bias

        # 4. Discretize dt
        A = -torch.exp(self.A_log.float())  # (n_heads,) always negative
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads)
        dt = dt.clamp(min=1e-4, max=0.5)

        # 5. Data-dependent RoPE on B and C
        B_mat, C_mat = self._apply_data_dependent_rope(B_mat, C_mat, dt, theta_raw)

        # 6. Reshape x to multi-head
        x_heads = x_raw.view(B, T, self.n_heads, self.head_dim)

        # 7. SSD computation
        if ssm_state is None:
            # Training: Two-SSD decomposition
            y_heads = self._two_ssd_forward(x_heads, A, B_mat, C_mat, dt, lambda_raw)
        else:
            # Inference: trapezoidal recurrence
            y_heads, ssm_state = self._trapezoidal_recurrent(
                x_heads, A, B_mat, C_mat, dt, lambda_raw, ssm_state)

        # 8. Output gate: norm then gate with z
        y = y_heads.reshape(B, T, self.d_inner)
        y = self._fused_norm_gate(y, z)

        # 9. Output projection
        y = self.out_proj(y)

        return y, ssm_state, None

    def _split_projection(self, proj):
        """Split the input projection into its components."""
        d = self.d_inner
        bc = self.bc_dim
        nh = self.n_heads
        half_d = self.d_state // 2

        z          = proj[..., :d]                                  # (B, T, d_inner)
        x_raw      = proj[..., d:2*d]                              # (B, T, d_inner)
        B_mat      = proj[..., 2*d:2*d+bc]                         # (B, T, ngroups*d_state)
        C_mat      = proj[..., 2*d+bc:2*d+2*bc]                    # (B, T, ngroups*d_state)
        dt_raw     = proj[..., 2*d+2*bc:2*d+2*bc+nh]               # (B, T, n_heads)
        lambda_raw = proj[..., 2*d+2*bc+nh:2*d+2*bc+2*nh]          # (B, T, n_heads)
        theta_raw  = proj[..., 2*d+2*bc+2*nh:]                     # (B, T, d_state//2)

        B_mat = B_mat.view(*B_mat.shape[:-1], self.ngroups, self.d_state)
        C_mat = C_mat.view(*C_mat.shape[:-1], self.ngroups, self.d_state)

        return z, x_raw, B_mat, C_mat, dt_raw, lambda_raw, theta_raw

    def _apply_data_dependent_rope(self, B, C, dt, theta_raw):
        """Apply data-dependent RoPE to B and C.

        Args:
            B:         (batch, T, ngroups, d_state)
            C:         (batch, T, ngroups, d_state)
            dt:        (batch, T, n_heads)
            theta_raw: (batch, T, d_state//2)
        Returns:
            B_rotated, C_rotated: same shapes as input
        """
        # Use mean dt across heads as the group-level timestep for RoPE
        dt_mean = dt.mean(dim=-1, keepdim=True)  # (batch, T, 1)

        # raw_angles: (batch, T, d_state//2)
        raw_angles = dt_mean * theta_raw

        # Cumulative angles over the full sequence (computed before chunking)
        cum_angles = -torch.cumsum(raw_angles.float(), dim=1)

        cos_a = torch.cos(cum_angles).unsqueeze(2)  # (batch, T, 1, d_state//2)
        sin_a = torch.sin(cum_angles).unsqueeze(2)  # (batch, T, 1, d_state//2)

        B_rotated = _apply_rope_pairs(B, cos_a, sin_a)
        C_rotated = _apply_rope_pairs(C, cos_a, sin_a)

        return B_rotated, C_rotated

    def _two_ssd_forward(self, x_heads, A, B, C, dt, lambda_raw):
        """Two-SSD decomposition for trapezoidal discretization (training).

        Decomposes the trapezoidal recurrence into two standard SSD calls:
        - gamma-SSD: current timestep contribution
        - beta-SSD:  previous timestep contribution

        The SSD kernel internally computes x * dt for the input weighting.
        To get the correct trapezoidal coefficients, we pre-weight x so that
        the kernel's x * dt gives us x * gamma or x * beta:
        - gamma = lam * dt, so pass x * lam → kernel does (x*lam)*dt = x*gamma
        - beta = (1-lam) * dt * alpha, so pass x * (1-lam) * alpha → kernel does
          (x*(1-lam)*alpha)*dt = x*beta

        Args:
            x_heads:    (B, T, n_heads, head_dim)
            A:          (n_heads,)
            B:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            C:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
        Returns:
            y: (B, T, n_heads, head_dim)
        """
        lam = torch.sigmoid(lambda_raw)       # (B, T, n_heads) in (0, 1)
        alpha = torch.exp(A * dt)             # (B, T, n_heads) decay

        # gamma-SSD: pre-weight x by lam so kernel's x*dt gives x*lam*dt = x*gamma
        x_gamma = x_heads * lam.unsqueeze(-1)

        # beta-SSD: pre-weight shifted x by (1-lam)*alpha so kernel's x*dt gives x*beta
        x_shifted = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 1, 0))
        x_beta = x_shifted * ((1 - lam) * alpha).unsqueeze(-1)

        B_shifted = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))

        # Both SSD calls reuse the Mamba-2 kernel with the same A and dt
        y_gamma = self._ssd_dispatch(x_gamma, A, B, C, dt)
        y_beta = self._ssd_dispatch(x_beta, A, B_shifted, C, dt)

        return y_gamma + y_beta

    def _ssd_dispatch(self, x, A, B, C, dt):
        """Dispatch to Triton or PyTorch SSD (reuses Mamba-2 infrastructure)."""
        B_batch, T, n_heads, head_dim = x.shape
        if _HAS_TRITON and x.device.type == 'cuda' and T >= self.chunk_size:
            return self._ssd_triton(x, A, B, C, dt)
        else:
            return self._ssd_chunked(x, A, B, C, dt)

    def _ssd_triton(self, x, A, B, C, dt):
        """Triton forward (reuses Mamba-2 custom op)."""
        B_batch, T, n_heads, head_dim = x.shape
        chunk_size = min(self.chunk_size, T)

        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))

        y = _ssd_fwd_op(x, A, B, C, dt, chunk_size, self._ssd_scale)

        if pad_len > 0:
            y = y[:, :T, :, :]
        return y

    def _ssd_chunked(self, x, A, B, C, dt):
        """PyTorch chunked SSD (reuses Mamba-2 standalone function)."""
        B_batch, T, n_heads, head_dim = x.shape
        chunk_size = min(self.chunk_size, T)

        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))

        y = _ssd_chunked_pytorch(x, A, B, C, dt, self._causal_mask,
                                  self.ngroups, self._ssd_scale, chunk_size)

        if pad_len > 0:
            y = y[:, :T, :, :]
        return y

    def _trapezoidal_recurrent(self, x, A, B, C, dt, lambda_raw, ssm_state):
        """Trapezoidal recurrent computation for inference.

        State carries both h (SSM state) and prev_Bx (previous B*x outer product).

        Args:
            x:          (B, T, n_heads, head_dim) — typically T=1
            A:          (n_heads,)
            B:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            C:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
            ssm_state:  dict with 'h' and 'prev_Bx'
        Returns:
            y: (B, T, n_heads, head_dim)
            new_state: updated dict
        """
        B_batch, T, n_heads, head_dim = x.shape

        B_exp = self._expand_groups(B)  # (B, T, n_heads, d_state)
        C_exp = self._expand_groups(C)

        lam = torch.sigmoid(lambda_raw)

        h = ssm_state['h']             # (B, n_heads, head_dim, d_state)
        prev_Bx = ssm_state['prev_Bx'] # (B, n_heads, head_dim, d_state)

        outputs = []
        for t in range(T):
            x_t = x[:, t]        # (B, n_heads, head_dim)
            B_t = B_exp[:, t]    # (B, n_heads, d_state)
            C_t = C_exp[:, t]    # (B, n_heads, d_state)
            dt_t = dt[:, t]      # (B, n_heads)
            lam_t = lam[:, t]    # (B, n_heads)

            alpha_t = torch.exp(A * dt_t)            # (B, n_heads)
            beta_t = (1 - lam_t) * dt_t * alpha_t
            gamma_t = lam_t * dt_t

            # Current outer product: B_t * x_t
            Bx_t = torch.einsum('bhi, bhj -> bhij', x_t, B_t)  # (B, nh, hd, d_state)

            # Trapezoidal state update
            h = (alpha_t[:, :, None, None] * h
                 + beta_t[:, :, None, None] * prev_Bx
                 + gamma_t[:, :, None, None] * Bx_t)

            prev_Bx = Bx_t

            # Output
            y_t = torch.einsum('bhn, bhpn -> bhp', C_t, h) * self._ssd_scale
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, {'h': h, 'prev_Bx': prev_Bx}

    @torch.compile
    def _fused_norm_gate(self, y_raw, z):
        """Fused RMSNorm and SiLU gate via torch.compile."""
        y_norm = F.rms_norm(y_raw, (y_raw.size(-1),))
        return y_norm * F.silu(z)
