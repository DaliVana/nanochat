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

5. MIMO (Multi-Input Multi-Output): Rank-R expansion of x and B projections.
   State update becomes a rank-R matmul instead of rank-1 outer product,
   increasing arithmetic intensity from ~2.5 to ~2.5R FLOPs/byte.
   Controlled by mimo_rank parameter (1=SISO, >1=MIMO).

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
    Mamba-3 (Trapezoidal SSD) layer with optional MIMO.

    Architecture:
    1. Input projection: x -> (z, x_raw, B, C, dt, lambda, theta)
    2. SiLU on x_raw (no conv1d)
    3. QK-norm on B and C
    4. Data-dependent RoPE on B and C
    5. Two-SSD scan (gamma-SSD + beta-SSD), with MIMO rank-R if mimo_rank > 1
    6. Output gate: y = rms_norm(y) * silu(z)
    7. Output projection
    """

    def __init__(self, d_model, d_state=64, expand=2, n_heads=None, ngroups=1,
                 chunk_size=256, mimo_rank=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.mimo_rank = mimo_rank
        assert d_state % 2 == 0, f"d_state must be even for RoPE, got {d_state}"
        assert mimo_rank >= 1, f"mimo_rank must be >= 1, got {mimo_rank}"

        # Inner dimension (expanded)
        self.d_inner = int(expand * d_model)
        self.n_heads = n_heads if n_heads is not None else self.d_inner // d_state
        assert self.d_inner % self.n_heads == 0, f"d_inner ({self.d_inner}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_inner // self.n_heads

        # B/C group count (like GQA)
        self.ngroups = ngroups
        assert self.n_heads % self.ngroups == 0, f"n_heads ({self.n_heads}) must be divisible by ngroups ({self.ngroups})"

        # Input projection: z + x*R + B*R + C + dt + lambda + theta
        R = mimo_rank
        self.bc_dim = self.ngroups * self.d_state
        d_proj = (self.d_inner                    # z (gate, single rank)
                  + self.d_inner * R              # x (R copies for MIMO)
                  + self.bc_dim * R               # B (R copies for MIMO)
                  + self.bc_dim                   # C (single rank output)
                  + 2 * self.n_heads              # dt + lambda
                  + self.d_state // 2)            # theta (RoPE frequencies)
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

        # Scale normalizes both the C·B dot product (1/√d_state) and the
        # MIMO rank summation (1/√R) so total output variance matches SISO.
        self._ssd_scale = (d_state * mimo_rank) ** -0.5
        self.register_buffer(
            '_causal_mask',
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _expand_groups(self, x, groups_dim=-2):
        """Expand grouped B/C from (*, ngroups, ...) to (*, n_heads, ...) along groups_dim."""
        if self.ngroups == self.n_heads:
            return x
        return x.repeat_interleave(self.n_heads // self.ngroups, dim=groups_dim)

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
        R = self.mimo_rank

        # 1. Input projection
        proj = self.in_proj(x)  # (B, T, d_proj)
        z, x_raw, B_mat, C_mat, dt_raw, lambda_raw, theta_raw = self._split_projection(proj)

        # 2. SiLU activation (replaces conv1d + SiLU from Mamba-2)
        # x_raw is flat: (B, T, d_inner*R) — SiLU is element-wise
        x_raw = F.silu(x_raw)

        # 3. QK-normalization on B and C
        if R > 1:
            # B: (B, T, ngroups, R, d_state) — norm over d_state, bias broadcasts over R
            B_mat = F.rms_norm(B_mat, (self.d_state,)) * self.B_norm_bias.unsqueeze(-2)
        else:
            # B: (B, T, ngroups, d_state)
            B_mat = F.rms_norm(B_mat, (self.d_state,)) * self.B_norm_bias
        C_mat = F.rms_norm(C_mat, (self.d_state,)) * self.C_norm_bias

        # 4. Discretize dt
        A = -torch.exp(self.A_log.float())  # (n_heads,) always negative
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads)
        dt = dt.clamp(min=1e-4, max=0.5)

        # 5. Data-dependent RoPE on B and C
        B_mat, C_mat = self._apply_data_dependent_rope(B_mat, C_mat, dt, theta_raw)

        # 6. Reshape x to multi-head (with optional R dimension)
        if R > 1:
            x_heads = x_raw.view(B, T, self.n_heads, R, self.head_dim)
        else:
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
        R = self.mimo_rank
        bc = self.bc_dim
        nh = self.n_heads

        z          = proj[..., :d]                                  # (B, T, d_inner)
        offset = d
        x_raw      = proj[..., offset:offset + d * R]              # (B, T, d_inner*R)
        offset += d * R
        B_mat      = proj[..., offset:offset + bc * R]             # (B, T, bc_dim*R)
        offset += bc * R
        C_mat      = proj[..., offset:offset + bc]                 # (B, T, bc_dim)
        offset += bc
        dt_raw     = proj[..., offset:offset + nh]                 # (B, T, n_heads)
        offset += nh
        lambda_raw = proj[..., offset:offset + nh]                 # (B, T, n_heads)
        offset += nh
        theta_raw  = proj[..., offset:]                            # (B, T, d_state//2)

        if R > 1:
            B_mat = B_mat.view(*B_mat.shape[:-1], self.ngroups, R, self.d_state)
        else:
            B_mat = B_mat.view(*B_mat.shape[:-1], self.ngroups, self.d_state)
        C_mat = C_mat.view(*C_mat.shape[:-1], self.ngroups, self.d_state)

        return z, x_raw, B_mat, C_mat, dt_raw, lambda_raw, theta_raw

    def _apply_data_dependent_rope(self, B, C, dt, theta_raw):
        """Apply data-dependent RoPE to B and C.

        Args:
            B:         (batch, T, ngroups, [R,] d_state)
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

        # Negative cumsum: B at time s rotates by -sum(0..s), C at time t by -sum(0..t),
        # so C_t @ B_s encodes positional distance via angle difference sum(s..t).
        cum_angles = -torch.cumsum(raw_angles.float(), dim=1)

        # cos/sin — compute in fp32, cast to input dtype to free fp32 copies
        input_dtype = B.dtype
        cos_a = torch.cos(cum_angles).to(input_dtype).unsqueeze(2)  # (batch, T, 1, d_state//2)
        sin_a = torch.sin(cum_angles).to(input_dtype).unsqueeze(2)
        del cum_angles

        C_rotated = _apply_rope_pairs(C, cos_a, sin_a)

        # cos/sin for B: needs extra unsqueeze when MIMO to broadcast over R
        if self.mimo_rank > 1:
            B_rotated = _apply_rope_pairs(B, cos_a.unsqueeze(3), sin_a.unsqueeze(3))
        else:
            B_rotated = _apply_rope_pairs(B, cos_a, sin_a)

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
            x_heads:    (B, T, n_heads, [R,] head_dim)
            A:          (n_heads,)
            B:          (B, T, ngroups, [R,] d_state) — after QK-norm and RoPE
            C:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
        Returns:
            y: (B, T, n_heads, head_dim)
        """
        lam = torch.sigmoid(lambda_raw)       # (B, T, n_heads) in (0, 1)
        alpha = torch.exp(A * dt)             # (B, T, n_heads) decay
        coeff_beta = (1 - lam) * alpha        # pre-compute for clarity

        if self.mimo_rank > 1:
            # MIMO: x is (B, T, nh, R, hd), need extra unsqueezes
            lam_x = lam.unsqueeze(-1).unsqueeze(-1)           # (B, T, nh, 1, 1)
            coeff_beta_x = coeff_beta.unsqueeze(-1).unsqueeze(-1)

            x_gamma = x_heads * lam_x
            x_beta = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0)) * coeff_beta_x
            B_shifted = F.pad(B[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        else:
            # SISO: x is (B, T, nh, hd)
            x_gamma = x_heads * lam.unsqueeze(-1)
            x_beta = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 1, 0)) * coeff_beta.unsqueeze(-1)
            B_shifted = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))
        del lam, coeff_beta

        # Dispatch to most efficient path
        R = self.mimo_rank
        T = x_heads.shape[1]
        if R == 1 and _HAS_TRITON and x_heads.device.type == 'cuda' and T >= self.chunk_size:
            # SISO + Triton: two kernel calls (Triton computes decay on-the-fly in registers)
            y_gamma = self._ssd_triton(x_gamma, A, B, C, dt)
            del x_gamma
            y_beta = self._ssd_triton(x_beta, A, B_shifted, C, dt)
            del x_beta, B_shifted
            return y_gamma + y_beta
        elif R == 1:
            # SISO + PyTorch: fused path — shared decay + single state propagation
            return self._two_ssd_fused_siso(x_gamma, x_beta, A, B, B_shifted, C, dt)
        else:
            # MIMO: fused path — shared decay + combined state propagation
            return self._two_ssd_fused_mimo(x_gamma, x_beta, A, B, B_shifted, C, dt)

    def _ssd_dispatch(self, x, A, B, C, dt):
        """Dispatch to MIMO, Triton, or PyTorch SSD (single-call path)."""
        if self.mimo_rank > 1:
            # MIMO: always use PyTorch path (can't modify Triton kernel for rank-R)
            return self._ssd_mimo_chunked(x, A, B, C, dt)

        # SISO: use Triton on CUDA, PyTorch fallback otherwise
        B_batch, T, n_heads, head_dim = x.shape
        if _HAS_TRITON and x.device.type == 'cuda' and T >= self.chunk_size:
            return self._ssd_triton(x, A, B, C, dt)
        else:
            return self._ssd_chunked(x, A, B, C, dt)

    def _two_ssd_fused_siso(self, x_gamma, x_beta, A, B_gamma, B_beta, C, dt):
        """Fused Two-SSD for SISO PyTorch path.

        Computes decay infrastructure once (instead of twice) and combines
        delta_h before state propagation (one sequential loop instead of two).
        """
        B_batch, T, n_heads, head_dim = x_gamma.shape
        d_state = C.shape[-1]
        ngroups = self.ngroups
        scale = self._ssd_scale
        chunk_size = min(self.chunk_size, T)

        # Pad to multiple of chunk_size
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x_gamma = F.pad(x_gamma, (0, 0, 0, 0, 0, pad_len))
            x_beta  = F.pad(x_beta,  (0, 0, 0, 0, 0, pad_len))
            B_gamma = F.pad(B_gamma, (0, 0, 0, 0, 0, pad_len))
            B_beta  = F.pad(B_beta,  (0, 0, 0, 0, 0, pad_len))
            C       = F.pad(C,       (0, 0, 0, 0, 0, pad_len))
            dt      = F.pad(dt,      (0, 0, 0, pad_len))
        T_padded = T + pad_len
        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks
        x_g = x_gamma.view(B_batch, n_chunks, L, n_heads, head_dim)
        x_b = x_beta.view(B_batch, n_chunks, L, n_heads, head_dim)
        Bg  = B_gamma.view(B_batch, n_chunks, L, ngroups, d_state)
        Bb  = B_beta.view(B_batch, n_chunks, L, ngroups, d_state)
        C_  = C.view(B_batch, n_chunks, L, ngroups, d_state)
        dt_ = dt.view(B_batch, n_chunks, L, n_heads)

        # === SHARED DECAY (computed once instead of twice) ===
        log_dA = (A * dt_).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)

        log_decay_matrix = cum_log_dA.unsqueeze(3) - cum_log_dA.unsqueeze(2)
        mask = self._causal_mask[:L, :L].reshape(1, 1, L, L, 1)
        log_decay_matrix = log_decay_matrix.masked_fill(mask, float('-inf'))
        decay_matrix = torch.exp(log_decay_matrix)
        del log_decay_matrix

        # === WITHIN-CHUNK: both passes share decay_matrix ===
        C_f = C_.float()
        scores_g = torch.einsum('bcign, bcjgn -> bcijg', C_f, Bg.float()) * scale
        scores_b = torch.einsum('bcign, bcjgn -> bcijg', C_f, Bb.float()) * scale
        if ngroups < n_heads:
            scores_g = scores_g.repeat_interleave(n_heads // ngroups, dim=-1)
            scores_b = scores_b.repeat_interleave(n_heads // ngroups, dim=-1)
        scores_g = scores_g * decay_matrix
        scores_b = scores_b * decay_matrix
        del decay_matrix

        x_dt_g = (x_g * dt_.unsqueeze(-1)).float()
        x_dt_b = (x_b * dt_.unsqueeze(-1)).float()
        y_intra = (torch.einsum('bcijh, bcjhp -> bcihp', scores_g, x_dt_g)
                   + torch.einsum('bcijh, bcjhp -> bcihp', scores_b, x_dt_b))
        del scores_g, scores_b

        # === BETWEEN-CHUNK: shared decay, combined delta_h ===
        chunk_decay = torch.exp(cum_log_dA[:, :, -1, :])
        decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)

        if ngroups < n_heads:
            Bg_exp = Bg.repeat_interleave(n_heads // ngroups, dim=-2)
            Bb_exp = Bb.repeat_interleave(n_heads // ngroups, dim=-2)
        else:
            Bg_exp, Bb_exp = Bg, Bb

        # Sum delta_h from both passes BEFORE state propagation (linearity of recurrence)
        delta_h = (torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                decay_to_end, x_dt_g, Bg_exp.float())
                   + torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                  decay_to_end, x_dt_b, Bb_exp.float()))
        del x_dt_g, x_dt_b

        # Single state propagation pass (instead of two)
        states = torch.zeros(B_batch, n_heads, head_dim, d_state,
                             device=x_gamma.device, dtype=torch.float32)
        all_states = []
        for c in range(n_chunks):
            all_states.append(states)
            states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
        all_states = torch.stack(all_states, dim=1)
        del delta_h

        # Inter-chunk output
        if ngroups < n_heads:
            C_expanded = C_.repeat_interleave(n_heads // ngroups, dim=-2)
        else:
            C_expanded = C_
        decay_from_start = torch.exp(cum_log_dA)
        state_contribution = torch.einsum('bcihn, bchpn -> bcihp',
                                           C_expanded.float(), all_states)
        y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)

        y = (y_intra + y_inter).to(x_gamma.dtype)
        y = y.contiguous().view(B_batch, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T]
        return y

    def _two_ssd_fused_mimo(self, x_g, x_b, A, Bg, Bb, C, dt):
        """Fused Two-SSD for MIMO: shared decay + combined state propagation.

        Optimizations over two separate _ssd_mimo_chunked calls:
        1. Decay infrastructure (decay_matrix, cum_log_dA, etc.) computed once
        2. delta_h combined before state propagation (one sequential loop)
        3. Broadcasting replaces repeat_interleave for within-chunk scores
        4. Group-aware einsum avoids B expansion for between-chunk
        5. x*dt pre-computed once for all ranks
        """
        batch_size, T, n_heads, R, head_dim = x_g.shape
        d_state = C.shape[-1]
        ngroups = C.shape[-2]
        scale = self._ssd_scale
        chunk_size = min(self.chunk_size, T)

        # Pad T to multiple of chunk_size
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x_g = F.pad(x_g, (0, 0, 0, 0, 0, 0, 0, pad_len))
            x_b = F.pad(x_b, (0, 0, 0, 0, 0, 0, 0, pad_len))
            Bg  = F.pad(Bg,  (0, 0, 0, 0, 0, 0, 0, pad_len))
            Bb  = F.pad(Bb,  (0, 0, 0, 0, 0, 0, 0, pad_len))
            C   = F.pad(C,   (0, 0, 0, 0, 0, pad_len))
            dt  = F.pad(dt,  (0, 0, 0, pad_len))
        T_padded = T + pad_len
        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks
        x_g = x_g.view(batch_size, n_chunks, L, n_heads, R, head_dim)
        x_b = x_b.view(batch_size, n_chunks, L, n_heads, R, head_dim)
        Bg  = Bg.view(batch_size, n_chunks, L, ngroups, R, d_state)
        Bb  = Bb.view(batch_size, n_chunks, L, ngroups, R, d_state)
        C_  = C.view(batch_size, n_chunks, L, ngroups, d_state)
        dt_ = dt.view(batch_size, n_chunks, L, n_heads)

        # === SHARED DECAY (computed once instead of twice) ===
        log_dA = (A * dt_).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)

        log_decay_matrix = cum_log_dA.unsqueeze(3) - cum_log_dA.unsqueeze(2)
        mask = self._causal_mask[:L, :L].reshape(1, 1, L, L, 1)
        log_decay_matrix = log_decay_matrix.masked_fill(mask, float('-inf'))
        decay_matrix = torch.exp(log_decay_matrix)
        del log_decay_matrix

        # Pre-compute x*dt for all ranks (reused in within-chunk and between-chunk)
        dt_broad = dt_.unsqueeze(-1).unsqueeze(-1)  # (batch, nc, L, nh, 1, 1)
        x_dt_g = (x_g * dt_broad).float()
        x_dt_b = (x_b * dt_broad).float()

        C_f = C_.float()

        # Pre-reshape decay for group broadcasting (avoid repeat_interleave in loop)
        hpg = n_heads // ngroups
        if ngroups < n_heads:
            decay_grouped = decay_matrix.view(
                batch_size, n_chunks, L, L, ngroups, hpg)

        # === WITHIN-CHUNK: R-loop, both passes share decay_matrix ===
        y_intra = torch.zeros(batch_size, n_chunks, L, n_heads, head_dim,
                               device=x_g.device, dtype=torch.float32)

        for r in range(R):
            Bg_r = Bg[:, :, :, :, r, :].float()
            Bb_r = Bb[:, :, :, :, r, :].float()
            scores_g = torch.einsum('bcign, bcjgn -> bcijg', C_f, Bg_r) * scale
            scores_b = torch.einsum('bcign, bcjgn -> bcijg', C_f, Bb_r) * scale

            # Apply decay via broadcasting (no repeat_interleave allocation)
            if ngroups < n_heads:
                scores_g = (scores_g.unsqueeze(-1) * decay_grouped
                            ).reshape(batch_size, n_chunks, L, L, n_heads)
                scores_b = (scores_b.unsqueeze(-1) * decay_grouped
                            ).reshape(batch_size, n_chunks, L, L, n_heads)
            else:
                scores_g = scores_g * decay_matrix
                scores_b = scores_b * decay_matrix

            y_intra = y_intra + (
                torch.einsum('bcijh, bcjhp -> bcihp',
                             scores_g, x_dt_g[:, :, :, :, r, :])
                + torch.einsum('bcijh, bcjhp -> bcihp',
                               scores_b, x_dt_b[:, :, :, :, r, :]))

        del decay_matrix
        if ngroups < n_heads:
            del decay_grouped

        # === BETWEEN-CHUNK: shared decay, combined delta_h ===
        chunk_decay = torch.exp(cum_log_dA[:, :, -1, :])
        decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)

        # Group-aware einsum: keep B in group space, restructure heads as (ngroups, hpg)
        # This avoids expanding B from ngroups to n_heads entirely.
        delta_h = torch.zeros(batch_size, n_chunks, n_heads, head_dim, d_state,
                               device=x_g.device, dtype=torch.float32)
        if ngroups < n_heads:
            decay_to_end_g = decay_to_end.view(
                batch_size, n_chunks, L, ngroups, hpg)
            for r in range(R):
                xg_r = x_dt_g[:, :, :, :, r, :].view(
                    batch_size, n_chunks, L, ngroups, hpg, head_dim)
                xb_r = x_dt_b[:, :, :, :, r, :].view(
                    batch_size, n_chunks, L, ngroups, hpg, head_dim)
                delta_h = delta_h + (
                    torch.einsum('bclgk, bclgkp, bclgn -> bcgkpn',
                                 decay_to_end_g, xg_r,
                                 Bg[:, :, :, :, r, :].float())
                    + torch.einsum('bclgk, bclgkp, bclgn -> bcgkpn',
                                   decay_to_end_g, xb_r,
                                   Bb[:, :, :, :, r, :].float())
                ).reshape(batch_size, n_chunks, n_heads, head_dim, d_state)
        else:
            for r in range(R):
                delta_h = delta_h + (
                    torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                 decay_to_end, x_dt_g[:, :, :, :, r, :],
                                 Bg[:, :, :, :, r, :].float())
                    + torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                   decay_to_end, x_dt_b[:, :, :, :, r, :],
                                   Bb[:, :, :, :, r, :].float()))
        del x_dt_g, x_dt_b

        # Single state propagation pass (instead of two)
        states = delta_h.new_zeros(batch_size, n_heads, head_dim, d_state)
        all_states = []
        for c in range(n_chunks):
            all_states.append(states)
            states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
        all_states = torch.stack(all_states, dim=1)
        del delta_h

        # Inter-chunk output
        if ngroups < n_heads:
            C_expanded = C_.repeat_interleave(n_heads // ngroups, dim=-2)
        else:
            C_expanded = C_
        decay_from_start = torch.exp(cum_log_dA)
        state_contribution = torch.einsum('bcihn, bchpn -> bcihp',
                                           C_expanded.float(), all_states)
        y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)

        y = (y_intra + y_inter).to(x_g.dtype)
        y = y.contiguous().view(batch_size, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T]
        return y

    def _ssd_triton(self, x, A, B, C, dt):
        """Triton forward (reuses Mamba-2 custom op). SISO only."""
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
        """PyTorch chunked SSD (reuses Mamba-2 standalone function). SISO only."""
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

    def _ssd_mimo_chunked(self, x, A, B, C, dt):
        """MIMO SSD: loop over R ranks for within-chunk, contract R for between-chunk.

        The key insight: within-chunk quadratic computation is already compute-bound,
        so we loop over R to keep memory constant. Between-chunk state updates sum
        over R (rank-R matmul), which is where the arithmetic intensity benefit lives.

        Args:
            x:  (batch, T, n_heads, R, head_dim)
            A:  (n_heads,)
            B:  (batch, T, ngroups, R, d_state)
            C:  (batch, T, ngroups, d_state)  — single-rank output
            dt: (batch, T, n_heads)
        Returns:
            y:  (batch, T, n_heads, head_dim)
        """
        batch_size, T, n_heads, R, head_dim = x.shape
        d_state = C.shape[-1]
        ngroups = C.shape[-2]
        chunk_size = min(self.chunk_size, T)
        scale = self._ssd_scale

        # Pad T to multiple of chunk_size
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_len))      # 5D: pad T
            B = F.pad(B, (0, 0, 0, 0, 0, 0, 0, pad_len))       # 5D: pad T
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))              # 4D: pad T
            dt = F.pad(dt, (0, 0, 0, pad_len))                   # 3D: pad T
            T_padded = T + pad_len
        else:
            T_padded = T

        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks
        x  = x.view(batch_size, n_chunks, L, n_heads, R, head_dim)
        B_ = B.view(batch_size, n_chunks, L, ngroups, R, d_state)
        C_ = C.view(batch_size, n_chunks, L, ngroups, d_state)
        dt = dt.view(batch_size, n_chunks, L, n_heads)

        # Decay computation (same as SISO — doesn't depend on R)
        log_dA = (A * dt).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)  # (batch, nc, L, nh)

        # Within-chunk decay matrix
        log_decay_matrix = cum_log_dA.unsqueeze(3) - cum_log_dA.unsqueeze(2)
        mask = self._causal_mask[:L, :L].reshape(1, 1, L, L, 1)
        log_decay_matrix = log_decay_matrix.masked_fill(mask, float('-inf'))
        decay_matrix = torch.exp(log_decay_matrix)  # (batch, nc, L, L, nh)

        # === WITHIN-CHUNK: loop over R for memory efficiency ===
        y_intra = torch.zeros(batch_size, n_chunks, L, n_heads, head_dim,
                               device=x.device, dtype=torch.float32)

        C_f = C_.float()  # cast once, reuse across R iterations
        dt_exp = dt.unsqueeze(-1)  # pre-expand for x * dt

        for r in range(R):
            # Scores for rank r: C @ B_r^T
            scores_r = torch.einsum('bcign, bcjgn -> bcijg',
                                     C_f, B_[:, :, :, :, r, :].float()) * scale
            if ngroups < n_heads:
                scores_r = scores_r.repeat_interleave(n_heads // ngroups, dim=-1)
            scores_r = scores_r * decay_matrix

            x_dt_r = (x[:, :, :, :, r, :] * dt_exp).float()
            y_intra = y_intra + torch.einsum('bcijh, bcjhp -> bcihp', scores_r, x_dt_r)

        del decay_matrix  # free large (batch, nc, L, L, nh) tensor

        # === BETWEEN-CHUNK: contract over R for shared-state rank-R update ===
        chunk_decay = torch.exp(cum_log_dA[:, :, -1, :])  # (batch, nc, nh)
        decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)  # (batch, nc, L, nh)

        delta_h = torch.zeros(batch_size, n_chunks, n_heads, head_dim, d_state,
                               device=x.device, dtype=torch.float32)

        # Pre-expand B groups once (avoid repeat_interleave per rank)
        if ngroups < n_heads:
            B_expanded = B_.repeat_interleave(n_heads // ngroups, dim=-3)
        else:
            B_expanded = B_

        for r in range(R):
            x_dt_r = (x[:, :, :, :, r, :] * dt_exp).float()
            delta_h = delta_h + torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                              decay_to_end, x_dt_r,
                                              B_expanded[:, :, :, :, r, :].float())

        del B_expanded  # free expanded B

        # State propagation — shared state (same as SISO)
        # NOTE: must use list+stack (not in-place assignment) to preserve autograd
        states = delta_h.new_zeros(batch_size, n_heads, head_dim, d_state)
        all_states = []
        for c in range(n_chunks):
            all_states.append(states)
            states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
        all_states = torch.stack(all_states, dim=1)  # (batch, nc, nh, hd, ds)
        del delta_h

        # Inter-chunk output — same as SISO (single-rank C)
        if ngroups < n_heads:
            C_expanded = C_.repeat_interleave(n_heads // ngroups, dim=-2)
        else:
            C_expanded = C_
        decay_from_start = torch.exp(cum_log_dA)
        state_contribution = torch.einsum('bcihn, bchpn -> bcihp',
                                           C_expanded.float(), all_states)
        y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)

        y = (y_intra + y_inter).to(x.dtype)
        y = y.contiguous().view(batch_size, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T, :, :]
        return y

    def _trapezoidal_recurrent(self, x, A, B, C, dt, lambda_raw, ssm_state):
        """Trapezoidal recurrent computation for inference.

        State carries both h (SSM state) and prev_Bx (previous B*x product).
        With MIMO, the B*x product is a rank-R matmul instead of rank-1 outer product.

        Args:
            x:          (B, T, n_heads, [R,] head_dim) — typically T=1
            A:          (n_heads,)
            B:          (B, T, ngroups, [R,] d_state) — after QK-norm and RoPE
            C:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
            ssm_state:  dict with 'h' and 'prev_Bx'
        Returns:
            y: (B, T, n_heads, head_dim)
            new_state: updated dict
        """
        R = self.mimo_rank

        if R > 1:
            B_batch, T, n_heads, _, head_dim = x.shape
        else:
            B_batch, T, n_heads, head_dim = x.shape

        # Expand B groups to heads
        if R > 1:
            B_exp = self._expand_groups(B, groups_dim=-3)  # (B, T, n_heads, R, d_state)
        else:
            B_exp = self._expand_groups(B)                  # (B, T, n_heads, d_state)
        C_exp = self._expand_groups(C)                       # (B, T, n_heads, d_state)

        lam = torch.sigmoid(lambda_raw)

        h = ssm_state['h']             # (B, n_heads, head_dim, d_state)
        prev_Bx = ssm_state['prev_Bx'] # (B, n_heads, head_dim, d_state)

        outputs = []
        for t in range(T):
            C_t = C_exp[:, t]    # (B, n_heads, d_state)
            dt_t = dt[:, t]      # (B, n_heads)
            lam_t = lam[:, t]    # (B, n_heads)

            alpha_t = torch.exp(A * dt_t)            # (B, n_heads)
            # Note: dt is included in beta/gamma (not in Bx below), matching the forward
            # path where x is pre-weighted by (1-lam)*alpha and the SSD kernel applies dt.
            beta_t = (1 - lam_t) * dt_t * alpha_t
            gamma_t = lam_t * dt_t

            # Current B*x product (rank-R matmul for MIMO, rank-1 outer product for SISO)
            if R > 1:
                x_t = x[:, t]        # (B, n_heads, R, head_dim)
                B_t = B_exp[:, t]    # (B, n_heads, R, d_state)
                Bx_t = torch.einsum('bhri, bhrj -> bhij', x_t, B_t)  # rank-R matmul
            else:
                x_t = x[:, t]        # (B, n_heads, head_dim)
                B_t = B_exp[:, t]    # (B, n_heads, d_state)
                Bx_t = torch.einsum('bhi, bhj -> bhij', x_t, B_t)    # rank-1 outer product

            # Trapezoidal state update
            h = (alpha_t[:, :, None, None] * h
                 + beta_t[:, :, None, None] * prev_Bx
                 + gamma_t[:, :, None, None] * Bx_t)

            prev_Bx = Bx_t

            # Output (same for SISO and MIMO — C is single-rank)
            y_t = torch.einsum('bhn, bhpn -> bhp', C_t, h) * self._ssd_scale
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, {'h': h, 'prev_Bx': prev_Bx}

    def _fused_norm_gate(self, y_raw, z):
        """RMSNorm and SiLU gate (fused by whole-model torch.compile)."""
        y_norm = F.rms_norm(y_raw, (y_raw.size(-1),))
        return y_norm * F.silu(z)
