"""
Mamba-3 (Trapezoidal SSD) layer for nanochat.

================================================================================
Architecture Overview
================================================================================

Mamba-3 extends Mamba-2's Structured State Space Duality (SSD) with three key
improvements that close the gap with Transformers on state-tracking tasks while
preserving linear-time inference:

1. Trapezoidal discretization (2nd-order) via Two-SSD decomposition:
   h_t = alpha_t * h_{t-1} + beta_t * B_{t-1} * x_{t-1} + gamma_t * B_t * x_t
   Mamba-2 uses Euler discretization (1st-order): h_t = alpha * h_{t-1} + B_t * x_t,
   which only looks at the current input x_t when updating state. The trapezoidal
   rule adds a dependency on the *previous* input x_{t-1} via B_{t-1}, giving the
   model a two-point stencil — it can reason about how the input changed between
   consecutive timesteps, not just its current value. This is implemented
   efficiently by running the standard SSD kernel twice with shifted inputs
   (gamma-SSD for current, beta-SSD for previous) and summing the results.

2. Data-dependent RoPE on B and C projections:
   Cumulative rotation angles derived from input, enabling state-tracking
   capabilities (e.g. parity, modular arithmetic) that Mamba-2 cannot learn.
   The rotation angle at each position is a function of the input (via learned
   theta frequencies modulated by dt), and angles accumulate via cumsum. The
   dot product C_t @ B_s then encodes a relative-position-like signal based on
   the angular distance sum(s..t), analogous to RoPE in attention.

3. QK-normalization on B and C (RMSNorm + learnable bias):
   Stabilizes training, analogous to QK-norm in attention. B and C are
   normalized to unit RMS then scaled by a learnable per-group bias vector.

4. No conv1d: The trapezoidal rule's dependency on x_{t-1} subsumes
   the role of the short causal convolution from Mamba-2.

5. MIMO (Multi-Input Multi-Output): Rank-R expansion of x and B projections.
   State update becomes a rank-R matmul instead of rank-1 outer product,
   increasing arithmetic intensity from ~2.5 to ~2.5R FLOPs/byte.
   Controlled by mimo_rank parameter (1=SISO, >1=MIMO).

================================================================================
Data Flow (per layer)
================================================================================

  input x: (B, T, d_model)
      |
      v
  in_proj ──> z (gate), x_raw (R copies), B (R copies), C, dt, lambda, theta
      |
      v
  SiLU(x_raw)                          -- nonlinear activation (replaces conv1d)
      |
      v
  QK-norm on B, C                      -- RMSNorm + learnable bias
      |
      v
  Data-dependent RoPE on B, C          -- cumulative rotary encoding
      |
      v
  Two-SSD scan                         -- gamma-SSD + beta-SSD (trapezoidal)
      |                                   with MIMO rank-R
      v
  RMSNorm(y) * SiLU(z)                 -- output gate (like GLU)
      |
      v
  out_proj ──> output: (B, T, d_model)

================================================================================
Key Parameters
================================================================================

d_model (int):
    Model embedding dimension. This is the input and output width of the layer.
    The Mamba layer reads d_model-wide vectors, processes them in an expanded
    internal space, and projects back to d_model.

expand (int, default=2):
    Expansion factor for the inner working dimension. Sets d_inner = expand * d_model.
    This controls how much wider the SSM computation is compared to the residual
    stream, analogous to the 4x expansion in a Transformer MLP.

    The inner dimension d_inner is divided into n_heads, each of width head_dim.
    A larger expand increases capacity per layer at the cost of more parameters
    and compute in the input/output projections.

    NOTE: In the Samba config, expand=1 is used because the Mamba layer already
    acts as both the "attention" and "MLP" (the z-gate provides channel mixing),
    so there is no separate MLP block after Mamba. In a pure Mamba stack you'd
    typically use expand=2 to match the parameter count of attn+MLP.

d_state (int, default=64):
    SSM state dimension per head. Each of the n_heads heads maintains a hidden
    state matrix h of shape (head_dim, d_state). This is the "memory" of the
    recurrence — at each timestep, the state is updated via:
        h_t = decay * h_{t-1} + x_t ⊗ B_t    (outer product, rank-1 or rank-R)
    and the output is read via:
        y_t = C_t @ h_t

    Larger d_state = more expressive state (more information retained across
    time), but costs O(d_state) per head in both the B/C projections and the
    state update. The total recurrent state size is n_heads * head_dim * d_state.

    d_state also determines the RoPE dimensionality (must be even for the
    paired rotation). In the SSD algorithm, d_state controls the size of the
    C @ B^T attention-like score matrix within each chunk (L x L x ngroups,
    contracted over d_state).

    Typical values: 64 (default), 128 for higher capacity. Diminishing returns
    beyond ~128 since the state is low-rank updated (rank-1 or rank-R per step).

ngroups (int, default=1):
    Number of B/C groups, analogous to Grouped Query Attention (GQA) in
    Transformers. Controls how B and C projection matrices are shared across
    heads:

    - ngroups=1: All heads share one (B, C) pair. Maximum parameter sharing,
      minimum projection cost. Each B/C vector has d_state dimensions.
      Total B/C projection size = 1 * d_state (+ R copies for MIMO B).

    - ngroups=n_heads: Each head gets its own (B, C) — "multi-head" SSM,
      analogous to MHA. Maximum expressiveness, but B/C projections scale
      with n_heads.

    - 1 < ngroups < n_heads: Intermediate sharing. Each group of
      (n_heads / ngroups) heads shares one (B, C) pair, like GQA.

    The B/C projection width is ngroups * d_state. Within the SSD algorithm,
    grouped B/C are expanded to per-head via repeat_interleave before the
    score computation. Using ngroups=1 is the most parameter-efficient and is
    recommended unless you have evidence that per-head B/C helps.

mimo_rank (int, default=1):
    MIMO (Multi-Input Multi-Output) rank. Controls the rank of the state
    update at each timestep:

    - mimo_rank=1 (SISO): Standard Mamba. The state update is a rank-1 outer
      product: delta_h = x_t ⊗ B_t (head_dim x d_state, rank 1). This means
      each timestep can only write a rank-1 "slice" of information into the
      state matrix. The input x and B projection are single vectors.

    - mimo_rank=R > 1 (MIMO): The input x is projected to R copies
      (d_inner * R total), and B is also projected to R copies (ngroups * d_state * R).
      The state update becomes a rank-R matmul:
          delta_h = sum_{r=1}^{R} x_t^r ⊗ B_t^r    (rank-R update)
      This allows R times more information to be written into state per timestep.

      The output C remains single-rank (not replicated), so MIMO is
      "multi-input, single-output" for the read side.

    Why MIMO matters — arithmetic intensity:
      In SISO, the state update is memory-bandwidth-bound: each timestep reads
      x (head_dim) and B (d_state) to produce a (head_dim x d_state) outer
      product — only ~2.5 FLOPs per byte loaded. With MIMO rank R, the same
      state matrix receives R outer products, giving ~2.5R FLOPs per byte.
      This makes the between-chunk state propagation compute-bound rather than
      memory-bound on modern GPUs, improving hardware utilization.

    Cost: MIMO adds R copies of x and B in the input projection, increasing
    projection parameters by roughly (d_inner + ngroups * d_state) * (R - 1).
    The output normalization scale includes 1/sqrt(R) to keep output variance
    stable: scale = 1 / sqrt(d_state * mimo_rank).

    Typical values: 1 (SISO, default), 2-4 for improved state capacity.

chunk_size (int, default=256):
    Chunk size for the SSD (Structured State Space Duality) algorithm. The SSD
    algorithm splits the sequence into non-overlapping chunks of this size and
    computes:
    1. Within-chunk: quadratic attention-like scores (O(L^2) per chunk)
    2. Between-chunk: linear recurrent state propagation across chunks

    Larger chunk_size = more work done in the efficient quadratic path (better
    GPU utilization via large matmuls) but more memory for the L x L score
    matrix. Smaller chunk_size = less memory but more sequential state
    propagation steps. 256 is a good default for most sequence lengths.

================================================================================
Derived Dimensions
================================================================================

    d_inner   = expand * d_model      -- inner working dimension
    n_heads   = d_inner / d_state     -- number of SSM heads (if n_heads not given)
    head_dim  = d_inner / n_heads     -- per-head dimension
    bc_dim    = ngroups * d_state      -- total B or C projection width

    Input projection output size:
        d_inner                       -- z (output gate)
      + d_inner * R                   -- x (R copies for MIMO)
      + bc_dim * R                    -- B (R copies for MIMO)
      + bc_dim                        -- C (single rank)
      + 2 * n_heads                   -- dt + lambda (per-head scalars)
      + d_state // 2                  -- theta (RoPE frequencies)

    Recurrent state per layer: n_heads * head_dim * d_state floats

References:
- Mamba-3 (Trapezoidal SSD): https://openreview.net/forum?id=HwCvaJOiCj
- Mamba-2 (SSD): https://arxiv.org/abs/2405.21060
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton kernels for CUDA (optional — falls back to PyTorch on CPU/MPS)
_HAS_TRITON = False
try:
    from nanochat.ssd_triton import mamba3_chunk_scan_fwd, state_propagation_fwd, delta_h_fwd
    _HAS_TRITON = True
except ImportError:
    pass


def _mamba3_intra_pytorch(Bg, Bb, x_dt_g, x_dt_b, cum_log_dA, C, scale, R, causal_mask):
    """Pure PyTorch within-chunk computation (materializes L). Used as CPU fallback
    and for backward recomputation.

    Args:
        Bg:          (batch, nc, L, ngroups, R, dstate) float32
        Bb:          (batch, nc, L, ngroups, R, dstate) float32
        x_dt_g:      (batch, nc, L, nheads, R, headdim) float32
        x_dt_b:      (batch, nc, L, nheads, R, headdim) float32
        cum_log_dA:  (batch, nc, L, nheads) float32
        C:           (batch, nc, L, ngroups, dstate) float32
        scale:       float
        R:           int (mimo_rank)
        causal_mask: (L, L) bool upper-triangular mask
    Returns:
        y_intra: (batch, nc, L, nheads, headdim) float32
    """
    batch_size, n_chunks, L, ngroups, _, dstate = Bg.shape
    n_heads = cum_log_dA.shape[-1]
    headdim = x_dt_g.shape[-1]

    log_decay_matrix = cum_log_dA.unsqueeze(3) - cum_log_dA.unsqueeze(2)
    mask = causal_mask[:L, :L].reshape(1, 1, L, L, 1)
    log_decay_matrix = log_decay_matrix.masked_fill(mask, float('-inf'))
    decay_matrix = torch.exp(log_decay_matrix)
    del log_decay_matrix

    hpg = n_heads // ngroups

    # Batch CB scores across all R ranks in one GEMM (instead of R separate launches).
    # einsum handles the non-adjacent G dimension correctly (no manual reshape needed).
    # C: (B, nc, L_i, G, D), Bg: (B, nc, L_j, G, R, D) -> (B, nc, L_i, L_j, G, R)
    all_scores_g = torch.einsum('bcign, bcjgrn -> bcijgr', C, Bg) * scale
    all_scores_b = torch.einsum('bcign, bcjgrn -> bcijgr', C, Bb) * scale

    y_intra = torch.zeros(batch_size, n_chunks, L, n_heads, headdim,
                           device=Bg.device, dtype=torch.float32)

    for r in range(R):
        scores_g = all_scores_g[..., r]   # (B, nc, L, L, ngroups)
        scores_b = all_scores_b[..., r]

        if ngroups < n_heads:
            decay_grouped = decay_matrix.view(
                batch_size, n_chunks, L, L, ngroups, hpg)
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

    return y_intra



# ---- Custom ops for torch.compile compatibility (Triton path) ----
if _HAS_TRITON:
    @torch.library.custom_op("nanochat::mamba3_intra_fwd", mutates_args=())
    def _mamba3_intra_fwd_op(
        Bg: torch.Tensor, Bb: torch.Tensor,
        x_dt_g: torch.Tensor, x_dt_b: torch.Tensor,
        dA_cumsum: torch.Tensor, C: torch.Tensor,
        cos_ang: torch.Tensor, sin_ang: torch.Tensor,
        cos_angs: torch.Tensor, sin_angs: torch.Tensor,
        scale: float, chunk_size: int, mimo_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mamba3_chunk_scan_fwd(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                     cos_ang, sin_ang, cos_angs, sin_angs,
                                     scale, chunk_size, mimo_rank)

    @_mamba3_intra_fwd_op.register_fake
    def _mamba3_intra_fwd_fake(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                                cos_ang, sin_ang, cos_angs, sin_angs,
                                scale, chunk_size, mimo_rank):
        batch, nchunks, L, nheads, R, headdim = x_dt_g.shape
        ngroups = C.shape[3]
        dstate = C.shape[4]
        return (
            x_dt_g.new_empty(batch, nchunks, L, nheads, headdim, dtype=torch.bfloat16),
            Bg.new_empty(batch, nchunks, L, ngroups, R, dstate, dtype=torch.bfloat16),
            Bb.new_empty(batch, nchunks, L, ngroups, R, dstate, dtype=torch.bfloat16),
            C.new_empty(batch, nchunks, L, ngroups, dstate),
        )

    def _mamba3_intra_autograd_bwd(ctx, dy_intra, dBg_rot_ext, dBb_rot_ext, dC_rot_ext):
        """Backward for within-chunk + between-chunk gradient accumulation.

        The custom_op returns (y_intra, Bg_rot, Bb_rot, C_rot). Gradients arrive
        from both within-chunk (dy_intra) and between-chunk (dBg/Bb/C_rot_ext)
        uses of the rotated tensors. Both are inverse-rotated and summed.

        cuBLAS batched GEMMs outperform hand-written Triton tiling for these shapes.
        aot_autograd (from whole-model torch.compile) traces through this function,
        so these operations ARE compiled."""
        (Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
         cos_ang, sin_ang, cos_angs, sin_angs) = ctx.saved_tensors
        scale = ctx.scale
        R = ctx.mimo_rank
        L = ctx.chunk_size

        # Reconstruct rotated B/C from saved cos/sin (no trig recomputation)
        fused = cos_ang.numel() > 0
        if fused:
            cos_a = cos_ang.unsqueeze(3)    # (batch, nc, L, 1, half_d)
            sin_a = sin_ang.unsqueeze(3)
            cos_s = cos_angs.unsqueeze(3)
            sin_s = sin_angs.unsqueeze(3)
            C_rot = _apply_rope_pairs(C, cos_a, sin_a).float()
            Bg_rot = _apply_rope_pairs(Bg, cos_a.unsqueeze(4), sin_a.unsqueeze(4))
            Bb_rot = _apply_rope_pairs(Bb, cos_s.unsqueeze(4), sin_s.unsqueeze(4))
        else:
            C_rot = C
            Bg_rot = Bg
            Bb_rot = Bb

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=Bg.device), diagonal=1)
        with torch.enable_grad():
            Bg_ = Bg_rot.float().detach().requires_grad_()
            Bb_ = Bb_rot.float().detach().requires_grad_()
            x_dt_g_ = x_dt_g.float().detach().requires_grad_()
            x_dt_b_ = x_dt_b.float().detach().requires_grad_()
            dA_cumsum_ = dA_cumsum.detach().requires_grad_()
            C_ = C_rot.float().detach().requires_grad_()
            y_intra = _mamba3_intra_pytorch(Bg_, Bb_, x_dt_g_, x_dt_b_,
                                             dA_cumsum_, C_, scale, R, causal_mask)
            y_intra.backward(dy_intra)

        # Inverse-rotate gradients: dL/dBg_raw = R^{-1}(dL/dBg_rot)
        # Inverse rotation uses -sin (negate sin_a/sin_s).
        if fused:
            neg_sin_a = (-sin_a).unsqueeze(4)
            neg_sin_s = (-sin_s).unsqueeze(4)
            cos_a4 = cos_a.unsqueeze(4)
            cos_s4 = cos_s.unsqueeze(4)
            # Within-chunk gradients (from dy_intra)
            dBg = _apply_rope_pairs(Bg_.grad, cos_a4, neg_sin_a)
            dBb = _apply_rope_pairs(Bb_.grad, cos_s4, neg_sin_s)
            dC = _apply_rope_pairs(C_.grad, cos_a, -sin_a)
            # Between-chunk gradients (from rotated tensor outputs)
            if dBg_rot_ext is not None:
                dBg = dBg + _apply_rope_pairs(
                    dBg_rot_ext.float(), cos_a4, neg_sin_a)
                dBb = dBb + _apply_rope_pairs(
                    dBb_rot_ext.float(), cos_s4, neg_sin_s)
                dC = dC + _apply_rope_pairs(dC_rot_ext, cos_a, -sin_a)
        else:
            dBg = Bg_.grad
            dBb = Bb_.grad
            dC = C_.grad

        return (dBg, dBb, x_dt_g_.grad, x_dt_b_.grad,
                dA_cumsum_.grad, dC, None, None, None, None, None, None, None)

    def _mamba3_intra_setup_context(ctx, inputs, output):
        (Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
         cos_ang, sin_ang, cos_angs, sin_angs, scale, chunk_size, mimo_rank) = inputs
        ctx.save_for_backward(Bg, Bb, x_dt_g, x_dt_b, dA_cumsum, C,
                              cos_ang, sin_ang, cos_angs, sin_angs)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.mimo_rank = mimo_rank

    _mamba3_intra_fwd_op.register_autograd(
        _mamba3_intra_autograd_bwd, setup_context=_mamba3_intra_setup_context)



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
    """Mamba-3 (Trapezoidal SSD) layer with MIMO.

    Always uses MIMO tensor layout (x and B carry an R dimension).
    For SISO semantics, use Mamba-2 instead.

    See module docstring for full architecture description and parameter guide.

    Args:
        d_model:    Model dimension (input/output width).
        d_state:    SSM state dimension per head. Must be even (for RoPE).
        expand:     Expansion factor: d_inner = expand * d_model.
        n_heads:    Number of SSM heads. Default: d_inner // d_state.
        ngroups:    B/C group count (1=shared, n_heads=per-head, like GQA).
        chunk_size: SSD chunk size for quadratic/linear decomposition.
        mimo_rank:  MIMO rank (>=1, rank-R state updates).
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

        # Scale normalizes the C·B dot product (1/sqrt(d_state)) and the
        # MIMO rank summation (1/sqrt(R)) so total output variance is stable.
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
        x_raw = F.silu(x_raw)

        # 3. QK-normalization on B and C
        # B: (B, T, ngroups, R, d_state) — norm over d_state, bias broadcasts over R
        B_mat = F.rms_norm(B_mat, (self.d_state,)) * self.B_norm_bias.unsqueeze(-2)
        C_mat = F.rms_norm(C_mat, (self.d_state,)) * self.C_norm_bias

        # 4. Discretize dt
        A = -torch.exp(self.A_log.float())  # (n_heads,) always negative
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads)
        dt = dt.clamp(min=1e-4, max=0.5)

        # 5. Reshape x to multi-head with R dimension: (B, T, n_heads, R, head_dim)
        x_heads = x_raw.view(B, T, self.n_heads, R, self.head_dim)

        # 6. SSD computation
        if ssm_state is None:
            # Training: Two-SSD decomposition
            # Triton path: pass raw B/C + cum_angles, fuse RoPE inside kernel
            # CPU fallback: pre-rotate B/C in Python
            if _HAS_TRITON and B_mat.device.type == 'cuda':
                cum_angles = self._compute_rope_angles(dt, theta_raw)
                y_heads = self._two_ssd_forward(
                    x_heads, A, B_mat, C_mat, dt, lambda_raw,
                    cum_angles=cum_angles)
            else:
                B_mat, C_mat = self._apply_data_dependent_rope(
                    B_mat, C_mat, dt, theta_raw)
                y_heads = self._two_ssd_forward(
                    x_heads, A, B_mat, C_mat, dt, lambda_raw)
        else:
            # Inference: trapezoidal recurrence (always pre-rotate)
            B_mat, C_mat = self._apply_data_dependent_rope(
                B_mat, C_mat, dt, theta_raw)
            y_heads, ssm_state = self._trapezoidal_recurrent(
                x_heads, A, B_mat, C_mat, dt, lambda_raw, ssm_state)

        # 8. Output gate: norm then gate with z
        y = y_heads.reshape(B, T, self.d_inner)
        y = self._fused_norm_gate(y, z)

        # 9. Output projection
        y = self.out_proj(y)

        return y, ssm_state, None

    def _split_projection(self, proj):
        """Split the input projection into its components.

        Returns B with shape (B, T, ngroups, R, d_state) — always 5D.
        """
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

        B_mat = B_mat.view(*B_mat.shape[:-1], self.ngroups, R, self.d_state)
        C_mat = C_mat.view(*C_mat.shape[:-1], self.ngroups, self.d_state)

        return z, x_raw, B_mat, C_mat, dt_raw, lambda_raw, theta_raw

    def _compute_rope_angles(self, dt, theta_raw):
        """Compute cumulative RoPE angles (without applying rotation).

        Args:
            dt:        (batch, T, n_heads)
            theta_raw: (batch, T, d_state//2)
        Returns:
            cum_angles: (batch, T, d_state//2) float32
        """
        dt_mean = dt.mean(dim=-1, keepdim=True)  # (batch, T, 1)
        raw_angles = dt_mean * theta_raw
        # Negative cumsum: B at time s rotates by -sum(0..s), C at time t by -sum(0..t),
        # so C_t @ B_s encodes positional distance via angle difference sum(s..t).
        return -torch.cumsum(raw_angles.float(), dim=1)

    def _apply_data_dependent_rope(self, B, C, dt, theta_raw):
        """Apply data-dependent RoPE to B and C.

        Args:
            B:         (batch, T, ngroups, R, d_state)
            C:         (batch, T, ngroups, d_state)
            dt:        (batch, T, n_heads)
            theta_raw: (batch, T, d_state//2)
        Returns:
            B_rotated, C_rotated: same shapes as input
        """
        cum_angles = self._compute_rope_angles(dt, theta_raw)

        # cos/sin — compute in fp32, cast to input dtype to free fp32 copies
        input_dtype = B.dtype
        cos_a = torch.cos(cum_angles).to(input_dtype).unsqueeze(2)  # (batch, T, 1, d_state//2)
        sin_a = torch.sin(cum_angles).to(input_dtype).unsqueeze(2)
        del cum_angles

        C_rotated = _apply_rope_pairs(C, cos_a, sin_a)
        # Extra unsqueeze to broadcast over R dim in B: (batch, T, 1, 1, d_state//2)
        B_rotated = _apply_rope_pairs(B, cos_a.unsqueeze(3), sin_a.unsqueeze(3))

        return B_rotated, C_rotated

    def _two_ssd_forward(self, x_heads, A, B, C, dt, lambda_raw,
                         cum_angles=None):
        """Two-SSD decomposition for trapezoidal discretization (training).

        Decomposes the trapezoidal recurrence into two standard SSD calls:
        - gamma-SSD: current timestep contribution (weighted by lambda)
        - beta-SSD:  previous timestep contribution (weighted by (1-lambda)*alpha)

        Fused implementation: shared decay infrastructure + combined state
        propagation (one sequential loop instead of two).

        Args:
            x_heads:    (B, T, n_heads, R, head_dim)
            A:          (n_heads,)
            B:          (B, T, ngroups, R, d_state) — after QK-norm (raw if cum_angles given)
            C:          (B, T, ngroups, d_state) — after QK-norm (raw if cum_angles given)
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
            cum_angles: (B, T, d_state//2) float32, or None if B/C already rotated
        Returns:
            y: (B, T, n_heads, head_dim)
        """
        fused_rope = cum_angles is not None

        R = self.mimo_rank
        lam = torch.sigmoid(lambda_raw)       # (B, T, n_heads) in (0, 1)
        alpha = torch.exp(A * dt)             # (B, T, n_heads) decay
        coeff_beta = (1 - lam) * alpha

        # x is (B, T, nh, R, hd), need extra unsqueezes for broadcasting
        lam_x = lam.unsqueeze(-1).unsqueeze(-1)               # (B, T, nh, 1, 1)
        coeff_beta_x = coeff_beta.unsqueeze(-1).unsqueeze(-1)

        x_gamma = x_heads * lam_x
        x_beta = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0)) * coeff_beta_x
        B_shifted = F.pad(B[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        del lam, coeff_beta

        # Shifted cum_angles for Bb: position t-1 angles (Bb uses B from prev timestep)
        if fused_rope:
            cum_angles_shifted = F.pad(cum_angles[:, :-1], (0, 0, 1, 0), value=0.0)

        # Fused Two-SSD: shared decay + combined state propagation
        batch_size, T, n_heads, _, head_dim = x_gamma.shape
        d_state = C.shape[-1]
        ngroups = C.shape[-2]
        scale = self._ssd_scale
        chunk_size = min(self.chunk_size, T)

        # Pad T to multiple of chunk_size
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x_gamma = F.pad(x_gamma, (0, 0, 0, 0, 0, 0, 0, pad_len))
            x_beta  = F.pad(x_beta,  (0, 0, 0, 0, 0, 0, 0, pad_len))
            B       = F.pad(B,       (0, 0, 0, 0, 0, 0, 0, pad_len))
            B_shifted = F.pad(B_shifted, (0, 0, 0, 0, 0, 0, 0, pad_len))
            C       = F.pad(C,       (0, 0, 0, 0, 0, pad_len))
            dt      = F.pad(dt,      (0, 0, 0, pad_len))
            if fused_rope:
                cum_angles = F.pad(cum_angles, (0, 0, 0, pad_len))
                cum_angles_shifted = F.pad(cum_angles_shifted, (0, 0, 0, pad_len))
        T_padded = T + pad_len
        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks
        x_g = x_gamma.view(batch_size, n_chunks, L, n_heads, R, head_dim)
        x_b = x_beta.view(batch_size, n_chunks, L, n_heads, R, head_dim)
        Bg  = B.view(batch_size, n_chunks, L, ngroups, R, d_state)
        Bb  = B_shifted.view(batch_size, n_chunks, L, ngroups, R, d_state)
        C_  = C.view(batch_size, n_chunks, L, ngroups, d_state)
        dt_ = dt.view(batch_size, n_chunks, L, n_heads)

        if fused_rope:
            half_d = d_state // 2
            angles_chunked = cum_angles.view(batch_size, n_chunks, L, half_d)
            angles_shifted_chunked = cum_angles_shifted.view(
                batch_size, n_chunks, L, half_d)
            # Pre-compute cos/sin once for the within-chunk kernel.
            # Eliminates trig from Triton kernel.
            cos_a_f32 = torch.cos(angles_chunked)   # (batch, nc, L, half_d) float32
            sin_a_f32 = torch.sin(angles_chunked)
            cos_s_f32 = torch.cos(angles_shifted_chunked)
            sin_s_f32 = torch.sin(angles_shifted_chunked)

        # === SHARED DECAY (computed once instead of twice) ===
        log_dA = (A * dt_).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)

        # Pre-compute x*dt for all ranks (reused in within-chunk and between-chunk)
        dt_broad = dt_.unsqueeze(-1).unsqueeze(-1)  # (batch, nc, L, nh, 1, 1)
        x_dt_g = (x_g * dt_broad).bfloat16()
        x_dt_b = (x_b * dt_broad).bfloat16()

        # === WITHIN-CHUNK: Triton on CUDA (L never materialized), PyTorch fallback ===
        hpg = n_heads // ngroups
        C_f = C_.float()
        if _HAS_TRITON and x_gamma.device.type == 'cuda':
            assert fused_rope, "Triton path requires fused RoPE (cum_angles must be provided)"
            # Custom op returns pre-rotated B/C alongside y_intra — rotation
            # happens once (inside the scan kernel) and is shared with between-chunk.
            y_intra, Bg_rot, Bb_rot, C_rot = _mamba3_intra_fwd_op(
                Bg, Bb, x_dt_g, x_dt_b, cum_log_dA, C_f,
                cos_a_f32, sin_a_f32, cos_s_f32, sin_s_f32,
                scale, L, R)
            del cos_a_f32, sin_a_f32, cos_s_f32, sin_s_f32
        else:
            y_intra = _mamba3_intra_pytorch(
                Bg.float(), Bb.float(), x_dt_g.float(), x_dt_b.float(),
                cum_log_dA, C_f, scale, R, self._causal_mask)
            # B/C already rotated in non-fused path
            C_rot = C_
            Bg_rot = Bg
            Bb_rot = Bb

        chunk_decay = torch.exp(cum_log_dA[:, :, -1, :])

        # Between-chunk delta_h: outer-product reduction over (L, R).
        # Triton kernel computes decay_to_end in registers from cum_log_dA.
        if (_HAS_TRITON and x_gamma.device.type == 'cuda'
                and not torch.compiler.is_compiling()):
            delta_h = delta_h_fwd(x_dt_g, x_dt_b, Bg_rot, Bb_rot, cum_log_dA)
        else:
            decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)
            delta_h = torch.zeros(batch_size, n_chunks, n_heads, head_dim, d_state,
                                   device=x_gamma.device, dtype=torch.float32)
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
                                     Bg_rot[:, :, :, :, r, :].float())
                        + torch.einsum('bclgk, bclgkp, bclgn -> bcgkpn',
                                       decay_to_end_g, xb_r,
                                       Bb_rot[:, :, :, :, r, :].float())
                    ).reshape(batch_size, n_chunks, n_heads, head_dim, d_state)
            else:
                for r in range(R):
                    delta_h = delta_h + (
                        torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                     decay_to_end, x_dt_g[:, :, :, :, r, :],
                                     Bg_rot[:, :, :, :, r, :].float())
                        + torch.einsum('bclh, bclhp, bclhn -> bchpn',
                                       decay_to_end, x_dt_b[:, :, :, :, r, :],
                                       Bb_rot[:, :, :, :, r, :].float()))
        del x_dt_g, x_dt_b

        # Single state propagation pass (instead of two)
        # Use Triton kernel only outside torch.compile — the custom_op boundary
        # fragments compile's backward graph, causing worse fusion overall.
        # Under torch.compile, the Python loop is traced and fused natively.
        if (_HAS_TRITON and delta_h.device.type == 'cuda'
                and not torch.compiler.is_compiling()):
            all_states = state_propagation_fwd(delta_h, chunk_decay)
        else:
            states = delta_h.new_zeros(batch_size, n_heads, head_dim, d_state)
            all_states = []
            for c in range(n_chunks):
                all_states.append(states)
                states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
            all_states = torch.stack(all_states, dim=1)
        del delta_h

        # Inter-chunk output — group-aware einsum avoids repeat_interleave.
        # For ngroups=1: saves 268 MB (no C expansion) and uses 1 large matmul
        # instead of n_heads small ones.
        C_rot_f = C_rot.float() if fused_rope else C_f
        decay_from_start = torch.exp(cum_log_dA)
        if ngroups < n_heads:
            # all_states: (B, nc, nheads, headdim, dstate) -> (B, nc, ngroups, hpg, headdim, dstate)
            all_states_g = all_states.view(
                batch_size, n_chunks, ngroups, hpg, head_dim, d_state)
            state_contribution = torch.einsum(
                'bcign, bcgkpn -> bcigkp', C_rot_f, all_states_g)
            state_contribution = state_contribution.reshape(
                batch_size, n_chunks, L, n_heads, head_dim)
        else:
            state_contribution = torch.einsum(
                'bcihn, bchpn -> bcihp', C_rot_f, all_states)
        y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)

        y = (y_intra + y_inter).to(x_gamma.dtype)
        y = y.contiguous().view(batch_size, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T]
        return y

    def _trapezoidal_recurrent(self, x, A, B, C, dt, lambda_raw, ssm_state):
        """Trapezoidal recurrent computation for inference.

        State carries both h (SSM state) and prev_Bx (previous B*x product).
        The B*x product is a rank-R matmul.

        Args:
            x:          (B, T, n_heads, R, head_dim) — typically T=1
            A:          (n_heads,)
            B:          (B, T, ngroups, R, d_state) — after QK-norm and RoPE
            C:          (B, T, ngroups, d_state) — after QK-norm and RoPE
            dt:         (B, T, n_heads)
            lambda_raw: (B, T, n_heads)
            ssm_state:  dict with 'h' and 'prev_Bx'
        Returns:
            y: (B, T, n_heads, head_dim)
            new_state: updated dict
        """
        _, T, _, _, _ = x.shape

        # Expand B groups to heads
        B_exp = self._expand_groups(B, groups_dim=-3)  # (B, T, n_heads, R, d_state)
        C_exp = self._expand_groups(C)                  # (B, T, n_heads, d_state)

        lam = torch.sigmoid(lambda_raw)

        h = ssm_state['h']             # (B, n_heads, head_dim, d_state)
        prev_Bx = ssm_state['prev_Bx'] # (B, n_heads, head_dim, d_state)

        outputs = []
        for t in range(T):
            C_t = C_exp[:, t]    # (B, n_heads, d_state)
            dt_t = dt[:, t]      # (B, n_heads)
            lam_t = lam[:, t]    # (B, n_heads)

            alpha_t = torch.exp(A * dt_t)            # (B, n_heads)
            beta_t = (1 - lam_t) * dt_t * alpha_t
            gamma_t = lam_t * dt_t

            # Rank-R matmul: sum of R outer products
            x_t = x[:, t]        # (B, n_heads, R, head_dim)
            B_t = B_exp[:, t]    # (B, n_heads, R, d_state)
            Bx_t = torch.einsum('bhri, bhrj -> bhij', x_t, B_t)

            # Trapezoidal state update
            h = (alpha_t[:, :, None, None] * h
                 + beta_t[:, :, None, None] * prev_Bx
                 + gamma_t[:, :, None, None] * Bx_t)

            prev_Bx = Bx_t

            # Output (C is single-rank). Cast C_t to match h dtype — h is promoted
            # to float32 by alpha_t (from A.float()) when the model runs in bf16.
            y_t = torch.einsum('bhn, bhpn -> bhp', C_t.to(h.dtype), h) * self._ssd_scale
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, {'h': h, 'prev_Bx': prev_Bx}

    def _fused_norm_gate(self, y_raw, z):
        """RMSNorm and SiLU gate (fused by whole-model torch.compile)."""
        y_norm = F.rms_norm(y_raw, (y_raw.size(-1),))
        return y_norm * F.silu(z)
