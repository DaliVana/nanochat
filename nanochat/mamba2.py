"""
Mamba-2 (SSD) layer for nanochat.

Pure PyTorch implementation of the Structured State Space Duality (SSD) algorithm.
No external mamba-ssm dependency.

Key ideas:
- The selective state space model with diagonal A can be computed equivalently as
  a form of linear attention (the "duality" in SSD).
- For training, we use a chunk-wise algorithm: quadratic (attention-like) within
  chunks, recurrent between chunks. This gives O(T * N * D) complexity.
- For inference, we use the standard sequential recurrence.

Memory optimization (following official Mamba-2 implementation):
- ngroups: B/C projections are shared across heads (like GQA), reducing parameters
  and activation memory. Default ngroups=1 matches the official implementation.
- For long sequences that OOM, use model-level gradient checkpointing to limit
  peak memory to one block's SSD intermediates at a time.

References:
- Mamba-2: https://arxiv.org/abs/2405.21060
- Samba: https://arxiv.org/abs/2406.07522
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Mamba2Layer(nn.Module):
    """
    Mamba-2 (SSD) layer.

    Architecture:
    1. Input projection: x -> (z, x_conv, B, C, dt)
    2. Short causal 1D depthwise convolution on x_conv, then SiLU
    3. SSD scan: selective state space (chunked for training, recurrent for inference)
    4. Output gate: y = rms_norm(y) * silu(z)
    5. Output projection: linear back to model dimension
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, n_heads=None, ngroups=1, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk_size = chunk_size

        # Inner dimension (expanded)
        self.d_inner = int(expand * d_model)
        # Number of heads: d_inner must be divisible by n_heads
        # head_dim = d_inner // n_heads; in Mamba-2, head_dim = d_state
        self.n_heads = n_heads if n_heads is not None else self.d_inner // d_state
        assert self.d_inner % self.n_heads == 0, f"d_inner ({self.d_inner}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_inner // self.n_heads

        # B/C group count (like GQA: ngroups=1 means all heads share one B/C,
        # ngroups=n_heads means per-head B/C like the original implementation)
        self.ngroups = ngroups
        assert self.n_heads % self.ngroups == 0, f"n_heads ({self.n_heads}) must be divisible by ngroups ({self.ngroups})"

        # Input projection: single big linear for efficiency
        # z(d_inner) + x_conv(d_inner) + B(ngroups*d_state) + C(ngroups*d_state) + dt(n_heads)
        d_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(d_model, d_proj, bias=False)

        # Short causal 1D depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # causal: we'll truncate the future
            bias=False,
        )

        # Learnable log(A) parameter (per-head, always negative after exp)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))

        # dt bias (per-head, added to dt before softplus)
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._ssd_scale = d_state ** -0.5
        self.register_buffer(
            '_causal_mask',
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _expand_groups(self, x):
        """Expand grouped B/C from (*, ngroups, d_state) to (*, n_heads, d_state).
        No-op when ngroups == n_heads."""
        if self.ngroups == self.n_heads:
            return x
        return x.repeat_interleave(self.n_heads // self.ngroups, dim=-2)

    def forward(self, x, ssm_state=None, conv_state=None):
        """
        Args:
            x: (B, T, d_model) input tensor
            ssm_state: None for training, (B, n_heads, head_dim, d_state) for inference
            conv_state: None for training, (B, d_inner, d_conv-1) for inference
        Returns:
            y: (B, T, d_model) output tensor
            new_ssm_state: updated SSM state (None during training)
            new_conv_state: updated conv state (None during training)
        """
        B, T, _ = x.shape

        # 1. Input projection
        proj = self.in_proj(x)  # (B, T, d_proj)

        # Split into components
        z, x_conv, B_mat, C_mat, dt_raw = self._split_projection(proj)

        # 2. Short causal convolution
        if conv_state is None:
            # Training: full convolution
            x_conv = x_conv.transpose(1, 2)  # (B, d_inner, T)
            x_conv = self.conv1d(x_conv)[:, :, :T]  # causal: truncate future
            x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        else:
            # Inference: step-by-step convolution
            x_conv, conv_state = self._conv_step(x_conv, conv_state)
        x_conv = F.silu(x_conv)

        # 3. Compute discretized parameters
        A = -torch.exp(self.A_log.float())  # (n_heads,) always negative
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads) always positive
        dt = dt.clamp(min=1e-4, max=0.5)

        # Reshape x_conv to multi-head: (B, T, n_heads, head_dim)
        x_heads = x_conv.view(B, T, self.n_heads, self.head_dim)

        # 4. SSD computation
        if ssm_state is None:
            # Training: vectorized chunked SSD (all chunks in parallel)
            y_heads = self._ssd_chunked(x_heads, A, B_mat, C_mat, dt)
        else:
            # Inference: sequential recurrent scan
            y_heads, ssm_state = self._ssd_recurrent(x_heads, A, B_mat, C_mat, dt, ssm_state)

        # 5. Output gate: norm then gate with z
        y = y_heads.reshape(B, T, self.d_inner)
        y = F.rms_norm(y, (y.size(-1),))
        y = y * F.silu(z)

        # 6. Output projection
        y = self.out_proj(y)

        return y, ssm_state, conv_state

    def _split_projection(self, proj):
        """Split the input projection into its components."""
        d_inner = self.d_inner
        ngroups = self.ngroups
        d_state = self.d_state

        z = proj[..., :d_inner]                                                         # (B, T, d_inner)
        x_conv = proj[..., d_inner:2*d_inner]                                           # (B, T, d_inner)
        B_mat = proj[..., 2*d_inner:2*d_inner + ngroups*d_state]                        # (B, T, ngroups*d_state)
        C_mat = proj[..., 2*d_inner + ngroups*d_state:2*d_inner + 2*ngroups*d_state]    # (B, T, ngroups*d_state)
        dt_raw = proj[..., 2*d_inner + 2*ngroups*d_state:]                              # (B, T, n_heads)

        B_mat = B_mat.view(*B_mat.shape[:-1], ngroups, d_state)  # (B, T, ngroups, d_state)
        C_mat = C_mat.view(*C_mat.shape[:-1], ngroups, d_state)  # (B, T, ngroups, d_state)

        return z, x_conv, B_mat, C_mat, dt_raw

    def _conv_step(self, x_conv, conv_state):
        """
        Single-step convolution for inference.
        Args:
            x_conv: (B, T, d_inner) - typically T=1 during generation
            conv_state: (B, d_inner, d_conv-1) - buffer of recent inputs
        Returns:
            y: (B, T, d_inner)
            new_conv_state: (B, d_inner, d_conv-1)
        """
        B, T, d_inner = x_conv.shape
        outputs = []
        for t in range(T):
            x_t = x_conv[:, t, :]  # (B, d_inner)
            # Shift and insert
            conv_state = torch.cat([conv_state[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)
            # Apply conv: weight is (d_inner, 1, d_conv)
            y_t = (conv_state * self.conv1d.weight.squeeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        return y, conv_state

    def _ssd_chunked(self, x, A, B, C, dt):
        """
        Chunked SSD computation for training (CPU/MPS path).
        Materializes all chunks at once — faster but uses more memory.

        Args:
            x:  (B, T, n_heads, head_dim) - input after conv+silu
            A:  (n_heads,) - diagonal state decay (negative)
            B:  (B, T, ngroups, d_state) - input-to-state matrix
            C:  (B, T, ngroups, d_state) - state-to-output matrix
            dt: (B, T, n_heads) - discretization timestep
        Returns:
            y:  (B, T, n_heads, head_dim) - output
        """
        B_batch, T, n_heads, head_dim = x.shape
        ngroups = B.shape[-2]
        d_state = B.shape[-1]
        chunk_size = min(self.chunk_size, T)

        # Pad T to multiple of chunk_size if needed
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T

        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks: (B, n_chunks, L, ...)
        x  = x.view(B_batch, n_chunks, L, n_heads, head_dim)
        B_ = B.view(B_batch, n_chunks, L, ngroups, d_state)
        C_ = C.view(B_batch, n_chunks, L, ngroups, d_state)
        dt = dt.view(B_batch, n_chunks, L, n_heads)

        # Decay computation (float32 for cumsum/exp stability)
        log_dA = (A * dt).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)  # (B, nc, L, nh)

        # --- Within-chunk computation (quadratic / attention-like) ---
        # Pairwise decay matrix with causal mask applied in log-space
        log_decay_matrix = cum_log_dA.unsqueeze(3) - cum_log_dA.unsqueeze(2)  # (B, nc, L, L, nh)
        mask = self._causal_mask[:L, :L].reshape(1, 1, L, L, 1)
        log_decay_matrix = log_decay_matrix.masked_fill(mask, float('-inf'))
        decay_matrix = torch.exp(log_decay_matrix)

        # Scores: CB in grouped space, then expand to per-head and multiply by decay
        scale = self._ssd_scale
        scores = torch.einsum('bcign, bcjgn -> bcijg', C_.float(), B_.float())  # (B, nc, L, L, ngroups)
        scores = scores * scale
        if ngroups < n_heads:
            scores = scores.repeat_interleave(n_heads // ngroups, dim=-1)  # -> (B, nc, L, L, nh)
        scores = scores * decay_matrix

        # Weighted sum
        x_dt = (x * dt.unsqueeze(-1)).float()
        y_intra = torch.einsum('bcijh, bcjhp -> bcihp', scores, x_dt)

        # --- Between-chunk computation (recurrent state propagation) ---
        chunk_decay = torch.exp(cum_log_dA[:, :, -1, :])  # (B, nc, nh)
        decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :] - cum_log_dA)  # (B, nc, L, nh)
        B_expanded = self._expand_groups(B_)  # (B, nc, L, n_heads, d_state)
        delta_h = torch.einsum('bclh, bclhp, bclhn -> bchpn', decay_to_end, x_dt.float(), B_expanded.float())

        # Propagate states across chunks sequentially
        states = torch.zeros(B_batch, n_heads, head_dim, d_state, device=x.device, dtype=torch.float32)
        all_states = []
        for c in range(n_chunks):
            all_states.append(states)
            states = chunk_decay[:, c, :, None, None] * states + delta_h[:, c]
        all_states = torch.stack(all_states, dim=1)  # (B, nc, nh, hd, d_state)

        # Contribution from previous state to current chunk's output
        C_expanded = self._expand_groups(C_)  # (B, nc, L, n_heads, d_state)
        decay_from_start = torch.exp(cum_log_dA)
        state_contribution = torch.einsum('bcihn, bchpn -> bcihp', C_expanded.float(), all_states)
        y_inter = state_contribution * (decay_from_start.unsqueeze(-1) * scale)

        # Combine
        y = (y_intra + y_inter).to(x.dtype)
        y = y.contiguous().view(B_batch, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T, :, :]

        return y

    def _ssd_chunk_compute(self, x_c, B_c, C_c, dt_c, states, A, cum_log_dA_c):
        """
        Compute intra-chunk output and inter-chunk state update for a single chunk.
        Designed to be wrapped in torch.utils.checkpoint for memory efficiency.

        Args:
            x_c:          (B, L, n_heads, head_dim)
            B_c:          (B, L, ngroups, d_state)
            C_c:          (B, L, ngroups, d_state)
            dt_c:         (B, L, n_heads)
            states:       (B, n_heads, head_dim, d_state) - incoming state
            A:            (n_heads,)
            cum_log_dA_c: (B, L, n_heads)
        Returns:
            y_c:        (B, L, n_heads, head_dim)
            new_states: (B, n_heads, head_dim, d_state)
        """
        B_batch, L, n_heads, head_dim = x_c.shape
        scale = self._ssd_scale

        # --- Intra-chunk (quadratic) ---
        log_decay = cum_log_dA_c.unsqueeze(2) - cum_log_dA_c.unsqueeze(1)  # (B, L, L, nh)
        mask = self._causal_mask[:L, :L].unsqueeze(0).unsqueeze(-1)  # (1, L, L, 1)
        log_decay = log_decay.masked_fill(mask, float('-inf'))
        decay = torch.exp(log_decay)  # (B, L, L, nh)

        # CB scores in grouped space, then expand to per-head
        scores = torch.einsum('bign, bjgn -> bijg', C_c.float(), B_c.float())  # (B, L, L, ngroups)
        scores = scores * scale
        if self.ngroups < n_heads:
            scores = scores.repeat_interleave(n_heads // self.ngroups, dim=-1)
        scores = scores * decay

        x_dt = (x_c * dt_c.unsqueeze(-1)).float()
        y_intra = torch.einsum('bijh, bjhp -> bihp', scores, x_dt)

        # --- Inter-chunk contribution from incoming state ---
        decay_from_start = torch.exp(cum_log_dA_c)  # (B, L, nh)
        C_expanded = self._expand_groups(C_c)  # (B, L, n_heads, d_state)
        state_contrib = torch.einsum('bihn, bhpn -> bihp', C_expanded.float(), states)
        y_inter = state_contrib * (decay_from_start.unsqueeze(-1) * scale)

        y_c = y_intra + y_inter

        # --- Update state for next chunk ---
        chunk_decay = torch.exp(cum_log_dA_c[:, -1, :])  # (B, nh)
        decay_to_end = torch.exp(cum_log_dA_c[:, -1:, :] - cum_log_dA_c)  # (B, L, nh)
        B_expanded = self._expand_groups(B_c)  # (B, L, n_heads, d_state)
        delta_h = torch.einsum('blh, blhp, blhn -> bhpn', decay_to_end, x_dt.float(), B_expanded.float())
        new_states = chunk_decay[:, :, None, None] * states + delta_h

        return y_c, new_states

    def _ssd_chunked_sequential(self, x, A, B, C, dt):
        """
        Memory-efficient chunked SSD for CUDA training.
        Processes one chunk at a time with per-chunk gradient checkpointing.
        Peak quadratic memory: O(B * L^2 * nh) instead of O(B * n_chunks * L^2 * nh).

        Args:
            x:  (B, T, n_heads, head_dim)
            A:  (n_heads,)
            B:  (B, T, ngroups, d_state)
            C:  (B, T, ngroups, d_state)
            dt: (B, T, n_heads)
        Returns:
            y:  (B, T, n_heads, head_dim)
        """
        B_batch, T, n_heads, head_dim = x.shape
        d_state = B.shape[-1]
        chunk_size = min(self.chunk_size, T)

        # Pad T to multiple of chunk_size if needed
        pad_len = 0
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T

        n_chunks = T_padded // chunk_size
        L = chunk_size

        # Reshape into chunks: (B, n_chunks, L, ...)
        x  = x.view(B_batch, n_chunks, L, n_heads, head_dim)
        B_ = B.view(B_batch, n_chunks, L, self.ngroups, d_state)
        C_ = C.view(B_batch, n_chunks, L, self.ngroups, d_state)
        dt = dt.view(B_batch, n_chunks, L, n_heads)

        # Precompute cumulative log(dA) for all chunks (small: B, nc, L, nh)
        log_dA = (A * dt).float()
        cum_log_dA = torch.cumsum(log_dA, dim=2)  # (B, nc, L, nh)

        # Process chunks sequentially with per-chunk checkpointing
        states = torch.zeros(B_batch, n_heads, head_dim, d_state,
                             device=x.device, dtype=torch.float32)
        y_chunks = []

        for c in range(n_chunks):
            y_c, states = checkpoint(
                self._ssd_chunk_compute,
                x[:, c], B_[:, c], C_[:, c], dt[:, c],
                states, A, cum_log_dA[:, c],
                use_reentrant=False,
            )
            y_chunks.append(y_c.to(x.dtype))

        y = torch.stack(y_chunks, dim=1)  # (B, nc, L, nh, hd)
        y = y.contiguous().view(B_batch, T_padded, n_heads, head_dim)
        if pad_len > 0:
            y = y[:, :T, :, :]

        return y

    def _ssd_recurrent(self, x, A, B, C, dt, ssm_state):
        """
        Recurrent SSD computation for inference (one or few tokens at a time).

        Args:
            x:  (B, T, n_heads, head_dim) where T is typically 1 during generation
            A:  (n_heads,)
            B:  (B, T, ngroups, d_state)
            C:  (B, T, ngroups, d_state)
            dt: (B, T, n_heads)
            ssm_state: (B, n_heads, head_dim, d_state)
        Returns:
            y: (B, T, n_heads, head_dim)
            new_state: (B, n_heads, head_dim, d_state)
        """
        B_batch, T, n_heads, head_dim = x.shape

        # Expand B/C from ngroups to n_heads
        B = self._expand_groups(B)  # (B, T, n_heads, d_state)
        C = self._expand_groups(C)  # (B, T, n_heads, d_state)

        outputs = []
        h = ssm_state  # (B, n_heads, head_dim, d_state)

        for t in range(T):
            x_t = x[:, t]       # (B, n_heads, head_dim)
            B_t = B[:, t]       # (B, n_heads, d_state)
            C_t = C[:, t]       # (B, n_heads, d_state)
            dt_t = dt[:, t]     # (B, n_heads)

            # Discretize: dA_t = exp(A * dt_t)
            dA_t = torch.exp(A * dt_t)  # (B, n_heads)

            # Update state: h = dA * h + outer(x * dt, B)
            h = dA_t.unsqueeze(-1).unsqueeze(-1) * h + torch.einsum(
                'bhi, bhj -> bhij',
                x_t * dt_t.unsqueeze(-1),
                B_t,
            )

            # Output: y = C @ h, scaled by 1/sqrt(d_state) like attention
            y_t = torch.einsum('bhn, bhpn -> bhp', C_t, h) * (self.d_state ** -0.5)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, n_heads, head_dim)
        return y, h
