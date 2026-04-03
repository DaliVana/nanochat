"""
Mamba-3 layer using the official mamba-ssm package (state-spaces/mamba).

Drop-in replacement for nanochat's Mamba3Layer that delegates to
mamba_ssm.modules.mamba3.Mamba3 for training. Uses the official's optimized
TileLang (MIMO) and Triton (SISO) kernels instead of nanochat's custom Triton.

Install: uv sync --extra official-mamba

Limitations:
- CUDA-only (official kernels have no CPU/MPS fallback)
- Inference (ssm_state != None) not supported — different state formats
- Weights not transferable to/from nanochat backend (different in_proj layout,
  data-dependent A vs fixed A_log)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _get_mamba3_cls():
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3
    except ImportError:
        raise ImportError(
            "mamba_ssm package is required for the official Mamba-3 backend. "
            "Install with: uv sync --extra official-mamba"
        )


class Mamba3LayerOfficial(nn.Module):
    """Mamba-3 layer wrapping the official mamba_ssm.Mamba3.

    Accepts the same constructor signature as nanochat's Mamba3Layer and
    returns the same 3-tuple from forward(). Internally delegates to the
    official implementation for its optimized kernels.
    """

    # The official SISO kernel materializes a [chunk_size, chunk_size] QK score
    # matrix in SRAM. At chunk_size=128 the working set (~307 KB) exceeds H100
    # per-SM SRAM (228 KB), forcing a slow two-phase global-memory roundtrip.
    # Cap at 64 (the official's own default) to keep everything in SRAM (~160 KB).
    _OFFICIAL_MAX_CHUNK_SIZE = 64

    def __init__(self, d_model, d_state=64, expand=2, n_heads=None, ngroups=1,
                 chunk_size=256, mimo_rank=1, **official_kwargs):
        super().__init__()

        # Derive dimensions matching nanochat's logic
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.n_heads = n_heads if n_heads is not None else self.d_inner // d_state
        self.head_dim = self.d_inner // self.n_heads
        self.ngroups = ngroups
        self.mimo_rank = mimo_rank
        self._ssd_scale = (d_state * mimo_rank) ** -0.5

        # Cap chunk_size to avoid SRAM overflow in the official kernel
        official_chunk_size = min(chunk_size, self._OFFICIAL_MAX_CHUNK_SIZE)
        self.chunk_size = official_chunk_size

        assert d_state % 2 == 0, "d_state must be even for RoPE"
        assert mimo_rank >= 1, "mimo_rank must be >= 1"
        assert self.d_inner % self.n_heads == 0, "d_inner must be divisible by n_heads"
        assert self.n_heads % ngroups == 0, "n_heads must be divisible by ngroups"

        # Build the official Mamba3 with translated args
        Mamba3 = _get_mamba3_cls()
        self._inner = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=self.head_dim,
            ngroups=ngroups,
            rope_fraction=1.0,  # nanochat always uses full d_state for RoPE
            is_mimo=(mimo_rank > 1),
            mimo_rank=mimo_rank,
            chunk_size=official_chunk_size,
            **official_kwargs,
        )

    # ------------------------------------------------------------------
    # Property aliases so gpt_samba.py init_weights / setup_optimizer work
    # ------------------------------------------------------------------

    @property
    def in_proj(self):
        return self._inner.in_proj

    @property
    def out_proj(self):
        return self._inner.out_proj

    @property
    def dt_bias(self):
        return self._inner.dt_bias

    @property
    def B_norm_bias(self):
        """Official B_bias shape is (nheads, mimo_rank, d_state).
        Nanochat's is (ngroups, d_state). In-place ops like .fill_() work on either."""
        return self._inner.B_bias

    @property
    def C_norm_bias(self):
        """Official C_bias shape is (nheads, mimo_rank, d_state).
        Nanochat's is (ngroups, d_state). In-place ops like .fill_() work on either."""
        return self._inner.C_bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def forward(self, x, ssm_state=None):
        """
        Args:
            x: (B, T, d_model) input tensor
            ssm_state: must be None (inference not supported via this wrapper)
        Returns:
            y: (B, T, d_model) output tensor
            None: ssm_state placeholder
            None: conv_state placeholder
        """
        if ssm_state is not None:
            raise NotImplementedError(
                "Inference with ssm_state is not supported via the official Mamba-3 "
                "backend wrapper. The official Mamba3 uses a different state format "
                "(4 separate tensors vs nanochat's dict). Use mamba_ssm.Mamba3.step() "
                "directly for inference."
            )
        # The official kernel saves ~27 large intermediate tensors per layer for
        # backward (SSM_States, Q_rot, K_scaled, Out_v, ...) — ~500 MB/layer in
        # bf16. Gradient checkpointing frees them after forward and recomputes
        # one layer at a time during backward, matching nanochat's memory profile.
        y = grad_checkpoint(self._inner, x, use_reentrant=True)
        return y, None, None
