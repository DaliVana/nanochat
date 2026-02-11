"""
Unified Flash Attention interface with automatic FA4/FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, with:
- FA4 (flash_attn.cute) on Blackwell GPUs (sm100) — training forward+backward
- FA3 (varunneal/flash-attention-3) on Hopper GPUs (sm90) — full support
- PyTorch SDPA fallback on all other hardware (Ada, Ampere, CPU, MPS)

Note: FA4 does not expose flash_attn_with_kvcache, so inference with KV cache
on Blackwell falls back to SDPA until the FA4 API adds KV cache support.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA4 (Blackwell sm100) or FA3 (Hopper sm90)
# =============================================================================
def _load_flash_attention_4():
    """Try to load Flash Attention 4 (requires Blackwell GPU, sm100)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            return None
        from flash_attn.cute.interface import flash_attn_func
        return flash_attn_func
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Blackwell GPU detected but FA4 failed to load: {e}")
        return None


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa4_func = _load_flash_attention_4()
_fa3 = _load_flash_attention_3()
HAS_FA4 = _fa4_func is not None
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa4', 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _active_impl():
    """Determine which implementation to use: 'fa4', 'fa3', or 'sdpa'."""
    if _override_impl == 'fa4':
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return 'fa4'
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'sdpa':
        return 'sdpa'
    # auto
    if HAS_FA4:
        return 'fa4'
    if HAS_FA3:
        return 'fa3'
    return 'sdpa'


# Keep _use_fa3() for backward compatibility with tests
def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    return _active_impl() == 'fa3'


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# FA4 helpers
# =============================================================================
def _fa4_window_size(window_size):
    """Convert FA3-style window_size to FA4-style.

    FA3 uses (-1, -1) for unlimited. FA4 uses (None, None).
    """
    left, right = window_size
    return (None if left < 0 else left, None if right < 0 else right)


# =============================================================================
# FA4 call wrapper: exclude only the CuTe-DSL kernel from torch.compile.
# FA4 uses its own JIT compilation (cute.compile) which is incompatible with
# Dynamo tracing. We wrap just the FA4 call, not the entire attention function,
# so Inductor can still fuse the surrounding ops (transposes, projections, etc.)
# =============================================================================
@torch.compiler.disable
def _fa4_call(q, k, v, causal, window_size):
    """Call FA4 outside of torch.compile tracing."""
    return _fa4_func(q, k, v, causal=causal,
                     window_size=_fa4_window_size(window_size))


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    impl = _active_impl()

    if impl == 'fa4':
        return _fa4_call(q, k, v, causal, window_size)

    if impl == 'fa3':
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.
    FA4 does not expose a KV cache API, so Blackwell falls back to SDPA here.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    impl = _active_impl()

    if impl == 'fa3':
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # FA4 and SDPA: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
