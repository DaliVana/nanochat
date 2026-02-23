"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

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
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_sliding_window_chunked(q, k, v, window, enable_gqa):
    """
    Efficient O(T*W) sliding window attention for SDPA using overlapping chunks.
    Avoids materializing a full T×T attention mask.

    q: (B, H_q, T, D), k: (B, H_kv, T, D), v: (B, H_kv, T, D)
    window: number of tokens to the left each token can attend to (window_size[0])
    """
    B, H_q, T, D = q.shape
    H_kv = k.size(1)
    W = window  # chunk size = window size

    # Pad T to next multiple of W
    pad_len = (W - T % W) % W
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
    T_pad = T + pad_len
    n_chunks = T_pad // W

    y = q.new_empty(B, H_q, T_pad, D)

    # Chunk 0: standard causal (no previous context to attend to)
    y[:, :, :W] = F.scaled_dot_product_attention(
        q[:, :, :W], k[:, :, :W], v[:, :, :W],
        is_causal=True, enable_gqa=enable_gqa
    )

    if n_chunks > 1:
        nc = n_chunks - 1
        # Queries: split remaining sequence into nc chunks of W
        q_rest = (q[:, :, W:]
                  .reshape(B, H_q, nc, W, D)
                  .transpose(1, 2)
                  .reshape(B * nc, H_q, W, D))

        # Keys/values: overlapping windows of 2W with stride W via unfold
        # chunk i (1-indexed) gets k[(i-1)*W : (i+1)*W]
        k_rest = (k.unfold(2, 2 * W, W)          # (B, H_kv, nc, D, 2W)
                  .permute(0, 2, 1, 4, 3)         # (B, nc, H_kv, 2W, D)
                  .reshape(B * nc, H_kv, 2 * W, D))
        v_rest = (v.unfold(2, 2 * W, W)
                  .permute(0, 2, 1, 4, 3)
                  .reshape(B * nc, H_kv, 2 * W, D))

        # Sliding window mask: (W, 2W), same for all chunks
        # q at local pos i, k at local pos j (offset by W from q's chunk start)
        # Causal: j <= i + W, Window: j >= i
        qi = torch.arange(W, device=q.device).unsqueeze(1)
        kj = torch.arange(2 * W, device=q.device).unsqueeze(0)
        mask = (kj >= qi) & (kj <= qi + W)

        y_rest = F.scaled_dot_product_attention(
            q_rest, k_rest, v_rest,
            attn_mask=mask, enable_gqa=enable_gqa
        )
        y[:, :, W:] = (y_rest
                       .reshape(B, nc, H_q, W, D)
                       .permute(0, 2, 1, 3, 4)
                       .reshape(B, H_q, nc * W, D))

    if pad_len > 0:
        y = y[:, :, :T]
    return y


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

    # Training with sliding window: use efficient chunked approach O(T*W)
    if Tq == Tk and window >= 0 and window < Tk:
        return _sdpa_sliding_window_chunked(q, k, v, window, enable_gqa)

    # Fallback: explicit mask for chunk inference (Tq != Tk)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

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
    if _use_fa3():
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
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
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
