"""
Test Multi-Head Latent Attention (MLA) implementation.

Run: python -m pytest tests/test_mla.py -v -s
"""
import torch
import pytest
from nanochat.gpt import GPTConfig, GPT, MLACausalSelfAttention, CausalSelfAttention, get_mla_dims


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestMLAConfig:
    """Test MLA configuration and dimension calculations."""

    def test_default_mla_dims(self):
        """Test auto-scaling of MLA dimensions."""
        config = GPTConfig(n_embd=768, n_head=6, use_mla=True)
        d_c, d_c1, d_rope = get_mla_dims(config)

        # d_c should be n_embd // 3
        assert d_c == 256, f"Expected d_c=256, got {d_c}"
        # d_c1 should equal d_c by default
        assert d_c1 == d_c, f"Expected d_c1={d_c}, got {d_c1}"
        # d_rope should be head_dim // 2 by default (768/6=128, 128/2=64)
        head_dim = config.n_embd // config.n_head
        assert d_rope == head_dim // 2, f"Expected d_rope={head_dim // 2}, got {d_rope}"

    def test_custom_mla_dims(self):
        """Test custom MLA dimensions."""
        config = GPTConfig(n_embd=768, use_mla=True, mla_d_c=128, mla_d_c1=96, mla_d_rope=32)
        d_c, d_c1, d_rope = get_mla_dims(config)

        assert d_c == 128
        assert d_c1 == 96
        assert d_rope == 32

    def test_mla_dims_scaling(self):
        """Test that MLA dimensions scale with model size."""
        for n_embd in [384, 768, 1536, 2048]:
            n_head = n_embd // 128  # keeps head_dim = 128
            config = GPTConfig(n_embd=n_embd, n_head=n_head, use_mla=True)
            d_c, d_c1, d_rope = get_mla_dims(config)
            assert d_c == n_embd // 3, f"d_c should be {n_embd // 3} for n_embd={n_embd}"
            head_dim = n_embd // n_head
            assert d_rope == head_dim // 2, f"d_rope should be {head_dim // 2} for head_dim={head_dim}"


class TestMLAAttention:
    """Test MLA attention module."""

    def test_mla_output_shape(self):
        """Test MLA produces correct output shape."""
        config = GPTConfig(n_embd=384, n_head=6, n_layer=4, use_mla=True)
        mla = MLACausalSelfAttention(config, layer_idx=0).to(DEVICE)

        B, T, C = 2, 32, config.n_embd
        x = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE)

        # Create dummy cos/sin for RoPE (d_rope dimensions)
        _, _, d_rope = get_mla_dims(config)
        cos = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)
        sin = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)

        y = mla(x, ve=None, cos_sin=(cos, sin), window_size=(T, 0), kv_cache=None)

        assert y.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_mla_with_value_embedding(self):
        """Test MLA with value embedding."""
        config = GPTConfig(n_embd=384, n_head=6, n_layer=4, use_mla=True)
        # Layer 3 should have VE (alternating, last layer included)
        mla = MLACausalSelfAttention(config, layer_idx=3).to(DEVICE)

        B, T, C = 2, 32, config.n_embd
        x = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE)

        # Value embedding has shape (B, T, n_head * head_dim) for MLA
        head_dim = config.n_embd // config.n_head
        ve = torch.randn(B, T, config.n_head * head_dim, device=DEVICE, dtype=DTYPE)

        _, _, d_rope = get_mla_dims(config)
        cos = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)
        sin = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)

        y = mla(x, ve=ve, cos_sin=(cos, sin), window_size=(T, 0), kv_cache=None)

        assert y.shape == (B, T, C)
        assert not torch.isnan(y).any()

    def test_mla_gradients_flow(self):
        """Test gradients flow through MLA."""
        config = GPTConfig(n_embd=384, n_head=6, n_layer=4, use_mla=True)
        mla = MLACausalSelfAttention(config, layer_idx=0).to(DEVICE)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE, requires_grad=True)

        _, _, d_rope = get_mla_dims(config)
        cos = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)
        sin = torch.randn(1, T, 1, d_rope // 2, device=DEVICE, dtype=DTYPE)

        y = mla(x, ve=None, cos_sin=(cos, sin), window_size=(T, 0), kv_cache=None)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        # Check key parameters have gradients
        assert mla.w_dkv.weight.grad is not None, "No gradient for w_dkv"
        assert mla.w_uk.weight.grad is not None, "No gradient for w_uk"
        assert mla.w_uv.weight.grad is not None, "No gradient for w_uv"


class TestMLAModel:
    """Test full GPT model with MLA."""

    def test_mla_model_creation(self):
        """Test GPT model with MLA can be created."""
        config = GPTConfig(
            n_embd=384, n_head=6, n_layer=4, n_kv_head=6,
            vocab_size=1024, sequence_len=128,
            use_mla=True
        )

        with torch.device('meta'):
            model = GPT(config)

        # Verify attention modules are MLA
        for i, block in enumerate(model.transformer.h):
            assert isinstance(block.attn, MLACausalSelfAttention), \
                f"Block {i} should use MLACausalSelfAttention"

    def test_standard_model_creation(self):
        """Test GPT model without MLA uses standard attention."""
        config = GPTConfig(
            n_embd=384, n_head=6, n_layer=4, n_kv_head=6,
            vocab_size=1024, sequence_len=128,
            use_mla=False
        )

        with torch.device('meta'):
            model = GPT(config)

        # Verify attention modules are standard
        for i, block in enumerate(model.transformer.h):
            assert isinstance(block.attn, CausalSelfAttention), \
                f"Block {i} should use CausalSelfAttention"

    @pytest.mark.slow
    def test_mla_model_forward(self):
        """Test MLA model forward pass."""
        config = GPTConfig(
            n_embd=384, n_head=6, n_layer=4, n_kv_head=6,
            vocab_size=1024, sequence_len=128,
            use_mla=True
        )

        model = GPT(config).to(DEVICE)
        model.init_weights()

        B, T = 2, 32
        idx = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
        targets = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)

        loss = model(idx, targets=targets)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

    @pytest.mark.slow
    def test_mla_model_backward(self):
        """Test MLA model backward pass."""
        config = GPTConfig(
            n_embd=384, n_head=6, n_layer=4, n_kv_head=6,
            vocab_size=1024, sequence_len=128,
            use_mla=True
        )

        model = GPT(config).to(DEVICE)
        model.init_weights()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
        targets = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)

        loss = model(idx, targets=targets)
        loss.backward()

        # Check some parameters have gradients
        assert model.transformer.h[0].attn.w_dkv.weight.grad is not None
        assert model.lm_head.weight.grad is not None


class TestMLAVsStandard:
    """Compare MLA and standard attention properties."""

    def test_mla_has_fewer_kv_params_per_token(self):
        """Test that MLA compresses KV representation."""
        config_std = GPTConfig(n_embd=768, n_head=6, n_kv_head=6, n_layer=1, use_mla=False)
        config_mla = GPTConfig(n_embd=768, n_head=6, n_kv_head=6, n_layer=1, use_mla=True)

        # Standard: K and V each have n_kv_head * head_dim dimensions
        head_dim = config_std.n_embd // config_std.n_head
        std_kv_dim = 2 * config_std.n_kv_head * head_dim  # K + V

        # MLA: compressed to d_c + d_rope
        d_c, _, d_rope = get_mla_dims(config_mla)
        mla_cache_dim = d_c + d_rope  # for inference cache (compressed)

        print(f"Standard KV dim per token: {std_kv_dim}")
        print(f"MLA cache dim per token: {mla_cache_dim}")
        print(f"Compression ratio: {std_kv_dim / mla_cache_dim:.2f}x")

        # MLA should use less storage (though we use compatibility mode for inference)
        assert mla_cache_dim < std_kv_dim, "MLA should have smaller cache footprint"


class TestMLARoPE:
    """Test decoupled RoPE in MLA."""

    def test_rope_dimension_split(self):
        """Test that head_dim is correctly split between nope and rope."""
        config = GPTConfig(n_embd=768, n_head=6, use_mla=True)
        mla = MLACausalSelfAttention(config, layer_idx=0)

        head_dim = config.n_embd // config.n_head  # 128
        _, _, d_rope = get_mla_dims(config)  # auto-scaled
        d_nope = head_dim - d_rope

        assert mla.d_rope == d_rope
        assert mla.d_nope == d_nope
        assert mla.d_rope + mla.d_nope == head_dim

    def test_rope_applied_only_to_rope_part(self):
        """Test that RoPE is only applied to the d_rope dimensions."""
        config = GPTConfig(n_embd=384, n_head=6, use_mla=True)
        mla = MLACausalSelfAttention(config, layer_idx=0).to(DEVICE)

        _, _, d_rope = get_mla_dims(config)

        # The w_kr projection outputs d_rope dimensions for the key RoPE part
        assert mla.w_kr.weight.shape == (d_rope, config.n_embd)

        # The w_uk projection outputs n_head * d_nope (non-positional part)
        d_nope = (config.n_embd // config.n_head) - d_rope
        assert mla.w_uk.weight.shape == (config.n_head * d_nope, mla.d_c)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Test device: {DEVICE}, dtype: {DTYPE}")
    print()

    pytest.main([__file__, "-v", "-s"])
