# Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models

**Paper**: arXiv:2404.02258
**Authors**: David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro (Google DeepMind)
**Date**: April 2024

## Summary

This paper introduces **Mixture-of-Depths (MoD)**, a technique for dynamically allocating compute in transformers by allowing tokens to either pass through a transformer block (self-attention + MLP) or bypass it via a residual connection. Unlike traditional transformers that process all tokens uniformly, MoD uses learned routing to decide which tokens need computation and which can skip.

### Key Insight

Not all tokens require equal processing. By learning which tokens to route through blocks vs. around them, MoD can:
1. Match baseline performance with **fewer FLOPs per forward pass**
2. Achieve **better performance** than baselines at equivalent training FLOPs
3. Be **up to 50% faster** during inference sampling

## How It Works

### Routing Mechanism

1. **Per-block router**: A linear projection produces a scalar weight for each token: `r_i = w_θᵀ x_i`

2. **Top-k selection**: The top-k tokens (by router weight) participate in the block's computations; the rest bypass via residual connection

3. **Output formula**:
   ```
   x_i^{l+1} = r_i * f(X̃) + x_i   if token is in top-k
   x_i^{l+1} = x_i                  if token is not in top-k
   ```
   Where `f` is self-attention + MLP, and `X̃` is the set of top-k tokens.

4. **Router weight multiplication**: The router weight `r_i` multiplies the block output, putting it in the gradient path for learning.

### Expert-Choice Routing (vs Token-Choice)

The paper uses **expert-choice routing** where each block selects its top-k tokens, rather than tokens choosing which block to use. Benefits:
- No auxiliary balancing loss needed
- Router weights can express relative importance (critical tokens get higher weights to ensure selection)
- With two paths (compute vs. bypass), top-k cleanly splits tokens into mutually exclusive sets

### Key Hyperparameters

- **Capacity**: 12.5% of tokens participate in computation (87.5% bypass) - surprisingly aggressive!
- **Routing frequency**: Every other block has routing (alternating with full-capacity blocks)
- **Static compute budget**: Total FLOPs are known ahead of time, only token identities are dynamic

## Results

### IsoFLOP Analysis

At equivalent training FLOPs:
- Optimal MoD transformers achieve **~1.5% better** training loss than vanilla baselines
- MoD models are larger (more parameters) but use fewer FLOPs per parameter
- There exist MoD variants that match baseline loss while being **60%+ faster** to step

### Why It Works

1. **Learned routing is crucial**: Random/stochastic routing performs much worse
2. **FLOPs-per-forward-pass determines optimal size**: The best MoD uses same FLOPs/forward as the optimal baseline
3. **Better to add depth than width**: When adding capacity, more layers beats wider layers
4. **Attention savings compound**: Q×K is O(T²), so routing 12.5% of tokens makes attention 25× cheaper

### Auto-regressive Inference Challenge

Top-k is **non-causal** (depends on future tokens), which breaks auto-regressive sampling. Solutions:
1. **Auxiliary loss**: Binary cross-entropy to predict if token will be in top-k (centers sigmoid at 0.5)
2. **Predictor MLP**: Small auxiliary network predicts top-k membership from token embedding

Both achieve **>97% accuracy** quickly during training, with minimal performance degradation at inference.

## MoDE: Combining with MoE

MoD integrates naturally with Mixture-of-Experts (MoE):
- **Staged MoDE**: MoD routing before self-attention, then MoE for MLPs
- **Integrated MoDE**: Include "no-op" as one of the experts

MoD gains compound with MoE gains!

## Key Observations from Routing Analysis

- Some tokens engage every block; others route around whenever possible
- Tokens that engage more frequently correlate with **higher entropy predictions** (harder predictions)
- Vertical bands in routing patterns suggest certain positions (e.g., end of sequence) consistently need more compute

## Discussion & Future Directions

1. **Decoupled Q/K/V routing**: Currently all are routed together; could have tokens be keys but not queries, or vice versa
2. **Long-term memory**: Tokens could decide at encoding time whether to be available as keys for future attention (cheaper than full content-based lookup)
3. **Heterogeneous compute types**: Route to different computation types (memory lookup, tool use) not just compute vs. no-op
4. **Memory savings**: Smaller KV cache during inference since fewer tokens participate

---

## Relevance to Nanochat

### Direct Implementation Opportunity

Nanochat's `Block` class in `nanochat/gpt.py` is a good candidate for MoD:

```python
# Current nanochat Block (nanochat/gpt.py:291-303)
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

An MoD version would add a router and selectively apply the block:

```python
class MoDBlock(nn.Module):
    def __init__(self, config, layer_idx, use_routing=True, capacity_ratio=0.125):
        super().__init__()
        self.use_routing = use_routing
        self.capacity_ratio = capacity_ratio
        if config.use_mla:
            self.attn = MLACausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        if use_routing:
            self.router = nn.Linear(config.n_embd, 1, bias=False)
            # Optional: auxiliary predictor for inference
            self.aux_predictor = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if not self.use_routing:
            # Standard full-capacity block
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
            return x, None

        B, T, C = x.size()
        k = int(T * self.capacity_ratio)

        # Compute router weights
        router_weights = self.router(x).squeeze(-1)  # (B, T)

        # Top-k selection per batch element
        _, top_indices = torch.topk(router_weights, k, dim=-1)

        # Gather tokens for processing
        # ... (implementation details for efficient gathering/scattering)

        # Process only selected tokens
        # ...

        # Auxiliary loss for inference routing
        aux_loss = F.binary_cross_entropy_with_logits(
            router_weights,
            (router_weights > router_weights.median(dim=-1, keepdim=True).values).float()
        )

        return x, aux_loss
```

### Specific Nanochat Synergies

1. **Sliding Window Attention**: Nanochat already uses `window_pattern` (SSSL) for sliding windows. MoD could compound: fewer tokens × smaller windows = massive attention savings.

2. **MLA Integration**: The paper suggests MoD works with MoE; nanochat's MLA could similarly benefit. Fewer tokens going through the KV compression path.

3. **Value Embeddings**: Nanochat uses per-layer value embeddings (`value_embeds`). MoD routing could help determine which tokens most benefit from value embedding contribution.

4. **resid_lambdas/x0_lambdas**: Nanochat already has learnable per-layer scalars. Router weights are similar in spirit - per-token, per-layer modulation.

### Implementation Considerations for Nanochat

1. **Every-other routing**: Paper found routing every other block works best. Could align with nanochat's alternating value embedding pattern (`has_ve`).

2. **Capacity 12.5%**: Aggressive but effective. For 2048 sequence length, only 256 tokens would pass through routed blocks.

3. **Training vs Inference**:
   - Training: top-k selection (non-causal is fine)
   - Inference: predictor-based routing (needs aux loss during training)

4. **KV Cache**: MoD complicates the KV cache since fewer tokens produce K/V. Need to track which positions have cached values.

5. **Static graph**: Unlike some conditional compute methods, MoD maintains static tensor shapes (always process k tokens), making it friendly to compilation/optimization.

### Estimated Impact

Based on paper results:
- **Training**: Could match baseline loss ~60% faster, or achieve 1-2% better loss at same compute
- **Inference**: Up to 50% faster per-token generation
- **Memory**: Smaller effective KV cache (only k tokens per routed layer)

### Experiment Suggestions

1. **Baseline comparison**: Add MoD routing to d20 speedrun model, compare against vanilla at same FLOPs
2. **Capacity sweep**: Test 12.5%, 25%, 50% capacity ratios
3. **Routing pattern**: Test every-other vs all-layer routing
4. **Combine with existing**: See if MoD compounds with sliding window and MLA savings
5. **Inference predictor**: Validate that auxiliary predictor reaches >95% accuracy

### Potential Challenges

1. **Implementation complexity**: Gathering/scattering tokens efficiently on GPU
2. **KV cache management**: Tracking which tokens participated in each layer
3. **Training stability**: Router learning dynamics, ensuring convergence
4. **Interaction with DDP**: Routing decisions may vary across ranks; need careful handling
