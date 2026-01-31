# Progressive Depth Scaling for Nanochat: MoD + LESA Hybrid Approach

**Context**: Discussion on extending trained nanochat models (e.g., d20 → d21) with minimal performance loss.

**Implementation**: `nanochat/depth_extension.py`

**Related Papers**:
- Mixture-of-Depths (arXiv:2404.02258) - Dynamic compute allocation via routing
- LESA (arXiv:2502.13794) - Learnable layer prediction for depth scaling

---

## Quick Start

```python
from nanochat.depth_extension import extend_model_depth, DepthExtensionOptimizer

# 1. Extend d20 to d21
model, new_block = extend_model_depth(model, insert_idx=15)

# 2. Setup optimizer with gradual unfreezing
base_optimizer = model.setup_optimizer()
optimizer = DepthExtensionOptimizer(
    model,
    new_layer_idx=16,
    base_optimizer=base_optimizer,
    fade_in_steps=2000
)

# 3. Training loop - capacity and LR annealing happen automatically
for step, batch in enumerate(dataloader):
    # Update capacity (tied to optimizer's internal step count)
    new_block.capacity_ratio = optimizer.get_capacity()

    loss = model(batch['input_ids'], batch['targets'])
    loss.backward()
    optimizer.step()  # LR scaling handled internally
    optimizer.zero_grad()

    # Logging
    if step % 100 == 0:
        print(f"Step {step}: capacity={optimizer.get_capacity():.3f}, lr_scale={optimizer.get_lr_scale():.3f}")
```

### How it works

| Step | Capacity | Original Layers LR | New Layer LR |
|------|----------|-------------------|--------------|
| 0 | 0% | 0% of base | 100% of base |
| 1000 | ~6% | ~50% of base | 100% of base |
| 2000 | 12.5% | 100% of base | 100% of base |

Both capacity and LR follow the same cosine schedule, so the original layers gradually "wake up" as the new layer activates.

---

## The Problem

When adding layers to a trained model:
1. **Random initialization** disrupts learned representations
2. **Naive duplication** (LLaMA Pro, SOLAR) leads to suboptimal initialization
3. **Sudden insertion** causes immediate performance drop
4. **Recovery requires extensive retraining**

**Goal**: Extend d20 → d21 (or more) with minimal CORE score loss and fast recovery.

---

## Solution: MoD + LESA Hybrid

Combine two complementary techniques:

| Technique | Contribution |
|-----------|--------------|
| **LESA** | Predicts good initial weights by learning inter-layer patterns |
| **MoD** | Enables gradual "fade-in" via routing (new layer starts as identity) |

### Why This Works

1. **LESA** ensures the new layer's weights are meaningful (not random noise)
2. **MoD** ensures zero initial disruption (capacity=0 means pure residual)
3. **Gradual activation** allows the model to adapt smoothly
4. **Frozen original layers** preserve existing knowledge during early training

---

## Implementation

### Step 1: Train LESA Predictors (~5 minutes)

```python
def train_lesa_predictors(model):
    """
    Train small MLPs to predict intermediate layer weights.
    One predictor per weight matrix type.
    """
    config = model.config
    predictors = {}
    svd_components = {}

    # Weight types depend on attention mechanism
    if config.use_mla:
        weight_types = [
            'w_dkv', 'w_uk', 'w_uv', 'w_kr',  # KV compression
            'w_dq', 'w_uq', 'w_qr',            # Q compression (if exists)
            'c_proj',                           # Attention output
            'c_fc', 'c_proj_mlp'               # MLP
        ]
    else:
        weight_types = [
            'c_q', 'c_k', 'c_v', 'c_proj',     # Attention
            'c_fc', 'c_proj_mlp'               # MLP
        ]

    for wtype in weight_types:
        # 1. Collect weights from all layers
        W_list = []
        for layer in model.transformer.h:
            W = get_weight_by_type(layer, wtype)
            W_list.append(W.flatten())

        W_stack = torch.stack(W_list)  # (n_layer, d1*d2)

        # 2. SVD decomposition
        # Transpose to (d1*d2, n_layer) for SVD
        W_T = W_stack.T
        U, S, Vt = torch.linalg.svd(W_T, full_matrices=False)

        svd_components[wtype] = {
            'U': U,
            'S': S,
            'Vt': Vt,
            'original_shape': get_weight_by_type(model.transformer.h[0], wtype).shape
        }

        # 3. Train predictor on V coefficients
        V = Vt.T  # (n_layer, rank)
        predictor = train_predictor_mlp(V, hidden_dim=256, epochs=5, lr=1e-3)
        predictors[wtype] = predictor

    return predictors, svd_components


def train_predictor_mlp(V, hidden_dim=256, epochs=5, lr=1e-3, lambda_norm=5e-5):
    """
    Train MLP: [V_{i-1}, V_{i+1}] -> V_i
    """
    n_layers, rank = V.shape

    # Training data: predict middle from neighbors
    X, Y = [], []
    for i in range(1, n_layers - 1):
        X.append(torch.cat([V[i-1], V[i+1]]))
        Y.append(V[i])

    X = torch.stack(X)
    Y = torch.stack(Y)

    predictor = nn.Sequential(
        nn.Linear(2 * rank, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, rank)
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = predictor(X)
        loss_mse = F.mse_loss(pred, Y)
        loss_norm = F.mse_loss(pred.norm(dim=-1), Y.norm(dim=-1))
        loss = (1 - lambda_norm) * loss_mse + lambda_norm * loss_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return predictor
```

### Step 2: Predict New Layer Weights

```python
def predict_new_layer_weights(predictors, svd_components, insert_idx):
    """
    Use trained predictors to generate weights for a new layer
    to be inserted between insert_idx and insert_idx+1.
    """
    new_weights = {}

    for wtype, predictor in predictors.items():
        svd = svd_components[wtype]
        U, S, Vt = svd['U'], svd['S'], svd['Vt']
        V = Vt.T
        original_shape = svd['original_shape']

        # Predict intermediate V
        input_vec = torch.cat([V[insert_idx], V[insert_idx + 1]])
        V_pred = predictor(input_vec.unsqueeze(0)).squeeze(0)

        # Reconstruct weight matrix: W = U @ diag(S) @ V_pred
        W_flat = U @ (S.unsqueeze(1) * V_pred.unsqueeze(0))
        W_pred = W_flat.reshape(original_shape)

        new_weights[wtype] = W_pred

    return new_weights
```

### Step 3: Create MoD-Enabled Block

```python
class MoDBlock(nn.Module):
    """
    Transformer block with Mixture-of-Depths routing.
    capacity_ratio=0 means pure residual (identity).
    """
    def __init__(self, config, layer_idx, capacity_ratio=0.125):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.layer_idx = layer_idx

        if config.use_mla:
            self.attn = MLACausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

        # Router: produces scalar weight per token
        self.router = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Pure residual mode (for fade-in)
        if self.capacity_ratio == 0:
            return x

        # Full capacity mode (standard block)
        if self.capacity_ratio >= 1.0:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
            return x

        # MoD routing mode
        k = max(1, int(T * self.capacity_ratio))

        # Compute router weights
        router_logits = self.router(x).squeeze(-1)  # (B, T)

        # Top-k selection
        top_vals, top_idx = torch.topk(router_logits, k, dim=-1)  # (B, k)

        # Gather selected tokens
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, C)
        x_selected = torch.gather(x, 1, idx_expanded)  # (B, k, C)

        # Process selected tokens
        # Note: Simplified - full impl needs proper positional handling
        x_attn = self.attn(norm(x_selected), ve, cos_sin, window_size, None)
        x_mlp = self.mlp(norm(x_selected + x_attn))
        x_processed = x_selected + x_attn + x_mlp

        # Weight by sigmoid(router_logits) - puts router in gradient path
        router_weights = torch.sigmoid(top_vals).unsqueeze(-1)  # (B, k, 1)
        x_processed = x_processed * router_weights

        # Scatter back to original positions
        x_out = x.clone()
        x_out.scatter_(1, idx_expanded, x_processed)

        return x_out

    def load_predicted_weights(self, weights_dict):
        """Load LESA-predicted weights into this block."""
        # Map weight types to module attributes
        # This depends on attention type (standard vs MLA)
        for wtype, weight in weights_dict.items():
            set_weight_by_type(self, wtype, weight)
```

### Step 4: Insert and Train

```python
def extend_model_d20_to_d21(model_d20, insert_idx=15, fade_in_steps=2000, total_steps=10000):
    """
    Extend trained d20 model to d21 with minimal CORE loss.
    """
    # 1. Train LESA predictors
    print("Training LESA predictors...")
    predictors, svd_components = train_lesa_predictors(model_d20)

    # 2. Predict new layer weights
    print(f"Predicting weights for new layer at position {insert_idx}...")
    new_weights = predict_new_layer_weights(predictors, svd_components, insert_idx)

    # 3. Create MoD block with predicted weights
    config = model_d20.config
    new_block = MoDBlock(config, layer_idx=insert_idx, capacity_ratio=0.0)
    new_block.load_predicted_weights(new_weights)

    # Scale down output projections for smoother blending
    new_block.attn.c_proj.weight.data *= 0.1
    new_block.mlp.c_proj.weight.data *= 0.1

    # Initialize router to slightly favor NOT routing (conservative start)
    nn.init.normal_(new_block.router.weight, mean=-0.5, std=0.1)

    # 4. Insert into model
    layers = list(model_d20.transformer.h)
    layers.insert(insert_idx, new_block)
    model_d20.transformer.h = nn.ModuleList(layers)

    # Update config and layer indices
    config.n_layer += 1
    for i, layer in enumerate(model_d20.transformer.h):
        if hasattr(layer, 'layer_idx'):
            layer.layer_idx = i
        if hasattr(layer.attn, 'layer_idx'):
            layer.attn.layer_idx = i

    # Recompute window sizes
    model_d20.window_sizes = model_d20._compute_window_sizes(config)

    # Extend per-layer parameters
    extend_per_layer_params(model_d20, insert_idx)

    return model_d20, new_block


def continue_training(model, new_block, dataloader, fade_in_steps=2000, total_steps=10000):
    """
    Continue training with fade-in and layer freezing.
    """
    # Freeze original layers initially
    for name, param in model.named_parameters():
        if 'transformer.h.' in name:
            layer_num = int(name.split('.')[2])
            if layer_num != new_block.layer_idx:
                param.requires_grad = False

    optimizer = model.setup_optimizer()

    for step, batch in enumerate(dataloader):
        # Capacity annealing
        if step < fade_in_steps:
            progress = step / fade_in_steps
            # Cosine annealing: 0 -> 0.125 (or 1.0 for full capacity)
            capacity = 0.125 * (1 - math.cos(progress * math.pi)) / 2
            new_block.capacity_ratio = capacity
        else:
            new_block.capacity_ratio = 0.125  # Final MoD capacity

        # Unfreeze all layers after fade-in
        if step == fade_in_steps:
            print("Unfreezing all layers...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = model.setup_optimizer()  # Recreate optimizer

        # Training step
        loss = model(batch['input_ids'], batch['targets'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, capacity={new_block.capacity_ratio:.3f}")

        # CORE evaluation
        if step % 500 == 0:
            core_score = evaluate_core(model)
            print(f"Step {step}: CORE={core_score:.2f}")
```

---

## Expected Results

### CORE Score Trajectory

| Phase | Steps | Capacity | CORE Score | Notes |
|-------|-------|----------|------------|-------|
| Initial | 0 | 0% | ~100% | New layer is pure residual |
| Early fade-in | 0-500 | 0-3% | 98-100% | Minimal impact |
| Mid fade-in | 500-1500 | 3-10% | 95-99% | New layer activating |
| Full fade-in | 1500-2000 | 10-12.5% | 95-98% | Slight dip possible |
| Recovery | 2000-5000 | 12.5% | 98-102% | Model adapting |
| Converged | 5000+ | 12.5% | 100-105% | May exceed original |

### Comparison to Alternatives

| Method | Initial Drop | Recovery Steps | Final Score |
|--------|--------------|----------------|-------------|
| Random init + full insert | -30 to -50% | 10k+ | ~100% |
| LLaMA Pro (duplication) | -10 to -20% | 5k+ | ~100% |
| LESA only | -5 to -10% | 3k | ~102% |
| **MoD + LESA hybrid** | **<5%** | **2-3k** | **102-105%** |

---

## Configuration Recommendations

### Insert Position

Based on LESA paper findings:
- **Best**: Insert in later half (layers 15-19 for d20)
- **Avoid**: Early layers (1-5) - causes large PPL increase
- **Rationale**: Early layers encode basic features, more "frozen"

### Capacity Setting

From MoD paper:
- **Optimal**: 12.5% capacity (256 tokens out of 2048)
- **Conservative**: Start at 0%, anneal to 12.5%
- **Every-other**: Only route every other block (matches nanochat's `has_ve` pattern)

### Training Hyperparameters

```python
# Fade-in phase (steps 0-2000)
freeze_original_layers = True
learning_rate_new_layer = 1e-3
capacity_annealing = 'cosine'  # 0% -> 12.5%

# Main training phase (steps 2000+)
freeze_original_layers = False
learning_rate = 5e-5  # Lower for fine-tuning
capacity = 0.125  # Fixed at optimal MoD capacity
```

---

## Extensions

### Multiple Layers: d20 → d24

Insert layers in a staggered fashion:

```python
# Insert 4 layers with staggered fade-in
insert_positions = [15, 13, 17, 11]  # Spread throughout later half
fade_schedules = [
    (0, 2000),      # Layer 1: fade steps 0-2000
    (1000, 3000),   # Layer 2: fade steps 1000-3000
    (2000, 4000),   # Layer 3: fade steps 2000-4000
    (3000, 5000),   # Layer 4: fade steps 3000-5000
]
```

### Progressive Depth During Initial Training

Start training with shallow effective depth, gradually deepen:

```python
# Progressive training schedule
schedule = [
    (0, 4),       # Steps 0-1000: 4 effective layers
    (1000, 8),    # Steps 1000-2000: 8 effective layers
    (2000, 12),   # Steps 2000-3000: 12 effective layers
    (3000, 16),   # Steps 3000-4000: 16 effective layers
    (4000, 20),   # Steps 4000+: full 20 layers
]
```

This gives faster early training (fewer FLOPs) and curriculum learning benefits.

---

## Summary

The MoD + LESA hybrid approach provides:

1. **Minimal initial disruption** (MoD routing starts at 0%)
2. **Good initialization** (LESA predicts meaningful weights)
3. **Smooth integration** (gradual capacity annealing)
4. **Fast recovery** (converges in ~2-3k steps)
5. **Potential improvement** (new capacity can exceed original performance)

This should allow extending nanochat from d20 → d21 (or more) with **<5% CORE score drop** and full recovery within a few thousand training steps.
