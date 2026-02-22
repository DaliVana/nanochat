# Sparsity Variants Comparison

Generated: 2026-02-20 01:13:43

## Summary

| Variant | Training Time (s) | Val BPB | Tokens/s | MFU (%) | Peak Memory (MiB) |
|---------|------------------|---------|----------|---------|-------------------|
| Baseline | 119.2 | 2.346375 | N/A | N/A | 0.0 |
| MoE (4x2) | 169.5 | 3.210729 | N/A | N/A | 0.0 |
| MoE (8x2) | 209.5 | 3.210729 | N/A | N/A | 0.0 |
| MoE (8x1) | 1334.7 | 3.210729 | N/A | N/A | 0.0 |

## Detailed Results

### Baseline

```json
{
  "variant": "Baseline",
  "training_time": 119.17030882835388,
  "success": true,
  "sparsity_args": [],
  "val_bpb": 2.346375,
  "peak_memory_mib": 0.0
}
```

### MoE (4x2)

```json
{
  "variant": "MoE (4x2)",
  "training_time": 169.4977672100067,
  "success": true,
  "sparsity_args": [
    "--num-experts=4",
    "--top-k=2",
    "--num-shared-experts=1"
  ],
  "val_bpb": 3.210729,
  "peak_memory_mib": 0.0
}
```

### MoE (8x2)

```json
{
  "variant": "MoE (8x2)",
  "training_time": 209.49753212928772,
  "success": true,
  "sparsity_args": [
    "--num-experts=8",
    "--top-k=2",
    "--num-shared-experts=1"
  ],
  "val_bpb": 3.210729,
  "peak_memory_mib": 0.0
}
```

### MoE (8x1)

```json
{
  "variant": "MoE (8x1)",
  "training_time": 1334.7314438819885,
  "success": true,
  "sparsity_args": [
    "--num-experts=8",
    "--top-k=1",
    "--num-shared-experts=0"
  ],
  "val_bpb": 3.210729,
  "peak_memory_mib": 0.0
}
```
