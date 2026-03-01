# SSD Backward Kernel — Working Notes

## The Problem
Forward: custom Triton kernel, L never materialized (registers only).
Backward: PyTorch fallback, materializes full L matrix, ~60-80 kernel launches, ~200-400MB temp memory.

## Forward Recap
```
y[m,h,p] = scale * Σ_r Σ_{k≤m} (C[m,g]·Bg[k,g,r]) * L[m,k,h] * x_dt_g[k,h,r,p]
                                + (C[m,g]·Bb[k,g,r]) * L[m,k,h] * x_dt_b[k,h,r,p]
```
where L[m,k,h] = exp(min(dA_cs[m]-dA_cs[k], 0)) * (m>=k)

Intermediate: `scores_g[m,k,h,r] = (C[m,g]·Bg[k,g,r]) * L[m,k,h] * scale`
Then: `y[m,h,p] = Σ_r Σ_k scores_g * x_dt_g + scores_b * x_dt_b`

## Gradients (given dy[m,h,p])

### 1. dx_dt_g[k,h,r,p] = Σ_{m≥k} scores_g[m,k,h,r] * dy[m,h,p]
- Transposed forward: sum over m instead of k
- Kernel 2 (k-direction grid, inner loop over m≥k)

### 2. dx_dt_b — same structure with scores_b

### 3. dscores_g[m,k,h,r] = Σ_p dy[m,h,p] * x_dt_g[k,h,r,p]
- Intermediate, not an output. Dot product over headdim.

### 4. dC[m,g,d] = Σ_{k≤m} Σ_r Σ_{h∈g} (dscores_g*L*scale) * Bg[k,g,r,d] + (dscores_b*L*scale) * Bb[k,g,r,d]
- Kernel 1 (m-direction grid, inner loop over k≤m)
- The Σ_{h∈g} reduction is over heads in the group

### 5. dBg[k,g,r,d] = Σ_{m≥k} Σ_{h∈g} (dscores_g*L*scale) * C[m,g,d]
- Computed in Kernel 2 via atomic_add (multiple heads write to same group)

### 6. dBb — same structure

### 7. ddA_cumsum[i,h]
- Most complex gradient. L depends on dA_cs[m] and dA_cs[k].
- ddA_cs[i,h] = Σ_{k≤i} dL[i,k,h]*L_act - Σ_{m≥i} dL[m,i,h]*L_act
  where dL = scale * Σ_r (dscores_g*cb_g + dscores_b*cb_b)
  and L_act = L * 1{dA_cs[m]-dA_cs[k] < 0}
- m-side computed in Kernel 1, k-side in Kernel 2, summed afterward

## Architecture Decision: 2 Kernels

**Kernel 1 (dC + ddA_m):** Grid over (m-blocks × dstate-blocks, batch*chunks, groups)
- Holds m fixed, sweeps k ≤ m
- Natural for dC[m,g,d] which accumulates over k

**Kernel 2 (dx_dt + dBg/dBb + ddA_k):** Grid over (k-blocks × headdim-blocks, batch*chunks, heads)
- Holds k fixed, sweeps m ≥ k
- Natural for dx_dt[k,h,r,p] which accumulates over m
- dBg via atomic_add across heads in group

Both recompute L and cb products tile-by-tile (same as forward).

## Block Sizes (more conservative than forward)
- BLOCK_M=64, BLOCK_K=32 preferred; fallback to 32×32 or 32×16
- BLOCK_DSTATE=min(64, next_pow2(dstate))
- num_stages=1 (SRAM tight with backward's extra accumulators)

## Key Complexity: dscores computation
Both kernels need `dscores_g[m,k,h,r] = Σ_p dy[m,h,p] * x_dt_g[k,h,r,p]`
This is a reduction over headdim. Need to tile if headdim > BLOCK_HDIM.
In Kernel 1 (m-grid): we need dscores for each (m,k,h) — expensive because we iterate heads in the group.
In Kernel 2 (k-grid): we only handle one head, so dscores is straightforward.

## Progress

### Step 1: Kernel 2 written ✅
`_mamba3_chunk_scan_bwd_dx_kernel` added to ssd_triton.py (lines 350-578).
Computes: dx_dt_g, dx_dt_b, dBg (atomic), dBb (atomic), ddA_k (k-side).

Key decisions made:
- Grid: (cdiv(L,BLOCK_K)*cdiv(headdim,BLOCK_N), batch*nchunks, nheads)
- R-loop is serial (not unrolled) to save registers
- Bg[k], Bb[k], x_dt[k] pre-loaded per rank (persistent across m-loop)
- C[m] loaded fresh each m-iteration (varies with m)
- dBg/dBb via tl.atomic_add (one write per (m,k,r) tile per head)
- ddA_k stored only from pid_n==0 to avoid double-counting across headdim blocks
- ddA_k accumulated across ranks in register (BLOCK_K vector persists outside r-loop, single store after all ranks)

Issues fixed:
- dscores was using partial headdim (BLOCK_N slice) — WRONG for dBg/dBb/ddA.
  Fix: dBg/dBb/ddA only computed from pid_n==0, with inner headdim tile loop
  for the full dscores reduction. dx_dt unaffected (tiles over headdim correctly).
- Removed xg_k/xb_k preloads (only used for buggy dscores).

### Step 2: Kernel 1 design (dC + ddA_m)

**What it computes:**
```
dC[m,g,d] = Σ_{k≤m} Σ_r Σ_{h∈g} (dscores_g[m,k,h,r] * L[m,k,h] * scale) * Bg[k,g,r,d]
          + Σ_{k≤m} Σ_r Σ_{h∈g} (dscores_b[m,k,h,r] * L[m,k,h] * scale) * Bb[k,g,r,d]

ddA_m[m,h] = Σ_{k≤m} Σ_r dL[m,k,h,r] * L_active[m,k,h]
```
where dscores_g[m,k,h,r] = Σ_p dy[m,h,p] * x_dt_g[k,h,r,p]

**Grid:** (cdiv(L, BLOCK_M), batch * nchunks, ngroups)
- No headdim tiling needed — dC output is (m, group, dstate)
- No headdim in output at all, but dscores needs full headdim reduction

**Structure mirrors forward kernel** (m-direction, inner loop over k ≤ m):
```
for each m-block (pid_m):
  for each k-block (k ≤ m):
    compute decay L[m,k] — same as forward, tile-by-tile in registers
    for each head h in this group:
      compute full dscores[m,k,h,r] via headdim tile loop
      dcb_h = dscores * L[m,k,h] * scale
      accumulate dcb across heads -> dcb_g[m,k,r] (for dC)
      accumulate ddA_m[m,h] += dL * L_active
    for each rank r:
      dC[m,g,:] += dcb_g[m,k,r] * Bg[k,g,r,:]  (matmul: (BM,BK) @ (BK,BD))
      dC[m,g,:] += dcb_b[m,k,r] * Bb[k,g,r,:]
```

**Key difference from Kernel 2:** Must loop over heads within group for:
1. dscores (per-head dy and L)
2. dcb reduction (sum dscores*L*scale across heads -> group-level dcb)
3. ddA_m (per-head output)

**dscores inside head loop:** For each h in group:
- Load dy[m,h,:] and x_dt[k,h,r,:] — full headdim tile loop
- Compute dscores_g[m,k,h,r] = Σ_p dy * x_dt  (BLOCK_M, BLOCK_K)
- Multiply by L[m,k,h] * scale -> per-head dcb contribution
- Add to group-level dcb_g accumulator
- Compute dL contribution for ddA_m[m,h]

**Output tiling for dC:** dC is (m, group, dstate). We could tile dstate dim:
- Option A: Grid includes dstate tiling -> (cdiv(L,BM)*cdiv(dstate,BD), B*nc, ngroups)
  Then dC writes are tiled and clean, but Bg[k] loads need matching dstate tile.
- Option B: No dstate tiling, write full dstate per m-block.
  Only works if dstate fits in one or two BLOCK_DSTATE tiles (already true — we assert dstate ≤ 2*BLOCK_DSTATE).

**Go with Option B** — same as forward's C tile strategy. dC accumulator is (BLOCK_M, BLOCK_DSTATE) × C_TILES, matching how we load Bg[k].

**Register budget estimate (Option B):**
```
dC_acc tile 0:        BLOCK_M * BLOCK_DSTATE * 4 = 32*64*4 = 8 KB
dC_acc tile 1:        8 KB (if C_TILES > 1)
ddA_m_acc (1 head):   BLOCK_M * 4 = 128 B (stored per head, written immediately)
decay (1 head):       BLOCK_M * BLOCK_K * 4 = 32*32*4 = 4 KB
dcb_g, dcb_b:         BLOCK_M * BLOCK_K * 4 * 2 = 8 KB (accumulated over heads)
dscores (1 head):     BLOCK_M * BLOCK_K * 4 = 4 KB
Bg[k] tile:           BLOCK_K * BLOCK_DSTATE * 2 = 32*64*2 = 4 KB (BF16)
Bb[k] tile:           4 KB
dy tile (headdim loop): BLOCK_M * BLOCK_N * 4 = 32*64*4 = 8 KB
x_dt tile (headdim loop): BLOCK_K * BLOCK_N * 4 = 32*64*4 = 8 KB
dA_cs_m, dA_cs_k:     BLOCK_M + BLOCK_K = 256 B
Total: ~57 KB (manageable with BLOCK_M=32, BLOCK_K=32, BLOCK_DSTATE=64, BLOCK_N=64)
```

**ddA_m storage:** Each head's ddA_m[m,h] is independent. We compute it inside the head loop and write to global memory immediately after processing all k-blocks for that head. But wait — the head loop is INSIDE the k-loop. So we need to accumulate ddA_m across k-blocks.

Revised approach: ddA_m_acc is (BLOCK_M,) per head. We'd need nheads_per_group of these... that's up to 12 vectors. 12 * 32 * 4 = 1.5 KB — fine.

Actually simpler: accumulate ddA_m per-head in the head loop, write with atomic_add after each k-block (since same m positions written from different k iterations). Or: accumulate across ALL k-blocks, then write once.

**Simplest:** ddA_m_acc shape (BLOCK_M, nheads_per_group). After all k-blocks done, write each head's column. But nheads_per_group isn't constexpr-friendly for indexing.

**Pragmatic:** Use atomic_add for ddA_m. Each (m-block, k-block, head) contributes its piece. This is safe because m-blocks are in different grid programs (no conflict), and within one program we process heads sequentially.

Wait — within one program, we have ONE m-block. We loop over k-blocks and heads. The ddA_m contributions for the SAME m positions accumulate across k-blocks. So we CAN just keep a (BLOCK_M,) accumulator per head, add to it in the k-loop, and write once after all k-blocks.

**Final plan for ddA_m:**
- Before k-loop: for each h in group, init ddA_m_h_acc = zeros(BLOCK_M)
- Can't do this easily with variable nheads_per_group...
- **Just use atomic_add.** Write ddA_m contribution after each (k, h) iteration. Same m-block = same program = sequential writes = no actual contention. The "atomic" is just to handle accumulation cleanly.
### Step 2b: Kernel 1 written ✅
`_mamba3_chunk_scan_bwd_dC_kernel` added to ssd_triton.py (lines 582-770).
Computes: dC, ddA_m (m-side of ddA_cumsum).

Key decisions made:
- Grid: (cdiv(L,BLOCK_M), batch*nchunks, ngroups) — no headdim/dstate tiling
- Pre-loads C[m,g,:] tiles (persistent, m is fixed per program)
- Loop order: k-loop → r-loop → head-loop (heads innermost)
- Per-rank: loads Bg/Bb[k,g,r,:], computes cb=C@B^T, then sweeps heads
- dscores per (m,k,h,r) via full headdim tile loop inside head loop
- dcb_g accumulated across heads per rank, then used for dC matmul
- ddA_m via load-add-store per (k,r,h) — no cross-program contention (one program per m-block)
- dC matmul: dcb_g @ Bg_k → (BLOCK_M, BLOCK_DSTATE), accumulated across k and r
- k_max optimization: only iterate k ≤ max(m in block), same as forward

### Step 3: SRAM estimators + block selection ✅
Added `_sram_estimate_bwd_dx`, `_sram_estimate_bwd_dC`, `_select_block_sizes_bwd`.
- Separate SRAM estimates for each kernel (Kernel 2 has dx accumulators + dscores path; Kernel 1 has persistent C/dC tiles)
- Bg/Bb tile count conditional on C_TILES: `2 * C_TILES * BLOCK_K * BLOCK_DSTATE * 2`
- Both estimates include `diff` (BLOCK_M × BLOCK_K × 4) for the long-lived decay difference tensor
- dC estimate includes `L_active` (BLOCK_M × BLOCK_K × 4) for the ddA indicator mask
- Block candidates: (64,32) → (32,32) → (32,16), picks first that fits both kernels
- Always num_stages=1 (backward has more live data than forward)
- BLOCK_N = min(64, next_pow2(headdim)), BLOCK_DSTATE = min(64, next_pow2(dstate))

### Step 4: Python wrapper ✅
`mamba3_chunk_scan_bwd` added.
- Allocates outputs: dx_dt_g/b (empty), dBg/dBb (zeros, atomic target), dC (empty), ddA_m (zeros), ddA_k (empty)
- Launches Kernel 2 first (dx_dt + dBg/dBb + ddA_k), then Kernel 1 (dC + ddA_m)
- Combines: ddA_cumsum = ddA_m + ddA_k
- Returns (dx_dt_g, dx_dt_b, dBg, dBb, dC, ddA_cumsum)

### Step 5: Autograd integration ✅
- Import: `from nanochat.ssd_triton import mamba3_chunk_scan_fwd, mamba3_chunk_scan_bwd`
- `_mamba3_intra_autograd_bwd`: if `Bg.is_cuda` → Triton path, else PyTorch fallback
- Return order matches custom_op inputs: (dBg, dBb, dx_dt_g, dx_dt_b, ddA_cumsum, dC, None, None, None)

### Step 6: Tests ✅
Added 3 backward tests to test_mamba3.py:
- `test_triton_bwd_per_gradient`: Each gradient vs PyTorch (dBg, dBb, dx_dt_g/b, dC, ddA_cumsum), tol 0.02
- `test_triton_bwd_multigroup`: 4 heads / 1 group, tests atomic dBg/dBb correctness
- `test_triton_bwd_full_layer`: Full Mamba3Layer forward-backward with Triton, checks all params have gradients
