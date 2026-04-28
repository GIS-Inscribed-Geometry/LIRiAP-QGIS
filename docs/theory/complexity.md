# Computational Complexity Analysis

Formal time and space complexity analysis for all LIRiAP algorithms.

## Symbols

| Symbol | Meaning |
|--------|---------|
| n | Total polygon vertices (exterior + holes) |
| g_coarse | Coarse grid size (GRID_COARSE) |
| g_fine | Fine grid size (GRID_FINE) |
| k | Candidate count kept for refinement (TOP_K) |
| m | Edge-guided initial angle candidates (<=12 in Contained/BCRS, <=10 in Approximation) |
| s90 | Fallback sweep size: ceil(90 / a), where a = ANGLE_STEP |
| s180 | Approximation fallback sweep size: ceil(180 / a) |
| p | Brent objective evaluations (maxiter=60 where explicitly set) |
| t | Stage 4-5 angle trials per candidate (<=4 in BCRS, <=2 in BCRS Fast) |
| X, Y | Unique boundary x/y coordinates after rotation (BCRS grid lines) |
| nu | BCRS cell count: (|X| - 1) * (|Y| - 1), max 89401 (300x300) |

## Primitive Solver Costs

### Uniform Grid Solver
`T_grid(g) = O(g^2)`, `M_grid(g) = O(g^2)`

### BCRS Variable-Pitch Solver
`T_bcrs = O(n log n + nu)`, `M_bcrs = O(n + nu)`

Implementation guard: if |X| > 300 or |Y| > 300, BCRS is skipped (seed fallback).

### CABF Expansion
`T_cabf = O(n)`

Bounded iteration counts; worst-case geometric predicate cost.

### Certification + Best-Effort Shrink
`T_cert = O(n)`, `T_shrink = O(n)`

### Axis-Aligned Solvers (Alt/Amenta, Daniels et al.)
`T_axis = O(n^2)`, `M_axis = O(n^2)`

## Per-Feature Complexity

### Generic Pipeline
```
T_feature = T_angles + T_stage1 + T_refine + T_cert/fallback
```

### BCRS-Family
```
T_feature,BCRS = T_angles + T_stage1 + T_refine + T_expand + T_cert/fallback
```

## Algorithm Worst-Case Complexity

| Algorithm | Time (single feature) | Memory |
|-----------|----------------------|--------|
| Approximation Standard | O(n + (m+s180)g_coarse^2 + (p+1)g_fine^2) | O(max(g_coarse^2, g_fine^2)) |
| Approximation Fast | Same as Standard (batch changes constants) | Same |
| Contained Standard | O(n + (m+s90)g_coarse^2 + k((p+2)g_fine^2 + n)) | O(max(g_coarse^2, g_fine^2)) |
| Contained Fast | O(n + (m+s90)g_coarse^2 + k(pg_coarse^2 + g_fine^2 + n)) | O(max(g_coarse^2, g_fine^2)) |
| BCRS | O(n + (m+s90)g_coarse^2 + k(pg_coarse^2 + t(g_fine^2 + n log n + nu + n) + n)) | O(max(g_fine^2, nu)) |
| BCRS Fast | O(n + (m+s90)g_coarse^2 + k((p+4)g_coarse^2 + t(n log n + nu + n) + n)), t<=2 | O(max(g_coarse^2, nu)) |
| Axis-Aligned LIR | O(n^2) | O(n^2) |

## Fast Variant Speedup

### Contained
Delta approx: `p * (g_fine^2 - g_coarse^2)` saved per candidate

### BCRS
Delta approx: `(4-2)(n log n + nu) + 4g_fine^2 - 4g_coarse^2`

## Default Parameter Operation Model

Defaults used for calculation:

| Family | g_coarse | g_fine | k | ANGLE_STEP |
|--------|----------|--------|---|------------|
| Approximation | 40 | 100 | 1 effective | 5 |
| Contained/BCRS | 40 | 120 | 3 | 5 |

Derived values: s180 = 36, s90 = 18, 40^2 = 1600, 100^2 = 10000, 120^2 = 14400

### Dominant Term Estimates

| Algorithm | Estimated Grid Units |
|-----------|---------------------|
| Approximation Standard/Fast | (10+36)*1600 + (60+1)*10000 = 683,600 |
| Contained Standard | (12+18)*1600 + 3*(60+2)*14400 = 2,726,400 |
| Contained Fast | (12+18)*1600 + 3*(60*1600 + 14400) = 379,200 |
| BCRS | (12+18)*1600 + 3*(60*1600 + 4*(14400 + 89401)) ≈ 1,581,612 |
| BCRS Fast | (12+18)*1600 + 3*((60+4)*1600 + 2*89401) ≈ 891,606 |
| Axis-Aligned LIR | O(n^2) — depends on vertex count, not grid resolution |

Note: These are complexity-weighted operation counts, not wall-clock predictions.

## Verification Against Wall-Clock Times

Using baseline 5406-feature wall times (N_WORKERS=1, no chunking):

| Relation Check | Model Expectation | Measured | Result |
|----------------|------------------|----------|--------|
| Approximation Fast vs Standard | Nearly equal | 125.93s vs 127.25s | Consistent |
| Contained Fast vs Standard | Fast should be lower | 226.05s < 574.13s | Consistent |
| BCRS Fast vs BCRS | Fast should be lower | 445.01s < 772.05s | Consistent |
| Contained Standard vs BCRS | Model estimate favors BCRS | 574.13s < 772.05s | Mismatch |

The model captures intra-family speed relations well. Cross-family ordering can vary due to solver semantics, settings, and constant factors not captured by asymptotic terms.

## Scaling Behavior

### Observed Scaling Exponents (5406 features / 290 features ≈ 18.65x)

| Algorithm | 1 Worker | 12 Workers |
|-----------|----------|------------|
| Approximation Standard | 0.9851 | 1.0031 |
| Approximation Fast | 0.9888 | 0.9951 |
| Contained Standard (strict) | 1.0039 | 0.9923 |
| Contained Standard (fallback) | 1.0002 | 0.9993 |
| Contained Fast (fallback) | 0.9965 | 1.0009 |
| BCRS (strict) | 0.9879 | 0.9851 |
| BCRS (fallback) | 0.9994 | 0.9975 |
| BCRS Fast (fallback) | 1.0038 | 1.0005 |
| Axis-Aligned LIR | 1.0058 | 1.0067 |

All exponents near 1.0 indicate linear scaling with feature count.

### Parallel Efficiency

| Algorithm | Speedup @290 | Efficiency @290 | Speedup @5406 | Efficiency @5406 |
|-----------|-------------|-----------------|---------------|------------------|
| Approx Standard | 1.19 | 9.95% | 1.13 | 9.44% |
| Approx Fast | 1.18 | 9.86% | 1.16 | 9.68% |
| Contained Standard | 1.37 | 11.39% | 1.41 | 11.79% |
| Contained Fast | 1.02 | 8.49% | 1.01 | 8.38% |
| BCRS | 0.83 | 6.90% | 0.83 | 6.96% |
| BCRS Fast | 0.79 | 6.59% | 0.80 | 6.66% |
| Axis-Aligned LIR | 0.80 | 6.63% | 0.76 | 6.32% |

BCRS and Axis-Aligned families show negative speedup (slowdown) with multithreading due to synchronization overhead exceeding per-feature computation time.