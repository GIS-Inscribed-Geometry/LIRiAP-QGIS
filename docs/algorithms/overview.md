# Algorithm Families Overview

LIRiAP implements four problem variants and multiple execution modes for each.

## Family Comparison

| Family | Primary Objective | Strict Containment | Boundary Expansion | Best For |
|--------|------------------|-------------------|-------------------|----------|
| Approximation | Fast area-focused search | No (not certified) | No | Quick candidates, exploratory analysis |
| Contained | Certified contained rectangle | Yes (unless fallback enabled) | No | Guaranteed containment without expansion |
| BCRS | Full contained + fit improvement | Yes (unless fallback enabled) | Yes (CABF) | Maximum accuracy, closest to theoretical optimum |
| Axis-Aligned | Exact fixed-axis solve | Exact (vertex-coordinate) | N/A | Fastest for axis-aligned problems, exact solution |

## Performance (baseline, 1 worker)

| Algorithm | Time @290 (s) | Time @5406 (s) | Scaling |
|-----------|--------------|---------------|---------|
| Approx Fast | 6.98 | 125.93 | 18.0x |
| Axis-Aligned LIR | 11.81 | 120.24 | 10.2x |
| Contained Fast | 12.25 | 226.05 | 18.5x |
| BCRS Fast | 23.61 | 445.01 | 18.8x |

## Algorithm Matrix

| Algorithm | Variant | Containment Semantics | Expansion | Complexity |
|-----------|---------|----------------------|-----------|------------|
| Approximation Standard | Approx | Not certified | No | O(n + (m+s180)g² + (p+1)g²) |
| Approximation Fast | Approx | Not certified | No | Same as Standard |
| Contained Standard | Contained | Certified / best-effort | No | O(n + (m+s90)g² + k((p+2)g² + n)) |
| Contained Fast | Contained | Certified / best-effort | No | O(n + (m+s90)g² + k(pg² + g² + n)) |
| BCRS | BCRS | Certified / best-effort | CABF | O(n + (m+s90)g² + k(pg² + t(g² + n log n + nu + n) + n)) |
| BCRS Fast | BCRS | Certified / best-effort | CABF | O(n + (m+s90)g² + k((p+4)g² + t(n log n + nu + n) + n)) |
| Axis-Aligned LIR | Exact | Exact (vertex-coordinate) | N/A | O(n²) |

## Execution Modes

All algorithms support multiple execution modes via parameters:

| Mode | N_WORKERS | USE_CHUNKING | Characteristics |
|------|-----------|--------------|-----------------|
| Serial | 1 | False | Single-threaded, no overhead |
| Parallel | >1 | False | Per-feature parallel execution |
| Chunked | >1 | True | Chunk-based parallel, better for canceling |

## Selection Guide

```
Start: Do you need strict containment?
├─ No → Approximation family (fast candidates)
│
└─ Yes → Do you need axis-aligned rectangles?
    ├─ Yes → Axis-Aligned LIR (exact, fastest for this case)
    │
    └─ No → Need boundary expansion?
        ├─ No → Contained family
        │   ├─ Need speed → Contained Fast
        │   └─ Simpler → Contained Standard
        │
        └─ Yes → BCRS family
            ├─ Need speed → BCRS Fast
            └─ Maximum accuracy → BCRS
```

## Axis-Aligned LIR

For fixed-orientation (axis-aligned) rectangles, use Axis-Aligned LIR:
- Exact solution (vertex-coordinate precision)
- O(n²) complexity
- Supports convex, concave, with/without holes
- Optional rotation parameter for tilted axis
- Fastest algorithm for axis-aligned problems (11.81s @290, 120.24s @5406)

See [Axis-Aligned LIR](./axis-aligned.md) for details.