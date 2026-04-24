# lir_solver — Rust/PyO3 LIR backend

Fastest axis-aligned Largest Inscribed Rectangle solver for the LIRiAP-QGIS plugin.

## Pipeline

```
classify → rotate → [convex: O(n²) Alt/Amenta] or [general: vertex-grid LRH]
         → 4-side binary-search refine (52 steps × 2 passes, ≈1e-15 m precision)
         → certify → rotate back
```

## Build

```bash
# One-time (installs Rust + maturin):
bash install_deps.sh

# Subsequent builds:
bash build.sh
```

Requires Python ≥ 3.9 and a Rust stable toolchain.

## QGIS integration

Copy `python/lir_rust_shim.py` into `LIRiAP-QGIS/LIRiAP_pack/` and add to the top
of `axis_aligned_lir_worker.py`:

```python
try:
    from lir_rust_shim import solve_lir_shim as _solve_axis_aligned_lir
except ImportError:
    pass  # keeps original Python implementation
```

The shim signature is identical; `grid_fine` is accepted but ignored
(the Rust solver always uses the exact vertex-coordinate grid).

## API

```python
import lir_rust

# Single polygon
result = lir_rust.solve_axis_aligned_lir(
    exterior_flat,  # [x0,y0, x1,y1, ...]
    holes_flat_list,  # [[x,y,...], ...]
    axis_angle,  # degrees
    max_ratio,  # e.g. 1.6; 0 = unconstrained
    always_return,  # bool
    buf_value,  # metres, negative = inset
)
# Returns (corners_list, area, ratio, poly_type, best_effort) or None

# Batch
results = lir_rust.solve_axis_aligned_lir_batch(
    [(ext, holes), ...], axis_angle, max_ratio, always_return, buf_value
)
```

## Benchmark

```bash
python python/benchmark.py testt_cases-2.geojson
```

Expected: ~3–5 s for 290 Polish geodata features (vs ~24 s Python).
