# Code Structure

LIRiAP code organization and folder layout.

## Top-Level Structure

```
LIRiAP/
├── LIRiAP_pack/              # Core algorithm implementations
│   ├── *[_algorithm.py]      # QGIS Processing wrappers
│   ├── *[_worker.py]         # Geometry solvers (QGIS-independent)
│   ├── numba_bootstrap.py    # Numba JIT helper
│   └── help_descriptions.py  # Shared help text
├── LiRiAP_provider/          # QGIS plugin integration
│   ├── LIRiAP_plugin.py      # Plugin entry point
│   └── algorithms/           # Algorithm provider files
├── tests/                    # Test suite
├── docs/                     # Documentation
└── README.md                 # Main readme
```

## LIRiAP_pack/

### Algorithm Wrappers (`*_algorithm.py`)

Each algorithm has a QGIS Processing wrapper that:
- Declares parameters
- Handles feature serialization
- Manages parallel execution
- Writes output

| File | Algorithm |
|------|-----------|
| `approximation_standard_algorithm.py` | Approximation Standard |
| `approximation_fast_algorithm.py` | Approximation Fast |
| `contained_standard_algorithm.py` | Contained Standard |
| `contained_fast_algorithm.py` | Contained Fast |
| `bcrs_algorithm.py` | BCRS |
| `bcrs_fast_algorithm.py` | BCRS Fast |
| `axis_aligned_lir_algorithm.py` | Axis-Aligned LIR |

### Worker Modules (`*_worker.py`)

Geometry solvers independent of QGIS/Qt runtime:
- Pure computational logic
- Can be called programmatically
- Numba JIT compilation where applicable

| File | Solves |
|------|--------|
| `approximation_standard_worker.py` | Approximation geometric search |
| `approximation_fast_worker.py` | Approximation with slice execution |
| `contained_standard_worker.py` | Contained Stage 1-4 |
| `contained_fast_worker.py` | Contained with optimizations |
| `bcrs_worker.py` | BCRS Stage 1-7 with CABF |
| `bcrs_fast_worker.py` | BCRS with trial ranking |
| `axis_aligned_lir_worker.py` | Exact axis-aligned solvers |

### Support Files

| File | Purpose |
|------|---------|
| `numba_bootstrap.py` | Safe Numba import/install helper |
| `help_descriptions.py` | Shared right-panel algorithm descriptions |

## LiRiAP_provider/

QGIS plugin integration files that wrap LIRiAP algorithms as QGIS Processing algorithms.

## tests/

Test files for algorithm validation:
- `test_axis_aligned_lir.py` — Axis-Aligned LIR exactness
- `test_event_emission.py` — Event handling
- `test_real_world_comparison.py` — Integration tests
- `test_numba_bootstrap.py` — Numba bootstrap safety
- `test_tuning_constants.py` — Parameter guardrails
- `test_worker_edge_cases.py` — Edge case handling

## Module Dependencies

```
QGIS Processing Framework
    │
    ├── algorithm.py (wrapper)
    │       │
    │       ├── worker.py (geometry solver)
    │       │       │
    │       │       ├── numpy, shapely
    │       │       └── numba (optional, for acceleration)
    │       │
    │       ├── help_descriptions.py
    │       └── numba_bootstrap.py
    │
    └── QGIS Runtime (Qt)
```

## Calling Workers Directly

Worker modules can be imported and used programmatically without QGIS:

```python
import sys
sys.path.insert(0, 'LIRiAP_pack')

# Approximation
from approximation_standard_worker import worker_process_feature

# Contained
from contained_standard_worker import worker_process_feature

# BCRS
from bcrs_worker import worker_process_feature

# Axis-Aligned
from axis_aligned_lir_worker import _worker_process_feature
```

See [Usage Guide](./usage.md) for detailed examples.