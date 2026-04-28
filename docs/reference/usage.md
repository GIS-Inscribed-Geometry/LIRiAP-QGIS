# Programmatic Usage

Call worker functions directly from Python code without QGIS.

## Basic Usage Pattern

All worker modules share a common pattern:
1. Import the worker module
2. Prepare geometry as WKB bytes
3. Call the worker function with parameters
4. Process results

## Approximation Standard

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))

from approximation_standard_worker import process_feature
from shapely.wkb import dumps as wkb_dumps
from shapely.geometry import Polygon

# Prepare input
polygon = Polygon([(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)])
wkb_bytes = bytes(wkb_dumps(polygon))

params = (
    40,          # grid_coarse 
    100,         # grid_fine
    5.0,         # angle_step
    1.6,         # max_ratio
    None,        # buffer_value
    False,       # use_buffer
)

# Call worker
result = process_feature((1, wkb_bytes, *params))

# Result: (feat_id, wkt, area, angle, ratio) or None
if result:
    feat_id, wkt, area, angle, ratio = result
    print(f"Area: {area}, Angle: {angle}, Ratio: {ratio}")
```

## Approximation Fast

```python
from approximation_fast_worker import process_slice

job_array = [
    (feat_id, wkb_bytes, *params),
    ...
]
results, _ = process_slice(job_array, 0, len(job_array), params)
```

## Contained Standard

```python
from contained_standard_worker import process_feature
from shapely.wkb import dumps as wkb_dumps

polygon = Polygon([(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)])
wkb_bytes = bytes(wkb_dumps(polygon))

params = (
    40,          # grid_coarse
    120,         # grid_fine
    5.0,         # angle_step
    3,           # top_k
    1.6,         # max_ratio
    True,        # always_return
    False,       # use_buffer
    -0.5,        # buffer_value
)

result = process_feature((1, wkb_bytes, *params))

# Result: (feat_id, wkt, area, angle, ratio, cand_rank, s2_gain, best_effort) or None
```

## Contained Fast

```python
from contained_fast_worker import process_feature

# Same interface as Contained Standard
result = process_feature((1, wkb_bytes, *params))
```

## BCRS

```python
from bcrs_worker import process_feature
from shapely.wkb import dumps as wkb_dumps

polygon = Polygon([(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)])
wkb_bytes = bytes(wkb_dumps(polygon))

params = (
    40,          # grid_coarse
    120,         # grid_fine
    5.0,         # angle_step
    3,           # top_k
    1.6,         # max_ratio
    True,        # always_return
    False,       # use_buffer
    -0.5,        # buffer_value
)

result = process_feature((1, wkb_bytes, *params))

# Result: (feat_id, wkt, area, angle, ratio, cand_rank, s2_gain, s4_gain, s5_gain, best_effort) or None
```

## BCRS Fast

```python
from bcrs_fast_worker import process_feature

# Same interface as BCRS
result = process_feature((1, wkb_bytes, *params))
```

## Axis-Aligned LIR

```python
from axis_aligned_lir_worker import _worker_process_feature
from shapely.wkb import dumps as wkb_dumps
from shapely.geometry import Polygon

polygon = Polygon([(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)])
wkb_bytes = bytes(wkb_dumps(polygon))

params = (
    0.0,         # axis_angle (degrees)
    120,         # grid_fine (fallback)
    1.6,         # max_ratio
    False,       # use_buffer
    -0.5,        # buffer_value
    True,        # always_return
)

result = _worker_process_feature((1, wkb_bytes, *params))

# Result: (feat_id, wkt, area, axis_angle, poly_type, ratio, best_effort) or None
```

## Using Exact Solvers Directly (Axis-Aligned)

```python
from axis_aligned_lir_worker import (
    _detect_polygon_type,
    _exact_solve_convex,
    _exact_solve_vertex_grid,
)
from shapely.geometry import Polygon

# Create polygon
hexagon = Polygon([
    (10, 0), (5, 8.66), (-5, 8.66),
    (-10, 0), (-5, -8.66), (5, -8.66)
])

# Detect type
poly_type = _detect_polygon_type(hexagon)
# Returns: 'convex_no_holes', 'convex_with_holes', 
#          'concave_no_holes', 'concave_with_holes'

# Solve based on type
if poly_type == 'convex_no_holes':
    rect = _exact_solve_convex(
        hexagon, 
        max_ratio=1.6,
        always_return=True,
        use_buffer=False,
        buffer_value=-0.5
    )
else:
    rect = _exact_solve_vertex_grid(
        hexagon,
        max_ratio=1.6,
        always_return=True,
        use_buffer=False,
        buffer_value=-0.5
    )

if rect:
    print(f"Area: {rect.area}")
```

## Sliced Execution (Fast Variants)

For processing multiple features efficiently:

```python
from approximation_fast_worker import process_slice

job_array = []
for feat_id, polygon in enumerate(polygons):
    wkb = bytes(wkb_dumps(polygon))
    job_array.append((feat_id, wkb, *shared_params))

# Process in parallel-friendly slices
results, best_effort_count = process_slice(
    job_array, 
    start=0, 
    end=len(job_array), 
    shared_params=shared_params
)
```

## Error Handling

Workers return `None` for invalid inputs:
- Empty geometry
- Invalid parameters
- Certification failures (when ALWAYS_RETURN=False)

Always check for `None` before accessing results.

## Parameter Tuples by Algorithm

| Algorithm | Params Length | Key Parameters |
|-----------|---------------|----------------|
| Approx Standard | 6 | grid_coarse, grid_fine, angle_step, max_ratio |
| Approx Fast | 6 | Same as Standard |
| Contained Standard | 8 | + top_k, always_return, use_buffer, buffer_value |
| Contained Fast | 8 | Same as Standard |
| BCRS | 8 | Same as Contained |
| BCRS Fast | 8 | Same as BCRS |
| Axis-Aligned | 6 | axis_angle, grid_fine, max_ratio, use_buffer, buffer_value, always_return |