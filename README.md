# LIRiAP

LIRiAP (Largest Inscribed Rectangle in Arbitrary Polygon) is a set of QGIS Processing algorithms for computing the largest inscribed rectangles in polygon features.

## Problem statement

Given an input polygon, find a large non axis aligned interior rectangle (concave polygons and polygons with holes supported). In this pack, **four different problem variants** are implemented:

1. **Approximation family**: maximize area quickly, without strict containment certification. Good for finding candidates
2. **Contained family**: enforce containment certification, but do not run boundary expansion after certification.
3. **BCRS family**: containment certification **plus** boundary-coordinate expansion (CABF) - contain & extend. This is the only family in this pack intended to mostly solve the full "largest-area, non axis aligned, fully contained rectangle with expansion" target. Best for finding results closer to solves on more limited set of features.
4. **Axis-Aligned family**: exact fixed-axis solve with vertex-coordinate precision.

## Result screenshots (constrained to 16:10 resolution)

### Approximation (less vs denser grid)

![Approximation result](media/Approximate.png)

![Approximation (improved candidate)](media/Approximate_better.png)

### Contained

![Contained result](media/Contained.png)

### BCRS (Boundary-Coordinate Raster Solve)

![BCRS result](media/BCRS.png)

![BCRS result (zoom)](media/BCRS_zoom.png)

---

## Potential uses

- **Suitability analysis**: search candidate locations for building or infrastructure placement by finding the largest feasible rectangular footprint inside constrained parcels (e.g., houses, warehouses, solar arrays, staging pads, retention structures) while respecting parcel boundaries and holes/exclusions.
- **Remote sensing**: derive stable interior rectangular patches for spectral sampling, calibration windows, texture statistics, and object-level summaries where centroid or full-polygon sampling is noisy.
- **Dynamic cartographic label placement**: place labels in the largest interior rectangle instead of using only centroid or bounding box, improving readability in concave polygons and polygons with holes. An axis-aligned version could be fast enough to handle this task.
- **Other scenarios**: map tiling anchors, drone landing-zone preselection, interior ROI extraction for QA workflows, and standardized shape descriptors for downstream analytics.
- **Computer vision**: find maximum rectangular regions of interest within arbitrary shaped detection masks
- **Game development**: calculate valid placement areas for rectangular game objects within complex terrain polygons

The less the features the denser the grid can be whilst still maintaining reasonable accuracy.

### Potential for other algorithms

The ideas in this pack could potentially be used to get solutions for other contained shapes, as well as the reverse problem - finding positions for inscribed polygons in a rectangle in a way that maximizes used space.

## At a glance

From the fastest to slowest. BCRS without multithreaded processing is usually the best option for finding the maximum area. "Approximation fast" with multithreaded processing should be the best at finding candidates in large datasets. But this may vary depending on device and dataset. Mind that chunking blocks cancelling the run. I advise experimenting with grid parameters for the result best fitting your requirements (time of processing vs accuracy).


| Family        | Primary objective                            | Strict containment               | Boundary expansion |
| ------------- | -------------------------------------------- | -------------------------------- | ------------------ |
| Approximation | Fast area-focused search                     | No                               | No                 |
| Contained     | Certified contained rectangle search         | Yes (unless fallback is enabled) | No                 |
| BCRS          | Certified contained search + fit improvement | Yes (unless fallback is enabled) | Yes (CABF)         |
| Axis-Aligned  | Exact fixed-axis solve                       | Yes (vertex-coordinate)         | N/A                |

Best execution mode by algorithm (@290 @5406 are number of run features in a dataset):


| Algorithm                     | Best mode @290 | Best mode @5406 |
| ----------------------------- | -------------- | --------------- |
| Approximation Standard        | 12w            | 12w+chunk       |
| Approximation Fast            | 12w            | 12w+chunk       |
| Contained Standard (strict)   | 12w+chunk      | 12w             |
| Contained Standard (fallback) | 12w+chunk      | 12w+chunk       |
| Contained Fast (fallback)     | 12w+chunk      | 12w+chunk       |
| BCRS (strict)                 | 1w             | 1w              |
| BCRS (fallback)               | 1w             | 1w              |
| BCRS Fast (fallback)          | 1w             | 1w              |
| Axis-Aligned LIR              | 1w             | 1w              |

## Shared components

All algorithms in `LIRiAP_pack` follow the same structure:

1. **Input normalization**: read polygon geometry; for multipolygons, use the largest part.
2. **Angle candidates**: extract likely orientations from polygon edge directions, with a fallback sweep.
3. **Rectangle solve in rotated frame**: solve axis-aligned rectangle candidates on a rotated polygon to recover non axis aligned solutions in map coordinates.
4. **Refinement and checks**: apply finer search and containment-related adjustments (depending on variant).
5. **Output**: write rectangle geometry and metrics (area, angle, ratio, and variant-specific diagnostics).


## Algorithms


| Algorithm                                               | What problem it solves                                       | Containment semantics                                                                                   | Expansion semantics                                                |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Approximation Standard                                  | Fast area-focused approximation                              | Not certified; rectangle can violate containment in difficult cases                                     | No expansion stage                                                 |
| Approximation Fast                                      | Same as Approximation Standard with lower overhead execution | Not certified; same semantics as Standard                                                               | No expansion stage                                                 |
| Contained Standard                                      | Certified contained rectangle search                         | Certified contained when strict mode succeeds; optional best-effort fallback can relax strict guarantee | No expansion stage after certification                             |
| Contained Fast                                          | Same as Contained Standard with optimized execution          | Same certified/best-effort semantics as Standard                                                        | No expansion stage after certification                             |
| BCRS (Boundary-Coordinate Raster Solve)                 | Full contained-plus-expansion solve                          | Certified contained when strict mode succeeds; optional best-effort fallback can relax strict guarantee | Includes CABF boundary expansion (full target method in this pack) |
| BCRS Fast (Boundary-Coordinate Raster Solve, optimized) | Same as BCRS with prioritized/optimized execution            | Same certified/best-effort semantics as BCRS                                                            | Includes CABF boundary expansion                                   |
| Axis-Aligned LIR                                        | Exact fixed-axis solve                                      | Exact (vertex-coordinate precision)                                                                    | N/A                                                                |

## Setting semantics

- `ALWAYS_RETURN` (Contained/BCRS):
  - `False`: strict certification only; features may return no rectangle if strict containment cannot be certified.
  - `True`: returns best-effort fallback when strict certification fails (`best_effort=1`), so strict guarantee is no longer universal.
- `USE_BUFFER` + `BUFFER_VALUE` (Contained/BCRS): applies an additional containment margin in map units (usually reducing area to increase margin from boundaries/holes).
- `MAX_RATIO`: constrains the admissible rectangle aspect ratio; tighter cap can reduce max area.
- `GRID_*`, `ANGLE_STEP`, `TOP_K`: search density and candidate breadth controls; they change result quality/runtime tradeoff, not the solver family semantics.
- `N_WORKERS`, `USE_CHUNKING`, `AUTO_INSTALL_NUMBA`: runtime/performance controls only; they do not change geometric guarantees.

## Processing benchmark (default settings)

All runs assume default algorithm parameters and Numba installed. 290 and 5406 are the number of features in the testing dataset.

Benchmarked with:

- i5-12400F
- 32GB DDR4 RAM

### Baseline profile (N_WORKERS=1, USE_CHUNKING=False)


| Profile | Algorithm              | ALWAYS_RETURN            | Time @ 290 (s)<br />*5 run average | Time @ 5406 (s) | Scale ratio (5406 / 290) |
| ------- | ---------------------- | ------------------------ | ----------------------------------- | --------------- | ------------------------ |
| P1      | Approximation Standard | n/a                      | 7.13*                               | 127.25          | 17.8471                  |
| P2      | Approximation Fast     | n/a                      | 6.98*                               | 125.93          | 18.0415                  |
| P3      | Contained Standard     | False (strict)           | 30.45                               | 574.13          | 18.8548                  |
| P4      | Contained Standard     | True (fallback enabled)  | 30.75                               | 573.59          | 18.6533                  |
| P5      | Contained Fast         | True (fallback enabled) | 12.25*                              | 226.05          | 18.4531                  |
| P6      | BCRS                   | False (strict)           | 42.91                               | 772.05          | 17.9923                  |
| P7      | BCRS                   | True (fallback enabled)  | 42.35                               | 788.03          | 18.6076                  |
| P8      | BCRS Fast              | True (fallback enabled)  | 23.61                               | 445.01          | 18.8484                  |
| P9      | Axis-Aligned LIR       | True (fallback enabled)  | 11.81                               | 120.24          | 10.1812                  |

### Parallel profile (N_WORKERS=12, USE_CHUNKING=False)


| Profile | Algorithm              | ALWAYS_RETURN            | Time @ 290 (s)<br />*5 run average | Time @ 5406 (s) | Scale ratio (5406 / 290) |
| ------- | ---------------------- | ------------------------ | ---------------------------------- | --------------- | ------------------------ |
| P1      | Approximation Standard | n/a                      | 5.97*                              | 112.30          | 18.8107                  |
| P2      | Approximation Fast     | n/a                      | 5.90*                              | 108.43          | 18.3780                  |
| P3      | Contained Standard     | False (strict)           | 22.27                              | 405.91          | 18.2268                  |
| P4      | Contained Standard     | True (fallback enabled)  | 22.05                              | 410.21          | 18.6036                  |
| P5      | Contained Fast         | True (fallback enabled) | 12.03*                             | 224.82          | 18.6883                  |
| P6      | BCRS                   | False (strict)           | 51.83                              | 925.01          | 17.8470                  |
| P7      | BCRS                   | True (fallback enabled)  | 50.88                              | 941.69          | 18.5081                  |
| P8      | BCRS Fast              | True (fallback enabled) | 29.84                              | 557.11          | 18.6699                  |
| P9      | Axis-Aligned LIR       | True (fallback enabled)  | 14.83                              | 158.53          | 10.6897                  |

### Parallel + chunking profile (N_WORKERS=12, USE_CHUNKING=True)


| Profile | Algorithm              | ALWAYS_RETURN            | Time @ 290 (s)<br />*5 run average | Time @ 5406 (s) | Scale ratio (5406 / 290) |
| ------- | ---------------------- | ------------------------ | ---------------------------------- | --------------- | ------------------------ |
| P1      | Approximation Standard | n/a                      | 6.04*                              | 109.76          | 18.1722                  |
| P2      | Approximation Fast     | n/a                      | 5.90*                              | 108.43          | 18.3780                  |
| P3      | Contained Standard     | False (strict)           | 21.98                              | 405.95          | 18.4691                  |
| P4      | Contained Standard     | True (fallback enabled)  | 21.96                              | 405.15          | 18.4495                  |
| P5      | Contained Fast         | True (fallback enabled) | 12.01*                             | 224.82          | 18.7194                  |
| P6      | BCRS                   | False (strict)           | 51.10                              | **              |                          |
| P7      | BCRS                   | True (fallback enabled)  | 51.30                              | **              |                          |
| P8      | BCRS Fast              | True (fallback enabled)  | 30.19                              | **              |                          |
| P9      | Axis-Aligned LIR       | True (fallback enabled) | 14.91                              | 157.89          | 10.5912                  |

## Folder layout

- `LIRiAP_pack/*_algorithm.py`: QGIS Processing wrappers (parameters, execution, output fields, help text).
- `LIRiAP_pack/*_worker.py`: geometry solvers independent from QGIS/Qt runtime.
- `LIRiAP_pack/numba_bootstrap.py`: optional Numba bootstrap helper.
- `LIRiAP_pack/help_descriptions.py`: shared right-panel algorithm descriptions.
- `tests/*.py`: unit tests for bootstrap safety, edge cases, and tuning-constant guardrails.

## Documentation

Detailed documentation is available in the [GitHub Wiki](https://github.com/Wolren/LIRiAP-QGIS/wiki):

- [Home](https://github.com/Wolren/LIRiAP-QGIS/wiki/Home) — Overview and quick start
- [Algorithms](https://github.com/Wolren/LIRiAP-QGIS/wiki/Algorithms) — Family comparison with flowcharts
- [Approximation](https://github.com/Wolren/LIRiAP-QGIS/wiki/Approximation) — Approximation algorithm details
- [Contained](https://github.com/Wolren/LIRiAP-QGIS/wiki/Contained) — Contained algorithm details
- [BCRS](https://github.com/Wolren/LIRiAP-QGIS/wiki/BCRS) — BCRS algorithm details
- [Axis-Aligned](https://github.com/Wolren/LIRiAP-QGIS/wiki/Axis-Aligned) — Exact axis-aligned solver
- [Complexity](https://github.com/Wolren/LIRiAP-QGIS/wiki/Complexity) — Formal complexity analysis
- [Foundations](https://github.com/Wolren/LIRiAP-QGIS/wiki/Foundations) — Geometric background
- [Parameters](https://github.com/Wolren/LIRiAP-QGIS/wiki/Parameters) — Full parameter reference
- [Folder Layout](https://github.com/Wolren/LIRiAP-QGIS/wiki/Folder-Layout) — Code structure
- [Usage](https://github.com/Wolren/LIRiAP-QGIS/wiki/Usage) — Programmatic API usage