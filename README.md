# LIRiAP

LIRiAP (Largest Inscribed Rectangle in Arbitary Polygon) is a set of QGIS Processing algorithms for computing largest inscribed rectangles for polygon features.

## Problem statement

Given an input polygon, compute a largest-area non axis aligned rectangle. The target geometry includes concave polygons and polygons with holes, with an optional maximum aspect-ratio constraint.

## Uses

- **Suitability analysis task scenarios**: search candidate locations for building or infrastructure placement by finding the largest feasible rectangular footprint inside constrained parcels (e.g., houses, warehouses, solar arrays, staging pads, retention structures) while respecting parcel boundaries and holes/exclusions.
- **Remote sensing scenarios**: derive stable interior rectangular patches for spectral sampling, calibration windows, texture statistics, and object-level summaries where centroid or full-polygon sampling is noisy.
- **Dynamic cartographic label placement**: place labels in the largest interior rectangle instead of using only centroid or bounding box, improving readability in concave polygons and polygons with holes.
- **Other scenarios**: map tiling anchors, drone landing-zone preselection, interior ROI extraction for QA workflows, and standardized shape descriptors for downstream analytics.

## Shared components

All algorithms in `LIRiAP_pack` follow the same high-level structure:

1. **Input normalization**: read polygon geometry; for multipolygons, use the largest part.
2. **Angle candidates**: extract likely orientations from polygon edge directions, with a fallback sweep.
3. **Rectangle solve in rotated frame**: solve axis-aligned rectangle candidates on a rotated polygon to recover non axis aligned solutions in map coordinates.
4. **Refinement and checks**: apply finer search and containment-related adjustments (depending on variant).
5. **Output**: write rectangle geometry and metrics (area, angle, ratio, and variant-specific diagnostics).

## Algorithms

| Algorithm | Main steps | Approach |
| --- | --- | --- |
| Approximation Standard | Coarse candidate evaluation -> local angle refinement -> fine solve | Fast approximate search, tuned for stable quality |
| Approximation Fast | Same steps as Approximation Standard, executed with lower-overhead slice workers | Same approximation method with runtime optimizations |
| Contained Standard | Top-K candidates -> angle polish -> fine solve -> containment certification (with optional fallback) | Certified containment workflow |
| Contained Fast | Same steps as Contained Standard with optimized execution | Certified containment with faster batch execution |
| BCRS (Boundary-Coordinate Raster Solve) | Candidate generation -> angle polish -> boundary-coordinate raster solve -> clamped coordinate-ascent boundary fitting -> certification | Boundary-coordinate method for stronger fit on straight-sided polygons |
| BCRS Fast (Boundary-Coordinate Raster Solve, optimized) | Same BCRS/CABF pipeline with prioritized trials and runtime parallel optimizations | Faster BCRS path for larger datasets |

## Detailed algorithm breakdown

### Approximation Standard

1. Prepare geometry and keep the largest polygon component for multipolygons.
2. Generate orientation candidates from boundary edge directions.
3. Compute a cheap area upper bound per angle and skip weak candidates early.
4. Run coarse grid search on rotated geometry to get the current best rectangle.
5. If candidates are weak or ambiguous, run fallback uniform sweep by `ANGLE_STEP`.
6. Refine around the current best angle with bounded scalar optimization.
7. Recompute at fine grid resolution and rotate rectangle back to map orientation.
8. Apply optional containment buffer and export area/angle/ratio.

### Approximation Fast

1. Uses the same geometric search logic as Approximation Standard.
2. Executes work as index slices (`process_slice`) to reduce per-feature overhead.
3. Preserves identical output fields while improving throughput on larger batches.

### Contained Standard

1. Stage 1: edge-guided coarse search produces top-K candidate angles.
2. Stage 2: local angle polishing around each candidate.
3. Stage 3: fine-grid solve at polished and original angles.
4. Stage 4: explicit containment certification with symmetric shrink if required.
5. If strict certification fails and `ALWAYS_RETURN` is enabled, apply best-effort shrink fallback.
6. Optionally apply user buffer and export diagnostics (`cand_rank`, `s2_gain`, `best_effort`).

### Contained Fast

1. Uses the same Stage 1-4 contained workflow as Contained Standard.
2. Uses optimized execution and chunk/slice processing to scale over many features.
3. Keeps containment guarantees and diagnostics behavior aligned with Standard.

### BCRS (novel method)

1. **Stage 1 (geometry preparation)**: validate geometry, normalize multipart inputs, optional precision snapping.
2. **Stage 2 (heuristic candidates)**:
   - edge-orientation histogram proposes angles,
   - convex-hull upper bound prunes weak directions,
   - coarse grid search keeps top-K candidates.
3. **Stage 3 (angle refinement)**: bounded Brent optimization around each Stage 2 angle.
4. **Stage 4 (Boundary-Coordinate Raster Solve, BCRS)**:
   - rotate polygon to test angle,
   - create boundary-coordinate grid from polygon vertex x/y values,
   - run variable-pitch histogram solver to get best axis-aligned rectangle at that angle.
5. **Stage 5 (Coordinate-Ascent Boundary Fitting, CABF, expansion)**:
   - expand each side by coordinate-ascent binary search,
   - clamp expansion to nearest boundary coordinates to avoid floating-point overreach.
6. **Stage 6 (containment certification)**:
   - verify full containment in original polygon frame,
   - apply controlled shrink when needed,
   - optional best-effort fallback if strict certification fails and fallback is enabled.
7. **Stage 7 (selection and output)**: keep best certified candidate, compute ratio/gain/best-effort metadata, and return rectangle.

### BCRS Fast

1. Uses the same BCRS Stage 1-7 geometry logic.
2. Adds trial ranking and limits expensive Stage 4-5 runs to strongest nearby angles.
3. Reuses Stage 3 area cache and optimized arrays to reduce repeated computation.
4. Keeps the same containment, fallback, and output semantics as BCRS Standard.

## Folder layout

- `LIRiAP_pack/*_algorithm.py`: QGIS Processing wrappers (parameters, execution, output fields, help text).
- `LIRiAP_pack/*_worker.py`: geometry solvers independent from QGIS/Qt runtime.
- `LIRiAP_pack/numba_bootstrap.py`: optional Numba bootstrap helper.
- `LIRiAP_pack/help_descriptions.py`: shared right-panel algorithm descriptions.
