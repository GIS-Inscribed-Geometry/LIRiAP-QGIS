# Axis-Aligned LIR

The Axis-Aligned LIR algorithm solves the Largest Inscribed Rectangle problem under a fixed-axis (axis-aligned) orientation constraint. Unlike other LIRiAP families, this provides an **exact** solution at vertex-coordinate precision.

## Overview

- **Exactness**: Guaranteed optimal rectangle for the given axis orientation
- **Complexity**: O(n²) in polygon vertex count
- **Orientation**: Fixed axis (or rotated via AXIS_ANGLE parameter)
- **Supports**: Convex, concave, with holes, without holes

## Algorithm Flow

```mermaid
flowchart TD
    A[Input polygon] --> B[Detect polygon type]
    B --> C{Convex, no holes?}
    C -- Yes --> D[Alt/Amenta O(n²) solver]
    C -- No --> E[Daniels et al. vertex-grid solver]
    D --> F[Epsilon-inset certification]
    E --> F
    F --> G[Apply max_ratio constraint]
    G --> H[Optional axis rotation]
    H --> I[Output exact rectangle]
```

## Exact Solvers

The algorithm dispatches to one of four exact solvers based on polygon topology:

| Polygon Type | Solver | Complexity | Guarantee |
|--------------|--------|------------|----------|
| Convex, no holes | Alt/Amenta | O(n²) | Exact, vertex-pair enumeration |
| Convex with holes | Daniels et al. | O(n²) | Exact, vertex-coordinate grid |
| Concave, no holes | Daniels et al. | O(n²) | Exact, vertex-coordinate grid |
| Concave with holes | Daniels et al. | O(n²) | Exact, vertex-coordinate grid |

### Alt/Amenta Solver (Convex No Holes)

For convex polygons, the optimal axis-aligned rectangle has its bottom and top sides tangent to the polygon boundary, with each contact at a vertex y-coordinate.

- Enumerate all O(n²) pairs of vertex y-coordinates as (y_lo, y_hi) candidates
- Compute horizontal polygon chord intersection at each y level
- Take maximum-width rectangle

**Reference**: Alt, H. et al. (1987); Amenta, N. (1994)

### Daniels et al. Solver (All Other Cases)

Daniels et al. (1997) proved that the largest axis-aligned rectangle inscribed in a simple polygon always has at least two sides determined by vertex x- or y-coordinates.

- Build vertex-coordinate grid from all unique vertex x/y values
- Create cell-centre PIP (point-in-polygon) mask
- Run Largest Rectangle in Histogram (LRH) stack sweep row-by-row
- Returns exact answer at vertex-coordinate precision

**References**:
- Daniels, K., Milenkovic, V., Roth, D. (1997). Finding the largest axis-aligned rectangle in a polygon. Proc. 13th CCCG.
- Klingel, E. (1986). Finding the largest rectangle in a polygon (LRH).

## Vertex-Grid Midpoint Augmentation

The solver augments raw vertex coordinates with midpoints between consecutive unique values on each axis. This doubles grid density and handles cases where optimal rectangle interior lies between two vertex coordinates (e.g., right triangles, regular hexagons).

Without augmentation, the single cell center can land exactly on the polygon boundary, causing Shapely's strict `contains` to return False for a cell that is geometrically inside.

## Rotation Axis

The AXIS_ANGLE parameter (degrees) rotates the frame in which "axis-aligned" is interpreted:
- Rotate polygon by -axis_angle before solving
- Solve exact axis-aligned problem in rotated frame
- Rotate result rectangle back by +axis_angle

This provides exact solutions for tilted-but-rectangular inscribed rectangles.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| AXIS_ANGLE | Rotation of axis-aligned frame (0 = horizontal) | 0.0 |
| GRID_FINE | Fallback grid resolution (used when vertex density > 500) | 120 |
| MAX_RATIO | Maximum aspect ratio (long:short), 0 = unlimited | 1.6 |
| ALWAYS_RETURN | Return best-effort if epsilon-inset fails | True |
| USE_BUFFER | Apply containment buffer after certification | False |
| BUFFER_VALUE | Buffer distance in map units (negative = inward) | -0.5 |

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| feat_id | int | Source feature ID |
| area | double | Rectangle area in CRS map units |
| axis_angle | double | Axis angle used (echoes input parameter) |
| poly_type | string | convex_no_holes / convex_with_holes / concave_no_holes / concave_with_holes |
| ratio | double | Actual long:short aspect ratio |
| best_effort | int | 1 if result from shrink fallback, 0 otherwise |

## Semantics

- **Exact**: Rectangle sides snap exactly to vertex coordinates (or midpoint-augmented grid lines)
- **Containment**: Guaranteed up to floating-point epsilon via light-weight inset certification
- **No expansion needed**: Unlike BCRS, exact vertex-coordinate precision means no CABF expansion is required

## Performance

The algorithm is O(n²) in vertex count, making it suitable for polygons with typical vertex counts (<1000). For very high-density polygons, the fallback grid solver is used (>500 unique vertices triggers grid fallback).

## References

1. Alt, H., Hagerup, T., Mehlhorn, K., Preparata, F.P. (1987). Deterministic simulation of idealized parallel computers. Information and Computation.
2. Amenta, N. (1994). A short proof of an interesting heuristic result. Proc. 5th ACM-SIAM SODA.
3. Daniels, K., Milenkovic, V., Roth, D. (1997). Finding the largest axis-aligned rectangle in a polygon. Proc. 13th Canadian Conf. Computational Geometry.
4. Klingel, E. (1986). Finding the largest rectangle in a polygon.