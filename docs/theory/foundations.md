# Geometric Foundations

Background on the Largest Inscribed Rectangle problem and the geometric principles underlying LIRiAP algorithms.

## Problem Definition

Given a simple polygon P (possibly with holes), find the rectangle R of maximum area such that R is strictly contained in P. The rectangle may be rotated (non-axis-aligned).

### Variants

1. **Axis-aligned**: Rectangle sides parallel to coordinate axes
2. **Rotated**: Rectangle may have any orientation
3. **With constraints**: Maximum aspect ratio, minimum area, etc.

## Theoretical Background

### Axis-Aligned Case

For axis-aligned rectangles, Daniels et al. (1997) proved that the optimal rectangle always has at least two sides determined by vertex coordinates of the polygon. This enables efficient exact solvers using vertex-coordinate grids.

### Rotated Case

The general (rotated) case is NP-hard to solve exactly. Approaches include:
- Heuristic search over orientation space
- Grid-based approximation
- Boundary-coordinate raster methods (BCRS)

## Key Algorithms and References

### Largest Rectangle in Histogram (LRH)

The classic stack-based algorithm for finding the largest rectangle in a histogram. Runs in O(n) time.

**Reference**: Klingel, E. (1986). Finding the largest rectangle in a polygon.

### Alt/Amenta (Convex Polygons)

For convex polygons, the optimal axis-aligned rectangle has its top and bottom edges tangent at polygon vertices.

**References**:
- Alt, H., Hagerup, T., Mehlhorn, K., Preparata, F.P. (1987). Deterministic simulation of idealized parallel computers. Information and Computation.
- Amenta, N. (1994). A short proof of an interesting heuristic result. Proc. 5th ACM-SIAM SODA.

### Daniels et al. (General Polygons)

Theorem: The largest axis-aligned rectangle in a simple polygon always has at least two sides determined by vertex x- or y-coordinates.

**Reference**: Daniels, K., Milenkovic, V., Roth, D. (1997). Finding the largest axis-aligned rectangle in a polygon. Proc. 13th Canadian Conf. Computational Geometry.

### Brent Optimization

Bounded scalar optimization for angle refinement.

**Reference**: Brent, R.P. (1973). Algorithms for Finding Zeros and Extrema of Functions Without Calculating Derivatives.

### Computational Geometry Foundations

**References**:
- Bentley, J.L. (1977). Programming Pearls: Fast Algorithms for Polygon Containment.
- Preparata, F.P., Shamos, M.I. (1985). Computational Geometry: An Introduction.

## Algorithm Families in LIRiAP

### Approximation Family
- Heuristic edge-direction analysis
- Grid-based search with refinement
- No containment guarantee

### Contained Family
- Edge-guided candidate generation
- Coarse-to-fine grid search
- Explicit containment certification
- Best-effort fallback when strict fails

### BCRS Family
- Boundary-coordinate raster solve (BCRS)
- Uses polygon vertex coordinates as grid lines
- CABF expansion for boundary fitting
- Novel contributions in variable-pitch LRH and coordinate-ascent fitting

### Axis-Aligned LIR
- Exact solvers for fixed-orientation
- Alt/Amenta for convex polygons
- Daniels et al. vertex-grid for general polygons
- Vertex-coordinate precision

## Practical Considerations

### Polygon Complexity
- Low vertex count (<100): Exact methods feasible
- Medium vertex count (100-1000): Grid-based methods appropriate
- High vertex count (>1000): Need downsampling or approximation

### Containment Guarantees
- Approximation: No guarantee
- Contained: Strict when certification passes
- BCRS: Strict when certification passes
- Axis-Aligned: Exact (vertex-coordinate precision)

### Performance Tradeoffs
- Speed vs. accuracy
- Strictness vs. always-return
- Single-threaded vs. parallel