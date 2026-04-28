# LIRiAP Documentation

LIRiAP (Largest Inscribed Rectangle in Arbitrary Polygon) provides QGIS Processing algorithms for computing largest inscribed rectangles in polygon features.

## Problem Statement

Given an input polygon, find a large non-axis-aligned interior rectangle. Four problem variants are implemented:

| Family | Primary Objective | Strict Containment | Boundary Expansion |
|--------|-------------------|-------------------|-------------------|
| Approximation | Fast area-focused search | No | No |
| Contained | Certified contained rectangle | Yes (unless fallback enabled) | No |
| BCRS | Certified + fit improvement | Yes (unless fallback enabled) | Yes (CABF) |
| Axis-Aligned | Exact fixed-axis solve | Exact (vertex-coordinate) | N/A |

## Algorithm Overview

| Algorithm | Problem | Containment | Expansion |
|-----------|---------|-------------|----------|
| Approximation Standard | Fast approximation | Not certified | No |
| Approximation Fast | Same as Standard, lower overhead | Not certified | No |
| Contained Standard | Certified search | Certified / best-effort | No |
| Contained Fast | Optimized Contained | Certified / best-effort | No |
| BCRS | Full contained + expansion | Certified / best-effort | CABF |
| BCRS Fast | Optimized BCRS | Certified / best-effort | CABF |
| Axis-Aligned LIR | Exact fixed-axis solve | Exact (vertex-coordinate) | N/A |

## Documentation Sections

### Algorithms
- [Overview](./algorithms/overview.md) — Family comparison table
- [Approximation](./algorithms/approximation.md) — Approx Standard/Fast with flowchart
- [Contained](./algorithms/contained.md) — Contained Standard/Fast with flowchart
- [BCRS](./algorithms/bcrs.md) — BCRS/CABF Standard/Fast with flowchart
- [Axis-Aligned](./algorithms/axis-aligned.md) — Exact solvers for fixed-axis

### Theory
- [Complexity Analysis](./theory/complexity.md) — Formal complexity analysis
- [Foundations](./theory/foundations.md) — Geometric background and academic references

### Reference
- [Parameters](./reference/parameters.md) — Complete parameter reference
- [Folder Layout](./reference/folder-layout.md) — Code structure
- [Usage](./reference/usage.md) — Programmatic API usage

## Quick Start

1. Install LIRiAP in QGIS
2. Open Processing Toolbox → LIRiAP
3. Select algorithm family based on your needs:
   - Approximation: quick candidates
   - Contained: guaranteed containment
   - BCRS: best accuracy with boundary fitting
   - Axis-Aligned: exact fixed-orientation solution

## Performance Notes

- Approximation Fast with multithreading: best throughput for large datasets
- Axis-Aligned LIR: fastest single-threaded for fixed-axis problems (exact solution)
- BCRS (single-threaded): best accuracy for rotated rectangles
- See [Complexity Analysis](./theory/complexity.md) for detailed performance characteristics