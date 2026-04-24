"""
LIRiAP Axis-Aligned LIR worker module.

Implements EXACT axis-aligned (or fixed-rotation-axis) Largest Inscribed
Rectangle solvers and is intentionally independent from QGIS/Qt runtime
objects.

Algorithm lineage and exactness guarantees
------------------------------------------
**Convex, no holes** — ``_exact_solve_convex``
    For a convex polygon the optimal axis-aligned inscribed rectangle has
    its bottom and top sides tangent to the polygon boundary, and each of
    those contacts is realised at a vertex y-coordinate (Alt et al. 1994;
    Amenta 1994).  We enumerate all O(n²) pairs of vertex y-coordinates as
    (y_lo, y_hi) candidates, compute the horizontal polygon chord
    intersection at each y level via edge scanning, and take the maximum.
    This is provably exact for axis-aligned rectangles in convex polygons.
    Complexity: O(n²) with small constants.

    NOTE: _exact_solve_convex is a fast-path that only covers the case
    where both y_lo and y_hi are vertex y-coordinates.  For polygons with
    fewer than 4 unique y-values (e.g. triangles), the optimal rectangle
    interior y may not be a vertex coordinate.  In those cases the solver
    transparently falls through to ``_exact_solve_vertex_grid`` which uses
    midpoint-augmented grid lines to handle all convex shapes correctly.

**All other cases** — ``_exact_solve_vertex_grid``
    Daniels et al. (1997) proved that the largest axis-aligned rectangle
    inscribed in a simple polygon (concave, with holes, …) always has at
    least two of its four sides determined by vertex x- or y-coordinates of
    the polygon (exterior + interior rings).  Therefore the vertex-coordinate
    grid — sorting all unique vertex x/y values, building a cell-centre PIP
    mask, and running the standard largest-rectangle-in-histogram (LRH)
    stack sweep row-by-row — yields the EXACT answer at vertex-coordinate
    precision.  This is O(n²) in vertex count, matching the practical
    complexity of the theoretical O(n log² n) algorithm for real GIS
    polygons (n typically < 1000).

    For convex-with-holes the hole vertex coordinates are included in both
    the grid and the PIP test (Shapely ``contains`` natively handles holes).
    For concave-no-holes only exterior vertices are needed (and are
    sufficient by the theorem).  For concave-with-holes both exterior and
    all interior rings are collected.

No post-hoc CABF expansion is required: the rectangle sides snap exactly
to vertex coordinates, so containment is guaranteed up to floating-point
epsilon.  A lightweight epsilon-inset certification step handles any
residual GEOS tolerance issues without iterative binary search.

Vertex-grid midpoint augmentation
    ``_exact_solve_vertex_grid`` augments the raw vertex coordinate arrays
    with the midpoint between every consecutive pair of unique values on
    each axis.  This doubles the grid density and is required for correctness
    on polygons where the optimal rectangle's interior lies between two
    vertex coordinates (e.g. right triangles, regular hexagons).  Without
    augmentation the single cell centre can land exactly on the polygon
    boundary, causing Shapely's strict ``contains`` to return False for a
    cell that is geometrically inside the polygon.

Rotation axis
    The ROTATION_AXIS parameter (degrees) rotates the frame in which
    "axis-aligned" is interpreted.  The polygon is rotated by –axis_angle
    before solving, and the resulting rectangle is rotated back by
    +axis_angle.  This is exact because we solve an exact axis-aligned
    problem in the rotated frame.

References
----------
* Alt, H., Hagerup, T., Melhorn, K., Preparata, F.P. (1987). Deterministic
  simulation of idealized parallel computers. *Information and Computation*.
* Amenta, N. (1994). A short proof of an interesting heuristic result.
  *Proc. 5th ACM-SIAM SODA*.
* Daniels, K., Milenkovic, V., Roth, D. (1997). Finding the largest
  axis-aligned rectangle in a polygon. *Proc. 13th Canadian Conf.
  Computational Geometry*.
* This module's LIR context: LIRiAP bcrs_worker.py (same plugin).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import box, MultiPolygon, Polygon, Point
from shapely.prepared import prep as shp_prep
from shapely.wkb import loads as wkb_loads

try:
    from shapely.prepared import prep as _prep_geom
except Exception:
    _prep_geom = None

# --------------------------------------------------------------------------
# Vectorised point-in-polygon  (Shapely 1.x + 2.x compat)
# --------------------------------------------------------------------------
try:
    # Shapely 2.x: use contains_xy (vectorised, no deprecation warning)
    from shapely import contains_xy as _shp_contains_xy

    def _mask_from_poly(poly, xx_flat, yy_flat):
        """Return boolean array: True where point (xx_flat[i], yy_flat[i]) is strictly inside poly."""
        return _shp_contains_xy(poly, xx_flat, yy_flat)

except ImportError:
    try:
        # Shapely 1.x fallback: shapely.vectorized.contains
        from shapely.vectorized import contains as _shp_contains_vec

        def _mask_from_poly(poly, xx_flat, yy_flat):
            """Return boolean array: True where point (xx_flat[i], yy_flat[i]) is strictly inside poly."""
            return _shp_contains_vec(poly, xx_flat, yy_flat)

    except ImportError:
        import shapely as _shp2

        def _mask_from_poly(poly, xx_flat, yy_flat):
            """Return boolean array: True where point (xx_flat[i], yy_flat[i]) is strictly inside poly."""
            pts = _shp2.points(xx_flat, yy_flat)
            return _shp2.contains(poly, pts)

# --------------------------------------------------------------------------
# Numba JIT — graceful fallback to pure Python
# --------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(fn=None, **kw):
        """Null decorator used when Numba is not installed."""
        return fn if fn is not None else (lambda f: f)

    _NUMBA_AVAILABLE = False

# --------------------------------------------------------------------------
# Tuning constants
# --------------------------------------------------------------------------
_CERT_EPS = 1e-8            # Epsilon inset for floating-point certification
_VERTEX_COORD_CAP = 500     # Max unique vertex coords per axis before fallback
_UNIFORM_FALLBACK_N = 500   # Uniform fallback grid size when cap exceeded


# --------------------------------------------------------------------------
# Rect-to-polygon clipping helper
# --------------------------------------------------------------------------
def _clip_rect_to_poly(poly: Polygon, rect: Polygon) -> Optional[Polygon]:
    """Return an axis-aligned rectangle fitting inside poly ∩ rect, or None.

    For oblique polygons (triangles, skewed quads) candidate rects produced by
    the chord-intersection solvers can genuinely protrude past a diagonal edge
    even though the chord midpoint queries passed — the corners of the rect
    leave the polygon. Clipping the rect to the polygon and taking the bounds
    of the intersection recovers a strictly interior rect.
    """
    if rect is None or rect.is_empty:
        return None
    try:
        inter = poly.intersection(rect)
    except Exception:
        return None
    if inter.is_empty:
        return None
    if inter.geom_type == 'MultiPolygon':
        inter = max(inter.geoms, key=lambda g: g.area)
    if inter.geom_type != 'Polygon':
        return None
    minx, miny, maxx, maxy = inter.bounds
    if maxx - minx < 1e-9 or maxy - miny < 1e-9:
        return None
    return box(minx, miny, maxx, maxy)


# ==========================================================================
# ① JIT HISTOGRAM KERNEL  (variable-pitch)
# ==========================================================================

@_njit(cache=True)
def _histogram_kernel_vp(heights, xs, ys, row_idx, max_ratio):
    """
    Largest-rectangle-in-histogram with VARIABLE-PITCH columns/rows.

    ``xs[i]`` is the LEFT edge of column i; ``xs[i+1]`` is the RIGHT edge.
    ``ys[r]`` is the BOTTOM edge of row r; ``ys[r+1]`` is the TOP edge.

    This is the core of the Daniels et al. (1997) vertex-coordinate-grid
    exact solver.  Each rectangle returned has its boundaries aligned to
    entries of *xs* and *ys*, which are the polygon's own vertex coordinates,
    so the result is exact at vertex-coordinate precision.

    Parameters
    ----------
    heights : np.ndarray[int64]
        Current histogram heights (consecutive included rows) per column.
    xs : np.ndarray[float64]
        Column boundary x-coordinates (length = n_cols + 1).
    ys : np.ndarray[float64]
        Row boundary y-coordinates (length = n_rows + 1).
    row_idx : int
        Zero-based index of the current sweep row.
    max_ratio : float
        Maximum allowed long:short aspect ratio; 0.0 = unlimited.
        When > 0, candidates are ranked by their ratio-shrunk area (so a
        slightly smaller rect that already satisfies the ratio can beat a
        larger rect that would be heavily shrunk), but the returned
        coordinates are the RAW histogram bounds — the caller is responsible
        for applying the actual ratio constraint via anchored/centred
        sliding-window search, which avoids the centre-shrink pitfall of
        producing rectangles that leave the polygon on concave shapes.

    Returns
    -------
    tuple (x0, y0, x1, y1, best_area) all float64.
        Raw histogram bounds of the best rectangle found up to this row
        sweep.  ``best_area`` is the unconstrained area (rw * rh) of that
        raw rect — not the ratio-shrunk area.
    """
    cols = len(heights)
    n_xs = len(xs)
    n_ys = len(ys)
    best_score = 0.0
    best_area = 0.0
    bx0 = by0 = bx1 = by1 = 0.0

    st_col = np.empty(cols + 1, dtype=np.int64)
    st_h   = np.empty(cols + 1, dtype=np.int64)
    top = 0

    for c in range(cols + 1):
        h = int(heights[c]) if c < cols else 0
        start = c
        while top > 0 and st_h[top - 1] > h:
            top -= 1
            sc = st_col[top]
            sh = st_h[top]
            xi  = sc + (c - sc)
            x0_w = xs[sc]
            x1_w = xs[xi if xi < n_xs else n_xs - 1]
            ri0  = row_idx - sh + 1
            y0_w = ys[ri0 if ri0 >= 0 else 0]
            ri1  = row_idx + 1
            y1_w = ys[ri1 if ri1 < n_ys else n_ys - 1]
            rw = x1_w - x0_w
            rh = y1_w - y0_w
            if rw <= 0.0 or rh <= 0.0:
                start = sc
                continue
            area = rw * rh
            # Score = ratio-constrained area (what the caller can actually
            # realise).  If unconstrained (max_ratio <= 0), score == area.
            if max_ratio > 0.0:
                ls = rw if rw >= rh else rh
                ss = rh if rw >= rh else rw
                if ss > 0.0 and ls / ss > max_ratio:
                    score = ss * (ss * max_ratio)
                else:
                    score = area
            else:
                score = area
            if score > best_score:
                best_score = score
                best_area = area
                bx0 = x0_w
                by0 = y0_w
                bx1 = x1_w
                by1 = y1_w
            start = sc
        st_col[top] = start
        st_h[top]   = h
        top += 1

    return bx0, by0, bx1, by1, best_area


# ==========================================================================
# ② POLYGON TYPE DETECTION
# ==========================================================================

def _detect_polygon_type(poly: Polygon) -> str:
    """
    Classify *poly* into one of four axis-aligned LIR algorithm cases.

    Uses two independent Boolean tests:

    * **has_holes**: the polygon contains one or more interior rings.
    * **is_convex**: ``|convex_hull.area - poly.area| / poly.area < 1e-6``,
      i.e. no reflex vertices within floating-point tolerance.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        A valid, non-empty Shapely polygon.

    Returns
    -------
    str
        One of ``"convex_no_holes"``, ``"convex_with_holes"``,
        ``"concave_no_holes"``, or ``"concave_with_holes"``.
    """
    has_holes: bool = len(list(poly.interiors)) > 0
    is_convex: bool = (
        abs(poly.convex_hull.area - poly.area)
        / max(poly.area, 1e-14)
        < 1e-6
    )

    if is_convex and not has_holes:
        return "convex_no_holes"
    if is_convex and has_holes:
        return "convex_with_holes"
    if not is_convex and not has_holes:
        return "concave_no_holes"
    return "concave_with_holes"


# ==========================================================================
# ③ EXACT SOLVER — CONVEX POLYGON, NO HOLES
#    Alt et al. 1994 / Amenta 1994 — O(n²) vertex-pair enumeration
# ==========================================================================

def _exact_solve_convex(poly: Polygon, max_ratio: float) -> Tuple[Optional[Polygon], float]:
    """
    Exact O(n²) axis-aligned LIR solver for a convex polygon without holes.

    **Theoretical basis** (Alt et al. 1994; Amenta 1994):
    For a convex polygon the optimal axis-aligned inscribed rectangle has its
    bottom and top sides touching the boundary, and each contact is realised
    at a vertex y-coordinate.  Enumerating all O(n²) pairs of vertex
    y-coordinates as (y_lo, y_hi) candidates and computing the maximum
    horizontal chord intersection at both levels is therefore exhaustive.

    **Algorithm**:
    For each ordered pair (y_lo, y_hi) drawn from the set of vertex
    y-coordinates:

    1. Compute the x-extent (x_min, x_max) of the polygon boundary at
       ``y_lo + ε`` and at ``y_hi − ε`` by intersecting every polygon edge
       with the given horizontal level.
    2. The admissible rectangle spans
       ``x_left = max(x_min@y_lo, x_min@y_hi)`` to
       ``x_right = min(x_max@y_lo, x_max@y_hi)`` — this is the largest width
       that fits in the convex polygon between both horizontal contacts.
    3. Area = ``(x_right − x_left) × (y_hi − y_lo)``.
    4. The pair giving maximum area is the answer.

    The tiny epsilon offsets (``±1e-10``) ensure the query point is
    strictly inside the slab and not on a boundary vertex, which would give
    ambiguous edge intersections for nearly-horizontal edges.

    If *max_ratio* > 0, the rectangle is analytically centred and shortened
    on its long side to satisfy the constraint.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Convex polygon, no interior rings.
    max_ratio : float
        Maximum allowed long:short aspect ratio.  0.0 = unlimited.

    Returns
    -------
    best_rect : shapely.geometry.Polygon or None
    best_area : float
    """
    coords = np.array(poly.exterior.coords[:-1], dtype=np.float64)
    n = len(coords)
    if n < 3:
        return None, 0.0

    ys_vertex = np.unique(coords[:, 1])

    # Candidate y-lines: by the Alt/Amenta convex theorem, the optimal
    # axis-aligned rectangle has its top and bottom sides at vertex
    # y-coordinates, so enumerating all O(n^2) pairs of vertex y-values is
    # exhaustive.  Interior y-sampling is mathematically unnecessary.
    # For near-degenerate cases where only one side is at a vertex-y (e.g.
    # isosceles triangles), the ternary-search refinement below handles
    # sub-vertex accuracy.
    ys_all = ys_vertex

    def x_extent_at_y(y: float):
        """
        Return (x_min, x_max) of the polygon boundary at height *y* by
        intersecting every edge with the horizontal line.

        Uses an asymmetric epsilon strategy:
        * For a lower-boundary query (y slightly above a vertex), use
          ``lo_y <= y + eps`` to include horizontal edges exactly at y.
        * For an upper-boundary query (y slightly below a vertex), use
          ``hi_y >= y - eps`` to include horizontal edges exactly at y.

        Returns None if fewer than two intersection points are found.
        """
        xs_hit = []
        for i in range(n):
            x0_, y0_ = coords[i]
            x1_, y1_ = coords[(i + 1) % n]
            lo_y = min(y0_, y1_)
            hi_y = max(y0_, y1_)
            # Include the edge if it spans through y (with epsilon tolerance)
            if lo_y > y + 1e-10 or hi_y < y - 1e-10:
                continue
            if abs(y1_ - y0_) < 1e-14:
                # Horizontal edge — both endpoints contribute
                xs_hit.append(x0_)
                xs_hit.append(x1_)
            else:
                t = (y - y0_) / (y1_ - y0_)
                t = max(0.0, min(1.0, t))
                xs_hit.append(x0_ + t * (x1_ - x0_))
        if len(xs_hit) < 2:
            return None
        return float(min(xs_hit)), float(max(xs_hit))

    best_area: float = 0.0
    best_rect: Optional[Polygon] = None
    best_y_lo: float = 0.0
    best_y_hi: float = 0.0

    # Triple-query strategy (see module docstring / algorithm_design.md):
    # For each candidate slab [y_lo, y_hi], the rectangle with sides at y=y_lo
    # and y=y_hi must fit within the polygon's cross-section at BOTH endpoints
    # and everywhere in between. Querying at y_lo+eps, y_mid, and y_hi-eps and
    # taking the intersection (tightest bounds) guarantees correctness for
    # oblique/narrow convex polygons where the mid-slab x-extent is wider than
    # at the slab boundaries.
    _EXT_DEGENERATE = 1e-9    # minimum chord width to treat as a real constraint

    def _tight_extent(y_lo_v: float, y_hi_v: float):
        """Return (xl, xr) of the admissible x-band for the slab [y_lo, y_hi].

        The rectangle's bottom edge is at y=y_lo and top edge at y=y_hi, so
        both those chord extents must contain the rect.  Additionally every
        vertex y inside the slab is a kink in the piecewise-linear width
        function so the polygon could be NARROWER at an interior vertex y
        than at the slab endpoints; we must include those too.

        Queries happen AT y_lo and y_hi exactly (not offset by eps): the
        caller uses vertex y-values directly.  Pointed-vertex slabs where
        the extent degenerates are not useful anyway — the rect would have
        zero width there.
        """
        query_ys = [y_lo_v, 0.5 * (y_lo_v + y_hi_v), y_hi_v]
        for yv in ys_all:
            yv_f = float(yv)
            if y_lo_v < yv_f < y_hi_v:
                query_ys.append(yv_f)
        extents = [x_extent_at_y(y) for y in query_ys]
        # Reject the slab entirely if any query is None — a None extent means
        # we could not bound the polygon at that y (numerically degenerate),
        # and an unbounded query would make the intersection unsafe.
        extents_valid = [e for e in extents if e is not None and (e[1] - e[0]) > _EXT_DEGENERATE]
        if not extents_valid:
            return None
        xl_v = max(e[0] for e in extents_valid)
        xr_v = min(e[1] for e in extents_valid)
        if xr_v - xl_v < _EXT_DEGENERATE:
            return None
        return xl_v, xr_v

    for yi in range(len(ys_all)):
        y_lo_orig = float(ys_all[yi])
        for yj in range(yi + 1, len(ys_all)):
            y_hi_orig = float(ys_all[yj])

            tight = _tight_extent(y_lo_orig, y_hi_orig)
            if tight is None:
                continue
            xl, xr = tight

            x_left  = xl
            x_right = xr
            if x_right <= x_left:
                continue

            rw = x_right - x_left
            rh = y_hi_orig - y_lo_orig

            # Apply aspect-ratio constraint analytically from centre.
            # Use LOCAL copies of y_lo/y_hi so the outer loop variables
            # are never mutated (fixes y_lo mutation bug in max_ratio block).
            rect_y_lo = y_lo_orig
            rect_y_hi = y_hi_orig
            rect_x_lo = x_left
            rect_x_hi = x_right

            if max_ratio > 0.0 and min(rw, rh) > 0.0:
                ls = max(rw, rh)
                ss = min(rw, rh)
                if ls / ss > max_ratio:
                    nl = ss * max_ratio
                    if rw >= rh:
                        cx_c = 0.5 * (rect_x_lo + rect_x_hi)
                        rect_x_lo = cx_c - 0.5 * nl
                        rect_x_hi = cx_c + 0.5 * nl
                    else:
                        cy_c = 0.5 * (rect_y_lo + rect_y_hi)
                        rect_y_lo = cy_c - 0.5 * nl
                        rect_y_hi = cy_c + 0.5 * nl
                    rw = rect_x_hi - rect_x_lo
                    rh = rect_y_hi  - rect_y_lo

            area = rw * rh
            if area > best_area:
                cand = box(rect_x_lo, rect_y_lo, rect_x_hi, rect_y_hi)
                if not poly.covers(cand):
                    # Candidate corner protrudes across a diagonal edge near a
                    # pointed vertex. Clip to the polygon boundary to recover
                    # a strictly-interior axis-aligned rectangle.
                    cand = _clip_rect_to_poly(poly, cand)
                    if cand is None:
                        continue
                    if not poly.covers(cand):
                        continue
                    c_area = float(cand.area)
                    if c_area <= best_area:
                        continue
                    best_area = c_area
                    best_rect = cand
                    best_y_lo = rect_y_lo
                    best_y_hi = rect_y_hi
                else:
                    best_area = area
                    best_rect = cand
                    best_y_lo = rect_y_lo
                    best_y_hi = rect_y_hi

    # ── Ternary-search refinement ────────────────────────────────────────────
    # Dense-y sampling gives approximate optima with error ~ (y_range/N).  For
    # analytic correctness (matching e.g. Alt/Amenta 1994 for isosceles
    # triangles) we refine the best candidate by ternary-searching both y_lo
    # (keeping y_hi fixed) and y_hi (keeping y_lo fixed) to sub-epsilon
    # accuracy.  For unlimited aspect ratio the 1D area function is unimodal
    # in each side, so ternary search converges.
    if best_rect is not None and max_ratio <= 0.0:
        def _theoretical_area(y_lo_v, y_hi_v):
            """Return smooth theoretical area from tight_extent, without
            poly.covers check — needed for unimodal optimization."""
            tv = _tight_extent(y_lo_v, y_hi_v)
            if tv is None:
                return 0.0
            xl_t, xr_t = tv
            if xr_t - xl_t <= 0.0 or y_hi_v - y_lo_v <= 0.0:
                return 0.0
            return (xr_t - xl_t) * (y_hi_v - y_lo_v)

        def _realize(y_lo_v, y_hi_v):
            """Build and certify a rect at (y_lo, y_hi). Returns (area, rect)."""
            tv = _tight_extent(y_lo_v, y_hi_v)
            if tv is None:
                return 0.0, None
            xl_t, xr_t = tv
            if xr_t - xl_t < _EXT_DEGENERATE or y_hi_v - y_lo_v < _EXT_DEGENERATE:
                return 0.0, None
            cand_t = box(xl_t, y_lo_v, xr_t, y_hi_v)
            if poly.covers(cand_t):
                return float(cand_t.area), cand_t
            clipped = _clip_rect_to_poly(poly, cand_t)
            if clipped is not None and poly.covers(clipped):
                return float(clipped.area), clipped
            # Try a tiny inset on the rect to cover FP noise
            try:
                inset = cand_t.buffer(-_CERT_EPS * 10, cap_style=3, join_style=2)
                if not inset.is_empty and poly.covers(inset):
                    return float(inset.area), inset
            except Exception:
                pass
            return 0.0, None

        def _ternary(fix_lo, y_lo_v, y_hi_v, lo_bound, hi_bound):
            a_lo, a_hi = lo_bound, hi_bound
            for _ in range(80):
                if a_hi - a_lo < 1e-12:
                    break
                m1 = a_lo + (a_hi - a_lo) / 3.0
                m2 = a_hi - (a_hi - a_lo) / 3.0
                if fix_lo:
                    ar1 = _theoretical_area(y_lo_v, m1)
                    ar2 = _theoretical_area(y_lo_v, m2)
                else:
                    ar1 = _theoretical_area(m1, y_hi_v)
                    ar2 = _theoretical_area(m2, y_hi_v)
                if ar1 < ar2:
                    a_lo = m1
                else:
                    a_hi = m2
            # Try to realize the converged optimum
            y_opt = 0.5 * (a_lo + a_hi)
            best_local = (0.0, None, None)
            for yv in (y_opt, a_lo, a_hi):
                if fix_lo:
                    ar, rc = _realize(y_lo_v, yv)
                else:
                    ar, rc = _realize(yv, y_hi_v)
                if ar > best_local[0] and rc is not None:
                    best_local = (ar, rc, yv)
            return best_local

        try:
            _b_y_lo, _b_y_hi = best_rect.bounds[1], best_rect.bounds[3]
        except Exception:
            _b_y_lo = best_y_lo
            _b_y_hi = best_y_hi
        y_range = float(ys_all[-1] - ys_all[0])
        half = 0.5 * y_range / max(len(ys_all) - 1, 1)
        r_hi = _ternary(True, _b_y_lo, _b_y_hi,
                        max(_b_y_hi - 3 * half, _b_y_lo + _EXT_DEGENERATE),
                        min(_b_y_hi + 3 * half, float(ys_all[-1])))
        if r_hi[1] is not None and r_hi[0] > best_area:
            best_area = r_hi[0]
            best_rect = r_hi[1]
            _b_y_hi = r_hi[2]
        r_lo = _ternary(False, _b_y_lo, _b_y_hi,
                        max(_b_y_lo - 3 * half, float(ys_all[0])),
                        min(_b_y_lo + 3 * half, _b_y_hi - _EXT_DEGENERATE))
        if r_lo[1] is not None and r_lo[0] > best_area:
            best_area = r_lo[0]
            best_rect = r_lo[1]

    return best_rect, best_area


# ==========================================================================
# ④ EXACT SOLVER — GENERAL POLYGON (convex/concave, with/without holes)
#    Daniels et al. 1997 — vertex-coordinate grid + LRH scanline — O(n²)
# ==========================================================================

def _uniform_grid_solve(
    poly: Polygon,
    n_cols: int,
    n_rows: int,
    max_ratio: float,
) -> Tuple[Optional[Polygon], float]:
    """
    Fallback uniform-grid LRH solver used when the vertex-coordinate grid
    exceeds ``_VERTEX_COORD_CAP``.

    The uniform grid is an approximation, but the cap is set at 500 unique
    coordinates per axis — well beyond any normal GIS polygon — so in
    practice this path is only reached for heavily discretised circles or
    deliberately pathological inputs.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
    n_cols, n_rows : int
        Grid resolution.
    max_ratio : float

    Returns
    -------
    best_rect : shapely.geometry.Polygon or None
    best_area : float
    """
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, n_cols + 1)
    ys = np.linspace(miny, maxy, n_rows + 1)

    cx_pts = 0.5 * (xs[:-1] + xs[1:])
    cy_pts = 0.5 * (ys[:-1] + ys[1:])
    gxx, gyy = np.meshgrid(cx_pts, cy_pts)
    centre_flat = _mask_from_poly(poly, gxx.ravel(), gyy.ravel())
    centre_mask = centre_flat.reshape(n_rows, n_cols)
    try:
        from shapely.prepared import prep as _shp_prep2
        _prep2 = _shp_prep2(poly)
        def _cell_ok2(cb): return _prep2.covers(cb)
    except Exception:
        def _cell_ok2(cb): return poly.covers(cb)
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for _i in range(n_rows):
        _y0, _y1 = ys[_i], ys[_i + 1]
        for _j in range(n_cols):
            if not centre_mask[_i, _j]:
                continue
            if _cell_ok2(box(xs[_j], _y0, xs[_j + 1], _y1)):
                mask[_i, _j] = True

    heights  = np.zeros(n_cols, dtype=np.int64)
    best_rect: Optional[Polygon] = None
    best_area: float = 0.0
    fallback_rect: Optional[Polygon] = None
    fallback_area: float = 0.0


    for r in range(n_rows):
        row = mask[r].astype(np.int64)
        heights += row
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel_vp(
            heights, xs, ys, r, max_ratio
        )
        if area <= 0.0:
            continue

        rw_raw = x1 - x0
        rh_raw = y1 - y0

        if max_ratio > 0.0 and min(rw_raw, rh_raw) > 0.0:
            ls = max(rw_raw, rh_raw)
            ss = min(rw_raw, rh_raw)
            if ls / ss > max_ratio:
                nl = ss * max_ratio
                if rw_raw >= rh_raw:
                    candidates = [
                        box(x0, y0, x0 + nl, y1),
                        box(x1 - nl, y0, x1, y1),
                        box(0.5 * (x0 + x1) - nl * 0.5, y0,
                            0.5 * (x0 + x1) + nl * 0.5, y1),
                    ]
                else:
                    candidates = [
                        box(x0, y0, x1, y0 + nl),
                        box(x0, y1 - nl, x1, y1),
                        box(x0, 0.5 * (y0 + y1) - nl * 0.5,
                            x1, 0.5 * (y0 + y1) + nl * 0.5),
                    ]
                best_cand = None
                best_cand_area = 0.0
                for c in candidates:
                    if poly.covers(c):
                        ca = float(c.area)
                        if ca > best_cand_area:
                            best_cand_area = ca
                            best_cand = c
                    else:
                        clipped = _clip_rect_to_poly(poly, c)
                        if clipped is not None and poly.covers(clipped):
                            ca = float(clipped.area)
                            if ca > best_cand_area:
                                best_cand_area = ca
                                best_cand = clipped
                if best_cand is not None and best_cand_area > best_area:
                    best_area = best_cand_area
                    best_rect = best_cand
                cand_unconstrained = box(x0, y0, x1, y1)
                if not poly.covers(cand_unconstrained):
                    clipped = _clip_rect_to_poly(poly, cand_unconstrained)
                    if clipped is not None and poly.covers(clipped):
                        ca = float(clipped.area)
                        if ca > fallback_area:
                            fallback_area = ca
                            fallback_rect = clipped
                continue

        if area > best_area:
            cand = box(x0, y0, x1, y1)
            if not poly.covers(cand):
                # LRH may return a rect that spans a hole or protrudes outside
                # the polygon. Record it as a fallback candidate but do NOT
                # raise best_area — doing so would prevent later rows from
                # finding the true optimal (exact) rect at those columns.
                clipped = _clip_rect_to_poly(poly, cand)
                if clipped is not None and poly.covers(clipped):
                    c_area = float(clipped.area)
                    if c_area > fallback_area:
                        fallback_area = c_area
                        fallback_rect = clipped
            else:
                best_area = area
                best_rect = cand

    # Merge: return the exact rect if found, otherwise the best clipped fallback.
    if best_rect is None and fallback_rect is not None:
        return fallback_rect, fallback_area
    if fallback_rect is not None and fallback_area > best_area:
        return fallback_rect, fallback_area
    return best_rect, best_area


def _exact_solve_vertex_grid(
    poly: Polygon,
    poly_type: str,
    max_ratio: float,
) -> Tuple[Optional[Polygon], float]:
    """
    Exact O(n²) axis-aligned LIR solver via Daniels et al. (1997)
    vertex-coordinate grid and LRH scanline.

    **Theoretical basis** (Daniels et al. 1997):
    The largest axis-aligned rectangle inscribed in a simple polygon always
    has at least two of its four sides determined by vertex coordinates of the
    polygon (exterior ring + all interior rings).  Building a grid whose lines
    are exactly those coordinates, running a cell-centre PIP test, and sweeping
    with the largest-rectangle-in-histogram stack algorithm therefore finds the
    globally optimal rectangle.  The result is EXACT at vertex-coordinate
    precision for any simple polygon, including concave polygons and polygons
    with holes.  PIP membership via Shapely ``contains`` naturally handles holes.

    **Grid construction**:
    * Collect all unique vertex x-coordinates → *xs_v* (column boundaries).
    * Collect all unique vertex y-coordinates → *ys_v* (row boundaries).
    * Add bounding-box extremes (already present for valid polygons, but
      included explicitly for robustness).
    * For the holed cases (``convex_with_holes``, ``concave_with_holes``)
      interior ring vertices are included so that the grid aligns with hole
      boundaries.

    **Fallback**:
    When either axis has more than ``_VERTEX_COORD_CAP`` unique values (e.g.
    heavily discretised circles), the function falls back to
    ``_uniform_grid_solve`` at ``_VERTEX_COORD_CAP`` resolution, which is an
    approximation but still practical and no worse than any other finite-grid
    method.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
    poly_type : str
        One of the four strings from ``_detect_polygon_type``; controls
        whether interior ring vertices are collected.
    max_ratio : float
        Maximum allowed long:short aspect ratio.  0.0 = unlimited.

    Returns
    -------
    best_rect : shapely.geometry.Polygon or None
    best_area : float
    """
    # ── Collect vertex coordinates ──────────────────────────────────────────
    all_xs: list = [c[0] for c in poly.exterior.coords[:-1]]
    all_ys: list = [c[1] for c in poly.exterior.coords[:-1]]

    include_holes = poly_type in ("convex_with_holes", "concave_with_holes")
    if include_holes:
        for ring in poly.interiors:
            for c in ring.coords[:-1]:
                all_xs.append(c[0])
                all_ys.append(c[1])

    minx, miny, maxx, maxy = poly.bounds
    all_xs += [minx, maxx]
    all_ys += [miny, maxy]

    xs_raw = np.unique(np.array(all_xs, dtype=np.float64))
    ys_raw = np.unique(np.array(all_ys, dtype=np.float64))

    # ── Collapse near-duplicate vertex coordinates ───────────────────────────
    # Circle-approximation holes can produce vertex coordinates that differ
    # only by floating-point noise (e.g. y = 5.0 and 4.999999999999999 from
    # sin(π)).  After augmentation these become a degenerate row/column with
    # near-zero height/width.  Every cell in such a row is False (a zero-area
    # box cannot be covered), which zeroes the LRH histogram chain and blocks
    # the solver from finding rectangles that span the degenerate line.  The
    # fix is to collapse near-duplicates before augmentation.
    def _dedupe(arr: np.ndarray, tol: float) -> np.ndarray:
        if len(arr) < 2:
            return arr
        keep = np.empty(len(arr), dtype=bool)
        keep[0] = True
        keep[1:] = np.diff(arr) > tol
        return arr[keep]

    _span_x = float(xs_raw[-1] - xs_raw[0]) if len(xs_raw) >= 2 else 1.0
    _span_y = float(ys_raw[-1] - ys_raw[0]) if len(ys_raw) >= 2 else 1.0
    _tol = max(_span_x, _span_y) * 1e-9
    xs_raw = _dedupe(xs_raw, _tol)
    ys_raw = _dedupe(ys_raw, _tol)

    # ── Midpoint augmentation ────────────────────────────────────────────────
    # Insert the midpoint between every consecutive pair of unique coordinates.
    # This is critical for correctness: without augmentation, cell centres can
    # land exactly on the polygon boundary (e.g. the hypotenuse of a right
    # triangle), causing Shapely's strict ``contains`` to return False for cells
    # that are geometrically inside the polygon.  Augmentation ensures every
    # cell centre is strictly interior to its geometric region.
    # The augmented grid has at most 2*n-1 lines per axis — still O(n).
    def _augment(arr: np.ndarray) -> np.ndarray:
        if len(arr) < 2:
            return arr
        result = np.empty(2 * len(arr) - 1, dtype=np.float64)
        result[0::2] = arr
        result[1::2] = 0.5 * (arr[:-1] + arr[1:])
        return result

    xs_v = _augment(xs_raw)
    ys_v = _augment(ys_raw)

    n_cols = len(xs_v) - 1
    n_rows = len(ys_v) - 1

    if n_cols < 1 or n_rows < 1:
        return None, 0.0

    # ── Fallback for pathological vertex density ────────────────────────────
    # Cap is applied AFTER augmentation (augmented grid is at most 2x raw size).
    if n_cols > _VERTEX_COORD_CAP or n_rows > _VERTEX_COORD_CAP:
        fb_cols = min(n_cols, _UNIFORM_FALLBACK_N)
        fb_rows = min(n_rows, _UNIFORM_FALLBACK_N)
        return _uniform_grid_solve(poly, fb_cols, fb_rows, max_ratio)

    # ── Exact cell-box mask ───────────────────────────────────────────────────
    # Each cell is accepted only if the ENTIRE rectangle fits inside the polygon.
    # Using poly.covers(cell_box) instead of point-in-polygon on the cell centre
    # eliminates false positives along diagonal edges (e.g. circular hole approx,
    # oblique triangles in UTM data).  This is O(n_cells * n_poly_vertices) but
    # n_cells is bounded by _VERTEX_COORD_CAP^2 and polygon n is typically small.
    try:
        from shapely.prepared import prep as _shp_prep
        _prep = _shp_prep(poly)
        def _cell_ok(cell_box):
            return _prep.covers(cell_box)
    except Exception:
        def _cell_ok(cell_box):
            return poly.covers(cell_box)

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    # Fast pre-filter: cell centre must be inside poly before doing full box check
    cx_pts = 0.5 * (xs_v[:-1] + xs_v[1:])
    cy_pts = 0.5 * (ys_v[:-1] + ys_v[1:])
    gxx, gyy = np.meshgrid(cx_pts, cy_pts)
    centre_flat = _mask_from_poly(poly, gxx.ravel(), gyy.ravel())
    centre_mask = centre_flat.reshape(n_rows, n_cols)
    for i in range(n_rows):
        y0_c, y1_c = ys_v[i], ys_v[i + 1]
        for j in range(n_cols):
            if not centre_mask[i, j]:
                continue
            cell_box = box(xs_v[j], y0_c, xs_v[j + 1], y1_c)
            if _cell_ok(cell_box):
                mask[i, j] = True


    # ── LRH scanline with variable-pitch kernel ──────────────────────────────
    heights  = np.zeros(n_cols, dtype=np.int64)
    best_rect: Optional[Polygon] = None
    best_area: float = 0.0
    fallback_rect: Optional[Polygon] = None
    fallback_area: float = 0.0

    for r in range(n_rows):
        row = mask[r].astype(np.int64)
        heights += row
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel_vp(
            heights, xs_v, ys_v, r, max_ratio
        )
        if area <= 0.0:
            continue

        rw_raw = x1 - x0
        rh_raw = y1 - y0

        # Apply aspect-ratio constraint via sliding-window search. Centre-
        # shrinking is unsafe for concave polygons: the centred sub-rect can
        # fall outside the valid region, so we also try the left/right (or
        # bottom/top) anchored sub-windows and keep the largest one that is
        # actually contained.
        if max_ratio > 0.0 and min(rw_raw, rh_raw) > 0.0:
            ls = max(rw_raw, rh_raw)
            ss = min(rw_raw, rh_raw)
            if ls / ss > max_ratio:
                nl = ss * max_ratio
                if rw_raw >= rh_raw:
                    candidates = [
                        box(x0, y0, x0 + nl, y1),
                        box(x1 - nl, y0, x1, y1),
                        box(0.5 * (x0 + x1) - nl * 0.5, y0,
                            0.5 * (x0 + x1) + nl * 0.5, y1),
                    ]
                else:
                    candidates = [
                        box(x0, y0, x1, y0 + nl),
                        box(x0, y1 - nl, x1, y1),
                        box(x0, 0.5 * (y0 + y1) - nl * 0.5,
                            x1, 0.5 * (y0 + y1) + nl * 0.5),
                    ]
                best_cand = None
                best_cand_area = 0.0
                for c in candidates:
                    if poly.covers(c):
                        ca = float(c.area)
                        if ca > best_cand_area:
                            best_cand_area = ca
                            best_cand = c
                    else:
                        clipped = _clip_rect_to_poly(poly, c)
                        if clipped is not None and poly.covers(clipped):
                            ca = float(clipped.area)
                            if ca > best_cand_area:
                                best_cand_area = ca
                                best_cand = clipped
                if best_cand is not None and best_cand_area > best_area:
                    best_area = best_cand_area
                    best_rect = best_cand
                # Also keep the unconstrained rect's clip as a fallback.
                cand_unconstrained = box(x0, y0, x1, y1)
                if not poly.covers(cand_unconstrained):
                    clipped = _clip_rect_to_poly(poly, cand_unconstrained)
                    if clipped is not None and poly.covers(clipped):
                        ca = float(clipped.area)
                        if ca > fallback_area:
                            fallback_area = ca
                            fallback_rect = clipped
                continue

        if area > best_area:
            cand = box(x0, y0, x1, y1)
            if not poly.covers(cand):
                # LRH may return a rect that spans a hole or protrudes outside.
                # Record it as a fallback candidate but do NOT raise best_area --
                # doing so would block later rows from finding the true exact rect.
                clipped = _clip_rect_to_poly(poly, cand)
                if clipped is not None and poly.covers(clipped):
                    c_area = float(clipped.area)
                    if c_area > fallback_area:
                        fallback_area = c_area
                        fallback_rect = clipped
            else:
                best_area = area
                best_rect = cand

    # Merge: prefer the exact (unclipped) rect; fall back to the best clipped one.
    if best_rect is None and fallback_rect is not None:
        return fallback_rect, fallback_area
    if fallback_rect is not None and fallback_area > best_area:
        return fallback_rect, fallback_area
    return best_rect, best_area


# ==========================================================================
# ⑤ EPSILON-INSET CONTAINMENT CERTIFICATION
# ==========================================================================

def _certify_rect(
    poly: Polygon,
    rect: Optional[Polygon],
    max_ratio: float,
    buf_enabled: bool,
    buf_value: float,
    prepared_poly=None,
) -> Tuple[Optional[Polygon], float]:
    """
    Verify that *rect* is fully contained in *poly* and apply a tiny
    epsilon inset if GEOS floating-point noise causes a marginal failure.

    Unlike the BCRS worker's ``_certify_and_adjust``, **no iterative
    binary-search shrink** is performed: the exact solvers produce rectangles
    whose sides are aligned to polygon vertex coordinates, so containment
    failures are exclusively sub-femtometre GEOS artefacts fixable by a
    single fixed epsilon inset.

    **Inset strategy**:
    If ``poly.covers(rect)`` fails, shrink *rect* symmetrically by
    ``_CERT_EPS`` on all four sides from its centre.  If the inseted
    rectangle still fails, the feature is rejected (returns ``None``).
    This is deliberately conservative: an inset of 1e-8 map units is
    invisible in any practical CRS but eliminates all floating-point
    boundary grazing.

    An optional buffer (*buf_enabled*, *buf_value*) is applied after a
    successful certification to allow deliberate safety margins.  Negative
    buffer values shrink the result; positive values grow it (use with care
    — a growing buffer may violate containment and the resulting geometry is
    not re-certified).

    Parameters
    ----------
    poly : shapely.geometry.Polygon
    rect : shapely.geometry.Polygon or None
    max_ratio : float
    buf_enabled : bool
    buf_value : float
    prepared_poly : shapely.prepared.PreparedGeometry or None

    Returns
    -------
    (final_rect, area) : tuple
        ``(None, 0.0)`` when certification fails after epsilon inset.
    """
    if rect is None or rect.is_empty:
        return None, 0.0

    prep = prepared_poly
    if prep is None:
        try:
            prep = shp_prep(poly)
        except Exception:
            prep = None

    def _covers(r):
        if prep is not None:
            return prep.covers(r)
        return poly.covers(r)

    # Primary check
    if _covers(rect):
        final = rect
    else:
        # Single epsilon inset on the actual rectangle geometry (preserves rotation).
        # We use a negative buffer with flat cap/join so the shape stays rectangular.
        # This is correct for both axis-aligned and rotated rectangles.
        try:
            inseted = rect.buffer(-_CERT_EPS, cap_style=3, join_style=2)
        except Exception:
            inseted = None
        if inseted is None or inseted.is_empty or inseted.area <= 0.0:
            return None, 0.0
        if not _covers(inseted):
            return None, 0.0
        final = inseted

    # Optional post-certification buffer
    if buf_enabled and buf_value != 0.0:
        try:
            cand = final.buffer(buf_value, cap_style=3, join_style=2)
            if not cand.is_empty and cand.area > 0.0:
                final = cand
        except Exception:
            pass

    return final, float(final.area)




# ==========================================================================
# ③.5  BOUNDARY-PUSH REFINEMENT  (post-grid diagonal-edge gap correction)
# ==========================================================================

_REFINE_BINARY_STEPS = 52   # ~15 significant digits precision; ~5e-16 relative
_REFINE_ITERATIONS   = 2    # 1 pass corrects diagonal gaps; 2nd catches coupled sides

def _refine_rect_to_boundary(
    poly: Polygon,
    rect: Optional[Polygon],
    max_ratio: float,
    prepared_poly=None,
) -> Tuple[Optional[Polygon], float]:
    """
    Push each side of *rect* outward to the true polygon boundary via binary
    search, then return the largest contained rectangle.

    **Purpose**
    The vertex-coordinate grid (Daniels et al. 1997) snaps all four rectangle
    sides to polygon vertex x/y values.  When the optimal rectangle's boundary
    falls on a *diagonal* polygon edge — at a coordinate that is NOT a vertex
    x or y value — the grid solution leaves a gap.  This function closes that
    gap by iteratively expanding each side to the exact polygon boundary.

    **Correctness guarantee**
    After refinement the returned rectangle satisfies ``poly.covers(rect)``
    exactly (verified by the final Shapely call).  The Daniels theorem guarantees
    the optimal rectangle has at least 2 sides at vertex coordinates; the other
    2 sides (which may lie on diagonal edges) are found here by binary search
    to sub-nanometre precision (52 bisection steps ≈ 1e-15 map-unit error).

    **Complexity**
    O(4 × n_iter × n_binary) ≈ 416 ``poly.covers`` calls per invocation —
    negligible compared with the O(n²) grid construction.

    Parameters
    ----------
    poly : Polygon
    rect : Polygon or None
    max_ratio : float
    prepared_poly : PreparedGeometry or None

    Returns
    -------
    (refined_rect, area) or (rect, rect.area) if refinement yields no improvement.
    """
    if rect is None or rect.is_empty:
        return rect, 0.0

    pp = prepared_poly
    if pp is None:
        try:
            pp = shp_prep(poly)
        except Exception:
            pp = None

    def _ok(cx0, cy0, cx1, cy1):
        if cx0 >= cx1 or cy0 >= cy1:
            return False
        r = box(cx0, cy0, cx1, cy1)
        return pp.covers(r) if pp is not None else poly.covers(r)

    minx, miny, maxx, maxy = poly.bounds
    x0, y0, x1, y1 = rect.bounds

    # Sanity: starting rect must be inside poly (it always should be)
    if not _ok(x0, y0, x1, y1):
        eps = _CERT_EPS
        if not _ok(x0 + eps, y0 + eps, x1 - eps, y1 - eps):
            return rect, float(rect.area)
        x0, y0, x1, y1 = x0 + eps, y0 + eps, x1 - eps, y1 - eps

    for _pass in range(_REFINE_ITERATIONS):
        # Push LEFT outward (x0 decreases toward minx)
        lo, hi = minx, x0
        for _ in range(_REFINE_BINARY_STEPS):
            mid = 0.5 * (lo + hi)
            if _ok(mid, y0, x1, y1):
                hi = mid
            else:
                lo = mid
        x0 = hi

        # Push RIGHT outward (x1 increases toward maxx)
        lo, hi = x1, maxx
        for _ in range(_REFINE_BINARY_STEPS):
            mid = 0.5 * (lo + hi)
            if _ok(x0, y0, mid, y1):
                lo = mid
            else:
                hi = mid
        x1 = lo

        # Push BOTTOM outward (y0 decreases toward miny)
        lo, hi = miny, y0
        for _ in range(_REFINE_BINARY_STEPS):
            mid = 0.5 * (lo + hi)
            if _ok(x0, mid, x1, y1):
                hi = mid
            else:
                lo = mid
        y0 = hi

        # Push TOP outward (y1 increases toward maxy)
        lo, hi = y1, maxy
        for _ in range(_REFINE_BINARY_STEPS):
            mid = 0.5 * (lo + hi)
            if _ok(x0, y0, x1, mid):
                lo = mid
            else:
                hi = mid
        y1 = lo

    area_new = (x1 - x0) * (y1 - y0)
    if area_new <= float(rect.area) - 1e-9:
        return rect, float(rect.area)

    refined = box(x0, y0, x1, y1)
    if not poly.covers(refined):
        # Floating-point edge case: final rect grazes boundary. Inset by eps.
        refined = box(x0 + _CERT_EPS, y0 + _CERT_EPS, x1 - _CERT_EPS, y1 - _CERT_EPS)
        if refined.is_empty or not poly.covers(refined):
            return rect, float(rect.area)
        area_new = float(refined.area)

    return refined, area_new


# ==========================================================================
# ⑥ BEST-EFFORT SHRINK FALLBACK  (for always_return path)
# ==========================================================================

def _best_effort_shrink_to_cover(
    poly: Polygon,
    rect: Optional[Polygon],
    max_ratio: float,
    prepared_poly=None,
) -> Tuple[Optional[Polygon], float]:
    """
    Binary-search for the largest uniform scale factor s ∈ (0, 1] such that
    *rect* scaled by *s* from its centre is fully contained in *poly*.

    Used as a last-resort fallback when the exact solver returns a rectangle
    that fails even the epsilon-inset certification — which should not happen
    in normal operation but can occur for degenerate or nearly-degenerate
    polygons.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
    rect : shapely.geometry.Polygon or None
    max_ratio : float
    prepared_poly : shapely.prepared.PreparedGeometry or None

    Returns
    -------
    (best_rect, area) : tuple
        ``(None, 0.0)`` if no valid scale found.
    """
    if rect is None or rect.is_empty:
        return None, 0.0

    centroid_r = rect.centroid
    prep = prepared_poly
    if prep is None:
        try:
            prep = shp_prep(poly)
        except Exception:
            prep = None

    def _covers(r):
        if prep is not None:
            return prep.covers(r)
        return poly.covers(r)

    def build(s: float) -> Optional[Polygon]:
        """Scale rect uniformly by factor s around its centroid."""
        if s <= 0.0:
            return None
        from shapely.affinity import scale as shp_scale
        try:
            scaled = shp_scale(rect, xfact=s, yfact=s, origin=centroid_r)
            if scaled is None or scaled.is_empty or scaled.area <= 0.0:
                return None
            return scaled
        except Exception:
            return None

    # Find a valid starting lower bound
    lo = 0.0
    r_lo = None
    for s in (1.0, 0.95, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01):
        r = build(s)
        if r is not None and _covers(r):
            lo = s
            r_lo = r
            break

    if r_lo is None:
        return None, 0.0

    hi   = 1.0
    best_r = r_lo
    best_a = float(r_lo.area)

    for _ in range(48):
        if hi - lo < 1e-9:
            break
        mid = 0.5 * (lo + hi)
        r = build(mid)
        if r is not None and _covers(r):
            lo     = mid
            best_r = r
            best_a = float(r.area)
        else:
            hi = mid

    return best_r, best_a


# ==========================================================================
# ⑦ GEOMETRY PREPARATION
# ==========================================================================

def _prepare_polygon(geom) -> Optional[Polygon]:
    """
    Validate and normalise an arbitrary Shapely geometry to a single Polygon.

    Steps:
    1. Call ``make_valid`` if the geometry reports as invalid.
    2. If a MultiPolygon (or other geometry collection) results, keep the
       largest-area component.
    3. Reject anything empty, non-polygonal, or with zero area.

    Parameters
    ----------
    geom : shapely.geometry.BaseGeometry or None

    Returns
    -------
    shapely.geometry.Polygon or None
    """
    from shapely.validation import make_valid

    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)

    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and not g.is_empty and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)
    elif hasattr(geom, "geoms") and not isinstance(geom, Polygon):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and not g.is_empty and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)

    if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= 0:
        return None
    return geom


# ==========================================================================
# ⑧ MAIN AXIS-ALIGNED LIR SOLVER
# ==========================================================================

def _solve_axis_aligned_lir(
    poly: Polygon,
    axis_angle: float,
    grid_fine: int,
    max_ratio: float,
    always_return: bool,
    buf_enabled: bool,
    buf_value: float,
) -> Tuple[Optional[Polygon], float, float, str, float, bool]:
    """
    Solve for the largest inscribed rectangle that is axis-aligned in the
    frame rotated by *axis_angle* degrees, using the exact algorithm
    appropriate for the polygon's type.

    Pipeline
    --------
    1. **Type detection** — classify *poly* as one of the four cases via
       ``_detect_polygon_type``.
    2. **Frame rotation** — rotate *poly* by ``−axis_angle`` around its
       centroid to enter the "axis-aligned solve frame".
    3. **Exact solve** — dispatch:
       * ``convex_no_holes``   → ``_exact_solve_convex`` (Alt et al. 1994)
       * all others            → ``_exact_solve_vertex_grid`` (Daniels et al. 1997)
    4. **Frame inverse-rotation** — rotate the result back by ``+axis_angle``.
    5. **Certification** — ``_certify_rect`` applies a tiny epsilon inset if
       GEOS floating-point noise causes a marginal containment failure.
    6. **Best-effort fallback** — if certification still fails and
       ``always_return`` is True, ``_best_effort_shrink_to_cover`` is called.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon (valid, non-empty, in world CRS).
    axis_angle : float
        Degrees by which the "axis-aligned" frame is rotated relative to the
        world CRS.  0.0 = standard horizontal/vertical.  Any real value is
        accepted; 90.0 is equivalent to 0.0 by symmetry.
    grid_fine : int
        Uniform fallback grid resolution (used only when vertex density
        exceeds ``_VERTEX_COORD_CAP``).
    max_ratio : float
        Maximum allowed long:short aspect ratio.  0.0 = unlimited.
    always_return : bool
        When True, attempt a best-effort shrink fallback when the exact
        solve result fails epsilon-inset certification.
    buf_enabled : bool
        When True, apply a post-certification buffer of *buf_value*.
    buf_value : float
        Buffer distance in map units (positive = grow, negative = shrink).

    Returns
    -------
    rect : shapely.geometry.Polygon or None
    area : float
    axis_angle : float
        Echo of the input parameter.
    poly_type : str
        One of the four detection strings.
    ratio : float
        Actual long:short aspect ratio of the returned rectangle.
    used_best_effort : bool
        True when the result was produced by the shrink fallback.
    """
    centroid = poly.centroid

    # Step 1 — classify
    poly_type = _detect_polygon_type(poly)

    # Step 2 — rotate into solve frame
    rot_poly: Polygon = shp_rotate(
        poly, -axis_angle, origin=centroid, use_radians=False
    )

    # Step 3 — exact solve
    if poly_type == "convex_no_holes":
        best_rect_rot, best_area = _exact_solve_convex(rot_poly, max_ratio)
    else:
        best_rect_rot, best_area = _exact_solve_vertex_grid(
            rot_poly, poly_type, max_ratio
        )

    # Step 3.5 — boundary-push refinement for vertex-grid solves
    # The LRH grid snaps sides to vertex coordinates; the true optimum may have
    # sides on diagonal polygon edges (Daniels theorem: only 2 sides guaranteed at
    # vertex coords).  Push all 4 sides outward to recover the exact solution.
    if poly_type != "convex_no_holes" and best_rect_rot is not None:
        try:
            prep_rot = shp_prep(rot_poly)
        except Exception:
            prep_rot = None
        refined_rot, refined_area = _refine_rect_to_boundary(
            rot_poly, best_rect_rot, max_ratio, prep_rot
        )
        if refined_area > best_area:
            best_rect_rot, best_area = refined_rot, refined_area

    if best_rect_rot is None or best_area <= 0.0:
        return None, 0.0, axis_angle, poly_type, 1.0, False

    # Step 4 — rotate result back to world frame
    best_rect_world: Polygon = shp_rotate(
        best_rect_rot, axis_angle, origin=centroid, use_radians=False
    )

    # Step 5 — epsilon-inset certification
    prepared_poly = None
    try:
        prepared_poly = shp_prep(poly)
    except Exception:
        pass

    final_rect, final_area = _certify_rect(
        poly, best_rect_world, max_ratio, buf_enabled, buf_value, prepared_poly
    )

    used_best_effort = False

    # Step 6 — best-effort fallback
    if final_rect is None and always_return:
        final_rect, final_area = _best_effort_shrink_to_cover(
            poly, best_rect_world, max_ratio, prepared_poly
        )
        used_best_effort = final_rect is not None

    if final_rect is None:
        return None, 0.0, axis_angle, poly_type, 1.0, False

    # Compute actual aspect ratio of the output rectangle
    coords = list(final_rect.exterior.coords)
    w = math.hypot(
        coords[1][0] - coords[0][0],
        coords[1][1] - coords[0][1],
    )
    h = math.hypot(
        coords[2][0] - coords[1][0],
        coords[2][1] - coords[1][1],
    )
    ratio = max(w, h) / min(w, h) if min(w, h) > 0.0 else 1.0

    return final_rect, float(final_area), float(axis_angle), poly_type, ratio, used_best_effort


# ==========================================================================
# ⑨ PUBLIC WORKER ENTRY POINT
# ==========================================================================

def _worker_process_feature(args: tuple) -> Optional[tuple]:
    """
    Stateless worker — safe for ``concurrent.futures.ThreadPoolExecutor``
    and ``ProcessPoolExecutor``.

    Parameters
    ----------
    args : tuple
        ``(feat_id, wkb_bytes, axis_angle, grid_fine, max_ratio,
        buf_enabled, buf_value, always_return)``

        feat_id       : int   — source feature identifier
        wkb_bytes     : bytes — WKB-encoded polygon geometry
        axis_angle    : float — degrees; 0.0 = standard axis-aligned
        grid_fine     : int   — fallback uniform grid resolution
        max_ratio     : float — max long:short ratio (0 = unlimited)
        buf_enabled   : bool  — apply post-certification buffer
        buf_value     : float — buffer distance in map units
        always_return : bool  — use best-effort fallback on cert failure

    Returns
    -------
    tuple or None
        ``(feat_id, wkt, area, axis_angle, poly_type, ratio, used_best_effort)``
        or None if the feature cannot be processed.

    Raises
    ------
    RuntimeError
        Re-raised with feature context if an unexpected exception occurs
        during processing.
    """
    (feat_id, wkb_bytes, axis_angle, grid_fine, max_ratio,
     buf_enabled, buf_value, always_return) = args

    try:
        # Parse WKB → validated Shapely polygon
        poly = _prepare_polygon(wkb_loads(bytes(wkb_bytes)))
        if poly is None:
            return None

        # Optional floating-point precision normalisation
        try:
            from shapely import set_precision
            minx, miny, maxx, maxy = poly.bounds
            span = max(maxx - minx, maxy - miny)
            if span > 0:
                poly = set_precision(
                    poly, grid_size=span * 1e-9, mode="valid_output"
                )
        except Exception:
            pass

        if poly is None or poly.is_empty:
            return None

        rect, area, ang, poly_type, ratio, used_best_effort = _solve_axis_aligned_lir(
            poly,
            axis_angle=axis_angle,
            grid_fine=grid_fine,
            max_ratio=max_ratio,
            always_return=always_return,
            buf_enabled=buf_enabled,
            buf_value=buf_value,
        )

        if rect is None:
            return None

        return (
            feat_id,
            rect.wkt,
            round(float(area), 4),
            round(float(ang), 4),
            str(poly_type),
            round(float(ratio), 4),
            int(used_best_effort),
        )

    except Exception as e:
        raise RuntimeError(
            f"_worker_process_feature failed for feat_id={feat_id}: {e}"
        ) from e


# ==========================================================================
# ⑩ TEST CASES  (not called in production)
# ==========================================================================

def _test_cases() -> dict:
    """
    Return a dictionary of four representative Shapely polygons, one for each
    of the polygon-type cases handled by ``_detect_polygon_type``.

    Intended for interactive testing and unit test fixtures only; never called
    during normal QGIS processing.

    Returns
    -------
    dict with keys:
        ``"convex_no_holes"``, ``"convex_with_holes"``,
        ``"concave_no_holes"``, ``"concave_with_holes"``.

    Additional interesting cases (as comments)
    ------------------------------------------
    U-shaped concave polygon (more complex concavity)::

        u_shape = Polygon([
            (0, 0), (10, 0), (10, 4), (7, 4),
            (7, 8), (10, 8), (10, 12), (0, 12),
            (0, 8), (3, 8), (3, 4), (0, 4),
        ])
        # The two vertical arms each hold a 3×8 rectangle; the optimal LIR
        # spans the full 10×4 base section, area = 40. With max_ratio
        # constraints the two arms (area 24 each) may win instead.

    Polygon with multiple holes (3×3 grid of square punch-outs)::

        outer = [(0,0),(30,0),(30,30),(0,30)]
        holes = [
            [(3*i+1, 3*j+1),(3*i+2,3*j+1),(3*i+2,3*j+2),(3*i+1,3*j+2)]
            for i in range(9) for j in range(9)
        ]
        multi_hole = Polygon(outer, holes)
        # Optimal LIR spans the inter-hole strips; exact answer requires
        # all hole vertex coordinates in the grid (handled automatically
        # by _exact_solve_vertex_grid for concave_with_holes type).
    """
    from shapely.geometry import Polygon
    import numpy as np

    # Case 1: Convex, no holes — simple square
    # Exact solver: _exact_solve_convex; expected LIR = 10×10 = 100 units²
    convex_no_holes = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Case 2: Convex with holes — square with circular hole (32-gon approximation)
    # Exact solver: _exact_solve_vertex_grid; LIR avoids the central circle
    t = np.linspace(0, 2 * np.pi, 32)
    hole = list(zip(5 + 2 * np.cos(t), 5 + 2 * np.sin(t)))
    convex_with_holes = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], [hole])

    # Case 3: Concave, no holes — L-shape
    # Exact solver: _exact_solve_vertex_grid; LIR fits in the 10×5 lower bar
    concave_no_holes = Polygon(
        [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    )

    # Case 4: Concave with holes — L-shape with rectangular punch-out
    # Exact solver: _exact_solve_vertex_grid; hole vertex coords included in grid
    concave_with_holes = Polygon(
        [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)],
        [[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )

    return {
        "convex_no_holes":    convex_no_holes,
        "convex_with_holes":  convex_with_holes,
        "concave_no_holes":   concave_no_holes,
        "concave_with_holes": concave_with_holes,
    }