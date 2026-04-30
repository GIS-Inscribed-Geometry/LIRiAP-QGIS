"""
LIRiAP BCRS Fast worker module.

Pure geometry solver for the optimized BCRS solve path.
No QGIS or Qt runtime dependencies.

Pipeline
======
Same as BCRS Standard:
Stage 1: Geometry preparation
Stage 2: Heuristic candidates
Stage 3: Angle refinement with trial ranking
Stage 4: BCRS boundary-coordinate solve (limited to top candidates)
Stage 5: SDF-guided boundary expansion
Stage 6: Containment certification
Stage 7: Selection and output

Optimization Differences
=======================
- Trial ranking: Ranks Stage 3 angles by likelihood before expensive Stage 4-5
- Limited runs: Only top candidates proceed to boundary-coordinate solve
- Area cache: Reuses computed values to reduce repeated work
- Reduced trials: t <= 2 instead of t <= 4

Algorithm Semantics
==================
Same as BCRS Standard: strict containment or best-effort fallback.

Output
======
(feat_id, wkt, area, angle_deg, ratio, cand_rank, s2_gain, s4_gain, s5_gain, best_effort) or None

See Also
========
bcrs_fast_algorithm: QGIS wrapper
bcrs_worker: Non-optimized variant
sdf_oracle: SDF-based signed-distance field oracle
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize_scalar
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import box, MultiPolygon, Polygon, Point
from shapely.prepared import prep as shp_prep
from shapely.wkb import loads as wkb_loads

try:
    from shapely.prepared import prep as _prep_geom
except Exception:
    _prep_geom = None

# --------------------------------------------------------------------------
# Vectorised point-in-polygon (Shapely 1.x + 2.x compat)
# --------------------------------------------------------------------------
try:
    from shapely.vectorized import contains as _shp_contains_vec


    def _mask_from_poly(poly, xx_flat, yy_flat):
        return _shp_contains_vec(poly, xx_flat, yy_flat)
except ImportError:
    import shapely as _shp2


    def _mask_from_poly(poly, xx_flat, yy_flat):
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
        return fn if fn is not None else (lambda f: f)


    _NUMBA_AVAILABLE = False

# --------------------------------------------------------------------------
# SDF oracle — signed-distance-field primitive (inlined from sdf_oracle.py)
# --------------------------------------------------------------------------
def _polygon_sdf(poly, x, y):
    """Signed distance: negative inside, positive outside/in-hole."""
    from shapely.geometry import Point
    pt = Point(x, y)
    d_poly = poly.distance(pt)
    if d_poly > 0.0:
        return d_poly
    d_ext = poly.exterior.distance(pt)
    if poly.contains(pt):
        min_d = d_ext
        for ring in poly.interiors:
            d_h = ring.distance(pt)
            if d_h < min_d:
                min_d = d_h
        return -min_d
    if d_ext < 1e-12:
        return 0.0
    for ring in poly.interiors:
        hp = Polygon(ring)
        if hp.contains(pt):
            return hp.exterior.distance(pt)
    return 0.0

# --------------------------------------------------------------------------
# Tuning constants
# --------------------------------------------------------------------------
_PHASE_A_XATOL = 0.02  # Brent angle tolerance [deg]
_PHASE_A_HALFWIDTH = 3.0  # Brent bracket half-width [deg]
_CERT_EPS = 1e-7  # Safety inset after certification
_CERT_MAX_SHRINK = 0.20  # Max symmetric shrink as fraction of shorter side
_PRUNE_MARGIN = 0.90  # Upper-bound pruning factor
_SIMPLIFY_THRESHOLD = 300  # Vertex count above which simplification is tried
_SIMPLIFY_TOL_FRAC = 0.001  # Simplification tol as fraction of short bbox side
_EXPAND_ITERS = 3  # Boundary expansion outer iterations
_EXPAND_STEPS = 24  # Binary-search steps per expansion side
_ANGLE_DELTA_DEG = 0.5  # ± delta tested around each Brent-polished angle
_STAGE2_MAX_TRIALS = 2  # Run expensive BCRS only on top-N angle trials


# ==========================================================================
# ① JIT HISTOGRAM KERNELS
# ==========================================================================

@_njit(cache=True)
def _histogram_kernel_vp(heights, xs, ys, row_idx, max_ratio):
    """
    Largest-rectangle-in-histogram with VARIABLE-PITCH columns/rows.

    xs[i] is the LEFT edge of column i; xs[i+1] is the RIGHT edge.
    ys[r] is the BOTTOM edge of row r; ys[r+1] is the TOP edge.

    Used by BCRS where grid lines are polygon vertex coordinates.
    """
    cols = len(heights)
    n_xs = len(xs)
    n_ys = len(ys)
    best = 0.0
    bx0 = by0 = bx1 = by1 = 0.0

    st_col = np.empty(cols + 1, dtype=np.int64)
    st_h = np.empty(cols + 1, dtype=np.int64)
    top = 0

    for c in range(cols + 1):
        h = int(heights[c]) if c < cols else 0
        start = c
        while top > 0 and st_h[top - 1] > h:
            top -= 1
            sc = st_col[top]
            sh = st_h[top]
            xi = sc + (c - sc)
            x0_w = xs[sc]
            x1_w = xs[xi if xi < n_xs else n_xs - 1]
            ri0 = row_idx - sh + 1
            y0_w = ys[ri0 if ri0 >= 0 else 0]
            ri1 = row_idx + 1
            y1_w = ys[ri1 if ri1 < n_ys else n_ys - 1]
            rw = x1_w - x0_w
            rh = y1_w - y0_w
            if rw <= 0.0 or rh <= 0.0:
                start = sc
                continue
            if max_ratio > 0.0:
                ls = rw if rw >= rh else rh
                ss = rh if rw >= rh else rw
                if ss > 0.0 and ls / ss > max_ratio:
                    nl = ss * max_ratio
                    if rw >= rh:
                        cx = (x0_w + x1_w) * 0.5
                        x0_w = cx - nl * 0.5
                        x1_w = cx + nl * 0.5
                    else:
                        cy = (y0_w + y1_w) * 0.5
                        y0_w = cy - nl * 0.5
                        y1_w = cy + nl * 0.5
                    rw = x1_w - x0_w
                    rh = y1_w - y0_w
            area = rw * rh
            if area > best:
                best = area
                bx0 = x0_w;
                by0 = y0_w
                bx1 = x1_w;
                by1 = y1_w
            start = sc
        st_col[top] = start
        st_h[top] = h
        top += 1

    return bx0, by0, bx1, by1, best


@_njit(cache=True)
def _histogram_kernel(heights, xs, ys, row_idx, max_ratio):
    """Uniform-pitch version — used by the coarse grid solver and Brent polisher."""
    cols = len(heights)
    n_xs = len(xs)
    n_ys = len(ys)
    best = 0.0
    bx0 = by0 = bx1 = by1 = 0.0
    st_col = np.empty(cols + 1, dtype=np.int64)
    st_h = np.empty(cols + 1, dtype=np.int64)
    top = 0

    for c in range(cols + 1):
        h = int(heights[c]) if c < cols else 0
        start = c
        while top > 0 and st_h[top - 1] > h:
            top -= 1
            sc = st_col[top];
            sh = st_h[top]
            w = c - sc
            xi = sc + w
            x0_w = xs[sc]
            x1_w = xs[xi if xi < n_xs else n_xs - 1]
            ri0 = row_idx - sh + 1
            y0_w = ys[ri0 if ri0 >= 0 else 0]
            y1_w = ys[row_idx if row_idx < n_ys else n_ys - 1]
            rw = x1_w - x0_w
            rh = y1_w - y0_w
            if rw <= 0.0 or rh <= 0.0:
                start = sc;
                continue
            if max_ratio > 0.0:
                ls = rw if rw >= rh else rh
                ss = rh if rw >= rh else rw
                if ss > 0.0 and ls / ss > max_ratio:
                    nl = ss * max_ratio
                    if rw >= rh:
                        cx = (x0_w + x1_w) * 0.5
                        x0_w = cx - nl * 0.5;
                        x1_w = cx + nl * 0.5
                    else:
                        cy = (y0_w + y1_w) * 0.5
                        y0_w = cy - nl * 0.5;
                        y1_w = cy + nl * 0.5
                    rw = x1_w - x0_w;
                    rh = y1_w - y0_w
            area = rw * rh
            if area > best:
                best = area;
                bx0 = x0_w;
                by0 = y0_w;
                bx1 = x1_w;
                by1 = y1_w
            start = sc
        st_col[top] = start;
        st_h[top] = h;
        top += 1

    return bx0, by0, bx1, by1, best


# --------------------------------------------------------------------------
# Prepared-geometry helpers
# --------------------------------------------------------------------------
def _make_prepared(poly):
    if poly is None or poly.is_empty or _prep_geom is None:
        return None
    try:
        return _prep_geom(poly)
    except Exception:
        return None


def _covers(poly, candidate, prepared_poly=None):
    if prepared_poly is not None:
        return prepared_poly.covers(candidate)
    return poly.covers(candidate)


# ==========================================================================
# ② UNIFORM GRID SOLVER (Stage 1 / Brent)
# ==========================================================================
def _solve_axis_rect_grid(poly, grid_steps, max_ratio):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, grid_steps)
    ys = np.linspace(miny, maxy, grid_steps)
    xx, yy = np.meshgrid(xs, ys)
    flat = _mask_from_poly(poly, xx.ravel(), yy.ravel())
    mask_i64 = flat.reshape(grid_steps, grid_steps).astype(np.int64, copy=False)
    heights = np.zeros(grid_steps, dtype=np.int64)
    best_rect = None;
    best_area = 0.0

    for r in range(grid_steps):
        row = mask_i64[r]
        heights += row;
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel(heights, xs, ys, r, max_ratio)
        if area > best_area:
            best_area = area
            best_rect = box(x0, y0, x1, y1)

    return best_rect, best_area


# ==========================================================================
# ③ BCRS — Boundary-Coordinate Raster Solve
# ==========================================================================
def _solve_axis_rect_bcrs(rot_poly, seed_bounds, max_ratio):
    """
    Uses polygon vertex x/y coordinates as histogram grid lines.
    Provably finds the globally optimal axis-aligned rectangle at
    vertex-coordinate precision for any straight-sided polygon.

    Falls back gracefully for smooth polygons (>300 unique coords per axis)
    by returning (None, 0.0) — the caller uses the uniform grid seed instead.
    """
    all_xs_raw = [c[0] for c in rot_poly.exterior.coords[:-1]]
    all_ys_raw = [c[1] for c in rot_poly.exterior.coords[:-1]]
    for interior in rot_poly.interiors:
        for c in interior.coords[:-1]:
            all_xs_raw.append(c[0])
            all_ys_raw.append(c[1])

    minx, miny, maxx, maxy = rot_poly.bounds
    all_xs_raw += [minx, maxx]
    all_ys_raw += [miny, maxy]

    xs_v = np.unique(np.array(all_xs_raw, dtype=np.float64))
    ys_v = np.unique(np.array(all_ys_raw, dtype=np.float64))

    # Cap for smooth polygons (hundreds of vertices → BCRS is expensive)
    if len(xs_v) > 300 or len(ys_v) > 300:
        return None, 0.0

    n_cols = len(xs_v) - 1
    n_rows = len(ys_v) - 1
    if n_cols < 1 or n_rows < 1:
        return None, 0.0

    # Cell-centre point-in-polygon test
    cx_pts = 0.5 * (xs_v[:-1] + xs_v[1:])
    cy_pts = 0.5 * (ys_v[:-1] + ys_v[1:])
    gxx, gyy = np.meshgrid(cx_pts, cy_pts)
    flat = _mask_from_poly(rot_poly, gxx.ravel(), gyy.ravel())
    mask_i64 = flat.reshape(n_rows, n_cols).astype(np.int64, copy=False)

    heights = np.zeros(n_cols, dtype=np.int64)
    best_rect = None
    best_area = 0.0

    if seed_bounds is not None:
        sx0, sy0, sx1, sy1 = seed_bounds
        if sx1 > sx0 and sy1 > sy0:
            best_area = (sx1 - sx0) * (sy1 - sy0)
            best_rect = box(sx0, sy0, sx1, sy1)

    for r in range(n_rows):
        row = mask_i64[r]
        heights += row
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel_vp(
            heights, xs_v, ys_v, r, max_ratio)
        if area > best_area:
            best_area = area
            best_rect = box(x0, y0, x1, y1)

    return best_rect, best_area


# ==========================================================================
# ④ SDF-GUIDED BOUNDARY EXPANSION
# ==========================================================================
def _expand_rect_to_boundary(rot_poly, x0, y0, x1, y1, max_ratio, emitter=None):
    """
    SDF-guided coordinate-ascent boundary expansion.

    Uses the signed-distance field at each side's midpoint to compute a
    tight upper bound on the expansion distance, then binary-searches to
    the exact boundary using ``prep.covers``.  The SDF bounds are tighter
    than raw bounding-box extent, requiring fewer binary steps.
    """
    minx, miny, maxx, maxy = rot_poly.bounds
    prep = shp_prep(rot_poly)

    def _v(ax0, ay0, ax1, ay1):
        if ax1 - ax0 < 1e-12 or ay1 - ay0 < 1e-12:
            return False
        return prep.covers(box(ax0, ay0, ax1, ay1))

    # Shrink to valid start (diagonal edges / slightly-invalid seeds)
    if not _v(x0, y0, x1, y1):
        cx_c = 0.5 * (x0 + x1);
        cy_c = 0.5 * (y0 + y1)
        hw = 0.5 * (x1 - x0);
        hh = 0.5 * (y1 - y0)
        lo, hi = 0.0, 1.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if _v(cx_c - hw * mid, cy_c - hh * mid,
                  cx_c + hw * mid, cy_c + hh * mid):
                lo = mid
            else:
                hi = mid
        if lo < 1e-9:
            return x0, y0, x1, y1
        x0 = cx_c - hw * lo;
        y0 = cy_c - hh * lo
        x1 = cx_c + hw * lo;
        y1 = cy_c + hh * lo

    SDF_BINARY_STEPS = 10

    for _ in range(_EXPAND_ITERS):
        # Left
        if x0 > minx:
            sdf = _polygon_sdf(rot_poly, x0, 0.5 * (y0 + y1))
            hi_d = min(x0 - minx, abs(sdf)) if sdf < 0 else x0 - minx
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0 - mid, y0, x1, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                x0 -= lo_d

        # Right
        if x1 < maxx:
            sdf = _polygon_sdf(rot_poly, x1, 0.5 * (y0 + y1))
            hi_d = min(maxx - x1, abs(sdf)) if sdf < 0 else maxx - x1
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0, x1 + mid, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                x1 += lo_d

        # Bottom
        if y0 > miny:
            sdf = _polygon_sdf(rot_poly, 0.5 * (x0 + x1), y0)
            hi_d = min(y0 - miny, abs(sdf)) if sdf < 0 else y0 - miny
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0 - mid, x1, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                y0 -= lo_d

        # Top
        if y1 < maxy:
            sdf = _polygon_sdf(rot_poly, 0.5 * (x0 + x1), y1)
            hi_d = min(maxy - y1, abs(sdf)) if sdf < 0 else maxy - y1
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0, x1, y1 + mid):
                        lo_d = mid
                    else:
                        hi_d = mid
                y1 += lo_d

    # Ratio constraint (analytical, from centre)
    if max_ratio > 0.0:
        rw = x1 - x0;
        rh = y1 - y0
        if rw > 0 and rh > 0:
            ls = max(rw, rh);
            ss = min(rw, rh)
            if ss > 0 and ls / ss > max_ratio:
                nl = ss * max_ratio
                if rw >= rh:
                    cx_r = 0.5 * (x0 + x1);
                    x0 = cx_r - 0.5 * nl;
                    x1 = cx_r + 0.5 * nl
                else:
                    cy_r = 0.5 * (y0 + y1);
                    y0 = cy_r - 0.5 * nl;
                    y1 = cy_r + 0.5 * nl

    return x0, y0, x1, y1


# ==========================================================================
# ⑤ EDGE-GUIDED ANGLE CANDIDATE GENERATOR
# ==========================================================================
def _edge_candidate_angles(poly, min_sep_deg=4.0, max_candidates=12):
    coord_sets = [np.asarray(poly.exterior.coords, dtype=np.float64)]
    for interior in poly.interiors:
        coord_sets.append(np.asarray(interior.coords, dtype=np.float64))

    all_edges = [];
    all_lengths = []
    for coords in coord_sets:
        edges = np.diff(coords, axis=0)
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        valid = lengths > 1e-12
        if valid.any():
            all_edges.append(edges[valid])
            all_lengths.append(lengths[valid])

    if not all_edges:
        return np.array([0.0, 45.0])

    edges = np.vstack(all_edges)
    lengths = np.concatenate(all_lengths)
    angles = np.degrees(np.arctan2(np.abs(edges[:, 1]),
                                   np.abs(edges[:, 0]))) % 90.0

    bins = np.zeros(91, dtype=np.float64)
    idx = np.clip(np.round(angles).astype(np.int64), 0, 90)
    np.add.at(bins, idx, lengths)
    bins = np.convolve(bins, np.array([0.1, 0.2, 0.4, 0.2, 0.1]), mode='same')

    sep = max(1, int(min_sep_deg))
    peaks = []
    for idx_p in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx_p) - p) >= sep for p in peaks):
            peaks.append(int(idx_p))
        if len(peaks) >= max_candidates:
            break

    return np.asarray(sorted(set(peaks) | {0, 45}), dtype=np.float64)


def _upper_bound_area(hull_poly, angle, max_ratio, centroid):
    rot = shp_rotate(hull_poly, -angle, origin=centroid, use_radians=False)
    bw = rot.bounds[2] - rot.bounds[0]
    bh = rot.bounds[3] - rot.bounds[1]
    if max_ratio > 0.0:
        ls = max(bw, bh);
        ss = min(bw, bh)
        if ss > 0 and ls / ss > max_ratio:
            ls = ss * max_ratio
        return ls * ss * 0.5
    return bw * bh * 0.5


def _simplify_for_solve(poly):
    n_verts = len(poly.exterior.coords)
    for interior in poly.interiors:
        n_verts += len(interior.coords)
    if n_verts <= _SIMPLIFY_THRESHOLD:
        return poly, False
    minx, miny, maxx, maxy = poly.bounds
    tol = min(maxx - minx, maxy - miny) * _SIMPLIFY_TOL_FRAC
    if tol <= 0:
        return poly, False
    try:
        simplified = poly.simplify(tol, preserve_topology=True)
        if (simplified.is_empty
                or not isinstance(simplified, Polygon)
                or simplified.area <= 0):
            return poly, False
        return simplified, True
    except Exception:
        return poly, False


# ==========================================================================
# ⑥ STAGE 1 — HEURISTIC CANDIDATE GENERATOR
# ==========================================================================
def _heuristic_candidates(poly, angle_step, grid_coarse, grid_fine,
                          max_ratio, top_k, emitter=None):
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    hull = poly.convex_hull
    simplified, _ = _simplify_for_solve(poly)

    raw = []
    best_area = 0.0

    def _solve_coarse(angle_f):
        rot_s = shp_rotate(simplified, -angle_f,
                           origin=centroid, use_radians=False)
        return _solve_axis_rect_grid(rot_s, grid_coarse, max_ratio)

    edge_angles = _edge_candidate_angles(poly)
    if emitter:
        emitter.emit(
            phase="CANDIDATES", type_="edge_angles_found",
            label=f"{len(edge_angles)} edge angles",
            narration="Edge-direction angles extracted from polygon boundary.",
            angles_deg=edge_angles.tolist(),
            edge_lengths=[],
            smoothed=True,
        )

    for angle in edge_angles:
        a = float(angle)
        ub = _upper_bound_area(hull, a, max_ratio, centroid)
        if ub <= best_area * _PRUNE_MARGIN:
            if emitter:
                emitter.emit(
                    phase="CANDIDATES", type_="upper_bound_computed",
                    label=f"Angle {a:.1f}° pruned",
                    narration="Upper bound below current best; angle pruned.",
                    angle_deg=a,
                    upper_bound=round(ub, 4),
                    pruned=True,
                    prune_threshold=round(best_area * _PRUNE_MARGIN, 4),
                )
            continue
        if emitter:
            emitter.emit(
                phase="CANDIDATES", type_="upper_bound_computed",
                label=f"Angle {a:.1f}° UB={ub:.1f}",
                narration="Upper bound exceeds current best; evaluating angle.",
                angle_deg=a,
                upper_bound=round(ub, 4),
                pruned=False,
                prune_threshold=round(best_area * _PRUNE_MARGIN, 4),
            )
        rect, area = _solve_coarse(a)
        if area > 0:
            raw.append((area, a, rect))
        if area > best_area:
            best_area = area

    if len(raw) < 3:
        for a_int in range(0, 90, angle_step):
            a = float(a_int)
            if any(abs(a - ar[1]) < 2.0 for ar in raw):
                continue
            ub = _upper_bound_area(hull, a, max_ratio, centroid)
            if ub <= best_area * _PRUNE_MARGIN:
                continue
            rect, area = _solve_coarse(a)
            if area > 0:
                raw.append((area, a, rect))
            if area > best_area:
                best_area = area

    raw.sort(key=lambda t: t[0], reverse=True)

    kept = [];
    seen = []
    for area, angle, rect_rot in raw:
        if any(abs(angle - s) < 2.0 for s in seen):
            continue
        seen.append(angle)
        rect_world = shp_rotate(rect_rot, angle,
                                origin=centroid, use_radians=False)
        if emitter:
            rect_bounds = rect_world.bounds
            emitter.emit(
                phase="CANDIDATES", type_="candidate_found",
                label=f"Cand {len(kept)}: {angle:.1f}° area={area:.1f}",
                narration=f"Candidate rectangle at {angle:.1f}°.",
                angle_deg=round(angle, 4),
                rect=[float(rect_bounds[0]), float(rect_bounds[1]),
                      float(rect_bounds[2]), float(rect_bounds[3])],
                area=round(float(area), 4),
                source="grid",
                rank=len(kept),
            )
        kept.append({
            'angle': angle,
            'area': area,
            'rect_rot': rect_rot,
            'rect_world': rect_world,
            'center': (cx, cy),
        })
        if len(kept) >= top_k:
            break
    return kept


# ==========================================================================
# ⑦ PHASE A — BRENT ANGLE POLISHER
# ==========================================================================
def _polish_angle(poly, candidate, grid_coarse, max_ratio, area_cache=None):
    angle_0 = candidate['angle']
    centroid = Point(candidate['center'])
    lo, hi = angle_0 - _PHASE_A_HALFWIDTH, angle_0 + _PHASE_A_HALFWIDTH

    def _neg_area_coarse(a):
        if area_cache is not None:
            key = round(float(a), 4)
            cached = area_cache.get(key)
            if cached is not None:
                return -cached
        rot = shp_rotate(poly, -a, origin=centroid, use_radians=False)
        _, area = _solve_axis_rect_grid(rot, grid_coarse, max_ratio)
        if area_cache is not None:
            area_cache[key] = float(area)
        return -area

    try:
        res = minimize_scalar(_neg_area_coarse, bounds=(lo, hi),
                              method='bounded',
                              options={'xatol': _PHASE_A_XATOL, 'maxiter': 60})
        best_angle = float(res.x)
        if abs(best_angle - angle_0) > 0.005:
            c = candidate.copy()
            c['angle'] = best_angle
            c['area'] = float(-res.fun)
            return c
    except Exception:
        pass
    return candidate


# ==========================================================================
# ⑧ RECTANGLE FRAME HELPERS
# ==========================================================================
def _rect_local_frame(rect):
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return None
    p0 = np.array(coords[0][:2]);
    p1 = np.array(coords[1][:2])
    p2 = np.array(coords[2][:2])
    e0 = p1 - p0;
    e1 = p2 - p1
    l0 = float(np.linalg.norm(e0));
    l1 = float(np.linalg.norm(e1))
    if l0 < 1e-14 or l1 < 1e-14:
        return None
    cx = float((p0[0] + p2[0]) / 2)
    cy = float((p0[1] + p2[1]) / 2)
    if l0 >= l1:
        ux, uy = e0[0] / l0, e0[1] / l0
        vx, vy = e1[0] / l1, e1[1] / l1
        a, b = l0 / 2, l1 / 2
    else:
        ux, uy = e1[0] / l1, e1[1] / l1
        vx, vy = e0[0] / l0, e0[1] / l0
        a, b = l1 / 2, l0 / 2
    return cx, cy, ux, uy, vx, vy, a, b


def _build_rect_from_frame(cx, cy, ux, uy, vx, vy, a, b):
    corners = [
        (cx + a * ux + b * vx, cy + a * uy + b * vy),
        (cx - a * ux + b * vx, cy - a * uy + b * vy),
        (cx - a * ux - b * vx, cy - a * uy - b * vy),
        (cx + a * ux - b * vx, cy + a * uy - b * vy),
    ]
    return Polygon(corners + [corners[0]])


# ==========================================================================
# ⑨ SDF-BASED CONTAINMENT CERTIFICATION
# ==========================================================================
def _rect_sdf_max(poly, rect):
    """SDF at all 4 corners + 4 edge midpoints; return the maximum."""
    coords = list(rect.exterior.coords)
    n = len(coords)
    best = _polygon_sdf(poly, coords[0][0], coords[0][1])
    for i in range(1, n - 1):
        v = _polygon_sdf(poly, coords[i][0], coords[i][1])
        if v > best:
            best = v
        mx = (coords[i - 1][0] + coords[i][0]) * 0.5
        my = (coords[i - 1][1] + coords[i][1]) * 0.5
        v = _polygon_sdf(poly, mx, my)
        if v > best:
            best = v
    return best


def _certify_and_adjust(poly, rect, max_ratio, buf_enabled, buf_value,
                        prepared_poly=None):
    """SDF-based certification: check corners+midpoints, shrink if needed."""
    if rect is None or rect.is_empty:
        return None, 0.0

    max_sdf = _rect_sdf_max(poly, rect)

    if max_sdf <= _CERT_EPS:
        final = rect
    else:
        frame = _rect_local_frame(rect)
        if frame is None:
            return None, 0.0
        cx, cy, ux, uy, vx, vy, a, b = frame

        shrink = max_sdf + _CERT_EPS
        if shrink > min(a, b) * _CERT_MAX_SHRINK:
            return None, 0.0

        new_a = a - shrink;
        new_b = b - shrink
        if new_a <= 0 or new_b <= 0:
            return None, 0.0
        if max_ratio > 0.0 and new_b > 0 and new_a / new_b > max_ratio:
            new_a = new_b * max_ratio

        final = _build_rect_from_frame(cx, cy, ux, uy, vx, vy, new_a, new_b)
        if _rect_sdf_max(poly, final) > _CERT_EPS * 10:
            return None, 0.0

    if buf_enabled and buf_value != 0.0:
        cand = final.buffer(buf_value, cap_style=3, join_style=2)
        if not cand.is_empty and cand.area > 0:
            final = cand

    return final, float(final.area)


# ==========================================================================
# ⑩ CONSERVATIVE FALLBACK
# ==========================================================================
def _conservative_inner_fallback(poly, grid_fine, max_ratio,
                                 centroid, angles, prepared_poly=None):
    best_rect = None;
    best_area = 0.0;
    best_angle = None
    minx, miny, maxx, maxy = poly.bounds
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        return None, 0.0, None

    for frac in (0.002, 0.005, 0.01, 0.02):
        inner = poly.buffer(-span * frac, cap_style=3, join_style=2)
        if inner.is_empty or inner.area <= 0:
            continue
        if isinstance(inner, MultiPolygon):
            inner = max(inner.geoms, key=lambda g: g.area)
        if not isinstance(inner, Polygon) or inner.is_empty:
            continue
        for angle in angles:
            rot = shp_rotate(inner, -angle, origin=centroid, use_radians=False)
            rect_rot, area = _solve_axis_rect_grid(rot, grid_fine, max_ratio)
            if rect_rot is None or area <= best_area:
                continue
            rect_world = shp_rotate(rect_rot, angle,
                                    origin=centroid, use_radians=False)
            if _rect_sdf_max(poly, rect_world) <= _CERT_EPS:
                best_rect = rect_world
                best_area = float(rect_world.area)
                best_angle = angle
        if best_rect is not None:
            return best_rect, best_area, best_angle

    return None, 0.0, None


def _best_effort_shrink_to_cover(poly, rect, max_ratio,
                                 tol=1e-7, max_iter=40,
                                 prepared_poly=None):
    """SDF-based single-pass shrink — no binary search needed."""
    if rect is None or rect.is_empty:
        return None, 0.0
    frame = _rect_local_frame(rect)
    if frame is None:
        return None, 0.0
    cx, cy, ux, uy, vx, vy, a0, b0 = frame
    if a0 <= 0 or b0 <= 0:
        return None, 0.0

    max_sdf = _rect_sdf_max(poly, rect)

    if max_sdf <= _CERT_EPS:
        return rect, float(rect.area)

    shrink = max_sdf + _CERT_EPS * 2
    a = a0 - shrink;
    b = b0 - shrink
    if a <= 0 or b <= 0:
        return None, 0.0
    if max_ratio > 0.0 and b > 0 and a / b > max_ratio:
        a = b * max_ratio
    if a <= 0 or b <= 0:
        return None, 0.0

    final = _build_rect_from_frame(cx, cy, ux, uy, vx, vy, a, b)
    if _rect_sdf_max(poly, final) > _CERT_EPS:
        return None, 0.0

    return final, float(final.area)


# ==========================================================================
# ⑪ BCRS + Boundary Expansion AT A GIVEN ANGLE
# ==========================================================================
def _bcrs_expand_at_angle(rot_poly, seed_bounds, max_ratio, angle_deg=0.0, emitter=None):
    """
    Stage 4 + Stage 5 in the already-rotated frame:
    run BCRS boundary-coordinate solve, then clamped boundary expansion.
    Returns (best_rect_in_rotated_frame, area).
    """
    bcrs_rect, bcrs_area = _solve_axis_rect_bcrs(rot_poly, seed_bounds, max_ratio)

    if bcrs_rect is None:
        if seed_bounds is None:
            return None, 0.0
        sx0, sy0, sx1, sy1 = seed_bounds
        if sx1 <= sx0 or sy1 <= sy0:
            return None, 0.0
        bcrs_rect = box(sx0, sy0, sx1, sy1)
        bcrs_area = (sx1 - sx0) * (sy1 - sy0)

    if bcrs_area <= 0:
        return None, 0.0

    if emitter:
        sb = bcrs_rect.bounds
        emitter.emit(
            phase="BCRS_SOLVE", type_="bcrs_seed_set",
            label=f"BCRS seed area={bcrs_area:.1f}",
            narration=f"BCRS seed for angle {angle_deg:.1f}°.",
            angle_deg=round(angle_deg, 4),
            seed_bounds=[round(sb[0], 4), round(sb[1], 4),
                         round(sb[2], 4), round(sb[3], 4)],
            seed_area=round(bcrs_area, 4),
        )

    bx0, by0, bx1, by1 = bcrs_rect.bounds
    if emitter:
        emitter.emit(
            phase="SDF_EXPAND", type_="sdf_expand_started",
            label="SDF expansion",
            narration="SDF-guided boundary expansion starting.",
            rect_in=[round(bx0, 4), round(by0, 4),
                     round(bx1, 4), round(by1, 4)],
            area_in=round((bx1 - bx0) * (by1 - by0), 4),
        )

    bx0, by0, bx1, by1 = _expand_rect_to_boundary(
        rot_poly, bx0, by0, bx1, by1, max_ratio, emitter=emitter)

    area = (bx1 - bx0) * (by1 - by0)
    if emitter:
        emitter.emit(
            phase="SDF_EXPAND", type_="sdf_expand_done",
            label=f"SDF done: area={area:.1f}",
            narration="SDF-guided boundary expansion completed.",
            rect_out=[round(bx0, 4), round(by0, 4),
                      round(bx1, 4), round(by1, 4)],
            area_out=round(area, 4),
            delta_area=round(area - bcrs_area, 4),
        )

    if area <= 0:
        return None, 0.0
    return box(bx0, by0, bx1, by1), area


# ==========================================================================
# ⑫ STAGE 2 ORCHESTRATOR
# ==========================================================================
def _refine_best_candidate(poly, candidates, grid_coarse, grid_fine,
                           max_ratio, buf_enabled, buf_value,
                           always_return, prepared_poly=None,
                           emitter=None):
    """
    Stage 3-7 orchestrator for each Stage 2 candidate:
    Stage 3 angle refinement -> Stage 4 BCRS -> Stage 5 SDF expansion ->
    Stage 6 certification/fallback -> Stage 7 selection/output.

    Critical design: for each candidate the original Stage-1 edge-candidate
    angle is ALWAYS tested first (before the Brent-polished angle and its
    ±_ANGLE_DELTA_DEG variants). This guarantees that exact-edge-aligned
    solutions are never lost when Brent drifts due to coarse-grid artifacts.

    The strict improvement threshold (+1e-6) prevents fp noise from a
    slightly-off delta angle displacing a clean boundary-aligned result.
    """
    certified = []
    fallback_best = None
    stage3_cache = {}

    for rank, cand in enumerate(candidates):
        area_s1 = cand['area']
        centroid = Point(cand['center'])
        orig_angle = cand['angle']  # exact Stage-1 edge direction

        # Stage 3: Brent polish (coarse grid)
        cand_a = _polish_angle(
            poly, cand, grid_coarse, max_ratio, area_cache=stage3_cache
        )
        brent_angle = cand_a['angle']

        # Build angles_to_try:
        # orig_angle is prepended and wins ties via strict +1e-6 threshold.
        # Brent ± delta variants follow for sub-degree refinement.
        angles_to_try = [orig_angle]
        for delta in (0.0, _ANGLE_DELTA_DEG, -_ANGLE_DELTA_DEG):
            a_try = brent_angle + delta
            if all(abs(a_try - x) > 0.01 for x in angles_to_try):
                angles_to_try.append(a_try)

        # ── Stage 4-5: trial ranking, then BCRS + expansion on selected trials ──
        trial_data = []
        for idx_try, angle_try in enumerate(angles_to_try):
            rot_poly = shp_rotate(poly, -angle_try,
                                  origin=centroid, use_radians=False)
            # Use a cheap coarse solve to rank nearby angle trials.
            seed_rect, seed_area = _solve_axis_rect_grid(
                rot_poly, grid_coarse, max_ratio
            )
            seed_bounds = seed_rect.bounds if seed_rect is not None else None
            trial_data.append({
                'idx': idx_try,
                'angle': angle_try,
                'rot_poly': rot_poly,
                'seed_bounds': seed_bounds,
                'seed_area': float(seed_area) if seed_area is not None else 0.0,
            })

        if not trial_data:
            continue

        # Always keep original edge-derived angle; add best remaining trials.
        selected_trials = [trial_data[0]]
        other_trials = sorted(
            trial_data[1:], key=lambda t: t['seed_area'], reverse=True
        )
        for t in other_trials:
            if len(selected_trials) >= _STAGE2_MAX_TRIALS:
                break
            selected_trials.append(t)

        best_raw_r = None
        best_raw_a = 0.0
        best_angle_this = orig_angle

        for trial in selected_trials:
            angle_try = trial['angle']
            rot_poly = trial['rot_poly']
            seed_bounds = trial['seed_bounds']
            # BCRS + SDF expansion
            rect_rot, area_rot = _bcrs_expand_at_angle(
                rot_poly, seed_bounds, max_ratio,
                angle_deg=angle_try, emitter=emitter)

            if rect_rot is None or area_rot <= 0:
                continue

            # Strict improvement threshold: +1e-6 required to displace
            # the current best (first entry = orig_angle is kept unless
            # a genuinely better result emerges).
            if area_rot > best_raw_a + 1e-6:
                best_raw_a = area_rot
                best_raw_r = shp_rotate(rect_rot, angle_try,
                                        origin=centroid,
                                        use_radians=False)
                best_angle_this = angle_try
            elif best_raw_r is None:
                best_raw_a = area_rot
                best_raw_r = shp_rotate(rect_rot, angle_try,
                                        origin=centroid,
                                        use_radians=False)
                best_angle_this = angle_try

        if best_raw_r is None:
            continue

        if fallback_best is None or best_raw_a > fallback_best['area']:
            fallback_best = {
                'rect': best_raw_r,
                'area': best_raw_a,
                'angle': best_angle_this,
                'rank': rank,
            }

        # Stage 6: certification
        if emitter:
            emitter.emit(
                phase="CERT", type_="cert_started",
                label="Certification",
                narration="Verifying rectangle containment.",
                rect=[float(best_raw_r.bounds[0]), float(best_raw_r.bounds[1]),
                      float(best_raw_r.bounds[2]), float(best_raw_r.bounds[3])],
                area=round(best_raw_a, 4),
                method="covers",
            )

        best_r, best_a = _certify_and_adjust(
            poly, best_raw_r, max_ratio, False, 0.0, prepared_poly)
        used_best_effort = False

        if best_r is not None:
            if emitter:
                emitter.emit(
                    phase="CERT", type_="cert_passed",
                    label="Cert passed",
                    narration="Rectangle fully inside polygon.",
                    rect=[float(best_r.bounds[0]), float(best_r.bounds[1]),
                          float(best_r.bounds[2]), float(best_r.bounds[3])],
                    area=round(best_a, 4),
                    inset=round(best_raw_a - best_a, 6),
                )
        elif always_return:
            if emitter:
                emitter.emit(
                    phase="CERT", type_="cert_failed_shrink",
                    label="Cert failed, shrinking",
                    narration="Shrinking rectangle for containment.",
                    attempt=1,
                    rect_before=[float(best_raw_r.bounds[0]),
                                 float(best_raw_r.bounds[1]),
                                 float(best_raw_r.bounds[2]),
                                 float(best_raw_r.bounds[3])],
                    rect_after=[0, 0, 0, 0],
                    eps=1e-7,
                )
                emitter.emit(
                    phase="CERT", type_="cert_fallback",
                    label="Fallback invoked",
                    narration="Best-effort shrink fallback.",
                    reason="shrink_exhausted",
                    fallback="best_effort_shrink",
                )

        if best_r is None and always_return:
            best_r, best_a = _best_effort_shrink_to_cover(
                poly, best_raw_r, max_ratio, prepared_poly=prepared_poly)
            used_best_effort = best_r is not None

        if best_r is None:
            continue

        # Post-rotation SDF check — shp_rotate introduces fp noise even at 0°.
        if best_r is not None and _rect_sdf_max(poly, best_r) > _CERT_EPS:
            best_r2, best_a2 = _certify_and_adjust(
                poly, best_r, max_ratio, False, 0.0, prepared_poly)
            if best_r2 is not None:
                best_r, best_a = best_r2, best_a2
            else:
                continue

        if emitter:
            poly_area = float(poly.area)
            pct = (best_a / poly_area * 100) if poly_area > 0 else 0.0
            emitter.emit(
                phase="RESULT", type_="best_updated",
                label=f"Best: area={best_a:.1f} ({pct:.1f}%)",
                narration="Best rectangle after BCRS+expansion.",
                rect=[float(best_r.bounds[0]), float(best_r.bounds[1]),
                      float(best_r.bounds[2]), float(best_r.bounds[3])],
                area=round(best_a, 4),
                pct_polygon=round(pct, 2),
                angle_deg=round(best_angle_this, 4),
                source="BCRS",
                prev_area=round(emitter._best_area, 4),
            )
            emitter._best_area = best_a

        if buf_enabled and buf_value != 0.0:
            cand_buf = best_r.buffer(buf_value, cap_style=3, join_style=2)
            if not cand_buf.is_empty and cand_buf.area > 0:
                best_r = cand_buf
                best_a = float(best_r.area)

        coords = list(best_r.exterior.coords)
        w = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        h = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

        certified.append({
            'rect': best_r,
            'area': best_a,
            'angle': best_angle_this,
            'ratio': ratio,
            'rank': rank,
            'stage2_gain': best_a - area_s1,
            'used_best_effort': used_best_effort,
        })

    # ── Fallback paths ─────────────────────────────────────────────────────
    if not certified:
        if always_return and fallback_best is not None:
            rect_fb, area_fb = _best_effort_shrink_to_cover(
                poly, fallback_best['rect'], max_ratio,
                prepared_poly=prepared_poly)
            if rect_fb is not None:
                coords = list(rect_fb.exterior.coords)
                w = math.hypot(coords[1][0] - coords[0][0],
                               coords[1][1] - coords[0][1])
                h = math.hypot(coords[2][0] - coords[1][0],
                               coords[2][1] - coords[1][1])
                ratio_fb = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                return (rect_fb, area_fb, fallback_best['angle'], ratio_fb,
                        fallback_best['rank'], area_fb, True)

        centroid_fb = Point(candidates[0]['center'])
        rescue_angs = [c['angle']
                       for c in candidates[:max(3, min(len(candidates), 8))]]
        rect_c, area_c, angle_c = _conservative_inner_fallback(
            poly, grid_fine, max_ratio, centroid_fb,
            rescue_angs, prepared_poly)
        if rect_c is not None:
            coords = list(rect_c.exterior.coords)
            w = math.hypot(coords[1][0] - coords[0][0],
                           coords[1][1] - coords[0][1])
            h = math.hypot(coords[2][0] - coords[1][0],
                           coords[2][1] - coords[1][1])
            ratio_c = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            return (rect_c, area_c, angle_c, ratio_c,
                    fallback_best['rank'] if fallback_best else 0,
                    area_c, True)
        return None

    best = max(certified, key=lambda c: c['area'])
    return (best['rect'], best['area'], best['angle'],
            best['ratio'], best['rank'], best['stage2_gain'],
            best['used_best_effort'])


# ==========================================================================
# ⑬ FAST-PATH: SIMPLE CONVEX POLYGONS
# ==========================================================================

def _maybe_fast_path(poly, max_ratio=0.0):
    """
    Return the optimal rectangle for simple convex polygons where the
    optimal LIR is guaranteed edge-aligned, skipping the full BCRS+expansion
    pipeline.

    Returns (rect, area, angle, ratio) or None.
    """
    coords = list(poly.exterior.coords)[:-1]
    nv = len(coords)
    has_holes = bool(poly.interiors)

    # ── Rectangle (identity) ──────────────────────────────────────────────
    if nv == 4 and not has_holes:
        for i in range(4):
            p0 = np.array(coords[i])
            p1 = np.array(coords[(i + 1) % 4])
            p2 = np.array(coords[(i + 2) % 4])
            v1 = p1 - p0;  v2 = p2 - p1
            n1 = float(np.linalg.norm(v1));  n2 = float(np.linalg.norm(v2))
            if n1 > 0 and n2 > 0 and abs(np.dot(v1, v2) / (n1 * n2)) > 1e-6:
                break
        else:
            a = float(poly.area)
            frame = _rect_local_frame(poly)
            ang = math.degrees(math.atan2(frame[3], frame[2])) % 90.0 if frame else 0.0
            r_ = 1.0
            if a > 0:
                cp = list(poly.exterior.coords)
                wp = math.hypot(cp[1][0] - cp[0][0], cp[1][1] - cp[0][1])
                hp = math.hypot(cp[2][0] - cp[1][0], cp[2][1] - cp[1][1])
                r_ = max(wp, hp) / min(wp, hp) if min(wp, hp) > 0 else 1.0
            return poly, a, ang, r_

    # ── Simple convex (≤ 8 vertices, near-convex, no holes) ──────────────
    if has_holes or nv < 3 or nv > 8:
        return None

    hull = poly.convex_hull
    hull_area = float(hull.area)
    poly_area = float(poly.area)
    if poly_area <= 0 or hull_area / poly_area > 1.005:
        return None

    raw_angles = []
    for hci in range(len(hull.exterior.coords) - 1):
        dx = hull.exterior.coords[hci + 1][0] - hull.exterior.coords[hci][0]
        dy = hull.exterior.coords[hci + 1][1] - hull.exterior.coords[hci][1]
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            a = math.degrees(math.atan2(dy, dx)) % 90.0
            if not any(abs(a - ra) < 1.0 for ra in raw_angles):
                raw_angles.append(a)
    for fixed in (0, 45):
        if not any(abs(fixed - ra) < 1.0 for ra in raw_angles):
            raw_angles.append(float(fixed))
    raw_angles.sort()

    centroid = poly.centroid
    best_rect = None
    best_area = 0.0
    best_angle = 0.0

    for a in raw_angles:
        rot = shp_rotate(poly, -a, origin=centroid, use_radians=False)
        seed, _ = _solve_axis_rect_grid(rot, 60, max_ratio)
        if seed is None:
            continue
        sb = seed.bounds
        bx0, by0, bx1, by1 = _expand_rect_to_boundary(
            rot, sb[0], sb[1], sb[2], sb[3], max_ratio)
        area = (bx1 - bx0) * (by1 - by0)
        if area <= best_area:
            continue
        rect_r = box(bx0, by0, bx1, by1)
        rect_w = shp_rotate(rect_r, a, origin=centroid, use_radians=False)
        cert_r, cert_a = _certify_and_adjust(poly, rect_w, max_ratio, False, 0.0)
        if cert_r is not None and cert_a > best_area:
            best_rect, best_area, best_angle = cert_r, cert_a, a

    if best_rect is None:
        return None
    cf = list(best_rect.exterior.coords)
    wf = math.hypot(cf[1][0] - cf[0][0], cf[1][1] - cf[0][1])
    hf = math.hypot(cf[2][0] - cf[1][0], cf[2][1] - cf[1][1])
    rf = max(wf, hf) / min(wf, hf) if min(wf, hf) > 0 else 1.0
    return best_rect, best_area, best_angle, rf


# ==========================================================================
# ⑭ GEOMETRY PREPARATION
# ==========================================================================
def _prepare_polygon(geom):
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
    elif hasattr(geom, 'geoms') and not isinstance(geom, Polygon):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and not g.is_empty and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)

    if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= 0:
        return None
    return geom


# ==========================================================================
# ⑭ PUBLIC ENTRY POINT
# ==========================================================================
def _worker_process_feature(args, emitter=None):
    """
    Stateless worker — safe for ThreadPoolExecutor and ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (feat_id, wkb_bytes, angle_step, grid_coarse, grid_fine,
         max_ratio, buf_enabled, buf_value, top_k, always_return)
    emitter : TraceEmitter or None
        Optional event emitter for visualisation traces.

    Returns
    -------
    tuple or None
        (feat_id, wkt, area, angle_deg, ratio,
         cand_rank, stage2_gain, used_best_effort)
    """
    (feat_id, wkb_bytes, angle_step, grid_coarse, grid_fine,
     max_ratio, buf_enabled, buf_value, top_k, always_return) = args

    try:
        poly = _prepare_polygon(wkb_loads(bytes(wkb_bytes)))
        if poly is None:
            return None

        try:
            from shapely import set_precision
            minx, miny, maxx, maxy = poly.bounds
            span = max(maxx - minx, maxy - miny)
            if span > 0:
                poly = set_precision(poly,
                                     grid_size=span * 1e-9,
                                     mode='valid_output')
        except Exception:
            pass

        if poly is None or poly.is_empty:
            return None

        # ── Fast path: simple convex cases → skip BCRS entirely ─────────
        fast = _maybe_fast_path(poly, max_ratio=max_ratio)
        if fast is not None:
            fp_r, fp_a, fp_ang, fp_rat = fast
            if emitter:
                fp_eb = fp_r.bounds
                emitter.emit(
                    phase="RESULT", type_="final_result",
                    label=f"Fast-path: area={fp_a:.1f}",
                    narration="Solved via edge-aligned fast-path.",
                    rect=[float(fp_eb[0]), float(fp_eb[1]),
                          float(fp_eb[2]), float(fp_eb[3])],
                    area=round(float(fp_a), 4),
                    angle_deg=round(float(fp_ang), 4),
                    algorithm="bcrs_fast_fastpath",
                )
            return (feat_id, fp_r.wkt,
                    round(float(fp_a), 4), round(float(fp_ang), 4),
                    round(float(fp_rat), 4), 0, 0.0, 0)

        if emitter:
            ext_coords = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
            hole_coords = [[[float(x), float(y)] for x, y in r.coords[:-1]] for r in poly.interiors]
            emitter.emit(
                phase="SETUP", type_="polygon_loaded",
                label="Polygon loaded",
                narration="Polygon loaded for BCRS fast solve.",
                exterior=ext_coords,
                holes=hole_coords,
                bbox=[float(poly.bounds[0]), float(poly.bounds[1]),
                      float(poly.bounds[2]), float(poly.bounds[3])],
                area=float(poly.area),
                vertex_count=len(poly.exterior.coords) - 1,
                poly_type="concave_no_holes",
                is_valid=poly.is_valid,
            )

        prepared_poly = _make_prepared(poly)

        candidates = _heuristic_candidates(
            poly, angle_step, grid_coarse, grid_fine, max_ratio, top_k,
            emitter=emitter)
        if not candidates:
            return None

        result = _refine_best_candidate(
            poly, candidates, grid_coarse, grid_fine,
            max_ratio, buf_enabled, buf_value, always_return,
            prepared_poly=prepared_poly, emitter=emitter)

        if result is None:
            return None

        rect, area, angle, ratio, rank, gain, used_best_effort = result

        if emitter:
            rect_bounds = rect.bounds
            poly_area = float(poly.area)
            pct = (area / poly_area * 100) if poly_area > 0 else 0.0
            emitter.emit(
                phase="RESULT", type_="final_result",
                label=f"Final: area={area:.1f} ({pct:.1f}%)",
                narration="BCRS fast LIR solve complete.",
                rect=[float(rect_bounds[0]), float(rect_bounds[1]),
                      float(rect_bounds[2]), float(rect_bounds[3])],
                area=round(float(area), 4),
                pct_polygon=round(pct, 2),
                angle_deg=round(float(angle), 4),
                algorithm="bcrs_fast",
                total_events=len(emitter.events),
                elapsed_ms=round(time.monotonic() * 1000 - emitter._start_ms, 2),
            )

        return (
            feat_id,
            rect.wkt,
            round(float(area), 4),
            round(float(angle), 4),
            round(float(ratio), 4),
            int(rank),
            round(float(gain), 6),
            int(used_best_effort),
        )

    except Exception as e:
        raise RuntimeError(
            f'_worker_process_feature failed for feat_id={feat_id}: {e}'
        ) from e


def process_slice(job_array, start, end,
                  angle_step, grid_coarse, grid_fine,
                  max_ratio, buf_enabled, buf_value,
                  top_k, always_return):
    """
    Process a slice of job_array in one worker.

    Parameters
    ----------
    job_array : list
        List of (feat_id, wkb_bytes) tuples.
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).
    angle_step : float
        Angle step for fallback sweep.
    grid_coarse : int
        Coarse grid resolution.
    grid_fine : int
        Fine grid resolution.
    max_ratio : float
        Maximum aspect ratio (0 = unlimited).
    buf_enabled : bool
        Enable containment buffer.
    buf_value : float
        Buffer distance.
    top_k : int
        Number of candidates to keep.
    always_return : bool
        Enable best-effort fallback.

    Returns
    -------
    tuple
        (results: dict, best_effort_count: int)
        results: {feat_id: (wkt, area, angle, ratio, cand_rank, s2_gain, best_effort)}
    """
    out = {}
    best_effort_count = 0
    for i in range(start, end):
        feat_id, wkb_bytes = job_array[i]
        try:
            res = _worker_process_feature((
                feat_id, wkb_bytes,
                angle_step, grid_coarse, grid_fine,
                max_ratio, buf_enabled, buf_value,
                top_k, always_return,
            ))
        except Exception:
            continue
        if res is None:
            continue
        (_, wkt, area, angle, ratio,
         cand_rank, s2_gain, best_effort) = res
        out[feat_id] = (wkt, area, angle, ratio, cand_rank, s2_gain, best_effort)
        best_effort_count += int(best_effort)
    return out, best_effort_count
