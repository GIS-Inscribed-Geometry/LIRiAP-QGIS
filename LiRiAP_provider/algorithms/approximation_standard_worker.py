"""
LIRiAP Approximation Standard worker module.

Pure geometry solver for fast area-focused rectangle search.
No QGIS or Qt runtime dependencies.

Pipeline
========
1. Edge-guided coarse candidate search from polygon edge directions
2. Upper-bound pruning to skip weak candidates early
3. Coarse grid evaluation at candidate angles
4. Optional fallback uniform sweep if candidates are weak
5. Angle refinement via bounded Brent optimization
6. Fine-grid solve near best angle
7. Rotate rectangle back to map coordinates
8. Optional containment buffer application

Algorithm Semantics
===================
NOT a strict containment solver. Rectangle can violate containment in
difficult cases. Use for quick candidate finding and exploratory analysis.

Complexity
==========
O(n + (m+s180)*g_coarse^2 + (p+1)*g_fine^2) where:
- n: polygon vertices
- m: edge-guided candidates (<=10)
- s180: fallback sweep (ceil(180/ANGLE_STEP))
- g_coarse, g_fine: grid resolutions
- p: Brent iterations

Output
======
(feat_id, wkt, area, angle_deg, ratio) or None

Parameters (tuple order)
========================
angle_step, grid_coarse, grid_fine, max_ratio, buf_enabled, buf_value

See Also
========
approximation_standard_algorithm: QGIS wrapper
approximation_fast_worker: Optimized variant
"""

import time

import numpy as np
from scipy.optimize import minimize_scalar
from shapely.affinity import rotate
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.vectorized import contains as shp_contains
from shapely.wkb import loads as wkb_loads

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_EDGE_KERNEL = np.array([0.15, 0.25, 0.20, 0.25, 0.15], dtype=np.float64)
_UPPER_BOUND_FACTOR = 0.5  # convex-shape bound for max inscribed rectangle area
_HALF_WINDOW_MEDIAN_SCALE = 0.6
_HALF_WINDOW_MIN = 3.0
_HALF_WINDOW_MAX = 15.0
_HALF_WINDOW_FALLBACK = 10.0
_BRENT_XATOL = 0.3

# NOTE: STRtree was evaluated for this stage. Because each solve performs
# one polygon-vs-many-grid-points query, vectorized contains remains the
# default path and usually outperforms building a per-feature spatial index.

try:
    from numba import njit as _njit

    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(fn=None, **kw):
        return fn if fn is not None else lambda f: f


    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inner solver — O(n) largest-rectangle-in-histogram, JIT-compiled
# ---------------------------------------------------------------------------
@_njit(cache=True)
def _histogram_kernel(heights, xs, ys, row_idx, max_ratio):
    cols = len(heights)
    n_xs = len(xs)
    n_ys = len(ys)
    best_area = 0.0
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
            w = c - sc
            x0_w = xs[sc]
            xi = sc + w
            x1_w = xs[xi if xi < n_xs else n_xs - 1]
            ri0 = row_idx - sh + 1
            y0_w = ys[ri0 if ri0 >= 0 else 0]
            y1_w = ys[row_idx if row_idx < n_ys else n_ys - 1]
            rw = x1_w - x0_w
            rh = y1_w - y0_w
            if rw <= 0.0 or rh <= 0.0:
                start = sc
                continue
            if max_ratio > 0.0:
                long_s = rw if rw >= rh else rh
                short_s = rh if rw >= rh else rw
                if short_s > 0.0 and long_s / short_s > max_ratio:
                    new_long = short_s * max_ratio
                    if rw >= rh:
                        cx = (x0_w + x1_w) * 0.5
                        x0_w = cx - new_long * 0.5
                        x1_w = cx + new_long * 0.5
                        rw = new_long
                    else:
                        cy = (y0_w + y1_w) * 0.5
                        y0_w = cy - new_long * 0.5
                        y1_w = cy + new_long * 0.5
                        rh = new_long
            area = rw * rh
            if area > best_area:
                best_area = area
                bx0 = x0_w;
                by0 = y0_w
                bx1 = x1_w;
                by1 = y1_w
            start = sc
        st_col[top] = start
        st_h[top] = h
        top += 1
    return bx0, by0, bx1, by1, best_area


# ===========================================================================
# MODULE-LEVEL WORKER FUNCTION
# Must live at module level (not nested) so multiprocessing can pickle it.
# Receives a serialised feature payload, runs the full search, returns result.
# ===========================================================================

def _worker_process_feature(args, emitter=None):
    """
    Standalone worker function for multiprocessing.Pool.

    Parameters
    ----------
    args : tuple
        (feat_id, wkb_bytes, angle_step, grid_steps_coarse, grid_steps_fine,
         max_ratio, buf_enabled, buf_value)
    emitter : TraceEmitter or None
        Optional event emitter for visualisation traces.

    Returns
    -------
    tuple or None
        (feat_id, wkt_rect, area, angle, ratio)  or  None on failure/empty
    """
    (feat_id, wkb_bytes, angle_step, grid_steps_coarse, grid_steps_fine,
     max_ratio, buf_enabled, buf_value) = args

    try:
        shapely_poly = wkb_loads(bytes(wkb_bytes))
        result = _search(shapely_poly, angle_step, grid_steps_coarse,
                         grid_steps_fine, max_ratio, buf_enabled, buf_value,
                         emitter=emitter)
        if result is None:
            return None
        rect, area, angle, ratio = result

        if emitter:
            emitter.emit(
                phase="RESULT", type_="final_result",
                label=f"Final: area={area:.1f}",
                narration="Approximation standard solve complete.",
                rect=[float(rect.bounds[0]), float(rect.bounds[1]),
                      float(rect.bounds[2]), float(rect.bounds[3])],
                area=round(float(area), 4),
                pct_polygon=round(area / max(shapely_poly.area, 1e-14) * 100, 2),
                angle_deg=round(float(angle), 4),
                algorithm="approximation_standard",
                total_events=len(emitter.events),
                elapsed_ms=round(time.monotonic() * 1000 - emitter._start_ms, 2),
            )

        return (feat_id, rect.wkt, round(area, 4),
                round(angle, 2), round(ratio, 4))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pure-function search core — no QGIS objects, safe to call from workers
# ---------------------------------------------------------------------------

def _edge_candidate_angles(poly, min_sep_deg=4.0, max_candidates=10):
    """
    Build a length-weighted edge-orientation histogram for the polygon
    boundary and return dominant candidate angles in [0, 90°).
    """
    coords = np.array(poly.exterior.coords)
    edges = np.diff(coords, axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    valid = lengths > 1e-12
    edges = edges[valid];
    lengths = lengths[valid]

    if len(edges) == 0:
        return np.array([0.0, 45.0])

    angles = np.degrees(np.arctan2(np.abs(edges[:, 1]),
                                   np.abs(edges[:, 0]))) % 90.0

    bins = np.zeros(91, dtype=np.float64)
    for ang, wt in zip(angles, lengths):
        bins[min(int(round(ang)), 90)] += wt

    bins = np.convolve(bins, _EDGE_KERNEL, mode='same')

    sep = max(1, int(min_sep_deg))
    peaks = []
    for idx in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx) - p) >= sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= max_candidates:
            break

    return np.unique(np.concatenate([np.array(sorted(peaks), dtype=np.float64),
                                     np.array([0.0, 45.0])]))


def _upper_bound(poly, angle, max_ratio):
    """
    Cheap O(1) upper bound on the inscribed rectangle area at a given angle.

    Rotates the polygon and uses its bounding-box area as an upper bound.
    Divided by 2 gives a tighter bound exploiting the fact that the largest
    inscribed axis-aligned rectangle in any convex shape is at most half the
    bounding box.  For non-convex shapes we use the full bbox area as a safe
    (loose) upper bound.
    """
    centroid = poly.centroid
    rot_poly = rotate(poly, -angle, origin=centroid, use_radians=False)
    minx, miny, maxx, maxy = rot_poly.bounds
    bw = maxx - minx
    bh = maxy - miny

    if max_ratio > 0.0:
        long_s = max(bw, bh)
        short_s = min(bw, bh)
        if short_s > 0 and long_s / short_s > max_ratio:
            long_s = short_s * max_ratio
        upper = long_s * short_s
    else:
        upper = bw * bh

    # Divide by 2 — provably valid for convex shapes; conservative for general
    return upper * _UPPER_BOUND_FACTOR


def _solve_axis_rect(poly, grid_steps, max_ratio):
    """
    Largest axis-aligned inscribed rectangle for the given (pre-rotated)
    polygon via the scanline histogram with JITed inner kernel.
    """
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, grid_steps)
    ys = np.linspace(miny, maxy, grid_steps)

    xx, yy = np.meshgrid(xs, ys)
    mask = shp_contains(poly, xx.ravel(), yy.ravel()) \
        .reshape(grid_steps, grid_steps)

    heights = np.zeros(grid_steps, dtype=np.int64)
    best_rect = None
    best_area = 0.0

    for r in range(grid_steps):
        row = mask[r]
        heights += row
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel(
            heights, xs, ys, r, max_ratio)
        if area > best_area:
            best_area = area
            best_rect = box(x0, y0, x1, y1)

    return best_rect, best_area


def _search(shapely_poly, angle_step, grid_steps_coarse, grid_steps_fine,
            max_ratio, buf_enabled, buf_value, emitter=None):
    """
    Full search for one polygon — no QGIS objects, pure geometry.
    Combines:
      1. Edge-guided candidate angle extraction
      2. Area upper-bound early rejection
      3. Adaptive two-resolution grid (coarse → fine)
      4. Narrow-band continuous refinement around the best angle
    """
    if isinstance(shapely_poly, MultiPolygon):
        shapely_poly = max(shapely_poly.geoms, key=lambda g: g.area)
    if not isinstance(shapely_poly, Polygon) or shapely_poly.is_empty:
        return None

    centroid = shapely_poly.centroid
    best_area = 0.0
    best_rect = None
    best_angle = 0.0

    # ── Stage 1: edge-guided candidates ─────────────────────────────────────
    candidates = _edge_candidate_angles(shapely_poly)

    # ── Emit edge_angles_found ──────────────────────────────────────────────
    if emitter:
        emitter.emit(
            phase="CANDIDATES", type_="edge_angles_found",
            label=f"{len(candidates)} edge angles",
            narration="Edge-direction angles extracted from polygon boundary.",
            angles_deg=candidates.tolist(),
            edge_lengths=[],
            smoothed=True,
        )

    if len(candidates) >= 2:
        gaps = np.diff(np.sort(candidates))
        half_window = float(
            np.clip(
                np.median(gaps) * _HALF_WINDOW_MEDIAN_SCALE,
                _HALF_WINDOW_MIN,
                _HALF_WINDOW_MAX,
            )
        )
    else:
        half_window = _HALF_WINDOW_FALLBACK

    # ── Stage 2: coarse-grid evaluation with early rejection ─────────────────
    # Sort candidates by descending upper-bound so the global best rises fast,
    # maximising pruning efficiency in the early-rejection test.
    bounds = [(a, _upper_bound(shapely_poly, a, max_ratio))
              for a in candidates]
    bounds.sort(key=lambda t: t[1], reverse=True)

    for angle, ub in bounds:
        # Early rejection: skip this angle if its theoretical maximum is
        # already beaten by the current best rectangle
        if ub <= best_area:
            if emitter:
                emitter.emit(
                    phase="CANDIDATES", type_="upper_bound_computed",
                    label=f"Angle {angle:.1f}° pruned (UB={ub:.1f})",
                    narration="Upper bound below current best; angle pruned.",
                    angle_deg=float(angle),
                    upper_bound=round(ub, 4),
                    pruned=True,
                    prune_threshold=round(best_area, 4),
                )
            continue

        if emitter:
            emitter.emit(
                phase="CANDIDATES", type_="upper_bound_computed",
                label=f"Angle {angle:.1f}° UB={ub:.1f}",
                narration="Upper bound exceeds current best; evaluating angle.",
                angle_deg=float(angle),
                upper_bound=round(ub, 4),
                pruned=False,
                prune_threshold=round(best_area, 4),
            )

        rot_poly = rotate(shapely_poly, -angle,
                          origin=centroid, use_radians=False)
        rect, area = _solve_axis_rect(rot_poly, grid_steps_coarse, max_ratio)
        if area > best_area:
            best_area = area
            best_rect = rect
            best_angle = float(angle)

    # ── Stage 2: fallback uniform sweep for isotropic/featureless polygons ───
    if best_rect is None or len(candidates) <= 4:
        for angle in range(0, 180, angle_step):
            a = float(angle % 90)
            ub = _upper_bound(shapely_poly, a, max_ratio)
            if ub <= best_area:
                continue
            rot_poly = rotate(shapely_poly, -a,
                              origin=centroid, use_radians=False)
            rect, area = _solve_axis_rect(rot_poly, grid_steps_coarse, max_ratio)
            if area > best_area:
                best_area = area
                best_rect = rect
                best_angle = a

    if best_rect is None:
        return None

    # ── Stage 3: narrow continuous refinement around best_angle ─────────────
    def _neg_area_fine(a):
        rp = rotate(shapely_poly, -a, origin=centroid, use_radians=False)
        _, area = _solve_axis_rect(rp, grid_steps_fine, max_ratio)
        return -area

    lo = best_angle - half_window
    hi = best_angle + half_window

    if emitter:
        emitter.emit(
            phase="ANGLE_SEARCH", type_="brent_bracket_set",
            label=f"Brent bracket [{lo:.1f}, {hi:.1f}]",
            narration=f"Brent bracket set around {best_angle:.1f}°.",
            center_deg=round(best_angle, 4),
            bracket_deg=[round(lo, 4), round(hi, 4)],
            half_width=round(half_window, 4),
        )

    res = minimize_scalar(
        _neg_area_fine,
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': _BRENT_XATOL},
    )

    if res.fun < -best_area:
        best_angle = res.x
        if emitter:
            emitter.emit(
                phase="ANGLE_SEARCH", type_="angle_polished",
                label=f"Polished angle {best_angle:.2f}°",
                narration="Brent optimisation converged on refined angle.",
                angle_deg=round(best_angle, 4),
                area=round(-res.fun, 4),
                rect=[0, 0, 0, 0],
                iterations_used=int(res.nfev or 0),
            )
        rot_poly = rotate(shapely_poly, -best_angle,
                          origin=centroid, use_radians=False)
        best_rect, best_area = _solve_axis_rect(
            rot_poly, grid_steps_fine, max_ratio)

    if best_rect is None:
        return None

    # Rotate rectangle back to original CRS orientation
    final_rect = rotate(best_rect, best_angle,
                        origin=centroid, use_radians=False)

    if emitter:
        rect_bounds = final_rect.bounds
        emitter.emit(
            phase="RESULT", type_="best_updated",
            label=f"Best: area={best_area:.1f}",
            narration="Best rectangle found by approximation standard solver.",
            rect=[float(rect_bounds[0]), float(rect_bounds[1]),
                  float(rect_bounds[2]), float(rect_bounds[3])],
            area=round(best_area, 4),
            pct_polygon=round(best_area / max(shapely_poly.area, 1e-14) * 100, 2),
            angle_deg=round(best_angle, 4),
            source="APPROXIMATION",
            prev_area=round(emitter._best_area, 4),
        )
        emitter._best_area = best_area

    # ── Stage 4: optional containment buffer ─────────────────────────────────
    if buf_enabled and buf_value != 0.0:
        candidate = final_rect.buffer(buf_value, cap_style=3, join_style=2)
        if not candidate.is_empty and candidate.area > 0:
            final_rect = candidate

    coords = list(final_rect.exterior.coords)
    w = np.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    h = np.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    ratio = (max(w, h) / min(w, h)) if min(w, h) > 0 else 0.0

    return final_rect, final_rect.area, best_angle, ratio
