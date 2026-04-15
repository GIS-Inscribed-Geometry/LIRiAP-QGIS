
"""
inscribed_rect_worker_claude.py
========================
Pure-geometry worker module — NO QGIS / Qt imports.
Workers import this file cleanly under all platforms.
"""

"""
LIRiAP Approximation Fast worker module.

Pure geometry routines used by the QGIS algorithm wrapper. No QGIS/Qt imports.
"""

import numpy as np
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.affinity import rotate
from shapely.vectorized import contains as shp_contains
from shapely.wkb import loads as wkb_loads
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Numba JIT — graceful no-op fallback
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(fn=None, **kw):
        return fn if fn is not None else lambda f: f
    _NUMBA_AVAILABLE = False


@_njit(cache=True)
def _histogram_kernel(heights, xs, ys, row_idx, max_ratio):
    cols      = len(heights)
    n_xs      = len(xs)
    n_ys      = len(ys)
    best_area = 0.0
    bx0 = by0 = bx1 = by1 = 0.0
    st_col = np.empty(cols + 1, dtype=np.int64)
    st_h   = np.empty(cols + 1, dtype=np.int64)
    top    = 0
    for c in range(cols + 1):
        h     = int(heights[c]) if c < cols else 0
        start = c
        while top > 0 and st_h[top - 1] > h:
            top -= 1
            sc = st_col[top]
            sh = st_h[top]
            w  = c - sc
            x0_w = xs[sc]
            xi   = sc + w
            x1_w = xs[xi if xi < n_xs else n_xs - 1]
            ri0  = row_idx - sh + 1
            y0_w = ys[ri0 if ri0 >= 0 else 0]
            y1_w = ys[row_idx if row_idx < n_ys else n_ys - 1]
            rw = x1_w - x0_w
            rh = y1_w - y0_w
            if rw <= 0.0 or rh <= 0.0:
                start = sc
                continue
            if max_ratio > 0.0:
                long_s  = rw if rw >= rh else rh
                short_s = rh if rw >= rh else rw
                if short_s > 0.0 and long_s / short_s > max_ratio:
                    new_long = short_s * max_ratio
                    if rw >= rh:
                        cx = (x0_w + x1_w) * 0.5
                        x0_w = cx - new_long * 0.5
                        x1_w = cx + new_long * 0.5
                        rw   = new_long
                    else:
                        cy = (y0_w + y1_w) * 0.5
                        y0_w = cy - new_long * 0.5
                        y1_w = cy + new_long * 0.5
                        rh   = new_long
            area = rw * rh
            if area > best_area:
                best_area = area
                bx0 = x0_w; by0 = y0_w
                bx1 = x1_w; by1 = y1_w
            start = sc
        st_col[top] = start
        st_h[top]   = h
        top += 1
    return bx0, by0, bx1, by1, best_area


def _edge_candidate_angles(poly, min_sep_deg=4.0, max_candidates=10):
    coords  = np.array(poly.exterior.coords)
    edges   = np.diff(coords, axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    valid   = lengths > 1e-12
    edges   = edges[valid]
    lengths = lengths[valid]
    if len(edges) == 0:
        return np.array([0.0, 45.0])
    angles = np.degrees(np.arctan2(np.abs(edges[:, 1]),
                                   np.abs(edges[:, 0]))) % 90.0
    bins = np.zeros(91, dtype=np.float64)
    for ang, wt in zip(angles, lengths):
        bins[min(int(round(ang)), 90)] += wt
    kernel = np.array([0.15, 0.25, 0.20, 0.25, 0.15])
    bins   = np.convolve(bins, kernel, mode='same')
    sep    = max(1, int(min_sep_deg))
    peaks  = []
    for idx in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx) - p) >= sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= max_candidates:
            break
    return np.unique(np.concatenate([
        np.array(sorted(peaks), dtype=np.float64),
        np.array([0.0, 45.0])
    ]))


def _upper_bound(poly, angle, max_ratio, centroid):
    rot_poly = rotate(poly, -angle, origin=centroid, use_radians=False)
    minx, miny, maxx, maxy = rot_poly.bounds
    bw = maxx - minx
    bh = maxy - miny
    if max_ratio > 0.0:
        long_s  = max(bw, bh)
        short_s = min(bw, bh)
        if short_s > 0 and long_s / short_s > max_ratio:
            long_s = short_s * max_ratio
        upper = long_s * short_s
    else:
        upper = bw * bh
    return upper * 0.5


def _solve_axis_rect(poly, grid_steps, max_ratio):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, grid_steps)
    ys = np.linspace(miny, maxy, grid_steps)
    xx, yy = np.meshgrid(xs, ys)
    mask   = shp_contains(poly, xx.ravel(), yy.ravel()) \
                 .reshape(grid_steps, grid_steps)
    heights   = np.zeros(grid_steps, dtype=np.int64)
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


def _search_one(shapely_poly, angle_step, grid_coarse, grid_fine,
                max_ratio, buf_enabled, buf_value):
    if isinstance(shapely_poly, MultiPolygon):
        shapely_poly = max(shapely_poly.geoms, key=lambda g: g.area)
    if not isinstance(shapely_poly, Polygon) or shapely_poly.is_empty:
        return None

    centroid   = shapely_poly.centroid
    best_area  = 0.0
    best_rect  = None
    best_angle = 0.0

    candidates = _edge_candidate_angles(shapely_poly)
    if len(candidates) >= 2:
        gaps = np.diff(np.sort(candidates))
        half_window = float(np.clip(np.median(gaps) * 0.6, 3.0, 15.0))
    else:
        half_window = 10.0

    bounds = [(a, _upper_bound(shapely_poly, a, max_ratio, centroid))
              for a in candidates]
    bounds.sort(key=lambda t: t[1], reverse=True)

    for angle, ub in bounds:
        if ub <= best_area:
            continue
        rot_poly = rotate(shapely_poly, -angle, origin=centroid, use_radians=False)
        rect, area = _solve_axis_rect(rot_poly, grid_coarse, max_ratio)
        if area > best_area:
            best_area  = area
            best_rect  = rect
            best_angle = float(angle)

    if best_rect is None or len(candidates) <= 4:
        for angle in range(0, 180, angle_step):
            a  = float(angle % 90)
            ub = _upper_bound(shapely_poly, a, max_ratio, centroid)
            if ub <= best_area:
                continue
            rot_poly = rotate(shapely_poly, -a, origin=centroid, use_radians=False)
            rect, area = _solve_axis_rect(rot_poly, grid_coarse, max_ratio)
            if area > best_area:
                best_area  = area
                best_rect  = rect
                best_angle = a

    if best_rect is None:
        return None

    def _neg_area_fine(a):
        rp = rotate(shapely_poly, -a, origin=centroid, use_radians=False)
        _, area = _solve_axis_rect(rp, grid_fine, max_ratio)
        return -area

    res = minimize_scalar(
        _neg_area_fine,
        bounds=(best_angle - half_window, best_angle + half_window),
        method='bounded',
        options={'xatol': 0.3}
    )
    if res.fun < -best_area:
        best_angle = res.x
        rot_poly   = rotate(shapely_poly, -best_angle, origin=centroid, use_radians=False)
        best_rect, best_area = _solve_axis_rect(rot_poly, grid_fine, max_ratio)

    if best_rect is None:
        return None

    final_rect = rotate(best_rect, best_angle, origin=centroid, use_radians=False)

    if buf_enabled and buf_value != 0.0:
        candidate = final_rect.buffer(buf_value, cap_style=3, join_style=2)
        if not candidate.is_empty and candidate.area > 0:
            final_rect = candidate

    coords = list(final_rect.exterior.coords)
    w = np.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    h = np.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    ratio = (max(w, h) / min(w, h)) if min(w, h) > 0 else 0.0

    return final_rect, final_rect.area, best_angle, ratio


# ---------------------------------------------------------------------------
# PUBLIC API — single entry point for the main script
# ---------------------------------------------------------------------------

def process_slice(job_array, start, end,
                  angle_step, grid_coarse, grid_fine,
                  max_ratio, buf_enabled, buf_value):
    """
    Process job_array[start:end] in this thread/process.

    job_array  : list of (feat_id, wkb_bytes) — shared, never copied
    start, end : half-open slice indices into job_array

    Returns a dict  {feat_id: (wkt, area, angle, ratio)}
    Built in one pass with no intermediate lists; caller does dict.update().
    """
    out = {}
    for i in range(start, end):
        feat_id, wkb_bytes = job_array[i]
        try:
            poly   = wkb_loads(wkb_bytes)          # bytes already interned
            result = _search_one(poly, angle_step, grid_coarse, grid_fine,
                                 max_ratio, buf_enabled, buf_value)
            if result is not None:
                rect, area, angle, ratio = result
                out[feat_id] = (rect.wkt,
                                round(area,  4),
                                round(angle, 2),
                                round(ratio, 4))
        except Exception:
            pass
    return out
