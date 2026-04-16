# ===========================================================================
# inscribed_rect_worker.py  ·  v5
# Pure-geometry worker — no QGIS objects.
#
# Pipeline per feature
# ──────────────────────────────────────────────────────────────────────────
#   Stage 1  – Edge-guided angle candidates + coarse grid → top-K candidates
#   Stage 2  – Local angle polishing around each Stage 1 candidate
#   Stage 3  – Fine-grid solve at polished and original angles
#   Stage 4  – Explicit containment certification (symmetric shrink fallback)
# ===========================================================================
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import box, MultiPolygon, Polygon, Point
from shapely.prepared import prep as shp_prep
from shapely.wkb import loads as wkb_loads

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


# ==========================================================================
# ① JIT HISTOGRAM KERNEL
#    Classic O(n) largest-rectangle-in-histogram, operating on
#    integer height arrays over real-valued CRS coordinate arrays.
#    Stack uses two pre-allocated arrays → zero heap allocation inside JIT.
# ==========================================================================
@_njit(cache=True)
def _histogram_kernel(heights, xs, ys, row_idx, max_ratio):
    cols = len(heights)
    n_xs = len(xs)
    n_ys = len(ys)
    best = 0.0
    bx0 = by0 = bx1 = by1 = 0.0
    # Pre-allocated stack
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
                best = area
                bx0 = x0_w;
                by0 = y0_w;
                bx1 = x1_w;
                by1 = y1_w
            start = sc
        st_col[top] = start;
        st_h[top] = h;
        top += 1

    return bx0, by0, bx1, by1, best


# ==========================================================================
# ② GRID-BASED AXIS-ALIGNED RECTANGLE SOLVER
#    Rasterises the polygon at grid_steps × grid_steps resolution,
#    builds a running scanline height-map, and calls the histogram kernel.
# ==========================================================================
def _solve_axis_rect_grid(poly, grid_steps, max_ratio):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, grid_steps)
    ys = np.linspace(miny, maxy, grid_steps)
    xx, yy = np.meshgrid(xs, ys)
    flat = _mask_from_poly(poly, xx.ravel(), yy.ravel())
    mask = flat.reshape(grid_steps, grid_steps)
    heights = np.zeros(grid_steps, dtype=np.int64)
    best_rect = None;
    best_area = 0.0

    for r in range(grid_steps):
        row = mask[r].astype(np.int64)
        heights += row;
        heights *= row  # in-place, no temp allocation
        x0, y0, x1, y1, area = _histogram_kernel(heights, xs, ys, r, max_ratio)
        if area > best_area:
            best_area = area;
            best_rect = box(x0, y0, x1, y1)

    return best_rect, best_area


# ==========================================================================
# ③ ANALYTIC SLAB SOLVER (utility / reference; not in the hot path)
#    For simple convex polygons this gives exact axis-aligned results.
#    Evaluates y-intervals at both slab endpoints to avoid mid-point bias.
# ==========================================================================
def _edge_y_crossings_at_x(ring_coords, x_query):
    crossings = []
    n = len(ring_coords) - 1
    for i in range(n):
        xa, ya = ring_coords[i, 0], ring_coords[i, 1]
        xb, yb = ring_coords[i + 1, 0], ring_coords[i + 1, 1]
        xlo = min(xa, xb);
        xhi = max(xa, xb)
        if xlo < x_query <= xhi:
            dx = xb - xa
            if abs(dx) < 1e-14: continue
            crossings.append(ya + (x_query - xa) / dx * (yb - ya))
    crossings.sort()
    return crossings


def _net_valid_y_intervals(rot_poly, x_mid):
    ext_arr = np.asarray(rot_poly.exterior.coords, dtype=np.float64)
    ext_c = _edge_y_crossings_at_x(ext_arr, x_mid)
    include = [(ext_c[k], ext_c[k + 1]) for k in range(0, len(ext_c) - 1, 2)]
    if not include:
        return []
    free = list(include)
    for interior in rot_poly.interiors:
        h_arr = np.asarray(interior.coords, dtype=np.float64)
        h_c = _edge_y_crossings_at_x(h_arr, x_mid)
        for k in range(0, len(h_c) - 1, 2):
            hlo, hhi = h_c[k], h_c[k + 1]
            new_free = []
            for (flo, fhi) in free:
                if hhi <= flo or hlo >= fhi:
                    new_free.append((flo, fhi))
                else:
                    if flo < hlo: new_free.append((flo, hlo))
                    if hhi < fhi: new_free.append((hhi, fhi))
            free = new_free
        if not free: return []
    return free


def _solve_axis_rect_slab(rot_poly, max_ratio, n_extra=6):
    """Analytic slab sweep. Evaluates y-intervals at both slab endpoints."""
    if not isinstance(rot_poly, Polygon) or rot_poly.is_empty:
        return None, 0.0
    minx, miny, maxx, maxy = rot_poly.bounds
    if maxx - minx < 1e-12 or maxy - miny < 1e-12:
        return None, 0.0

    all_x = list(np.asarray(rot_poly.exterior.coords)[:, 0])
    for interior in rot_poly.interiors:
        all_x.extend(np.asarray(interior.coords)[:, 0].tolist())
    xev = np.unique(np.array(all_x, dtype=np.float64))
    xev = xev[(xev > minx + 1e-12) & (xev < maxx - 1e-12)]
    xb = np.unique(np.concatenate([[minx], xev, [maxx]]))

    slabs = []
    for i in range(len(xb) - 1):
        xa, xb_i = float(xb[i]), float(xb[i + 1])
        dx = (xb_i - xa) / (n_extra + 1)
        for k in range(n_extra + 1):
            slabs.append((xa + k * dx, xa + (k + 1) * dx))

    EPS = 1e-9
    n = len(slabs)
    ylo = np.full(n, np.nan);
    yhi = np.full(n, np.nan)

    for i, (xl, xr) in enumerate(slabs):
        ivs_l = _net_valid_y_intervals(rot_poly, xl + EPS)
        ivs_r = _net_valid_y_intervals(rot_poly, xr - EPS)
        if not ivs_l or not ivs_r: continue
        bl = max(ivs_l, key=lambda iv: iv[1] - iv[0])
        br = max(ivs_r, key=lambda iv: iv[1] - iv[0])
        lo = max(bl[0], br[0]) + EPS
        hi = min(bl[1], br[1]) - EPS
        if hi > lo: ylo[i] = lo; yhi[i] = hi

    best_area = 0.0;
    best_box = None
    for l in range(n):
        if math.isnan(ylo[l]): continue
        cur_lo, cur_hi = ylo[l], yhi[l]
        for r in range(l, n):
            if math.isnan(ylo[r]): break
            cur_lo = max(cur_lo, ylo[r])
            cur_hi = min(cur_hi, yhi[r])
            if cur_hi <= cur_lo: break
            rx0 = slabs[l][0];
            rx1 = slabs[r][1]
            rw = rx1 - rx0;
            rh = cur_hi - cur_lo
            if rw <= 0 or rh <= 0: continue
            if max_ratio > 0.0:
                ls = max(rw, rh);
                ss = min(rw, rh)
                if ss > 0 and ls / ss > max_ratio:
                    nl = ss * max_ratio
                    if rw >= rh:
                        cx = (rx0 + rx1) * 0.5;
                        rx0 = cx - nl / 2;
                        rx1 = cx + nl / 2;
                        rw = nl
                    else:
                        cy = (cur_lo + cur_hi) * 0.5;
                        cur_lo = cy - nl / 2;
                        cur_hi = cy + nl / 2;
                        rh = nl
            area = rw * rh
            if area > best_area:
                best_area = area;
                best_box = box(rx0, cur_lo, rx1, cur_hi)
    return best_box, best_area


# ==========================================================================
# ④ EDGE-GUIDED ANGLE CANDIDATE GENERATOR
#    Builds a weighted edge-orientation histogram (weights = edge lengths),
#    smooths it, and picks local maxima as candidate angles.
#    Falls back to uniform sampling if the polygon has few distinct edges.
# ==========================================================================
def _edge_candidate_angles(poly, min_sep_deg=4.0, max_candidates=12):
    coords = np.asarray(poly.exterior.coords, dtype=np.float64)
    edges = np.diff(coords, axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    valid = lengths > 1e-12
    if not valid.any():
        return np.array([0.0, 45.0])
    edges = edges[valid];
    lengths = lengths[valid]
    angles = np.degrees(np.arctan2(np.abs(edges[:, 1]),
                                   np.abs(edges[:, 0]))) % 90.0
    bins = np.zeros(91, dtype=np.float64)
    for ang, wt in zip(angles, lengths):
        bins[min(int(round(ang)), 90)] += wt
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    bins = np.convolve(bins, kernel, mode='same')
    sep = max(1, int(min_sep_deg))
    peaks = []
    for idx in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx) - p) >= sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= max_candidates:
            break
    return np.asarray(sorted(set(peaks) | {0, 45}), dtype=np.float64)


def _upper_bound_area(poly, angle, max_ratio, centroid):
    rot = shp_rotate(poly, -angle, origin=centroid, use_radians=False)
    bw, bh = rot.bounds[2] - rot.bounds[0], rot.bounds[3] - rot.bounds[1]
    if max_ratio > 0.0:
        ls = max(bw, bh);
        ss = min(bw, bh)
        if ss > 0 and ls / ss > max_ratio: ls = ss * max_ratio
        return ls * ss * 0.5
    return bw * bh * 0.5


def _heuristic_candidates(poly, angle_step, grid_coarse, grid_fine,
                          max_ratio, top_k):
    """Stage 1 candidate generation using edge-guided heuristic search."""
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    raw = []
    best_area = 0.0

    # Priority 1: dominant edge orientations
    for angle in _edge_candidate_angles(poly):
        ub = _upper_bound_area(poly, float(angle), max_ratio, centroid)
        if ub <= best_area * 0.85:
            continue
        rot = shp_rotate(poly, -float(angle), origin=centroid, use_radians=False)
        rect, area = _solve_axis_rect_grid(rot, grid_coarse, max_ratio)
        if area > 0:
            raw.append((area, float(angle), rect))
            if area > best_area: best_area = area

    # Priority 2: uniform fallback if edge heuristic under-covers
    if len(raw) < 3:
        for a_int in range(0, 90, angle_step):
            a = float(a_int)
            if any(abs(a - ar[1]) < 2.0 for ar in raw):
                continue
            ub = _upper_bound_area(poly, a, max_ratio, centroid)
            if ub <= best_area * 0.85:
                continue
            rot = shp_rotate(poly, -a, origin=centroid, use_radians=False)
            rect, area = _solve_axis_rect_grid(rot, grid_coarse, max_ratio)
            if area > 0:
                raw.append((area, a, rect))
                if area > best_area: best_area = area

    raw.sort(key=lambda t: t[0], reverse=True)

    # Deduplicate by angle proximity
    kept = [];
    seen = []
    for area, angle, rect_rot in raw:
        if any(abs(angle - s) < 2.0 for s in seen):
            continue
        seen.append(angle)
        rect_world = shp_rotate(rect_rot, angle, origin=centroid, use_radians=False)
        kept.append({'angle': angle, 'area': area,
                     'rect_rot': rect_rot, 'rect_world': rect_world,
                     'center': (cx, cy)})
        if len(kept) >= top_k:
            break
    return kept


# ==========================================================================
# ⑤ STAGE 2 — REFINEMENT
# ==========================================================================
_PHASE_A_XATOL = 0.05  # degrees tolerance for Brent
_PHASE_A_HALFWIDTH = 3.0  # ± bracket in degrees
_CERT_EPS = 1e-7  # inset after certification
_CERT_MAX_SHRINK = 0.2  # maximum symmetric shrink as fraction of shorter side


def _polish_angle(poly, candidate, grid_fine, max_ratio):
    """Stage 2: Brent minimisation around candidate angle."""
    angle_0 = candidate['angle']
    centroid = Point(candidate['center'])
    lo, hi = angle_0 - _PHASE_A_HALFWIDTH, angle_0 + _PHASE_A_HALFWIDTH

    def _neg_area(a):
        rot = shp_rotate(poly, -a, origin=centroid, use_radians=False)
        _, area = _solve_axis_rect_grid(rot, grid_fine, max_ratio)
        return -area

    try:
        res = minimize_scalar(_neg_area, bounds=(lo, hi), method='bounded',
                              options={'xatol': _PHASE_A_XATOL, 'maxiter': 60})
        if res.fun < -candidate['area'] + 1e-10:
            c = candidate.copy()
            c['angle'] = float(res.x)
            c['area'] = float(-res.fun)
            return c
    except Exception:
        pass
    return candidate


def _rect_local_frame(rect):
    coords = list(rect.exterior.coords)
    if len(coords) < 5: return None
    p0 = np.array(coords[0][:2]);
    p1 = np.array(coords[1][:2])
    p2 = np.array(coords[2][:2])
    e0 = p1 - p0;
    e1 = p2 - p1
    l0 = float(np.linalg.norm(e0));
    l1 = float(np.linalg.norm(e1))
    if l0 < 1e-14 or l1 < 1e-14: return None
    cx = float((p0[0] + p2[0]) / 2);
    cy = float((p0[1] + p2[1]) / 2)
    if l0 >= l1:
        ux, uy = e0[0] / l0, e0[1] / l0;
        vx, vy = e1[0] / l1, e1[1] / l1;
        a, b = l0 / 2, l1 / 2
    else:
        ux, uy = e1[0] / l1, e1[1] / l1;
        vx, vy = e0[0] / l0, e0[1] / l0;
        a, b = l1 / 2, l0 / 2
    return cx, cy, ux, uy, vx, vy, a, b


def _build_rect_from_frame(cx, cy, ux, uy, vx, vy, a, b):
    corners = [(cx + a * ux + b * vx, cy + a * uy + b * vy),
               (cx - a * ux + b * vx, cy - a * uy + b * vy),
               (cx - a * ux - b * vx, cy - a * uy - b * vy),
               (cx + a * ux - b * vx, cy + a * uy - b * vy)]
    return Polygon(corners + [corners[0]])


def _certify_and_adjust(poly, rect, max_ratio, buf_enabled, buf_value):
    """
    Stage 4: guarantee containment, optionally apply user buffer.
    Returns (rect, area) or (None, 0.0).
    """
    if rect is None or rect.is_empty: return None, 0.0
    prep = shp_prep(poly)

    if prep.covers(rect):
        final = rect
    else:
        frame = _rect_local_frame(rect)
        if frame is None: return None, 0.0
        cx, cy, ux, uy, vx, vy, a, b = frame

        # Measure overflow at all corners (and interior holes)
        max_ov = 0.0
        for corner in list(rect.exterior.coords)[:-1]:
            pt = Point(corner)
            if not prep.contains(pt):
                d = poly.exterior.distance(pt)
                for interior in poly.interiors:
                    ip = Polygon(interior)
                    if ip.contains(pt): d = max(d, ip.exterior.distance(pt))
                max_ov = max(max_ov, d)

        shrink = max_ov + _CERT_EPS
        if shrink > min(a, b) * _CERT_MAX_SHRINK:
            return None, 0.0

        new_a = a - shrink;
        new_b = b - shrink
        if new_a <= 0 or new_b <= 0: return None, 0.0
        if max_ratio > 0.0 and new_b > 0 and new_a / new_b > max_ratio:
            new_a = new_b * max_ratio

        final = _build_rect_from_frame(cx, cy, ux, uy, vx, vy, new_a, new_b)
        if not poly.covers(final): return None, 0.0

    if buf_enabled and buf_value != 0.0:
        cand = final.buffer(buf_value, cap_style=3, join_style=2)
        if not cand.is_empty and cand.area > 0:
            final = cand

    return final, float(final.area)


def _conservative_inner_fallback(poly, grid_fine, max_ratio, centroid, angles):
    best_rect = None
    best_area = 0.0
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
            rect_world = shp_rotate(rect_rot, angle, origin=centroid, use_radians=False)
            if poly.covers(rect_world):
                best_rect = rect_world
                best_area = float(rect_world.area)
                best_angle = angle

        if best_rect is not None:
            return best_rect, best_area, best_angle

    return None, 0.0, None


def _best_effort_shrink_to_cover(poly, rect, max_ratio, tol=1e-7, max_iter=40):
    """
    Symmetrically shrink a rectangle in its local frame until poly.covers(rect).
    Returns (rect, area) or (None, 0.0).
    """
    if rect is None or rect.is_empty:
        return None, 0.0

    frame = _rect_local_frame(rect)
    if frame is None:
        return None, 0.0

    cx, cy, ux, uy, vx, vy, a0, b0 = frame
    if a0 <= 0 or b0 <= 0:
        return None, 0.0

    def build(scale):
        a = a0 * scale
        b = b0 * scale
        if a <= 0 or b <= 0:
            return None
        if max_ratio > 0.0:
            long_s = max(a, b)
            short_s = min(a, b)
            if short_s <= 0:
                return None
            if long_s / short_s > max_ratio:
                if a >= b:
                    a = b * max_ratio
                else:
                    b = a * max_ratio
        return _build_rect_from_frame(cx, cy, ux, uy, vx, vy, a, b)

    # Full size already valid
    r1 = build(1.0)
    if r1 is not None and poly.covers(r1):
        return r1, float(r1.area)

    # Find any valid lower bound
    lo, hi = 0.0, 1.0
    r_lo = None
    for s in (0.95, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01):
        r = build(s)
        if r is not None and poly.covers(r):
            lo = s
            r_lo = r
            break

    if r_lo is None:
        return None, 0.0

    # Binary search largest valid scale
    best_r = r_lo
    best_a = float(r_lo.area)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        r = build(mid)
        if r is not None and poly.covers(r):
            lo = mid
            best_r = r
            best_a = float(r.area)
        else:
            hi = mid

    return best_r, best_a


def _refine_best_candidate(poly, candidates, grid_fine, max_ratio,
                           buf_enabled, buf_value, always_return):
    """Full Stage 2 pipeline. Returns 7-tuple or None."""
    certified = []
    fallback_best = None

    for rank, cand in enumerate(candidates):
        area_s1 = cand['area']
        centroid = Point(cand['center'])
        angle_0 = cand['angle']

        rot0 = shp_rotate(poly, -angle_0, origin=centroid, use_radians=False)
        rect0_rot, area0 = _solve_axis_rect_grid(rot0, grid_fine, max_ratio)
        if rect0_rot is None or area0 <= 0:
            continue
        rect0_world = shp_rotate(rect0_rot, angle_0, origin=centroid, use_radians=False)

        best_raw_r = rect0_world
        best_raw_a = float(area0)
        best_raw_ang = angle_0

        cand_a = _polish_angle(poly, cand, grid_fine, max_ratio)
        if cand_a['angle'] != angle_0:
            angle_a = cand_a['angle']
            rot_a = shp_rotate(poly, -angle_a, origin=centroid, use_radians=False)
            rect_a_rot, area_a = _solve_axis_rect_grid(rot_a, grid_fine, max_ratio)
            if rect_a_rot is not None and area_a > best_raw_a:
                rect_a_world = shp_rotate(rect_a_rot, angle_a, origin=centroid, use_radians=False)
                best_raw_r = rect_a_world
                best_raw_a = float(area_a)
                best_raw_ang = angle_a

        if best_raw_r is None:
            continue

        if fallback_best is None or best_raw_a > fallback_best['area']:
            fallback_best = {
                'rect': best_raw_r,
                'area': best_raw_a,
                'angle': best_raw_ang,
                'rank': rank,
            }

        best_r, best_a = _certify_and_adjust(poly, best_raw_r, max_ratio, False, 0.0)
        used_best_effort = False

        if best_r is None and always_return:
            best_r, best_a = _best_effort_shrink_to_cover(poly, best_raw_r, max_ratio)
            used_best_effort = best_r is not None

        if best_r is None:
            continue

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
            'angle': best_raw_ang,
            'ratio': ratio,
            'rank': rank,
            'stage2_gain': best_a - area_s1,
            'used_best_effort': used_best_effort,
        })

    if not certified:
        if always_return and fallback_best is not None and fallback_best['rect'] is not None:
            rect_fb, area_fb = _best_effort_shrink_to_cover(poly, fallback_best['rect'], max_ratio)
            if rect_fb is not None:
                angle_fb = fallback_best['angle']
                rank_fb = fallback_best['rank']
                coords = list(rect_fb.exterior.coords)
                w = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                h = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
                ratio_fb = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                return (rect_fb, area_fb, angle_fb, ratio_fb, rank_fb, area_fb, True)

            centroid = Point(candidates[0]['center'])
            rescue_angles = [c['angle'] for c in candidates[:max(3, min(len(candidates), 8))]]
            rect_c, area_c, angle_c = _conservative_inner_fallback(
                poly, grid_fine, max_ratio, centroid, rescue_angles
            )
            if rect_c is not None:
                coords = list(rect_c.exterior.coords)
                w = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                h = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
                ratio_c = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                return (rect_c, area_c, angle_c, ratio_c, fallback_best['rank'], area_c, True)

        return None

    best = max(certified, key=lambda c: c['area'])
    return (best['rect'], best['area'], best['angle'],
            best['ratio'], best['rank'], best['stage2_gain'],
            best['used_best_effort'])


# ==========================================================================
# ⑥ GEOMETRY PREPARATION
# ==========================================================================
def _prepare_polygon(geom):
    from shapely.validation import make_valid

    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        geom = make_valid(geom)

    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)

    elif hasattr(geom, "geoms") and not isinstance(geom, Polygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)

    if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= 0:
        return None

    return geom


# ==========================================================================
# ⑦ MODULE-LEVEL MULTIPROCESSING ENTRY POINT
# ==========================================================================
def _worker_process_feature(args):
    """
    Stateless worker function — safe for multiprocessing.Pool.

    Parameters
    ----------
    args : tuple
    (feat_id, wkb_bytes, angle_step, grid_coarse, grid_fine,
     max_ratio, buf_enabled, buf_value, top_k, always_return)

    Returns
    -------
tuple or None
    (feat_id, wkt, area, angle_deg, ratio, cand_rank, stage2_gain, used_best_effort)
    """
    (feat_id, wkb_bytes, angle_step, grid_coarse, grid_fine,
     max_ratio, buf_enabled, buf_value, top_k, always_return) = args
    try:
        poly = _prepare_polygon(wkb_loads(bytes(wkb_bytes)))
        try:
            from shapely import set_precision
            minx, miny, maxx, maxy = poly.bounds
            span = max(maxx - minx, maxy - miny)
            if span > 0:
                poly = set_precision(poly, grid_size=span * 1e-9, mode="valid_output")
        except Exception:
            pass
        if poly is None:
            return None
        candidates = _heuristic_candidates(
            poly, angle_step, grid_coarse, grid_fine, max_ratio, top_k)
        if not candidates:
            return None
        result = _refine_best_candidate(
            poly, candidates, grid_fine, max_ratio, buf_enabled, buf_value, always_return
        )

        if result is None:
            return None

        rect, area, angle, ratio, rank, gain, used_best_effort = result

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
        raise RuntimeError(f"_worker_process_feature failed for feat_id={feat_id}: {e}") from e
