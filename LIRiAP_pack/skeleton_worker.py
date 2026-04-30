"""
LIRiAP Skeleton worker module.

Pure geometry solver using medial-axis skeleton decomposition for seed
generation, followed by SDF-guided boundary expansion.  No QGIS or Qt
runtime dependencies.  No BCRS — SDF + grid-solve replaces the expensive
vertex-coordinate raster.

Pipeline
========
Stage 0: Skeleton extraction — distance transform + ridge detection
Stage 1: Geometry preparation
Stage 2: Hybrid seeds — skeleton PCA angles + nearest edge-angle fillers
Stage 3: Grid solve per seed (40×40) → initial axis-aligned rect
Stage 4: SDF binary expansion → exact boundary contact
Stage 5: ±2° delta test → catches edge-aligned angular optima
Stage 6: SDF containment certification
Stage 7: Selection

Fallback
========
If skeleton extraction yields fewer than 2 viable seeds, falls through
to edge-angle candidates with the same grid-solve + SDF-expand pipeline.

See Also
========
skeleton_algorithm: QGIS wrapper
bcrs_worker:         Standard BCRS variant
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon, box, MultiPolygon
from shapely.prepared import prep as shp_prep
from shapely.wkb import loads as wkb_loads

try:
    from shapely.vectorized import contains as _shp_contains_vec
    def _mask_from_poly(poly, xx_flat, yy_flat):
        return _shp_contains_vec(poly, xx_flat, yy_flat)
except ImportError:
    import shapely as _shp2
    def _mask_from_poly(poly, xx_flat, yy_flat):
        pts = _shp2.points(xx_flat, yy_flat)
        return _shp2.contains(poly, pts)


# ==========================================================================
# Tuning constants
# ==========================================================================
_SKEL_GRID_RES    = 150      # raster resolution (long axis)
_SKEL_CLEAR_PCT   = 60       # clearance percentile for ridge pruning
_SKEL_MIN_REGION  = 5        # minimum pixels per ridge region
_SKEL_MIN_SEEDS   = 2        # fall through to edge-angle if fewer
_SDF_BINARY_STEPS = 10       # binary-search steps per side
_EXPAND_ITERS     = 3        # outer expansion iterations
_CERT_EPS         = 1e-7     # safety inset for certification
_CERT_MAX_SHRINK  = 0.20     # maximum symmetric shrink fraction
_DELTA_DEG        = 2.0      # ± degrees tested around each seed angle


# ==========================================================================
# ① SDF PRIMITIVE (inlined from sdf_oracle.py)
# ==========================================================================

def _polygon_sdf(poly, x, y):
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


# ==========================================================================
# ② RECTANGLE FRAME HELPERS
# ==========================================================================

def _rect_local_frame(rect):
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return None
    p0 = np.array(coords[0][:2]);  p1 = np.array(coords[1][:2])
    p2 = np.array(coords[2][:2])
    e0 = p1 - p0;  e1 = p2 - p1
    l0 = float(np.linalg.norm(e0));  l1 = float(np.linalg.norm(e1))
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
# ③ SDF CONTAINMENT CERTIFICATION
# ==========================================================================

def _rect_sdf_max(poly, rect):
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


def _certify_and_adjust(poly, rect, max_ratio):
    if rect is None or rect.is_empty:
        return None, 0.0
    max_sdf = _rect_sdf_max(poly, rect)
    if max_sdf <= _CERT_EPS:
        return rect, float(rect.area)
    frame = _rect_local_frame(rect)
    if frame is None:
        return None, 0.0
    cx, cy, ux, uy, vx, vy, a, b = frame
    shrink = max_sdf + _CERT_EPS
    if shrink > min(a, b) * _CERT_MAX_SHRINK:
        return None, 0.0
    new_a = a - shrink;  new_b = b - shrink
    if new_a <= 0 or new_b <= 0:
        return None, 0.0
    if max_ratio > 0.0 and new_b > 0 and new_a / new_b > max_ratio:
        new_a = new_b * max_ratio
    final = _build_rect_from_frame(cx, cy, ux, uy, vx, vy, new_a, new_b)
    if _rect_sdf_max(poly, final) > _CERT_EPS * 10:
        return None, 0.0
    return final, float(final.area)


# ==========================================================================
# ④ SDF-GUIDED BINARY EXPANSION
# ==========================================================================

def _sdf_expand_rect(rot_poly, x0, y0, x1, y1, max_ratio):
    minx, miny, maxx, maxy = rot_poly.bounds
    prep = shp_prep(rot_poly)

    def _v(ax0, ay0, ax1, ay1):
        if ax1 - ax0 < 1e-12 or ay1 - ay0 < 1e-12:
            return False
        return prep.covers(box(ax0, ay0, ax1, ay1))

    if not _v(x0, y0, x1, y1):
        cx_c = 0.5 * (x0 + x1);  cy_c = 0.5 * (y0 + y1)
        hw = 0.5 * (x1 - x0);  hh = 0.5 * (y1 - y0)
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
        x0 = cx_c - hw * lo;  y0 = cy_c - hh * lo
        x1 = cx_c + hw * lo;  y1 = cy_c + hh * lo

    for _ in range(_EXPAND_ITERS):
        if x0 > minx:
            sdf = _polygon_sdf(rot_poly, x0, 0.5 * (y0 + y1))
            hi_d = min(x0 - minx, abs(sdf)) if sdf < 0 else x0 - minx
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(_SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0 - mid, y0, x1, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                x0 -= lo_d
        if x1 < maxx:
            sdf = _polygon_sdf(rot_poly, x1, 0.5 * (y0 + y1))
            hi_d = min(maxx - x1, abs(sdf)) if sdf < 0 else maxx - x1
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(_SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0, x1 + mid, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                x1 += lo_d
        if y0 > miny:
            sdf = _polygon_sdf(rot_poly, 0.5 * (x0 + x1), y0)
            hi_d = min(y0 - miny, abs(sdf)) if sdf < 0 else y0 - miny
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(_SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0 - mid, x1, y1):
                        lo_d = mid
                    else:
                        hi_d = mid
                y0 -= lo_d
        if y1 < maxy:
            sdf = _polygon_sdf(rot_poly, 0.5 * (x0 + x1), y1)
            hi_d = min(maxy - y1, abs(sdf)) if sdf < 0 else maxy - y1
            if hi_d > 1e-12:
                lo_d = 0.0
                for _ in range(_SDF_BINARY_STEPS):
                    mid = 0.5 * (lo_d + hi_d)
                    if _v(x0, y0, x1, y1 + mid):
                        lo_d = mid
                    else:
                        hi_d = mid
                y1 += lo_d

    if max_ratio > 0.0:
        rw = x1 - x0;  rh = y1 - y0
        if rw > 0 and rh > 0:
            ls = max(rw, rh);  ss = min(rw, rh)
            if ss > 0 and ls / ss > max_ratio:
                nl = ss * max_ratio
                if rw >= rh:
                    cx_r = 0.5 * (x0 + x1)
                    x0 = cx_r - 0.5 * nl;  x1 = cx_r + 0.5 * nl
                else:
                    cy_r = 0.5 * (y0 + y1)
                    y0 = cy_r - 0.5 * nl;  y1 = cy_r + 0.5 * nl

    return x0, y0, x1, y1


# ==========================================================================
# ⑤ SKELETON EXTRACTION
# ==========================================================================

def _extract_skeleton_seeds(poly, top_k, emitter=None):
    seeds = []
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0.0 or h <= 0.0:
        return seeds

    aspect = h / w if w >= h else w / h
    nx = _SKEL_GRID_RES
    ny = max(3, int(round(_SKEL_GRID_RES * aspect)))
    if h > w:
        nx, ny = ny, nx

    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    dx = (maxx - minx) / (nx - 1) if nx > 1 else 0.0
    dy = (maxy - miny) / (ny - 1) if ny > 1 else 0.0

    xx, yy = np.meshgrid(xs, ys)
    flat = _mask_from_poly(poly, xx.ravel(), yy.ravel())
    mask = flat.reshape(ny, nx).astype(np.bool_, copy=False)
    dist = distance_transform_edt(mask)
    if not (dist > 0.0).any():
        return seeds

    local_max = maximum_filter(dist, size=3)
    ridges = (dist > 0.0) & (dist >= local_max - 1e-12)
    ridge_vals = dist[ridges]
    if len(ridge_vals) < _SKEL_MIN_REGION:
        return seeds
    threshold = np.percentile(ridge_vals, _SKEL_CLEAR_PCT)
    strong = ridges & (dist >= threshold)
    if not strong.any():
        return seeds

    regions, n_regs = label(strong)
    if n_regs < 1:
        return seeds

    if emitter:
        emitter.emit(
            phase="CANDIDATES", type_="skeleton_regions_found",
            label=f"{n_regs} skeleton regions",
            narration="Interior distance-field skeleton extracted.",
            region_count=n_regs, grid_res=f"{nx}×{ny}",
            clearance_threshold=round(float(threshold), 3),
        )

    for r_id in range(1, n_regs + 1):
        ry, rx = np.where(regions == r_id)
        if len(ry) < _SKEL_MIN_REGION:
            continue
        clearances = dist[regions == r_id]
        max_cl = float(clearances.max())
        total_cl = float(clearances.sum())
        cx = float(minx + np.mean(rx) * dx)
        cy = float(miny + np.mean(ry) * dy)
        rx_w = minx + rx.astype(np.float64) * dx
        ry_w = miny + ry.astype(np.float64) * dy
        pts = np.column_stack([rx_w, ry_w])
        pts_c = pts - np.mean(pts, axis=0)
        cov = (pts_c.T @ pts_c) / (len(rx) - 1)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        v = eigenvecs[:, np.argmax(eigenvals)]
        angle = math.degrees(math.atan2(v[1], v[0])) % 90.0
        seeds.append({
            'cx': cx, 'cy': cy, 'angle': angle,
            'clearance': max_cl, 'total_clearance': total_cl,
            'pixels': len(rx),
            'score': max_cl * max_cl * total_cl,
        })

    seeds.sort(key=lambda s: s['score'], reverse=True)
    kept = []
    for s in seeds:
        too_close = False
        for k in kept:
            d = math.hypot(s['cx'] - k['cx'], s['cy'] - k['cy'])
            if d < max(s['clearance'], k['clearance']) * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append(s)
        if len(kept) >= top_k:
            break

    if emitter:
        for i, s in enumerate(kept):
            emitter.emit(
                phase="CANDIDATES", type_="skeleton_seed",
                label=f"Seed {i}: {s['angle']:.1f}° cl={s['clearance']:.1f}",
                narration=f"Skeleton seed ({s['cx']:.1f},{s['cy']:.1f}).",
                angle_deg=round(s['angle'], 4),
                center=[round(s['cx'], 4), round(s['cy'], 4)],
                clearance=round(s['clearance'], 4),
                score=round(s['score'], 2), rank=i,
            )
    return kept


# ==========================================================================
# ⑥ EDGE-ANGLE HELPER
# ==========================================================================

def _edge_angles(poly):
    coord_sets = [np.asarray(poly.exterior.coords, dtype=np.float64)]
    for interior in poly.interiors:
        coord_sets.append(np.asarray(interior.coords, dtype=np.float64))
    all_edges = [];  all_lengths = []
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
    peaks = []
    for idx_p in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx_p) - p) >= 3 for p in peaks):
            peaks.append(int(idx_p))
        if len(peaks) >= 16:
            break
    return np.asarray(sorted(set(peaks) | {0, 45}), dtype=np.float64)


# ==========================================================================
# ⑦ FAST-PATH: SIMPLE CONVEX POLYGONS
# ==========================================================================

def _fast_path_solve(poly, max_ratio=0.0):
    """
    Detect simple convex polygons where the optimal rectangle is
    guaranteed to be edge-aligned, and solve directly via grid-solve
    + SDF expansion for each edge angle.

    Handles:
    1. Rectangle (4 vertices, right angles) → identity, O(1)
    2. Simple convex (≤ 8 vertices, no holes, near-convex) →
       grid-solve at each unique edge angle (60-res), SDF expand,
       certify, return best.

    Returns (rect, area, angle) or None.
    """
    coords = list(poly.exterior.coords)[:-1]
    nv = len(coords)
    has_holes = bool(poly.interiors)

    # ── Case 1: Rectangle (identity) ─────────────────────────────────────
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
            angle_deg = math.degrees(
                math.atan2(frame[3], frame[2])) % 90.0 if frame else 0.0
            return poly, a, angle_deg

    # ── Case 2: Simple convex (edge-aligned optimal guaranteed) ──────────
    # Criteria: no holes, ≤ 8 vertices, concavity < 0.5 %
    if has_holes:
        return None
    if nv < 3 or nv > 8:
        return None

    hull = poly.convex_hull
    hull_area = float(hull.area)
    poly_area = float(poly.area)
    if poly_area <= 0 or hull_area / poly_area > 1.005:
        return None  # too concave — fall back to skeleton

    # Extract unique edge angles
    raw_angles = []
    for i in range(len(hull.exterior.coords) - 1):
        dx = hull.exterior.coords[i + 1][0] - hull.exterior.coords[i][0]
        dy = hull.exterior.coords[i + 1][1] - hull.exterior.coords[i][1]
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            a = math.degrees(math.atan2(dy, dx)) % 90.0
            # Merge within 1°
            if not any(abs(a - ra) < 1.0 for ra in raw_angles):
                raw_angles.append(a)
    # Always include 0° and 45°
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
        seed, _ = _grid_solve(rot, 60)
        if seed is None:
            continue
        sb = seed.bounds
        bx0, by0, bx1, by1 = _sdf_expand_rect(
            rot, sb[0], sb[1], sb[2], sb[3], max_ratio)
        area = (bx1 - bx0) * (by1 - by0)
        if area <= best_area:
            continue
        rect_r = box(bx0, by0, bx1, by1)
        rect_w = shp_rotate(rect_r, a, origin=centroid, use_radians=False)
        cert_r, cert_a = _certify_and_adjust(poly, rect_w, max_ratio)
        if cert_r is not None and cert_a > best_area:
            best_rect, best_area, best_angle = cert_r, cert_a, a

    if best_rect is None:
        return None
    return best_rect, best_area, best_angle


# ==========================================================================
# ⑧ COARSE GRID SOLVER
# ==========================================================================

def _grid_solve(rot_poly, grid_steps):
    minx, miny, maxx, maxy = rot_poly.bounds
    xs = np.linspace(minx, maxx, grid_steps)
    ys = np.linspace(miny, maxy, grid_steps)
    xx, yy = np.meshgrid(xs, ys)
    flat = _mask_from_poly(rot_poly, xx.ravel(), yy.ravel())
    mask = flat.reshape(grid_steps, grid_steps)
    heights = np.zeros(grid_steps, dtype=np.int64)
    best_area = 0.0;  best_bounds = None
    cols = grid_steps
    st_col = np.empty(cols + 1, dtype=np.int64)
    st_h = np.empty(cols + 1, dtype=np.int64)
    for r in range(grid_steps):
        row = mask[r].astype(np.int64)
        heights += row;  heights *= row
        top = 0
        for c2 in range(cols + 1):
            h = int(heights[c2]) if c2 < cols else 0
            start = c2
            while top > 0 and st_h[top - 1] > h:
                top -= 1
                sc = st_col[top];  sh = st_h[top]
                w_s = c2 - sc
                xi = sc + w_s
                bx0 = xs[sc]
                bx1 = xs[xi if xi < cols else cols - 1]
                y0g = ys[int(r - sh + 1)]
                y1g = ys[int(r)]
                rw = bx1 - bx0;  rh = y1g - y0g
                if rw > 0 and rh > 0:
                    a = rw * rh
                    if a > best_area:
                        best_area = a
                        best_bounds = (bx0, y0g, bx1, y1g)
                start = sc
            st_col[top] = start;  st_h[top] = h;  top += 1
    if best_bounds is None:
        return None, 0.0
    return box(*best_bounds), best_area


# ==========================================================================
# ⑧ SOLVE FROM A SEED
# ==========================================================================

def _solve_from_seed(poly, centroid, angle_deg, max_ratio):
    rot_poly = shp_rotate(poly, -angle_deg, origin=centroid,
                           use_radians=False)
    seed_rect, _ = _grid_solve(rot_poly, 40)
    if seed_rect is None:
        return None, 0.0
    sb = seed_rect.bounds
    bx0, by0, bx1, by1 = _sdf_expand_rect(
        rot_poly, sb[0], sb[1], sb[2], sb[3], max_ratio)
    area = (bx1 - bx0) * (by1 - by0)
    if area <= 1e-12:
        return None, 0.0
    rect_rot = box(bx0, by0, bx1, by1)
    rect_world = shp_rotate(rect_rot, angle_deg, origin=centroid,
                             use_radians=False)
    return rect_world, float(rect_world.area)


# ==========================================================================
# ⑨ ORCHESTRATOR
# ==========================================================================

def _refine_candidates(poly, candidates, max_ratio, emitter=None):
    centroid = poly.centroid
    certified = []
    for rank, cand in enumerate(candidates):
        angle = cand['angle']
        rect_w, area = _solve_from_seed(poly, centroid, angle, max_ratio)
        if rect_w is None:
            continue
        best_angle = angle

        for delta in (-_DELTA_DEG, _DELTA_DEG):
            r2, a2 = _solve_from_seed(
                poly, centroid, best_angle + delta, max_ratio)
            if r2 is not None and a2 > area + 1e-6:
                rect_w, area, best_angle = r2, a2, best_angle + delta

        cert_r, cert_a = _certify_and_adjust(poly, rect_w, max_ratio)
        if cert_r is None:
            continue

        if emitter:
            eb = cert_r.bounds
            a_pct = (cert_a / poly.area * 100) if poly.area > 0 else 0.0
            emitter.emit(
                phase="RESULT", type_="best_updated",
                label=f"Best: area={cert_a:.1f} ({a_pct:.1f}%)",
                narration="Candidate after SDF expansion + cert.",
                rect=[float(eb[0]), float(eb[1]),
                      float(eb[2]), float(eb[3])],
                area=round(cert_a, 4),
                pct_polygon=round(a_pct, 2),
                angle_deg=round(best_angle, 4),
                source=cand.get('source', 'skeleton'),
            )
        coords = list(cert_r.exterior.coords)
        w_l = math.hypot(coords[1][0] - coords[0][0],
                         coords[1][1] - coords[0][1])
        h_l = math.hypot(coords[2][0] - coords[1][0],
                         coords[2][1] - coords[1][1])
        ratio_v = max(w_l, h_l) / min(w_l, h_l) if min(w_l, h_l) > 0 else 1.0
        certified.append({
            'rect': cert_r, 'area': cert_a, 'angle': best_angle,
            'ratio': ratio_v, 'rank': rank,
        })

    if not certified:
        return None
    best = max(certified, key=lambda c: c['area'])
    return (best['rect'], best['area'], best['angle'],
            best['ratio'], best['rank'], 0.0, False)


# ==========================================================================
# ⑩ CANDIDATE GENERATION
# ==========================================================================

def _generate_candidates(poly, angle_step, top_k, max_ratio, emitter=None):
    seeds = _extract_skeleton_seeds(poly, top_k, emitter)
    if len(seeds) >= _SKEL_MIN_SEEDS:
        edge_list = _edge_angles(poly).tolist()
        seen = {round(s['angle'], 1) for s in seeds}
        # Augment each skeleton seed with nearest 2 edge angles within ±10°
        for s in seeds:
            nearby = sorted(
                [a for a in edge_list
                 if abs((a - s['angle'] + 90) % 90 - 90) < 10.0
                 and round(a, 1) not in seen],
                key=lambda a: abs((a - s['angle'] + 90) % 90 - 90))
            for a in nearby[:2]:
                seeds.append({
                    'cx': s['cx'], 'cy': s['cy'], 'angle': float(a),
                    'clearance': s['clearance'], 'pixels': s['pixels'],
                    'score': s['score'] * 0.5, 'source': 'edge_nearby',
                })
                seen.add(round(a, 1))
        if emitter:
            emitter.emit(
                phase="CANDIDATES", type_="skeleton_used",
                label=f"Skeleton: {len(seeds)} seeds",
                narration="Using skeleton + edge-angle hybrid seeds.",
                candidate_count=len(seeds),
            )
        return seeds

    # Skeleton sparse → fall through to edge-angle
    edge_list = _edge_angles(poly).tolist()
    candidates = []
    for a in edge_list:
        candidates.append({
            'cx': 0, 'cy': 0, 'angle': float(a),
            'clearance': 0, 'pixels': 0, 'score': 0,
            'source': 'edge_angle',
        })
    if len(candidates) < 3:
        for a_int in range(0, 90, angle_step):
            a_f = float(a_int)
            if any(abs(a_f - c['angle']) < 2.0 for c in candidates):
                continue
            candidates.append({
                'cx': 0, 'cy': 0, 'angle': a_f,
                'clearance': 0, 'pixels': 0, 'score': 0,
                'source': 'grid_sweep',
            })
    if emitter:
        emitter.emit(
            phase="CANDIDATES", type_="skeleton_fallback",
            label=f"Edge-angle fallback: {len(candidates)} candidates",
            narration="Skeleton returned too few seeds.",
            candidate_count=len(candidates),
        )
    return candidates[:top_k * 2]


# ==========================================================================
# ⑪ GEOMETRY PREPARATION
# ==========================================================================

def _prepare_polygon(geom):
    from shapely.validation import make_valid
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and not g.is_empty
                 and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)
    elif hasattr(geom, 'geoms') and not isinstance(geom, Polygon):
        polys = [g for g in geom.geoms
                 if isinstance(g, Polygon) and not g.is_empty
                 and g.area > 0]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)
    if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= 0:
        return None
    return geom


# ==========================================================================
# ⑫ PUBLIC ENTRY POINT
# ==========================================================================

def _worker_process_feature(args, emitter=None):
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
                poly = set_precision(poly, grid_size=span * 1e-9,
                                     mode='valid_output')
        except Exception:
            pass
        if poly is None or poly.is_empty:
            return None

        # ── Fast path: analytically solvable cases ────────────────────────
        fast = _fast_path_solve(poly, max_ratio=max_ratio)
        if fast is not None:
            fp_rect, fp_area, fp_angle = fast
            fp_ratio = 1.0
            if fp_area > 0:
                coords_fp = list(fp_rect.exterior.coords)
                w_fp = math.hypot(coords_fp[1][0] - coords_fp[0][0],
                                  coords_fp[1][1] - coords_fp[0][1])
                h_fp = math.hypot(coords_fp[2][0] - coords_fp[1][0],
                                  coords_fp[2][1] - coords_fp[1][1])
                fp_ratio = max(w_fp, h_fp) / min(w_fp, h_fp) if min(w_fp, h_fp) > 0 else 1.0
            if emitter:
                emitter.emit(
                    phase="RESULT", type_="final_result",
                    label=f"Fast-path: area={fp_area:.1f}",
                    narration="Solved via analytical fast-path.",
                    rect=[float(fp_rect.bounds[0]), float(fp_rect.bounds[1]),
                          float(fp_rect.bounds[2]), float(fp_rect.bounds[3])],
                    area=round(float(fp_area), 4),
                    angle_deg=round(float(fp_angle), 4),
                    algorithm="skeleton_fastpath",
                )
            return (
                feat_id, fp_rect.wkt,
                round(float(fp_area), 4), round(float(fp_angle), 4),
                round(float(fp_ratio), 4), 0, 0.0, 0,
            )

        if emitter:
            emitter.emit(
                phase="SETUP", type_="polygon_loaded",
                label="Polygon loaded (skeleton)",
                narration="Polygon loaded for skeleton solve.",
                bbox=[float(poly.bounds[0]), float(poly.bounds[1]),
                      float(poly.bounds[2]), float(poly.bounds[3])],
                area=float(poly.area),
                vertex_count=len(poly.exterior.coords) - 1,
            )

        candidates = _generate_candidates(
            poly, angle_step, top_k, max_ratio, emitter=emitter)
        if not candidates:
            return None

        result = _refine_candidates(
            poly, candidates, max_ratio, emitter=emitter)
        if result is None:
            return None

        rect, area, angle, ratio, rank, gain, used_best_effort = result

        if buf_enabled and buf_value != 0.0:
            cand_b = rect.buffer(buf_value, cap_style=3, join_style=2)
            if not cand_b.is_empty and cand_b.area > 0:
                rect = cand_b
                area = float(rect.area)

        if emitter:
            pct = (area / poly.area * 100) if poly.area > 0 else 0.0
            emitter.emit(
                phase="RESULT", type_="final_result",
                label=f"Final: area={area:.1f} ({pct:.1f}%)",
                narration="Skeleton LIR solve complete.",
                rect=[float(rect.bounds[0]), float(rect.bounds[1]),
                      float(rect.bounds[2]), float(rect.bounds[3])],
                area=round(float(area), 4),
                pct_polygon=round(pct, 2),
                angle_deg=round(float(angle), 4),
                algorithm="skeleton",
            )
        return (
            feat_id, rect.wkt,
            round(float(area), 4), round(float(angle), 4),
            round(float(ratio), 4), int(rank),
            round(float(gain), 6), int(used_best_effort),
        )
    except Exception as e:
        raise RuntimeError(
            f'_worker_process_feature failed for feat_id={feat_id}: {e}'
        ) from e
