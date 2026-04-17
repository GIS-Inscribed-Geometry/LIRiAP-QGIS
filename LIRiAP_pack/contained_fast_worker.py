"""
LIRiAP Contained Fast worker module.

Pure geometry routines used by the corresponding algorithm wrapper.
No QGIS or Qt runtime dependencies.

Pipeline:
1. Edge-guided angle candidates with coarse-grid ranking.
2. Local angle polishing around top candidates.
3. Fine-grid solve at polished and original angles.
4. Containment certification with symmetric-shrink fallback.
"""
from __future__ import annotations

import math

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
# Vectorised point-in-polygon  (Shapely 1.x + 2.x compat)
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
# Tuning constants
# --------------------------------------------------------------------------
_PHASE_A_XATOL = 0.02  # Brent angle tolerance [deg]  (tightened from 0.05)
_PHASE_A_HALFWIDTH = 3.0  # Brent bracket half-width [deg]
_CERT_EPS = 1e-7  # Safety inset after certification
_CERT_MAX_SHRINK = 0.20  # Max symmetric shrink as fraction of shorter side
_PRUNE_MARGIN = 0.90  # Upper-bound pruning factor  (raised from 0.85)
_SIMPLIFY_THRESHOLD = 300  # Vertex count above which simplification is tried
_SIMPLIFY_TOL_FRAC = 0.001  # Simplification tol as fraction of short bbox side


# ==========================================================================
# ① JIT HISTOGRAM KERNEL
#    O(n) largest-rectangle-in-histogram via monotone stack.
#    Pre-allocated stack arrays → zero heap allocation inside JIT.
#    Aspect-ratio constraint applied analytically per candidate rectangle.
# ==========================================================================
@_njit(cache=True)
def _histogram_kernel(heights, xs, ys, row_idx, max_ratio):
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
# ② GRID-BASED AXIS-ALIGNED RECTANGLE SOLVER
#    Rasterises the (pre-rotated) polygon onto a grid_steps × grid_steps
#    binary mask and runs the JIT histogram kernel row by row.
#    Works correctly for all polygon types including non-convex and holed.
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
        heights *= row
        x0, y0, x1, y1, area = _histogram_kernel(heights, xs, ys, r, max_ratio)
        if area > best_area:
            best_area = area;
            best_rect = box(x0, y0, x1, y1)

    return best_rect, best_area


# ==========================================================================
# ③ EDGE-GUIDED ANGLE CANDIDATE GENERATOR
#    Builds a length-weighted edge-orientation histogram over [0°, 90°),
#    smooths it with a Gaussian-like kernel, and extracts local maxima as
#    candidate angles.  Both exterior and interior ring edges are included
#    so that hole boundaries contribute (important for parcels with buildings
#    or courtyards that share the dominant orientation).
#    Bin accumulation uses np.add.at (vectorised; replaces Python loop).
# ==========================================================================
def _edge_candidate_angles(poly: Polygon,
                           min_sep_deg: float = 4.0,
                           max_candidates: int = 12) -> np.ndarray:
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

    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    bins = np.convolve(bins, kernel, mode='same')

    sep = max(1, int(min_sep_deg))
    peaks = []
    for idx_p in np.argsort(bins)[::-1]:
        if not peaks or all(abs(int(idx_p) - p) >= sep for p in peaks):
            peaks.append(int(idx_p))
        if len(peaks) >= max_candidates:
            break

    return np.asarray(sorted(set(peaks) | {0, 45}), dtype=np.float64)


def _upper_bound_area(hull_poly, angle: float,
                      max_ratio: float, centroid) -> float:
    """
    Cheap O(h) upper bound on the inscribed rectangle area at a given angle
    (h = convex hull vertex count ≪ n).
    Rotates the convex hull and uses half its bounding-box area as the bound
    (provably valid for convex shapes; conservative for non-convex).
    """
    rot = shp_rotate(hull_poly, -angle, origin=centroid, use_radians=False)
    bw, bh = rot.bounds[2] - rot.bounds[0], rot.bounds[3] - rot.bounds[1]
    if max_ratio > 0.0:
        ls = max(bw, bh);
        ss = min(bw, bh)
        if ss > 0 and ls / ss > max_ratio:
            ls = ss * max_ratio
        return ls * ss * 0.5
    return bw * bh * 0.5


def _simplify_for_solve(poly: Polygon):
    """
    Return (simplified_polygon, was_simplified).
    Simplification tolerance = _SIMPLIFY_TOL_FRAC × shorter bbox side.
    Applied only for Stage 1 coarse-grid calls on high-vertex polygons;
    Stage 2 always uses the original polygon for accuracy.
    """
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
# ④ STAGE 1 — HEURISTIC CANDIDATE GENERATOR
#    Produces top_k (angle, coarse_rect, coarse_area) candidates.
#
#    Two-pass approach:
#      Pass 1 — edge-guided angles (dominant edge orientations as prior).
#      Pass 2 — uniform sweep fallback for featureless / isotropic polygons.
#    Upper-bound pruning with _PRUNE_MARGIN skips angles whose theoretical
#    maximum is already beaten by the running best, with a 10% safety margin.
#    Convex hull (O(h) rotation) is used for all upper-bound queries.
#    High-vertex polygons are simplified for the coarse grid call only.
# ==========================================================================
def _heuristic_candidates(poly: Polygon,
                          angle_step: int,
                          grid_coarse: int,
                          grid_fine: int,
                          max_ratio: float,
                          top_k: int) -> list:
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    hull = poly.convex_hull  # used for upper-bound queries
    simplified, _ = _simplify_for_solve(poly)

    raw = []
    best_area = 0.0

    def _solve_coarse(angle_f: float):
        rot_s = shp_rotate(simplified, -angle_f, origin=centroid, use_radians=False)
        return _solve_axis_rect_grid(rot_s, grid_coarse, max_ratio)

    # Pass 1: dominant edge orientations
    for angle in _edge_candidate_angles(poly):
        a = float(angle)
        ub = _upper_bound_area(hull, a, max_ratio, centroid)
        if ub <= best_area * _PRUNE_MARGIN:
            continue
        rect, area = _solve_coarse(a)
        if area > 0:
            raw.append((area, a, rect))
            if area > best_area:
                best_area = area

    # Pass 2: uniform sweep — only when edge heuristic yields < 3 candidates
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

    # Deduplicate by 2° angle proximity, keep top_k
    kept = [];
    seen = []
    for area, angle, rect_rot in raw:
        if any(abs(angle - s) < 2.0 for s in seen):
            continue
        seen.append(angle)
        rect_world = shp_rotate(rect_rot, angle, origin=centroid, use_radians=False)
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
# ⑤ STAGE 2 — REFINEMENT
# ==========================================================================

def _polish_angle(poly: Polygon, candidate: dict,
                  grid_coarse: int, max_ratio: float) -> dict:
    """
    Stage 2: Brent scalar minimisation ±_PHASE_A_HALFWIDTH degrees around
    the candidate angle, using the COARSE grid as the objective function.

    Design rationale
    ────────────────
    Using the coarse grid inside Brent keeps each evaluation cheap (O(g²)
    PIP tests, g = grid_coarse) while still capturing the shape of the
    area-vs-angle landscape.  The fine-grid solve happens exactly once per
    candidate in Stage 3 at the Brent-winning angle.

    Tolerance _PHASE_A_XATOL = 0.02° gives sub-pixel angular precision
    for any polygon whose shortest dimension is > 1 m in a 1:1000 CRS.

    The Brent winner replaces the candidate angle whenever the angle shift
    exceeds 0.005° (i.e. it actually moved — avoids floating-point noise
    producing spurious dict copies).  No area comparison is made: Stage 3
    always performs the definitive fine-grid solve regardless.
    """
    angle_0 = candidate['angle']
    centroid = Point(candidate['center'])
    lo, hi = angle_0 - _PHASE_A_HALFWIDTH, angle_0 + _PHASE_A_HALFWIDTH

    def _neg_area_coarse(a):
        rot = shp_rotate(poly, -a, origin=centroid, use_radians=False)
        _, area = _solve_axis_rect_grid(rot, grid_coarse, max_ratio)
        return -area

    try:
        res = minimize_scalar(_neg_area_coarse, bounds=(lo, hi),
                              method='bounded',
                              options={'xatol': _PHASE_A_XATOL, 'maxiter': 60})
        best_angle = float(res.x)
        if abs(best_angle - angle_0) > 0.005:
            c = candidate.copy()
            c['angle'] = best_angle
            c['area'] = float(-res.fun)  # coarse estimate; Stage 3 re-solves
            return c
    except Exception:
        pass
    return candidate


def _rect_local_frame(rect):
    """
    Decompose a rotated rectangle into its local frame:
    (cx, cy, ux, uy, vx, vy, a, b)
    where (cx,cy) is center, (ux,uy) is the long-axis unit vector,
    (vx,vy) is the short-axis unit vector, a = half long side, b = half short.
    """
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
    cx = float((p0[0] + p2[0]) / 2);
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


def _certify_and_adjust(poly: Polygon, rect,
                        max_ratio: float,
                        buf_enabled: bool, buf_value: float,
                        prepared_poly=None):
    """
    Stage 4: guarantee containment and optionally apply user buffer.

    Containment strategy
    ────────────────────
    1. Fast path — prep.covers(rect): O(1) GEOS prepared-geometry check.
       Done immediately if the rectangle already lies inside the polygon.

    2. Corner-distance sweep — for each of the 4 exterior corners that lies
       outside the polygon, measure its distance to the nearest polygon
       boundary (exterior ring and any hole ring where the corner falls
       inside the hole).  max_ov = maximum of these distances.
       This handles the common case where grid snap or rotation drift
       pushes one or two corners slightly outside.

    3. Interior-crossing fallback (secondary; rare) — called ONLY when all
       4 corners pass the prep.covers() point test but the full covers(rect)
       still fails.  This means a concave polygon boundary cuts through the
       rectangle interior without intersecting any corner.  In this case
       rect.difference(poly) computes the exact overflow geometry and
       sqrt(overflow.area) provides a tighter shrink proxy than the
       bounding-box half-dimension used in earlier versions.

    4. Symmetric shrink of max_ov + _CERT_EPS in the local rectangle frame.
       Rejects (returns None) if the required shrink exceeds _CERT_MAX_SHRINK
       fraction of the shorter side (catastrophic overflow → fallback path).

    5. User buffer applied last (positive = expand, negative = shrink).

    Returns (certified_rect, area) or (None, 0.0).
    """
    if rect is None or rect.is_empty:
        return None, 0.0
    prep = prepared_poly if prepared_poly is not None else shp_prep(poly)

    # 1. Fast path
    if _covers(poly, rect, prep):
        final = rect
    else:
        frame = _rect_local_frame(rect)
        if frame is None:
            return None, 0.0
        cx, cy, ux, uy, vx, vy, a, b = frame

        # 2. Corner-distance sweep
        max_ov = 0.0
        for corner in list(rect.exterior.coords)[:-1]:
            pt = Point(corner)
            if not prep.covers(pt):
                d = poly.exterior.distance(pt)
                for interior in poly.interiors:
                    ip_poly = Polygon(interior)
                    if ip_poly.covers(pt):
                        d = max(d, ip_poly.exterior.distance(pt))
                max_ov = max(max_ov, d)

        # 3. Interior-crossing fallback
        if max_ov < _CERT_EPS:
            try:
                overflow_geom = rect.difference(poly)
                if not overflow_geom.is_empty and overflow_geom.area > 1e-14:
                    # sqrt(area) is a tighter geometric proxy than bbox half-dim
                    max_ov = math.sqrt(overflow_geom.area) + _CERT_EPS
            except Exception:
                max_ov = _CERT_EPS

        # 4. Symmetric shrink
        shrink = max_ov + _CERT_EPS
        if shrink > min(a, b) * _CERT_MAX_SHRINK:
            return None, 0.0

        new_a = a - shrink;
        new_b = b - shrink
        if new_a <= 0 or new_b <= 0:
            return None, 0.0
        if max_ratio > 0.0 and new_b > 0 and new_a / new_b > max_ratio:
            new_a = new_b * max_ratio

        final = _build_rect_from_frame(cx, cy, ux, uy, vx, vy, new_a, new_b)
        if not _covers(poly, final, prep):
            return None, 0.0

    # 5. Optional user buffer
    if buf_enabled and buf_value != 0.0:
        cand = final.buffer(buf_value, cap_style=3, join_style=2)
        if not cand.is_empty and cand.area > 0:
            final = cand

    return final, float(final.area)


def _conservative_inner_fallback(poly, grid_fine, max_ratio,
                                 centroid, angles, prepared_poly=None):
    """
    Fallback solver: progressively inset the polygon boundary and solve
    inside the inset.  Guarantees containment at the cost of some area.
    Used only when all top-K candidates fail certification.
    """
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
            rect_world = shp_rotate(rect_rot, angle, origin=centroid, use_radians=False)
            if _covers(poly, rect_world, prepared_poly):
                best_rect = rect_world
                best_area = float(rect_world.area)
                best_angle = angle
        if best_rect is not None:
            return best_rect, best_area, best_angle

    return None, 0.0, None


def _best_effort_shrink_to_cover(poly, rect, max_ratio,
                                 tol=1e-7, max_iter=40, prepared_poly=None):
    """
    Binary-search the largest uniform scale s ∈ (0, 1] such that
    poly.covers(scale_rect(rect, s)).  Used when _certify_and_adjust
    rejects a candidate (overflow > 20% of shorter side).
    The binary search converges to tol = 1e-7 relative scale, which for a
    10 m rectangle corresponds to a 1 µm precision — sub-grid resolution.
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
        a = a0 * scale;
        b = b0 * scale
        if a <= 0 or b <= 0:
            return None
        if max_ratio > 0.0:
            if max(a, b) / min(a, b) > max_ratio:
                if a >= b:
                    a = b * max_ratio
                else:
                    b = a * max_ratio
        return _build_rect_from_frame(cx, cy, ux, uy, vx, vy, a, b)

    r1 = build(1.0)
    if r1 is not None and _covers(poly, r1, prepared_poly):
        return r1, float(r1.area)

    lo = 0.0;
    r_lo = None
    for s in (0.95, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01):
        r = build(s)
        if r is not None and _covers(poly, r, prepared_poly):
            lo = s;
            r_lo = r;
            break

    if r_lo is None:
        return None, 0.0

    hi = 1.0;
    best_r = r_lo;
    best_a = float(r_lo.area)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        r = build(mid)
        if r is not None and _covers(poly, r, prepared_poly):
            lo = mid;
            best_r = r;
            best_a = float(r.area)
        else:
            hi = mid

    return best_r, best_a


# ==========================================================================
# ⑥ STAGE 2 ORCHESTRATOR
#    Runs Stage 2 → 3 → 4 for each Stage 1 candidate and returns the
#    highest-area certified rectangle.
# ==========================================================================
def _refine_best_candidate(poly: Polygon,
                           candidates: list,
                           grid_coarse: int,
                           grid_fine: int,
                           max_ratio: float,
                           buf_enabled: bool,
                           buf_value: float,
                           always_return: bool,
                           prepared_poly=None):
    """Full Stage 2 pipeline.  Returns 7-tuple or None."""
    certified = []
    fallback_best = None

    for rank, cand in enumerate(candidates):
        area_s1 = cand['area']
        centroid = Point(cand['center'])

        # Stage 2: Brent angle polish (coarse grid only)
        cand_a = _polish_angle(poly, cand, grid_coarse, max_ratio)
        angle_work = cand_a['angle']

        # Stage 3: definitive fine-grid solve at polished angle
        rot_work = shp_rotate(poly, -angle_work,
                              origin=centroid, use_radians=False)
        rect_rot, area_work = _solve_axis_rect_grid(rot_work, grid_fine, max_ratio)

        if rect_rot is None or area_work <= 0:
            continue

        best_raw_r = shp_rotate(rect_rot, angle_work,
                                origin=centroid, use_radians=False)
        best_raw_a = float(area_work)
        best_raw_ang = angle_work

        if fallback_best is None or best_raw_a > fallback_best['area']:
            fallback_best = {
                'rect': best_raw_r,
                'area': best_raw_a,
                'angle': best_raw_ang,
                'rank': rank,
            }

        # Stage 4: containment certification
        best_r, best_a = _certify_and_adjust(
            poly, best_raw_r, max_ratio, False, 0.0, prepared_poly)
        used_best_effort = False

        if best_r is None and always_return:
            best_r, best_a = _best_effort_shrink_to_cover(
                poly, best_raw_r, max_ratio, prepared_poly=prepared_poly)
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
        if always_return and fallback_best is not None:
            rect_fb, area_fb = _best_effort_shrink_to_cover(
                poly, fallback_best['rect'], max_ratio,
                prepared_poly=prepared_poly)
            if rect_fb is not None:
                angle_fb = fallback_best['angle']
                rank_fb = fallback_best['rank']
                coords = list(rect_fb.exterior.coords)
                w = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                h = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
                ratio_fb = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                return (rect_fb, area_fb, angle_fb, ratio_fb,
                        rank_fb, area_fb, True)

            centroid_fb = Point(candidates[0]['center'])
            rescue_angs = [c['angle']
                           for c in candidates[:max(3, min(len(candidates), 8))]]
            rect_c, area_c, angle_c = _conservative_inner_fallback(
                poly, grid_fine, max_ratio, centroid_fb,
                rescue_angs, prepared_poly)
            if rect_c is not None:
                coords = list(rect_c.exterior.coords)
                w = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                h = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
                ratio_c = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                return (rect_c, area_c, angle_c, ratio_c,
                        fallback_best['rank'], area_c, True)

        return None

    best = max(certified, key=lambda c: c['area'])
    return (best['rect'], best['area'], best['angle'],
            best['ratio'], best['rank'], best['stage2_gain'],
            best['used_best_effort'])


# ==========================================================================
# ⑦ GEOMETRY PREPARATION
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
# ⑧ MODULE-LEVEL MULTIPROCESSING ENTRY POINT
# ==========================================================================
def _worker_process_feature(args):
    """
    Stateless worker — safe for ThreadPoolExecutor and ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (feat_id, wkb_bytes, angle_step, grid_coarse, grid_fine,
         max_ratio, buf_enabled, buf_value, top_k, always_return)

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

        # Precision normalisation (Shapely ≥ 2.x) — snaps near-coincident
        # vertices to a sub-nanometre grid, preventing degenerate GEOS results.
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

        prepared_poly = _make_prepared(poly)

        candidates = _heuristic_candidates(
            poly, angle_step, grid_coarse, grid_fine, max_ratio, top_k)
        if not candidates:
            return None

        result = _refine_best_candidate(
            poly, candidates, grid_coarse, grid_fine,
            max_ratio, buf_enabled, buf_value, always_return,
            prepared_poly=prepared_poly)

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
        raise RuntimeError(
            f'_worker_process_feature failed for feat_id={feat_id}: {e}') from e
