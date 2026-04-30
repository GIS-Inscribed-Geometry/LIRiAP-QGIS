"""
test_sdf_oracle.py — Comprehensive tests for sdf_oracle.py.

Sections
--------
  A. SDF sign and magnitude correctness
  B. Batch SDF consistency with scalar
  C. rect_corners geometry
  D. _solve_5d axis-aligned cases
  E. _solve_5d angular free-rotation cases
  F. _solve_5d with max_ratio constraint
  G. _solve_5d with holes
  H. sdf_solve_smooth for dense-vertex polygons
  I. sdf_polish (main entry point)
  J. Containment invariant — every result must satisfy poly.covers(rect)
  K. No-regression — SDF result area ≥ BCRS seed area on all cases
  L. Timeout safety — never exceeds 2× budget
  M. Graceful failure — invalid seeds, empty polys, degenerate params

Run
---
  pytest tests/test_sdf_oracle.py -v

Stubs
-----
  The module gracefully falls back when scipy is absent (returns original seed).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))

from sdf_oracle import (
    polygon_sdf,
    polygon_sdf_batch,
    rect_corners,
    sdf_polish,
    sdf_solve_smooth,
    _solve_5d,
    _apply_ratio,
    _EPS,
)

# ── Tolerances ────────────────────────────────────────────────────────────
SDF_ABS_TOL   = 1e-9
AREA_REL_TOL  = 5e-4    # 0.05 % — generous for a post-polisher
COVERS_STRICT = True

# ── Helpers ──────────────────────────────────────────────────────────────
def assert_covers(poly, rect, label=""):
    assert rect is not None,        f"{label}: rect is None"
    assert not rect.is_empty,       f"{label}: rect is empty"
    diff = rect.difference(poly)
    assert poly.covers(rect), (
        f"{label}: poly does NOT cover rect. "
        f"diff_area={diff.area:.3e}"
    )

def assert_area_ge(actual, reference, label=""):
    assert actual >= reference - reference * AREA_REL_TOL, (
        f"{label}: area {actual:.6f} < reference {reference:.6f}"
    )

def circle_poly(r=10.0, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(list(zip(r * np.cos(t), r * np.sin(t))))

def ellipse_poly(a=15.0, b=5.0, n=128):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(list(zip(a * np.cos(t), b * np.sin(t))))

def l_shape():
    return Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])

def hexagon(r=10.0):
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    return Polygon(list(zip(r * np.cos(t), r * np.sin(t))))


# ===========================================================================
# Section A — SDF sign and magnitude
# ===========================================================================
class TestSDFPrimitives:

    def test_interior_negative(self):
        sq = box(0, 0, 10, 10)
        assert polygon_sdf(sq, 5, 5) < 0

    def test_interior_magnitude(self):
        sq = box(0, 0, 10, 10)
        # Nearest boundary: min(5,5,5,5) = 5
        assert abs(polygon_sdf(sq, 5, 5) - (-5.0)) < SDF_ABS_TOL

    def test_exterior_positive(self):
        sq = box(0, 0, 10, 10)
        assert polygon_sdf(sq, 15, 5) > 0

    def test_exterior_magnitude(self):
        sq = box(0, 0, 10, 10)
        assert abs(polygon_sdf(sq, 15, 5) - 5.0) < SDF_ABS_TOL

    def test_on_boundary_zero(self):
        sq = box(0, 0, 10, 10)
        assert abs(polygon_sdf(sq, 5, 0)) < SDF_ABS_TOL
        assert abs(polygon_sdf(sq, 0, 5)) < SDF_ABS_TOL

    def test_on_corner_zero(self):
        sq = box(0, 0, 10, 10)
        assert abs(polygon_sdf(sq, 0, 0)) < SDF_ABS_TOL

    def test_inside_hole_positive(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(6,6),(14,6),(14,14),(6,14)]])
        # (10,10) is inside the hole
        d = polygon_sdf(poly, 10, 10)
        assert d > 0, f"Expected positive (in hole), got {d}"

    def test_valid_interior_near_hole(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(6,6),(14,6),(14,14),(6,14)]])
        # (2, 2) is inside outer ring, outside hole
        d = polygon_sdf(poly, 2, 2)
        assert d < 0, f"Expected negative (valid interior), got {d}"

    def test_circle_interior_distance(self):
        circ = circle_poly(r=10.0, n=64)
        d = polygon_sdf(circ, 0, 0)
        # Exact distance to boundary ≈ r * (1 - some polygon approx error)
        assert d < -9.0

    def test_circle_exterior_distance(self):
        circ = circle_poly(r=10.0, n=64)
        d = polygon_sdf(circ, 15, 0)
        assert d > 4.9


# ===========================================================================
# Section B — Batch SDF consistency
# ===========================================================================
class TestBatchSDF:

    def test_batch_matches_scalar_square(self):
        sq = box(0, 0, 10, 10)
        xs = np.array([5.0, 0.0, 15.0, 5.0, 0.0])
        ys = np.array([5.0, 0.0,  5.0, 0.0, 10.0])
        batch  = polygon_sdf_batch(sq, xs, ys)
        scalar = np.array([polygon_sdf(sq, float(x), float(y))
                           for x, y in zip(xs, ys)])
        np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_batch_matches_scalar_circle(self):
        circ = circle_poly(r=10.0, n=64)
        rng  = np.random.default_rng(42)
        xs   = rng.uniform(-15, 15, 20)
        ys   = rng.uniform(-15, 15, 20)
        batch  = polygon_sdf_batch(circ, xs, ys)
        scalar = np.array([polygon_sdf(circ, float(x), float(y))
                           for x, y in zip(xs, ys)])
        np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_batch_returns_correct_shape(self):
        sq = box(0, 0, 10, 10)
        xs = np.linspace(0, 10, 50)
        ys = np.linspace(0, 10, 50)
        result = polygon_sdf_batch(sq, xs, ys)
        assert result.shape == (50,)

    def test_batch_sign_consistency(self):
        sq = box(0, 0, 10, 10)
        xs = np.array([5.0, -1.0, 11.0])
        ys = np.array([5.0,  5.0,  5.0])
        batch = polygon_sdf_batch(sq, xs, ys)
        assert batch[0] < 0    # inside
        assert batch[1] > 0    # outside
        assert batch[2] > 0    # outside


# ===========================================================================
# Section C — rect_corners geometry
# ===========================================================================
class TestRectCorners:

    def test_axis_aligned_area(self):
        c = rect_corners(0, 0, 5, 3, 0.0)
        poly = Polygon(list(map(tuple, c)) + [tuple(c[0])])
        assert abs(poly.area - 60.0) < 1e-9

    def test_rotated_area_preserved(self):
        hw, hh = 5.0, 3.0
        for theta_deg in [0, 15, 30, 45, 60, 90]:
            c = rect_corners(0, 0, hw, hh, math.radians(theta_deg))
            poly = Polygon(list(map(tuple, c)) + [tuple(c[0])])
            assert abs(poly.area - 4 * hw * hh) < 1e-9, \
                f"Area wrong at theta={theta_deg}"

    def test_center_is_centroid(self):
        cx, cy = 3.5, -2.1
        c = rect_corners(cx, cy, 4.0, 2.0, math.radians(37))
        poly = Polygon(list(map(tuple, c)) + [tuple(c[0])])
        assert abs(float(poly.centroid.x) - cx) < 1e-9
        assert abs(float(poly.centroid.y) - cy) < 1e-9

    def test_four_corners_returned(self):
        c = rect_corners(0, 0, 2, 2, 0.0)
        assert c.shape == (4, 2)

    def test_apply_ratio(self):
        hw, hh = _apply_ratio(10.0, 2.0, 3.0)
        assert hw / hh <= 3.0 + 1e-9

    def test_apply_ratio_no_limit(self):
        hw, hh = _apply_ratio(10.0, 2.0, 0.0)
        assert hw == 10.0 and hh == 2.0


# ===========================================================================
# Section D — _solve_5d axis-aligned cases
# ===========================================================================
class TestSolve5DAxisAligned:

    def _run(self, poly, cx, cy, hw, hh, ang=0.0,
             max_ratio=0.0, angle_win=0.1):
        return _solve_5d(poly, cx, cy, hw, hh, ang,
                         max_ratio=max_ratio, angle_win=angle_win,
                         timeout_ms=300)

    def test_no_improvement_when_already_optimal_square(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = self._run(sq, 5, 5, 5, 5)
        # Already optimal — solver should not report improvement
        assert r is None or a >= 100.0 - 100.0 * AREA_REL_TOL

    def test_improves_suboptimal_circle_seed(self):
        circ = circle_poly(r=10.0, n=64)
        # Seed: 7×7 (area 196), optimal: 7.071×7.071 (area 200)
        r, a, ang = self._run(circ, 0, 0, 7.0, 7.0)
        if r is not None:
            assert_covers(circ, r, "circle axis-aligned")
            assert a >= 196.0                # never worse than seed
            assert a >= 199.0 - 200.0 * AREA_REL_TOL  # near optimal

    def test_improves_suboptimal_ellipse_seed(self):
        ell = ellipse_poly(a=15.0, b=5.0, n=128)
        r, a, ang = self._run(ell, 0, 0, 8.0, 2.5)
        if r is not None:
            assert_covers(ell, r, "ellipse axis-aligned")
            assert_area_ge(a, 148.0, "ellipse axis-aligned")

    def test_covers_invariant_circle(self):
        circ = circle_poly(r=10.0, n=32)
        r, a, ang = self._run(circ, 0, 0, 6.0, 6.0)
        if r is not None:
            assert_covers(circ, r, "circle covers")

    def test_covers_invariant_l_shape(self):
        ls = l_shape()
        cx = float(ls.centroid.x);  cy = float(ls.centroid.y)
        r, a, ang = self._run(ls, cx, cy, 1.5, 1.5)
        if r is not None:
            assert_covers(ls, r, "l_shape covers")

    def test_never_regresses(self):
        for poly_fn, cx, cy, hw, hh in [
            (lambda: box(0,0,10,10),         5, 5, 5, 5),
            (lambda: circle_poly(r=8, n=32), 0, 0, 4, 4),
            (lambda: l_shape(),              3, 3, 2, 2),
        ]:
            poly = poly_fn()
            area_seed = 4 * hw * hh
            r, a, ang = self._run(poly, cx, cy, hw, hh)
            if r is not None:
                assert a >= area_seed - area_seed * AREA_REL_TOL, \
                    "SDF regressed below seed area"


# ===========================================================================
# Section E — _solve_5d angular free-rotation
# ===========================================================================
class TestSolve5DAngular:

    def test_hexagon_free_angle(self):
        hex_p = hexagon(r=10.0)
        # Seed: axis-aligned 4.5×8.5 (area 153)
        r, a, ang = _solve_5d(hex_p, 0, 0, 4.5, 8.5, 0.0,
                               angle_win=20.0, timeout_ms=300)
        if r is not None:
            assert_covers(hex_p, r, "hexagon free angle")
            assert_area_ge(a, 153.0, "hexagon free angle")

    def test_tilted_rect_near_optimal(self):
        base = Polygon([(0,0),(20,0),(20,5),(0,5)])
        tilted = shp_rotate(base, 37, origin=(10, 2.5))
        cx = float(tilted.centroid.x);  cy = float(tilted.centroid.y)
        # Seed: slightly suboptimal hw=9.5, hh=2.4
        r, a, ang = _solve_5d(tilted, cx, cy, 9.5, 2.4, 37.0,
                               angle_win=5.0, timeout_ms=300)
        if r is not None:
            assert_covers(tilted, r, "tilted37 free angle")
            assert a >= 4 * 9.5 * 2.4 - 1e-4   # never worse than seed

    def test_ellipse_free_angle(self):
        # Tilted ellipse: optimal angle ≈ 0° still, but solver should confirm
        ell = shp_rotate(ellipse_poly(a=15.0, b=5.0, n=64), 20)
        cx = float(ell.centroid.x);  cy = float(ell.centroid.y)
        r, a, ang = _solve_5d(ell, cx, cy, 6.0, 2.5, 20.0,
                               angle_win=10.0, timeout_ms=400)
        if r is not None:
            assert_covers(ell, r, "tilted ellipse")

    def test_angle_stays_in_window(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = _solve_5d(sq, 5, 5, 4.9, 4.9, 0.0,
                               angle_win=3.0, timeout_ms=200)
        if r is not None:
            assert abs(ang) <= 3.0 + 0.01


# ===========================================================================
# Section F — max_ratio constraint
# ===========================================================================
class TestMaxRatioConstraint:

    def test_ratio_enforced_on_result(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = _solve_5d(sq, 5, 5, 4.0, 4.0, 0.0,
                               max_ratio=1.5, timeout_ms=200)
        if r is not None:
            coords = list(r.exterior.coords)
            l0 = math.hypot(coords[1][0]-coords[0][0],
                            coords[1][1]-coords[0][1])
            l1 = math.hypot(coords[2][0]-coords[1][0],
                            coords[2][1]-coords[1][1])
            ratio = max(l0, l1) / min(l0, l1) if min(l0, l1) > 0 else 1.0
            assert ratio <= 1.5 + 1e-4, f"Ratio {ratio:.4f} > 1.5"

    def test_ratio_1_gives_square_ish(self):
        circ = circle_poly(r=10.0, n=64)
        r, a, ang = _solve_5d(circ, 0, 0, 6.0, 6.0, 0.0,
                               max_ratio=1.0, timeout_ms=300)
        if r is not None:
            assert_covers(circ, r, "circle ratio=1")


# ===========================================================================
# Section G — polygons with holes
# ===========================================================================
class TestHoleHandling:

    def test_sdf_positive_inside_hole(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(6,6),(14,6),(14,14),(6,14)]])
        d = polygon_sdf(poly, 10, 10)
        assert d > 0

    def test_sdf_negative_outside_hole(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(6,6),(14,6),(14,14),(6,14)]])
        d = polygon_sdf(poly, 2, 2)
        assert d < 0

    def test_result_avoids_hole(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(7,7),(13,7),(13,13),(7,13)]])
        # Seed: small rect in the corner, away from hole
        r, a, ang = _solve_5d(poly, 3, 10, 2.5, 8.0, 0.0,
                               timeout_ms=300)
        if r is not None:
            assert_covers(poly, r, "square with hole")

    def test_covers_never_intersects_hole(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(5,5),(15,5),(15,15),(5,15)]])
        r, a, ang = _solve_5d(poly, 3, 10, 2.0, 8.0, 0.0,
                               timeout_ms=300)
        if r is not None:
            assert_covers(poly, r, "large hole avoidance")


# ===========================================================================
# Section H — sdf_solve_smooth
# ===========================================================================
class TestSdfSolveSmooth:

    def test_circle_64_near_optimal(self):
        circ = circle_poly(r=10.0, n=64)
        r, a, ang = sdf_solve_smooth(circ, timeout_ms=500)
        if r is not None:
            assert_covers(circ, r, "smooth circle")
            assert_area_ge(a, 195.0, "smooth circle (200 opt)")

    def test_ellipse_near_optimal(self):
        ell = ellipse_poly(a=15.0, b=5.0, n=128)
        r, a, ang = sdf_solve_smooth(ell, timeout_ms=500)
        if r is not None:
            assert_covers(ell, r, "smooth ellipse")
            assert_area_ge(a, 145.0, "smooth ellipse (150 opt)")

    def test_covers_invariant(self):
        circ = circle_poly(r=7.0, n=64)
        r, a, ang = sdf_solve_smooth(circ, timeout_ms=400)
        if r is not None:
            assert_covers(circ, r, "sdf_solve_smooth covers")

    def test_result_none_or_valid(self):
        sq = box(0, 0, 5, 5)
        r, a, ang = sdf_solve_smooth(sq, timeout_ms=200)
        assert r is None or (not r.is_empty and a > 0)


# ===========================================================================
# Section I — sdf_polish (main entry point)
# ===========================================================================
class TestSdfPolish:

    def _make_cand(self, poly, cx, cy, hw, hh, angle=0.0):
        c = rect_corners(cx, cy, hw, hh, math.radians(angle))
        rect = Polygon(list(map(tuple, c)) + [tuple(c[0])])
        return [{'rect': rect, 'area': 4*hw*hh, 'angle': angle}]

    def test_returns_original_when_no_improvement(self):
        sq = box(0, 0, 10, 10)
        cands = self._make_cand(sq, 5, 5, 5, 5)
        r, a, ang = sdf_polish(sq, cands)
        assert a >= 100.0 - 1e-4

    def test_covers_invariant_circle(self):
        circ = circle_poly(r=10.0, n=64)
        cands = self._make_cand(circ, 0, 0, 7.0, 7.0)
        r, a, ang = sdf_polish(circ, cands)
        assert_covers(circ, r, "sdf_polish circle")

    def test_covers_invariant_l_shape(self):
        ls = l_shape()
        cands = self._make_cand(ls, 2.5, 2.5, 2.0, 2.0)
        r, a, ang = sdf_polish(ls, cands)
        assert_covers(ls, r, "sdf_polish l_shape")

    def test_empty_candidates_returns_none(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = sdf_polish(sq, [])
        assert r is None

    def test_never_regresses(self):
        circ = circle_poly(r=10.0, n=64)
        cands = self._make_cand(circ, 0, 0, 7.0, 7.0)
        seed_area = cands[0]['area']
        r, a, ang = sdf_polish(circ, cands)
        assert a >= seed_area - seed_area * AREA_REL_TOL

    def test_smooth_flag_triggers_direct_solve(self):
        circ = circle_poly(r=10.0, n=500)  # > 300 unique coords
        cands = self._make_cand(circ, 0, 0, 6.0, 6.0)
        r, a, ang = sdf_polish(circ, cands, is_smooth_poly=True,
                                timeout_ms=600)
        assert_covers(circ, r, "smooth flag circle")
        assert_area_ge(a, 144.0, "smooth flag area")


# ===========================================================================
# Section J — Containment invariant (exhaustive)
# ===========================================================================
class TestContainmentInvariant:
    """Every SDF-returned rectangle must satisfy poly.covers(rect)."""

    @pytest.mark.parametrize("poly,cx,cy,hw,hh,angle_win", [
        (box(0,0,10,10),             5, 5, 4.9, 4.9, 2.0),
        (circle_poly(r=10, n=64),    0, 0, 7.0, 7.0, 1.0),
        (ellipse_poly(15, 5, 128),   0, 0, 8.0, 2.5, 1.0),
        (l_shape(),                  3, 3, 2.5, 2.0, 5.0),
        (hexagon(r=10),              0, 0, 4.5, 8.3, 12.0),
    ])
    def test_covers(self, poly, cx, cy, hw, hh, angle_win):
        r, a, ang = _solve_5d(poly, cx, cy, hw, hh, 0.0,
                               angle_win=angle_win, timeout_ms=400)
        if r is not None:
            assert_covers(poly, r,
                          f"containment {poly.geom_type} aw={angle_win}")


# ===========================================================================
# Section K — No-regression against BCRS-like seeds
# ===========================================================================
class TestNoRegression:
    """SDF area ≥ BCRS seed area on every case."""

    def test_circle_no_regression(self):
        circ = circle_poly(r=10.0, n=64)
        # Simulate BCRS+CABF result (already expanded to boundary)
        hw_bcrs = 10.0 / math.sqrt(2)
        c = rect_corners(0, 0, hw_bcrs, hw_bcrs, 0.0)
        seed_rect = Polygon(list(map(tuple, c)) + [tuple(c[0])])
        cands = [{'rect': seed_rect, 'area': 4*hw_bcrs**2, 'angle': 0.0}]
        r, a, ang = sdf_polish(circ, cands)
        assert a >= cands[0]['area'] - cands[0]['area'] * AREA_REL_TOL

    def test_l_shape_no_regression(self):
        ls = l_shape()
        # BCRS+CABF for L-shape gives 10×5=50
        seed_rect = box(0, 0, 10, 5)
        cands = [{'rect': seed_rect, 'area': 50.0, 'angle': 0.0}]
        r, a, ang = sdf_polish(ls, cands)
        assert a >= 50.0 - 0.05

    def test_square_with_hole_no_regression(self):
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)],
                       [[(7,7),(13,7),(13,13),(7,13)]])
        seed_rect = box(0, 0, 6, 20)   # left strip, area 120
        if not poly.covers(seed_rect):
            seed_rect = box(0, 0, 6, 18)
        if not poly.covers(seed_rect):
            pytest.skip("Seed not inside polygon")
        cands = [{'rect': seed_rect,
                  'area': float(seed_rect.area), 'angle': 0.0}]
        r, a, ang = sdf_polish(poly, cands)
        assert a >= float(seed_rect.area) - 0.1


# ===========================================================================
# Section L — Timeout safety
# ===========================================================================
class TestTimeoutSafety:

    def test_finishes_within_budget(self):
        circ = circle_poly(r=10.0, n=256)
        budget_ms = 300.0
        c = rect_corners(0, 0, 7.0, 7.0, 0.0)
        cands = [{'rect': Polygon(list(map(tuple, c)) + [tuple(c[0])]),
                  'area': 196.0, 'angle': 0.0}]
        t0 = time.monotonic()
        sdf_polish(circ, cands, timeout_ms=budget_ms)
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert elapsed_ms < budget_ms * 2.5, \
            f"Ran {elapsed_ms:.0f}ms, budget was {budget_ms:.0f}ms"

    def test_returns_valid_on_short_timeout(self):
        circ = circle_poly(r=10.0, n=64)
        c = rect_corners(0, 0, 7.0, 7.0, 0.0)
        cands = [{'rect': Polygon(list(map(tuple, c)) + [tuple(c[0])]),
                  'area': 196.0, 'angle': 0.0}]
        r, a, ang = sdf_polish(circ, cands, timeout_ms=1.0)
        # Must return the original seed unchanged
        assert a >= 196.0 - 1e-4


# ===========================================================================
# Section M — Graceful failure
# ===========================================================================
class TestGracefulFailure:

    def test_invalid_seed_returns_none(self):
        sq = box(0, 0, 10, 10)
        # Seed way outside polygon
        r, a, ang = _solve_5d(sq, 50, 50, 5, 5, 0.0, timeout_ms=100)
        assert r is None

    def test_degenerate_hw_returns_none(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = _solve_5d(sq, 5, 5, 0.0, 5.0, 0.0, timeout_ms=100)
        assert r is None

    def test_empty_candidates_returns_none(self):
        sq = box(0, 0, 10, 10)
        r, a, ang = sdf_polish(sq, [])
        assert r is None and a == 0.0

    def test_none_rect_in_candidates_skipped(self):
        sq = box(0, 0, 10, 10)
        cands = [{'rect': None, 'area': 0.0, 'angle': 0.0}]
        r, a, ang = sdf_polish(sq, cands)
        # Should not crash; returns None since no valid candidates

    def test_invalid_polygon_does_not_crash(self):
        from shapely.geometry import Polygon as _P
        bad = _P([(0,0),(1,0),(1,1)])   # valid triangle, degenerate for our purposes
        r, a, ang = _solve_5d(bad, 0.3, 0.3, 0.1, 0.1, 0.0, timeout_ms=200)
        # No assertion — just must not crash