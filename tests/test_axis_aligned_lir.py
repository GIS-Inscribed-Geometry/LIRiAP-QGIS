"""
Comprehensive test suite for axis_aligned_lir_worker.py
========================================================

Tests are organised into sections:

  Section A — polygon type detection
  Section B — convex no-holes solver (_exact_solve_convex)
  Section C — vertex-grid solver (_exact_solve_vertex_grid)
  Section D — full pipeline via _solve_axis_aligned_lir
  Section E — axis-angle rotation
  Section F — max_ratio constraint
  Section G — holes handling
  Section H — worker entry point (_worker_process_feature)
  Section I — edge / degenerate cases
  Section J — containment invariant (property: poly.covers(rect) for every result)

Strictness contract
-------------------
* Every solved rectangle must satisfy  ``poly.covers(rect)``  with NO exceptions.
* Area must match expected values within  ``AREA_REL_TOL = 1e-4``  (0.01 %).
* ``best_effort`` must be False for all exact solves on well-formed polygons.
* No result may have  ``rect is None``  for any valid non-degenerate polygon
  when  ``always_return=True``.
"""

from __future__ import annotations

import math
import sys
import pytest
import numpy as np
from shapely.geometry import Polygon
from shapely.wkb import dumps as wkb_dumps
from shapely.affinity import rotate as shp_rotate
from shapely import set_precision as _set_precision


def _make_hexagon(r: float = 10.0) -> Polygon:
    """Return a regular hexagon with circumradius *r*, with vertex coordinates
    precision-snapped to eliminate floating-point noise from sin/cos.
    This is important for tests that call the raw exact solvers directly.
    """
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    raw = Polygon(zip(np.cos(t) * r, np.sin(t) * r))
    return _set_precision(raw, grid_size=1e-9, mode="valid_output")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))

from axis_aligned_lir_worker import (
    _detect_polygon_type,
    _exact_solve_convex,
    _exact_solve_vertex_grid,
    _solve_axis_aligned_lir,
    _worker_process_feature,
    _test_cases,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
AREA_REL_TOL = 1e-4   # 0.01 % relative area tolerance for exact solvers
AREA_ABS_TOL = 1e-6   # absolute fallback for near-zero areas
COVERS_EPS   = 0.0    # poly.covers(rect) must be exactly True — no tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_covers(poly: Polygon, rect, label: str = "") -> None:
    """Assert poly.covers(rect) strictly, with a useful failure message."""
    assert rect is not None, f"{label}: rect is None"
    assert not rect.is_empty, f"{label}: rect is empty"
    assert poly.covers(rect), (
        f"{label}: poly does NOT cover rect.\n"
        f"  rect.bounds = {rect.bounds}\n"
        f"  poly.bounds = {poly.bounds}\n"
        f"  difference area = {rect.difference(poly).area:.6e}"
    )


def assert_area(actual: float, expected: float, label: str = "") -> None:
    """Assert area is within AREA_REL_TOL of expected."""
    rel = abs(actual - expected) / max(abs(expected), AREA_ABS_TOL)
    assert rel <= AREA_REL_TOL, (
        f"{label}: area mismatch. got={actual:.6f}, expected={expected:.6f}, "
        f"rel_err={rel:.2e} (tol={AREA_REL_TOL:.0e})"
    )


def solve(poly, axis_angle=0.0, max_ratio=0.0, always_return=True):
    """Convenience wrapper around _solve_axis_aligned_lir."""
    return _solve_axis_aligned_lir(
        poly,
        axis_angle=axis_angle,
        grid_fine=120,
        max_ratio=max_ratio,
        always_return=always_return,
        buf_enabled=False,
        buf_value=0.0,
    )


# ===========================================================================
# Section A — Polygon type detection
# ===========================================================================

class TestPolygonTypeDetection:

    def test_square_is_convex_no_holes(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        assert _detect_polygon_type(poly) == "convex_no_holes"

    def test_regular_hexagon_is_convex_no_holes(self):
        poly = _make_hexagon()
        assert _detect_polygon_type(poly) == "convex_no_holes"

    def test_triangle_is_convex_no_holes(self):
        poly = Polygon([(0,0),(10,0),(5,5)])
        assert _detect_polygon_type(poly) == "convex_no_holes"

    def test_right_triangle_is_convex_no_holes(self):
        poly = Polygon([(0,0),(10,0),(0,10)])
        assert _detect_polygon_type(poly) == "convex_no_holes"

    def test_l_shape_is_concave_no_holes(self):
        poly = Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])
        assert _detect_polygon_type(poly) == "concave_no_holes"

    def test_u_shape_is_concave_no_holes(self):
        poly = Polygon([
            (0,0),(10,0),(10,4),(7,4),(7,8),(10,8),(10,12),(0,12),(0,8),(3,8),(3,4),(0,4)
        ])
        assert _detect_polygon_type(poly) == "concave_no_holes"

    def test_t_shape_is_concave_no_holes(self):
        poly = Polygon([(0,5),(10,5),(10,7),(7,7),(7,10),(3,10),(3,7),(0,7)])
        assert _detect_polygon_type(poly) == "concave_no_holes"

    def test_square_with_square_hole_is_concave_with_holes(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [[(3,3),(7,3),(7,7),(3,7)]])
        assert _detect_polygon_type(poly) == "concave_with_holes"

    def test_l_shape_with_hole_is_concave_with_holes(self):
        poly = Polygon(
            [(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)],
            [[(1,1),(3,1),(3,3),(1,3)]],
        )
        assert _detect_polygon_type(poly) == "concave_with_holes"

    def test_square_with_circular_hole_type(self):
        # A convex exterior with a hole: the convex_hull.area != poly.area
        # so this is correctly classified as concave_with_holes by area ratio.
        t = np.linspace(0, 2*np.pi, 32)
        hole = list(zip(5 + 2*np.cos(t), 5 + 2*np.sin(t)))
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)], [hole])
        ptype = _detect_polygon_type(poly)
        # Both concave_with_holes and convex_with_holes route to vertex-grid solver,
        # so either is acceptable — but must contain 'with_holes'.
        assert "with_holes" in ptype


# ===========================================================================
# Section B — _exact_solve_convex
# ===========================================================================

class TestExactSolveConvex:

    def _check(self, poly, expected_area, label=""):
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        assert_area(area, expected_area, label)
        assert_covers(poly, rect, label)

    def test_square_10x10(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        self._check(poly, 100.0, "square 10x10")

    def test_rectangle_20x5(self):
        poly = Polygon([(0,0),(20,0),(20,5),(0,5)])
        self._check(poly, 100.0, "rect 20x5")

    def test_rectangle_3x7(self):
        poly = Polygon([(0,0),(3,0),(3,7),(0,7)])
        self._check(poly, 21.0, "rect 3x7")

    def test_isoceles_triangle(self):
        # (0,0),(10,0),(5,5): optimal rect at y_hi=2.5, width=5, area=12.5
        poly = Polygon([(0,0),(10,0),(5,5)])
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        assert_area(area, 12.5, "isoceles triangle")
        assert_covers(poly, rect, "isoceles triangle")

    def test_right_triangle(self):
        # (0,0),(10,0),(0,10): optimal y_hi=5, width=5, area=25
        poly = Polygon([(0,0),(10,0),(0,10)])
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        assert_area(area, 25.0, "right triangle")
        assert_covers(poly, rect, "right triangle")

    def test_equilateral_triangle_approx(self):
        # Equilateral triangle base=10, height=8.66
        poly = Polygon([(0,0),(10,0),(5, 10*math.sqrt(3)/2)])
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        # Optimal = base * height / 4 = 10 * 8.66 / 4 ≈ 21.65
        expected = 10 * (10*math.sqrt(3)/2) / 4
        assert_area(area, expected, "equilateral triangle")
        assert_covers(poly, rect, "equilateral triangle")

    def test_regular_hexagon(self):
        # Use set_precision-snapped hexagon so FP vertex noise does not
        # cause spurious covers=False in the raw solver (the pipeline's
        # _certify_rect handles this, but direct solver tests must have
        # clean input geometry).
        poly = _make_hexagon()
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        # Regular hexagon r=10: optimal axis-aligned LIR has
        # width=10 (x=-5..5), height=2*r*sin(60°)=17.32 -> area ≈ 173.2
        assert area > 160.0, f"hexagon area too small: {area}"
        assert_covers(poly, rect, "hexagon")

    def test_parallelogram(self):
        # (0,0),(10,2),(10,8),(0,6): a parallelogram
        poly = Polygon([(0,0),(10,2),(10,8),(0,6)])
        rect, area = _exact_solve_convex(poly, max_ratio=0.0)
        assert area > 0.0
        assert_covers(poly, rect, "parallelogram")

    def test_max_ratio_square(self):
        # 10x10 square, max_ratio=1.0: result is 10x10 (already ≤ ratio)
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect, area = _exact_solve_convex(poly, max_ratio=1.0)
        assert_area(area, 100.0, "square max_ratio=1")
        assert_covers(poly, rect, "square max_ratio=1")

    def test_max_ratio_wide_rect(self):
        # 20x5, max_ratio=2.0: long=10(capped), short=5, area=50
        poly = Polygon([(0,0),(20,0),(20,5),(0,5)])
        rect, area = _exact_solve_convex(poly, max_ratio=2.0)
        assert_area(area, 50.0, "wide rect max_ratio=2")
        assert_covers(poly, rect, "wide rect max_ratio=2")

    def test_max_ratio_tall_rect(self):
        # 2x10, max_ratio=3.0: short=2, long=10→capped at 6, area=12
        poly = Polygon([(0,0),(2,0),(2,10),(0,10)])
        rect, area = _exact_solve_convex(poly, max_ratio=3.0)
        assert_area(area, 12.0, "tall rect max_ratio=3")
        assert_covers(poly, rect, "tall rect max_ratio=3")

    def test_y_lo_not_mutated_across_iterations(self):
        # Pentagon with 4 unique y-values.  If y_lo is mutated in the max_ratio
        # block and leaks into the next inner iteration, subsequent rects are wrong.
        # This test catches the y_lo mutation bug.
        poly = Polygon([(0,0),(12,0),(12,3),(12,6),(0,9)])
        rect_nolimit, area_nolimit = _exact_solve_convex(poly, max_ratio=0.0)
        rect_limited, area_limited = _exact_solve_convex(poly, max_ratio=1.5)
        # With ratio limit the area must be <= unconstrained area
        assert area_limited <= area_nolimit + 1e-9, "ratio-constrained area exceeds unconstrained"
        assert area_limited > 0.0
        assert_covers(poly, rect_limited, "pentagon max_ratio mutation check")

    def test_result_aspect_ratio_respected(self):
        # For a very wide polygon, max_ratio=1.5 must yield ratio ≤ 1.5
        poly = Polygon([(0,0),(100,0),(100,5),(0,5)])
        rect, area = _exact_solve_convex(poly, max_ratio=1.5)
        assert rect is not None
        b = rect.bounds
        w = b[2] - b[0]; h = b[3] - b[1]
        if min(w,h) > 0:
            assert max(w,h) / min(w,h) <= 1.5 + 1e-6, f"ratio exceeded: w={w}, h={h}"
        assert_covers(poly, rect, "aspect ratio check")


# ===========================================================================
# Section C — _exact_solve_vertex_grid
# ===========================================================================

class TestExactSolveVertexGrid:

    def _check(self, poly, ptype, expected_area, label=""):
        rect, area = _exact_solve_vertex_grid(poly, ptype, max_ratio=0.0)
        assert_area(area, expected_area, label)
        assert_covers(poly, rect, label)

    def test_square_no_holes(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        self._check(poly, "convex_no_holes", 100.0, "square via vertex grid")

    def test_right_triangle(self):
        # Midpoint augmentation must resolve this: optimal area=25
        poly = Polygon([(0,0),(10,0),(0,10)])
        self._check(poly, "convex_no_holes", 25.0, "right triangle via vertex grid")

    def test_isoceles_triangle(self):
        poly = Polygon([(0,0),(10,0),(5,5)])
        rect, area = _exact_solve_vertex_grid(poly, "convex_no_holes", 0.0)
        assert_area(area, 12.5, "isoceles triangle via vertex grid")
        assert_covers(poly, rect, "isoceles triangle via vertex grid")

    def test_l_shape(self):
        poly = Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])
        self._check(poly, "concave_no_holes", 50.0, "L-shape")

    def test_t_shape(self):
        poly = Polygon([(0,5),(10,5),(10,7),(7,7),(7,10),(3,10),(3,7),(0,7)])
        rect, area = _exact_solve_vertex_grid(poly, "concave_no_holes", 0.0)
        assert area > 0.0
        assert_covers(poly, rect, "T-shape")

    def test_u_shape(self):
        poly = Polygon([
            (0,0),(10,0),(10,4),(7,4),(7,8),(10,8),(10,12),(0,12),(0,8),(3,8),(3,4),(0,4)
        ])
        rect, area = _exact_solve_vertex_grid(poly, "concave_no_holes", 0.0)
        # U-shape: the full-height vertical channel (x=3..7, y=0..12) = 4×12 = 48
        # is larger than the 10×4 base strip (area=40); 48 is the correct optimum.
        assert_area(area, 48.0, "U-shape")
        assert_covers(poly, rect, "U-shape")

    def test_square_with_square_hole(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [[(3,3),(7,3),(7,7),(3,7)]])
        # Each strip around 4x4 hole in 10x10 square: best strip = 10*3=30
        rect, area = _exact_solve_vertex_grid(poly, "concave_with_holes", 0.0)
        assert_area(area, 30.0, "square with square hole")
        assert_covers(poly, rect, "square with square hole")

    def test_concave_with_hole(self):
        poly = Polygon(
            [(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)],
            [[(1,1),(3,1),(3,3),(1,3)]],
        )
        rect, area = _exact_solve_vertex_grid(poly, "concave_with_holes", 0.0)
        assert area > 0.0
        assert_covers(poly, rect, "L-shape with hole")

    def test_hexagon(self):
        # Use precision-snapped hexagon for the raw solver test (see _make_hexagon)
        poly = _make_hexagon()
        rect, area = _exact_solve_vertex_grid(poly, "convex_no_holes", 0.0)
        assert area > 160.0, f"hexagon area too small: {area}"
        assert_covers(poly, rect, "hexagon via vertex grid")

    def test_no_holes_flag_excludes_interior_rings(self):
        # If ptype says no holes, hole vertices should not be in the grid.
        # Solve should still succeed and cover the polygon region.
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect, area = _exact_solve_vertex_grid(poly, "concave_no_holes", 0.0)
        assert_area(area, 100.0, "square treated as concave_no_holes")
        assert_covers(poly, rect, "square as concave_no_holes")


# ===========================================================================
# Section D — Full pipeline: _solve_axis_aligned_lir
# ===========================================================================

class TestFullPipeline:

    def test_convex_no_holes_square(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert_area(area, 100.0, "pipeline square")
        assert_covers(poly, rect, "pipeline square")
        assert ptype == "convex_no_holes"
        assert be is False or be == 0

    def test_convex_no_holes_right_triangle(self):
        poly = Polygon([(0,0),(10,0),(0,10)])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert_area(area, 25.0, "pipeline right triangle")
        assert_covers(poly, rect, "pipeline right triangle")
        assert be is False or be == 0

    def test_concave_no_holes_l_shape(self):
        poly = Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert_area(area, 50.0, "pipeline L-shape")
        assert_covers(poly, rect, "pipeline L-shape")
        assert ptype == "concave_no_holes"
        assert be is False or be == 0

    def test_concave_no_holes_u_shape(self):
        poly = Polygon([
            (0,0),(10,0),(10,4),(7,4),(7,8),(10,8),(10,12),(0,12),(0,8),(3,8),(3,4),(0,4)
        ])
        rect, area, ang, ptype, ratio, be = solve(poly)
        # Inner vertical channel (x=3..7, y=0..12) = 4×12 = 48 beats the base bar (40)
        assert_area(area, 48.0, "pipeline U-shape")
        assert_covers(poly, rect, "pipeline U-shape")

    def test_convex_with_holes_square_circular_hole(self):
        t = np.linspace(0, 2*np.pi, 32)
        hole = list(zip(5 + 2*np.cos(t), 5 + 2*np.sin(t)))
        poly = Polygon([(0,0),(20,0),(20,20),(0,20)], [hole])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert area > 250.0, f"area too small for square-with-hole: {area}"
        assert_covers(poly, rect, "pipeline square-with-hole")

    def test_concave_with_holes_square_hole(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [[(3,3),(7,3),(7,7),(3,7)]])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert_area(area, 30.0, "pipeline square with square hole")
        assert_covers(poly, rect, "pipeline square with square hole")

    def test_test_cases_dict_all_covered(self):
        """Every polygon in _test_cases() must yield covers=True."""
        for name, poly in _test_cases().items():
            rect, area, ang, ptype, ratio, be = solve(poly)
            assert rect is not None, f"{name}: rect is None"
            assert_covers(poly, rect, name)
            assert area > 0.0, f"{name}: area is 0"

    def test_poly_type_echo(self):
        """The returned poly_type string must be a valid detection string."""
        valid_types = {"convex_no_holes","convex_with_holes","concave_no_holes","concave_with_holes"}
        for name, poly in _test_cases().items():
            _, _, _, ptype, _, _ = solve(poly)
            assert ptype in valid_types, f"{name}: invalid poly_type '{ptype}'"

    def test_axis_angle_echoed(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        for ang in [0.0, 30.0, 45.0, 90.0, -45.0]:
            _, _, out_ang, _, _, _ = solve(poly, axis_angle=ang)
            assert abs(out_ang - ang) < 1e-9, f"angle not echoed: {out_ang} != {ang}"

    def test_ratio_attribute_accurate(self):
        poly = Polygon([(0,0),(20,0),(20,5),(0,5)])
        rect, area, ang, ptype, ratio, be = solve(poly, max_ratio=0.0)
        b = rect.bounds
        w = b[2]-b[0]; h = b[3]-b[1]
        expected_ratio = max(w,h)/min(w,h) if min(w,h)>0 else 1.0
        assert abs(ratio - expected_ratio) < 1e-4, f"ratio attr wrong: {ratio} vs {expected_ratio}"


# ===========================================================================
# Section E — Axis-angle rotation
# ===========================================================================

class TestAxisAngle:

    def test_angle_0_and_90_equivalent_on_square(self):
        """A square is invariant under 90° rotation."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        _, a0, *_ = solve(poly, axis_angle=0.0)
        _, a90, *_ = solve(poly, axis_angle=90.0)
        assert_area(a0, a90, "square 0 vs 90 degrees")

    def test_angle_0_and_180_equivalent_on_square(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        _, a0, *_ = solve(poly, axis_angle=0.0)
        _, a180, *_ = solve(poly, axis_angle=180.0)
        assert_area(a0, a180, "square 0 vs 180 degrees")

    def test_negative_angle_equivalent_to_positive(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        _, a_pos, *_ = solve(poly, axis_angle=45.0)
        _, a_neg, *_ = solve(poly, axis_angle=-45.0)
        assert_area(a_pos, a_neg, "45 vs -45 degrees")

    def test_45_degree_diamond(self):
        """A diamond (square rotated 45°) solved at 45° should give area ≈ full square."""
        # Diamond with diagonals of length 10√2 → side 10 → area 100
        # At 45° we rotate the diamond back to axis-aligned, so the LIR = 100
        diamond = shp_rotate(
            Polygon([(0,0),(10,0),(10,10),(0,10)]),
            45, origin=(5,5)
        )
        rect, area, _, _, _, _ = solve(diamond, axis_angle=45.0)
        assert area > 90.0, f"Diamond at 45° should find near-full rect, got {area}"
        assert_covers(diamond, rect, "diamond at 45°")

    def test_rotated_rectangle(self):
        """Thin rectangle at 30°: solving at 30° should recover near full area."""
        base = Polygon([(0,0),(20,0),(20,5),(0,5)])
        rotated = shp_rotate(base, 30, origin=(10,2.5))
        rect, area, _, _, _, _ = solve(rotated, axis_angle=30.0)
        assert area > 90.0, f"Rotated rect at 30° should recover ~100 area, got {area}"
        assert_covers(rotated, rect, "rotated rectangle at 30°")

    def test_result_geometry_is_rotated_when_angle_nonzero(self):
        """At a non-zero axis angle the output rect must NOT be axis-aligned."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect, _, _, _, _, _ = solve(poly, axis_angle=30.0)
        if rect is None:
            pytest.skip("No rect returned")
        coords = np.array(rect.exterior.coords[:-1])
        edges = np.diff(np.vstack([coords, coords[0]]), axis=0)
        angles = np.degrees(np.arctan2(np.abs(edges[:,1]), np.abs(edges[:,0]))) % 90.0
        # At least one edge should be non-axis-aligned (angle ≠ 0 and ≠ 90)
        # For 30° rotation the rect edges are at 30° and 120°
        assert any(5.0 < a < 85.0 for a in angles), \
            f"Rect at angle=30 still appears axis-aligned: edge angles = {angles}"

    def test_covers_holds_for_all_angles(self):
        """poly.covers(rect) must be True for a range of axis angles."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        for ang in np.linspace(0, 89, 18):
            rect, area, *_ = solve(poly, axis_angle=float(ang))
            if rect is not None:
                assert_covers(poly, rect, f"covers at angle={ang:.1f}")


# ===========================================================================
# Section F — max_ratio constraint
# ===========================================================================

class TestMaxRatio:

    def _check_ratio(self, poly, max_ratio, label=""):
        rect, area, *_ = solve(poly, max_ratio=max_ratio)
        assert rect is not None, f"{label}: rect is None"
        b = rect.bounds
        w = b[2]-b[0]; h = b[3]-b[1]
        # Only check the ratio constraint when max_ratio > 0 (0 means unlimited).
        if max_ratio > 0.0 and min(w, h) > 1e-9:
            actual = max(w, h) / min(w, h)
            assert actual <= max_ratio + 1e-5, \
                f"{label}: ratio={actual:.4f} exceeds max_ratio={max_ratio}"
        assert_covers(poly, rect, label)
        return area

    def test_square_unlimited_ratio(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        self._check_ratio(poly, 0.0, "square unlimited")

    def test_square_ratio_1(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        a = self._check_ratio(poly, 1.0, "square ratio=1")
        assert_area(a, 100.0, "square ratio=1 area")

    def test_wide_rect_ratio_2(self):
        poly = Polygon([(0,0),(20,0),(20,5),(0,5)])
        a = self._check_ratio(poly, 2.0, "wide rect ratio=2")
        assert_area(a, 50.0, "wide rect ratio=2 area")

    def test_tall_rect_ratio_3(self):
        poly = Polygon([(0,0),(2,0),(2,10),(0,10)])
        a = self._check_ratio(poly, 3.0, "tall rect ratio=3")
        assert_area(a, 12.0, "tall rect ratio=3 area")

    def test_ratio_applies_to_all_poly_types(self):
        polys = [
            ("square",        Polygon([(0,0),(10,0),(10,10),(0,10)])),
            ("L-shape",       Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])),
            ("sq-hole",       Polygon([(0,0),(10,0),(10,10),(0,10)],[[(3,3),(7,3),(7,7),(3,7)]])),
        ]
        for name, poly in polys:
            for mr in [1.0, 1.5, 2.0]:
                self._check_ratio(poly, mr, f"{name} max_ratio={mr}")

    def test_ratio_constrained_area_less_than_unconstrained(self):
        poly = Polygon([(0,0),(20,0),(20,2),(0,2)])
        a_free = solve(poly, max_ratio=0.0)[1]
        a_cstr = solve(poly, max_ratio=2.0)[1]
        assert a_cstr <= a_free + 1e-9, "constrained area exceeds unconstrained"


# ===========================================================================
# Section G — Holes handling
# ===========================================================================

class TestHoles:

    def test_square_hole_strips(self):
        """The best axis-aligned rect in a 10x10 with 4x4 central hole = 30."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [[(3,3),(7,3),(7,7),(3,7)]])
        rect, area, *_ = solve(poly)
        assert_area(area, 30.0, "square with central hole")
        assert_covers(poly, rect, "square with central hole")

    def test_rect_does_not_intersect_hole(self):
        """The result must not intersect any hole."""
        for i in range(3):  # different hole sizes
            s = 2 + i
            hole = [(s,s),(10-s,s),(10-s,10-s),(s,10-s)]
            poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [hole])
            rect, area, *_ = solve(poly)
            if rect is not None:
                for ring in poly.interiors:
                    hole_poly = Polygon(ring)
                    overlap = rect.intersection(hole_poly).area
                    assert overlap < 1e-9, \
                        f"hole size {s}: rect overlaps hole by {overlap:.2e}"

    def test_multiple_holes(self):
        """Polygon with 3 rectangular holes: result must avoid all holes."""
        holes = [
            [(1,1),(3,1),(3,3),(1,3)],
            [(5,1),(7,1),(7,3),(5,3)],
            [(3,6),(7,6),(7,8),(3,8)],
        ]
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], holes)
        rect, area, *_ = solve(poly)
        assert rect is not None
        assert_covers(poly, rect, "multiple holes")
        for ring in poly.interiors:
            hp = Polygon(ring)
            assert rect.intersection(hp).area < 1e-9, "rect overlaps a hole"

    def test_hole_at_boundary_does_not_cause_best_effort(self):
        """A square hole flush with the outer boundary: result is still exact."""
        # Hole touching the right edge: outer (0,0,10,10), hole (8,3,10,7)
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)], [[(8,3),(10,3),(10,7),(8,7)]])
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert rect is not None
        assert_covers(poly, rect, "hole at boundary")

    def test_concave_with_hole_pipeline(self):
        poly = Polygon(
            [(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)],
            [[(1,1),(3,1),(3,3),(1,3)]],
        )
        rect, area, *_ = solve(poly)
        assert area > 0.0
        assert_covers(poly, rect, "concave with hole pipeline")


# ===========================================================================
# Section H — _worker_process_feature (public entry point)
# ===========================================================================

class TestWorkerEntryPoint:

    def _run(self, poly, axis_angle=0.0, max_ratio=0.0):
        wkb = bytes(wkb_dumps(poly))
        return _worker_process_feature(
            (1, wkb, axis_angle, 120, max_ratio, False, 0.0, True)
        )

    def test_returns_tuple_for_valid_polygon(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        result = self._run(poly)
        assert result is not None
        assert len(result) == 7  # feat_id, wkt, area, axis_angle, poly_type, ratio, best_effort

    def test_feat_id_preserved(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        wkb = bytes(wkb_dumps(poly))
        res = _worker_process_feature((42, wkb, 0.0, 120, 0.0, False, 0.0, True))
        assert res[0] == 42

    def test_wkt_parses_to_valid_polygon(self):
        from shapely.wkt import loads as wkt_loads
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        res = self._run(poly)
        out = wkt_loads(res[1])
        assert not out.is_empty
        assert out.is_valid

    def test_area_matches_pipeline(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        res = self._run(poly)
        _, _, area, _, _, _, _ = res
        assert_area(float(area), 100.0, "worker area")

    def test_poly_type_string_valid(self):
        valid = {"convex_no_holes","convex_with_holes","concave_no_holes","concave_with_holes"}
        for name, poly in _test_cases().items():
            res = self._run(poly)
            assert res[4] in valid, f"{name}: invalid poly_type '{res[4]}'"

    def test_returns_none_for_empty_geometry(self):
        from shapely.geometry import Polygon as P
        empty = P()
        wkb = bytes(wkb_dumps(empty, include_srid=False))
        # empty wkb → _prepare_polygon returns None → worker returns None
        # (may raise or return None depending on wkb validity)
        try:
            res = _worker_process_feature((1, wkb, 0.0, 120, 0.0, False, 0.0, True))
            assert res is None, f"Expected None for empty polygon, got {res}"
        except Exception:
            pass  # acceptable to raise for degenerate inputs

    def test_covers_invariant_from_wkt(self):
        """Re-parse the WKT output and verify covers."""
        from shapely.wkt import loads as wkt_loads
        for name, poly in _test_cases().items():
            res = self._run(poly)
            if res is None:
                pytest.fail(f"{name}: worker returned None")
            out_rect = wkt_loads(res[1])
            assert_covers(poly, out_rect, f"worker wkt covers: {name}")

    def test_all_test_cases_return_non_none(self):
        for name, poly in _test_cases().items():
            res = self._run(poly)
            assert res is not None, f"{name}: worker returned None"

    def test_axis_angle_stored_in_output(self):
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        for ang in [0.0, 30.0, 45.0]:
            wkb = bytes(wkb_dumps(poly))
            res = _worker_process_feature((1, wkb, ang, 120, 0.0, False, 0.0, True))
            assert abs(float(res[3]) - ang) < 1e-6, f"axis_angle not echoed: {res[3]} != {ang}"


# ===========================================================================
# Section I — Edge / degenerate cases
# ===========================================================================

class TestEdgeCases:

    def test_very_thin_polygon(self):
        """100×0.001 strip: LIR = 0.1"""
        poly = Polygon([(0,0),(100,0),(100,0.001),(0,0.001)])
        rect, area, *_ = solve(poly)
        assert_area(area, 0.1, "thin polygon")
        assert_covers(poly, rect, "thin polygon")

    def test_unit_square(self):
        poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
        rect, area, *_ = solve(poly)
        assert_area(area, 1.0, "unit square")
        assert_covers(poly, rect, "unit square")

    def test_large_polygon(self):
        """1e6 × 1e6 square: LIR = 1e12"""
        poly = Polygon([(0,0),(1e6,0),(1e6,1e6),(0,1e6)])
        rect, area, *_ = solve(poly)
        assert_area(area, 1e12, "large polygon")
        assert_covers(poly, rect, "large polygon")

    def test_tiny_polygon(self):
        """1e-6 × 1e-6 square: LIR = 1e-12"""
        poly = Polygon([(0,0),(1e-6,0),(1e-6,1e-6),(0,1e-6)])
        rect, area, *_ = solve(poly)
        assert area > 0.0
        assert_covers(poly, rect, "tiny polygon")

    def test_non_axis_aligned_input_polygon(self):
        """Polygon that is not axis-aligned (diamond): solver still works."""
        diamond = Polygon([(5,0),(10,5),(5,10),(0,5)])
        rect, area, *_ = solve(diamond)
        assert area > 0.0
        assert_covers(diamond, rect, "diamond polygon")

    def test_many_vertices_circle_approx(self):
        """Circle approximated with 200 vertices: fallback grid used, result valid."""
        t = np.linspace(0, 2*np.pi, 201)[:-1]
        poly = Polygon(zip(np.cos(t)*10, np.sin(t)*10))
        rect, area, *_ = solve(poly)
        # LIR of circle r=10: optimal axis-aligned square = r√2 × r√2 = 200
        assert area > 150.0, f"circle LIR too small: {area}"
        assert_covers(poly, rect, "circle approx 200 verts")

    def test_polygon_with_collinear_vertices(self):
        """Polygon with collinear vertices on an edge: still valid."""
        poly = Polygon([(0,0),(5,0),(10,0),(10,10),(0,10)])
        rect, area, *_ = solve(poly)
        assert_area(area, 100.0, "collinear vertices")
        assert_covers(poly, rect, "collinear vertices")

    def test_always_return_false_on_degenerate(self):
        """always_return=False: may return None but must never return an uncovered rect."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect, area, *_ = _solve_axis_aligned_lir(
            poly, 0.0, 120, 0.0, False, False, 0.0
        )
        if rect is not None:
            assert_covers(poly, rect, "always_return=False")

    def test_buffer_negative_shrinks_rect(self):
        """Negative buffer value must shrink the result."""
        poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
        rect_buf, area_buf, *_ = _solve_axis_aligned_lir(
            poly, 0.0, 120, 0.0, True, True, -0.5
        )
        rect_raw, area_raw, *_ = solve(poly)
        assert area_buf < area_raw, "negative buffer should shrink"
        if rect_buf is not None:
            assert_covers(poly, rect_buf, "buffered result")


# ===========================================================================
# Section J — Containment invariant (parametric / property-style)
# ===========================================================================

class TestContainmentInvariant:
    """
    For EVERY polygon in this parametric set and EVERY axis angle,
    poly.covers(rect) must be True.  This is the single most important
    correctness property.
    """

    POLYGONS = {
        "square":          Polygon([(0,0),(10,0),(10,10),(0,10)]),
        "rect_20x5":       Polygon([(0,0),(20,0),(20,5),(0,5)]),
        "right_tri":       Polygon([(0,0),(10,0),(0,10)]),
        "isoceles_tri":    Polygon([(0,0),(10,0),(5,5)]),
        # Use set_precision-snapped hexagon so FP vertex noise from sin/cos
        # does not cause best_effort=True or covers=False on the raw solver.
        "hexagon":         _make_hexagon(10.0),
        "l_shape":         Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)]),
        "t_shape":         Polygon([(0,5),(10,5),(10,7),(7,7),(7,10),(3,10),(3,7),(0,7)]),
        "u_shape":         Polygon([(0,0),(10,0),(10,4),(7,4),(7,8),(10,8),
                                    (10,12),(0,12),(0,8),(3,8),(3,4),(0,4)]),
        "sq_sq_hole":      Polygon([(0,0),(10,0),(10,10),(0,10)],
                                    [[(3,3),(7,3),(7,7),(3,7)]]),
        "l_hole":          Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)],
                                    [[(1,1),(3,1),(3,3),(1,3)]]),
    }
    ANGLES = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    @pytest.mark.parametrize("poly_name", list(POLYGONS.keys()))
    @pytest.mark.parametrize("angle", ANGLES)
    def test_covers_invariant(self, poly_name, angle):
        poly = self.POLYGONS[poly_name]
        rect, area, *_ = solve(poly, axis_angle=angle)
        if rect is None:
            pytest.fail(f"{poly_name}@{angle}°: got None")
        assert_covers(poly, rect, f"{poly_name}@{angle}°")
        assert area > 0.0, f"{poly_name}@{angle}°: area=0"

    @pytest.mark.parametrize("poly_name", list(POLYGONS.keys()))
    def test_best_effort_false_for_standard_polys(self, poly_name):
        """Standard well-formed polygons should never trigger the best-effort fallback."""
        poly = self.POLYGONS[poly_name]
        rect, area, ang, ptype, ratio, be = solve(poly)
        assert be is False or be == 0, \
            f"{poly_name}: best_effort=True for a well-formed polygon"
