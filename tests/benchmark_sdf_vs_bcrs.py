"""
benchmark_sdf_vs_bcrs.py — Quality and speed comparison.

Runs BCRS (via axis_aligned_lir_worker) and SDF oracle on a curated set of
polygons and reports area improvement, timing, and coverage.

Usage
-----
  python tests/benchmark_sdf_vs_bcrs.py

Output columns
--------------
  Polygon        : short name
  BCRS area      : area from the standard BCRS+CABF pipeline
  SDF area       : area from SDF post-polish
  Δ%             : percentage improvement (positive = SDF better)
  BCRS ms        : wall-clock time for BCRS solve
  SDF ms         : additional time for SDF post-polish
  covered        : True if poly.covers(SDF result)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon, box

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "LIRiAP_pack"))

from axis_aligned_lir_worker import _solve_axis_aligned_lir
from sdf_oracle import sdf_polish, _solve_5d, rect_corners

# ── Helpers ──────────────────────────────────────────────────────────────────

def circle_poly(r=10.0, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(list(zip(r * np.cos(t), r * np.sin(t))))

def ellipse_poly(a=15.0, b=5.0, n=128):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(list(zip(a * np.cos(t), b * np.sin(t))))

def make_bcrs_seed(poly, axis_angle=0.0, grid_fine=120):
    """Run BCRS+CABF and return (rect, area)."""
    t0 = time.monotonic()
    rect, area, angle, *_ = _solve_axis_aligned_lir(
        poly,
        axis_angle=axis_angle,
        grid_fine=grid_fine,
        max_ratio=0.0,
        always_return=True,
        buf_enabled=False,
        buf_value=0.0,
    )
    elapsed = (time.monotonic() - t0) * 1e3
    return rect, area, angle, elapsed

def run_sdf(poly, rect, area, angle):
    """Run SDF post-polish from a BCRS seed."""
    if rect is None or rect.is_empty:
        return None, 0.0, angle, 0.0
    cands = [{'rect': rect, 'area': area, 'angle': angle}]
    t0 = time.monotonic()
    r, a, ang = sdf_polish(poly, cands, timeout_ms=250)
    elapsed = (time.monotonic() - t0) * 1e3
    return r, a, ang, elapsed


# ── Test polygon suite ────────────────────────────────────────────────────────

CASES: List[Tuple[str, Polygon, float, float]] = [
    # (name, polygon, axis_angle, known_optimal_area_or_0)
    ("Square 10×10",          box(0,0,10,10),                0.0, 100.0),
    ("Rectangle 20×5",        box(0,0,20,5),                 0.0, 100.0),
    ("Circle r=10 (64v)",     circle_poly(r=10,n=64),        0.0, 200.0),
    ("Circle r=10 (500v)",    circle_poly(r=10,n=500),       0.0, 200.0),
    ("Ellipse 15×5",          ellipse_poly(15,5,128),        0.0, 150.0),
    ("L-shape",               Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)]),
                                                              0.0, 50.0),
    ("U-shape",               Polygon([(0,0),(10,0),(10,4),(7,4),(7,8),
                                        (10,8),(10,12),(0,12),(0,8),(3,8),
                                        (3,4),(0,4)]),        0.0, 0.0),
    ("Hexagon r=10",          Polygon(list(zip(
                                  10*np.cos(np.linspace(0,2*np.pi,7)[:-1]),
                                  10*np.sin(np.linspace(0,2*np.pi,7)[:-1])))),
                                                              0.0, 173.205),
    ("Tilted rect 37°",       shp_rotate(box(0,0,20,5), 37, origin=(10,2.5)),
                                                              37.0, 100.0),
    ("Tilted rect 22.5°",     shp_rotate(box(0,0,15,6), 22.5, origin=(7.5,3)),
                                                              22.5, 90.0),
    ("Pentagon",              Polygon([(-5,-3),(7,-3),(9,4),(0,8),(-6,3)]),
                                                              0.0, 0.0),
    ("Sq w/ hole",            Polygon([(0,0),(20,0),(20,20),(0,20)],
                                       [[(7,7),(13,7),(13,13),(7,13)]]),
                                                              0.0, 0.0),
    ("Stadium",               Point(0,0).buffer(5, resolution=32).union(
                                  Point(10,0).buffer(5, resolution=32)).union(
                                  box(-2,-3,12,3)),            0.0, 0.0),
]


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    print(f"\n{'Polygon':<26} {'BCRS area':>12} {'SDF area':>12} "
          f"{'Δ%':>8} {'BCRS ms':>9} {'SDF ms':>8} {'✓':>4}")
    print("─" * 82)

    total_bcrs = 0.0;  total_sdf = 0.0;  n_improved = 0;  n_cases = 0

    for name, poly, axis_angle, known_opt in CASES:
        try:
            # BCRS solve
            bcrs_rect, bcrs_area, bcrs_angle, bcrs_ms = make_bcrs_seed(
                poly, axis_angle=axis_angle)

            if bcrs_rect is None:
                print(f"  {name:<24}  BCRS returned None")
                continue

            # SDF post-polish
            sdf_rect, sdf_area, sdf_angle, sdf_ms = run_sdf(
                poly, bcrs_rect, bcrs_area, bcrs_angle)

            best_area = max(bcrs_area, sdf_area)
            delta_pct = ((best_area - bcrs_area) / bcrs_area * 100.0
                         if bcrs_area > 0 else 0.0)
            covered   = poly.covers(sdf_rect) if sdf_rect else True
            ok        = "✓" if covered else "✗ FAIL"

            if delta_pct > 0.001:
                n_improved += 1

            total_bcrs += bcrs_area
            total_sdf  += best_area
            n_cases    += 1

            print(f"  {name:<24} {bcrs_area:>12.4f} {best_area:>12.4f} "
                  f"{delta_pct:>+7.3f}% {bcrs_ms:>8.0f}ms "
                  f"{sdf_ms:>7.0f}ms {ok:>4}")

            if known_opt > 0:
                ratio = best_area / known_opt
                if ratio < 0.98:
                    print(f"    ↳ NOTE: below known optimal "
                          f"{known_opt:.2f} (ratio={ratio:.4f})")

        except Exception as exc:
            print(f"  {name:<24}  ERROR: {exc}")

    if n_cases > 0:
        overall_gain = (total_sdf - total_bcrs) / total_bcrs * 100.0
        print("─" * 82)
        print(f"  {'TOTAL':<24} {total_bcrs:>12.4f} {total_sdf:>12.4f} "
              f"{overall_gain:>+7.3f}%")
        print(f"\n  Cases improved by SDF: {n_improved}/{n_cases}")
        print(f"  Overall area gain:     {overall_gain:+.4f}%")


if __name__ == "__main__":
    main()