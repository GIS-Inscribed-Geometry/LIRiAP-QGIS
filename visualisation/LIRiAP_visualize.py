#!/usr/bin/env python3
"""LIRiAP Visual Preview Tool

Generates HTML files to visualize polygons and their largest inscribed rectangles.
Output goes to `output/liriap_output/` directory.

Run with: python LIRiAP_visualize.py [--algorithm NAME]

Available algorithms (from LIRiAP_pack):
  axis_aligned      - axis_aligned_lir_worker
  contained         - contained_standard_worker  
  contained_fast   - contained_fast_worker
  bcrs             - bcrs_worker
  bcrs_fast         - bcrs_fast_worker
  approximation    - approximation_standard_worker
  approximation_fast - approximation_fast_worker

Run with: python LIRiAP_visualize.py --algorithm bcrs_fast
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.wkb import dumps as _shp_wkb_dumps
from shapely.wkt import loads as _shp_wkt_loads

from LIRiAP_pack.axis_aligned_lir_worker import _worker_process_feature as aa_worker
from LIRiAP_pack.contained_standard_worker import _worker_process_feature as cont_worker
from LIRiAP_pack.contained_fast_worker import _worker_process_feature as cont_fast_worker
from LIRiAP_pack.bcrs_worker import _worker_process_feature as bcrs_worker
from LIRiAP_pack.bcrs_fast_worker import _worker_process_feature as bcrs_fast_worker
from LIRiAP_pack.approximation_standard_worker import _worker_process_feature as approx_worker
from LIRiAP_pack.approximation_fast_worker import _worker_process_feature as approx_fast_worker


def parse_polygon(geom: dict) -> ShapelyPolygon:
    """Parse a GeoJSON polygon geometry."""
    coords = geom["coordinates"]
    exterior = coords[0]
    if len(coords) > 1:
        holes = coords[1:]
        return ShapelyPolygon(exterior, holes)
    return ShapelyPolygon(exterior)


def load_polygons() -> list[tuple[int, ShapelyPolygon]]:
    """Load polygons from realworld.geojson."""
    geojson_path = Path(__file__).parent.parent / "tests" / "real_world_data" / "realworld.geojson"
    with open(geojson_path, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data["features"]:
        fid = feature["properties"]["fid"]
        geom = feature["geometry"]
        poly = parse_polygon(geom)
        polygons.append((fid, poly))

    return sorted(polygons, key=lambda x: x[0])


def make_l_shape(cx: float, cy: float, size: float) -> ShapelyPolygon:
    """Create an L-shaped polygon."""
    return ShapelyPolygon([
        (cx - size, cy - size),
        (cx + size, cy - size),
        (cx + size, cy - size * 0.3),
        (cx + size * 0.3, cy - size * 0.3),
        (cx + size * 0.3, cy + size),
        (cx - size, cy + size),
        (cx - size, cy - size),
    ])


def make_u_shape(cx: float, cy: float, size: float) -> ShapelyPolygon:
    """Create a U-shaped polygon."""
    return ShapelyPolygon([
        (cx - size, cy - size),
        (cx + size, cy - size),
        (cx + size, cy + size),
        (cx + size * 0.4, cy + size),
        (cx + size * 0.4, cy),
        (cx - size * 0.4, cy),
        (cx - size * 0.4, cy + size),
        (cx - size, cy + size),
        (cx - size, cy - size),
    ])


def make_zigzag(cx: float, cy: float, size: float) -> ShapelyPolygon:
    """Create a zigzag polygon."""
    return ShapelyPolygon([
        (cx - size, cy - size),
        (cx - size * 0.6, cy - size),
        (cx - size * 0.2, cy),
        (cx + size * 0.2, cy),
        (cx + size * 0.6, cy - size),
        (cx + size, cy - size),
        (cx + size, cy + size),
        (cx + size * 0.6, cy + size),
        (cx + size * 0.2, cy),
        (cx - size * 0.2, cy),
        (cx - size * 0.6, cy + size),
        (cx - size, cy + size),
        (cx - size, cy - size),
    ])


def get_bounds(poly: ShapelyPolygon) -> tuple[float, float, float, float]:
    """Get polygon bounding box."""
    minx, miny, maxx, maxy = poly.bounds
    return minx, miny, maxx, maxy


def generate_svg_for_polygon(
    poly_id: str,
    poly: ShapelyPolygon,
    rect: ShapelyPolygon | None,
    time_ms: float,
) -> str:
    """Generate SVG HTML for a single polygon."""
    min_x, min_y, max_x, max_y = get_bounds(poly)
    poly_area = poly.area
    rect_area = rect.area if rect is not None else 0.0

    size = 200.0
    padding = 10.0
    draw_size = size - 2.0 * padding

    width = max_x - min_x
    height = max_y - min_y
    scale = draw_size / max(width, height) if width > 0 and height > 0 else draw_size

    offset_x = padding + (draw_size - width * scale) / 2.0
    offset_y = padding + (draw_size - height * scale) / 2.0

    def to_svg(x: float, y: float) -> tuple[float, float]:
        return (offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale)

    exterior_points = " ".join(
        f"{to_svg(c[0], c[1])[0]:.1f},{to_svg(c[0], c[1])[1]:.1f}"
        for c in poly.exterior.coords[:-1]
    )

    holes_svg = ""
    for hole in poly.interiors:
        points = " ".join(
            f"{to_svg(c[0], c[1])[0]:.1f},{to_svg(c[0], c[1])[1]:.1f}"
            for c in hole.coords[:-1]
        )
        holes_svg += f'<polygon class="hole" points="{points}"/>'

    rect_svg = ""
    if rect is not None:
        # Use polygon to support rotated rectangles
        rect_points = " ".join(
            f"{to_svg(c[0], c[1])[0]:.1f},{to_svg(c[0], c[1])[1]:.1f}"
            for c in rect.exterior.coords[:-1]
        )
        rect_svg = f'<polygon class="rect" points="{rect_points}"/>'

    fill_ratio = (rect_area / poly_area * 100.0) if poly_area > 0 else 0.0

    return f'''<div class="card">
        <svg viewBox="0 0 {size:.0f} {size:.0f}">
            <polygon class="polygon" points="{exterior_points}"/>
            {holes_svg}
            {rect_svg}
        </svg>
        <div class="info">
            <strong>{poly_id}</strong><br/>
            Polygon: {poly_area:.1f}<br/>
            Rectangle: {rect_area:.1f}<br/>
            Fill: {fill_ratio:.1f}%<br/>
            Time: {time_ms:.2f}ms
        </div>
    </div>'''


def solve_with_algorithm(poly: ShapelyPolygon, algorithm: str, max_ratio: float = 0.0):
    """Solve LIR with the specified worker - matching QGIS default parameters."""
    try:
        wkb = _shp_wkb_dumps(poly)
        fid = 0
        
        # QGIS default parameters:
        # angle_step=5, grid_coarse=40, grid_fine=120, max_ratio=1.6, top_k=3
        if algorithm == "axis_aligned":
            result = aa_worker((fid, wkb, 0.0, 120, max_ratio, False, 0.0, False))
        elif algorithm == "contained":
            result = cont_worker((fid, wkb, 5, 40, 120, max_ratio, False, 0.0, 3, False))
        elif algorithm == "contained_fast":
            result = cont_fast_worker((fid, wkb, 5, 40, 120, max_ratio, False, 0.0, 3, False))
        elif algorithm == "bcrs":
            result = bcrs_worker((fid, wkb, 5, 40, 120, max_ratio, False, 0.0, 3, False))
        elif algorithm == "bcrs_fast":
            result = bcrs_fast_worker((fid, wkb, 5, 40, 120, max_ratio, False, 0.0, 3, False))
        elif algorithm == "approximation":
            result = approx_worker((fid, wkb, 5, 40, 100, max_ratio, False, 0.0))
        elif algorithm == "approximation_fast":
            result = approx_fast_worker((fid, wkb, 5, 40, 100, max_ratio, False, 0.0))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if result is None:
            return (None, 0.0, 0.0, "failed", 1.0, False)
        
        # Handle different return signatures
        if algorithm == "axis_aligned":
            _, wkt, area, angle, poly_type, ratio, best_effort = result
            return (_shp_wkt_loads(wkt), area, angle, poly_type, ratio, best_effort)
        elif algorithm in ("bcrs", "bcrs_fast"):
            _, wkt, area, angle, ratio, cand_rank, stage2_gain, best_effort = result
            return (_shp_wkt_loads(wkt), area, angle, f"rank={cand_rank}", ratio, best_effort)
        elif algorithm in ("contained", "contained_fast"):
            _, wkt, area, angle, ratio, cand_rank, stage2_gain, best_effort = result
            return (_shp_wkt_loads(wkt), area, angle, result[5], ratio, best_effort)
        elif algorithm in ("approximation", "approximation_fast"):
            _, wkt, area, angle, ratio = result
            return (_shp_wkt_loads(wkt), area, angle, algorithm, ratio, False)
    except Exception as e:
        print(f"Error in {algorithm}: {e}")
        return (None, 0.0, 0.0, f"error: {e}", 1.0, False)


def generate_preview_html(output_dir: Path, algorithm: str = "axis_aligned", max_polygons: int | None = None) -> None:
    """Generate the full HTML preview."""
    all_polygons: list[tuple[str, ShapelyPolygon]] = []

    # Add synthetic polygons
    all_polygons.append(("Square 10x10", ShapelyPolygon([(0, 0), (10, 0), (10, 10), (0, 10)])))
    all_polygons.append(("Rectangle 10x1", ShapelyPolygon([(0, 0), (10, 0), (10, 1), (0, 1)])))
    all_polygons.append(("Triangle", ShapelyPolygon([(0, 0), (10, 0), (5, 10)])))
    all_polygons.append(("L-Shape", make_l_shape(5.0, 5.0, 5.0)))
    all_polygons.append(("U-Shape", make_u_shape(5.0, 5.0, 5.0)))
    all_polygons.append(("Zigzag", make_zigzag(5.0, 5.0, 5.0)))

    # Load real polygons
    real_polygons = load_polygons()
    for fid, poly in real_polygons:
        vertex_count = len(poly.exterior.coords) - 1
        all_polygons.append((f"Real #{fid} ({vertex_count}v)", poly))
        if max_polygons is not None and len(all_polygons) >= max_polygons:
            break

    output_dir = output_dir / "liriap_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_names = {
        "axis_aligned": "Axis-Aligned LIR",
        "contained": "Contained Standard",
        "contained_fast": "Contained Fast", 
        "bcrs": "BCRS",
        "bcrs_fast": "BCRS Fast",
        "approximation": "Approximation",
        "approximation_fast": "Approximation Fast",
    }

    choices = ["axis_aligned", "contained", "contained_fast", "bcrs", "bcrs_fast", "approximation", "approximation_fast"]

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LIRiAP Visual Preview - {algorithm}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #eee; margin-bottom: 10px; }}
        h2 {{ color: #aaa; font-size: 14px; font-weight: normal; margin-top: -5px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        svg {{ width: 100%; height: 200px; background: #0f0f23; border-radius: 4px; }}
        .polygon {{ fill: #e94560; stroke: #ff6b6b; stroke-width: 1; }}
        .rect {{ fill: rgba(66, 133, 244, 0.4); stroke: #4285f4; stroke-width: 2; }}
        .hole {{ fill: none; stroke: #666; stroke-width: 1; stroke-dasharray: 3; }}
        .info {{ margin-top: 8px; font-size: 11px; color: #aaa; line-height: 1.4; }}
        .stats {{ background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .stats p {{ margin: 5px 0; color: #ccc; }}
        .stats strong {{ color: #fff; }}
    </style>
</head>
<body>
    <h1>LIRiAP - Largest Inscribed Rectangle Preview</h1>
    <h2>Algorithm: {algorithm_names.get(algorithm, algorithm)}</h2>
    <div class="stats">
'''

    success_count = 0
    failed_count = 0
    total_rect_area = 0.0
    total_poly_area = 0.0
    total_time_ms = 0.0
    cards_html = ""

    for poly_id, poly in all_polygons:
        poly_area = poly.area
        total_poly_area += poly_area

        start = time.perf_counter()
        try:
            result = solve_with_algorithm(poly, algorithm)
            rect = result[0] if result[0] is not None else None
        except Exception as e:
            rect = None
        elapsed = (time.perf_counter() - start) * 1000.0
        total_time_ms += elapsed

        if rect is not None:
            success_count += 1
            total_rect_area += rect.area
        else:
            failed_count += 1
            rect = None

        cards_html += generate_svg_for_polygon(poly_id, poly, rect, elapsed)

    fill_ratio = (total_rect_area / total_poly_area * 100.0) if total_poly_area > 0 else 0.0
    avg_time = total_time_ms / len(all_polygons) if all_polygons else 0

    html += f'''
        <p><strong>Total shapes:</strong> {len(all_polygons)}</p>
        <p><strong>Successfully processed:</strong> {success_count} ({success_count / len(all_polygons) * 100:.1f}%)</p>
        <p><strong>Failed:</strong> {failed_count}</p>
        <p><strong>Total polygon area:</strong> {total_poly_area:.0f}</p>
        <p><strong>Total inscribed area:</strong> {total_rect_area:.0f} ({fill_ratio:.1f}%)</p>
        <p><strong>Total processing time:</strong> {total_time_ms:.1f}ms ({avg_time:.2f}ms avg per shape)</p>
    </div>
    <div class="grid">
'''

    html += cards_html

    html += '''
    </div>
</body>
</html>
'''

    output_path = output_dir / "index.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated preview: {output_path}")
    print(f"  Total shapes: {len(all_polygons)}")
    print(f"  Success: {success_count}, Failed: {failed_count}")
    print(f"  Total time: {total_time_ms:.1f}ms ({avg_time:.2f}ms avg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIRiAP Visual Preview Tool")
    parser.add_argument(
        "--algorithm", "-a",
        choices=["axis_aligned", "contained", "contained_fast", "bcrs", "bcrs_fast", "approximation", "approximation_fast"],
        default="axis_aligned",
        help="Algorithm to use (default: axis_aligned)",
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=None,
        help="Maximum number of polygons to process",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: output/liriap_output)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(__file__).parent / "output"
    generate_preview_html(output_dir, algorithm=args.algorithm, max_polygons=args.max)