#!/usr/bin/env python3
"""LIRiAP Visualizer - Shows actual algorithm process step by step."""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from LIRiAP_pack.axis_aligned_lir_worker import (
    _detect_polygon_type,
    _build_row_mask_scanline,
    _histogram_kernel_vp,
)


def load_polygons(path: str, n: int) -> list:
    with open(path) as f:
        data = json.load(f)
    
    polys = []
    for feat in data.get('features', []):
        geom = feat.get('geometry', {})
        if geom.get('type') != 'Polygon':
            continue
        coords = geom.get('coordinates', [])
        if not coords:
            continue
        try:
            poly = ShapelyPolygon(coords[0], coords[1:] if len(coords) > 1 else None)
            if not poly.is_valid or poly.is_empty:
                continue
        except:
            continue
        props = feat.get('properties', {})
        name = props.get('NAZWA') or props.get('fid') or str(len(polys))
        polys.append({'id': len(polys), 'name': str(name)[:50], 'poly': poly, 'area': float(poly.area)})
        if len(polys) >= n:
            break
    return polys


def build_grid_coords(poly: ShapelyPolygon) -> Tuple[np.ndarray, np.ndarray]:
    """Build vertex coordinate arrays."""
    minx, miny, maxx, maxy = poly.bounds
    all_xs = [c[0] for c in poly.exterior.coords[:-1]]
    all_ys = [c[1] for c in poly.exterior.coords[:-1]]
    if poly.interiors:
        for ring in poly.interiors:
            all_xs.extend([c[0] for c in ring.coords[:-1]])
            all_ys.extend([c[1] for c in ring.coords[:-1]])
    all_xs.extend([minx, maxx])
    all_ys.extend([miny, maxy])
    xs_raw = np.unique(np.array(all_xs, dtype=np.float64))
    ys_raw = np.unique(np.array(all_ys, dtype=np.float64))
    span = float(max(xs_raw[-1] - xs_raw[0], ys_raw[-1] - ys_raw[0])) if len(xs_raw) >= 2 else 1.0
    tol = span * 1e-9
    keep = np.empty(len(xs_raw), dtype=bool)
    keep[0] = True
    keep[1:] = np.diff(xs_raw) > tol
    xs_raw = xs_raw[keep]
    keep = np.empty(len(ys_raw), dtype=bool)
    keep[0] = True
    keep[1:] = np.diff(ys_raw) > tol
    ys_raw = ys_raw[keep]
    xs_v = np.empty(2 * len(xs_raw) - 1, dtype=np.float64)
    xs_v[0::2] = xs_raw
    xs_v[1::2] = 0.5 * (xs_raw[:-1] + xs_raw[1:])
    ys_v = np.empty(2 * len(ys_raw) - 1, dtype=np.float64)
    ys_v[0::2] = ys_raw
    ys_v[1::2] = 0.5 * (ys_raw[:-1] + ys_raw[1:])
    return xs_v, ys_v


def solve_with_steps(poly: ShapelyPolygon) -> dict:
    """Run solver and capture each step."""
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    scale = 200 / max(w, h)
    pad = 30
    
    def to_svg(x, y):
        return int(pad + (x - minx) * scale), int(250 - (y - miny) * scale)
    
    def coords_to_svg(coords):
        return " ".join(str(x) + "," + str(y) for x, y in [to_svg(c[0], c[1]) for c in coords[:-1]])
    
    ptype = _detect_polygon_type(poly)
    xs_v, ys_v = build_grid_coords(poly)
    n_rows, n_cols = len(ys_v) - 1, len(xs_v) - 1
    
    # Build mask
    mask = _build_row_mask_scanline(poly, xs_v, ys_v)
    
    frames = []
    
    # Frame 0: Input polygon
    poly_svg = '<polygon points="' + coords_to_svg(poly.exterior.coords) + '" class="poly"/>'
    if poly.interiors:
        for ring in poly.interiors:
            poly_svg += '<polygon points="' + coords_to_svg(ring.coords) + '" class="hole"/>'
    frames.append({'title': 'Input Polygon', 'desc': ptype + ', area=' + str(int(poly.area)), 'svg': poly_svg})
    
    # Frame 1: Raw vertex grid
    grid_svg = poly_svg
    for x in xs_v:
        px, _ = to_svg(x, miny)
        grid_svg += '<line x1="' + str(px) + '" y1="30" x2="' + str(px) + '" y2="230" class="grid"/>'
    for y in ys_v:
        _, py = to_svg(minx, y)
        grid_svg += '<line x1="30" y1="' + str(py) + '" x2="230" y2="' + str(py) + '" class="grid"/>'
    frames.append({'title': 'Vertex Grid', 'desc': str(n_cols) + ' x ' + str(n_rows) + ' = ' + str(n_cols*n_cols) + ' cells', 'svg': grid_svg})
    
    # Frame 2-N: Show each row being processed
    current_cells = []
    for row_idx in range(n_rows):
        row_svg = poly_svg + grid_svg
        for j in range(n_cols):
            if mask[row_idx, j]:
                cx0, cy0 = xs_v[j], ys_v[row_idx]
                cx1, cy1 = xs_v[j+1], ys_v[row_idx+1]
                x0, y0 = to_svg(cx0, cy0)
                x1, y1 = to_svg(cx1, cy1)
                row_svg += '<rect x="' + str(min(x0, x1)) + '" y="' + str(min(y0, y1)) + '" width="' + str(abs(x1-x0)) + '" height="' + str(abs(y1-y0)) + '" class="valid"/>'
        desc_y = int(ys_v[row_idx])
        frames.append({'title': 'Scan Row ' + str(row_idx+1) + '/' + str(n_rows), 'desc': 'Check row at y=' + str(desc_y), 'svg': row_svg})
    
    # Skip histogram for now - just show rows accumulating
    # Frame: Show all valid cells highlighted
    all_valid_svg = poly_svg + grid_svg
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                cx0, cy0 = xs_v[j], ys_v[i]
                cx1, cy1 = xs_v[j+1], ys_v[i+1]
                x0, y0 = to_svg(cx0, cy0)
                x1, y1 = to_svg(cx1, cy1)
                all_valid_svg += '<rect x="' + str(min(x0, x1)) + '" y="' + str(min(y0, y1)) + '" width="' + str(abs(x1-x0)) + '" height="' + str(abs(y1-y0)) + '" class="valid"/>'
    frames.append({'title': 'All Valid Cells', 'desc': str(int(mask.sum())) + ' cells inside polygon', 'svg': all_valid_svg})
    
    # Final result - use actual solver
    from LIRiAP_pack.axis_aligned_lir_worker import _exact_solve_vertex_grid
    rect, best_area = _exact_solve_vertex_grid(poly, ptype, 0.0)
    
    # Final frame
    final_svg = poly_svg + grid_svg
    if rect:
        x0, y0, x1, y1 = rect.bounds
        sx0, sy0 = to_svg(x0, y0)
        sx1, sy1 = to_svg(x1, y1)
        final_svg += '<rect x="' + str(min(sx0, sx1)) + '" y="' + str(min(sy0, sy1)) + '" width="' + str(abs(sx1-sx0)) + '" height="' + str(abs(sy1-sy0)) + '" class="result"/>'
    pct = best_area / poly.area * 100 if poly.area > 0 and best_area else 0
    frames.append({'title': 'Final Result', 'desc': 'area=' + str(int(best_area)) + ' (' + str(int(pct)) + '%)', 'svg': final_svg, 'final': True})
    
    return {'name': '', 'area': poly.area, 'type': ptype, 'frames': frames}


def main():
    geojson_path = Path(__file__).parent.parent / 'tests' / 'real_world_data' / 'realworld.geojson'
    if not geojson_path.exists():
        print("Not found:", geojson_path)
        return
    
    print("Loading polygons...")
    polys = load_polygons(str(geojson_path), 30)  # Limited for process
    print("Loaded:", len(polys))
    
    print("Running solver with steps...")
    all_data = {}
    for i, p in enumerate(polys):
        if i % 10 == 0:
            print("  " + str(i) + "/" + str(len(polys)))
        try:
            frames = solve_with_steps(p['poly'])
            frames['name'] = p['name']
            frames['area'] = p['area']
            all_data[p['id']] = frames
        except Exception as e:
            print("  Error:", p['name'][:25], str(e))
    
    features_json = json.dumps([{'id': k, 'name': v['name'], 'area': v['area'], 'type': v['type']} for k, v in all_data.items()])
    frames_json = json.dumps(all_data)
    
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LIRiAP Algorithm Visualization</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1419; color: #e7e9ea; padding: 24px; min-height: 100vh; }
        h1 { color: #1d9bf0; font-size: 28px; margin-bottom: 8px; font-weight: 700; }
        .subtitle { color: #71767b; margin-bottom: 20px; font-size: 15px; }
        .controls { background: #16181c; border: 1px solid #2f3337; border-radius: 16px; padding: 20px; margin-bottom: 20px; }
        .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        select, button, input { background: #000; color: #e7e9ea; border: 1px solid #2f3337; padding: 12px 16px; border-radius: 8px; font-size: 14px; }
        select { min-width: 340px; }
        button { background: #1d9bf0; color: #000; font-weight: 600; cursor: pointer; border: none; }
        button:hover { background: #1a8cd8; }
        button:disabled { background: #2f3337; color: #71767b; cursor: not-allowed; }
        .viewer { background: #16181c; border: 1px solid #2f3337; border-radius: 16px; padding: 24px; }
        .viewer.hidden { display: none; }
        .frame-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2f3337; }
        .frame-num { color: #1d9bf0; font-size: 14px; font-weight: 600; }
        .frame-title { color: #fff; font-size: 18px; font-weight: 600; }
        .frame-desc { color: #71767b; font-size: 14px; margin-bottom: 16px; }
        svg { display: block; background: #000; margin: 0 auto; border-radius: 8px; }
        .poly { fill: rgba(231, 69, 96, 0.12); stroke: #e94560; stroke-width: 2; }
        .hole { fill: none; stroke: #71767b; stroke-width: 2; stroke-dasharray: 6,4; }
        .grid { stroke: #2f3337; stroke-width: 1; }
        .valid { fill: rgba(35, 134, 54, 0.7); stroke: #238636; stroke-width: 1; }
        .hist { fill: rgba(56, 139, 253, 0.6); stroke: #388bfd; stroke-width: 1; }
        .result { fill: rgba(56, 139, 253, 0.4); stroke: #388bfd; stroke-width: 3; }
    </style>
</head>
<body>
    <h1>LIRiAP Algorithm Visualization</h1>
    <p class="subtitle">Step-by-step vertex-grid solver on real polygons</p>
    <div class="controls">
        <div class="row">
            <select id="polySel"><option value="">Select polygon...</option></select>
            <button onclick="run()">Visualize</button>
        </div>
        <div class="row" style="margin-top:12px;">
            <label style="color:#71767b;">Speed:</label>
            <input type="number" id="delay" value="300" min="50" max="2000" step="50" style="width:80px;"/>
        </div>
    </div>
    <div id="viewer" class="viewer hidden">
        <div class="frame-info">
            <span id="frameNum" class="frame-num">Frame 0</span>
            <span id="frameTitle" class="frame-title"></span>
        </div>
        <div id="frameDesc" class="frame-desc"></div>
        <svg viewBox="0 0 260 260" width="360" height="360"><g id="frameSvg"></g></svg>
        <div class="row" style="margin-top:20px;justify-content:center;gap:12px;">
            <button id="prevBtn" onclick="prev()">Previous</button>
            <button id="playBtn" onclick="togglePlay()">Play</button>
            <button id="nextBtn" onclick="next()">Next</button>
        </div>
    </div>
    <script>
    var features = ''' + features_json + ''';
    var framesData = ''' + frames_json + ''';
    var frames = [], idx = 0, timer = null, playing = false;
    
    var polySel = document.getElementById('polySel');
    
    features.forEach(function(f) {
        var opt = document.createElement('option');
        opt.value = f.id;
        opt.textContent = f.name.substring(0,45) + ' ' + Math.round(f.area).toLocaleString() + ' [' + f.type + ']';
        polySel.appendChild(opt);
    });
    
    function run() {
        var pid = parseInt(polySel.value);
        if(isNaN(pid)) return;
        frames = framesData[pid].frames;
        idx = 0;
        document.getElementById('viewer').classList.remove('hidden');
        render();
    }
    
    function render() {
        var f = frames[idx];
        document.getElementById('frameNum').textContent = 'Frame ' + idx + ' / ' + (frames.length - 1);
        document.getElementById('frameTitle').textContent = f.title;
        document.getElementById('frameDesc').textContent = f.desc || '';
        document.getElementById('frameSvg').innerHTML = f.svg || '';
        document.getElementById('prevBtn').disabled = idx === 0;
        document.getElementById('nextBtn').disabled = idx >= frames.length - 1;
    }
    
    function prev() { if(idx > 0){stop(); idx--; render();} }
    function next() { if(idx < frames.length - 1){stop(); idx++; render();} }
    function togglePlay() { playing ? stop() : play(); }
    
    function play() {
        playing = true;
        document.getElementById('playBtn').textContent = 'Pause';
        var d = parseInt(document.getElementById('delay').value) || 300;
        function loop(){ if(!playing || idx >= frames.length - 1){stop(); return;} idx++; render(); timer = setTimeout(loop, d); }
        loop();
    }
    
    function stop() {
        playing = false;
        document.getElementById('playBtn').textContent = 'Play';
        if(timer){ clearTimeout(timer); timer = null; }
    }
    </script>
</body>
</html>'''
    
    output_dir = Path(__file__).parent / 'output' / 'visualize_steps'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'interactive.html'
    output_path.write_text(html)
    print("Created:", output_path)


if __name__ == '__main__':
    main()