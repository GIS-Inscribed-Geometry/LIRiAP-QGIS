"""Drop-in replacement for _solve_axis_aligned_lir using the Rust backend.
Falls back to the Python worker automatically if the .so is missing."""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

try:
    import lir_rust as _rs
    _RUST_AVAILABLE = True
except Exception as e:
    _rs = None
    _RUST_AVAILABLE = False
    _log.warning("lir_solver not found – falling back to Python solver: %r", e)


def _ring_flat(ring):
    c = list(ring.coords)
    if c and c[0] == c[-1]:
        c = c[:-1]
    return [v for pt in c for v in pt]


def _poly_flat(poly):
    return _ring_flat(poly.exterior), [_ring_flat(h) for h in poly.interiors]


def _corners_to_poly(corners):
    from shapely.geometry import Polygon
    pts = [tuple(p) for p in corners]
    if len(pts) < 4:
        return None
    return Polygon(pts + [pts[0]])


def solve_lir_shim(poly, axis_angle=0.0, grid_fine=300,
                   max_ratio=1.6, always_return=True,
                   use_buffer=True, buf_value=-0.5):
    """Same signature and return value as _solve_axis_aligned_lir."""
    null = (None, 0.0, axis_angle, "unknown", 1.0, False)

    if not _RUST_AVAILABLE:
        return _py_fallback(poly, axis_angle, grid_fine, max_ratio,
                            always_return, use_buffer, buf_value)

    if poly is None or poly.is_empty:
        return null

    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly is None or poly.is_empty:
            return null

    ext, holes = _poly_flat(poly)
    buf = buf_value if use_buffer else 0.0

    try:
        r = _rs.solve_axis_aligned_lir(
            ext, holes, axis_angle, max_ratio, always_return, buf
        )
    except Exception as e:
        _log.exception("Rust solve failed, falling back to Python: %s", e)
        return _py_fallback(poly, axis_angle, grid_fine, max_ratio,
                            always_return, use_buffer, buf_value)

    if r is None:
        return null

    corners, area, ratio, ptype, be = r
    rect = _corners_to_poly(corners)
    if rect is None or rect.is_empty:
        return null

    return (rect, float(area), float(axis_angle), str(ptype), float(ratio), bool(be))


def solve_lir_batch_shim(polys, axis_angle=0.0, grid_fine=300,
                         max_ratio=1.6, always_return=True,
                         use_buffer=True, buf_value=-0.5):
    """Vectorised variant."""
    if not _RUST_AVAILABLE:
        return [solve_lir_shim(p, axis_angle, grid_fine, max_ratio,
                               always_return, use_buffer, buf_value) for p in polys]

    buf = buf_value if use_buffer else 0.0
    data = []

    for poly in polys:
        if poly is None or poly.is_empty:
            data.append(([], []))
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly is None or poly.is_empty:
            data.append(([], []))
            continue
        data.append(_poly_flat(poly))

    null = (None, 0.0, axis_angle, "unknown", 1.0, False)
    out = []

    try:
        raw = _rs.solve_axis_aligned_lir_batch(
            data, axis_angle, max_ratio, always_return, buf
        )
    except Exception as e:
        _log.exception("Rust batch solve failed, falling back to Python: %s", e)
        return [solve_lir_shim(p, axis_angle, grid_fine, max_ratio,
                               always_return, use_buffer, buf_value) for p in polys]

    for r in raw:
        if r is None:
            out.append(null)
            continue

        corners, area, ratio, ptype, be = r
        rect = _corners_to_poly(corners)
        if rect is None or rect.is_empty:
            out.append(null)
            continue

        out.append((rect, float(area), float(axis_angle), str(ptype), float(ratio), bool(be)))

    return out


def _py_fallback(poly, axis_angle, grid_fine, max_ratio, always_return, use_buffer, buf_value):
    try:
        from axis_aligned_lir_worker import _solve_axis_aligned_lir_python
        return _solve_axis_aligned_lir_python(
            poly,
            axis_angle,
            grid_fine,
            max_ratio,
            always_return,
            use_buffer,
            buf_value,
        )
    except Exception:
        return (None, 0.0, axis_angle, "unknown", 1.0, False)