"""
Shared user-facing descriptions for LIRiAP processing algorithms.
"""

_BASELINE = (
    "<div style='font-family:Segoe UI, Arial, sans-serif; font-size:10pt; line-height:1.35;'>"
    "<p><b>Problem framing</b><br/>"
    "LIRiAP exposes three solver families with different guarantees:</p>"
    "<ul>"
    "<li><b>Approximation</b>: fast area-focused search, not strict containment-certified.</li>"
    "<li><b>Contained</b>: strict containment certification (optional best-effort fallback), no expansion stage.</li>"
    "<li><b>BCRS</b>: containment certification plus CABF boundary expansion; full target method in this plugin.</li>"
    "</ul>"
    "<p>Concave polygons and polygons with holes are supported.</p>"
)

_ALGORITHM_DETAILS = {
    "approximation_standard": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Score candidate angles using edge orientation and upper-bound pruning.</li>"
        "<li>Evaluate candidates on a coarse grid.</li>"
        "<li>Refine near the best angle with bounded scalar optimization and a fine grid.</li>"
        "<li>Rotate the best rectangle back to map coordinates.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>No strict containment certification.</li>"
        "<li><code>GRID_*</code> and <code>ANGLE_STEP</code> tune quality-vs-runtime only.</li>"
        "<li><code>MAX_RATIO</code> constrains admissible rectangles.</li>"
        "</ul>"
    ),
    "approximation_fast": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Score candidate angles using edge orientation and upper-bound pruning.</li>"
        "<li>Evaluate candidates on a coarse grid.</li>"
        "<li>Refine near the best angle with bounded scalar optimization and a fine grid.</li>"
        "<li>Run via slice-based worker execution for lower overhead.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>Same geometry semantics as Approximation Standard (not strict containment-certified).</li>"
        "<li>Worker/chunking settings affect runtime only.</li>"
        "<li><code>MAX_RATIO</code> constrains admissible rectangles.</li>"
        "</ul>"
    ),
    "contained_standard": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Build top-K candidates from edge-guided coarse search.</li>"
        "<li>For each candidate, run angle polish and fine-grid solve.</li>"
        "<li>Certify containment and symmetrically shrink when needed.</li>"
        "<li>If strict certification fails and fallback is enabled, return a best-effort result.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li><code>ALWAYS_RETURN=False</code>: strict certification only (can return no rectangle).</li>"
        "<li><code>ALWAYS_RETURN=True</code>: best-effort fallback allowed.</li>"
        "<li>No post-certification expansion stage.</li>"
        "<li><code>USE_BUFFER/BUFFER_VALUE</code> add containment margin (usually smaller area).</li>"
        "</ul>"
    ),
    "contained_fast": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Build top-K candidates from edge-guided coarse search.</li>"
        "<li>For each candidate, run angle polish and fine-grid solve.</li>"
        "<li>Certify containment and symmetrically shrink when needed.</li>"
        "<li>If strict certification fails and fallback is enabled, return a best-effort result.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>Same containment/fallback semantics as Contained Standard.</li>"
        "<li>No post-certification expansion stage.</li>"
        "<li>Worker/chunking settings affect runtime only.</li>"
        "</ul>"
    ),
    "bcrs": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Build top-K candidates from edge-guided coarse search.</li>"
        "<li>Polish angle around each candidate.</li>"
        "<li>Run BCRS (boundary-coordinate raster solve) in rotated space.</li>"
        "<li>Apply clamped CABF boundary expansion, then containment certification.</li>"
        "<li>Use best-effort fallback only when strict certification cannot pass and fallback is enabled.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>This is the only family here with explicit boundary expansion (CABF).</li>"
        "<li><code>ALWAYS_RETURN=False</code>: strict certification only (can return no rectangle).</li>"
        "<li><code>ALWAYS_RETURN=True</code>: best-effort fallback allowed.</li>"
        "<li><code>USE_BUFFER/BUFFER_VALUE</code> add containment margin after solve.</li>"
        "</ul>"
    ),
    "bcrs_fast": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Build top-K candidates from edge-guided coarse search.</li>"
        "<li>Polish angle around each candidate and prioritize the strongest trials.</li>"
        "<li>Run BCRS (boundary-coordinate raster solve) in rotated space.</li>"
        "<li>Apply clamped CABF boundary expansion, then containment certification.</li>"
        "<li>Use best-effort fallback only when strict certification cannot pass and fallback is enabled.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>Same geometry semantics as BCRS Standard (includes CABF expansion).</li>"
        "<li>Worker/chunking settings affect runtime only.</li>"
        "<li><code>ALWAYS_RETURN</code> controls strict-only vs best-effort fallback behavior.</li>"
        "</ul>"
    ),
}


def build_short_help(algorithm_title, algorithm_key, numba_available):
    details = _ALGORITHM_DETAILS[algorithm_key]
    return (
        f"{_BASELINE}"
        f"<h3 style='margin:10px 0 6px 0;'>{algorithm_title}</h3>"
        f"{details}"
        f"<p><b>Runtime note</b>: Numba acceleration available: "
        f"{'yes' if numba_available else 'no'}.</p>"
        f"</div>"
    )
