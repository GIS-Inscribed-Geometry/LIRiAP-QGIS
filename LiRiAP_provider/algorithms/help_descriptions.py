"""
Shared user-facing descriptions for LIRiAP processing algorithms.
"""

_BASELINE = (
    "<div style='font-family:Segoe UI, Arial, sans-serif; font-size:10pt; line-height:1.35;'>"
    "<p><b>Problem framing</b><br/>"
    "LIRiAP exposes four solver families with different guarantees:</p>"
    "<ul>"
    "<li><b>Approximation</b>: fast area-focused search, not strict containment-certified.</li>"
    "<li><b>Skeleton</b>: BCRS-free solver using medial-axis skeleton decomposition for seed generation.</li>"
    "<li><b>BCRS</b>: containment certification plus boundary expansion (historical earlier implementation).</li>"
    "<li><b>Axis-Aligned</b>: exact fixed-axis solution at vertex-coordinate precision.</li>"
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
    "skeleton": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Extract medial-axis skeleton from polygon distance transform.</li>"
        "<li>Generate hybrid seeds from skeleton PCA angles + edge angles.</li>"
        "<li>Run grid solve at each seed angle (40x40).</li>"
        "<li>Apply SDF binary expansion for boundary contact.</li>"
        "<li>Test ±2° delta angles around candidates.</li>"
        "<li>SDF containment certification with symmetric shrink if needed.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>BCRS-free alternative using skeleton-guided seed generation.</li>"
        "<li><code>ALWAYS_RETURN=False</code>: strict certification only (can return no rectangle).</li>"
        "<li><code>ALWAYS_RETURN=True</code>: best-effort fallback allowed.</li>"
        "<li><code>USE_BUFFER/BUFFER_VALUE</code> add containment margin after solve.</li>"
        "</ul>"
    ),
    "bcrs": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Build top-K candidates from edge-guided coarse search.</li>"
        "<li>Polish angle around each candidate.</li>"
        "<li>Run BCRS (boundary-coordinate raster solve) in rotated space.</li>"
        "<li>Apply clamped boundary expansion, then containment certification.</li>"
        "<li>Use best-effort fallback only when strict certification cannot pass and fallback is enabled.</li>"
        "</ol>"
        "<p><b>Guarantee and setting semantics</b></p>"
        "<ul>"
        "<li>Uses SDF-guided boundary expansion.</li>"
        "<li><code>ALWAYS_RETURN=False</code>: strict certification only (can return no rectangle).</li>"
        "<li><code>ALWAYS_RETURN=True</code>: best-effort fallback allowed.</li>"
        "<li><code>USE_BUFFER/BUFFER_VALUE</code> add containment margin after solve.</li>"
        "</ul>"
    ),
"bcrs_fast": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Angle scan with upper-bound pruning to find promising candidates.</li>"
        "<li>Evaluate on coarse/fine vertex grid (Daniels et al. 1997).</li>"
        "<li>Binary-search boundary push for exact placement.</li>"
        "<li>Containment certification with boundary expansion.</li>"
        "<li>Rotate back to map coordinates.</li>"
        "</ol>"
        "<p><b>Guarantee</b></p>"
        "<ul>"
        "<li>Full containment certification + boundary expansion (historical).</li>"
        "<li><code>TOP_K</code> and <code>ANGLE_STEP</code> tune quality.</li>"
        "</ul>"
    ),
    "axis_aligned": (
        "<p><b>Algorithm steps</b></p>"
        "<ol>"
        "<li>Classify polygon type (convex/concave, with/without holes).</li>"
        "<li>For convex: Alt/Amenta O(n2) vertex-pair enumeration.</li>"
        "<li>For concave: Daniels et al. vertex-coordinate grid solver.</li>"
        "<li>Binary-search boundary refinement for diagonal edges.</li>"
        "<li>Epsilon-inset containment certification.</li>"
        "</ol>"
        "<p><b>Guarantee</b></p>"
        "<ul>"
        "<li>Exact solution for convex polygons.</li>"
        "<li>Vertex-grid optimal for concave.</li>"
        "<li>Strict containment with epsilon inset.</li>"
        "<li><code>GRID_FINE</code> for fallback grid resolution.</li>"
        "<li><code>ALWAYS_RETURN</code> enables best-effort fallback.</li>"
        "</ul>"
    ),
}


def build_short_help(algorithm_title, algorithm_key, numba_available):
    """
    Build HTML help string for an algorithm.

    Parameters
    ----------
    algorithm_title : str
        Human-readable title for the algorithm.
    algorithm_key : str
        Key matching entries in _ALGORITHM_DETAILS.
    numba_available : bool
        Whether Numba JIT is available.

    Returns
    -------
    str
        HTML-formatted help string.
    """
    details = _ALGORITHM_DETAILS[algorithm_key]
    return (
        f"{_BASELINE}"
        f"<h3 style='margin:10px 0 6px 0;'>{algorithm_title}</h3>"
        f"{details}"
        f"<p><b>Runtime note</b>: Numba acceleration available: "
        f"{'yes' if numba_available else 'no'}.</p>"
        f"</div>"
    )
