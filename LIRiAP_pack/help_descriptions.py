"""
Shared user-facing descriptions for LIRiAP processing algorithms.
"""


_BASELINE = (
    "Problem solved:\n"
    "Find a largest-area non axis aligned rectangle fully contained in each input polygon, "
    "including concave polygons and polygons with holes.\n"
)


_ALGORITHM_DETAILS = {
    "approximation_standard": (
        "Algorithm steps:\n"
        "1. Score candidate angles using edge orientation and upper-bound pruning.\n"
        "2. Evaluate candidates on a coarse grid.\n"
        "3. Refine near the best angle with bounded scalar optimization and a fine grid.\n"
        "4. Rotate the best rectangle back to map coordinates.\n"
        "\n"
        "Approach used:\n"
        "Approximate two-resolution search."
    ),
    "approximation_fast": (
        "Algorithm steps:\n"
        "1. Score candidate angles using edge orientation and upper-bound pruning.\n"
        "2. Evaluate candidates on a coarse grid.\n"
        "3. Refine near the best angle with bounded scalar optimization and a fine grid.\n"
        "4. Run via slice-based worker execution for lower overhead.\n"
        "\n"
        "Approach used:\n"
        "Same approximation method as Standard, with a faster execution path for large batches."
    ),
    "contained_standard": (
        "Algorithm steps:\n"
        "1. Build top-K candidates from edge-guided coarse search.\n"
        "2. For each candidate, run angle polish and fine-grid solve.\n"
        "3. Certify containment and symmetrically shrink when needed.\n"
        "4. If strict certification fails and fallback is enabled, return best-effort contained result.\n"
        "\n"
        "Approach used:\n"
        "Certified contained search with explicit containment checks and optional fallback."
    ),
    "contained_fast": (
        "Algorithm steps:\n"
        "1. Build top-K candidates from edge-guided coarse search.\n"
        "2. For each candidate, run angle polish and fine-grid solve.\n"
        "3. Certify containment and symmetrically shrink when needed.\n"
        "4. If strict certification fails and fallback is enabled, return best-effort contained result.\n"
        "\n"
        "Approach used:\n"
        "Same certified contained method as Standard, with optimized execution for throughput."
    ),
    "bcrs": (
        "Algorithm steps:\n"
        "1. Build top-K candidates from edge-guided coarse search.\n"
        "2. Polish angle around each candidate.\n"
        "3. Run BCRS (boundary-coordinate raster solve) in rotated space.\n"
        "4. Apply clamped CABF boundary expansion, then containment certification.\n"
        "5. Use best-effort fallback only when strict certification cannot pass and fallback is enabled.\n"
        "\n"
        "Approach used:\n"
        "Boundary-coordinate method designed for stronger geometric fit on straight-sided shapes."
    ),
    "bcrs_fast": (
        "Algorithm steps:\n"
        "1. Build top-K candidates from edge-guided coarse search.\n"
        "2. Polish angle around each candidate and prioritize the strongest trials.\n"
        "3. Run BCRS (boundary-coordinate raster solve) in rotated space.\n"
        "4. Apply clamped CABF boundary expansion, then containment certification.\n"
        "5. Use best-effort fallback only when strict certification cannot pass and fallback is enabled.\n"
        "\n"
        "Approach used:\n"
        "Same BCRS/CABF method as Standard, with trial limiting and runtime parallel optimizations."
    ),
}


def build_short_help(algorithm_title, algorithm_key, numba_available):
    details = _ALGORITHM_DETAILS[algorithm_key]
    return (
        f"{algorithm_title}\n\n"
        f"{_BASELINE}\n"
        f"{details}\n\n"
        f"Numba acceleration available: {'yes' if numba_available else 'no'}."
    )
