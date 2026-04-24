"""lir_solver — Largest Inscribed Rectangle, Rust/PyO3 backend."""
from .lir_solver import solve_axis_aligned_lir, solve_axis_aligned_lir_batch
__all__ = ["solve_axis_aligned_lir", "solve_axis_aligned_lir_batch"]
