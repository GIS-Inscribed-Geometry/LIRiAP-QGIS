mod geometry; mod pip; mod lrh; mod grid; mod convex; mod ratio;
mod refine; mod certify; mod classify; pub mod solver;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn result_to_py(py: Python<'_>, r: solver::SolveResult) -> Py<PyTuple> {
    let corners: Vec<Py<PyTuple>> = r.corners.iter()
        .map(|&(x,y)| PyTuple::new_bound(py, [x,y]).unbind()).collect();
    PyTuple::new_bound(py, &[
        PyList::new_bound(py, corners).into_py(py),
        r.area.into_py(py),
        r.ratio.into_py(py),
        r.poly_type.into_py(py),
        r.used_best_effort.into_py(py),
    ]).unbind()
}

#[pyfunction]
fn solve_axis_aligned_lir(py: Python<'_>,
    exterior: Vec<f64>, holes: Vec<Vec<f64>>,
    axis_angle: f64, max_ratio: f64, always_return: bool, buf_value: f64,
) -> PyResult<Option<Py<PyTuple>>> {
    Ok(solver::solve(&exterior, &holes, axis_angle, max_ratio, always_return, buf_value)
        .map(|r| result_to_py(py, r)))
}

#[pyfunction]
fn solve_axis_aligned_lir_batch(py: Python<'_>,
    polygons: Vec<(Vec<f64>, Vec<Vec<f64>>)>,
    axis_angle: f64, max_ratio: f64, always_return: bool, buf_value: f64,
) -> PyResult<Vec<Option<Py<PyTuple>>>> {
    polygons.iter().map(|(ext, holes)|
        Ok(solver::solve(ext, holes, axis_angle, max_ratio, always_return, buf_value)
            .map(|r| result_to_py(py, r)))
    ).collect()
}

#[pymodule]
fn lir_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_axis_aligned_lir, m)?)?;
    m.add_function(wrap_pyfunction!(solve_axis_aligned_lir_batch, m)?)?;
    Ok(())
}
