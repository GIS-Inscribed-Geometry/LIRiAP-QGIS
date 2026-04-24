use crate::classify::classify;
use crate::convex::exact_solve_convex;
use crate::grid::vertex_grid_solve;
use crate::refine::refine_to_boundary;
use crate::certify::certify;
use crate::geometry::{rotate_coords, rotate_rect_back, ring_centroid, Rect};

pub struct SolveResult {
    pub corners:          [(f64,f64); 4],
    pub area:             f64,
    pub ratio:            f64,
    pub poly_type:        String,
    pub used_best_effort: bool,
}

pub fn solve(exterior: &[f64], holes: &[Vec<f64>],
             axis_angle: f64, max_ratio: f64,
             always_return: bool, buf_value: f64) -> Option<SolveResult> {
    if exterior.len() < 6 { return None; }

    let (ext_b, holes_b): (Vec<f64>, Vec<Vec<f64>>) = if buf_value != 0.0 {
        apply_buffer(exterior, holes, buf_value)
    } else {
        (exterior.to_vec(), holes.to_vec())
    };
    if ext_b.len() < 6 { return None; }

    let poly_type = classify(&ext_b, &holes_b);
    let (cx, cy) = ring_centroid(&ext_b);
    let ext_r = rotate_coords(&ext_b, cx, cy, axis_angle);
    let holes_r: Vec<Vec<f64>> = holes_b.iter()
        .map(|h| rotate_coords(h, cx, cy, axis_angle)).collect();

    let rect_r: Option<Rect> = match poly_type {
        "convex_no_holes" => exact_solve_convex(&ext_r, max_ratio),
        _                 => vertex_grid_solve(&ext_r, &holes_r, max_ratio),
    };

    let rect_r = match rect_r {
        Some(r) => r,
        None => {
            return if always_return {
                best_effort(&ext_r, &holes_r, poly_type, cx, cy, axis_angle)
            } else { None };
        }
    };

    let rect_r = if poly_type != "convex_no_holes" {
        refine_to_boundary(rect_r, &ext_r, &holes_r, max_ratio)
    } else { rect_r };

    let rect_r = match certify(rect_r, &ext_r, &holes_r) {
        Some(r) => r,
        None => {
            return if always_return {
                best_effort(&ext_r, &holes_r, poly_type, cx, cy, axis_angle)
            } else { None };
        }
    };

    emit(rect_r, poly_type, false, cx, cy, axis_angle)
}

fn best_effort(ext_r: &[f64], holes_r: &[Vec<f64>], poly_type: &str,
               cx: f64, cy: f64, angle: f64) -> Option<SolveResult> {
    let r = vertex_grid_solve(ext_r, holes_r, 0.0)?;
    let r = refine_to_boundary(r, ext_r, holes_r, 0.0);
    let r = certify(r, ext_r, holes_r)?;
    emit(r, poly_type, true, cx, cy, angle)
}

fn emit(r: Rect, pt: &str, be: bool, cx: f64, cy: f64, angle: f64) -> Option<SolveResult> {
    let area = r.area();
    let (w, h) = (r.width(), r.height());
    let ratio = if w > 0.0 && h > 0.0 { w.max(h)/w.min(h) } else { 1.0 };
    Some(SolveResult {
        corners: rotate_rect_back(&r, cx, cy, angle),
        area, ratio, poly_type: pt.to_string(), used_best_effort: be,
    })
}

fn apply_buffer(exterior: &[f64], holes: &[Vec<f64>], buf: f64)
    -> (Vec<f64>, Vec<Vec<f64>>) {
    let scale = |ring: &[f64], inward: bool| -> Vec<f64> {
        let (rcx, rcy) = crate::geometry::ring_centroid(ring);
        let shift = if inward { buf } else { -buf };
        ring.chunks_exact(2).flat_map(|pt| {
            let (dx, dy) = (pt[0]-rcx, pt[1]-rcy);
            let d = (dx*dx + dy*dy).sqrt();
            if d < 1e-12 { return vec![pt[0], pt[1]]; }
            let s = ((d + shift) / d).max(0.0);
            vec![rcx + dx*s, rcy + dy*s]
        }).collect()
    };
    let ext_b = scale(exterior, buf < 0.0);
    let holes_b = holes.iter().map(|h| scale(h, buf >= 0.0)).collect();
    (ext_b, holes_b)
}
