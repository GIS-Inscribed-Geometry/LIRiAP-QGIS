use crate::geometry::Rect;
use crate::pip::cell_is_valid_inside;
use crate::lrh::lrh_sweep;
use crate::ratio::apply_ratio_constraint;

const MIN_SPAN: f64 = 1e-9;

fn build_axis(coords: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = coords.to_vec();
    v.sort_unstable_by(|a,b| a.partial_cmp(b).unwrap());
    v.dedup_by(|a,b| (*a - *b).abs() < 1e-9);
    let mut aug: Vec<f64> = Vec::with_capacity(2 * v.len());
    for i in 0..v.len().saturating_sub(1) {
        aug.push(v[i]);
        aug.push(0.5 * (v[i] + v[i+1]));
    }
    if let Some(&last) = v.last() { aug.push(last); }
    aug
}

pub fn vertex_grid_solve(exterior: &[f64], holes: &[Vec<f64>],
                         max_ratio: f64) -> Option<Rect> {
    let n = exterior.len() / 2;
    if n < 3 { return None; }

    let mut all_xs: Vec<f64> = exterior.chunks_exact(2).map(|c| c[0]).collect();
    let mut all_ys: Vec<f64> = exterior.chunks_exact(2).map(|c| c[1]).collect();
    for hole in holes {
        all_xs.extend(hole.chunks_exact(2).map(|c| c[0]));
        all_ys.extend(hole.chunks_exact(2).map(|c| c[1]));
    }
    let xs = build_axis(&all_xs);
    let ys = build_axis(&all_ys);

    let ncols = xs.len().saturating_sub(1);
    let nrows = ys.len().saturating_sub(1);
    if ncols == 0 || nrows == 0 { return None; }

    let mut mask = vec![false; nrows * ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            mask[r*ncols + c] = cell_is_valid_inside(
                exterior, holes, xs[c], ys[r], xs[c+1], ys[r+1], MIN_SPAN);
        }
    }

    let mut heights = vec![0usize; ncols];
    let mut best_area = 0.0_f64;
    let mut best: Option<Rect> = None;

    for r in 0..nrows {
        for c in 0..ncols {
            heights[c] = if mask[r*ncols + c] { heights[c]+1 } else { 0 };
        }
        if let Some(raw) = lrh_sweep(&xs, &ys, r, &heights) {
            if raw.area() > best_area {
                if let Some(rect) = apply_ratio_constraint(raw, max_ratio, exterior, holes) {
                    if rect.area() > best_area { best_area = rect.area(); best = Some(rect); }
                }
            }
        }
    }
    best
}
