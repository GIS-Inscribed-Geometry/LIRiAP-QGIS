use crate::geometry::Rect;
use crate::ratio::apply_ratio_constraint;

/// O(n²) exact convex solver with midpoint-augmented y-set.
pub fn exact_solve_convex(exterior: &[f64], max_ratio: f64) -> Option<Rect> {
    let n = exterior.len() / 2;
    if n < 3 { return None; }
    let coords: Vec<(f64,f64)> = exterior.chunks_exact(2).map(|c|(c[0],c[1])).collect();

    let mut ys: Vec<f64> = coords.iter().map(|&(_,y)| y).collect();
    ys.sort_unstable_by(|a,b| a.partial_cmp(b).unwrap());
    ys.dedup_by(|a,b| (*a - *b).abs() < 1e-9);

    // Augment with midpoints
    let mut ys_aug: Vec<f64> = Vec::with_capacity(2 * ys.len());
    for i in 0..ys.len().saturating_sub(1) {
        ys_aug.push(ys[i]);
        ys_aug.push(0.5 * (ys[i] + ys[i+1]));
    }
    if let Some(&last) = ys.last() { ys_aug.push(last); }

    let (mut best_area, mut best) = (0.0_f64, None::<Rect>);
    for i in 0..ys_aug.len() {
        for j in i+1..ys_aug.len() {
            let (y_lo, y_hi) = (ys_aug[i], ys_aug[j]);
            if let Some((xl, xr)) = tight_x_extent(&coords, n, y_lo, y_hi) {
                const EPS: f64 = 1e-7;
                let raw = Rect { x0: xl+EPS, y0: y_lo+EPS, x1: xr-EPS, y1: y_hi-EPS };
                if raw.area() <= best_area { continue; }
                if let Some(r) = apply_ratio_constraint(raw, max_ratio, exterior, &[]) {
                    if r.area() > best_area { best_area = r.area(); best = Some(r); }
                }
            }
        }
    }
    best
}

fn tight_x_extent(coords: &[(f64,f64)], n: usize, y_lo: f64, y_hi: f64) -> Option<(f64,f64)> {
    let mut xl = f64::MIN;
    let mut xr = f64::MAX;
    for &y in &[y_lo, y_hi, 0.5*(y_lo+y_hi)] {
        let (lx, rx) = x_extent_at_y(coords, n, y)?;
        if lx > xl { xl = lx; }
        if rx < xr { xr = rx; }
    }
    if xl < xr { Some((xl, xr)) } else { None }
}

fn x_extent_at_y(coords: &[(f64,f64)], n: usize, y: f64) -> Option<(f64,f64)> {
    let mut xs: Vec<f64> = Vec::new();
    for i in 0..n {
        let j = (i+1) % n;
        let (x0,y0) = coords[i]; let (x1,y1) = coords[j];
        if (y0 <= y && y < y1) || (y1 <= y && y < y0) {
            let t = (y - y0) / (y1 - y0);
            xs.push(x0 + t*(x1-x0));
        }
        if (y1 - y).abs() < 1e-9 { xs.push(x1); }
    }
    if xs.len() < 2 { return None; }
    xs.sort_unstable_by(|a,b| a.partial_cmp(b).unwrap());
    Some((*xs.first()?, *xs.last()?))
}
