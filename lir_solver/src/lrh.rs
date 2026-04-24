use crate::geometry::Rect;

/// Variable-pitch Largest Rectangle in Histogram (Daniels 1997 variant).
pub fn lrh_sweep(xs: &[f64], ys: &[f64], row_idx: usize, heights: &[usize]) -> Option<Rect> {
    let ncols = xs.len().saturating_sub(1);
    if ncols == 0 { return None; }
    let mut best_area = 0.0_f64;
    let mut best_rect: Option<Rect> = None;
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(ncols + 1);

    for c in 0..=ncols {
        let h = if c < ncols { heights[c] } else { 0 };
        let mut start = c;
        while let Some(&(sc, sh)) = stack.last() {
            if sh <= h { break; }
            stack.pop();
            let x0 = xs[sc]; let x1 = xs[c];
            let ri0 = (row_idx as i64 - sh as i64 + 1).max(0) as usize;
            let ri1 = row_idx + 1;
            if ri1 >= ys.len() { continue; }
            let area = (x1-x0) * (ys[ri1]-ys[ri0]);
            if area > best_area {
                best_area = area;
                best_rect = Some(Rect { x0, y0: ys[ri0], x1, y1: ys[ri1] });
            }
            start = sc;
        }
        if h > 0 { stack.push((start, h)); }
    }
    best_rect
}
