use crate::geometry::Rect;
use crate::pip::rect_inside_polygon;

#[inline]
fn ratio_ok(r: &Rect, max_ratio: f64) -> bool {
    if max_ratio <= 0.0 { return true; }
    let (w, h) = (r.width(), r.height());
    if w <= 0.0 || h <= 0.0 { return false; }
    w.max(h) / w.min(h) <= max_ratio + 1e-9
}

/// Try all 5 candidate sub-rects (raw + 4 shrink variants), return the best valid one.
pub fn apply_ratio_constraint(raw: Rect, max_ratio: f64,
                               exterior: &[f64], holes: &[Vec<f64>]) -> Option<Rect> {
    if raw.width() <= 0.0 || raw.height() <= 0.0 { return None; }
    let candidates: [Option<Rect>; 5] = [
        Some(raw),
        shrink_width_left (raw, max_ratio),
        shrink_width_right(raw, max_ratio),
        shrink_height_bot (raw, max_ratio),
        shrink_height_top (raw, max_ratio),
    ];
    let mut best_area = 0.0_f64;
    let mut best: Option<Rect> = None;
    for r in candidates.iter().flatten() {
        if !ratio_ok(r, max_ratio) { continue; }
        if r.area() <= best_area   { continue; }
        if rect_inside_polygon(exterior, holes, r.x0, r.y0, r.x1, r.y1) {
            best_area = r.area(); best = Some(*r);
        }
    }
    best
}

fn shrink_width_left (r: Rect, mr: f64) -> Option<Rect> {
    if mr <= 0.0 { return None; }
    Some(Rect { x1: r.x0 + r.height()*mr, ..r })
}
fn shrink_width_right(r: Rect, mr: f64) -> Option<Rect> {
    if mr <= 0.0 { return None; }
    Some(Rect { x0: r.x1 - r.height()*mr, ..r })
}
fn shrink_height_bot (r: Rect, mr: f64) -> Option<Rect> {
    if mr <= 0.0 { return None; }
    Some(Rect { y1: r.y0 + r.width()*mr, ..r })
}
fn shrink_height_top (r: Rect, mr: f64) -> Option<Rect> {
    if mr <= 0.0 { return None; }
    Some(Rect { y0: r.y1 - r.width()*mr, ..r })
}
