use crate::geometry::Rect;
use crate::pip::rect_inside_polygon;

pub fn certify(rect: Rect, exterior: &[f64], holes: &[Vec<f64>]) -> Option<Rect> {
    if rect_inside_polygon(exterior, holes, rect.x0, rect.y0, rect.x1, rect.y1) {
        return Some(rect);
    }
    const EPS: f64 = 1e-3;
    let r = Rect { x0: rect.x0+EPS, y0: rect.y0+EPS, x1: rect.x1-EPS, y1: rect.y1-EPS };
    if r.width() > 0.0 && r.height() > 0.0 &&
       rect_inside_polygon(exterior, holes, r.x0, r.y0, r.x1, r.y1) { Some(r) } else { None }
}
