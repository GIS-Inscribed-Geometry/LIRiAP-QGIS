use crate::geometry::Rect;
use crate::pip::rect_inside_polygon;

const STEPS: u32 = 52;
const PASSES: u32 = 2;

fn ratio_ok(r: &Rect, mr: f64) -> bool {
    if mr <= 0.0 { return true; }
    let (w, h) = (r.width(), r.height());
    if w <= 0.0 || h <= 0.0 { return false; }
    w.max(h) / w.min(h) <= mr + 1e-9
}

fn valid(r: &Rect, ext: &[f64], holes: &[Vec<f64>], mr: f64) -> bool {
    ratio_ok(r, mr) && rect_inside_polygon(ext, holes, r.x0, r.y0, r.x1, r.y1)
}

pub fn refine_to_boundary(rect: Rect, ext: &[f64], holes: &[Vec<f64>],
                           max_ratio: f64) -> Rect {
    let (bmnx, bmny, bmxx, bmxy) = crate::geometry::ring_bbox(ext);
    let search = (bmxx-bmnx).max(bmxy-bmny);
    if search <= 0.0 { return rect; }
    let mut r = rect;
    for _ in 0..PASSES {
        // push left
        let (mut lo, mut hi) = (0.0_f64, search);
        for _ in 0..STEPS { let m=0.5*(lo+hi); let c=Rect{x0:r.x0-m,..r}; if valid(&c,ext,holes,max_ratio){lo=m}else{hi=m}; }
        r = Rect{x0: r.x0-lo, ..r};
        // push right
        let (mut lo, mut hi) = (0.0_f64, search);
        for _ in 0..STEPS { let m=0.5*(lo+hi); let c=Rect{x1:r.x1+m,..r}; if valid(&c,ext,holes,max_ratio){lo=m}else{hi=m}; }
        r = Rect{x1: r.x1+lo, ..r};
        // push bottom
        let (mut lo, mut hi) = (0.0_f64, search);
        for _ in 0..STEPS { let m=0.5*(lo+hi); let c=Rect{y0:r.y0-m,..r}; if valid(&c,ext,holes,max_ratio){lo=m}else{hi=m}; }
        r = Rect{y0: r.y0-lo, ..r};
        // push top
        let (mut lo, mut hi) = (0.0_f64, search);
        for _ in 0..STEPS { let m=0.5*(lo+hi); let c=Rect{y1:r.y1+m,..r}; if valid(&c,ext,holes,max_ratio){lo=m}else{hi=m}; }
        r = Rect{y1: r.y1+lo, ..r};
    }
    r
}
