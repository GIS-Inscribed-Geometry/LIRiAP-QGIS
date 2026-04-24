#[derive(Debug, Clone, Copy)]
pub struct Rect { pub x0: f64, pub y0: f64, pub x1: f64, pub y1: f64 }

impl Rect {
    #[inline] pub fn width(&self)  -> f64 { self.x1 - self.x0 }
    #[inline] pub fn height(&self) -> f64 { self.y1 - self.y0 }
    #[inline] pub fn area(&self)   -> f64 { self.width() * self.height() }
    pub fn corners(&self) -> [(f64,f64);4] {
        [(self.x0,self.y0),(self.x1,self.y0),(self.x1,self.y1),(self.x0,self.y1)]
    }
}

pub fn rotate_coords(flat: &[f64], cx: f64, cy: f64, angle_deg: f64) -> Vec<f64> {
    if angle_deg == 0.0 { return flat.to_vec(); }
    let (s, c) = angle_deg.to_radians().sin_cos();
    flat.chunks_exact(2).flat_map(|pt| {
        let (dx, dy) = (pt[0]-cx, pt[1]-cy);
        [cx + dx*c - dy*s, cy + dx*s + dy*c]
    }).collect()
}

pub fn rotate_rect_back(rect: &Rect, cx: f64, cy: f64, angle_deg: f64) -> [(f64,f64);4] {
    if angle_deg == 0.0 { return rect.corners(); }
    let (s, c) = (-angle_deg).to_radians().sin_cos();
    rect.corners().map(|(x,y)| {
        let (dx, dy) = (x-cx, y-cy);
        (cx + dx*c - dy*s, cy + dx*s + dy*c)
    })
}

pub fn ring_centroid(flat: &[f64]) -> (f64, f64) {
    let n = flat.len() / 2;
    if n == 0 { return (0.0, 0.0); }
    let sx: f64 = flat.chunks_exact(2).map(|c| c[0]).sum();
    let sy: f64 = flat.chunks_exact(2).map(|c| c[1]).sum();
    (sx / n as f64, sy / n as f64)
}

pub fn ring_bbox(flat: &[f64]) -> (f64,f64,f64,f64) {
    let (mut mnx, mut mny, mut mxx, mut mxy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for c in flat.chunks_exact(2) {
        if c[0] < mnx { mnx = c[0]; } if c[0] > mxx { mxx = c[0]; }
        if c[1] < mny { mny = c[1]; } if c[1] > mxy { mxy = c[1]; }
    }
    (mnx, mny, mxx, mxy)
}
