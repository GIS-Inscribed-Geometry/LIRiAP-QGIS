#[inline]
pub fn point_in_ring(ring: &[f64], px: f64, py: f64) -> bool {
    let n = ring.len() / 2;
    if n < 3 { return false; }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[2*i], ring[2*i+1]);
        let (xj, yj) = (ring[2*j], ring[2*j+1]);
        if ((yi > py) != (yj > py)) && px < ((xj-xi)*(py-yi)/(yj-yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[inline]
pub fn point_in_polygon(exterior: &[f64], holes: &[Vec<f64>], px: f64, py: f64) -> bool {
    if !point_in_ring(exterior, px, py) { return false; }
    !holes.iter().any(|h| point_in_ring(h, px, py))
}

/// True iff rect [rx0,ry0]×[rx1,ry1] is fully inside polygon.
/// Uses epsilon-inset probes + hole-enclosure guard (non-strict inequalities).
pub fn rect_inside_polygon(exterior: &[f64], holes: &[Vec<f64>],
                            rx0: f64, ry0: f64, rx1: f64, ry1: f64) -> bool {
    const EPS: f64 = 1e-7;
    if rx1 <= rx0 || ry1 <= ry0 { return false; }
    let (mx, my) = (0.5*(rx0+rx1), 0.5*(ry0+ry1));
    if !point_in_polygon(exterior, holes, mx, my) { return false; }
    let (ix0, iy0, ix1, iy1) = (rx0+EPS, ry0+EPS, rx1-EPS, ry1-EPS);
    if ix0 >= ix1 || iy0 >= iy1 { return false; }
    let probes = [(ix0,iy0),(ix1,iy0),(ix1,iy1),(ix0,iy1),
                  (mx,iy0),(mx,iy1),(ix0,my),(ix1,my)];
    for (px,py) in probes {
        if !point_in_polygon(exterior, holes, px, py) { return false; }
    }
    // Hole-enclosure guard: no hole vertex may lie on/inside the rect
    for hole in holes {
        let nv = hole.len() / 2;
        for i in 0..nv {
            let (hx, hy) = (hole[2*i], hole[2*i+1]);
            if hx >= rx0 && hx <= rx1 && hy >= ry0 && hy <= ry1 { return false; }
        }
    }
    true
}

#[inline]
pub fn cell_is_valid_inside(exterior: &[f64], holes: &[Vec<f64>],
                             rx0: f64, ry0: f64, rx1: f64, ry1: f64, min_span: f64) -> bool {
    rx1-rx0 >= min_span && ry1-ry0 >= min_span &&
    rect_inside_polygon(exterior, holes, rx0, ry0, rx1, ry1)
}
