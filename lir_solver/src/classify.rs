pub fn classify(exterior: &[f64], holes: &[Vec<f64>]) -> &'static str {
    let has_holes = !holes.is_empty();
    if is_convex(exterior) {
        if has_holes { "convex_with_holes" } else { "convex_no_holes" }
    } else {
        if has_holes { "concave_with_holes" } else { "concave_no_holes" }
    }
}

fn cross(o: (f64,f64), a: (f64,f64), b: (f64,f64)) -> f64 {
    (a.0-o.0)*(b.1-o.1) - (a.1-o.1)*(b.0-o.0)
}

fn is_convex(flat: &[f64]) -> bool {
    let n = flat.len() / 2;
    if n < 3 { return true; }
    let pts: Vec<(f64,f64)> = flat.chunks_exact(2).map(|c|(c[0],c[1])).collect();
    let mut sign = 0i8;
    for i in 0..n {
        let (o,a,b) = (pts[i], pts[(i+1)%n], pts[(i+2)%n]);
        let c = cross(o, a, b);
        if c.abs() < 1e-10 { continue; }
        let s: i8 = if c > 0.0 { 1 } else { -1 };
        if sign == 0 { sign = s; } else if sign != s { return false; }
    }
    true
}
