//! Conservative containment of a draw's padded bounds in one opaque fill.
//! A Bezier can be replaced by its chord for winding only when its entire
//! control box misses the query rectangle: that deformation cannot cross the
//! query. Subdivide ambiguous boxes; a depth/work limit means "unknown".
//! This is a sufficient test, not general path boolean arithmetic.
use resvg::{
    tiny_skia::PathSegment,
    usvg::{self, Node},
};

type Point = [f64; 2];
type Rect = [f64; 4];

pub(super) struct Cover {
    segments: Vec<Vec<Point>>,
    bounds: Rect,
    rule: usvg::FillRule,
}

fn opaque_path(mut node: &Node) -> Option<&usvg::Path> {
    loop {
        match node {
            Node::Group(g) => {
                if g.opacity().get() != 1.0
                    || g.blend_mode() != usvg::BlendMode::Normal
                    || g.clip_path().is_some()
                    || g.mask().is_some()
                    || !g.filters().is_empty()
                {
                    return None;
                }
                let [child] = g.children() else { return None };
                node = child;
            }
            Node::Path(p) => {
                let fill = p.fill()?;
                return (p.is_visible()
                    && fill.opacity().get() == 1.0
                    && matches!(fill.paint(), usvg::Paint::Color(_)))
                .then_some(p);
            }
            _ => return None,
        }
    }
}

impl Cover {
    pub(super) fn new(node: &Node) -> Option<Self> {
        let path = opaque_path(node)?;
        let data = path.data().clone().transform(path.abs_transform())?;
        let b = data.bounds();
        let point = |p: resvg::tiny_skia::Point| [p.x as f64, p.y as f64];
        let mut segments = Vec::new();
        let mut start = [0.0; 2];
        let mut current = start;
        let mut open = false;
        // Explicitly close every subpath, including SVG's implicit fill close.
        for segment in data.segments() {
            match segment {
                PathSegment::MoveTo(p) => {
                    if open && current != start {
                        segments.push(vec![current, start]);
                    }
                    start = point(p);
                    current = start;
                    open = true;
                }
                PathSegment::LineTo(p) => {
                    let end = point(p);
                    segments.push(vec![current, end]);
                    current = end;
                }
                PathSegment::QuadTo(a, b) => {
                    let end = point(b);
                    segments.push(vec![current, point(a), end]);
                    current = end;
                }
                PathSegment::CubicTo(a, b, c) => {
                    let end = point(c);
                    segments.push(vec![current, point(a), point(b), end]);
                    current = end;
                }
                PathSegment::Close => {
                    if current != start {
                        segments.push(vec![current, start]);
                    }
                    current = start;
                    open = false;
                }
            }
        }
        if open && current != start {
            segments.push(vec![current, start]);
        }
        Some(Self {
            segments,
            bounds: [
                b.left() as f64,
                b.top() as f64,
                b.right() as f64,
                b.bottom() as f64,
            ],
            rule: path.fill()?.rule(),
        })
    }

    pub(super) fn contains(&self, bounds: [f32; 4]) -> bool {
        // collect() already pads layer bounds by one source pixel. Add one
        // more so the proof stays away from raster AA and f32 boundary noise.
        let r = [
            bounds[0] as f64 - 1.0,
            bounds[1] as f64 - 1.0,
            bounds[2] as f64 + 1.0,
            bounds[3] as f64 + 1.0,
        ];
        if !r.iter().all(|v| v.is_finite())
            || r[0] >= r[2]
            || r[1] >= r[3]
            || r[0] <= self.bounds[0]
            || r[1] <= self.bounds[1]
            || r[2] >= self.bounds[2]
            || r[3] >= self.bounds[3]
        {
            return false;
        }
        let q = [(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5];
        let mut budget = 4096;
        let mut winding = 0;
        for segment in &self.segments {
            let Some(w) = outside_winding(segment, r, q, 0, &mut budget) else {
                return false;
            };
            winding += w;
        }
        match self.rule {
            usvg::FillRule::NonZero => winding != 0,
            usvg::FillRule::EvenOdd => winding % 2 != 0,
        }
    }
}

fn chord_winding(a: Point, b: Point, q: Point) -> i32 {
    let cross = (b[0] - a[0]) * (q[1] - a[1]) - (q[0] - a[0]) * (b[1] - a[1]);
    if a[1] <= q[1] && b[1] > q[1] && cross > 0.0 {
        1
    } else if a[1] > q[1] && b[1] <= q[1] && cross < 0.0 {
        -1
    } else {
        0
    }
}

fn outside_winding(
    p: &[Point],
    r: Rect,
    q: Point,
    depth: usize,
    budget: &mut usize,
) -> Option<i32> {
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    let disjoint = p.iter().all(|p| p[0] < r[0])
        || p.iter().all(|p| p[0] > r[2])
        || p.iter().all(|p| p[1] < r[1])
        || p.iter().all(|p| p[1] > r[3]);
    if disjoint {
        return Some(chord_winding(p[0], *p.last()?, q));
    }
    let inside = |p: Point| p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3];
    if inside(p[0]) || inside(*p.last()?) || depth == 16 {
        return None;
    }
    // De Casteljau works for lines, quadratics and cubics. No global
    // flattening tolerance or sampled point containment is used.
    let n = p.len();
    let mut work = [[0.0; 2]; 4];
    work[..n].copy_from_slice(p);
    let mut left = work;
    let mut right = work;
    for level in 1..n {
        for i in 0..n - level {
            work[i] = [
                (work[i][0] + work[i + 1][0]) * 0.5,
                (work[i][1] + work[i + 1][1]) * 0.5,
            ];
        }
        left[level] = work[0];
        right[n - 1 - level] = work[n - 1 - level];
    }
    Some(
        outside_winding(&left[..n], r, q, depth + 1, budget)?
            + outside_winding(&right[..n], r, q, depth + 1, budget)?,
    )
}
