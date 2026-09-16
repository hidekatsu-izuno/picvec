//! Exact point-to-polyline distances with bounding-box branch and bound.
use super::Point;

#[derive(Clone, Copy)]
struct Segment {
    a: Point,
    d: Point,
    denominator: f32,
    bounds: [f32; 4],
}

impl Segment {
    fn new(a: Point, b: Point) -> Self {
        let d = Point {
            x: b.x - a.x,
            y: b.y - a.y,
        };
        // The endpoint reached by rounded a + t*d may differ from b. Enclose
        // the actual floating-point interpolation, not only ideal endpoints.
        let end = Point {
            x: a.x + d.x,
            y: a.y + d.y,
        };
        Self {
            a,
            d,
            denominator: (d.x * d.x + d.y * d.y).max(1e-12),
            bounds: [
                a.x.min(end.x),
                a.y.min(end.y),
                a.x.max(end.x),
                a.y.max(end.y),
            ],
        }
    }

    fn distance(self, p: Point) -> f32 {
        let t = (((p.x - self.a.x) * self.d.x + (p.y - self.a.y) * self.d.y) / self.denominator)
            .clamp(0.0, 1.0);
        p.distance(Point {
            x: self.a.x + self.d.x * t,
            y: self.a.y + self.d.y * t,
        })
    }
}

enum Contents {
    Leaf(Vec<Segment>),
    Branch(Box<(Node, Node)>),
}

struct Node {
    bounds: [f32; 4],
    contents: Contents,
}

impl Node {
    fn build(mut segments: Vec<Segment>) -> Self {
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for s in &segments {
            bounds[0] = bounds[0].min(s.bounds[0]);
            bounds[1] = bounds[1].min(s.bounds[1]);
            bounds[2] = bounds[2].max(s.bounds[2]);
            bounds[3] = bounds[3].max(s.bounds[3]);
        }
        let contents = if segments.len() <= 8 {
            Contents::Leaf(segments)
        } else {
            let axis = usize::from(bounds[3] - bounds[1] > bounds[2] - bounds[0]);
            let middle = segments.len() / 2;
            segments.select_nth_unstable_by(middle, |a, b| {
                (a.bounds[axis] * 0.5 + a.bounds[axis + 2] * 0.5)
                    .total_cmp(&(b.bounds[axis] * 0.5 + b.bounds[axis + 2] * 0.5))
            });
            let right = segments.split_off(middle);
            Contents::Branch(Box::new((Self::build(segments), Self::build(right))))
        };
        Self { bounds, contents }
    }

    fn lower_bound(&self, point: Point) -> f32 {
        point.distance(Point {
            x: point.x.clamp(self.bounds[0], self.bounds[2]),
            y: point.y.clamp(self.bounds[1], self.bounds[3]),
        })
    }

    fn search(&self, point: Point, best: &mut f32) {
        match &self.contents {
            Contents::Leaf(segments) => {
                for &segment in segments {
                    *best = best.min(segment.distance(point));
                }
            }
            Contents::Branch(children) => {
                let a = children.0.lower_bound(point);
                let b = children.1.lower_bound(point);
                let (first, second, first_bound, second_bound) = if a <= b {
                    (&children.0, &children.1, a, b)
                } else {
                    (&children.1, &children.0, b, a)
                };
                if first_bound <= *best {
                    first.search(point, best);
                }
                if second_bound <= *best {
                    second.search(point, best);
                }
            }
        }
    }
}

pub(super) struct SegmentIndex {
    root: Node,
}

impl SegmentIndex {
    pub(super) fn new(points: &[Point]) -> Self {
        Self {
            root: Node::build(
                points
                    .windows(2)
                    .map(|p| Segment::new(p[0], p[1]))
                    .collect(),
            ),
        }
    }

    pub(super) fn distance(&self, point: Point) -> f32 {
        let mut best = f32::INFINITY;
        self.root.search(point, &mut best);
        best
    }
}

#[cfg(test)]
include!("../tests/unit/geometry_distance.rs");
