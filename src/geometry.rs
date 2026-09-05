use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use rayon::prelude::*;
use serde::Serialize;

use crate::color::{delta_e2000, Lab};
use crate::hierarchy::HierarchicalTopology;
use crate::segment::Segmentation;
use crate::union_find::UnionFind;

#[path = "geometry_primitives.rs"]
mod geometry_primitives;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Clone, Debug)]
pub enum Primitive {
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    Circle {
        cx: f32,
        cy: f32,
        radius: f32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    },
}

#[derive(Clone, Debug)]
pub struct RegionGeometry {
    pub region: u32,
    pub loops: Vec<Vec<Point>>,
    pub path_data: String,
    /// Equivalent opaque painter-stack geometry with holes removed only when
    /// every covered owner is later in the final paint order.
    pub occlusion_path_data: Option<String>,
    pub primitive: Option<Primitive>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GeometrySummary {
    pub regions: usize,
    pub loops: usize,
    pub empty_regions: usize,
    pub empty_region_pixels: usize,
    pub largest_empty_region: usize,
    pub source_boundary_edges: usize,
    pub shared_boundary_edges: usize,
    pub simplified_vertices: usize,
    pub cubic_segments: usize,
    pub line_segments: usize,
    pub rectangles: usize,
    pub circles: usize,
    pub ellipses: usize,
    /// Hole contours omitted because every raster owner inside the hole is
    /// painted later.  The later opaque faces cover the underpaint exactly,
    /// including the existing shared-boundary overlap.
    pub covered_holes_removed: usize,
    /// Faces that could not be assembled exclusively from the canonical
    /// shared curves and therefore used the conservative grid fallback.
    pub shared_loop_fallbacks: usize,
    pub shared_loop_missing_edges: usize,
    pub shared_loop_discontinuities: usize,
    pub shared_loop_invalid_areas: usize,
    pub adaptive_optimal_polygons: usize,
    pub continuity_faired_masters: usize,
    pub regularized_corner_excursions: usize,
    pub regularized_corner_vertices: usize,
    /// Canonical curves replaced by their shared positioned-grid polyline
    /// after a whole-partition topology validation.
    pub shared_curve_downgrades: usize,
    /// Rounded shared Paint-graph nodes whose open-boundary degree exceeds
    /// two.  The structural graph reuses these exact topology coordinates.
    #[serde(skip)]
    pub paint_junctions: Vec<Point>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct GridEdge {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct EdgeKey(u64, u64);

impl EdgeKey {
    fn new(first: u64, second: u64) -> Self {
        if first < second {
            Self(first, second)
        } else {
            Self(second, first)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct RegionPair(i32, i32);

impl RegionPair {
    fn new(first: i32, second: i32) -> Self {
        if first < second {
            Self(first, second)
        } else {
            Self(second, first)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CurveSegment {
    Line {
        start: Point,
        end: Point,
    },
    Cubic {
        start: Point,
        first: Point,
        second: Point,
        end: Point,
    },
}

impl CurveSegment {
    fn start(self) -> Point {
        match self {
            Self::Line { start, .. } | Self::Cubic { start, .. } => start,
        }
    }

    fn end(self) -> Point {
        match self {
            Self::Line { end, .. } | Self::Cubic { end, .. } => end,
        }
    }

    fn reversed(self) -> Self {
        match self {
            Self::Line { start, end } => Self::Line {
                start: end,
                end: start,
            },
            Self::Cubic {
                start,
                first,
                second,
                end,
            } => Self::Cubic {
                start: end,
                first: second,
                second: first,
                end: start,
            },
        }
    }
}

fn interpolate_point(first: Point, second: Point, amount: f32) -> Point {
    Point {
        x: first.x + amount * (second.x - first.x),
        y: first.y + amount * (second.y - first.y),
    }
}

fn split_curve(segment: CurveSegment, amount: f64) -> (CurveSegment, CurveSegment) {
    let amount = amount.clamp(0.0, 1.0);
    let amount_f32 = amount as f32;
    match segment {
        CurveSegment::Line { start, end } => {
            let middle = interpolate_point(start, end, amount_f32);
            (
                CurveSegment::Line { start, end: middle },
                CurveSegment::Line { start: middle, end },
            )
        }
        CurveSegment::Cubic {
            start,
            first,
            second,
            end,
        } => {
            // Match NumPy's de Casteljau expression in `_cubic_interval`.
            // `A + t * (B - A)` is algebraically identical, but its float32
            // rounding changes a sliced master's endpoint tangent.
            // Python computes both scalar coefficients in f64, then NumPy
            // casts each weak scalar independently for its float32 array
            // multiply. In particular, `(1 - 2/3) as f32` differs from
            // `1f32 - (2/3) as f32` by one ulp.
            let inverse = (1.0 - amount) as f32;
            let coordinate = |left: f32, right: f32| {
                let left_term = inverse * left;
                let right_term = amount_f32 * right;
                left_term + right_term
            };
            let blend = |left: Point, right: Point| Point {
                x: coordinate(left.x, right.x),
                y: coordinate(left.y, right.y),
            };
            let first_edge = blend(start, first);
            let middle_edge = blend(first, second);
            let last_edge = blend(second, end);
            let first_face = blend(first_edge, middle_edge);
            let last_face = blend(middle_edge, last_edge);
            let middle = blend(first_face, last_face);
            (
                CurveSegment::Cubic {
                    start,
                    first: first_edge,
                    second: first_face,
                    end: middle,
                },
                CurveSegment::Cubic {
                    start: middle,
                    first: last_face,
                    second: last_edge,
                    end,
                },
            )
        }
    }
}

fn curve_interval(segment: CurveSegment, start: f64, end: f64) -> CurveSegment {
    let start = start.clamp(0.0, 1.0);
    let end = end.clamp(start, 1.0);
    if start <= 1e-8 && end >= 1.0 - 1e-8 {
        return segment;
    }
    let (before_end, _) = split_curve(segment, end);
    if start <= 1e-8 {
        return before_end;
    }
    split_curve(before_end, start / end.max(1e-8)).1
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveCurveSpan {
    master_id: usize,
    curve: CurveSegment,
    start_parameter: f64,
    end_parameter: f64,
}

#[derive(Clone, Debug, Default)]
struct AdaptiveBoundaryGeometry {
    edge_spans: HashMap<EdgeKey, Vec<AdaptiveCurveSpan>>,
    vertex_positions: VertexPositions,
    regularized_observations: VertexPositions,
    regularized_fixed_points: HashSet<u64>,
    regularized_excursions: usize,
    optimal_polygons: usize,
    continuity_faired_master_ids: HashSet<usize>,
}

type TaggedCurve = (f64, f64, usize, CurveSegment);

#[derive(Clone, Debug)]
struct SharedChain {
    points: Vec<Point>,
    segments: Vec<CurveSegment>,
    closed: bool,
}

type EdgeChainLookup = HashMap<EdgeKey, (usize, u64, u64)>;
type VertexTangents = HashMap<(u64, u64), Point>;
type VertexPositions = HashMap<u64, Point>;

fn vertex_id(x: usize, y: usize, stride: usize) -> u64 {
    (y * stride + x) as u64
}

fn point_from_vertex(value: u64, stride: usize) -> Point {
    Point {
        x: (value as usize % stride) as f32,
        y: (value as usize / stride) as f32,
    }
}

#[inline]
fn vertex_lex_key(value: u64, stride: usize) -> (usize, usize) {
    (value as usize % stride, value as usize / stride)
}

#[inline]
fn edge_lex_key(value: EdgeKey, stride: usize) -> ((usize, usize), (usize, usize)) {
    let first = vertex_lex_key(value.0, stride);
    let second = vertex_lex_key(value.1, stride);
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn is_canvas_vertex(vertex: u64, stride: usize, width: usize, height: usize) -> bool {
    let point = point_from_vertex(vertex, stride);
    point.x == 0.0 || point.y == 0.0 || point.x == width as f32 || point.y == height as f32
}

fn normalized(vector: Point) -> Point {
    let length = (vector.x * vector.x + vector.y * vector.y).sqrt();
    if length <= 1e-8 {
        Point::default()
    } else {
        Point {
            x: vector.x / length,
            y: vector.y / length,
        }
    }
}

fn trace_edge_chains(
    edges: &HashSet<EdgeKey>,
    forced_junctions: &HashSet<u64>,
    stride: usize,
) -> Vec<Vec<u64>> {
    let mut adjacency = HashMap::<u64, Vec<u64>>::new();
    for edge in edges {
        adjacency.entry(edge.0).or_default().push(edge.1);
        adjacency.entry(edge.1).or_default().push(edge.0);
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_by_key(|&value| vertex_lex_key(value, stride));
        neighbours.dedup();
    }
    let mut remaining = edges.clone();
    let mut chains = Vec::new();
    let trace =
        |start: u64, second: u64, remaining: &mut HashSet<EdgeKey>, chains: &mut Vec<Vec<u64>>| {
            let mut points = vec![start, second];
            remaining.remove(&EdgeKey::new(start, second));
            let (mut previous, mut current) = (start, second);
            while adjacency.get(&current).map(Vec::len) == Some(2)
                && !forced_junctions.contains(&current)
            {
                let Some(following) = adjacency.get(&current).and_then(|neighbours| {
                    neighbours.iter().copied().find(|&candidate| {
                        candidate != previous
                            && remaining.contains(&EdgeKey::new(current, candidate))
                    })
                }) else {
                    break;
                };
                points.push(following);
                remaining.remove(&EdgeKey::new(current, following));
                previous = current;
                current = following;
                if current == start {
                    break;
                }
            }
            chains.push(points);
        };
    let mut starts: Vec<u64> = adjacency
        .iter()
        .filter_map(|(&point, neighbours)| {
            (neighbours.len() != 2 || forced_junctions.contains(&point)).then_some(point)
        })
        .collect();
    starts.sort_by_key(|&value| vertex_lex_key(value, stride));
    for start in starts {
        let neighbours = adjacency.get(&start).cloned().unwrap_or_default();
        for second in neighbours {
            if remaining.contains(&EdgeKey::new(start, second)) {
                trace(start, second, &mut remaining, &mut chains);
            }
        }
    }
    while let Some(edge) = remaining
        .iter()
        .copied()
        .min_by_key(|&value| edge_lex_key(value, stride))
    {
        trace(edge.0, edge.1, &mut remaining, &mut chains);
    }
    chains
}

fn straight_cubic(start: Point, end: Point) -> CurveSegment {
    let delta = Point {
        x: end.x - start.x,
        y: end.y - start.y,
    };
    CurveSegment::Cubic {
        start,
        first: Point {
            x: start.x + delta.x / 3.0,
            y: start.y + delta.y / 3.0,
        },
        second: Point {
            x: start.x + 2.0 * delta.x / 3.0,
            y: start.y + 2.0 * delta.y / 3.0,
        },
        end,
    }
}

fn least_squares_line(points: &[Point]) -> (Point, Point) {
    if points.is_empty() {
        return (Point::default(), Point { x: 1.0, y: 0.0 });
    }
    let divisor = points.len() as f32;
    let centre = Point {
        x: points.iter().map(|point| point.x).sum::<f32>() / divisor,
        y: points.iter().map(|point| point.y).sum::<f32>() / divisor,
    };
    // `np.linalg.eigh(offsets.T @ offsets)` receives a float32 covariance
    // matrix.  Accumulating it in f64 changes a borderline half-pixel
    // straightness decision and, consequently, the optimal polygon nodes.
    let (mut xx, mut xy, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
    for point in points {
        let dx = point.x - centre.x;
        let dy = point.y - centre.y;
        // NumPy's 2xN @ Nx2 matmul kernel accumulates these short dot
        // products with float32 FMA. Separate multiply/add changes the fitted
        // line enough to move an adjusted polygon vertex by ~1e-4 px.
        xx = dx.mul_add(dx, xx);
        xy = dx.mul_add(dy, xy);
        yy = dy.mul_add(dy, yy);
    }
    let mut direction = if xx == yy && xy == 0.0 {
        // LAPACK returns the identity basis for a scalar 2x2 covariance and
        // NumPy selects the final column.
        Point { x: 0.0, y: 1.0 }
    } else {
        // `np.linalg.eigh` reaches LAPACK's DLAEV2 for this symmetric 2x2
        // matrix.  The common half-angle formula is mathematically equal but
        // not numerically equal: its last bits alter adjusted Potrace nodes,
        // then a later RDP pass can move an anchor by a whole raster sample.
        let (a, b, c) = (xx as f64, xy as f64, yy as f64);
        let sum = a + c;
        let difference = a - c;
        let absolute_difference = difference.abs();
        let twice_off_diagonal = b + b;
        let absolute_off_diagonal = twice_off_diagonal.abs();
        let (larger_diagonal, smaller_diagonal) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
        let radius = if absolute_difference > absolute_off_diagonal {
            absolute_difference
                * (1.0 + (absolute_off_diagonal / absolute_difference).powi(2)).sqrt()
        } else if absolute_difference < absolute_off_diagonal {
            absolute_off_diagonal
                * (1.0 + (absolute_difference / absolute_off_diagonal).powi(2)).sqrt()
        } else {
            absolute_off_diagonal * 2.0_f64.sqrt()
        };
        let (larger_eigenvalue, sign_eigenvalue) = if sum < 0.0 {
            (0.5 * (sum - radius), -1.0_f64)
        } else if sum > 0.0 {
            (0.5 * (sum + radius), 1.0_f64)
        } else {
            (0.5 * radius, 1.0_f64)
        };
        // Retain DLAEV2's evaluation of the other eigenvalue even though the
        // positive-semidefinite covariance always selects `larger_eigenvalue`.
        let _smaller_eigenvalue = if sum == 0.0 {
            -0.5 * radius
        } else {
            (larger_diagonal / larger_eigenvalue) * smaller_diagonal - (b / larger_eigenvalue) * b
        };
        let (candidate_cosine, sign_difference) = if difference >= 0.0 {
            (difference + radius, 1.0_f64)
        } else {
            (difference - radius, -1.0_f64)
        };
        let absolute_cosine = candidate_cosine.abs();
        let (mut cosine, mut sine) = if absolute_cosine > absolute_off_diagonal {
            let tangent = -twice_off_diagonal / candidate_cosine;
            let sine = 1.0 / (1.0 + tangent * tangent).sqrt();
            (tangent * sine, sine)
        } else if absolute_off_diagonal == 0.0 {
            (1.0, 0.0)
        } else {
            let tangent = -candidate_cosine / twice_off_diagonal;
            let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
            (cosine, tangent * cosine)
        };
        if sign_eigenvalue == sign_difference {
            (cosine, sine) = (-sine, cosine);
        }
        Point {
            x: cosine as f32,
            y: sine as f32,
        }
    };
    // The Rust callers store a directed tangent, whereas the line itself is
    // sign-invariant. Preserve their established source-walk orientation.
    if direction.x * (points[points.len() - 1].x - points[0].x)
        + direction.y * (points[points.len() - 1].y - points[0].y)
        < 0.0
    {
        direction.x = -direction.x;
        direction.y = -direction.y;
    }
    (centre, normalized(direction))
}

fn potrace_straight_subpath(points: &[Point], start: usize, end: usize, corridor: f32) -> bool {
    if end <= start + 2 {
        return true;
    }
    let mut directions = HashSet::<(i8, i8)>::new();
    for pair in points[start..=end].windows(2) {
        let direction = |delta: f32| -> i8 {
            if delta > 0.0 {
                1
            } else if delta < 0.0 {
                -1
            } else {
                0
            }
        };
        directions.insert((
            direction(pair[1].x - pair[0].x),
            direction(pair[1].y - pair[0].y),
        ));
    }
    if directions.len() == 4 {
        return false;
    }
    let midpoints: Vec<Point> = points[start..=end]
        .windows(2)
        .map(|pair| Point {
            x: 0.5 * (pair[0].x + pair[1].x),
            y: 0.5 * (pair[0].y + pair[1].y),
        })
        .collect();
    if midpoints.len() <= 2 {
        return true;
    }
    let (centre, direction) = least_squares_line(&midpoints);
    let spread = midpoints
        .iter()
        .map(|point| {
            let offset = Point {
                x: point.x - centre.x,
                y: point.y - centre.y,
            };
            let projection = offset.x * direction.x + offset.y * direction.y;
            projection * projection
        })
        .sum::<f32>();
    if spread <= 1e-8 {
        return false;
    }
    let normal = Point {
        x: -direction.y,
        y: direction.x,
    };
    let mut previous_projection = f32::NEG_INFINITY;
    for point in &midpoints {
        let offset = Point {
            x: point.x - centre.x,
            y: point.y - centre.y,
        };
        let projection = offset.x * direction.x + offset.y * direction.y;
        if projection < previous_projection - 1e-4
            || (offset.x * normal.x + offset.y * normal.y).abs() > corridor.max(0.0)
        {
            return false;
        }
        previous_projection = projection;
    }
    true
}

/// Potrace's lexicographic dynamic program: first minimize the number of
/// straight intervals inside the raster-edge corridor, then their normal
/// least-squares penalty.  A greedy longest-prefix pass is not equivalent and
/// was the reason shallow car silhouettes retained avoidable step anchors.
fn potrace_optimal_polygon(points: &[Point], corridor: f32) -> Vec<usize> {
    let count = points.len();
    if count <= 2 {
        return (0..count).collect();
    }
    let mut prefix = vec![[0.0_f64; 5]; count + 1];
    for (index, point) in points.iter().enumerate() {
        let x = point.x as f64;
        let y = point.y as f64;
        let terms = [x, y, x * x, x * y, y * y];
        for axis in 0..5 {
            prefix[index + 1][axis] = prefix[index][axis] + terms[axis];
        }
    }
    let mut furthest = vec![0_usize; count - 1];
    for (start, furthest_value) in furthest.iter_mut().enumerate() {
        let mut last = start + 1;
        let mut step = 2_usize;
        let mut failed = count;
        while start + step < count {
            let candidate = start + step;
            if !potrace_straight_subpath(points, start, candidate, corridor) {
                failed = candidate;
                break;
            }
            last = candidate;
            step *= 2;
        }
        if failed == count {
            let candidate = count - 1;
            if potrace_straight_subpath(points, start, candidate, corridor) {
                last = candidate;
            } else {
                failed = candidate;
            }
        }
        let mut low = last + 1;
        let mut high = failed.saturating_sub(1);
        while low <= high {
            let candidate = low + (high - low) / 2;
            if potrace_straight_subpath(points, start, candidate, corridor) {
                last = candidate;
                low = candidate + 1;
            } else if candidate == 0 {
                break;
            } else {
                high = candidate - 1;
            }
        }
        *furthest_value = last;
    }

    let mut segment_count = vec![usize::MAX; count];
    let mut total_penalty = vec![f64::INFINITY; count];
    let mut predecessor = vec![usize::MAX; count];
    segment_count[0] = 0;
    total_penalty[0] = 0.0;
    for start in 0..count - 1 {
        if start != 0 && predecessor[start] == usize::MAX {
            continue;
        }
        for end in start + 1..=furthest[start] {
            let chord_x = (points[end].x - points[start].x) as f64;
            let chord_y = (points[end].y - points[start].y) as f64;
            let length = chord_x.hypot(chord_y);
            let safe_length = length.max(1e-10);
            let samples = (end - start + 1) as f64;
            let mut mean = [0.0_f64; 5];
            for axis in 0..5 {
                mean[axis] = (prefix[end + 1][axis] - prefix[start][axis]) / samples;
            }
            let normal_x = -chord_y / safe_length;
            let normal_y = chord_x / safe_length;
            let mean_normal = normal_x * mean[0] + normal_y * mean[1];
            let mean_normal_squared = normal_x * normal_x * mean[2]
                + 2.0 * normal_x * normal_y * mean[3]
                + normal_y * normal_y * mean[4];
            let variance = (mean_normal_squared - mean_normal * mean_normal).max(0.0);
            let penalty = length * variance.sqrt();
            let candidate_count = segment_count[start] + 1;
            let candidate_penalty = total_penalty[start] + penalty;
            if candidate_count < segment_count[end]
                || (candidate_count == segment_count[end] && candidate_penalty < total_penalty[end])
            {
                segment_count[end] = candidate_count;
                total_penalty[end] = candidate_penalty;
                predecessor[end] = start;
            }
        }
    }
    let mut indices = vec![count - 1];
    while *indices.last().unwrap_or(&0) > 0 {
        let current = *indices.last().unwrap();
        let previous = if predecessor[current] == usize::MAX {
            current - 1
        } else {
            predecessor[current]
        };
        indices.push(previous);
    }
    indices.reverse();
    if cfg!(feature = "diagnostics")
        && std::env::var_os("PICVEC_STRAND_DIAGNOSTICS").is_some()
        && count == 261
        && points
            .iter()
            .any(|point| point.x == 1136.0 && point.y == 577.0)
    {
        eprintln!(
            "picvec target furthest: {:?}",
            &(130..215).map(|i| (i, furthest[i])).collect::<Vec<_>>()
        );
        eprintln!(
            "picvec target predecessor: {:?}",
            &(130..215)
                .map(|i| (i, predecessor[i], segment_count[i], total_penalty[i]))
                .collect::<Vec<_>>()
        );
    }
    indices
}

fn adjust_polygon_vertices(points: &[Point], indices: &[usize], closed: bool) -> Vec<Point> {
    if indices.len() < 2 {
        return Vec::new();
    }
    let polygon_indices = if closed {
        &indices[..indices.len() - 1]
    } else {
        indices
    };
    let lines: Vec<(Point, Point)> = indices
        .windows(2)
        .map(|span| least_squares_line(&points[span[0]..=span[1]]))
        .collect();
    let mut adjusted = Vec::with_capacity(polygon_indices.len());
    for (position, &point_index) in polygon_indices.iter().enumerate() {
        let raw = points[point_index];
        if !closed && (position == 0 || position + 1 == polygon_indices.len()) {
            adjusted.push(raw);
            continue;
        }
        let (first_point, first_direction) = lines[(position + lines.len() - 1) % lines.len()];
        let (second_point, second_direction) = lines[position % lines.len()];
        let denominator =
            first_direction.x * second_direction.y - first_direction.y * second_direction.x;
        let candidate = if denominator.abs() <= 1e-6 {
            raw
        } else {
            let offset = Point {
                x: second_point.x - first_point.x,
                y: second_point.y - first_point.y,
            };
            let distance =
                (offset.x * second_direction.y - offset.y * second_direction.x) / denominator;
            Point {
                x: first_point.x + distance * first_direction.x,
                y: first_point.y + distance * first_direction.y,
            }
        };
        adjusted.push(Point {
            x: candidate.x.clamp(raw.x - 0.5, raw.x + 0.5),
            y: candidate.y.clamp(raw.y - 0.5, raw.y + 0.5),
        });
    }
    adjusted
}

fn potrace_corner_curves(
    previous_midpoint: Point,
    vertex: Point,
    following_midpoint: Point,
    parameters: [f64; 3],
    alphamax: f32,
    next_master_id: &mut usize,
) -> Vec<TaggedCurve> {
    let [left_parameter, vertex_parameter, right_parameter] = parameters;
    let chord = Point {
        x: following_midpoint.x - previous_midpoint.x,
        y: following_midpoint.y - previous_midpoint.y,
    };
    let chord_length = chord.x.hypot(chord.y);
    if chord_length <= 1e-8 {
        return Vec::new();
    }
    let normal = Point {
        x: -chord.y / chord_length,
        y: chord.x / chord_length,
    };
    let distance = ((vertex.x - previous_midpoint.x) * normal.x
        + (vertex.y - previous_midpoint.y) * normal.y)
        .abs();
    let square_support = 0.5 * (normal.x.abs() + normal.y.abs());
    let gamma = (1.0 - square_support / distance.max(1e-8)).max(0.0);
    let mut alpha = 4.0 * gamma / 3.0;
    let mut tagged = |start: f64, end: f64, curve: CurveSegment| {
        let identifier = *next_master_id;
        *next_master_id += 1;
        (start, end, identifier, curve)
    };
    if alpha > alphamax {
        return vec![
            tagged(
                left_parameter,
                vertex_parameter,
                straight_cubic(previous_midpoint, vertex),
            ),
            tagged(
                vertex_parameter,
                right_parameter,
                straight_cubic(vertex, following_midpoint),
            ),
        ];
    }
    alpha = alpha.clamp(0.55, 1.0);
    vec![tagged(
        left_parameter,
        right_parameter,
        CurveSegment::Cubic {
            start: previous_midpoint,
            first: Point {
                x: previous_midpoint.x + alpha * (vertex.x - previous_midpoint.x),
                y: previous_midpoint.y + alpha * (vertex.y - previous_midpoint.y),
            },
            second: Point {
                x: following_midpoint.x + alpha * (vertex.x - following_midpoint.x),
                y: following_midpoint.y + alpha * (vertex.y - following_midpoint.y),
            },
            end: following_midpoint,
        },
    )]
}

fn potrace_master_curves(
    points: &[Point],
    closed: bool,
    corridor: f32,
    alphamax: f32,
    next_master_id: &mut usize,
    forced_vertices: &HashSet<usize>,
) -> (Vec<usize>, Vec<TaggedCurve>) {
    let mut indices = potrace_optimal_polygon(points, corridor);
    indices.extend(forced_vertices.iter().copied());
    indices.sort_unstable();
    indices.dedup();
    let mut adjusted = adjust_polygon_vertices(points, &indices, closed);
    let polygon_indices = if closed {
        &indices[..indices.len().saturating_sub(1)]
    } else {
        indices.as_slice()
    };
    for (position, &point_index) in polygon_indices.iter().enumerate() {
        if forced_vertices.contains(&point_index) {
            adjusted[position] = points[point_index];
        }
    }
    if adjusted.len() < 2 {
        return (indices, Vec::new());
    }
    let mut curves = Vec::new();
    if !closed {
        if adjusted.len() == 2 {
            let identifier = *next_master_id;
            *next_master_id += 1;
            curves.push((
                0.0,
                (points.len() - 1) as f64,
                identifier,
                straight_cubic(adjusted[0], adjusted[1]),
            ));
            return (indices, curves);
        }
        let first_midpoint = interpolate_point(adjusted[0], adjusted[1], 0.5);
        let identifier = *next_master_id;
        *next_master_id += 1;
        curves.push((
            indices[0] as f64,
            0.5 * (indices[0] + indices[1]) as f64,
            identifier,
            straight_cubic(adjusted[0], first_midpoint),
        ));
        for position in 1..adjusted.len() - 1 {
            curves.extend(potrace_corner_curves(
                interpolate_point(adjusted[position - 1], adjusted[position], 0.5),
                adjusted[position],
                interpolate_point(adjusted[position], adjusted[position + 1], 0.5),
                [
                    0.5 * (indices[position - 1] + indices[position]) as f64,
                    indices[position] as f64,
                    0.5 * (indices[position] + indices[position + 1]) as f64,
                ],
                if forced_vertices.contains(&indices[position]) {
                    -1.0
                } else {
                    alphamax
                },
                next_master_id,
            ));
        }
        let last = adjusted.len() - 1;
        let last_midpoint = interpolate_point(adjusted[last - 1], adjusted[last], 0.5);
        let identifier = *next_master_id;
        *next_master_id += 1;
        curves.push((
            0.5 * (indices[last - 1] + indices[last]) as f64,
            indices[last] as f64,
            identifier,
            straight_cubic(last_midpoint, adjusted[last]),
        ));
        return (indices, curves);
    }

    let period = (points.len() - 1) as f64;
    for (position, &vertex) in adjusted.iter().enumerate() {
        let previous = (position + adjusted.len() - 1) % adjusted.len();
        let following = (position + 1) % adjusted.len();
        let mut previous_index = polygon_indices[previous] as f64;
        let vertex_index = polygon_indices[position] as f64;
        let mut following_index = polygon_indices[following] as f64;
        if previous > position {
            previous_index -= period;
        }
        if following < position {
            following_index += period;
        }
        curves.extend(potrace_corner_curves(
            interpolate_point(adjusted[previous], vertex, 0.5),
            vertex,
            interpolate_point(vertex, adjusted[following], 0.5),
            [
                0.5 * (previous_index + vertex_index),
                vertex_index,
                0.5 * (vertex_index + following_index),
            ],
            if forced_vertices.contains(&(vertex_index as usize)) {
                -1.0
            } else {
                alphamax
            },
            next_master_id,
        ));
    }
    (indices, curves)
}

#[derive(Clone, Debug)]
struct RegularizedStrand {
    points: Vec<Point>,
    changed: HashSet<usize>,
    corners: HashSet<usize>,
    fixed: HashSet<usize>,
    reusable_polygon: bool,
}

#[derive(Clone, Debug)]
struct FittedBaseStrand {
    strand: Vec<u64>,
    raw: Vec<Point>,
    regularized: RegularizedStrand,
    polygon: Vec<usize>,
    curves: Vec<TaggedCurve>,
}

#[derive(Clone, Debug)]
struct ExcursionCandidate {
    excess: f32,
    area: f32,
    indices: Vec<usize>,
    start: Point,
    intersection: Point,
    end: Point,
}

fn point_segment_distance(point: Point, start: Point, end: Point) -> f32 {
    let direction = Point {
        x: end.x - start.x,
        y: end.y - start.y,
    };
    let length_squared = direction.x * direction.x + direction.y * direction.y;
    let parameter = if length_squared <= 1e-8 {
        0.0
    } else {
        (((point.x - start.x) * direction.x + (point.y - start.y) * direction.y) / length_squared)
            .clamp(0.0, 1.0)
    };
    point.distance(Point {
        x: start.x + parameter * direction.x,
        y: start.y + parameter * direction.y,
    })
}

fn regularize_short_corner_excursions(
    points: &[Point],
    edge_pairs: &[RegionPair],
    protected_vertices: &HashSet<u64>,
    stride: usize,
    corridor: f32,
) -> RegularizedStrand {
    if points.len() < 24 {
        return RegularizedStrand {
            points: points.to_vec(),
            changed: HashSet::new(),
            corners: HashSet::new(),
            fixed: HashSet::new(),
            reusable_polygon: false,
        };
    }
    let closed = points.len() > 2 && points.first() == points.last();
    let base = if closed {
        &points[..points.len() - 1]
    } else {
        points
    };
    let count = base.len();
    let uncertainty = corridor.max(0.25);
    let support_edges = (16.0 * uncertainty).ceil().max(8.0) as usize;
    let support_window = (32.0 * uncertainty).ceil().max(12.0) as usize;
    let minimum_detour_edges = (16.0 * uncertainty).ceil().max(8.0) as usize;
    let maximum_detour_edges = (64.0 * uncertainty).ceil().max(16.0) as usize;
    let maximum_ray = (32.0 * uncertainty).max(12.0);
    let minimum_excess = (8.0 * uncertainty).max(4.0);
    let maximum_area = (64.0 * uncertainty).max(12.0);
    let minimum_deviation = (3.0 * uncertainty).max(1.5);
    let straight_corridor = uncertainty.max(0.6);
    let offsets: Vec<usize> = if closed {
        let second = count / 2;
        if second == 0 {
            vec![0]
        } else {
            vec![0, second]
        }
    } else {
        vec![0]
    };
    let mut candidates = Vec::<ExcursionCandidate>::new();
    let mut reusable_polygon = None::<Vec<usize>>;
    for offset in offsets {
        let mut rotated = Vec::with_capacity(count);
        rotated.extend_from_slice(&base[offset..]);
        rotated.extend_from_slice(&base[..offset]);
        let rotated_pairs: Vec<RegionPair> = if closed {
            (0..count.saturating_sub(1))
                .map(|index| edge_pairs[(offset + index) % count])
                .collect()
        } else {
            edge_pairs.to_vec()
        };
        let polygon = potrace_optimal_polygon(&rotated, uncertainty);
        if cfg!(feature = "diagnostics")
            && std::env::var_os("PICVEC_STRAND_DIAGNOSTICS").is_some()
            && base
                .iter()
                .any(|point| point.x == 1136.0 && point.y == 577.0)
        {
            eprintln!("picvec target excursion polygon offset={offset}: {polygon:?}");
        }
        if !closed && offset == 0 {
            reusable_polygon = Some(polygon.clone());
        }
        if polygon.len() < 5 {
            continue;
        }
        for left in 1..polygon.len() - 3 {
            let maximum_right = (left + 7).min(polygon.len() - 1);
            for right in left + 3..maximum_right {
                let start_index = polygon[left];
                let end_index = polygon[right];
                let previous = polygon[left - 1];
                let following = polygon[right + 1];
                let detour_edges = end_index - start_index;
                if detour_edges < minimum_detour_edges || detour_edges > maximum_detour_edges {
                    continue;
                }
                if start_index - previous < support_edges || following - end_index < support_edges {
                    continue;
                }
                if end_index > rotated_pairs.len() {
                    continue;
                }
                if rotated_pairs[start_index..end_index]
                    .iter()
                    .any(|pair| *pair != rotated_pairs[start_index])
                {
                    continue;
                }
                let original_indices: Vec<usize> = (start_index..=end_index)
                    .map(|index| {
                        if closed {
                            (offset + index) % count
                        } else {
                            offset + index
                        }
                    })
                    .collect();
                if original_indices.iter().any(|&index| {
                    let point = base[index];
                    let vertex =
                        vertex_id(point.x.round() as usize, point.y.round() as usize, stride);
                    protected_vertices.contains(&vertex)
                }) {
                    continue;
                }
                let incoming_start = previous.max(start_index.saturating_sub(support_window));
                let outgoing_end = following.min(end_index + support_window);
                let incoming = &rotated[incoming_start..=start_index];
                let outgoing = &rotated[end_index..=outgoing_end];
                if incoming.len() < support_edges + 1 || outgoing.len() < support_edges + 1 {
                    continue;
                }
                if !potrace_straight_subpath(incoming, 0, incoming.len() - 1, straight_corridor)
                    || !potrace_straight_subpath(outgoing, 0, outgoing.len() - 1, straight_corridor)
                {
                    continue;
                }
                let incoming_midpoints: Vec<Point> = incoming
                    .windows(2)
                    .map(|pair| Point {
                        x: 0.5 * (pair[0].x + pair[1].x),
                        y: 0.5 * (pair[0].y + pair[1].y),
                    })
                    .collect();
                let outgoing_midpoints: Vec<Point> = outgoing
                    .windows(2)
                    .map(|pair| Point {
                        x: 0.5 * (pair[0].x + pair[1].x),
                        y: 0.5 * (pair[0].y + pair[1].y),
                    })
                    .collect();
                let (first_centre, mut first_direction) = least_squares_line(&incoming_midpoints);
                let (second_centre, mut second_direction) = least_squares_line(&outgoing_midpoints);
                let incoming_chord = Point {
                    x: incoming[incoming.len() - 1].x - incoming[0].x,
                    y: incoming[incoming.len() - 1].y - incoming[0].y,
                };
                if first_direction.x * incoming_chord.x + first_direction.y * incoming_chord.y < 0.0
                {
                    first_direction.x = -first_direction.x;
                    first_direction.y = -first_direction.y;
                }
                let outgoing_chord = Point {
                    x: outgoing[outgoing.len() - 1].x - outgoing[0].x,
                    y: outgoing[outgoing.len() - 1].y - outgoing[0].y,
                };
                if second_direction.x * outgoing_chord.x + second_direction.y * outgoing_chord.y
                    < 0.0
                {
                    second_direction.x = -second_direction.x;
                    second_direction.y = -second_direction.y;
                }
                let dot = (first_direction.x * second_direction.x
                    + first_direction.y * second_direction.y)
                    .clamp(-1.0, 1.0);
                let angle = (dot as f64).acos().to_degrees();
                if !(25.0..=155.0).contains(&angle) {
                    continue;
                }
                let denominator =
                    first_direction.x * second_direction.y - first_direction.y * second_direction.x;
                if denominator.abs() <= 1e-5 {
                    continue;
                }
                let centre_offset = Point {
                    x: second_centre.x - first_centre.x,
                    y: second_centre.y - first_centre.y,
                };
                let distance = (centre_offset.x * second_direction.y
                    - centre_offset.y * second_direction.x)
                    / denominator;
                let intersection = Point {
                    x: first_centre.x + distance * first_direction.x,
                    y: first_centre.y + distance * first_direction.y,
                };
                let project = |point: Point, line_point: Point, line_direction: Point| {
                    let parameter = (point.x - line_point.x) * line_direction.x
                        + (point.y - line_point.y) * line_direction.y;
                    Point {
                        x: line_point.x + parameter * line_direction.x,
                        y: line_point.y + parameter * line_direction.y,
                    }
                };
                let incoming_anchor = project(rotated[start_index], first_centre, first_direction);
                let outgoing_anchor = project(rotated[end_index], second_centre, second_direction);
                if incoming_anchor.distance(rotated[start_index]) > straight_corridor
                    || outgoing_anchor.distance(rotated[end_index]) > straight_corridor
                {
                    continue;
                }
                let forward_in = (intersection.x - incoming_anchor.x) * first_direction.x
                    + (intersection.y - incoming_anchor.y) * first_direction.y;
                let forward_out = (outgoing_anchor.x - intersection.x) * second_direction.x
                    + (outgoing_anchor.y - intersection.y) * second_direction.y;
                if forward_in < -1.0
                    || forward_out < -1.0
                    || forward_in > maximum_ray
                    || forward_out > maximum_ray
                {
                    continue;
                }
                let detour = &rotated[start_index..=end_index];
                let minimum_x = detour
                    .iter()
                    .map(|point| point.x)
                    .fold(f32::INFINITY, f32::min)
                    - uncertainty;
                let maximum_x = detour
                    .iter()
                    .map(|point| point.x)
                    .fold(f32::NEG_INFINITY, f32::max)
                    + uncertainty;
                let minimum_y = detour
                    .iter()
                    .map(|point| point.y)
                    .fold(f32::INFINITY, f32::min)
                    - uncertainty;
                let maximum_y = detour
                    .iter()
                    .map(|point| point.y)
                    .fold(f32::NEG_INFINITY, f32::max)
                    + uncertainty;
                if intersection.x < minimum_x
                    || intersection.x > maximum_x
                    || intersection.y < minimum_y
                    || intersection.y > maximum_y
                {
                    continue;
                }
                let raw_length = detour
                    .windows(2)
                    .map(|pair| pair[0].distance(pair[1]))
                    .sum::<f32>();
                let replacement_length =
                    incoming_anchor.distance(intersection) + outgoing_anchor.distance(intersection);
                let excess = raw_length - replacement_length;
                if excess < minimum_excess || excess < 0.35 * raw_length {
                    continue;
                }
                let mut polygon_points = detour.to_vec();
                polygon_points.extend([outgoing_anchor, intersection, incoming_anchor]);
                let mut twice_area = 0.0_f32;
                for index in 0..polygon_points.len() {
                    let first = polygon_points[index];
                    let second = polygon_points[(index + 1) % polygon_points.len()];
                    twice_area += first.x * second.y - first.y * second.x;
                }
                let area = 0.5 * twice_area.abs();
                let area_limit = if angle <= 120.0 {
                    maximum_area
                } else {
                    (48.0 * uncertainty).max(12.0)
                };
                if area > area_limit {
                    continue;
                }
                let mut deviation = 0.0_f32;
                for pair in detour.windows(2) {
                    let observation = Point {
                        x: 0.5 * (pair[0].x + pair[1].x),
                        y: 0.5 * (pair[0].y + pair[1].y),
                    };
                    deviation = deviation.max(
                        point_segment_distance(observation, incoming_anchor, intersection).min(
                            point_segment_distance(observation, intersection, outgoing_anchor),
                        ),
                    );
                }
                if deviation < minimum_deviation {
                    continue;
                }
                candidates.push(ExcursionCandidate {
                    excess,
                    area,
                    indices: original_indices,
                    start: incoming_anchor,
                    intersection,
                    end: outgoing_anchor,
                });
            }
        }
    }
    candidates.sort_by(|first, second| {
        second
            .excess
            .total_cmp(&first.excess)
            .then(first.area.total_cmp(&second.area))
    });
    let mut result = base.to_vec();
    let mut changed = HashSet::<usize>::new();
    let mut corners = HashSet::<usize>::new();
    let mut fixed = HashSet::<usize>::new();
    for candidate in candidates {
        if candidate
            .indices
            .iter()
            .any(|index| changed.contains(index))
        {
            continue;
        }
        let first_length = candidate.start.distance(candidate.intersection);
        let second_length = candidate.end.distance(candidate.intersection);
        let span = candidate.indices.len() - 1;
        let position =
            span as f64 * first_length as f64 / (first_length + second_length).max(1e-8) as f64;
        let corner_step = position.round_ties_even().clamp(1.0, (span - 1) as f64) as usize;
        corners.insert(candidate.indices[corner_step]);
        fixed.extend([
            candidate.indices[0],
            candidate.indices[corner_step],
            candidate.indices[span],
        ]);
        for (step, &index) in candidate.indices.iter().enumerate() {
            let position = if step <= corner_step {
                let parameter = step as f32 / corner_step as f32;
                Point {
                    x: candidate.start.x
                        + parameter * (candidate.intersection.x - candidate.start.x),
                    y: candidate.start.y
                        + parameter * (candidate.intersection.y - candidate.start.y),
                }
            } else {
                let parameter = (step - corner_step) as f32 / (span - corner_step) as f32;
                Point {
                    x: candidate.intersection.x
                        + parameter * (candidate.end.x - candidate.intersection.x),
                    y: candidate.intersection.y
                        + parameter * (candidate.end.y - candidate.intersection.y),
                }
            };
            result[index] = position;
            if position.distance(base[index]) > 1e-5 {
                changed.insert(index);
            }
        }
    }
    if closed {
        result.push(result[0]);
    }
    let reusable_polygon = reusable_polygon.is_some() && changed.is_empty();
    RegularizedStrand {
        points: result,
        changed,
        corners,
        fixed,
        reusable_polygon,
    }
}

fn boundary_topology(
    stride: usize,
    directed_edges: &[Vec<GridEdge>],
    pair_edges: &HashMap<RegionPair, Vec<EdgeKey>>,
) -> (VertexTangents, VertexPositions, Vec<Vec<u64>>, HashSet<u64>) {
    let mut remaining = BTreeSet::<EdgeKey>::new();
    for edges in pair_edges.values() {
        remaining.extend(edges.iter().copied());
    }
    let mut adjacency = HashMap::<u64, BTreeSet<u64>>::new();
    for &edge in &remaining {
        adjacency.entry(edge.0).or_default().insert(edge.1);
        adjacency.entry(edge.1).or_default().insert(edge.0);
    }

    // `_collect_label_edges`: one directed half-edge per owning face, with
    // that face on its right.  The exact insertion order is immaterial to
    // `_trace_region_loops`, which always starts from the lexicographically
    // smallest remaining directed edge.
    let mut votes = HashMap::<(u64, EdgeKey), f64>::new();
    for edges in directed_edges {
        for points in trace_region_vertex_loops(edges, stride) {
            if points.len() < 3 {
                continue;
            }
            let weight = ((points.len() as f64).sqrt() / 2.0).clamp(1.0, 12.0);
            for index in 0..points.len() {
                let point = points[index];
                let pair = EdgeKey::new(
                    points[(index + points.len() - 1) % points.len()],
                    points[(index + 1) % points.len()],
                );
                *votes.entry((point, pair)).or_default() += weight;
            }
        }
    }
    let mut continuation = HashMap::<(u64, u64), u64>::new();
    let mut tangents = VertexTangents::new();
    for (&vertex, neighbours) in &adjacency {
        let centre = point_from_vertex(vertex, stride);
        let pairing_score = |first: u64, second: u64| {
            let a = normalized(Point {
                x: point_from_vertex(first, stride).x - centre.x,
                y: point_from_vertex(first, stride).y - centre.y,
            });
            let b = normalized(Point {
                x: point_from_vertex(second, stride).x - centre.x,
                y: point_from_vertex(second, stride).y - centre.y,
            });
            votes
                .get(&(vertex, EdgeKey::new(first, second)))
                .copied()
                .unwrap_or(0.0)
                - 2.0 * (a.x * b.x + a.y * b.y) as f64
        };
        let mut ordered: Vec<u64> = neighbours.iter().copied().collect();
        ordered.sort_by_key(|&value| vertex_lex_key(value, stride));
        let pairings: Vec<(u64, u64)> = match ordered.as_slice() {
            [first, second] => vec![(*first, *second)],
            [first, second, third] => {
                let alternatives = [(*first, *second), (*first, *third), (*second, *third)];
                let mut best = alternatives[0];
                let mut best_score = pairing_score(best.0, best.1);
                for candidate in alternatives.into_iter().skip(1) {
                    let score = pairing_score(candidate.0, candidate.1);
                    if score > best_score {
                        best = candidate;
                        best_score = score;
                    }
                }
                vec![best]
            }
            [first, second, third, fourth] => {
                let alternatives = [
                    [(*first, *second), (*third, *fourth)],
                    [(*first, *third), (*second, *fourth)],
                    [(*first, *fourth), (*second, *third)],
                ];
                let score = |pairs: &[(u64, u64); 2]| {
                    pairing_score(pairs[0].0, pairs[0].1) + pairing_score(pairs[1].0, pairs[1].1)
                };
                let mut best = alternatives[0];
                let mut best_score = score(&best);
                for candidate in alternatives.into_iter().skip(1) {
                    let candidate_score = score(&candidate);
                    if candidate_score > best_score {
                        best = candidate;
                        best_score = candidate_score;
                    }
                }
                best.to_vec()
            }
            _ => {
                let mut unused = neighbours.clone();
                let mut pairs = Vec::new();
                while unused.len() >= 2 {
                    let mut ordered: Vec<u64> = unused.iter().copied().collect();
                    ordered.sort_by_key(|&value| vertex_lex_key(value, stride));
                    let mut selected = (ordered[0], ordered[1]);
                    let mut best_score = pairing_score(selected.0, selected.1);
                    for first_index in 0..ordered.len() {
                        for &second in &ordered[first_index + 1..] {
                            let candidate = (ordered[first_index], second);
                            let score = pairing_score(candidate.0, candidate.1);
                            if score > best_score {
                                selected = candidate;
                                best_score = score;
                            }
                        }
                    }
                    let (first, second) = selected;
                    unused.remove(&first);
                    unused.remove(&second);
                    pairs.push((first, second));
                }
                pairs
            }
        };
        for (first, second) in pairings {
            let first_point = point_from_vertex(first, stride);
            let second_point = point_from_vertex(second, stride);
            let tangent = normalized(Point {
                x: second_point.x - first_point.x,
                y: second_point.y - first_point.y,
            });
            continuation.insert((vertex, first), second);
            continuation.insert((vertex, second), first);
            tangents.insert(
                (vertex, first),
                Point {
                    x: -tangent.x,
                    y: -tangent.y,
                },
            );
            tangents.insert((vertex, second), tangent);
        }
    }

    // `_adaptive_boundary_strands` first consumes every open strand from a
    // lexicographically ordered snapshot, then consumes closed cycles.  This
    // ordering fixes master IDs and all later deterministic tie-breaks.
    let mut strands = Vec::<Vec<u64>>::new();
    let trace = |start: u64, following: u64, remaining: &mut BTreeSet<EdgeKey>| {
        let mut strand = vec![start, following];
        remaining.remove(&EdgeKey::new(start, following));
        let (mut previous, mut current) = (start, following);
        while let Some(&next) = continuation.get(&(current, previous)) {
            let edge = EdgeKey::new(current, next);
            if !remaining.remove(&edge) {
                break;
            }
            strand.push(next);
            previous = current;
            current = next;
            if current == start {
                break;
            }
        }
        strand
    };
    let mut ordered_edges: Vec<EdgeKey> = remaining.iter().copied().collect();
    ordered_edges.sort_by_key(|&value| edge_lex_key(value, stride));
    for edge in ordered_edges {
        if !remaining.contains(&edge) {
            continue;
        }
        if !continuation.contains_key(&(edge.0, edge.1)) {
            strands.push(trace(edge.0, edge.1, &mut remaining));
        } else if !continuation.contains_key(&(edge.1, edge.0)) {
            strands.push(trace(edge.1, edge.0, &mut remaining));
        }
    }
    while let Some(edge) = remaining
        .iter()
        .copied()
        .min_by_key(|&value| edge_lex_key(value, stride))
    {
        strands.push(trace(edge.0, edge.1, &mut remaining));
    }

    let positions: VertexPositions = adjacency
        .keys()
        .map(|&vertex| (vertex, point_from_vertex(vertex, stride)))
        .collect();
    let junctions = adjacency
        .iter()
        .filter_map(|(&vertex, neighbours)| (neighbours.len() != 2).then_some(vertex))
        .collect();
    (tangents, positions, strands, junctions)
}

fn direction(first: u64, second: u64, stride: usize) -> u8 {
    let a = point_from_vertex(first, stride);
    let b = point_from_vertex(second, stride);
    if b.x > a.x {
        0
    } else if b.y > a.y {
        1
    } else if b.x < a.x {
        2
    } else {
        3
    }
}

fn trace_region_vertex_loops(edges: &[GridEdge], stride: usize) -> Vec<Vec<u64>> {
    let mut outgoing = HashMap::<u64, Vec<u64>>::new();
    let mut remaining: HashSet<GridEdge> = edges.iter().copied().collect();
    for edge in &remaining {
        outgoing.entry(edge.start).or_default().push(edge.end);
    }
    for values in outgoing.values_mut() {
        values.sort_by_key(|&value| vertex_lex_key(value, stride));
    }
    let mut loops = Vec::new();
    while let Some(first) = remaining.iter().copied().min_by_key(|edge| {
        (
            vertex_lex_key(edge.start, stride),
            vertex_lex_key(edge.end, stride),
        )
    }) {
        let start = first.start;
        let mut current = first;
        let mut vertices = Vec::new();
        while remaining.remove(&current) {
            vertices.push(current.start);
            let vertex = current.end;
            if vertex == start {
                break;
            }
            let incoming = direction(current.start, current.end, stride);
            let Some(candidates) = outgoing.get(&vertex) else {
                break;
            };
            let selected = candidates
                .iter()
                .copied()
                .filter_map(|end| {
                    let edge = GridEdge { start: vertex, end };
                    remaining.contains(&edge).then_some(edge)
                })
                .min_by_key(|edge| {
                    let next_direction = direction(edge.start, edge.end, stride);
                    let turn = (next_direction + 4 - incoming) % 4;
                    match turn {
                        1 => 0,
                        0 => 1,
                        3 => 2,
                        _ => 3,
                    }
                });
            let Some(next) = selected else {
                break;
            };
            current = next;
        }
        if !vertices.is_empty() {
            loops.push(vertices);
        }
    }
    loops
}

fn perpendicular_distance(point: Point, first: Point, last: Point) -> f32 {
    let dx = last.x - first.x;
    let dy = last.y - first.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= 1e-12 {
        return point.distance(first);
    }
    let parameter =
        (((point.x - first.x) * dx + (point.y - first.y) * dy) / length_squared).clamp(0.0, 1.0);
    point.distance(Point {
        x: first.x + parameter * dx,
        y: first.y + parameter * dy,
    })
}

pub fn simplify_open(points: &[Point], tolerance: f32) -> Vec<Point> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut maximum = 0.0;
    let mut split = 0;
    for index in 1..points.len() - 1 {
        let distance = perpendicular_distance(points[index], points[0], points[points.len() - 1]);
        if distance > maximum {
            maximum = distance;
            split = index;
        }
    }
    if maximum <= tolerance {
        return vec![points[0], points[points.len() - 1]];
    }
    let mut first = simplify_open(&points[..=split], tolerance);
    let second = simplify_open(&points[split..], tolerance);
    first.pop();
    first.extend(second);
    first
}

fn resample_open_polyline(points: &[Point], spacing: f32) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    // NumPy computes each source norm/cumsum in float32, but concatenating
    // the leading Python literal promotes the interpolation abscissae to
    // float64. Preserve that mixed precision for exact porting parity.
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0_f64);
    let mut accumulated = 0.0_f32;
    for pair in points.windows(2) {
        let dx = pair[1].x - pair[0].x;
        let dy = pair[1].y - pair[0].y;
        accumulated += (dx * dx + dy * dy).sqrt();
        cumulative.push(accumulated as f64);
    }
    let length = *cumulative.last().unwrap_or(&0.0);
    if length <= 1e-6 {
        return vec![points[0]];
    }
    let count = ((length / spacing.max(0.1) as f64).ceil() as usize + 1).max(2);
    let mut result = Vec::with_capacity(count);
    let mut segment = 0_usize;
    for index in 0..count {
        // np.linspace(..., dtype=float32), followed by np.interp's float64
        // conversion of its query positions.
        let position = (length * index as f64 / (count - 1) as f64) as f32 as f64;
        while segment + 1 < cumulative.len() - 1 && cumulative[segment + 1] < position {
            segment += 1;
        }
        let span = (cumulative[segment + 1] - cumulative[segment]).max(1e-6_f64);
        let amount = ((position - cumulative[segment]) / span).clamp(0.0, 1.0);
        result.push(Point {
            x: (points[segment].x as f64
                + amount * (points[segment + 1].x - points[segment].x) as f64)
                as f32,
            y: (points[segment].y as f64
                + amount * (points[segment + 1].y - points[segment].y) as f64)
                as f32,
        });
    }
    result
}

fn gaussian_fair_points(points: &[Point], sigma: f32) -> Vec<Point> {
    if points.len() < 5 || sigma <= 1e-3 {
        return points.to_vec();
    }
    // scipy.ndimage.gaussian_filter1d builds its kernel in float64 with
    // radius=int(truncate*sigma + 0.5), then accumulates into a float32
    // destination for this source dtype.
    let radius = (4.0 * sigma as f64 + 0.5).floor() as isize;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|offset| {
            let value = offset as f64;
            -0.5 * value * value / ((sigma as f64) * sigma as f64)
        })
        .collect();
    // SciPy constructs ndimage's Gaussian kernel with scalar double-precision
    // libm. NumPy's dispatched SVML exp is correct for NumPy ufunc parity but
    // changes RDP decisions here after the Gaussian result is cast to f32.
    weights.iter_mut().for_each(|value| *value = value.exp());
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total.max(1e-12);
    }
    (0..points.len())
        .map(|index| {
            let mut x = 0.0_f64;
            let mut y = 0.0_f64;
            for (kernel_index, &weight) in weights.iter().enumerate() {
                let source = (index as isize + kernel_index as isize - radius)
                    .clamp(0, points.len() as isize - 1) as usize;
                x += weight * points[source].x as f64;
                y += weight * points[source].y as f64;
            }
            Point {
                x: x as f32,
                y: y as f32,
            }
        })
        .collect()
}

fn gaussian_smooth_points(points: &[Point], sigma: f32, wrap: bool) -> Vec<Point> {
    if points.len() < 5 || sigma <= 1e-3 {
        return points.to_vec();
    }
    let radius = (4.0 * sigma as f64 + 0.5).floor() as isize;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|offset| {
            let value = offset as f64;
            -0.5 * value * value / ((sigma as f64) * sigma as f64)
        })
        .collect();
    // Match scipy.ndimage's scalar double-precision kernel construction; see
    // `gaussian_fair_points` above.
    weights.iter_mut().for_each(|value| *value = value.exp());
    let total = weights.iter().sum::<f64>().max(1e-12);
    weights.iter_mut().for_each(|weight| *weight /= total);
    (0..points.len())
        .map(|index| {
            let mut x = 0.0_f64;
            let mut y = 0.0_f64;
            for (kernel_index, &weight) in weights.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let source = if wrap {
                    (index as isize + offset).rem_euclid(points.len() as isize) as usize
                } else {
                    (index as isize + offset).clamp(0, points.len() as isize - 1) as usize
                };
                x += weight * points[source].x as f64;
                y += weight * points[source].y as f64;
            }
            Point {
                x: x as f32,
                y: y as f32,
            }
        })
        .collect()
}

fn smooth_chain(points: &[Point], closed: bool, sigma: f32, fixed: &HashSet<usize>) -> Vec<Point> {
    let base = if closed && points.first() == points.last() {
        &points[..points.len() - 1]
    } else {
        points
    };
    let mut result = if sigma <= 0.0 || base.len() < 4 {
        base.to_vec()
    } else {
        gaussian_smooth_points(base, sigma, closed)
    };
    for &index in fixed {
        if index < result.len() {
            result[index] = base[index];
        }
    }
    if !closed && !result.is_empty() {
        result[0] = base[0];
        let last = result.len() - 1;
        result[last] = base[base.len() - 1];
    }
    if closed && !result.is_empty() {
        result.push(result[0]);
    }
    result
}

fn polyline_corner_indices(
    points: &[Point],
    closed: bool,
    tolerance: f32,
    angle_degrees: f32,
) -> HashSet<usize> {
    let base = if closed && points.first() == points.last() {
        &points[..points.len() - 1]
    } else {
        points
    };
    if base.len() < 3 {
        return HashSet::new();
    }
    let mut source = base.to_vec();
    if closed {
        source.push(base[0]);
    }
    let coarse = simplify_polyline(&source, (tolerance * 1.5).max(1.0), closed);
    let mut indices = Vec::<usize>::new();
    for point in coarse {
        let index = base
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first.distance(point).total_cmp(&second.distance(point))
            })
            .map(|value| value.0)
            .unwrap_or(0);
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    let mut corners = HashSet::new();
    if !closed {
        corners.extend([0, base.len() - 1]);
    }
    for position in 0..indices.len() {
        if !closed && (position == 0 || position + 1 == indices.len()) {
            continue;
        }
        let previous = base[indices[(position + indices.len() - 1) % indices.len()]];
        let current = base[indices[position]];
        let following = base[indices[(position + 1) % indices.len()]];
        let left = normalized(Point {
            x: previous.x - current.x,
            y: previous.y - current.y,
        });
        let right = normalized(Point {
            x: following.x - current.x,
            y: following.y - current.y,
        });
        let cosine = (left.x * right.x + left.y * right.y).clamp(-1.0, 1.0);
        if cosine.acos().to_degrees() <= angle_degrees {
            corners.insert(indices[position]);
        }
    }
    corners
}

fn chord_parameters(points: &[Point]) -> Vec<f32> {
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0_f32);
    for pair in points.windows(2) {
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + pair[0].distance(pair[1]));
    }
    let total = *cumulative.last().unwrap_or(&0.0);
    if total > 1e-8 {
        cumulative.iter_mut().for_each(|value| *value /= total);
        cumulative
    } else if points.len() <= 1 {
        vec![0.0; points.len()]
    } else {
        (0..points.len())
            .map(|index| index as f32 / (points.len() - 1) as f32)
            .collect()
    }
}

fn numpy_pairwise_sum_f32(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return values.iter().fold(-0.0_f32, |sum, &value| sum + value);
    }
    if values.len() <= 128 {
        let mut accumulators = [0.0_f32; 8];
        accumulators.copy_from_slice(&values[..8]);
        let mut index = 8;
        while index + 8 <= values.len() {
            for lane in 0..8 {
                accumulators[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut result = ((accumulators[0] + accumulators[1])
            + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        for &value in &values[index..] {
            result += value;
        }
        return result;
    }
    let mut split = values.len() / 2;
    split -= split % 8;
    numpy_pairwise_sum_f32(&values[..split]) + numpy_pairwise_sum_f32(&values[split..])
}

fn least_squares_cubic(
    points: &[Point],
    parameters: &[f32],
    left_tangent: Point,
    right_tangent: Point,
) -> CurveSegment {
    let start = points[0];
    let end = points[points.len() - 1];
    let inverses: Vec<f32> = parameters.iter().map(|&value| 1.0 - value).collect();
    let inverse_squared: Vec<f32> = inverses.iter().map(|&value| value * value).collect();
    let parameter_squared: Vec<f32> = parameters.iter().map(|&value| value * value).collect();
    let mut b0 = inverses.clone();
    let mut b3 = parameters.to_vec();
    crate::elementary::pow_f32_in_place(&mut b0, 3.0);
    crate::elementary::pow_f32_in_place(&mut b3, 3.0);
    let mut aa_values = Vec::with_capacity(points.len() * 2);
    let mut ab_values = Vec::with_capacity(points.len() * 2);
    let mut bb_values = Vec::with_capacity(points.len() * 2);
    let mut av_values = Vec::with_capacity(points.len() * 2);
    let mut bv_values = Vec::with_capacity(points.len() * 2);
    for (index, &point) in points.iter().enumerate() {
        let parameter = parameters[index];
        let inverse = inverses[index];
        let b1 = (3.0 * parameter) * inverse_squared[index];
        let b2 = (3.0 * parameter_squared[index]) * inverse;
        let a1 = Point {
            x: b1 * left_tangent.x,
            y: b1 * left_tangent.y,
        };
        let a2 = Point {
            x: b2 * right_tangent.x,
            y: b2 * right_tangent.y,
        };
        let residual = Point {
            x: point.x - (b0[index] + b1) * start.x - (b2 + b3[index]) * end.x,
            y: point.y - (b0[index] + b1) * start.y - (b2 + b3[index]) * end.y,
        };
        aa_values.extend([a1.x * a1.x, a1.y * a1.y]);
        ab_values.extend([a1.x * a2.x, a1.y * a2.y]);
        bb_values.extend([a2.x * a2.x, a2.y * a2.y]);
        av_values.extend([a1.x * residual.x, a1.y * residual.y]);
        bv_values.extend([a2.x * residual.x, a2.y * residual.y]);
    }
    let aa = numpy_pairwise_sum_f32(&aa_values);
    let ab = numpy_pairwise_sum_f32(&ab_values);
    let bb = numpy_pairwise_sum_f32(&bb_values);
    let av = numpy_pairwise_sum_f32(&av_values);
    let bv = numpy_pairwise_sum_f32(&bv_values);
    let determinant = aa as f64 * bb as f64 - ab as f64 * ab as f64;
    let (mut alpha_left, mut alpha_right) = if determinant.abs() > 1e-12 {
        (
            (av as f64 * bb as f64 - bv as f64 * ab as f64) / determinant,
            (aa as f64 * bv as f64 - ab as f64 * av as f64) / determinant,
        )
    } else {
        (0.0, 0.0)
    };
    let chord = start.distance(end) as f64;
    let minimum = chord * 1e-3;
    if alpha_left < minimum
        || alpha_right < minimum
        || alpha_left > chord * 3.0
        || alpha_right > chord * 3.0
    {
        alpha_left = chord / 3.0;
        alpha_right = chord / 3.0;
    }
    CurveSegment::Cubic {
        start,
        first: Point {
            x: start.x + left_tangent.x * alpha_left as f32,
            y: start.y + left_tangent.y * alpha_left as f32,
        },
        second: Point {
            x: end.x + right_tangent.x * alpha_right as f32,
            y: end.y + right_tangent.y * alpha_right as f32,
        },
        end,
    }
}

fn cubic_points(segment: CurveSegment, parameters: &[f32]) -> Vec<Point> {
    let CurveSegment::Cubic {
        start,
        first,
        second,
        end,
    } = segment
    else {
        return parameters
            .iter()
            .map(|&parameter| cubic_point(segment, parameter))
            .collect();
    };
    let inverses: Vec<f32> = parameters.iter().map(|&value| 1.0 - value).collect();
    let inverse_squared: Vec<f32> = inverses.iter().map(|&value| value * value).collect();
    let parameter_squared: Vec<f32> = parameters.iter().map(|&value| value * value).collect();
    let mut inverse_cubed = inverses.clone();
    let mut parameter_cubed = parameters.to_vec();
    crate::elementary::pow_f32_in_place(&mut inverse_cubed, 3.0);
    crate::elementary::pow_f32_in_place(&mut parameter_cubed, 3.0);
    (0..parameters.len())
        .map(|index| {
            let coordinate = |start: f32, first: f32, second: f32, end: f32| {
                let first_term = inverse_cubed[index] * start;
                let second_term = ((3.0 * inverse_squared[index]) * parameters[index]) * first;
                let third_term = ((3.0 * inverses[index]) * parameter_squared[index]) * second;
                let fourth_term = parameter_cubed[index] * end;
                ((first_term + second_term) + third_term) + fourth_term
            };
            Point {
                x: coordinate(start.x, first.x, second.x, end.x),
                y: coordinate(start.y, first.y, second.y, end.y),
            }
        })
        .collect()
}

fn fit_cubic_recursive(
    points: &[Point],
    left_tangent: Point,
    right_tangent: Point,
    tolerance_squared: f32,
) -> Vec<CurveSegment> {
    if points.len() == 2 {
        let distance = points[0].distance(points[1]) / 3.0;
        return vec![CurveSegment::Cubic {
            start: points[0],
            first: Point {
                x: points[0].x + left_tangent.x * distance,
                y: points[0].y + left_tangent.y * distance,
            },
            second: Point {
                x: points[1].x + right_tangent.x * distance,
                y: points[1].y + right_tangent.y * distance,
            },
            end: points[1],
        }];
    }
    let parameters = chord_parameters(points);
    let curve = least_squares_cubic(points, &parameters, left_tangent, right_tangent);
    let mut errors = Vec::with_capacity(points.len());
    let predicted = cubic_points(curve, &parameters);
    for (&point, predicted) in points.iter().zip(predicted) {
        let dx = predicted.x - point.x;
        let dy = predicted.y - point.y;
        errors.push(dx * dx + dy * dy);
    }
    let mut split = 0_usize;
    for index in 1..errors.len() {
        if errors[index] > errors[split] {
            split = index;
        }
    }
    if errors[split] <= tolerance_squared {
        return vec![curve];
    }
    if split == 0 || split + 1 == points.len() {
        split = points.len() / 2;
    }
    let mut centre_tangent = normalized(Point {
        x: points[split - 1].x - points[split + 1].x,
        y: points[split - 1].y - points[split + 1].y,
    });
    if centre_tangent.x == 0.0 && centre_tangent.y == 0.0 {
        centre_tangent = normalized(Point {
            x: points[split].x - points[split + 1].x,
            y: points[split].y - points[split + 1].y,
        });
    }
    let mut result = fit_cubic_recursive(
        &points[..=split],
        left_tangent,
        centre_tangent,
        tolerance_squared,
    );
    result.extend(fit_cubic_recursive(
        &points[split..],
        Point {
            x: -centre_tangent.x,
            y: -centre_tangent.y,
        },
        right_tangent,
        tolerance_squared,
    ));
    result
}

fn enforce_observed_coordinate_monotonicity(
    curve: CurveSegment,
    observations: &[Point],
) -> CurveSegment {
    let CurveSegment::Cubic {
        start,
        mut first,
        mut second,
        end,
    } = curve
    else {
        return curve;
    };
    let epsilon = 1e-5_f32;
    for axis in 0..2 {
        let coordinate = |point: Point| if axis == 0 { point.x } else { point.y };
        let nondecreasing = observations
            .windows(2)
            .all(|pair| coordinate(pair[1]) - coordinate(pair[0]) >= -epsilon);
        let nonincreasing = observations
            .windows(2)
            .all(|pair| coordinate(pair[1]) - coordinate(pair[0]) <= epsilon);
        if !nondecreasing && !nonincreasing {
            continue;
        }
        let start_value = coordinate(start);
        let end_value = coordinate(end);
        if (end_value - start_value).abs() <= epsilon {
            if axis == 0 {
                first.x = start_value;
                second.x = end_value;
            } else {
                first.y = start_value;
                second.y = end_value;
            }
            continue;
        }
        let sign = if end_value > start_value { 1.0 } else { -1.0 };
        if (sign > 0.0 && !nondecreasing) || (sign < 0.0 && !nonincreasing) {
            continue;
        }
        let mut transformed_first =
            (sign * coordinate(first)).clamp(sign * start_value, sign * end_value);
        let mut transformed_second =
            (sign * coordinate(second)).clamp(sign * start_value, sign * end_value);
        if transformed_first > transformed_second {
            let midpoint = 0.5 * (transformed_first + transformed_second);
            transformed_first = midpoint;
            transformed_second = midpoint;
        }
        if axis == 0 {
            first.x = sign * transformed_first;
            second.x = sign * transformed_second;
        } else {
            first.y = sign * transformed_first;
            second.y = sign * transformed_second;
        }
    }
    CurveSegment::Cubic {
        start,
        first,
        second,
        end,
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_smoothed_chain(
    smoothed: &[Point],
    raw: &[Point],
    closed: bool,
    smooth_closure: bool,
    fixed: &HashSet<usize>,
    tolerance: f32,
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
    monotone_anchors: &HashSet<usize>,
) -> Vec<CurveSegment> {
    let base = if closed {
        &smoothed[..smoothed.len() - 1]
    } else {
        smoothed
    };
    let raw_base = if closed { &raw[..raw.len() - 1] } else { raw };
    if base.len() < 2 {
        return Vec::new();
    }
    let (mut ordered, ordered_raw, split_indices, ordered_monotone) = if closed {
        let start = fixed.iter().copied().min().unwrap_or_else(|| {
            (0..base.len())
                .min_by(|&first, &second| {
                    base[first]
                        .y
                        .total_cmp(&base[second].y)
                        .then(base[first].x.total_cmp(&base[second].x))
                })
                .unwrap_or(0)
        });
        let mut ordered = base[start..].to_vec();
        ordered.extend_from_slice(&base[..=start]);
        let mut ordered_raw = raw_base[start..].to_vec();
        ordered_raw.extend_from_slice(&raw_base[..=start]);
        let mut splits: Vec<usize> = fixed
            .iter()
            .map(|&index| (index + base.len() - start) % base.len())
            .collect();
        splits.extend([0, base.len()]);
        splits.sort_unstable();
        splits.dedup();
        let monotone = monotone_anchors
            .iter()
            .map(|&index| (index + base.len() - start) % base.len())
            .collect();
        (ordered, ordered_raw, splits, monotone)
    } else {
        let mut splits: Vec<usize> = fixed.iter().copied().collect();
        splits.extend([0, base.len() - 1]);
        splits.sort_unstable();
        splits.dedup();
        (
            base.to_vec(),
            raw_base.to_vec(),
            splits,
            monotone_anchors.clone(),
        )
    };
    for &index in &split_indices {
        ordered[index] = ordered_raw[index];
    }
    let mut curves = Vec::new();
    let closure_tangent = smooth_closure.then(|| {
        normalized(Point {
            x: ordered[1].x - ordered[ordered.len() - 2].x,
            y: ordered[1].y - ordered[ordered.len() - 2].y,
        })
    });
    for pair in split_indices.windows(2) {
        let start_index = pair[0];
        let end_index = pair[1];
        let points = &ordered[start_index..=end_index];
        if points.len() < 2 {
            continue;
        }
        let left = if closed && start_index == 0 {
            closure_tangent.unwrap_or_else(|| {
                normalized(Point {
                    x: points[1].x - points[0].x,
                    y: points[1].y - points[0].y,
                })
            })
        } else if !closed && start_index == 0 {
            start_tangent.map(normalized).unwrap_or_else(|| {
                normalized(Point {
                    x: points[1].x - points[0].x,
                    y: points[1].y - points[0].y,
                })
            })
        } else {
            normalized(Point {
                x: points[1].x - points[0].x,
                y: points[1].y - points[0].y,
            })
        };
        let right = if closed && end_index == base.len() {
            closure_tangent
                .map(|value| Point {
                    x: -value.x,
                    y: -value.y,
                })
                .unwrap_or_else(|| {
                    normalized(Point {
                        x: points[points.len() - 2].x - points[points.len() - 1].x,
                        y: points[points.len() - 2].y - points[points.len() - 1].y,
                    })
                })
        } else if !closed && end_index + 1 == base.len() {
            end_tangent
                .map(|value| {
                    let value = normalized(value);
                    Point {
                        x: -value.x,
                        y: -value.y,
                    }
                })
                .unwrap_or_else(|| {
                    normalized(Point {
                        x: points[points.len() - 2].x - points[points.len() - 1].x,
                        y: points[points.len() - 2].y - points[points.len() - 1].y,
                    })
                })
        } else {
            normalized(Point {
                x: points[points.len() - 2].x - points[points.len() - 1].x,
                y: points[points.len() - 2].y - points[points.len() - 1].y,
            })
        };
        let mut fitted = geometry_primitives::fit(
            &ordered_raw[start_index..=end_index],
            tolerance.min(0.85),
            if !closed && start_index == 0 {
                start_tangent
            } else {
                None
            },
            if !closed && end_index + 1 == base.len() {
                end_tangent
            } else {
                None
            },
        )
        .unwrap_or_else(|| fit_cubic_recursive(points, left, right, tolerance.max(0.25).powi(2)));
        if ordered_monotone.contains(&start_index) || ordered_monotone.contains(&end_index) {
            let observations = &ordered_raw[start_index..=end_index];
            fitted = fitted
                .into_iter()
                .map(|curve| enforce_observed_coordinate_monotonicity(curve, observations))
                .collect();
        }
        curves.extend(fitted);
    }
    curves
}

#[allow(clippy::too_many_arguments)]
fn fit_shared_boundary_candidate(
    points: &[Point],
    closed: bool,
    tolerance: f32,
    smoothing_sigma: f32,
    corner_angle: f32,
    fixed_indices: &HashSet<usize>,
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
) -> Vec<CurveSegment> {
    let corner_probe = smooth_chain(
        points,
        closed,
        (smoothing_sigma * 2.0).max(1.0),
        &HashSet::new(),
    );
    let mut fixed = polyline_corner_indices(&corner_probe, closed, tolerance, corner_angle);
    fixed.extend(fixed_indices.iter().copied());
    // A closed shared chain must retain its topology-selected storage origin,
    // but that origin is not automatically a semantic corner. Remember the
    // distinction before adding the positional anchor: smooth contours need
    // one periodic tangent across this artificial seam.
    let smooth_closure = closed && !fixed.contains(&0);
    if closed {
        fixed.insert(0);
    }
    let smoothed = smooth_chain(points, closed, smoothing_sigma.max(0.0), &fixed);
    fit_smoothed_chain(
        &smoothed,
        points,
        closed,
        smooth_closure,
        &fixed,
        tolerance.max(0.25),
        start_tangent,
        end_tangent,
        fixed_indices,
    )
}

fn signed_supported_turn(points: &[Point], index: usize, support: usize) -> Option<f32> {
    if support == 0 || index < support || index + support >= points.len() {
        return None;
    }
    let before = Point {
        x: points[index].x - points[index - support].x,
        y: points[index].y - points[index - support].y,
    };
    let after = Point {
        x: points[index + support].x - points[index].x,
        y: points[index + support].y - points[index].y,
    };
    let before_length = before.x.hypot(before.y);
    let after_length = after.x.hypot(after.y);
    if before_length <= 1e-6 || after_length <= 1e-6 {
        return None;
    }
    let cross = before.x * after.y - before.y * after.x;
    let dot = before.x * after.x + before.y * after.y;
    Some(cross.atan2(dot))
}

fn persistent_open_corners(points: &[Point]) -> Vec<(usize, Point)> {
    let samples = resample_open_polyline(points, 2.0);
    if samples.len() < 5 {
        return Vec::new();
    }
    let local: Vec<f32> = (0..samples.len())
        .map(|index| signed_supported_turn(&samples, index, 2).unwrap_or(0.0))
        .collect();
    let coarse_support = 9_usize.min((samples.len().saturating_sub(1) / 2).max(2));
    let coarse: Vec<f32> = (0..samples.len())
        .map(|index| signed_supported_turn(&samples, index, coarse_support).unwrap_or(0.0))
        .collect();
    let mut result = Vec::new();
    for index in 0..samples.len() {
        let magnitude = local[index].abs();
        if magnitude < 65.0_f32.to_radians()
            || coarse[index].abs() < 45.0_f32.to_radians()
            || local[index] * coarse[index] <= 0.0
        {
            continue;
        }
        let first = index.saturating_sub(2);
        let last = (index + 2).min(samples.len() - 1);
        if (first..=last).any(|other| local[other].abs() > magnitude + 1e-6) {
            continue;
        }
        let (raw_index, point) = points
            .iter()
            .copied()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first
                    .distance(samples[index])
                    .total_cmp(&second.distance(samples[index]))
            })
            .unwrap_or((0, points[0]));
        if raw_index > 0
            && raw_index + 1 < points.len()
            && result
                .last()
                .map(|(previous, _)| *previous != raw_index)
                .unwrap_or(true)
        {
            result.push((raw_index, point));
        }
    }
    result
}

fn sample_open_catmull(points: &[Point], spacing: f32) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut result = vec![points[0]];
    for index in 0..points.len() - 1 {
        let current = points[index];
        let next = points[index + 1];
        let previous = if index == 0 {
            current
        } else {
            points[index - 1]
        };
        let after = if index + 2 >= points.len() {
            next
        } else {
            points[index + 2]
        };
        let smooth = is_smooth(previous, current, next) && is_smooth(current, next, after);
        let tension = 0.62 / 6.0;
        let first = Point {
            x: current.x + (next.x - previous.x) * tension,
            y: current.y + (next.y - previous.y) * tension,
        };
        let second = Point {
            x: next.x - (after.x - current.x) * tension,
            y: next.y - (after.y - current.y) * tension,
        };
        let estimated_length = if smooth {
            current.distance(first) + first.distance(second) + second.distance(next)
        } else {
            current.distance(next)
        };
        let steps = (estimated_length / spacing.max(0.1)).ceil().max(1.0) as usize;
        for step in 1..=steps {
            let amount = step as f32 / steps as f32;
            if smooth {
                let inverse = 1.0 - amount;
                result.push(Point {
                    x: inverse.powi(3) * current.x
                        + 3.0 * inverse.powi(2) * amount * first.x
                        + 3.0 * inverse * amount.powi(2) * second.x
                        + amount.powi(3) * next.x,
                    y: inverse.powi(3) * current.y
                        + 3.0 * inverse.powi(2) * amount * first.y
                        + 3.0 * inverse * amount.powi(2) * second.y
                        + amount.powi(3) * next.y,
                });
            } else {
                result.push(Point {
                    x: current.x + amount * (next.x - current.x),
                    y: current.y + amount * (next.y - current.y),
                });
            }
        }
    }
    result
}

fn nearest_sample_distances(
    query: &[Point],
    reference: &[Point],
    maximum: f32,
) -> Option<(f32, f32)> {
    if query.is_empty() || reference.is_empty() {
        return None;
    }
    let cell = maximum.max(0.25);
    let mut buckets = HashMap::<(i32, i32), Vec<Point>>::new();
    for &point in reference {
        buckets
            .entry((
                (point.x / cell).floor() as i32,
                (point.y / cell).floor() as i32,
            ))
            .or_default()
            .push(point);
    }
    let mut largest = 0.0_f32;
    let mut total = 0.0_f32;
    for &point in query {
        let key = (
            (point.x / cell).floor() as i32,
            (point.y / cell).floor() as i32,
        );
        let mut nearest = f32::INFINITY;
        for dy in -1_i32..=1 {
            for dx in -1_i32..=1 {
                if let Some(values) = buckets.get(&(key.0 + dx, key.1 + dy)) {
                    for &candidate in values {
                        nearest = nearest.min(point.distance(candidate));
                    }
                }
            }
        }
        if nearest > maximum {
            return None;
        }
        largest = largest.max(nearest);
        total += nearest;
    }
    Some((largest, total / query.len() as f32))
}

/// Replace long raster staircases with the simplest source-supported fair
/// centre-line. Candidates must cover both the baseline geometry and the
/// original detector chain in both directions within one raster-cell
/// diagonal, so this cannot shortcut a real corner or cross a narrow gap.
pub fn bounded_fairing_open(points: &[Point], tolerance: f32) -> Vec<Point> {
    let mut raw = Vec::with_capacity(points.len());
    for &point in points {
        if raw.last().copied() != Some(point) {
            raw.push(point);
        }
    }
    let baseline = simplify_open(&raw, tolerance.max(0.25));
    let length: f32 = raw.windows(2).map(|pair| pair[0].distance(pair[1])).sum();
    if raw.len() < 3 || baseline.len() < 3 || length < 16.0 {
        return baseline;
    }
    let reference = sample_open_catmull(&baseline, 0.25);
    let source = resample_open_polyline(&raw, 0.25);
    let source_corners = persistent_open_corners(&raw);
    let baseline_corner_error = source_corners
        .iter()
        .map(|(_, point)| {
            reference
                .iter()
                .map(|candidate| point.distance(*candidate))
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0_f32, f32::max);
    let fairing_input = resample_open_polyline(&reference, 2.0);
    let fairing_corners = persistent_open_corners(&reference);
    let maximum = std::f32::consts::SQRT_2;
    let scales = [
        0.5_f32, 0.659, 0.869, 1.145, 1.510, 1.991, 2.626, 3.463, 4.568, 6.0,
    ];
    let tolerances = [tolerance.max(0.35), maximum.min(1.0)];
    let mut best = baseline.clone();
    let mut best_key = (baseline.len() - 1, f32::INFINITY);
    for candidate_tolerance in tolerances {
        for sigma in scales {
            let smoothed = gaussian_fair_points(&fairing_input, sigma);
            let mut samples = fairing_input.clone();
            for index in 0..samples.len() {
                let turn = signed_supported_turn(&fairing_input, index, 2)
                    .unwrap_or(0.0)
                    .abs();
                let weight = (1.0 - turn / 65.0_f32.to_radians()).clamp(0.0, 1.0);
                samples[index].x += weight * (smoothed[index].x - samples[index].x);
                samples[index].y += weight * (smoothed[index].y - samples[index].y);
                let nearest = reference
                    .iter()
                    .copied()
                    .min_by(|first, second| {
                        samples[index]
                            .distance(*first)
                            .total_cmp(&samples[index].distance(*second))
                    })
                    .unwrap_or(samples[index]);
                let distance = samples[index].distance(nearest);
                if distance > candidate_tolerance {
                    let amount = candidate_tolerance / distance.max(1e-6);
                    samples[index] = Point {
                        x: nearest.x + amount * (samples[index].x - nearest.x),
                        y: nearest.y + amount * (samples[index].y - nearest.y),
                    };
                }
            }
            samples[0] = raw[0];
            let last = samples.len() - 1;
            samples[last] = *raw.last().unwrap();
            let mut anchors = simplify_open(&samples, candidate_tolerance.clamp(0.45, 1.0));
            let mut ordered: Vec<(usize, Point)> = anchors
                .drain(..)
                .map(|point| {
                    let index = samples
                        .iter()
                        .position(|candidate| *candidate == point)
                        .unwrap_or(0);
                    (index, point)
                })
                .collect();
            for (_, corner) in &fairing_corners {
                let index = samples
                    .iter()
                    .enumerate()
                    .min_by(|(_, first), (_, second)| {
                        first.distance(*corner).total_cmp(&second.distance(*corner))
                    })
                    .map(|value| value.0)
                    .unwrap_or(0);
                ordered.push((index, *corner));
            }
            ordered.sort_by_key(|value| value.0);
            ordered.dedup_by_key(|value| value.0);
            let candidate: Vec<Point> = ordered.into_iter().map(|value| value.1).collect();
            let complexity = candidate.len().saturating_sub(1);
            if complexity == 0 || complexity >= best_key.0 {
                continue;
            }
            let candidate_samples = sample_open_catmull(&candidate, 0.25);
            let Some((_, reference_to_candidate)) =
                nearest_sample_distances(&reference, &candidate_samples, maximum)
            else {
                continue;
            };
            let Some((_, candidate_to_reference)) =
                nearest_sample_distances(&candidate_samples, &reference, maximum)
            else {
                continue;
            };
            if nearest_sample_distances(&source, &candidate_samples, maximum).is_none()
                || nearest_sample_distances(&candidate_samples, &source, maximum).is_none()
            {
                continue;
            }
            if source_corners.iter().any(|(_, corner)| {
                candidate_samples
                    .iter()
                    .map(|point| point.distance(*corner))
                    .fold(f32::INFINITY, f32::min)
                    > baseline_corner_error + 0.125
            }) {
                continue;
            }
            let mean_error = 0.5 * (reference_to_candidate + candidate_to_reference);
            let key = (complexity, mean_error);
            if key.0 < best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1) {
                best = candidate;
                best_key = key;
            }
        }
    }
    best
}

fn structural_curve_path_data(curves: &[CurveSegment], closed: bool) -> String {
    let Some(first) = curves.first().map(|curve| curve.start()) else {
        return String::new();
    };
    let mut data = format!("M {} {}", fmt(first.x), fmt(first.y));
    for &curve in curves {
        let (first, second, end) = match curve {
            CurveSegment::Line { end, .. } => {
                data.push_str(&format!(" L {} {}", fmt(end.x), fmt(end.y)));
                continue;
            }
            CurveSegment::Cubic {
                first, second, end, ..
            } => (first, second, end),
        };
        data.push_str(&format!(
            " C {} {} {} {} {} {}",
            fmt(first.x),
            fmt(first.y),
            fmt(second.x),
            fmt(second.y),
            fmt(end.x),
            fmt(end.y),
        ));
    }
    if closed {
        data.push_str(" Z");
    }
    data
}

/// Fit a closed scalar-field contour without snapping it back to the raster
/// grid. Alpha mattes use this after marching-squares interpolation has
/// recovered the subpixel crossing of the coverage field. Running those
/// points through the ordinary Region geometry builder would discard that
/// crossing and recreate a pixel-edge staircase.
fn fit_alpha_contour(points: &[Point]) -> (Vec<Point>, Vec<CurveSegment>) {
    if points.len() < 3 {
        return (Vec::new(), Vec::new());
    }
    let mut source = points.to_vec();
    source.dedup_by(|left, right| left.distance(*right) <= 1e-5);
    if source.len() < 3 {
        return (Vec::new(), Vec::new());
    }
    if source[0].distance(*source.last().unwrap()) > 1e-5 {
        source.push(source[0]);
    } else {
        let first = source[0];
        *source.last_mut().unwrap() = first;
    }

    // Coverage isolines also contain raster stair steps (especially for binary
    // alpha). Fair them with the same corner-preserving fitter as colour
    // boundaries. Both the mask and its incident rim must use this exact fit.
    if source.len() <= 12 {
        // There is not enough support to distinguish a raster ripple from
        // the entire feature. Do not fair away tiny islands or holes.
        let curves = fit_shared_boundary_candidate(
            &source,
            true,
            0.3,
            0.0,
            100.0,
            &HashSet::new(),
            None,
            None,
        );
        return (source, curves);
    }
    // Keep sharp turns detected before fairing, including the tips of narrow
    // bands: a smoothed corner probe can otherwise erase their end caps.
    let corners = polyline_corner_indices(&source, true, 0.65, 100.0);
    let count = source.len() - 1;
    let mut fixed = corners.clone();
    for &corner in &corners {
        // Protect the small cap around a reversal as well as its vertex.
        let before = source[(corner + count - 3) % count];
        let after = source[(corner + 3) % count];
        if before.distance(after) < 3.0 {
            for offset in 0..=4.min(count - 1) {
                fixed.insert((corner + offset) % count);
                fixed.insert((corner + count - offset) % count);
            }
        }
    }
    let curves = fit_shared_boundary_candidate(&source, true, 0.65, 1.5, 100.0, &fixed, None, None);
    (source, curves)
}

pub(crate) fn fitted_alpha_contour_path_data(points: &[Point]) -> String {
    let (source, curves) = fit_alpha_contour(points);
    if source.is_empty() {
        return String::new();
    }
    if curves.is_empty() {
        let mut cubics = 0;
        let mut lines = 0;
        return closed_path_data(&source[..source.len() - 1], &mut cubics, &mut lines);
    }
    structural_curve_path_data(&curves, true)
}

/// Exact pieces of the same curve used by the alpha mask. Edge ink must not
/// independently refit this contour: subpixel disagreement creates dotted rims.
pub(crate) struct AlphaContourSpan {
    pub points: [Point; 3],
    pub path_data: String,
}

pub(crate) fn alpha_contour_spans(points: &[Point]) -> Vec<AlphaContourSpan> {
    let (source, mut curves) = fit_alpha_contour(points);
    if source.is_empty() {
        return Vec::new();
    }
    if curves.is_empty() {
        curves = source
            .windows(2)
            .map(|p| straight_cubic(p[0], p[1]))
            .collect();
    }
    let mut result = Vec::new();
    for curve in curves {
        let length: f32 = (0..8)
            .map(|i| {
                cubic_point(curve, i as f32 / 8.0)
                    .distance(cubic_point(curve, (i + 1) as f32 / 8.0))
            })
            .sum();
        let count = (length / 3.0).ceil().max(1.0) as usize;
        let mut remainder = curve;
        for i in 0..count {
            let (part, rest) = split_curve(remainder, 1.0 / (count - i) as f64);
            remainder = rest;
            result.push(AlphaContourSpan {
                points: [part.start(), cubic_point(part, 0.5), part.end()],
                path_data: structural_curve_path_data(&[part], false),
            });
        }
    }
    result
}

/// Port of the structural layer's constrained centre-line fitter followed by
/// its bounded fairing model selection.  The returned path retains exact
/// cubic controls; a knot-only representation cannot reproduce the Python
/// SVG because its fairing handles are not generic Catmull--Rom handles.
pub fn fitted_structural_open_path_data(
    points: &[Point],
    tolerance: f32,
    smoothing_sigma: f32,
) -> String {
    fitted_structural_open_path_data_with_tangents(points, tolerance, smoothing_sigma, None, None)
}

fn constrain_structural_endpoint_tangents(
    curves: &mut [CurveSegment],
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
) {
    if let (Some(curve), Some(tangent)) = (curves.first_mut(), start_tangent.map(normalized)) {
        if tangent.x != 0.0 || tangent.y != 0.0 {
            if let CurveSegment::Cubic {
                start,
                ref mut first,
                ..
            } = curve
            {
                let handle = start.distance(*first);
                *first = Point {
                    x: start.x + handle * tangent.x,
                    y: start.y + handle * tangent.y,
                };
            }
        }
    }
    if let (Some(curve), Some(tangent)) = (curves.last_mut(), end_tangent.map(normalized)) {
        if tangent.x != 0.0 || tangent.y != 0.0 {
            if let CurveSegment::Cubic {
                ref mut second,
                end,
                ..
            } = curve
            {
                let handle = end.distance(*second);
                *second = Point {
                    x: end.x - handle * tangent.x,
                    y: end.y - handle * tangent.y,
                };
            }
        }
    }
}

/// Fit an editable centre-line while preserving a graph-level continuation
/// direction at either endpoint. The tangent directions follow the path from
/// start to end. Candidate controls are still accepted only when their dense
/// samples remain in the detector-chain corridor, so G1 regularization cannot
/// invent an unsupported hook at a junction.
pub(crate) fn fitted_structural_open_path_data_with_tangents(
    points: &[Point],
    tolerance: f32,
    smoothing_sigma: f32,
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
) -> String {
    if points.len() < 2 {
        return String::new();
    }
    let raw = points.to_vec();
    let closed = raw.len() > 2 && raw.first() == raw.last();
    if let Some(curves) = geometry_primitives::fit(
        &raw,
        tolerance.clamp(0.25, 0.85),
        start_tangent,
        end_tangent,
    ) {
        return structural_curve_path_data(&curves, closed);
    }
    let length_factor = (raw.len() as f32 / 128.0).clamp(1.0, 20.0 / 3.0);
    let tolerance_factor = (raw.len() as f32 / 256.0).clamp(1.0, 3.0);
    let effective_sigma = smoothing_sigma.max(0.0) * length_factor;
    let effective_tolerance = tolerance.max(0.25) * tolerance_factor;
    let fitted = fit_shared_boundary_candidate(
        &raw,
        closed,
        effective_tolerance,
        effective_sigma,
        70.0,
        &HashSet::new(),
        start_tangent,
        end_tangent,
    );
    let fitted_samples = sample_curve_sequence(&fitted, 0.35);
    let fitted_supported = !fitted.is_empty()
        && nearest_sample_distances(&fitted_samples, &raw, tolerance.max(0.25)).is_some()
        && nearest_sample_distances(&raw, &fitted_samples, tolerance.max(0.25)).is_some();
    let mut baseline = if fitted_supported {
        fitted
    } else if closed {
        simplify_polyline(&raw, tolerance.max(0.25), true)
            .windows(2)
            .map(|pair| straight_cubic(pair[0], pair[1]))
            .collect::<Vec<_>>()
    } else {
        simplify_open(&raw, tolerance.max(0.25))
            .windows(2)
            .map(|pair| straight_cubic(pair[0], pair[1]))
            .collect::<Vec<_>>()
    };
    let unconstrained_baseline = baseline.clone();
    constrain_structural_endpoint_tangents(&mut baseline, start_tangent, end_tangent);
    if !boundary_corridor_supported(&raw, &baseline, tolerance.max(0.25)) {
        baseline = unconstrained_baseline;
    }
    if baseline.is_empty() {
        return String::new();
    }
    let length = raw
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum::<f32>();
    if length < 16.0 || baseline.len() < 2 {
        return structural_curve_path_data(&baseline, closed);
    }
    let reference = sample_curve_sequence(&baseline, 0.35);
    if reference.len() < 2 {
        return structural_curve_path_data(&baseline, closed);
    }
    // The Python fairing candidate is an endpoint-preserving open-curve
    // model. Closed structural counters retain their already constrained
    // baseline and explicit Z closure.
    if closed {
        return structural_curve_path_data(&baseline, true);
    }
    let reference_samples = sample_polyline_segments(&reference, 0.25);
    let source_samples = sample_polyline_segments(&raw, 0.25);
    let source_corners = persistent_open_corners(&raw)
        .into_iter()
        .map(|value| value.1)
        .collect::<Vec<_>>();
    let baseline_corner_error = source_corners
        .iter()
        .map(|&point| nearest_point(&reference_samples, point).1)
        .fold(0.0_f32, f32::max);
    let maximum = std::f32::consts::SQRT_2;
    let mut best = baseline.clone();
    let mut best_key = (baseline.len(), f32::INFINITY);
    let scale_ratio = (6.0_f64 / 0.5).powf(1.0 / 9.0);
    let scales = (0..10)
        .map(|index| (0.5_f64 * scale_ratio.powi(index)) as f32)
        .collect::<Vec<_>>();
    let mut candidate_tolerances = vec![tolerance.max(0.35), maximum.min(1.0)];
    candidate_tolerances.sort_by(f32::total_cmp);
    candidate_tolerances.dedup_by(|first, second| *first == *second);
    for candidate_tolerance in candidate_tolerances {
        for &sigma in &scales {
            let (mut candidate, _, _) =
                fairing_candidate_segments(&reference, candidate_tolerance, sigma, &source_corners);
            constrain_structural_endpoint_tangents(&mut candidate, start_tangent, end_tangent);
            let candidate_samples = sample_curve_sequence(&candidate, 0.35);
            let complexity = candidate.len();
            if candidate.is_empty() || complexity >= best_key.0 {
                continue;
            }
            let Some((_, reference_to_candidate)) =
                nearest_sample_distances(&reference_samples, &candidate_samples, maximum)
            else {
                continue;
            };
            let Some((_, candidate_to_reference)) =
                nearest_sample_distances(&candidate_samples, &reference_samples, maximum)
            else {
                continue;
            };
            if nearest_sample_distances(&source_samples, &candidate_samples, maximum).is_none()
                || nearest_sample_distances(&candidate_samples, &source_samples, maximum).is_none()
                || source_corners.iter().any(|&point| {
                    nearest_point(&candidate_samples, point).1
                        > (baseline_corner_error + 0.125).max(0.25)
                })
            {
                continue;
            }
            let mean_error = 0.5 * (reference_to_candidate + candidate_to_reference);
            let key = (complexity, mean_error);
            if key.0 < best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1) {
                best = candidate;
                best_key = key;
            }
        }
    }
    let best = geometry_primitives::regularize(
        &raw,
        &best,
        tolerance.max(0.5),
        start_tangent,
        end_tangent,
    );
    structural_curve_path_data(&best, false)
}

fn sample_curve_sequence(curves: &[CurveSegment], spacing: f32) -> Vec<Point> {
    let mut result = Vec::new();
    for (index, &curve) in curves.iter().enumerate() {
        let control_length = match curve {
            CurveSegment::Line { start, end } => start.distance(end),
            CurveSegment::Cubic {
                start,
                first,
                second,
                end,
            } => start.distance(first) + first.distance(second) + second.distance(end),
        };
        let count = (control_length / spacing.max(0.1)).ceil().max(1.0) as usize;
        let parameters: Vec<f32> = (0..=count)
            .map(|step| (step as f64 / count as f64) as f32)
            .collect();
        let sampled = match curve {
            CurveSegment::Line { .. } => parameters
                .iter()
                .map(|&parameter| cubic_point(curve, parameter))
                .collect::<Vec<_>>(),
            CurveSegment::Cubic {
                start,
                first,
                second,
                end,
            } => {
                // NumPy evaluates the two `** 3` array operations through
                // its dispatched float32 power ufunc. Repeated scalar
                // multiplication differs by an ulp and, over a long master,
                // can shift an RDP anchor by a whole raster sample.
                let inverses: Vec<f32> = parameters.iter().map(|&value| 1.0 - value).collect();
                let inverse_squared: Vec<f32> =
                    inverses.iter().map(|&value| value * value).collect();
                let parameter_squared: Vec<f32> =
                    parameters.iter().map(|&value| value * value).collect();
                let mut inverse_cubed = inverses.clone();
                let mut parameter_cubed = parameters.clone();
                crate::elementary::pow_f32_in_place(&mut inverse_cubed, 3.0);
                crate::elementary::pow_f32_in_place(&mut parameter_cubed, 3.0);
                (0..parameters.len())
                    .map(|sample| {
                        let coordinate = |start: f32, first: f32, second: f32, end: f32| {
                            let first_term = inverse_cubed[sample] * start;
                            let second_term =
                                ((3.0_f32 * inverse_squared[sample]) * parameters[sample]) * first;
                            let third_term =
                                ((3.0_f32 * inverses[sample]) * parameter_squared[sample]) * second;
                            let fourth_term = parameter_cubed[sample] * end;
                            ((first_term + second_term) + third_term) + fourth_term
                        };
                        Point {
                            x: coordinate(start.x, first.x, second.x, end.x),
                            y: coordinate(start.y, first.y, second.y, end.y),
                        }
                    })
                    .collect()
            }
        };
        for (step, point) in sampled.into_iter().enumerate() {
            if index > 0 && step == 0 {
                continue;
            }
            result.push(point);
        }
    }
    result
}

fn sample_polyline_segments(points: &[Point], spacing: f32) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let spacing = spacing.max(0.1);
    let mut result = Vec::new();
    for (index, pair) in points.windows(2).enumerate() {
        let count = ((pair[0].distance(pair[1]) / spacing).ceil() as usize + 1).max(2);
        for step in 0..count {
            if index > 0 && step == 0 {
                continue;
            }
            let parameter = (step as f64 / (count - 1) as f64) as f32;
            result.push(Point {
                x: pair[0].x + parameter * (pair[1].x - pair[0].x),
                y: pair[0].y + parameter * (pair[1].y - pair[0].y),
            });
        }
    }
    result
}

fn boundary_corridor_supported(source: &[Point], curves: &[CurveSegment], maximum: f32) -> bool {
    if source.len() < 2 || curves.is_empty() {
        return false;
    }
    let source_samples = sample_polyline_segments(source, 0.25);
    let rendered = sample_curve_sequence(curves, 0.25);
    nearest_sample_distances(&source_samples, &rendered, maximum).is_some()
        && nearest_sample_distances(&rendered, &source_samples, maximum).is_some()
}

fn simplify_polyline(points: &[Point], tolerance: f32, closed: bool) -> Vec<Point> {
    if points.len() <= 2 || tolerance <= 0.0 {
        return points.to_vec();
    }
    let explicitly_closed = points.first() == points.last();
    if !closed && !explicitly_closed {
        return simplify_open(points, tolerance);
    }
    let base = if explicitly_closed {
        &points[..points.len() - 1]
    } else {
        points
    };
    if base.len() <= 3 {
        let mut result = base.to_vec();
        result.push(base[0]);
        return result;
    }
    let mut split = 0_usize;
    let mut maximum_distance = 0.0_f32;
    for index in 1..base.len() {
        let dx = base[index].x - base[0].x;
        let dy = base[index].y - base[0].y;
        let distance = dx * dx + dy * dy;
        if distance > maximum_distance {
            maximum_distance = distance;
            split = index;
        }
    }
    if split == 0 {
        let mut result = base.to_vec();
        result.push(base[0]);
        return result;
    }
    let first = simplify_open(&base[..=split], tolerance);
    let mut second_input = base[split..].to_vec();
    second_input.push(base[0]);
    let second = simplify_open(&second_input, tolerance);
    let mut result = first[..first.len() - 1].to_vec();
    result.extend(second);
    result
}

fn corridor_fallback_curves(
    source: &[Point],
    tolerance: f32,
    simplification_tolerance: Option<f32>,
) -> Vec<CurveSegment> {
    let closed = source.len() > 2 && source.first() == source.last();
    let maximum = tolerance.max(0.0);
    let legacy = 0.45_f32.min(maximum * 0.45);
    let requested = simplification_tolerance
        .map(|value| value.max(0.0).min(maximum))
        .unwrap_or(legacy);
    let mut attempts = vec![requested];
    if legacy != requested {
        attempts.push(legacy);
    }
    for simplification in attempts {
        let simplified = simplify_polyline(source, simplification, closed);
        if simplified.len() < 2 {
            continue;
        }
        let curves: Vec<CurveSegment> = simplified
            .windows(2)
            .map(|pair| straight_cubic(pair[0], pair[1]))
            .collect();
        if raster_boundary_supported(
            source,
            &sample_curve_sequence(&curves, 0.25),
            maximum + 1e-6,
        ) {
            return curves;
        }
    }
    source
        .windows(2)
        .map(|pair| straight_cubic(pair[0], pair[1]))
        .collect()
}

fn cubic_is_linear(curve: CurveSegment, tolerance: f32) -> bool {
    let CurveSegment::Cubic {
        start,
        first,
        second,
        end,
    } = curve
    else {
        return true;
    };
    let expected_first = interpolate_point(start, end, 1.0 / 3.0);
    let expected_second = interpolate_point(start, end, 2.0 / 3.0);
    first.distance(expected_first) <= tolerance && second.distance(expected_second) <= tolerance
}

fn fairing_candidate_segments(
    reference: &[Point],
    tolerance: f32,
    sigma: f32,
    preserved_corners: &[Point],
) -> (Vec<CurveSegment>, Vec<Point>, Vec<Point>) {
    let samples = resample_open_polyline(reference, 2.0);
    if samples.len() < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let corner_samples: Vec<(usize, Point)> = preserved_corners
        .iter()
        .copied()
        .map(|point| {
            let index = samples
                .iter()
                .enumerate()
                .min_by(|(_, first), (_, second)| {
                    first.distance(point).total_cmp(&second.distance(point))
                })
                .map(|value| value.0)
                .unwrap_or(0);
            (index, point)
        })
        .collect();
    let smoothed = gaussian_fair_points(&samples, sigma);
    let mut fair_samples: Vec<(f64, f64)> = (0..samples.len())
        .map(|index| {
            let persistent_corner = corner_samples
                .iter()
                .any(|&(corner, _)| corner.abs_diff(index) <= 2);
            let weight = if persistent_corner { 0.0 } else { 1.0 };
            (
                samples[index].x as f64 + weight * (smoothed[index].x - samples[index].x) as f64,
                samples[index].y as f64 + weight * (smoothed[index].y - samples[index].y) as f64,
            )
        })
        .collect();
    for index in 0..fair_samples.len() {
        let nearest = reference
            .iter()
            .copied()
            .min_by(|first, second| {
                let first_distance = (fair_samples[index].0 - first.x as f64)
                    .hypot(fair_samples[index].1 - first.y as f64);
                let second_distance = (fair_samples[index].0 - second.x as f64)
                    .hypot(fair_samples[index].1 - second.y as f64);
                first_distance.total_cmp(&second_distance)
            })
            .unwrap_or(samples[index]);
        let dx = fair_samples[index].0 - nearest.x as f64;
        let dy = fair_samples[index].1 - nearest.y as f64;
        let distance = dx.hypot(dy);
        if distance > tolerance as f64 {
            let amount = tolerance as f64 / distance.max(1e-6);
            fair_samples[index] = (
                nearest.x as f64 + amount * dx,
                nearest.y as f64 + amount * dy,
            );
        }
    }
    fair_samples[0] = (reference[0].x as f64, reference[0].y as f64);
    let last = fair_samples.len() - 1;
    fair_samples[last] = (
        reference[reference.len() - 1].x as f64,
        reference[reference.len() - 1].y as f64,
    );
    let samples: Vec<Point> = fair_samples
        .into_iter()
        .map(|(x, y)| Point {
            x: x as f32,
            y: y as f32,
        })
        .collect();
    let anchors = simplify_open(&samples, tolerance.clamp(0.45, 1.0));
    let mut anchor_by_index: BTreeMap<usize, Point> = anchors
        .into_iter()
        .map(|point| {
            let index = samples
                .iter()
                .position(|candidate| *candidate == point)
                .unwrap_or(0);
            (index, point)
        })
        .collect();
    for (index, corner) in corner_samples {
        // Python assigns into a dict here: a persistent raw corner replaces
        // an RDP anchor at the same fairing-sample index.
        anchor_by_index.insert(index, corner);
    }
    let anchors: Vec<Point> = anchor_by_index.into_values().collect();
    if anchors.len() < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut curves = Vec::with_capacity(anchors.len() - 1);
    for index in 0..anchors.len() - 1 {
        let start = anchors[index];
        let end = anchors[index + 1];
        let tangent_start = if index == 0 {
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }
        } else {
            Point {
                x: anchors[index + 1].x - anchors[index - 1].x,
                y: anchors[index + 1].y - anchors[index - 1].y,
            }
        };
        let tangent_end = if index + 1 == anchors.len() - 1 {
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }
        } else {
            Point {
                x: anchors[index + 2].x - anchors[index].x,
                y: anchors[index + 2].y - anchors[index].y,
            }
        };
        let segment = start.distance(end);
        let limit = (0.5 * segment).max(0.25);
        let start_length = (tangent_start.x.hypot(tangent_start.y) / 6.0).min(limit);
        let end_length = (tangent_end.x.hypot(tangent_end.y) / 6.0).min(limit);
        let start_direction = normalized(tangent_start);
        let end_direction = normalized(tangent_end);
        curves.push(CurveSegment::Cubic {
            start,
            first: Point {
                x: start.x + start_length * start_direction.x,
                y: start.y + start_length * start_direction.y,
            },
            second: Point {
                x: end.x - end_length * end_direction.x,
                y: end.y - end_length * end_direction.y,
            },
            end,
        });
    }
    // Python keeps two views of this candidate.  `candidate_path.points` is
    // sampled from the original float32 controls, while the curves returned
    // to the caller are parsed from the millipixel-formatted path string.
    // The baseline/corner distance checks use the former and the raster
    // corridor uses the latter.
    let rendered = sample_curve_sequence(&curves, 0.35);
    let rounded = curves.into_iter().map(round_curve_to_milli).collect();
    (rounded, rendered, anchors)
}

fn round_curve_to_milli(curve: CurveSegment) -> CurveSegment {
    let rounded = |value: f32| {
        format!("{:.3}", value as f64)
            .parse::<f32>()
            .unwrap_or(value)
    };
    let point = |value: Point| Point {
        x: rounded(value.x),
        y: rounded(value.y),
    };
    match curve {
        CurveSegment::Line { start, end } => CurveSegment::Line {
            start: point(start),
            end: point(end),
        },
        CurveSegment::Cubic {
            start,
            first,
            second,
            end,
        } => CurveSegment::Cubic {
            start: point(start),
            first: point(first),
            second: point(second),
            end: point(end),
        },
    }
}

fn raster_boundary_supported(source: &[Point], rendered: &[Point], maximum: f32) -> bool {
    if source.len() < 2 || rendered.len() < 2 {
        return false;
    }
    let observations: Vec<Point> = source
        .windows(2)
        .map(|pair| Point {
            x: 0.5 * (pair[0].x + pair[1].x),
            y: 0.5 * (pair[0].y + pair[1].y),
        })
        .collect();
    let source_samples = sample_polyline_segments(source, 0.25);
    nearest_sample_distances(&observations, rendered, maximum).is_some()
        && nearest_sample_distances(rendered, &source_samples, maximum).is_some()
}

fn nearest_point(reference: &[Point], point: Point) -> (usize, f32) {
    reference
        .iter()
        .copied()
        .enumerate()
        .map(|(index, candidate)| (index, point.distance(candidate)))
        .min_by(|first, second| first.1.total_cmp(&second.1))
        .unwrap_or((0, f32::INFINITY))
}

fn least_squares_fairing_shared_boundary(
    source: &[Point],
    baseline: &[CurveSegment],
) -> Vec<CurveSegment> {
    if source.len() < 2 || baseline.len() < 6 {
        return baseline.to_vec();
    }
    let reference = sample_curve_sequence(baseline, 0.35);
    if reference.len() < 3 {
        return baseline.to_vec();
    }
    let reference_samples = sample_polyline_segments(&reference, 0.25);
    let source_corners: Vec<Point> = persistent_open_corners(source)
        .into_iter()
        .map(|value| value.1)
        .collect();
    let baseline_corner_error = source_corners
        .iter()
        .map(|&point| nearest_point(&reference_samples, point).1)
        .fold(0.0_f32, f32::max);
    let mut corner_indices: Vec<usize> = source_corners
        .iter()
        .map(|&point| nearest_point(&reference, point).0)
        .collect();
    corner_indices.sort_unstable();
    corner_indices.dedup();
    let mut split_indices = corner_indices.clone();
    split_indices.extend([0, reference.len() - 1]);
    split_indices.sort_unstable();
    split_indices.dedup();
    let allowed_baseline = std::f32::consts::FRAC_1_SQRT_2 + 0.5;
    let allowed_source = std::f32::consts::SQRT_2.max(allowed_baseline);
    let mut best = baseline.to_vec();
    let mut best_error = f32::INFINITY;
    for sigma in [
        0.75_f32,
        1.230_503,
        2.018_850_3,
        3.312_268_5,
        5.434_342,
        8.915_966,
        14.628_164,
        24.0,
    ] {
        let mut smoothed = gaussian_smooth_points(&reference, sigma, false);
        smoothed[0] = reference[0];
        let last = smoothed.len() - 1;
        smoothed[last] = reference[reference.len() - 1];
        for &index in &corner_indices {
            smoothed[index] = reference[index];
        }
        for fitting_tolerance in [0.75_f32, 1.0, 1.25] {
            let mut candidate = Vec::<CurveSegment>::new();
            for span in split_indices.windows(2) {
                let part = &smoothed[span[0]..=span[1]];
                if part.len() < 2 {
                    continue;
                }
                let start_index = 2.min(part.len() - 1);
                let end_index = part.len().saturating_sub(3);
                let start_direction = normalized(Point {
                    x: part[start_index].x - part[0].x,
                    y: part[start_index].y - part[0].y,
                });
                let end_direction = normalized(Point {
                    x: part[end_index].x - part[part.len() - 1].x,
                    y: part[end_index].y - part[part.len() - 1].y,
                });
                candidate.extend(fit_cubic_recursive(
                    part,
                    start_direction,
                    end_direction,
                    fitting_tolerance * fitting_tolerance,
                ));
            }
            if candidate.is_empty() || candidate.len() >= best.len() {
                continue;
            }
            if candidate[0].start().distance(baseline[0].start()) > 1e-3
                || candidate[candidate.len() - 1]
                    .end()
                    .distance(baseline[baseline.len() - 1].end())
                    > 1e-3
            {
                continue;
            }
            let candidate_samples = sample_curve_sequence(&candidate, 0.35);
            let Some((_, reference_to_candidate)) =
                nearest_sample_distances(&reference_samples, &candidate_samples, allowed_baseline)
            else {
                continue;
            };
            let Some((_, candidate_to_reference)) =
                nearest_sample_distances(&candidate_samples, &reference_samples, allowed_baseline)
            else {
                continue;
            };
            if !raster_boundary_supported(
                source,
                &sample_curve_sequence(&candidate, 0.25),
                allowed_source,
            ) {
                continue;
            }
            if source_corners.iter().any(|&corner| {
                nearest_point(&candidate_samples, corner).1
                    > (baseline_corner_error + 0.125).max(0.25)
            }) {
                continue;
            }
            let mean_error = 0.5 * (reference_to_candidate + candidate_to_reference);
            if candidate.len() < best.len()
                || (candidate.len() == best.len() && mean_error < best_error)
            {
                best = candidate;
                best_error = mean_error;
            }
        }
    }
    best
}

fn bounded_fairing_shared_boundary(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
) -> Vec<CurveSegment> {
    if source.len() < 2 || baseline.len() < 2 {
        return baseline.to_vec();
    }
    let reference = sample_curve_sequence(baseline, 0.35);
    let length: f32 = reference
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    if length < 16.0 {
        return baseline.to_vec();
    }
    let reference_samples = sample_polyline_segments(&reference, 0.25);
    let source_corners = persistent_open_corners(source);
    let baseline_corner_error = source_corners
        .iter()
        .map(|(_, corner)| nearest_point(&reference_samples, *corner).1)
        .fold(0.0_f32, f32::max);
    let allowed_baseline = (tolerance + 0.5).max(std::f32::consts::FRAC_1_SQRT_2 + 0.5);
    // Chain vertices live on pixel corners, whereas the observed transition
    // occupies the adjoining pixel cells. Keep a quarter-pixel sampling
    // margin beyond the cell diagonal for a smooth replacement.
    let allowed_source = (tolerance + 0.75).max(fairing_raster_corridor());
    let mut best = baseline.to_vec();
    let mut best_error = f32::INFINITY;
    for sigma in [
        0.5_f32,
        0.658_990_3,
        0.868_536_5,
        1.144_714_2,
        1.508_711_2,
        1.988_452_1,
        2.620_741_4,
        3.454_086_3,
        4.552_419,
        6.0,
    ] {
        let preserved_corners = source_corners
            .iter()
            .map(|value| value.1)
            .collect::<Vec<_>>();
        let (candidate, candidate_samples, _) =
            fairing_candidate_segments(&reference, tolerance.max(0.35), sigma, &preserved_corners);
        if candidate.is_empty() || candidate.len() >= best.len() {
            continue;
        }
        if candidate[0].start().distance(baseline[0].start()) > 1e-3
            || candidate[candidate.len() - 1]
                .end()
                .distance(baseline[baseline.len() - 1].end())
                > 1e-3
        {
            continue;
        }
        let Some((_, reference_to_candidate)) =
            nearest_sample_distances(&reference_samples, &candidate_samples, allowed_baseline)
        else {
            continue;
        };
        let Some((_, candidate_to_reference)) =
            nearest_sample_distances(&candidate_samples, &reference_samples, allowed_baseline)
        else {
            continue;
        };
        if !raster_boundary_supported(
            source,
            &sample_curve_sequence(&candidate, 0.25),
            allowed_source,
        ) {
            continue;
        }
        if source_corners.iter().any(|(_, corner)| {
            nearest_point(&candidate_samples, *corner).1 > (baseline_corner_error + 0.125).max(0.25)
        }) {
            continue;
        }
        let mean_error = 0.5 * (reference_to_candidate + candidate_to_reference);
        if candidate.len() < best.len()
            || (candidate.len() == best.len() && mean_error < best_error)
        {
            best = candidate;
            best_error = mean_error;
        }
    }
    best
}

fn curve_with_start(curve: CurveSegment, start: Point) -> CurveSegment {
    let delta = Point {
        x: start.x - curve.start().x,
        y: start.y - curve.start().y,
    };
    match curve {
        CurveSegment::Line { end, .. } => CurveSegment::Line { start, end },
        CurveSegment::Cubic {
            first, second, end, ..
        } => CurveSegment::Cubic {
            start,
            first: Point {
                x: first.x + delta.x,
                y: first.y + delta.y,
            },
            second,
            end,
        },
    }
}

fn curve_with_end(curve: CurveSegment, end: Point) -> CurveSegment {
    let delta = Point {
        x: end.x - curve.end().x,
        y: end.y - curve.end().y,
    };
    match curve {
        CurveSegment::Line { start, .. } => CurveSegment::Line { start, end },
        CurveSegment::Cubic {
            start,
            first,
            second,
            ..
        } => CurveSegment::Cubic {
            start,
            first,
            second: Point {
                x: second.x + delta.x,
                y: second.y + delta.y,
            },
            end,
        },
    }
}

fn bounded_fairing_direct_shared_boundary(
    source: &[Point],
    baseline: &[CurveSegment],
    closed: bool,
    continuity_master: bool,
) -> Vec<CurveSegment> {
    if closed || source.len() < 2 || baseline.len() < if continuity_master { 2 } else { 3 } {
        return baseline.to_vec();
    }
    if continuity_master {
        // Same-material tone contours may move farther than a structural
        // boundary: their purpose is to remove a raster staircase, while the
        // high-resolution material/occlusion anchors remain unchanged.
        return bounded_fairing_shared_boundary(source, baseline, 1.25);
    }
    let travelled: f32 = source
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    let displacement = source[0].distance(source[source.len() - 1]);
    let source_corners = persistent_open_corners(source);
    if travelled < 40.0 {
        return baseline.to_vec();
    }
    let direct_support = (source_corners.is_empty() && displacement >= 32.0)
        || (travelled >= 64.0
            && displacement >= 48.0
            && displacement / travelled.max(1e-6) >= std::f32::consts::FRAC_1_SQRT_2);
    // The Python direct route calls the ordinary shared-boundary model
    // verbatim before comparing it with the least-squares family.  Keeping a
    // second, almost-identical search here changed the effective tolerance
    // (sqrt(1/2) instead of 0.75) and therefore rejected valid Python
    // candidates at the final selection stage.
    let mut candidate = bounded_fairing_shared_boundary(source, baseline, 0.75);
    let least_squares = least_squares_fairing_shared_boundary(source, baseline);
    if least_squares.len() < candidate.len() {
        candidate = least_squares;
    }
    if candidate.len() < baseline.len() && candidate.len() >= 6 {
        let mut refined = bounded_fairing_shared_boundary(source, &candidate, 0.75);
        let refined_least_squares = least_squares_fairing_shared_boundary(source, &candidate);
        if refined_least_squares.len() < refined.len() {
            refined = refined_least_squares;
        }
        let original_reference = sample_curve_sequence(baseline, 0.35);
        let source_corner_points: Vec<Point> = source_corners.iter().map(|value| value.1).collect();
        let original_corner_error = source_corner_points
            .iter()
            .map(|&point| nearest_point(&original_reference, point).1)
            .fold(0.0_f32, f32::max);
        let refined_samples = sample_curve_sequence(&refined, 0.35);
        if refined.len() < candidate.len()
            && raster_boundary_supported(
                source,
                &sample_curve_sequence(&refined, 0.25),
                fairing_raster_corridor(),
            )
            && source_corner_points.iter().all(|&point| {
                nearest_point(&refined_samples, point).1
                    <= (original_corner_error + 0.125).max(0.25)
            })
        {
            candidate = refined;
        }
    }
    let sufficient_reduction = if direct_support {
        4 * candidate.len() <= 3 * baseline.len()
    } else {
        4 * candidate.len() <= baseline.len()
    };
    if sufficient_reduction {
        candidate
    } else {
        baseline.to_vec()
    }
}

fn fairing_raster_corridor() -> f32 {
    std::f32::consts::SQRT_2 + 0.25
}

fn parameterize_curves_by_source_arclength(
    source: &[Point],
    curves: Vec<CurveSegment>,
    next_master_id: &mut usize,
) -> Vec<(f64, f64, usize, CurveSegment)> {
    if source.len() < 2 || curves.is_empty() {
        return Vec::new();
    }
    let mut source_lengths = Vec::with_capacity(source.len());
    source_lengths.push(0.0_f32);
    for pair in source.windows(2) {
        source_lengths
            .push(source_lengths.last().copied().unwrap_or(0.0) + pair[0].distance(pair[1]));
    }
    let curve_lengths: Vec<f32> = curves
        .iter()
        .map(|curve| {
            sample_curve_sequence(std::slice::from_ref(curve), 0.25)
                .windows(2)
                .map(|pair| pair[0].distance(pair[1]))
                .sum::<f32>()
        })
        .collect();
    let curve_total = curve_lengths.iter().sum::<f32>();
    let source_total = *source_lengths.last().unwrap_or(&0.0);
    if curve_total <= 1e-6 || source_total <= 1e-6 {
        return Vec::new();
    }
    let source_parameter = |distance: f32| {
        let distance = distance.clamp(0.0, source_total);
        let mut index = source_lengths.partition_point(|&value| value < distance);
        index = index.clamp(1, source_lengths.len() - 1);
        let start = source_lengths[index - 1];
        let end = source_lengths[index];
        (index - 1) as f64
            + (distance as f64 - start as f64) / (end as f64 - start as f64).max(1e-6)
    };
    let mut accumulated = 0.0_f32;
    curves
        .into_iter()
        .zip(curve_lengths)
        .map(|(curve, length)| {
            let start = source_parameter(accumulated / curve_total * source_total);
            accumulated += length;
            let end = source_parameter(accumulated / curve_total * source_total);
            let identifier = *next_master_id;
            *next_master_id += 1;
            (start, end, identifier, curve)
        })
        .collect()
}

fn remove_collinear(points: &[Point]) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut result = Vec::new();
    for index in 0..points.len() {
        let previous = points[(index + points.len() - 1) % points.len()];
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        let cross = (current.x - previous.x) * (next.y - current.y)
            - (current.y - previous.y) * (next.x - current.x);
        if cross.abs() > 1e-6 {
            result.push(current);
        }
    }
    result
}

/// Simplify a pixel-grid interface from the observations carried by its unit
/// edges, rather than from the artificial outer corners of the staircase.
/// A diagonal one-pixel raster line has edge midpoints within sqrt(1/8) px of
/// its continuous source line even though every grid corner is sqrt(1/2) px
/// away.  Treating the corners as independent samples is what preserved the
/// raster staircase in earlier Rust output.
fn simplify_grid_open(points: &[Point], tolerance: f32) -> Vec<Point> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let straight = |start: usize, end: usize| {
        if end <= start + 2 {
            return true;
        }
        let midpoints: Vec<Point> = (start..end)
            .map(|index| Point {
                x: 0.5 * (points[index].x + points[index + 1].x),
                y: 0.5 * (points[index].y + points[index + 1].y),
            })
            .collect();
        let divisor = midpoints.len() as f32;
        let centre = Point {
            x: midpoints.iter().map(|point| point.x).sum::<f32>() / divisor,
            y: midpoints.iter().map(|point| point.y).sum::<f32>() / divisor,
        };
        let (mut xx, mut xy, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
        for point in &midpoints {
            let dx = point.x - centre.x;
            let dy = point.y - centre.y;
            xx += dx * dx;
            xy += dx * dy;
            yy += dy * dy;
        }
        if xx + yy <= 1e-8 {
            return false;
        }
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        let mut direction = Point {
            x: angle.cos(),
            y: angle.sin(),
        };
        let endpoint = Point {
            x: midpoints[midpoints.len() - 1].x - midpoints[0].x,
            y: midpoints[midpoints.len() - 1].y - midpoints[0].y,
        };
        if direction.x * endpoint.x + direction.y * endpoint.y < 0.0 {
            direction.x = -direction.x;
            direction.y = -direction.y;
        }
        let normal = Point {
            x: -direction.y,
            y: direction.x,
        };
        let mut previous_projection = f32::NEG_INFINITY;
        for point in &midpoints {
            let offset = Point {
                x: point.x - centre.x,
                y: point.y - centre.y,
            };
            let projection = offset.x * direction.x + offset.y * direction.y;
            if projection < previous_projection - 1e-4
                || (offset.x * normal.x + offset.y * normal.y).abs() > tolerance
            {
                return false;
            }
            previous_projection = projection;
        }
        // The pair-specific curve still emits its fixed shared endpoints.
        // Keep that chord in the same uncertainty corridor as the physical
        // midpoint observations.
        midpoints
            .iter()
            .all(|&point| perpendicular_distance(point, points[start], points[end]) <= tolerance)
    };
    let mut simplified = vec![points[0]];
    let mut start = 0_usize;
    while start + 1 < points.len() {
        let mut last = start + 1;
        let mut step = 2_usize;
        let mut failed = points.len() - 1;
        while start + step < points.len() {
            let candidate = start + step;
            if !straight(start, candidate) {
                failed = candidate;
                break;
            }
            last = candidate;
            step *= 2;
        }
        if failed == points.len() - 1 && straight(start, points.len() - 1) {
            last = points.len() - 1;
        } else {
            let mut low = last + 1;
            let mut high = failed.saturating_sub(1);
            while low <= high {
                let middle = low + (high - low) / 2;
                if straight(start, middle) {
                    last = middle;
                    low = middle + 1;
                } else if middle == 0 {
                    break;
                } else {
                    high = middle - 1;
                }
            }
        }
        simplified.push(points[last]);
        start = last;
    }
    simplified
}

fn simplify_grid_closed(points: &[Point], tolerance: f32) -> Vec<Point> {
    let points = remove_collinear(points);
    if points.len() <= 4 {
        return points;
    }
    let first_index = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)))
        .map(|value| value.0)
        .unwrap_or(0);
    let first = points[first_index];
    let second_index = points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.distance(first).total_cmp(&b.distance(first)))
        .map(|value| value.0)
        .unwrap_or(points.len() / 2);
    let arc = |start: usize, end: usize| {
        let mut result = Vec::new();
        let mut index = start;
        loop {
            result.push(points[index]);
            if index == end {
                break;
            }
            index = (index + 1) % points.len();
        }
        result
    };
    let mut first_arc = simplify_grid_open(&arc(first_index, second_index), tolerance);
    let second_arc = simplify_grid_open(&arc(second_index, first_index), tolerance);
    first_arc.pop();
    first_arc.extend(second_arc.into_iter().take_while(|point| *point != first));
    remove_collinear(&first_arc)
}

fn preserve_closed_points(simplified: Vec<Point>, raw: &[Point], required: &[Point]) -> Vec<Point> {
    if simplified.is_empty() || raw.is_empty() {
        return simplified;
    }
    let start = raw
        .iter()
        .position(|point| *point == simplified[0])
        .unwrap_or(0);
    let mut indices: Vec<usize> = simplified
        .iter()
        .filter_map(|point| raw.iter().position(|candidate| candidate == point))
        .collect();
    indices.extend(
        raw.iter()
            .enumerate()
            .filter_map(|(index, point)| required.contains(point).then_some(index)),
    );
    indices.sort_by_key(|&index| (index + raw.len() - start) % raw.len());
    indices.dedup();
    remove_collinear(
        &indices
            .into_iter()
            .map(|index| raw[index])
            .collect::<Vec<_>>(),
    )
}

fn signed_area(points: &[Point]) -> f32 {
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            a.x * b.y - b.x * a.y
        })
        .sum::<f32>()
        * 0.5
}

fn is_smooth(previous: Point, current: Point, next: Point) -> bool {
    let a = Point {
        x: current.x - previous.x,
        y: current.y - previous.y,
    };
    let b = Point {
        x: next.x - current.x,
        y: next.y - current.y,
    };
    let al = (a.x * a.x + a.y * a.y).sqrt().max(1e-6);
    let bl = (b.x * b.x + b.y * b.y).sqrt().max(1e-6);
    let cosine = (a.x * b.x + a.y * b.y) / (al * bl);
    cosine > 0.35
}

fn fmt(value: f32) -> String {
    let rounded = (value * 1000.0).round() / 1000.0;
    if (rounded - rounded.round()).abs() < 1e-5 {
        format!("{:.0}", rounded)
    } else {
        format!("{:.3}", rounded).trim_end_matches('0').to_string()
    }
}

fn closed_path_data(points: &[Point], cubics: &mut usize, lines: &mut usize) -> String {
    if points.len() < 3 {
        return String::new();
    }
    let mut output = format!("M{} {}", fmt(points[0].x), fmt(points[0].y));
    let count = points.len();
    for index in 0..count {
        let current = points[index];
        let next = points[(index + 1) % count];
        let previous = points[(index + count - 1) % count];
        let after = points[(index + 2) % count];
        if is_smooth(previous, current, next) && is_smooth(current, next, after) && count >= 6 {
            let tension = 0.52 / 6.0;
            let c1 = Point {
                x: current.x + (next.x - previous.x) * tension,
                y: current.y + (next.y - previous.y) * tension,
            };
            let c2 = Point {
                x: next.x - (after.x - current.x) * tension,
                y: next.y - (after.y - current.y) * tension,
            };
            output.push_str(&format!(
                "C{} {},{} {},{} {}",
                fmt(c1.x),
                fmt(c1.y),
                fmt(c2.x),
                fmt(c2.y),
                fmt(next.x),
                fmt(next.y)
            ));
            *cubics += 1;
        } else {
            output.push_str(&format!("L{} {}", fmt(next.x), fmt(next.y)));
            *lines += 1;
        }
    }
    output.push('Z');
    output
}

fn cubic_point(segment: CurveSegment, parameter: f32) -> Point {
    match segment {
        CurveSegment::Line { start, end } => Point {
            x: start.x + parameter * (end.x - start.x),
            y: start.y + parameter * (end.y - start.y),
        },
        CurveSegment::Cubic {
            start,
            first,
            second,
            end,
        } => {
            let inverse = 1.0 - parameter;
            // Preserve NumPy's float32 ufunc evaluation order from
            // `_cubic_point`: each power/multiply produces a float32 array,
            // then the four terms are added from left to right.
            let inverse_squared = inverse * inverse;
            let inverse_cubed = inverse_squared * inverse;
            let parameter_squared = parameter * parameter;
            let parameter_cubed = parameter_squared * parameter;
            let coordinate = |start: f32, first: f32, second: f32, end: f32| {
                let first_term = inverse_cubed * start;
                let second_term = ((3.0_f32 * inverse_squared) * parameter) * first;
                let third_term = ((3.0_f32 * inverse) * parameter_squared) * second;
                let fourth_term = parameter_cubed * end;
                ((first_term + second_term) + third_term) + fourth_term
            };
            Point {
                x: coordinate(start.x, first.x, second.x, end.x),
                y: coordinate(start.y, first.y, second.y, end.y),
            }
        }
    }
}

fn region_boundary_edges(
    segmentation: &Segmentation,
    stride: usize,
    topology: Option<&HierarchicalTopology>,
) -> (Vec<Vec<GridEdge>>, usize) {
    let count = segmentation.regions.len();
    let mut edges = vec![Vec::<GridEdge>::new(); count];
    let mut shared = 0_usize;
    let Some(topology) = topology.filter(|value| value.dimensions_match(segmentation)) else {
        for y in 0..segmentation.height {
            for x in 0..segmentation.width {
                let index = y * segmentation.width + x;
                let label = segmentation.labels[index] as usize;
                let neighbours = [
                    (y > 0).then(|| segmentation.labels[index - segmentation.width]),
                    (x + 1 < segmentation.width).then(|| segmentation.labels[index + 1]),
                    (y + 1 < segmentation.height)
                        .then(|| segmentation.labels[index + segmentation.width]),
                    (x > 0).then(|| segmentation.labels[index - 1]),
                ];
                let vertices = [
                    (vertex_id(x, y, stride), vertex_id(x + 1, y, stride)),
                    (vertex_id(x + 1, y, stride), vertex_id(x + 1, y + 1, stride)),
                    (vertex_id(x + 1, y + 1, stride), vertex_id(x, y + 1, stride)),
                    (vertex_id(x, y + 1, stride), vertex_id(x, y, stride)),
                ];
                for side in 0..4 {
                    if neighbours[side] == Some(label as u32) {
                        continue;
                    }
                    edges[label].push(GridEdge {
                        start: vertices[side].0,
                        end: vertices[side].1,
                    });
                    shared += usize::from(neighbours[side].is_some());
                }
            }
        }
        return (edges, shared);
    };
    for cell in &topology.cells {
        let label = cell.label as usize;
        for x in cell.x..cell.x + cell.width {
            let above =
                (cell.y > 0).then(|| segmentation.labels[(cell.y - 1) * segmentation.width + x]);
            if above != Some(cell.label) {
                edges[label].push(GridEdge {
                    start: vertex_id(x, cell.y, stride),
                    end: vertex_id(x + 1, cell.y, stride),
                });
                shared += usize::from(above.is_some());
            }
            let bottom_y = cell.y + cell.height;
            let below = (bottom_y < segmentation.height)
                .then(|| segmentation.labels[bottom_y * segmentation.width + x]);
            if below != Some(cell.label) {
                edges[label].push(GridEdge {
                    start: vertex_id(x + 1, bottom_y, stride),
                    end: vertex_id(x, bottom_y, stride),
                });
                shared += usize::from(below.is_some());
            }
        }
        for y in cell.y..cell.y + cell.height {
            let left =
                (cell.x > 0).then(|| segmentation.labels[y * segmentation.width + cell.x - 1]);
            if left != Some(cell.label) {
                edges[label].push(GridEdge {
                    start: vertex_id(cell.x, y + 1, stride),
                    end: vertex_id(cell.x, y, stride),
                });
                shared += usize::from(left.is_some());
            }
            let right_x = cell.x + cell.width;
            let right = (right_x < segmentation.width)
                .then(|| segmentation.labels[y * segmentation.width + right_x]);
            if right != Some(cell.label) {
                edges[label].push(GridEdge {
                    start: vertex_id(right_x, y, stride),
                    end: vertex_id(right_x, y + 1, stride),
                });
                shared += usize::from(right.is_some());
            }
        }
    }
    (edges, shared)
}

fn pair_boundary_edges(
    segmentation: &Segmentation,
    stride: usize,
    topology: Option<&HierarchicalTopology>,
) -> HashMap<RegionPair, Vec<EdgeKey>> {
    let mut pairs = HashMap::<RegionPair, Vec<EdgeKey>>::new();
    if let Some(topology) = topology.filter(|value| value.dimensions_match(segmentation)) {
        for cell in &topology.cells {
            for x in cell.x..cell.x + cell.width {
                let above = if cell.y == 0 {
                    -1
                } else {
                    segmentation.labels[(cell.y - 1) * segmentation.width + x] as i32
                };
                let below = cell.label as i32;
                if above != below {
                    pairs
                        .entry(RegionPair::new(above, below))
                        .or_default()
                        .push(EdgeKey::new(
                            vertex_id(x, cell.y, stride),
                            vertex_id(x + 1, cell.y, stride),
                        ));
                }
                if cell.y + cell.height == segmentation.height {
                    pairs
                        .entry(RegionPair::new(below, -1))
                        .or_default()
                        .push(EdgeKey::new(
                            vertex_id(x, segmentation.height, stride),
                            vertex_id(x + 1, segmentation.height, stride),
                        ));
                }
            }
            for y in cell.y..cell.y + cell.height {
                let left = if cell.x == 0 {
                    -1
                } else {
                    segmentation.labels[y * segmentation.width + cell.x - 1] as i32
                };
                let right = cell.label as i32;
                if left != right {
                    pairs
                        .entry(RegionPair::new(left, right))
                        .or_default()
                        .push(EdgeKey::new(
                            vertex_id(cell.x, y, stride),
                            vertex_id(cell.x, y + 1, stride),
                        ));
                }
                if cell.x + cell.width == segmentation.width {
                    pairs
                        .entry(RegionPair::new(right, -1))
                        .or_default()
                        .push(EdgeKey::new(
                            vertex_id(segmentation.width, y, stride),
                            vertex_id(segmentation.width, y + 1, stride),
                        ));
                }
            }
        }
        return pairs;
    }
    for y in 0..=segmentation.height {
        for x in 0..segmentation.width {
            let above = if y == 0 {
                -1
            } else {
                segmentation.labels[(y - 1) * segmentation.width + x] as i32
            };
            let below = if y == segmentation.height {
                -1
            } else {
                segmentation.labels[y * segmentation.width + x] as i32
            };
            if above != below {
                pairs
                    .entry(RegionPair::new(above, below))
                    .or_default()
                    .push(EdgeKey::new(
                        vertex_id(x, y, stride),
                        vertex_id(x + 1, y, stride),
                    ));
            }
        }
    }
    for y in 0..segmentation.height {
        for x in 0..=segmentation.width {
            let left = if x == 0 {
                -1
            } else {
                segmentation.labels[y * segmentation.width + x - 1] as i32
            };
            let right = if x == segmentation.width {
                -1
            } else {
                segmentation.labels[y * segmentation.width + x] as i32
            };
            if left != right {
                pairs
                    .entry(RegionPair::new(left, right))
                    .or_default()
                    .push(EdgeKey::new(
                        vertex_id(x, y, stride),
                        vertex_id(x, y + 1, stride),
                    ));
            }
        }
    }
    pairs
}

fn oriented_curve_interval(segment: CurveSegment, start: f64, end: f64) -> CurveSegment {
    if end < start {
        curve_interval(segment, end, start).reversed()
    } else {
        curve_interval(segment, start, end)
    }
}

fn enforce_convex_cubic_controls(segment: CurveSegment) -> CurveSegment {
    let CurveSegment::Cubic {
        start,
        first,
        second,
        end,
    } = segment
    else {
        return segment;
    };
    let chord = Point {
        x: end.x - start.x,
        y: end.y - start.y,
    };
    let chord_length = chord.x.hypot(chord.y);
    if chord_length <= 1e-6 {
        return straight_cubic(start, end);
    }
    let direction = Point {
        x: chord.x / chord_length,
        y: chord.y / chord_length,
    };
    let constrain = |anchor: Point, handle: Point, forward: Point| {
        let delta = Point {
            x: handle.x - anchor.x,
            y: handle.y - anchor.y,
        };
        let length = delta.x.hypot(delta.y).min(0.75 * chord_length);
        let mut handle_direction = normalized(delta);
        if handle_direction.x * forward.x + handle_direction.y * forward.y <= 1e-4 {
            handle_direction = forward;
        }
        Point {
            x: anchor.x + length * handle_direction.x,
            y: anchor.y + length * handle_direction.y,
        }
    };
    CurveSegment::Cubic {
        start,
        first: constrain(start, first, direction),
        second: constrain(
            end,
            second,
            Point {
                x: -direction.x,
                y: -direction.y,
            },
        ),
        end,
    }
}

fn connected_adaptive_edge_geometry(
    edge_spans: &HashMap<EdgeKey, Vec<AdaptiveCurveSpan>>,
    start_vertex: u64,
    end_vertex: u64,
    at_end: bool,
    stride: usize,
) -> Option<(Point, Point)> {
    let edge = EdgeKey::new(start_vertex, end_vertex);
    let mut pieces = edge_spans.get(&edge)?.clone();
    if start_vertex != edge.0 {
        pieces = pieces
            .into_iter()
            .rev()
            .map(|piece| AdaptiveCurveSpan {
                start_parameter: piece.end_parameter,
                end_parameter: piece.start_parameter,
                ..piece
            })
            .collect();
    }
    let piece = if at_end {
        pieces.last().copied()
    } else {
        pieces.first().copied()
    }?;
    let interval = oriented_curve_interval(piece.curve, piece.start_parameter, piece.end_parameter);
    let (position, tangent) = match (at_end, interval) {
        (true, CurveSegment::Line { start, end }) => Some((
            end,
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            },
        )),
        (true, CurveSegment::Cubic { second, end, .. }) => Some((
            end,
            Point {
                x: end.x - second.x,
                y: end.y - second.y,
            },
        )),
        (false, CurveSegment::Line { start, end }) => Some((
            start,
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            },
        )),
        (false, CurveSegment::Cubic { start, first, .. }) => Some((
            start,
            Point {
                x: first.x - start.x,
                y: first.y - start.y,
            },
        )),
    }?;
    // A Potrace master is parameterized by raster-chain index. On a long
    // curve its geometric speed need not be uniform, so the endpoint of one
    // sliced unit edge can lie several pixels beyond the grid vertex that the
    // slice represents. Vertex proposals are bounded later, but continuity
    // fitting used to consume the unbounded endpoint here and could therefore
    // begin by travelling backwards. Use the same source-cell constraint at
    // both stages so adjacent fits share a feasible endpoint.
    let connection_vertex = if at_end { end_vertex } else { start_vertex };
    let origin = point_from_vertex(connection_vertex, stride);
    Some((bounded_vertex_position(origin, position), tangent))
}

fn bounded_vertex_position(origin: Point, position: Point) -> Point {
    let displacement = Point {
        x: position.x - origin.x,
        y: position.y - origin.y,
    };
    let distance = displacement.x.hypot(displacement.y);
    // Match the shared-vertex displacement budget used by the reference
    // implementation for the 0.5-pixel Paint fitting corridor.
    const MAXIMUM_SHIFT: f32 = 0.25;
    if distance <= MAXIMUM_SHIFT {
        position
    } else {
        Point {
            x: origin.x + MAXIMUM_SHIFT * displacement.x / distance,
            y: origin.y + MAXIMUM_SHIFT * displacement.y / distance,
        }
    }
}

fn contextual_continuity_classes(
    labs: &[Lab],
    adjacency: &[HashMap<usize, usize>],
    has_interior: &[bool],
) -> Vec<i32> {
    assert_eq!(labs.len(), adjacency.len());
    assert_eq!(labs.len(), has_interior.len());
    if labs.is_empty() {
        return Vec::new();
    }

    // Continuity is a property of neighbouring material faces, not of a
    // global hue/lightness bucket.  A global neutral bucket conflated bright
    // desaturated glass, dark trim, tyres, and polished metal; their real
    // interfaces then disappeared before the master-curve fit.  Seed one
    // component per durable face and join only directly adjacent durable
    // faces whose CIEDE2000 distance is within three just-noticeable
    // differences.
    const THREE_JND_DELTA_E: f32 = 6.9;
    let mut durable = UnionFind::new(labs.len());
    for label in 0..labs.len() {
        if !has_interior[label] {
            continue;
        }
        for &neighbour in adjacency[label].keys() {
            if neighbour > label
                && has_interior[neighbour]
                && delta_e2000(labs[label], labs[neighbour]) <= THREE_JND_DELTA_E
            {
                durable.union(label, neighbour);
            }
        }
    }
    for (label, &interior) in has_interior.iter().enumerate() {
        if interior {
            continue;
        }
        let durable_neighbours = adjacency[label]
            .keys()
            .copied()
            .filter(|&neighbour| has_interior[neighbour])
            .collect::<Vec<_>>();
        for first in 0..durable_neighbours.len() {
            for second in first + 1..durable_neighbours.len() {
                let first_label = durable_neighbours[first];
                let second_label = durable_neighbours[second];
                if delta_e2000(labs[first_label], labs[second_label]) <= THREE_JND_DELTA_E {
                    durable.union(first_label, second_label);
                }
            }
        }
    }

    let mut representative = vec![usize::MAX; labs.len()];
    for (label, &interior) in has_interior.iter().enumerate() {
        if interior {
            let root = durable.find(label);
            representative[root] = representative[root].min(label);
        }
    }
    let mut classes = vec![-1_i32; labs.len()];
    for label in 0..labs.len() {
        if has_interior[label] {
            classes[label] = representative[durable.find(label)] as i32;
        }
    }

    // Propagate durable ownership through chains of coreless sleeves.  The
    // strongest shared-boundary support wins; perceptual distance resolves
    // only support ties.  Whether an antialias sleeve joins the first or the
    // second parent, the remaining interface has the same pair of durable
    // classes and can be fitted as one continuous curve.
    loop {
        let mut proposals = Vec::<(usize, i32)>::new();
        for label in 0..labs.len() {
            if classes[label] >= 0 {
                continue;
            }
            let mut support_by_class = HashMap::<i32, (usize, f32)>::new();
            for (&neighbour, &support) in &adjacency[label] {
                let class = classes[neighbour];
                if class < 0 {
                    continue;
                }
                let entry = support_by_class.entry(class).or_insert((0, f32::INFINITY));
                entry.0 += support;
                entry.1 = entry.1.min(delta_e2000(labs[label], labs[neighbour]));
            }
            let selected = support_by_class.into_iter().min_by(
                |(first_class, (first_support, first_delta)),
                 (second_class, (second_support, second_delta))| {
                    second_support
                        .cmp(first_support)
                        .then_with(|| first_delta.total_cmp(second_delta))
                        .then_with(|| first_class.cmp(second_class))
                },
            );
            if let Some((class, _)) = selected {
                proposals.push((label, class));
            }
        }
        if proposals.is_empty() {
            break;
        }
        for (label, class) in proposals {
            classes[label] = class;
        }
    }

    // A disconnected all-coreless component has no durable parent. Keep it
    // independent instead of inventing a colour/material relationship.
    for (label, class) in classes.iter_mut().enumerate() {
        if *class < 0 {
            *class = label as i32;
        }
    }
    classes
}

fn split_closed_continuity_track(track: &[u64], stride: usize) -> Vec<Vec<u64>> {
    let base = &track[..track.len().saturating_sub(1)];
    let points: Vec<Point> = base
        .iter()
        .map(|&vertex| point_from_vertex(vertex, stride))
        .collect();
    let count = points.len();
    if count < 12 {
        return Vec::new();
    }
    let turn = |index: usize, support: usize| {
        let previous = points[(index + count - support) % count];
        let current = points[index];
        let following = points[(index + support) % count];
        let before = Point {
            x: current.x - previous.x,
            y: current.y - previous.y,
        };
        let after = Point {
            x: following.x - current.x,
            y: following.y - current.y,
        };
        (before.x * after.y - before.y * after.x).atan2(before.x * after.x + before.y * after.y)
    };
    let local: Vec<f32> = (0..count).map(|index| turn(index, 2)).collect();
    let coarse_support = 9_usize.min((count / 4).max(2));
    let coarse: Vec<f32> = (0..count)
        .map(|index| turn(index, coarse_support))
        .collect();
    let mut corners: Vec<usize> = (0..count)
        .filter(|&index| {
            let magnitude = local[index].abs();
            magnitude >= 65.0_f32.to_radians()
                && coarse[index].abs() >= 45.0_f32.to_radians()
                && local[index] * coarse[index] > 0.0
                && (1..=2).all(|offset| {
                    local[(index + count - offset) % count].abs() <= magnitude + 1e-6
                        && local[(index + offset) % count].abs() <= magnitude + 1e-6
                })
        })
        .collect();
    if corners.len() < 2 {
        let first = points
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first
                    .x
                    .total_cmp(&second.x)
                    .then(first.y.total_cmp(&second.y))
            })
            .map(|value| value.0)
            .unwrap_or(0);
        let second = points
            .iter()
            .enumerate()
            .max_by(|(_, first_point), (_, second_point)| {
                first_point
                    .distance(points[first])
                    .total_cmp(&second_point.distance(points[first]))
            })
            .map(|value| value.0)
            .unwrap_or(count / 2);
        corners = vec![first, second];
        corners.sort_unstable();
        corners.dedup();
    }
    if corners.len() < 2 {
        return Vec::new();
    }
    (0..corners.len())
        .map(|position| {
            let start = corners[position];
            let end = corners[(position + 1) % corners.len()];
            let mut result = vec![base[start]];
            let mut index = start;
            while index != end {
                index = (index + 1) % count;
                result.push(base[index]);
            }
            result
        })
        .collect()
}

fn is_shallow_continuity_arc(track: &[u64], stride: usize) -> bool {
    let mut minimum_x = f32::INFINITY;
    let mut maximum_x = f32::NEG_INFINITY;
    let mut minimum_y = f32::INFINITY;
    let mut maximum_y = f32::NEG_INFINITY;
    for &vertex in track {
        let point = point_from_vertex(vertex, stride);
        minimum_x = minimum_x.min(point.x);
        maximum_x = maximum_x.max(point.x);
        minimum_y = minimum_y.min(point.y);
        maximum_y = maximum_y.max(point.y);
    }
    let horizontal_span = maximum_x - minimum_x;
    let vertical_span = maximum_y - minimum_y;
    horizontal_span >= 40.0 && horizontal_span >= 1.5 * vertical_span
}

fn fit_adaptive_boundary_geometry(
    segmentation: &Segmentation,
    stride: usize,
    strands: &[Vec<u64>],
    protected_vertices: &HashSet<u64>,
    pair_edges: &HashMap<RegionPair, Vec<EdgeKey>>,
) -> AdaptiveBoundaryGeometry {
    let mut edge_spans = HashMap::<EdgeKey, Vec<AdaptiveCurveSpan>>::new();
    let mut proposals = HashMap::<u64, Vec<(usize, Point)>>::new();
    let mut regularized_proposals = HashMap::<u64, Vec<(usize, Point)>>::new();
    let mut regularized_fixed_points = HashSet::<u64>::new();
    let mut next_master_id;
    let mut optimal_polygons = 0_usize;
    let mut regularized_excursions = 0_usize;
    let mut continuity_faired_master_ids = HashSet::<usize>::new();
    #[cfg(feature = "diagnostics")]
    let strand_diagnostics_enabled = std::env::var_os("PICVEC_STRAND_DIAGNOSTICS").is_some();
    #[cfg(feature = "diagnostics")]
    let mut strand_diagnostics = Vec::<serde_json::Value>::new();
    let mut edge_pair = HashMap::<EdgeKey, RegionPair>::new();
    for (&pair, edges) in pair_edges {
        for &edge in edges {
            edge_pair.insert(edge, pair);
        }
    }
    const MASTER_IDS_PER_STRAND: usize = 1_000_000;
    let fitted_base_strands: Vec<FittedBaseStrand> = strands
        .par_iter()
        .enumerate()
        .filter_map(|(strand_index, strand)| {
            if strand.len() < 2 {
                return None;
            }
            let closed = strand.len() > 2 && strand.first() == strand.last();
            let raw: Vec<Point> = strand
                .iter()
                .map(|&vertex| point_from_vertex(vertex, stride))
                .collect();
            let strand_pairs: Vec<RegionPair> = strand
                .windows(2)
                .map(|vertices| edge_pair[&EdgeKey::new(vertices[0], vertices[1])])
                .collect();
            let regularized = regularize_short_corner_excursions(
                &raw,
                &strand_pairs,
                protected_vertices,
                stride,
                0.5,
            );
            let mut master_id = strand_index.saturating_mul(MASTER_IDS_PER_STRAND);
            // Use Potrace's standard corner threshold for material
            // silhouettes. The previous 1.2 value rounded compact cusps into
            // bulb-shaped caps; same-material continuity contours use their
            // separate, more permissive fit below.
            let (polygon, curves) = potrace_master_curves(
                &regularized.points,
                closed,
                0.5,
                1.0,
                &mut master_id,
                &regularized.fixed,
            );
            Some(FittedBaseStrand {
                strand: strand.clone(),
                raw,
                regularized,
                polygon,
                curves,
            })
        })
        .collect();
    next_master_id = strands.len().saturating_mul(MASTER_IDS_PER_STRAND);
    for fitted_strand in fitted_base_strands {
        let FittedBaseStrand {
            strand,
            raw,
            regularized,
            polygon,
            curves,
        } = fitted_strand;
        #[cfg(not(feature = "diagnostics"))]
        let _ = (&raw, &polygon);
        let closed = strand.len() > 2 && strand.first() == strand.last();
        #[cfg(feature = "diagnostics")]
        if strand_diagnostics_enabled && raw.len() >= 24 {
            let point = |value: Point| serde_json::json!([value.x, value.y]);
            strand_diagnostics.push(serde_json::json!({
                "raw": raw.iter().copied().map(point).collect::<Vec<_>>(),
                "regularized": regularized.points.iter().copied().map(point).collect::<Vec<_>>(),
                "changed": regularized.changed.len(),
                "corners": regularized.corners.iter().copied().collect::<Vec<_>>(),
            }));
        }
        regularized_excursions += regularized.corners.len();
        optimal_polygons += usize::from(regularized.reusable_polygon);
        let fitting = regularized.points;
        let forced = regularized.fixed;
        let vertex_count = if closed {
            strand.len() - 1
        } else {
            strand.len()
        };
        for &index in &regularized.changed {
            let vertex = strand[index % vertex_count];
            regularized_proposals
                .entry(vertex)
                .or_default()
                .push((strand.len(), fitting[index]));
        }
        for &index in &forced {
            regularized_fixed_points.insert(strand[index % vertex_count]);
        }
        #[cfg(feature = "diagnostics")]
        if let Some(path) = std::env::var_os("PICVEC_ADAPTIVE_MASTER_DIAGNOSTICS") {
            if curves.iter().any(|value| value.2 == 7384) {
                let adjusted = adjust_polygon_vertices(&fitting, &polygon, closed);
                let point = |value: Point| [value.x, value.y];
                let mut forced_values = forced.iter().copied().collect::<Vec<_>>();
                forced_values.sort_unstable();
                let value = serde_json::json!({
                    "closed": closed,
                    "raw": raw.iter().copied().map(point).collect::<Vec<_>>(),
                    "fitting": fitting.iter().copied().map(point).collect::<Vec<_>>(),
                    "polygon": polygon,
                    "forced": forced_values,
                    "adjusted": adjusted.into_iter().map(point).collect::<Vec<_>>(),
                });
                if let Ok(encoded) = serde_json::to_vec(&value) {
                    let _ = std::fs::write(path, encoded);
                }
            }
        }
        if curves.is_empty() {
            continue;
        }
        let period = strand.len() - 1;
        let mut expanded = Vec::<TaggedCurve>::new();
        if closed {
            for &(start, end, identifier, curve) in &curves {
                for shift in [-(period as f64), 0.0, period as f64] {
                    expanded.push((start + shift, end + shift, identifier, curve));
                }
            }
        } else {
            expanded.extend(curves.iter().copied());
        }
        expanded.sort_by(|first, second| first.0.total_cmp(&second.0));

        let point_at = |parameter: f64| {
            expanded
                .iter()
                .find_map(|&(start, end, _, curve)| {
                    (start - 1e-6 <= parameter && parameter <= end + 1e-6 && end > start + 1e-8)
                        .then(|| cubic_point(curve, ((parameter - start) / (end - start)) as f32))
                })
                .unwrap_or_else(|| fitting[(parameter.round() as usize) % period.max(1)])
        };
        let weight = strand.len();
        let vertex_count = if closed {
            strand.len() - 1
        } else {
            strand.len()
        };
        for (index, &vertex) in strand.iter().take(vertex_count).enumerate() {
            proposals
                .entry(vertex)
                .or_default()
                .push((weight, point_at(index as f64)));
        }
        if closed {
            proposals
                .entry(strand[0])
                .or_default()
                .push((weight, point_at(period as f64)));
        }

        for (index, vertices) in strand.windows(2).enumerate() {
            let mut pieces = Vec::<AdaptiveCurveSpan>::new();
            for &(start, end, identifier, curve) in &expanded {
                let overlap_start = (index as f64).max(start);
                let overlap_end = ((index + 1) as f64).min(end);
                if overlap_end <= overlap_start + 1e-7 || end <= start + 1e-7 {
                    continue;
                }
                pieces.push(AdaptiveCurveSpan {
                    master_id: identifier,
                    curve,
                    start_parameter: (overlap_start - start) / (end - start),
                    end_parameter: (overlap_end - start) / (end - start),
                });
            }
            let edge = EdgeKey::new(vertices[0], vertices[1]);
            if vertices[0] != edge.0 {
                pieces = pieces
                    .into_iter()
                    .rev()
                    .map(|piece| AdaptiveCurveSpan {
                        start_parameter: piece.end_parameter,
                        end_parameter: piece.start_parameter,
                        ..piece
                    })
                    .collect();
            }
            edge_spans.insert(edge, pieces);
        }
    }

    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_ADAPTIVE_BASE_DIAGNOSTICS") {
        let encode_curve = |curve: CurveSegment| match curve {
            CurveSegment::Line { start, end } => serde_json::json!([
                [start.x, start.y],
                [start.x, start.y],
                [end.x, end.y],
                [end.x, end.y]
            ]),
            CurveSegment::Cubic {
                start,
                first,
                second,
                end,
            } => serde_json::json!([
                [start.x, start.y],
                [first.x, first.y],
                [second.x, second.y],
                [end.x, end.y]
            ]),
        };
        let mut edges = serde_json::Map::new();
        let mut positions = serde_json::Map::new();
        for x in 1133..1138 {
            for y in 488..492 {
                let vertex = vertex_id(x, y, stride);
                if let Some(values) = proposals.get(&vertex) {
                    let best_weight = values.iter().map(|(weight, _)| *weight).max().unwrap_or(0);
                    let best: Vec<Point> = values
                        .iter()
                        .filter_map(|&(weight, point)| (weight == best_weight).then_some(point))
                        .collect();
                    let mut position = Point {
                        x: best.iter().map(|point| point.x).sum::<f32>() / best.len() as f32,
                        y: best.iter().map(|point| point.y).sum::<f32>() / best.len() as f32,
                    };
                    let origin = Point {
                        x: x as f32,
                        y: y as f32,
                    };
                    let distance = position.distance(origin);
                    if distance > 0.25 {
                        position = Point {
                            x: origin.x + (position.x - origin.x) * 0.25 / distance,
                            y: origin.y + (position.y - origin.y) * 0.25 / distance,
                        };
                    }
                    positions.insert(
                        format!("{x},{y}"),
                        serde_json::json!([position.x, position.y]),
                    );
                }
                for (nx, ny) in [
                    (x.saturating_sub(1), y),
                    (x + 1, y),
                    (x, y.saturating_sub(1)),
                    (x, y + 1),
                ] {
                    let neighbour = vertex_id(nx, ny, stride);
                    let edge = EdgeKey::new(vertex, neighbour);
                    let Some(spans) = edge_spans.get(&edge) else {
                        continue;
                    };
                    let first = point_from_vertex(edge.0, stride);
                    let second = point_from_vertex(edge.1, stride);
                    let key = format!(
                        "{},{}-{},{}",
                        first.x as i32, first.y as i32, second.x as i32, second.y as i32
                    );
                    edges.insert(
                        key,
                        serde_json::Value::Array(
                            spans
                                .iter()
                                .map(|span| {
                                    serde_json::json!({
                                        "master": span.master_id,
                                        "start": span.start_parameter,
                                        "end": span.end_parameter,
                                        "curve": encode_curve(span.curve),
                                    })
                                })
                                .collect(),
                        ),
                    );
                }
            }
        }
        let value = serde_json::json!({"positions": positions, "edges": edges});
        if let Ok(encoded) = serde_json::to_vec(&value) {
            let _ = std::fs::write(path, encoded);
        }
    }

    // Port the Python continuity-class pass. Quantized Paint labels can
    // change many times along one visible contour; fitting each RegionPair
    // separately freezes those changes as staircase anchors. Trace the
    // boundary between coarse perceptual classes, fit it once, then slice the
    // same master back onto every original half-edge.
    let geometry_lab = crate::edge::lab_pixels(&segmentation.canonical);
    let mut all_by_label = vec![Vec::<usize>::new(); segmentation.regions.len()];
    let mut samples_by_label = vec![Vec::<usize>::new(); segmentation.regions.len()];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        all_by_label[label as usize].push(index);
        if segmentation.paint_samples[index] {
            samples_by_label[label as usize].push(index);
        }
    }
    let channel_median = |indices: &[usize], channel: usize| {
        let mut values: Vec<f32> = indices
            .iter()
            .map(|&index| match channel {
                0 => geometry_lab[index].l,
                1 => geometry_lab[index].a,
                _ => geometry_lab[index].b,
            })
            .collect();
        values.sort_by(f32::total_cmp);
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    };
    let continuity_lab_by_label: Vec<Lab> = (0..segmentation.regions.len())
        .map(|label| {
            let indices = if samples_by_label[label].is_empty() {
                &all_by_label[label]
            } else {
                &samples_by_label[label]
            };
            let l = channel_median(indices, 0);
            let a = channel_median(indices, 1);
            let b = channel_median(indices, 2);
            Lab { l, a, b }
        })
        .collect();
    let mut continuity_adjacency = vec![HashMap::<usize, usize>::new(); segmentation.regions.len()];
    for pair in edge_pair.values() {
        if pair.0 >= 0 && pair.1 >= 0 {
            let first = pair.0 as usize;
            let second = pair.1 as usize;
            *continuity_adjacency[first].entry(second).or_default() += 1;
            *continuity_adjacency[second].entry(first).or_default() += 1;
        }
    }
    let mut continuity_has_interior = vec![false; segmentation.regions.len()];
    if segmentation.width >= 3 && segmentation.height >= 3 {
        for y in 1..segmentation.height - 1 {
            for x in 1..segmentation.width - 1 {
                let index = y * segmentation.width + x;
                let label = segmentation.labels[index];
                if (-1_isize..=1).all(|dy| {
                    (-1_isize..=1).all(|dx| {
                        segmentation.labels[(y as isize + dy) as usize * segmentation.width
                            + (x as isize + dx) as usize]
                            == label
                    })
                }) {
                    continuity_has_interior[label as usize] = true;
                }
            }
        }
    }
    let continuity_class_by_label = contextual_continuity_classes(
        &continuity_lab_by_label,
        &continuity_adjacency,
        &continuity_has_interior,
    );
    let class_for_label = |label: i32| {
        if label < 0 {
            return -1;
        }
        continuity_class_by_label[label as usize]
    };
    let mut class_edges_by_pair = HashMap::<RegionPair, HashSet<EdgeKey>>::new();
    let mut class_pair_order = Vec::<RegionPair>::new();
    let mut class_adjacency = HashMap::<u64, HashSet<u64>>::new();
    let mut uncertain_class_edges = HashSet::<EdgeKey>::new();
    let mut add_class_edge =
        |first_class: i32, second_class: i32, edge: EdgeKey, uncertain: bool| {
            if first_class == second_class {
                return;
            }
            let class_pair = RegionPair::new(first_class, second_class);
            if !class_edges_by_pair.contains_key(&class_pair) {
                class_pair_order.push(class_pair);
            }
            class_edges_by_pair
                .entry(class_pair)
                .or_default()
                .insert(edge);
            class_adjacency.entry(edge.0).or_default().insert(edge.1);
            class_adjacency.entry(edge.1).or_default().insert(edge.0);
            if uncertain {
                uncertain_class_edges.insert(edge);
            }
        };
    // Match `_collect_label_edges` insertion order: opposite canvas edges are
    // interleaved first, followed by row-major horizontal and vertical label
    // transitions.  Python dict insertion order determines continuity master
    // IDs and equal-weight proposal resolution.
    for x in 0..segmentation.width {
        add_class_edge(
            -1,
            class_for_label(segmentation.labels[x] as i32),
            EdgeKey::new(vertex_id(x, 0, stride), vertex_id(x + 1, 0, stride)),
            false,
        );
        let bottom = (segmentation.height - 1) * segmentation.width + x;
        add_class_edge(
            -1,
            class_for_label(segmentation.labels[bottom] as i32),
            EdgeKey::new(
                vertex_id(x, segmentation.height, stride),
                vertex_id(x + 1, segmentation.height, stride),
            ),
            false,
        );
    }
    for y in 0..segmentation.height {
        add_class_edge(
            -1,
            class_for_label(segmentation.labels[y * segmentation.width] as i32),
            EdgeKey::new(vertex_id(0, y, stride), vertex_id(0, y + 1, stride)),
            false,
        );
        let right = y * segmentation.width + segmentation.width - 1;
        add_class_edge(
            -1,
            class_for_label(segmentation.labels[right] as i32),
            EdgeKey::new(
                vertex_id(segmentation.width, y, stride),
                vertex_id(segmentation.width, y + 1, stride),
            ),
            false,
        );
    }
    for y in 1..segmentation.height {
        for x in 0..segmentation.width {
            let above = segmentation.labels[(y - 1) * segmentation.width + x] as i32;
            let below = segmentation.labels[y * segmentation.width + x] as i32;
            if above != below {
                let edge = EdgeKey::new(vertex_id(x, y, stride), vertex_id(x + 1, y, stride));
                let uncertain = !continuity_has_interior[above as usize]
                    || !continuity_has_interior[below as usize];
                add_class_edge(
                    class_for_label(above),
                    class_for_label(below),
                    edge,
                    uncertain,
                );
            }
        }
    }
    for y in 0..segmentation.height {
        for x in 1..segmentation.width {
            let left = segmentation.labels[y * segmentation.width + x - 1] as i32;
            let right = segmentation.labels[y * segmentation.width + x] as i32;
            if left != right {
                let edge = EdgeKey::new(vertex_id(x, y, stride), vertex_id(x, y + 1, stride));
                let uncertain = !continuity_has_interior[left as usize]
                    || !continuity_has_interior[right as usize];
                add_class_edge(
                    class_for_label(left),
                    class_for_label(right),
                    edge,
                    uncertain,
                );
            }
        }
    }
    let class_junctions: HashSet<u64> = class_adjacency
        .iter()
        .filter_map(|(&point, neighbours)| (neighbours.len() != 2).then_some(point))
        .collect();
    #[cfg(feature = "diagnostics")]
    let continuity_diagnostics_enabled =
        std::env::var_os("PICVEC_CONTINUITY_DIAGNOSTICS").is_some();
    #[cfg(feature = "diagnostics")]
    let mut continuity_diagnostics = Vec::<serde_json::Value>::new();
    #[cfg(feature = "diagnostics")]
    if continuity_diagnostics_enabled {
        continuity_diagnostics.push(serde_json::json!({
            "kind": "classes",
            "labels": continuity_lab_by_label
                .iter()
                .zip(&continuity_class_by_label)
                .zip(&continuity_has_interior)
                .enumerate()
                .map(|(label, ((lab, class), interior))| serde_json::json!({
                    "label": label,
                    "class": class,
                    "interior": interior,
                    "lab": [lab.l, lab.a, lab.b],
                }))
                .collect::<Vec<_>>(),
        }));
    }
    for class_pair in class_pair_order {
        let edges = &class_edges_by_pair[&class_pair];
        let mut fitting_tracks = Vec::<(Vec<u64>, bool)>::new();
        let traced_tracks = trace_edge_chains(edges, &class_junctions, stride);
        #[cfg(feature = "diagnostics")]
        if continuity_diagnostics_enabled {
            continuity_diagnostics.push(serde_json::json!({
                "kind": "tracks",
                "class_pair": [class_pair.0, class_pair.1],
                "edge_count": edges.len(),
                "tracks": traced_tracks.iter().map(|track| serde_json::json!({
                    "length": track.len(),
                    "closed": track.first() == track.last(),
                    "start_junction": track.first().is_some_and(|value| class_junctions.contains(value)),
                    "end_junction": track.last().is_some_and(|value| class_junctions.contains(value)),
                })).collect::<Vec<_>>(),
            }));
        }
        for traced_track in traced_tracks {
            let closed_track = traced_track.first() == traced_track.last();
            if closed_track {
                // A closed material contour is especially likely to be split
                // into many short RegionPair chains: every quantized shade
                // inside a material changes the pair even though the visible
                // outer boundary remains continuous. Split
                // at persistent semantic corners and fit each full arc once.
                let split_tracks = split_closed_continuity_track(&traced_track, stride);
                #[cfg(feature = "diagnostics")]
                if continuity_diagnostics_enabled {
                    continuity_diagnostics.push(serde_json::json!({
                        "kind": "split",
                        "class_pair": [class_pair.0, class_pair.1],
                        "lengths": split_tracks.iter().map(Vec::len).collect::<Vec<_>>(),
                    }));
                }
                fitting_tracks.extend(
                    split_tracks
                        .into_iter()
                        // A general material contour must not hand a split
                        // endpoint to another class boundary: its connected
                        // base curve may place that endpoint differently and
                        // make the reconstructed face discontinuous.
                        .filter(|track| {
                            track
                                .first()
                                .zip(track.last())
                                .is_some_and(|(first, last)| {
                                    !class_junctions.contains(first)
                                        && !class_junctions.contains(last)
                                })
                        })
                        // The split points protect persistent contour corners.
                        .map(|track| (track, true)),
                );
            } else {
                fitting_tracks.push((traced_track, false));
            }
        }
        for (track, closed_track) in fitting_tracks {
            let coordinates: Vec<Point> = track
                .iter()
                .map(|&vertex| point_from_vertex(vertex, stride))
                .collect();
            let mut corners: Vec<usize> = if closed_track {
                // The closed contour was already split at stable cyclic
                // corners. Running the open corner detector again mistakes
                // raster steps for semantic corners and prevents each full
                // material arc from being fitted.
                Vec::new()
            } else {
                persistent_open_corners(&coordinates)
                    .into_iter()
                    .map(|(index, _)| index)
                    .filter(|&index| index > 0 && index + 1 < track.len())
                    .collect()
            };
            // Short fits are safe only for arcs cut from one closed contour:
            // both endpoints then remain anchored to the same observed loop.
            // Open tracks can end at independently fitted material junctions,
            // where lowering the old forty-vertex floor creates tiny gaps.
            const MINIMUM_CLOSED_ARC_VERTICES: usize = 17;
            const MINIMUM_OPEN_TRACK_VERTICES: usize = 40;
            let minimum_continuity_vertices = if closed_track {
                MINIMUM_CLOSED_ARC_VERTICES
            } else {
                MINIMUM_OPEN_TRACK_VERTICES
            };
            if track.len() < minimum_continuity_vertices {
                continue;
            }
            corners.extend([0, track.len() - 1]);
            corners.sort_unstable();
            corners.dedup();
            for corner_pair in corners.windows(2) {
                let corner_start = corner_pair[0];
                let corner_end = corner_pair[1];
                let guard_edges = 1_usize;
                let endpoint_needs_guard = |vertex: &u64| {
                    class_junctions.contains(vertex)
                        || (closed_track && protected_vertices.contains(vertex))
                };
                let fit_start = if endpoint_needs_guard(&track[corner_start]) {
                    (corner_start + guard_edges).min(corner_end)
                } else {
                    corner_start
                };
                let fit_end = if endpoint_needs_guard(&track[corner_end]) {
                    corner_end.saturating_sub(guard_edges).max(fit_start)
                } else {
                    corner_end
                };
                if fit_end + 1 < fit_start + minimum_continuity_vertices {
                    continue;
                }
                let fitting_vertices = &track[fit_start..=fit_end];
                let original_pairs = fitting_vertices
                    .windows(2)
                    .filter_map(|vertices| {
                        edge_pair
                            .get(&EdgeKey::new(vertices[0], vertices[1]))
                            .copied()
                    })
                    .collect::<HashSet<_>>();
                // The adaptive base fit already owns a contour made from one
                // RegionPair. Continuity fitting is needed only when multiple
                // quantized pairs fragment the same class boundary.
                if original_pairs.len() < 2 {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                let uncertain_edges = fitting_vertices
                    .windows(2)
                    .filter(|vertices| {
                        uncertain_class_edges.contains(&EdgeKey::new(vertices[0], vertices[1]))
                    })
                    .count();
                // Coreless ownership affects class connectivity, not the
                // geometric error budget. Every replacement remains inside
                // the same source-raster corridor.
                let continuity_raster_corridor = fairing_raster_corridor();
                let mut raw: Vec<Point> = fitting_vertices
                    .iter()
                    .map(|&vertex| point_from_vertex(vertex, stride))
                    .collect();
                let mut start_tangent = None;
                let mut end_tangent = None;
                if fit_start > 0 {
                    if let Some((position, tangent)) = connected_adaptive_edge_geometry(
                        &edge_spans,
                        track[fit_start - 1],
                        track[fit_start],
                        true,
                        stride,
                    ) {
                        raw[0] = position;
                        start_tangent = Some(tangent);
                    }
                }
                if fit_end + 1 < track.len() {
                    if let Some((position, tangent)) = connected_adaptive_edge_geometry(
                        &edge_spans,
                        track[fit_end],
                        track[fit_end + 1],
                        false,
                        stride,
                    ) {
                        let last = raw.len() - 1;
                        raw[last] = position;
                        end_tangent = Some(tangent);
                    }
                }
                let travelled = raw
                    .windows(2)
                    .map(|pair| pair[0].distance(pair[1]))
                    .sum::<f32>();
                let displacement = raw[0].distance(raw[raw.len() - 1]);
                let minimum_displacement = if closed_track && raw.len() < 40 {
                    12.0
                } else {
                    32.0
                };
                if displacement < minimum_displacement
                    || (!closed_track && displacement / travelled.max(1e-6) < 0.60)
                {
                    continue;
                }
                let forced = HashSet::new();
                let (_, mut baseline_values) = potrace_master_curves(
                    &raw,
                    false,
                    0.5,
                    4.0 / 3.0,
                    &mut next_master_id,
                    &forced,
                );
                if baseline_values.is_empty() {
                    continue;
                }
                if closed_track {
                    // Each arc of a general closed contour is fitted
                    // independently. Potrace may move its open endpoints to
                    // opposite sides of the same raster corner, leaving a
                    // subpixel gap when the shared face loop is reconstructed.
                    // Anchor both arcs to their common observed corner while
                    // moving the adjacent control handle by the same amount.
                    baseline_values[0].3 = curve_with_start(baseline_values[0].3, raw[0]);
                    let last = baseline_values.len() - 1;
                    baseline_values[last].3 =
                        curve_with_end(baseline_values[last].3, raw[raw.len() - 1]);
                }
                let align_tangent = |curve: CurveSegment, tangent: Point, at_end: bool| {
                    let chord = Point {
                        x: curve.end().x - curve.start().x,
                        y: curve.end().y - curve.start().y,
                    };
                    let mut direction = normalized(tangent);
                    if direction.x * chord.x + direction.y * chord.y < 0.0 {
                        direction.x = -direction.x;
                        direction.y = -direction.y;
                    }
                    match (at_end, curve) {
                        (
                            false,
                            CurveSegment::Cubic {
                                start,
                                first,
                                second,
                                end,
                            },
                        ) => {
                            let length = start.distance(first);
                            CurveSegment::Cubic {
                                start,
                                first: Point {
                                    x: start.x + direction.x * length,
                                    y: start.y + direction.y * length,
                                },
                                second,
                                end,
                            }
                        }
                        (
                            true,
                            CurveSegment::Cubic {
                                start,
                                first,
                                second,
                                end,
                            },
                        ) => {
                            let length = end.distance(second);
                            CurveSegment::Cubic {
                                start,
                                first,
                                second: Point {
                                    x: end.x - direction.x * length,
                                    y: end.y - direction.y * length,
                                },
                                end,
                            }
                        }
                        (_, value) => value,
                    }
                };
                if let Some(tangent) = start_tangent {
                    baseline_values[0].3 = align_tangent(baseline_values[0].3, tangent, false);
                }
                if let Some(tangent) = end_tangent {
                    let last = baseline_values.len() - 1;
                    baseline_values[last].3 = align_tangent(baseline_values[last].3, tangent, true);
                }
                let baseline: Vec<CurveSegment> =
                    baseline_values.iter().map(|value| value.3).collect();
                let strict_baseline = boundary_corridor_supported(&raw, &baseline, 0.5);
                let mut fair = if closed_track && is_shallow_continuity_arc(&track, stride) {
                    bounded_fairing_shared_boundary(&raw, &baseline, 1.25)
                } else {
                    bounded_fairing_direct_shared_boundary(&raw, &baseline, false, true)
                };
                #[cfg(feature = "diagnostics")]
                let fair_candidate_len = fair.len();
                #[cfg(feature = "diagnostics")]
                let mut fair_raster_error = None;
                if fair.len() < baseline.len() {
                    if let Some(tangent) = start_tangent {
                        fair[0] = align_tangent(fair[0], tangent, false);
                    }
                    if let Some(tangent) = end_tangent {
                        let last = fair.len() - 1;
                        fair[last] = align_tangent(fair[last], tangent, true);
                    }
                    let rendered = sample_curve_sequence(&fair, 0.25);
                    #[cfg(feature = "diagnostics")]
                    if continuity_diagnostics_enabled {
                        let observations: Vec<Point> = raw
                            .windows(2)
                            .map(|pair| interpolate_point(pair[0], pair[1], 0.5))
                            .collect();
                        let source_samples = sample_polyline_segments(&raw, 0.25);
                        let maximum_nearest = |query: &[Point], reference: &[Point]| {
                            query
                                .iter()
                                .map(|point| {
                                    reference
                                        .iter()
                                        .map(|candidate| point.distance(*candidate))
                                        .fold(f32::INFINITY, f32::min)
                                })
                                .fold(0.0_f32, f32::max)
                        };
                        fair_raster_error = Some(
                            maximum_nearest(&observations, &rendered)
                                .max(maximum_nearest(&rendered, &source_samples)),
                        );
                    }
                    if !raster_boundary_supported(&raw, &rendered, continuity_raster_corridor) {
                        fair = baseline.clone();
                    }
                }
                #[cfg(feature = "diagnostics")]
                if continuity_diagnostics_enabled {
                    let encode_curve = |curve: &CurveSegment| match *curve {
                        CurveSegment::Line { start, end } => serde_json::json!([
                            [start.x, start.y],
                            [
                                start.x + (end.x - start.x) / 3.0,
                                start.y + (end.y - start.y) / 3.0
                            ],
                            [
                                start.x + 2.0 * (end.x - start.x) / 3.0,
                                start.y + 2.0 * (end.y - start.y) / 3.0
                            ],
                            [end.x, end.y],
                        ]),
                        CurveSegment::Cubic {
                            start,
                            first,
                            second,
                            end,
                        } => serde_json::json!([
                            [start.x, start.y],
                            [first.x, first.y],
                            [second.x, second.y],
                            [end.x, end.y],
                        ]),
                    };
                    let fitting_start_point = point_from_vertex(fitting_vertices[0], stride);
                    let fitting_previous_point = (fit_start > 0).then(|| {
                        let point = point_from_vertex(track[fit_start - 1], stride);
                        [point.x, point.y]
                    });
                    let start_connection_spans = if fit_start > 0 {
                        edge_spans
                            .get(&EdgeKey::new(track[fit_start - 1], track[fit_start]))
                            .into_iter()
                            .flatten()
                            .map(|piece| {
                                serde_json::json!({
                                    "id": piece.master_id,
                                    "start": piece.start_parameter,
                                    "end": piece.end_parameter,
                                    "curve": encode_curve(&piece.curve),
                                })
                            })
                            .collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    };
                    continuity_diagnostics.push(serde_json::json!({
                        "kind": "fit",
                        "class_pair": [class_pair.0, class_pair.1],
                        "closed_track": closed_track,
                        "source_len": raw.len(),
                        "baseline_len": baseline.len(),
                        "candidate_len": fair_candidate_len,
                        "result_len": fair.len(),
                        "candidate_raster_error": fair_raster_error,
                        "uncertain_edges": uncertain_edges,
                        "raster_corridor": continuity_raster_corridor,
                        "start_tangent": start_tangent.map(|point| [point.x, point.y]),
                        "end_tangent": end_tangent.map(|point| [point.x, point.y]),
                        "adopted": fair.len() < baseline.len(),
                        "start": [raw[0].x, raw[0].y],
                        "end": [raw[raw.len() - 1].x, raw[raw.len() - 1].y],
                        "fitting_start": [fitting_start_point.x, fitting_start_point.y],
                        "fitting_previous": fitting_previous_point,
                        "start_connection_spans": start_connection_spans,
                        "source": raw.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                        "baseline": baseline.iter().map(&encode_curve).collect::<Vec<_>>(),
                        "result": fair.iter().map(&encode_curve).collect::<Vec<_>>(),
                    }));
                }
                // The bounded fairing routine already performs the same
                // bidirectional sqrt(2)-pixel raster corridor check as the
                // Python continuity pass.
                let fitted = if fair.len() < baseline.len() {
                    let fitted =
                        parameterize_curves_by_source_arclength(&raw, fair, &mut next_master_id);
                    continuity_faired_master_ids.extend(fitted.iter().map(|value| value.2));
                    fitted
                } else {
                    if !strict_baseline {
                        continue;
                    }
                    if closed_track {
                        // The Potrace polygon's native parameter interval can
                        // begin inside its first curve. On a split closed
                        // contour that makes the first raster half-edge start
                        // past the anchored corner. Reparameterize by observed
                        // arclength so adjacent arcs meet at exactly the same
                        // point even when fairing did not reduce the curve.
                        parameterize_curves_by_source_arclength(&raw, baseline, &mut next_master_id)
                    } else {
                        baseline_values
                    }
                };
                let correction_weight =
                    if closed_track { 2_000_000 } else { 1_000_000 } + raw.len();
                let point_at = |parameter: f64| {
                    fitted
                        .iter()
                        .find_map(|&(start, end, _, curve)| {
                            (start - 1e-6 <= parameter
                                && parameter <= end + 1e-6
                                && end > start + 1e-8)
                                .then(|| {
                                    cubic_point(curve, ((parameter - start) / (end - start)) as f32)
                                })
                        })
                        .unwrap_or_else(|| raw[parameter.round() as usize])
                };
                for (index, &vertex) in fitting_vertices.iter().enumerate() {
                    proposals
                        .entry(vertex)
                        .or_default()
                        .push((correction_weight, point_at(index as f64)));
                }
                for index in 0..fitting_vertices.len() - 1 {
                    let mut pieces = Vec::new();
                    for &(start, end, identifier, curve) in &fitted {
                        let overlap_start = (index as f64).max(start);
                        let overlap_end = ((index + 1) as f64).min(end);
                        if overlap_end <= overlap_start + 1e-7 || end <= start + 1e-7 {
                            continue;
                        }
                        pieces.push(AdaptiveCurveSpan {
                            master_id: identifier,
                            curve,
                            start_parameter: (overlap_start - start) / (end - start),
                            end_parameter: (overlap_end - start) / (end - start),
                        });
                    }
                    let start = fitting_vertices[index];
                    let end = fitting_vertices[index + 1];
                    let edge = EdgeKey::new(start, end);
                    if start != edge.0 {
                        pieces = pieces
                            .into_iter()
                            .rev()
                            .map(|piece| AdaptiveCurveSpan {
                                start_parameter: piece.end_parameter,
                                end_parameter: piece.start_parameter,
                                ..piece
                            })
                            .collect();
                    }
                    edge_spans.insert(edge, pieces);
                }
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_CONTINUITY_DIAGNOSTICS") {
        if let Ok(encoded) = serde_json::to_vec(&continuity_diagnostics) {
            let _ = std::fs::write(path, encoded);
        }
    }

    let vertices: HashSet<u64> = pair_edges
        .values()
        .flatten()
        .flat_map(|edge| [edge.0, edge.1])
        .collect();
    let mut vertex_positions = VertexPositions::new();
    for vertex in vertices {
        let origin = point_from_vertex(vertex, stride);
        if is_canvas_vertex(vertex, stride, segmentation.width, segmentation.height) {
            vertex_positions.insert(vertex, origin);
            continue;
        }
        let Some(values) = proposals.get(&vertex) else {
            vertex_positions.insert(vertex, origin);
            continue;
        };
        let best_weight = values.iter().map(|(weight, _)| *weight).max().unwrap_or(0);
        let best: Vec<Point> = values
            .iter()
            .filter_map(|&(weight, point)| (weight == best_weight).then_some(point))
            .collect();
        let position = Point {
            x: best.iter().map(|point| point.x).sum::<f32>() / best.len() as f32,
            y: best.iter().map(|point| point.y).sum::<f32>() / best.len() as f32,
        };
        vertex_positions.insert(vertex, bounded_vertex_position(origin, position));
    }
    let mut regularized_observations = VertexPositions::new();
    for (vertex, values) in regularized_proposals {
        let best_weight = values.iter().map(|(weight, _)| *weight).max().unwrap_or(0);
        let best: Vec<Point> = values
            .iter()
            .filter_map(|&(weight, point)| (weight == best_weight).then_some(point))
            .collect();
        regularized_observations.insert(
            vertex,
            Point {
                x: best.iter().map(|point| point.x).sum::<f32>() / best.len() as f32,
                y: best.iter().map(|point| point.y).sum::<f32>() / best.len() as f32,
            },
        );
    }
    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_STRAND_DIAGNOSTICS") {
        if let Ok(encoded) = serde_json::to_vec(&strand_diagnostics) {
            let _ = std::fs::write(path, encoded);
        }
    }
    AdaptiveBoundaryGeometry {
        edge_spans,
        vertex_positions,
        regularized_observations,
        regularized_fixed_points,
        regularized_excursions,
        optimal_polygons,
        continuity_faired_master_ids,
    }
}

fn adaptive_chain_curves(
    raw_edges: &[(u64, u64)],
    geometry: &AdaptiveBoundaryGeometry,
    stride: usize,
) -> (Vec<Point>, Vec<CurveSegment>) {
    if raw_edges.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut spans = Vec::<(AdaptiveCurveSpan, usize)>::new();
    for (edge_index, &(start, end)) in raw_edges.iter().enumerate() {
        let edge = EdgeKey::new(start, end);
        let mut pieces = geometry.edge_spans.get(&edge).cloned().unwrap_or_default();
        if start != edge.0 {
            pieces = pieces
                .into_iter()
                .rev()
                .map(|piece| AdaptiveCurveSpan {
                    start_parameter: piece.end_parameter,
                    end_parameter: piece.start_parameter,
                    ..piece
                })
                .collect();
        }
        spans.extend(pieces.into_iter().map(|piece| (piece, edge_index)));
    }
    let mut merged = Vec::<(AdaptiveCurveSpan, usize, usize)>::new();
    for (span, edge_index) in spans {
        if let Some(previous) = merged.last_mut() {
            if previous.0.master_id == span.master_id
                && (previous.0.end_parameter - span.start_parameter).abs() <= 1e-5
            {
                previous.0.end_parameter = span.end_parameter;
                previous.2 = edge_index;
                continue;
            }
        }
        merged.push((span, edge_index, edge_index));
    }
    let positioned = |vertex: u64| {
        geometry
            .vertex_positions
            .get(&vertex)
            .copied()
            .unwrap_or_else(|| point_from_vertex(vertex, stride))
    };
    let mut raw: Vec<Point> = raw_edges.iter().map(|edge| positioned(edge.0)).collect();
    raw.push(positioned(raw_edges[raw_edges.len() - 1].1));
    let mut continuity_curve = Vec::with_capacity(merged.len());
    let mut curves: Vec<CurveSegment> = merged
        .iter()
        .map(|(span, _, _)| {
            continuity_curve.push(
                geometry
                    .continuity_faired_master_ids
                    .contains(&span.master_id),
            );
            oriented_curve_interval(span.curve, span.start_parameter, span.end_parameter)
        })
        .collect();
    if curves.is_empty() {
        curves = raw
            .windows(2)
            .map(|pair| CurveSegment::Line {
                start: pair[0],
                end: pair[1],
            })
            .collect();
        continuity_curve.resize(curves.len(), false);
    }
    // A curve master is parameterized by source-chain index, but Bézier
    // speed within that master is not uniform. When two different masters
    // meet exactly between raster edges, their sliced endpoints can therefore
    // disagree even though both refer to the same topology vertex. Anchor
    // only such edge-boundary handoffs to the already bounded shared vertex;
    // transitions inside one raster edge retain their fitted parameter.
    if merged.len() == curves.len() {
        for index in 0..curves.len().saturating_sub(1) {
            let (_, _, left_edge) = merged[index];
            let (_, right_edge, _) = merged[index + 1];
            if left_edge + 1 == right_edge {
                let joint = raw[right_edge];
                // Moving a sliced endpoint also translates its adjacent
                // control handle. Keep that repair inside the same local
                // raster-support budget as fairing; otherwise a badly
                // parameterized master can turn a small corner into a long
                // spike even though the topology itself is valid.
                let maximum_handoff_shift = 2.0 * fairing_raster_corridor();
                if curves[index].end().distance(joint) <= maximum_handoff_shift
                    && curves[index + 1].start().distance(joint) <= maximum_handoff_shift
                {
                    curves[index] = curve_with_end(curves[index], joint);
                    curves[index + 1] = curve_with_start(curves[index + 1], joint);
                }
            }
        }
    }
    if let Some(first) = curves.first_mut() {
        let delta = Point {
            x: raw[0].x - first.start().x,
            y: raw[0].y - first.start().y,
        };
        *first = match *first {
            CurveSegment::Line { end, .. } => CurveSegment::Line { start: raw[0], end },
            CurveSegment::Cubic {
                first, second, end, ..
            } => CurveSegment::Cubic {
                start: raw[0],
                first: Point {
                    x: first.x + delta.x,
                    y: first.y + delta.y,
                },
                second,
                end,
            },
        };
    }
    if let Some(last) = curves.last_mut() {
        let endpoint = raw[raw.len() - 1];
        let delta = Point {
            x: endpoint.x - last.end().x,
            y: endpoint.y - last.end().y,
        };
        *last = match *last {
            CurveSegment::Line { start, .. } => CurveSegment::Line {
                start,
                end: endpoint,
            },
            CurveSegment::Cubic {
                start,
                first,
                second,
                ..
            } => CurveSegment::Cubic {
                start,
                first,
                second: Point {
                    x: second.x + delta.x,
                    y: second.y + delta.y,
                },
                end: endpoint,
            },
        };
    }
    for (curve, continuity) in curves.iter_mut().zip(continuity_curve) {
        // De Casteljau slicing already preserves the exact parent Bézier.
        // Re-constraining each slice independently changes its tangent at
        // every quantized RegionPair transition and recreates visible kinks.
        if !continuity {
            *curve = enforce_convex_cubic_controls(*curve);
        }
    }
    (raw, curves)
}

#[cfg(feature = "diagnostics")]
fn shared_chain_diagnostic(
    id: usize,
    pair: RegionPair,
    raw_edges: &[(u64, u64)],
    segments: &[CurveSegment],
    closed: bool,
    stride: usize,
    regularized_observations: &VertexPositions,
) -> serde_json::Value {
    let observed = |vertex: u64| {
        regularized_observations
            .get(&vertex)
            .copied()
            .unwrap_or_else(|| point_from_vertex(vertex, stride))
    };
    let mut source: Vec<Point> = raw_edges.iter().map(|edge| observed(edge.0)).collect();
    if let Some(edge) = raw_edges.last() {
        source.push(observed(edge.1));
    }
    let point = |value: Point| serde_json::json!([value.x, value.y]);
    let curves: Vec<serde_json::Value> = segments
        .iter()
        .map(|&segment| match segment {
            CurveSegment::Line { start, end } => serde_json::json!({
                "start": point(start),
                "first": point(interpolate_point(start, end, 1.0 / 3.0)),
                "second": point(interpolate_point(start, end, 2.0 / 3.0)),
                "end": point(end),
            }),
            CurveSegment::Cubic {
                start,
                first,
                second,
                end,
            } => serde_json::json!({
                "start": point(start),
                "first": point(first),
                "second": point(second),
                "end": point(end),
            }),
        })
        .collect();
    serde_json::json!({
        "id": id,
        "labels": [pair.0, pair.1],
        "closed": closed,
        "source": source.into_iter().map(point).collect::<Vec<_>>(),
        "curves": curves,
    })
}

fn build_shared_chains(
    segmentation: &Segmentation,
    stride: usize,
    directed_edges: &[Vec<GridEdge>],
    pair_edges: &HashMap<RegionPair, Vec<EdgeKey>>,
) -> (
    Vec<SharedChain>,
    EdgeChainLookup,
    VertexPositions,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let (_, _, strands, junctions) = boundary_topology(stride, directed_edges, pair_edges);
    let adaptive =
        fit_adaptive_boundary_geometry(segmentation, stride, &strands, &junctions, pair_edges);
    let positions = adaptive.vertex_positions.clone();
    let mut chains = Vec::<SharedChain>::new();
    let mut lookup = HashMap::<EdgeKey, (usize, u64, u64)>::new();
    #[cfg(feature = "diagnostics")]
    let diagnostics_enabled = std::env::var_os("PICVEC_GEOMETRY_DIAGNOSTICS").is_some();
    #[cfg(feature = "diagnostics")]
    let mut diagnostics = Vec::<serde_json::Value>::new();
    #[cfg(feature = "diagnostics")]
    let stage_diagnostics_enabled = std::env::var_os("PICVEC_GEOMETRY_STAGE_DIAGNOSTICS").is_some();
    #[cfg(feature = "diagnostics")]
    let mut stage_diagnostics = serde_json::Map::new();
    let mut pairs: Vec<(RegionPair, &[EdgeKey])> = pair_edges
        .iter()
        .map(|(&pair, edges)| (pair, edges.as_slice()))
        .collect();
    pairs.sort_by_key(|value| value.0);
    let mut tasks = Vec::<(RegionPair, Vec<u64>)>::new();
    for (pair, pair_edges) in pairs {
        let pair_edges: HashSet<EdgeKey> = pair_edges.iter().copied().collect();
        for track in trace_edge_chains(&pair_edges, &junctions, stride) {
            if track.len() >= 2 {
                tasks.push((pair, track));
            }
        }
    }
    let results: Vec<_> = tasks
        .into_par_iter()
        .enumerate()
        .map(|(chain_id, (pair, track))| {
            #[cfg(not(feature = "diagnostics"))]
            let _ = (chain_id, pair);
            let raw_edges: Vec<(u64, u64)> = track
                .windows(2)
                .map(|vertices| (vertices[0], vertices[1]))
                .collect();
            #[cfg(feature = "diagnostics")]
            let target_stage_boundary = [
                57_usize, 221, 462, 1473, 1519, 1788, 2360, 2361, 2398, 2410, 3278, 3417, 3418,
                3419, 3912, 5322,
            ]
            .contains(&chain_id);
            #[cfg(feature = "diagnostics")]
            let diagnostic_spans = if stage_diagnostics_enabled && target_stage_boundary {
                raw_edges
                    .iter()
                    .flat_map(|&(start, end)| {
                        let edge = EdgeKey::new(start, end);
                        let mut pieces =
                            adaptive.edge_spans.get(&edge).cloned().unwrap_or_default();
                        if start != edge.0 {
                            pieces = pieces
                                .into_iter()
                                .rev()
                                .map(|piece| AdaptiveCurveSpan {
                                    start_parameter: piece.end_parameter,
                                    end_parameter: piece.start_parameter,
                                    ..piece
                                })
                                .collect();
                        }
                        pieces
                    })
                    .map(|piece| {
                        let curve = match piece.curve {
                            CurveSegment::Line { start, end } => [start, start, end, end],
                            CurveSegment::Cubic {
                                start,
                                first,
                                second,
                                end,
                            } => [start, first, second, end],
                        };
                        serde_json::json!({
                            "id": piece.master_id,
                            "start": piece.start_parameter,
                            "end": piece.end_parameter,
                            "curve": curve.map(|point| [point.x, point.y]),
                        })
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let closed = track.first() == track.last();
            let (raw, mut segments) = adaptive_chain_curves(&raw_edges, &adaptive, stride);
            if segments.is_empty() {
                segments = raw
                    .windows(2)
                    .map(|points| straight_cubic(points[0], points[1]))
                    .collect();
            }
            #[cfg(feature = "diagnostics")]
            let base_segments = segments.clone();
            let observed = |vertex: u64| {
                adaptive
                    .regularized_observations
                    .get(&vertex)
                    .copied()
                    .unwrap_or_else(|| point_from_vertex(vertex, stride))
            };
            let mut source: Vec<Point> = raw_edges.iter().map(|edge| observed(edge.0)).collect();
            source.push(observed(raw_edges[raw_edges.len() - 1].1));
            let fair_continuity_edges = raw_edges
                .iter()
                .filter(|&&(start, end)| {
                    adaptive
                        .edge_spans
                        .get(&EdgeKey::new(start, end))
                        .is_some_and(|spans| {
                            spans.iter().any(|span| {
                                adaptive
                                    .continuity_faired_master_ids
                                    .contains(&span.master_id)
                            })
                        })
                })
                .count();
            let source_corridor =
                if fair_continuity_edges > 0 && fair_continuity_edges * 2 >= raw_edges.len() {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
            if !raster_boundary_supported(
                &source,
                &sample_curve_sequence(&segments, 0.25),
                source_corridor,
            ) {
                let mut fallback = source.clone();
                fallback[0] = raw[0];
                let last = fallback.len() - 1;
                fallback[last] = raw[raw.len() - 1];
                let mut replacement = corridor_fallback_curves(&fallback, 1.0, None);
                if !boundary_corridor_supported(&source, &replacement, 1.0) {
                    replacement = fallback
                        .windows(2)
                        .map(|pair| straight_cubic(pair[0], pair[1]))
                        .collect();
                }
                segments = replacement;
            } else if segments.len() > 2
                && segments
                    .iter()
                    .copied()
                    .all(|curve| cubic_is_linear(curve, 1e-4))
            {
                let replacement = corridor_fallback_curves(&raw, 1.0, Some(1.0));
                if replacement.len() < segments.len()
                    && raster_boundary_supported(
                        &source,
                        &sample_curve_sequence(&replacement, 0.25),
                        1.0,
                    )
                {
                    segments = replacement;
                }
            }
            #[cfg(feature = "diagnostics")]
            let encode_diagnostic_curves = |values: &[CurveSegment]| {
                values
                    .iter()
                    .map(|curve| match *curve {
                        CurveSegment::Line { start, end } => serde_json::json!([
                            [start.x, start.y],
                            [
                                start.x + (end.x - start.x) / 3.0,
                                start.y + (end.y - start.y) / 3.0
                            ],
                            [
                                start.x + 2.0 * (end.x - start.x) / 3.0,
                                start.y + 2.0 * (end.y - start.y) / 3.0
                            ],
                            [end.x, end.y],
                        ]),
                        CurveSegment::Cubic {
                            start,
                            first,
                            second,
                            end,
                        } => serde_json::json!([
                            [start.x, start.y],
                            [first.x, first.y],
                            [second.x, second.y],
                            [end.x, end.y],
                        ]),
                    })
                    .collect::<Vec<_>>()
            };
            #[cfg(feature = "diagnostics")]
            let mut smoothing_candidate_diagnostics = Vec::new();
            if source.len() >= 16 && segments.len() > 1 {
                let mut candidate_points: Vec<Point> = track
                    .iter()
                    .map(|&vertex| point_from_vertex(vertex, stride))
                    .collect();
                candidate_points[0] = raw[0];
                let last = candidate_points.len() - 1;
                candidate_points[last] = raw[raw.len() - 1];
                let mut fixed_indices = HashSet::<usize>::new();
                for (index, &vertex) in track.iter().enumerate() {
                    if let Some(&regularized) = adaptive.regularized_observations.get(&vertex) {
                        candidate_points[index] = regularized;
                    }
                    if adaptive.regularized_fixed_points.contains(&vertex) {
                        fixed_indices.insert(index);
                    }
                }
                let candidate_start_tangent = (!closed && source.len() >= 48).then(|| {
                    let start = segments[0].start();
                    match segments[0] {
                        CurveSegment::Line { end, .. } => Point {
                            x: end.x - start.x,
                            y: end.y - start.y,
                        },
                        CurveSegment::Cubic { first, .. } => Point {
                            x: first.x - start.x,
                            y: first.y - start.y,
                        },
                    }
                });
                let candidate_end_tangent = (!closed && source.len() >= 48).then(|| {
                    let end = segments[segments.len() - 1].end();
                    match segments[segments.len() - 1] {
                        CurveSegment::Line { start, .. } => Point {
                            x: end.x - start.x,
                            y: end.y - start.y,
                        },
                        CurveSegment::Cubic { second, .. } => Point {
                            x: end.x - second.x,
                            y: end.y - second.y,
                        },
                    }
                });
                let mut best = segments.clone();
                for sigma in [1.25_f32, 1.5, 2.0, 2.5, 3.0, 4.0] {
                    let candidate = fit_shared_boundary_candidate(
                        &candidate_points,
                        closed,
                        std::f32::consts::FRAC_1_SQRT_2,
                        sigma,
                        70.0,
                        &fixed_indices,
                        candidate_start_tangent,
                        candidate_end_tangent,
                    );
                    let accepted = !candidate.is_empty()
                        && candidate.len() < best.len()
                        && raster_boundary_supported(
                            &source,
                            &sample_curve_sequence(&candidate, 0.25),
                            fairing_raster_corridor(),
                        );
                    #[cfg(feature = "diagnostics")]
                    if stage_diagnostics_enabled && target_stage_boundary {
                        smoothing_candidate_diagnostics.push(serde_json::json!({
                            "sigma": sigma,
                            "accepted": accepted,
                            "curves": encode_diagnostic_curves(&candidate),
                        }));
                    }
                    if accepted {
                        best = candidate;
                    }
                }
                segments = best;
            }
            #[cfg(feature = "diagnostics")]
            let corridor_segments = segments.clone();
            #[cfg(feature = "diagnostics")]
            let direct_diagnostics = if stage_diagnostics_enabled && target_stage_boundary {
                let catmull = bounded_fairing_shared_boundary(&source, &segments, 0.75);
                let least = least_squares_fairing_shared_boundary(&source, &segments);
                let catmull_reference = sample_curve_sequence(&segments, 0.35);
                let catmull_candidates = [
                    0.5_f32,
                    0.658_990_3,
                    0.868_536_5,
                    1.144_714_2,
                    1.508_711_2,
                    1.988_452_1,
                    2.620_741_4,
                    3.454_086_3,
                    4.552_419,
                    6.0,
                ]
                .into_iter()
                .map(|sigma| {
                    let source_corners = persistent_open_corners(&source)
                        .into_iter()
                        .map(|value| value.1)
                        .collect::<Vec<_>>();
                    let (candidate, _, anchors) = fairing_candidate_segments(
                        &catmull_reference,
                        0.75,
                        sigma,
                        &source_corners,
                    );
                    serde_json::json!({
                        "sigma": sigma,
                        "curves": encode_diagnostic_curves(&candidate),
                        "anchors": anchors.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>();
                let selected = if least.len() < catmull.len() {
                    least.clone()
                } else {
                    catmull.clone()
                };
                let (refined_catmull, refined_least) =
                    if selected.len() < segments.len() && selected.len() >= 6 {
                        (
                            bounded_fairing_shared_boundary(&source, &selected, 0.75),
                            least_squares_fairing_shared_boundary(&source, &selected),
                        )
                    } else {
                        (Vec::new(), Vec::new())
                    };
                serde_json::json!({
                    "baseline": segments.len(),
                    "catmull": catmull.len(),
                    "least": least.len(),
                    "selected": selected.len(),
                    "refined_catmull": refined_catmull.len(),
                    "refined_least": refined_least.len(),
                    "catmull_curves": encode_diagnostic_curves(&catmull),
                    "least_curves": encode_diagnostic_curves(&least),
                    "selected_curves": encode_diagnostic_curves(&selected),
                    "refined_catmull_curves": encode_diagnostic_curves(&refined_catmull),
                    "refined_least_curves": encode_diagnostic_curves(&refined_least),
                    "catmull_candidates": catmull_candidates,
                })
            } else {
                serde_json::Value::Null
            };
            // The direct fairing is the final model-selection stage in the
            // Python graph builder, after corridor fallback and compaction.
            segments = bounded_fairing_direct_shared_boundary(&source, &segments, closed, false);
            if !segments.is_empty() {
                for index in 0..segments.len().saturating_sub(1) {
                    let left = segments[index];
                    let right = segments[index + 1];
                    // Closed perceptual contours are split into independently
                    // fitted arcs. Their de Casteljau slices can land on
                    // opposite sides of one native raster corner even though
                    // both remain inside the accepted sqrt(2)-pixel source
                    // corridor. Rejoin only that bounded artificial seam;
                    // larger gaps still invalidate the shared loop below.
                    if left.end().distance(right.start())
                        <= std::f32::consts::SQRT_2 + 0.125
                    {
                        let joint = interpolate_point(left.end(), right.start(), 0.5);
                        let left_delta = Point {
                            x: joint.x - left.end().x,
                            y: joint.y - left.end().y,
                        };
                        let right_delta = Point {
                            x: joint.x - right.start().x,
                            y: joint.y - right.start().y,
                        };
                        segments[index] = match left {
                            CurveSegment::Line { start, .. } => {
                                CurveSegment::Line { start, end: joint }
                            }
                            CurveSegment::Cubic {
                                start,
                                first,
                                second,
                                ..
                            } => CurveSegment::Cubic {
                                start,
                                first,
                                second: Point {
                                    x: second.x + left_delta.x,
                                    y: second.y + left_delta.y,
                                },
                                end: joint,
                            },
                        };
                        segments[index + 1] = match right {
                            CurveSegment::Line { end, .. } => {
                                CurveSegment::Line { start: joint, end }
                            }
                            CurveSegment::Cubic {
                                first, second, end, ..
                            } => CurveSegment::Cubic {
                                start: joint,
                                first: Point {
                                    x: first.x + right_delta.x,
                                    y: first.y + right_delta.y,
                                },
                                second,
                                end,
                            },
                        };
                    }
                }
                let start_delta = Point {
                    x: raw[0].x - segments[0].start().x,
                    y: raw[0].y - segments[0].start().y,
                };
                segments[0] = match segments[0] {
                    CurveSegment::Line { end, .. } => CurveSegment::Line { start: raw[0], end },
                    CurveSegment::Cubic {
                        first, second, end, ..
                    } => CurveSegment::Cubic {
                        start: raw[0],
                        first: Point {
                            x: first.x + start_delta.x,
                            y: first.y + start_delta.y,
                        },
                        second,
                        end,
                    },
                };
                let last = segments.len() - 1;
                let end_delta = Point {
                    x: raw[raw.len() - 1].x - segments[last].end().x,
                    y: raw[raw.len() - 1].y - segments[last].end().y,
                };
                segments[last] = match segments[last] {
                    CurveSegment::Line { start, .. } => CurveSegment::Line {
                        start,
                        end: raw[raw.len() - 1],
                    },
                    CurveSegment::Cubic {
                        start,
                        first,
                        second,
                        ..
                    } => CurveSegment::Cubic {
                        start,
                        first,
                        second: Point {
                            x: second.x + end_delta.x,
                            y: second.y + end_delta.y,
                        },
                        end: raw[raw.len() - 1],
                    },
                };
            }
            segments = geometry_primitives::regularize(&source, &segments, 1.0, None, None);
            let discontinuous = segments.windows(2).any(|pair| {
                pair[0].end().distance(pair[1].start()) > 1e-3
            }) || (closed
                && segments
                    .first()
                    .zip(segments.last())
                    .is_some_and(|(first, last)| last.end().distance(first.start()) > 1e-3));
            let mut curve_downgraded = false;
            if discontinuous {
                // A fitted master may be valid in isolation yet disagree at
                // a sliced junction with another winning master. Never pass
                // that gap into the shared face assembler: restore this one
                // chain from its positioned source corridor, which remains
                // exactly reversible for both incident Paint faces.
                let mut positioned_source = source.clone();
                positioned_source[0] = raw[0];
                let last = positioned_source.len() - 1;
                positioned_source[last] = raw[raw.len() - 1];
                segments = corridor_fallback_curves(&positioned_source, 1.0, None);
                curve_downgraded = true;
            }
            #[cfg(feature = "diagnostics")]
            let stage_diagnostic =
                (stage_diagnostics_enabled && target_stage_boundary).then(|| {
                    (
                        chain_id.to_string(),
                    serde_json::json!({
                        "raw": raw.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                        "source": source.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                        "spans": diagnostic_spans,
                        "base": encode_diagnostic_curves(&base_segments),
                        "corridor": encode_diagnostic_curves(&corridor_segments),
                        "final": encode_diagnostic_curves(&segments),
                        "direct_diagnostics": direct_diagnostics,
                        "smoothing_candidates": smoothing_candidate_diagnostics,
                    }),
                    )
                });
            let mut points: Vec<Point> = segments.iter().map(|segment| segment.start()).collect();
            if !closed {
                if let Some(last) = segments.last() {
                    points.push(last.end());
                }
            }
            #[cfg(feature = "diagnostics")]
            let diagnostic = diagnostics_enabled.then(|| {
                shared_chain_diagnostic(
                    chain_id,
                    pair,
                    &raw_edges,
                    &segments,
                    closed,
                    stride,
                    &adaptive.regularized_observations,
                )
            });
            if !closed && points.is_empty() {
                // Keep the representation structurally valid even when a
                // degenerate adaptive span returned no curve.
                points.extend(
                    track
                        .iter()
                        .map(|&vertex| point_from_vertex(vertex, stride)),
                );
            } else if !closed && points.last().copied() != segments.last().map(|value| value.end())
            {
                // The branch above normally appended this endpoint.  Retain
                // the explicit check to mirror the open Python chain.
                if let Some(last) = segments.last() {
                    points.push(last.end());
                }
            }
            #[cfg(feature = "diagnostics")]
            let result = (
                SharedChain {
                    points,
                    segments,
                    closed,
                },
                raw_edges,
                curve_downgraded,
                diagnostic,
                stage_diagnostic,
            );
            #[cfg(not(feature = "diagnostics"))]
            let result = (
                SharedChain {
                    points,
                    segments,
                    closed,
                },
                raw_edges,
                curve_downgraded,
            );
            result
        })
        .collect();
    let mut shared_curve_downgrades = 0_usize;
    #[cfg(feature = "diagnostics")]
    for (chain_id, (chain, raw_edges, curve_downgraded, diagnostic, stage_diagnostic)) in
        results.into_iter().enumerate()
    {
        shared_curve_downgrades += usize::from(curve_downgraded);
        if let Some(diagnostic) = diagnostic {
            diagnostics.push(diagnostic);
        }
        if let Some((key, diagnostic)) = stage_diagnostic {
            stage_diagnostics.insert(key, diagnostic);
        }
        for (first, second) in raw_edges {
            lookup.insert(EdgeKey::new(first, second), (chain_id, first, second));
        }
        chains.push(chain);
    }
    #[cfg(not(feature = "diagnostics"))]
    for (chain_id, (chain, raw_edges, curve_downgraded)) in results.into_iter().enumerate() {
        shared_curve_downgrades += usize::from(curve_downgraded);
        for (first, second) in raw_edges {
            lookup.insert(EdgeKey::new(first, second), (chain_id, first, second));
        }
        chains.push(chain);
    }
    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_GEOMETRY_DIAGNOSTICS") {
        if let Ok(encoded) = serde_json::to_vec(&diagnostics) {
            let _ = std::fs::write(path, encoded);
        }
    }
    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_GEOMETRY_STAGE_DIAGNOSTICS") {
        if let Ok(encoded) = serde_json::to_vec(&stage_diagnostics) {
            let _ = std::fs::write(path, encoded);
        }
    }
    (
        chains,
        lookup,
        positions,
        adaptive.optimal_polygons,
        adaptive.continuity_faired_master_ids.len(),
        adaptive.regularized_excursions,
        adaptive.regularized_observations.len(),
        shared_curve_downgrades,
    )
}

fn oriented_segments(chain: &SharedChain, forward: bool) -> Vec<CurveSegment> {
    if forward {
        chain.segments.clone()
    } else {
        chain
            .segments
            .iter()
            .rev()
            .copied()
            .map(CurveSegment::reversed)
            .collect()
    }
}

fn oriented_points(chain: &SharedChain, forward: bool) -> Vec<Point> {
    let mut points = chain.points.clone();
    if !forward {
        points.reverse();
        if chain.closed {
            points.rotate_right(1);
        }
    }
    points
}

#[derive(Clone, Copy, Debug)]
enum SharedLoopFailure {
    MissingEdge,
    Discontinuous,
    Empty,
}

fn shared_region_loop(
    vertices: &[u64],
    chains: &[SharedChain],
    lookup: &EdgeChainLookup,
    cubics: &mut usize,
    lines: &mut usize,
) -> Result<(Vec<Point>, String), SharedLoopFailure> {
    if vertices.len() < 3 {
        return Err(SharedLoopFailure::Empty);
    }
    let references: Option<Vec<(usize, bool)>> = (0..vertices.len())
        .map(|index| {
            let first = vertices[index];
            let second = vertices[(index + 1) % vertices.len()];
            let &(chain, stored_first, stored_second) = lookup.get(&EdgeKey::new(first, second))?;
            Some((chain, first == stored_first && second == stored_second))
        })
        .collect();
    let Some(references) = references else {
        return Err(SharedLoopFailure::MissingEdge);
    };
    let start = (0..references.len())
        .find(|&index| {
            references[index].0 != references[(index + references.len() - 1) % references.len()].0
        })
        .unwrap_or(0);
    let mut runs = Vec::<(usize, bool)>::new();
    for offset in 0..references.len() {
        let value = references[(start + offset) % references.len()];
        if runs.last().map(|previous| previous.0) != Some(value.0) {
            runs.push(value);
        }
    }
    if runs.len() > 1 && runs.first().map(|value| value.0) == runs.last().map(|value| value.0) {
        runs.pop();
    }
    let mut all_segments = Vec::<CurveSegment>::new();
    let mut loop_points = Vec::<Point>::new();
    for &(chain_id, forward) in &runs {
        let chain = &chains[chain_id];
        let points = oriented_points(chain, forward);
        if loop_points.last().copied() == points.first().copied() {
            loop_points.extend(points.into_iter().skip(1));
        } else {
            loop_points.extend(points);
        }
        all_segments.extend(oriented_segments(chain, forward));
    }
    let discontinuities = all_segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| {
            let following = all_segments[(index + 1) % all_segments.len()].start();
            let gap = segment.end().distance(following);
            (gap > 1e-3).then_some((index, segment.end(), following, gap))
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "diagnostics")]
    if !discontinuities.is_empty() && std::env::var_os("PICVEC_SHARED_LOOP_DIAGNOSTICS").is_some() {
        eprintln!("picvec shared loop discontinuity: runs={runs:?} gaps={discontinuities:?}");
    }
    if !discontinuities.is_empty() {
        return Err(SharedLoopFailure::Discontinuous);
    }
    let Some(first) = all_segments.first().map(|segment| segment.start()) else {
        return Err(SharedLoopFailure::Empty);
    };
    let mut data = format!("M{} {}", fmt(first.x), fmt(first.y));
    for segment in all_segments {
        match segment {
            CurveSegment::Line { end, .. } => {
                data.push_str(&format!("L{} {}", fmt(end.x), fmt(end.y)));
                *lines += 1;
            }
            CurveSegment::Cubic {
                first, second, end, ..
            } => {
                data.push_str(&format!(
                    "C{} {},{} {},{} {}",
                    fmt(first.x),
                    fmt(first.y),
                    fmt(second.x),
                    fmt(second.y),
                    fmt(end.x),
                    fmt(end.y)
                ));
                *cubics += 1;
            }
        }
    }
    data.push('Z');
    Ok((loop_points, data))
}

pub fn open_path_data(points: &[Point]) -> String {
    if points.is_empty() {
        return String::new();
    }
    let mut output = format!("M{} {}", fmt(points[0].x), fmt(points[0].y));
    if points.len() == 2 {
        output.push_str(&format!("L{} {}", fmt(points[1].x), fmt(points[1].y)));
        return output;
    }
    for index in 0..points.len() - 1 {
        let current = points[index];
        let next = points[index + 1];
        let previous = if index == 0 {
            current
        } else {
            points[index - 1]
        };
        let after = if index + 2 >= points.len() {
            next
        } else {
            points[index + 2]
        };
        if is_smooth(previous, current, next) && is_smooth(current, next, after) {
            let tension = 0.62 / 6.0;
            let c1 = Point {
                x: current.x + (next.x - previous.x) * tension,
                y: current.y + (next.y - previous.y) * tension,
            };
            let c2 = Point {
                x: next.x - (after.x - current.x) * tension,
                y: next.y - (after.y - current.y) * tension,
            };
            output.push_str(&format!(
                "C{} {},{} {},{} {}",
                fmt(c1.x),
                fmt(c1.y),
                fmt(c2.x),
                fmt(c2.y),
                fmt(next.x),
                fmt(next.y)
            ));
        } else {
            output.push_str(&format!("L{} {}", fmt(next.x), fmt(next.y)));
        }
    }
    output
}

fn paint_order_ranks(segmentation: &Segmentation) -> Vec<usize> {
    let count = segmentation.regions.len();
    let mut border_counts = vec![0_usize; count];
    if segmentation.width > 0 && segmentation.height > 0 {
        for x in 0..segmentation.width {
            border_counts[segmentation.labels[x] as usize] += 1;
            border_counts[segmentation.labels[(segmentation.height - 1) * segmentation.width + x]
                as usize] += 1;
        }
        for y in 1..segmentation.height.saturating_sub(1) {
            border_counts[segmentation.labels[y * segmentation.width] as usize] += 1;
            border_counts
                [segmentation.labels[y * segmentation.width + segmentation.width - 1] as usize] +=
                1;
        }
    }
    let background = border_counts
        .iter()
        .enumerate()
        .max_by_key(|&(label, &area)| (area, std::cmp::Reverse(label)))
        .map(|(label, _)| label)
        .unwrap_or(0);
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(
        |&left, &right| match (left == background, right == background) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => segmentation.regions[right]
                .area
                .cmp(&segmentation.regions[left].area)
                .then_with(|| left.cmp(&right)),
        },
    );
    let mut ranks = vec![0_usize; count];
    for (rank, region) in order.into_iter().enumerate() {
        ranks[region] = rank;
    }
    ranks
}

fn point_in_loop(point: Point, polygon: &[Point]) -> bool {
    let mut inside = false;
    for index in 0..polygon.len() {
        let first = polygon[index];
        let second = polygon[(index + 1) % polygon.len()];
        if (first.y > point.y) != (second.y > point.y) {
            let crossing =
                (second.x - first.x) * (point.y - first.y) / (second.y - first.y) + first.x;
            if point.x < crossing {
                inside = !inside;
            }
        }
    }
    inside
}

fn hole_is_covered_by_later_regions(
    polygon: &[Point],
    region: usize,
    segmentation: &Segmentation,
    order_ranks: &[usize],
) -> bool {
    if polygon.len() < 3 || segmentation.width == 0 || segmentation.height == 0 {
        return false;
    }
    let minimum_x = polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as usize;
    let minimum_y = polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as usize;
    let maximum_x = (polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .max(0.0) as usize)
        .min(segmentation.width);
    let maximum_y = (polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .max(0.0) as usize)
        .min(segmentation.height);
    let mut covered_pixels = 0_usize;
    let mut covering_owner = None::<usize>;
    for y in minimum_y..maximum_y {
        for x in minimum_x..maximum_x {
            if !point_in_loop(
                Point {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                },
                polygon,
            ) {
                continue;
            }
            covered_pixels += 1;
            let owner = segmentation.labels[y * segmentation.width + x] as usize;
            if owner == region || order_ranks[owner] <= order_ranks[region] {
                return false;
            }
            match covering_owner {
                Some(previous) if previous != owner => return false,
                None => covering_owner = Some(owner),
                _ => {}
            }
        }
    }
    covered_pixels > 0
}

pub fn build(segmentation: &Segmentation) -> (Vec<RegionGeometry>, GeometrySummary) {
    build_internal(segmentation, None)
}

pub fn build_with_topology(
    segmentation: &Segmentation,
    topology: &HierarchicalTopology,
) -> (Vec<RegionGeometry>, GeometrySummary) {
    build_internal(segmentation, Some(topology))
}

fn build_internal(
    segmentation: &Segmentation,
    topology: Option<&HierarchicalTopology>,
) -> (Vec<RegionGeometry>, GeometrySummary) {
    let count = segmentation.regions.len();
    let order_ranks = paint_order_ranks(segmentation);
    let stride = segmentation.width + 1;
    let (edges, shared) = region_boundary_edges(segmentation, stride, topology);
    let pair_edges = pair_boundary_edges(segmentation, stride, topology);
    let source_edges = edges.iter().map(Vec::len).sum();
    let (
        shared_chains,
        shared_lookup,
        positions,
        adaptive_optimal_polygons,
        continuity_faired_masters,
        regularized_corner_excursions,
        regularized_corner_vertices,
        shared_curve_downgrades,
    ) = build_shared_chains(segmentation, stride, &edges, &pair_edges);
    let mut endpoint_degree = HashMap::<(i64, i64), usize>::new();
    let mut endpoint_order = Vec::<(i64, i64)>::new();
    for chain in &shared_chains {
        if chain.closed || chain.points.len() < 2 {
            continue;
        }
        for point in [chain.points[0], chain.points[chain.points.len() - 1]] {
            let key = (
                (point.x * 10_000.0).round() as i64,
                (point.y * 10_000.0).round() as i64,
            );
            if !endpoint_degree.contains_key(&key) {
                endpoint_order.push(key);
            }
            *endpoint_degree.entry(key).or_default() += 1;
        }
    }
    let paint_junctions = endpoint_order
        .into_iter()
        .filter_map(|(x, y)| {
            let degree = endpoint_degree[&(x, y)];
            (degree > 2).then_some(Point {
                x: x as f32 / 10_000.0,
                y: y as f32 / 10_000.0,
            })
        })
        .collect::<Vec<_>>();
    let canvas_corners = [
        Point { x: 0.0, y: 0.0 },
        Point {
            x: segmentation.width as f32,
            y: 0.0,
        },
        Point {
            x: segmentation.width as f32,
            y: segmentation.height as f32,
        },
        Point {
            x: 0.0,
            y: segmentation.height as f32,
        },
    ];
    // Python's shared-half-edge graph accepts the one canonical curve after
    // its source-corridor check; it never estimates a curved face's area from
    // only the curve endpoints.  The former Rust-only stabilization did just
    // that and downgraded hundreds of valid shallow cubics to pixel-grid
    // polylines.  Topology is already exact because both faces reference the
    // same de Casteljau intervals.
    let mut summary = GeometrySummary {
        regions: count,
        source_boundary_edges: source_edges,
        shared_boundary_edges: shared / 2,
        shared_curve_downgrades,
        adaptive_optimal_polygons,
        continuity_faired_masters,
        regularized_corner_excursions,
        regularized_corner_vertices,
        paint_junctions,
        ..GeometrySummary::default()
    };
    let mut geometries = Vec::with_capacity(count);
    for (region, region_edges) in edges.iter().enumerate().take(count) {
        let traced = trace_region_vertex_loops(region_edges, stride);
        let mut loops = Vec::<Vec<Point>>::new();
        let mut data = String::new();
        let mut occlusion_data = String::new();
        let mut removed_hole = false;
        for vertices in traced {
            let source_points: Vec<Point> = vertices
                .iter()
                .map(|&value| point_from_vertex(value, stride))
                .collect();
            let raw_points: Vec<Point> = vertices
                .iter()
                .map(|&value| {
                    positions
                        .get(&value)
                        .copied()
                        .unwrap_or_else(|| point_from_vertex(value, stride))
                })
                .collect();
            let source_area = signed_area(&source_points);
            if source_area.abs() < 0.5 {
                continue;
            }
            let covered_hole = source_area < 0.0
                && hole_is_covered_by_later_regions(
                    &source_points,
                    region,
                    segmentation,
                    &order_ranks,
                );
            if covered_hole {
                summary.covered_holes_removed += 1;
                removed_hole = true;
            }
            match shared_region_loop(
                &vertices,
                &shared_chains,
                &shared_lookup,
                &mut summary.cubic_segments,
                &mut summary.line_segments,
            ) {
                Ok((points, path)) => {
                    loops.push(points);
                    data.push_str(&path);
                    if !covered_hole {
                        occlusion_data.push_str(&path);
                    }
                    continue;
                }
                Err(SharedLoopFailure::MissingEdge) => summary.shared_loop_missing_edges += 1,
                Err(SharedLoopFailure::Discontinuous) => summary.shared_loop_discontinuities += 1,
                Err(SharedLoopFailure::Empty) => summary.shared_loop_invalid_areas += 1,
            }
            summary.shared_loop_fallbacks += 1;
            let simplified = preserve_closed_points(
                simplify_grid_closed(&raw_points, std::f32::consts::FRAC_1_SQRT_2),
                &raw_points,
                &canvas_corners,
            );
            let simplified_area = signed_area(&simplified);
            let points = if simplified.len() >= 3
                && simplified_area.signum() == source_area.signum()
                && simplified_area.abs() >= 0.05 * source_area.abs()
            {
                simplified
            } else {
                let positioned = remove_collinear(&raw_points);
                let positioned_area = signed_area(&positioned);
                if positioned.len() >= 3
                    && positioned_area.signum() == source_area.signum()
                    && positioned_area.abs() >= 0.05 * source_area.abs()
                {
                    positioned
                } else {
                    remove_collinear(&source_points)
                }
            };
            if points.len() >= 3 {
                let path = closed_path_data(
                    &points,
                    &mut summary.cubic_segments,
                    &mut summary.line_segments,
                );
                data.push_str(&path);
                if !covered_hole {
                    occlusion_data.push_str(&path);
                }
                loops.push(points);
            }
        }
        loops.sort_by(|a, b| signed_area(b).abs().total_cmp(&signed_area(a).abs()));
        if loops.is_empty() {
            summary.empty_regions += 1;
            summary.empty_region_pixels += segmentation.regions[region].area;
            summary.largest_empty_region = summary
                .largest_empty_region
                .max(segmentation.regions[region].area);
        }
        summary.loops += loops.len();
        summary.simplified_vertices += loops.iter().map(Vec::len).sum::<usize>();
        // Keep the one-face-per-region shared path intact here.  The Python
        // pipeline recognizes exact primitives only after the structural
        // layer has been composited, as part of the final render-gated SVG
        // optimization.  Recognizing raster rectangles at this stage changes
        // both the serialized geometry and the optimizer's candidate tree.
        let primitive = None;
        geometries.push(RegionGeometry {
            region: region as u32,
            loops,
            path_data: data,
            occlusion_path_data: removed_hole.then_some(occlusion_data),
            primitive,
        });
    }
    // Python serializes the dominant border face first, then all remaining
    // faces by descending raster area with the label as the stable tie-break.
    // Preserve that order before any render-dependent structural selection.
    geometries.sort_by(|left, right| {
        let left_region = left.region as usize;
        let right_region = right.region as usize;
        order_ranks[left_region]
            .cmp(&order_ranks[right_region])
            .then_with(|| left_region.cmp(&right_region))
    });
    (geometries, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb_to_lab;
    use crate::raster::Raster;
    use crate::segment::{RegionStats, Segmentation, SegmentationSummary};

    #[test]
    fn alpha_fairing_retains_rectangle_corners_and_small_islands() {
        for (left, top, right, bottom) in [(20, 24, 76, 72), (40, 40, 41, 41)] {
            let values = (0..96 * 96)
                .map(|index| {
                    let (x, y) = (index % 96, index / 96);
                    if (left..right).contains(&x) && (top..bottom).contains(&y) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let matte = crate::chroma::AlphaMatte::new(96, 96, values);
            let contours = matte.isocontours(0.5);
            assert_eq!(contours.len(), 1);
            let (_, curves) = fit_alpha_contour(&contours[0]);
            let samples = sample_curve_sequence(&curves, 0.1);
            for (x, y) in [(left, top), (right, top), (right, bottom), (left, bottom)] {
                let corner = Point {
                    x: x as f32,
                    y: y as f32,
                };
                let distance = samples
                    .iter()
                    .map(|point| point.distance(corner))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    distance < 0.8,
                    "corner rounded away: {corner:?}, distance={distance}"
                );
            }
            let area = samples
                .windows(2)
                .map(|p| p[0].x * p[1].y - p[1].x * p[0].y)
                .sum::<f32>()
                .abs()
                * 0.5;
            assert!(
                area > 0.3 * ((right - left) * (bottom - top)) as f32,
                "island collapsed"
            );
        }
    }

    #[test]
    fn raster_alpha_circle_has_smooth_turning_without_losing_its_shape() {
        let size = 192;
        let radius = 80.0;
        let values = (0..size * size)
            .map(|index| {
                let x = (index % size) as f32 + 0.5 - 96.0;
                let y = (index / size) as f32 + 0.5 - 96.0;
                if x.hypot(y) < radius {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let matte = crate::chroma::AlphaMatte::new(size, size, values);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);
        let (_, curves) = fit_alpha_contour(&contours[0]);
        let samples = sample_curve_sequence(&curves, 0.5);
        for point in &samples {
            assert!(
                ((point.x - 96.0).hypot(point.y - 96.0) - radius).abs() < 0.9,
                "smoothed silhouette moved too far: {point:?}"
            );
        }
        let directions = samples
            .windows(2)
            .filter(|pair| pair[0].distance(pair[1]) > 1e-4)
            .map(|pair| {
                normalized(Point {
                    x: pair[1].x - pair[0].x,
                    y: pair[1].y - pair[0].y,
                })
            })
            .collect::<Vec<_>>();
        let total_turn: f32 = directions
            .iter()
            .zip(directions.iter().cycle().skip(1))
            .map(|(a, b)| (a.x * b.y - a.y * b.x).atan2(a.x * b.x + a.y * b.y).abs())
            .sum();
        assert!(
            total_turn < 8.0,
            "raster ripples remain: total turn {total_turn}"
        );
    }

    #[test]
    fn smooth_closed_fit_has_no_corner_at_its_storage_origin() {
        let mut points = (0..48)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / 48.0;
                Point {
                    x: 20.0 + 12.0 * angle.cos(),
                    y: 20.0 + 7.0 * angle.sin(),
                }
            })
            .collect::<Vec<_>>();
        points.push(points[0]);

        let curves = fit_shared_boundary_candidate(
            &points,
            true,
            std::f32::consts::FRAC_1_SQRT_2,
            1.5,
            70.0,
            &HashSet::new(),
            None,
            None,
        );

        let first = curves.first().copied().expect("closed curve");
        let last = curves.last().copied().expect("closed curve");
        assert_eq!(first.start(), points[0]);
        assert_eq!(last.end(), points[0]);
        let outgoing = match first {
            CurveSegment::Cubic { start, first, .. } => normalized(Point {
                x: first.x - start.x,
                y: first.y - start.y,
            }),
            CurveSegment::Line { start, end } => normalized(Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }),
        };
        let incoming = match last {
            CurveSegment::Cubic { second, end, .. } => normalized(Point {
                x: end.x - second.x,
                y: end.y - second.y,
            }),
            CurveSegment::Line { start, end } => normalized(Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }),
        };

        assert!(incoming.x * outgoing.x + incoming.y * outgoing.y > 0.999_999);
        assert!((incoming.x * outgoing.y - incoming.y * outgoing.x).abs() < 1e-4);
    }

    #[test]
    fn structural_endpoint_constraint_preserves_anchor_and_sets_g1_direction() {
        let mut curves = vec![CurveSegment::Cubic {
            start: Point { x: 0.0, y: 0.0 },
            first: Point { x: 1.0, y: 1.0 },
            second: Point { x: 3.0, y: 1.0 },
            end: Point { x: 4.0, y: 0.0 },
        }];
        constrain_structural_endpoint_tangents(
            &mut curves,
            Some(Point { x: 1.0, y: 0.0 }),
            Some(Point { x: 1.0, y: 0.0 }),
        );
        let CurveSegment::Cubic {
            start,
            first,
            second,
            end,
        } = curves[0]
        else {
            panic!("expected cubic");
        };
        assert_eq!(start, Point { x: 0.0, y: 0.0 });
        assert_eq!(end, Point { x: 4.0, y: 0.0 });
        assert_eq!(first.y, start.y);
        assert_eq!(second.y, end.y);
        assert!(first.x > start.x);
        assert!(second.x < end.x);
    }

    #[test]
    fn connected_curve_endpoint_stays_inside_shared_vertex_budget() {
        let stride = 64;
        let start = vertex_id(10, 10, stride);
        let end = vertex_id(11, 10, stride);
        let edge = EdgeKey::new(start, end);
        let spans = HashMap::from([(
            edge,
            vec![AdaptiveCurveSpan {
                master_id: 0,
                curve: straight_cubic(Point { x: 0.0, y: 10.0 }, Point { x: 30.0, y: 10.0 }),
                start_parameter: 0.5,
                end_parameter: 0.75,
            }],
        )]);

        let (at_start, _) =
            connected_adaptive_edge_geometry(&spans, start, end, false, stride).unwrap();
        let (at_end, _) =
            connected_adaptive_edge_geometry(&spans, start, end, true, stride).unwrap();

        assert_eq!(at_start, Point { x: 10.25, y: 10.0 });
        assert_eq!(at_end, Point { x: 11.25, y: 10.0 });
    }

    #[test]
    fn potrace_default_corner_threshold_keeps_a_compact_cusp() {
        let mut master_id = 0;
        let smooth = potrace_corner_curves(
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 3.0 },
            Point { x: 1.0, y: 0.0 },
            [0.0, 1.0, 2.0],
            1.2,
            &mut master_id,
        );
        let sharp = potrace_corner_curves(
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 3.0 },
            Point { x: 1.0, y: 0.0 },
            [0.0, 1.0, 2.0],
            1.0,
            &mut master_id,
        );

        assert_eq!(smooth.len(), 1);
        assert_eq!(sharp.len(), 2);
        assert_eq!(sharp[0].3.end(), Point { x: 0.0, y: 3.0 });
        assert_eq!(sharp[1].3.start(), Point { x: 0.0, y: 3.0 });
    }

    #[test]
    fn adjacent_masters_share_their_raster_vertex_exactly() {
        let stride = 64;
        let first = vertex_id(10, 10, stride);
        let joint = vertex_id(11, 10, stride);
        let last = vertex_id(12, 10, stride);
        let mut geometry = AdaptiveBoundaryGeometry::default();
        geometry.edge_spans.insert(
            EdgeKey::new(first, joint),
            vec![AdaptiveCurveSpan {
                master_id: 1,
                curve: straight_cubic(Point { x: 10.0, y: 10.0 }, Point { x: 11.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );
        geometry.edge_spans.insert(
            EdgeKey::new(joint, last),
            vec![AdaptiveCurveSpan {
                master_id: 2,
                curve: straight_cubic(Point { x: 8.5, y: 10.0 }, Point { x: 12.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );

        let (_, curves) =
            adaptive_chain_curves(&[(first, joint), (joint, last)], &geometry, stride);

        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].end(), Point { x: 11.0, y: 10.0 });
        assert_eq!(curves[1].start(), Point { x: 11.0, y: 10.0 });
    }

    #[test]
    fn distant_master_handoff_does_not_translate_a_control_handle() {
        let stride = 64;
        let first = vertex_id(10, 10, stride);
        let joint = vertex_id(11, 10, stride);
        let last = vertex_id(12, 10, stride);
        let mut geometry = AdaptiveBoundaryGeometry::default();
        geometry.edge_spans.insert(
            EdgeKey::new(first, joint),
            vec![AdaptiveCurveSpan {
                master_id: 1,
                curve: straight_cubic(Point { x: 10.0, y: 10.0 }, Point { x: 11.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );
        geometry.edge_spans.insert(
            EdgeKey::new(joint, last),
            vec![AdaptiveCurveSpan {
                master_id: 2,
                curve: straight_cubic(Point { x: 15.0, y: 10.0 }, Point { x: 12.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );

        let (_, curves) =
            adaptive_chain_curves(&[(first, joint), (joint, last)], &geometry, stride);

        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].end(), Point { x: 11.0, y: 10.0 });
        assert_eq!(curves[1].start(), Point { x: 15.0, y: 10.0 });
    }

    #[test]
    fn perceptually_close_adjacent_patch_shares_continuity_class() {
        let labs = vec![
            Lab {
                l: 20.0,
                a: 0.0,
                b: 0.0,
            },
            Lab {
                l: 74.8,
                a: -10.8,
                b: -15.4,
            },
            Lab {
                l: 76.0,
                a: -10.0,
                b: -14.5,
            },
            Lab {
                l: 100.0,
                a: 0.0,
                b: 0.0,
            },
        ];
        let adjacency = vec![
            HashMap::from([(2, 8)]),
            HashMap::from([(2, 20), (3, 8)]),
            HashMap::from([(0, 8), (1, 20)]),
            HashMap::from([(1, 8)]),
        ];

        let classes = contextual_continuity_classes(&labs, &adjacency, &[true, true, false, true]);

        assert_eq!(classes[2], classes[1]);
        assert_ne!(classes[0], classes[1]);
        assert_ne!(classes[3], classes[1]);
    }

    #[test]
    fn disconnected_or_visibly_distinct_neutral_faces_keep_separate_classes() {
        let labs = vec![
            Lab {
                l: 20.0,
                a: 0.0,
                b: 0.0,
            },
            Lab {
                l: 35.0,
                a: 1.0,
                b: -1.0,
            },
            Lab {
                l: 72.0,
                a: -18.0,
                b: -22.0,
            },
            Lab {
                l: 36.0,
                a: 0.0,
                b: 0.0,
            },
        ];
        let adjacency = vec![
            HashMap::from([(1, 12)]),
            HashMap::from([(0, 12), (2, 24)]),
            HashMap::from([(1, 24)]),
            HashMap::new(),
        ];

        let classes = contextual_continuity_classes(&labs, &adjacency, &[true; 4]);

        assert_ne!(classes[0], classes[1]);
        assert_ne!(classes[1], classes[2]);
        assert_ne!(classes[1], classes[3]);
    }

    #[test]
    fn coreless_line_cap_does_not_split_supported_surface_contour() {
        let labs = vec![
            Lab {
                l: 92.0,
                a: 0.0,
                b: 0.0,
            },
            Lab {
                l: 54.0,
                a: 62.0,
                b: 38.0,
            },
            Lab {
                l: 59.0,
                a: 58.0,
                b: 34.0,
            },
            Lab {
                l: 18.0,
                a: 20.0,
                b: 12.0,
            },
        ];
        // The coreless cap touches two durable shades of one surface over
        // nine edges and the exterior face over only two edges.
        let adjacency = vec![
            HashMap::from([(3, 2)]),
            HashMap::from([(3, 5)]),
            HashMap::from([(3, 4)]),
            HashMap::from([(0, 2), (1, 5), (2, 4)]),
        ];

        let classes = contextual_continuity_classes(&labs, &adjacency, &[true, true, true, false]);

        assert_eq!(classes[1], classes[2]);
        assert_eq!(classes[3], classes[1]);
        assert_ne!(classes[3], classes[0]);
    }

    #[test]
    fn closed_continuity_split_preserves_every_boundary_edge() {
        let stride = 64;
        let mut track = Vec::new();
        for x in 0..=48 {
            track.push(vertex_id(x, 0, stride));
        }
        for y in 1..=20 {
            track.push(vertex_id(48, y, stride));
        }
        for x in (0..48).rev() {
            track.push(vertex_id(x, 20, stride));
        }
        for y in (1..20).rev() {
            track.push(vertex_id(0, y, stride));
        }
        track.push(track[0]);

        let arcs = split_closed_continuity_track(&track, stride);

        assert!(arcs.len() >= 2);
        assert!(arcs
            .iter()
            .all(|arc| arc.len() >= 2 && arc[0] != arc[arc.len() - 1]));
        assert_eq!(
            arcs.iter().map(|arc| arc.len() - 1).sum::<usize>(),
            track.len() - 1
        );
        assert!(arcs
            .iter()
            .any(|arc| is_shallow_continuity_arc(arc, stride)));
    }

    #[test]
    fn closed_material_contour_is_fitted_across_quantized_face_transitions() {
        let width = 96;
        let height = 72;
        let centre_x = 48.0_f32;
        let centre_y = 36.0_f32;
        let colours = [
            [0.90, 0.10, 0.08],
            [0.025, 0.025, 0.025],
            [0.06, 0.06, 0.06],
            [0.11, 0.11, 0.11],
        ];
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = (x as f32 + 0.5 - centre_x) / 32.0;
                    let dy = (y as f32 + 0.5 - centre_y) / 24.0;
                    if dx * dx + dy * dy > 1.0 {
                        0
                    } else if y < 30 {
                        1
                    } else if y < 43 {
                        2
                    } else {
                        3
                    }
                })
            })
            .collect();
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| colours[label as usize])
                .collect(),
        );
        let regions = (0..colours.len())
            .map(|label| {
                let pixels: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &owner)| (owner as usize == label).then_some(index))
                    .collect();
                RegionStats {
                    id: label as u32,
                    area: pixels.len(),
                    min_x: pixels.iter().map(|index| index % width).min().unwrap(),
                    min_y: pixels.iter().map(|index| index / width).min().unwrap(),
                    max_x: pixels.iter().map(|index| index % width + 1).max().unwrap(),
                    max_y: pixels.iter().map(|index| index / width + 1).max().unwrap(),
                    mean_rgb: colours[label],
                    mean_lab: rgb_to_lab(colours[label]),
                }
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2, 3],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let (_, summary) = build(&segmentation);

        assert!(summary.continuity_faired_masters > 0, "{summary:?}");
        assert_eq!(summary.shared_loop_fallbacks, 0, "{summary:?}");
        assert_eq!(summary.shared_loop_discontinuities, 0, "{summary:?}");
    }

    #[test]
    fn rectangular_region_remains_shared_path_before_final_optimization() {
        let segmentation = Segmentation {
            width: 8,
            height: 6,
            labels: vec![0; 48],
            paint_keys: vec![0],
            paint_samples: vec![true; 48],
            canonical: Raster::blank(8, 6, [0.0; 3]),
            regions: vec![RegionStats {
                id: 0,
                area: 48,
                min_x: 0,
                min_y: 0,
                max_x: 8,
                max_y: 6,
                mean_rgb: [0.0; 3],
                mean_lab: rgb_to_lab([0.0; 3]),
            }],
            summary: SegmentationSummary {
                initial_regions: 1,
                merged_regions: 1,
                effective_minimum_area: 1,
                local_minimum_area: 1,
                local_median_area: 1,
                local_maximum_area: 1,
                ..SegmentationSummary::default()
            },
        };
        let (geometry, _) = build(&segmentation);
        assert!(geometry[0].primitive.is_none());
        assert!(!geometry[0].path_data.is_empty());
    }

    #[test]
    fn only_holes_owned_entirely_by_later_faces_are_removed() {
        let polygon = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 1.0, y: 4.0 },
            Point { x: 4.0, y: 4.0 },
            Point { x: 4.0, y: 1.0 },
        ];
        let mut labels = vec![0_u32; 25];
        for y in 1..4 {
            for x in 1..4 {
                labels[y * 5 + x] = 1;
            }
        }
        let segmentation = Segmentation {
            width: 5,
            height: 5,
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; 25],
            canonical: Raster::blank(5, 5, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: 16,
                    min_x: 0,
                    min_y: 0,
                    max_x: 5,
                    max_y: 5,
                    mean_rgb: [0.0; 3],
                    mean_lab: rgb_to_lab([0.0; 3]),
                },
                RegionStats {
                    id: 1,
                    area: 9,
                    min_x: 1,
                    min_y: 1,
                    max_x: 4,
                    max_y: 4,
                    mean_rgb: [1.0; 3],
                    mean_lab: rgb_to_lab([1.0; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        assert!(hole_is_covered_by_later_regions(
            &polygon,
            0,
            &segmentation,
            &[0, 1]
        ));
        assert!(!hole_is_covered_by_later_regions(
            &polygon,
            0,
            &segmentation,
            &[1, 0]
        ));
        let (geometry, summary) = build(&segmentation);
        let outer = geometry.iter().find(|item| item.region == 0).unwrap();
        assert_eq!(summary.covered_holes_removed, 1);
        assert!(outer.occlusion_path_data.is_some());
        assert!(outer.occlusion_path_data.as_ref().unwrap().len() < outer.path_data.len());
    }

    #[test]
    fn closed_simplification_keeps_canvas_corner() {
        let mut raw = Vec::<Point>::new();
        for x in 0..=32 {
            raw.push(Point {
                x: x as f32,
                y: 0.0,
            });
        }
        for y in 1..=12 {
            raw.push(Point {
                x: 32.0,
                y: y as f32,
            });
        }
        raw.extend([
            Point { x: 20.0, y: 9.0 },
            Point { x: 8.0, y: 6.0 },
            Point { x: 0.0, y: 3.0 },
        ]);
        let corner = Point { x: 32.0, y: 0.0 };
        let simplified = preserve_closed_points(
            simplify_grid_closed(&raw, std::f32::consts::FRAC_1_SQRT_2),
            &raw,
            &[corner],
        );
        assert!(simplified.contains(&corner));
    }

    #[test]
    fn bounded_fairing_removes_raster_staircase_without_leaving_source_corridor() {
        let mut raw = vec![Point { x: 0.0, y: 0.0 }];
        let mut y = 0.0_f32;
        for x in 0..96 {
            raw.push(Point {
                x: (x + 1) as f32,
                y,
            });
            if (x + 1) % 4 == 0 {
                y += 1.0;
                raw.push(Point {
                    x: (x + 1) as f32,
                    y,
                });
            }
        }
        let baseline = simplify_open(&raw, 0.55);
        let fair = bounded_fairing_open(&raw, 0.55);
        assert!(
            fair.len() < baseline.len(),
            "fair={} baseline={}",
            fair.len(),
            baseline.len()
        );

        let source = resample_open_polyline(&raw, 0.25);
        let candidate = sample_open_catmull(&fair, 0.25);
        let maximum = std::f32::consts::SQRT_2;
        assert!(nearest_sample_distances(&source, &candidate, maximum).is_some());
        assert!(nearest_sample_distances(&candidate, &source, maximum).is_some());
    }

    #[test]
    fn bounded_fairing_preserves_a_persistent_right_angle() {
        let mut raw: Vec<Point> = (0..=32)
            .map(|x| Point {
                x: x as f32,
                y: 0.0,
            })
            .collect();
        raw.extend((1..=32).map(|y| Point {
            x: 32.0,
            y: y as f32,
        }));
        let fair = bounded_fairing_open(&raw, 0.55);
        let corner = Point { x: 32.0, y: 0.0 };
        let corner_error = sample_open_catmull(&fair, 0.1)
            .into_iter()
            .map(|point| point.distance(corner))
            .fold(f32::INFINITY, f32::min);
        assert!(corner_error <= 0.25, "corner error={corner_error}");
    }

    #[test]
    fn short_cornerless_boundary_can_reject_a_one_pixel_phase_reversal() {
        let mut raw = vec![Point { x: 746.0, y: 574.0 }];
        for (target_x, target_y) in [
            (750.0, 574.0),
            (750.0, 572.0),
            (755.0, 572.0),
            (755.0, 571.0),
            (760.0, 571.0),
            (760.0, 570.0),
            (763.0, 570.0),
            (763.0, 571.0),
            (767.0, 571.0),
            (767.0, 570.0),
            (773.0, 570.0),
            (773.0, 569.0),
            (778.0, 569.0),
            (778.0, 568.0),
            (784.0, 568.0),
            (784.0, 567.0),
            (790.0, 567.0),
            (790.0, 566.0),
            (793.0, 566.0),
        ] {
            while (raw.last().unwrap().x - target_x).abs() > 1e-6 {
                let previous = *raw.last().unwrap();
                raw.push(Point {
                    x: previous.x + (target_x - previous.x).signum(),
                    y: previous.y,
                });
            }
            while (raw.last().unwrap().y - target_y).abs() > 1e-6 {
                let previous = *raw.last().unwrap();
                raw.push(Point {
                    x: previous.x,
                    y: previous.y + (target_y - previous.y).signum(),
                });
            }
        }
        let mut master_id = 0;
        let (_, tagged) =
            potrace_master_curves(&raw, false, 0.5, 1.2, &mut master_id, &HashSet::new());
        let baseline: Vec<CurveSegment> = tagged.iter().map(|value| value.3).collect();
        let catmull = bounded_fairing_shared_boundary(&raw, &baseline, 0.75);
        let least = least_squares_fairing_shared_boundary(&raw, &baseline);
        let fair = bounded_fairing_direct_shared_boundary(&raw, &baseline, false, false);

        assert!(persistent_open_corners(&raw).is_empty());
        assert!(
            fair.len() < baseline.len(),
            "fair={} baseline={} catmull={} least={}",
            fair.len(),
            baseline.len(),
            catmull.len(),
            least.len()
        );
        assert!(raster_boundary_supported(
            &raw,
            &sample_curve_sequence(&fair, 0.25),
            std::f32::consts::SQRT_2 + 0.25,
        ));
    }

    #[test]
    fn adjacent_faces_reuse_one_exact_reversed_boundary() {
        let width = 6;
        let height = 4;
        let labels: Vec<u32> = (0..height)
            .flat_map(|_| (0..width).map(|x| u32::from(x >= 3)))
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: 12,
                    min_x: 0,
                    min_y: 0,
                    max_x: 3,
                    max_y: 4,
                    mean_rgb: [0.5; 3],
                    mean_lab: rgb_to_lab([0.5; 3]),
                },
                RegionStats {
                    id: 1,
                    area: 12,
                    min_x: 3,
                    min_y: 0,
                    max_x: 6,
                    max_y: 4,
                    mean_rgb: [0.5; 3],
                    mean_lab: rgb_to_lab([0.5; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (directed_edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pair_edges = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, _, _, _, _, _, _) =
            build_shared_chains(&segmentation, stride, &directed_edges, &pair_edges);
        let chain_ids: Vec<usize> = (0..height)
            .map(|y| lookup[&EdgeKey::new(vertex_id(3, y, stride), vertex_id(3, y + 1, stride))].0)
            .collect();
        assert!(chain_ids.iter().all(|&value| value == chain_ids[0]));
        let forward = oriented_segments(&chains[chain_ids[0]], true);
        let restored: Vec<CurveSegment> = oriented_segments(&chains[chain_ids[0]], false)
            .into_iter()
            .rev()
            .map(CurveSegment::reversed)
            .collect();
        assert_eq!(forward, restored);
    }

    #[test]
    fn diagonal_grid_staircase_is_one_continuous_line() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 1.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 3.0, y: 2.0 },
            Point { x: 3.0, y: 3.0 },
        ];
        assert_eq!(
            simplify_grid_open(&points, std::f32::consts::FRAC_1_SQRT_2),
            vec![points[0], points[points.len() - 1]]
        );
        // The production shared-boundary path uses the Potrace dynamic
        // program, whose midpoint observations must make the same
        // minimum-description choice without preserving turn vertices.
        assert_eq!(potrace_optimal_polygon(&points, 0.5), vec![0, 6]);
    }

    #[test]
    fn material_transition_slices_one_continuous_master_boundary() {
        let width = 7;
        let height = 7;
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| if x < 3 { 0 } else { 1 + u32::from(y >= 3) }))
            .collect();
        let areas = [3 * height, 4 * 3, 4 * 4];
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: (0..3)
                .map(|id| RegionStats {
                    id: id as u32,
                    area: areas[id],
                    min_x: if id == 0 { 0 } else { 3 },
                    min_y: if id == 2 { 3 } else { 0 },
                    max_x: if id == 0 { 3 } else { width },
                    max_y: if id == 1 { 3 } else { height },
                    mean_rgb: [id as f32 * 0.3; 3],
                    mean_lab: rgb_to_lab([id as f32 * 0.3; 3]),
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (directed_edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pair_edges = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, _, _, _, _, _, _) =
            build_shared_chains(&segmentation, stride, &directed_edges, &pair_edges);
        let upper = lookup[&EdgeKey::new(vertex_id(3, 2, stride), vertex_id(3, 3, stride))].0;
        let lower = lookup[&EdgeKey::new(vertex_id(3, 3, stride), vertex_id(3, 4, stride))].0;
        assert_ne!(upper, lower);
        let endpoints = |chain: &SharedChain| {
            [
                chain.segments.first().unwrap().start(),
                chain.segments.last().unwrap().end(),
            ]
        };
        let junction = endpoints(&chains[upper])
            .into_iter()
            .find(|first| {
                endpoints(&chains[lower])
                    .into_iter()
                    .any(|second| first.distance(second) <= 1e-5)
            })
            .expect("material runs share one adjusted topology vertex");
        let tangent_away = |chain: &SharedChain| {
            chain.segments.iter().find_map(|segment| match *segment {
                CurveSegment::Line { start, end } if start.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: end.x - start.x,
                        y: end.y - start.y,
                    })
                }
                CurveSegment::Line { start, end } if end.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: start.x - end.x,
                        y: start.y - end.y,
                    })
                }
                CurveSegment::Cubic { start, first, .. } if start.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: first.x - start.x,
                        y: first.y - start.y,
                    })
                }
                CurveSegment::Cubic { second, end, .. } if end.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: second.x - end.x,
                        y: second.y - end.y,
                    })
                }
                _ => None,
            })
        };
        let first = tangent_away(&chains[upper]).expect("upper chain reaches junction");
        let second = tangent_away(&chains[lower]).expect("lower chain reaches junction");
        let cross = first.x * second.y - first.y * second.x;
        let dot = first.x * second.x + first.y * second.y;
        assert!(cross.abs() <= 1e-5);
        assert!(dot < 0.0);
        let (_, summary) = build(&segmentation);
        assert_eq!(summary.shared_loop_fallbacks, 0, "{summary:?}");
    }

    #[test]
    fn raster_supported_disc_remains_shared_path_before_final_optimization() {
        let width = 31;
        let height = 31;
        let centre = (15.5_f32, 15.5_f32);
        let radius = 9.5_f32;
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = x as f32 + 0.5 - centre.0;
                    let dy = y as f32 + 0.5 - centre.1;
                    u32::from(dx * dx + dy * dy <= radius * radius)
                })
            })
            .collect();
        let area = labels.iter().filter(|&&label| label == 1).count();
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: width * height - area,
                    min_x: 0,
                    min_y: 0,
                    max_x: width,
                    max_y: height,
                    mean_rgb: [1.0; 3],
                    mean_lab: rgb_to_lab([1.0; 3]),
                },
                RegionStats {
                    id: 1,
                    area,
                    min_x: 6,
                    min_y: 6,
                    max_x: 25,
                    max_y: 25,
                    mean_rgb: [0.0; 3],
                    mean_lab: rgb_to_lab([0.0; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let (geometry, _) = build(&segmentation);
        assert!(geometry[1].primitive.is_none());
        assert!(!geometry[1].path_data.is_empty());
    }

    #[test]
    fn every_binary_three_by_three_partition_has_closed_faces() {
        let width = 3;
        let height = 3;
        for bits in 1_u16..(1_u16 << 9) - 1 {
            let labels: Vec<u32> = (0..9)
                .map(|index| u32::from(bits & (1 << index) != 0))
                .collect();
            let first_area = labels.iter().filter(|&&label| label == 0).count();
            let second_area = labels.len() - first_area;
            let segmentation = Segmentation {
                width,
                height,
                labels: labels.clone(),
                paint_keys: vec![0, 1],
                paint_samples: vec![true; width * height],
                canonical: Raster::blank(width, height, [0.0; 3]),
                regions: [first_area, second_area]
                    .into_iter()
                    .enumerate()
                    .map(|(id, area)| RegionStats {
                        id: id as u32,
                        area,
                        min_x: 0,
                        min_y: 0,
                        max_x: width,
                        max_y: height,
                        mean_rgb: [id as f32; 3],
                        mean_lab: rgb_to_lab([id as f32; 3]),
                    })
                    .collect(),
                summary: SegmentationSummary::default(),
            };
            let (geometry, _) = build(&segmentation);
            assert!(
                geometry.iter().all(|face| !face.loops.is_empty()),
                "partition {bits:09b} lost faces {:?}",
                geometry
                    .iter()
                    .filter(|face| face.loops.is_empty())
                    .map(|face| face.region)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn every_ternary_three_by_three_partition_uses_shared_loops() {
        let width = 3;
        let height = 3;
        for mut code in 0_u32..3_u32.pow(9) {
            let mut labels = Vec::with_capacity(9);
            for _ in 0..9 {
                labels.push(code % 3);
                code /= 3;
            }
            let mut remap = [u32::MAX; 3];
            let mut next = 0_u32;
            for label in &mut labels {
                if remap[*label as usize] == u32::MAX {
                    remap[*label as usize] = next;
                    next += 1;
                }
                *label = remap[*label as usize];
            }
            let all_connected = (0..next).all(|id| {
                let Some(start) = labels.iter().position(|&label| label == id) else {
                    return false;
                };
                let mut seen = HashSet::from([start]);
                let mut queue = vec![start];
                while let Some(index) = queue.pop() {
                    let x = index % width;
                    let y = index / width;
                    for neighbour in [
                        (x > 0).then(|| index - 1),
                        (x + 1 < width).then(|| index + 1),
                        (y > 0).then(|| index - width),
                        (y + 1 < height).then(|| index + width),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if labels[neighbour] == id && seen.insert(neighbour) {
                            queue.push(neighbour);
                        }
                    }
                }
                seen.len() == labels.iter().filter(|&&label| label == id).count()
            });
            if !all_connected {
                continue;
            }
            let regions = (0..next)
                .map(|id| {
                    let area = labels.iter().filter(|&&label| label == id).count();
                    RegionStats {
                        id,
                        area,
                        min_x: 0,
                        min_y: 0,
                        max_x: width,
                        max_y: height,
                        mean_rgb: [id as f32 * 0.3; 3],
                        mean_lab: rgb_to_lab([id as f32 * 0.3; 3]),
                    }
                })
                .collect();
            let segmentation = Segmentation {
                width,
                height,
                labels: labels.clone(),
                paint_keys: (0..next).collect(),
                paint_samples: vec![true; width * height],
                canonical: Raster::blank(width, height, [0.0; 3]),
                regions,
                summary: SegmentationSummary::default(),
            };
            let (_, summary) = build(&segmentation);
            assert_eq!(
                summary.shared_loop_fallbacks, 0,
                "labels={labels:?}, summary={summary:?}"
            );
        }
    }
}
