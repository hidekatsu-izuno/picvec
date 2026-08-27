use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::segment::Segmentation;

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
    /// Faces that could not be assembled exclusively from the canonical
    /// shared curves and therefore used the conservative grid fallback.
    pub shared_loop_fallbacks: usize,
    pub shared_loop_missing_edges: usize,
    pub shared_loop_discontinuities: usize,
    pub shared_loop_invalid_areas: usize,
    pub adaptive_optimal_polygons: usize,
    pub continuity_faired_masters: usize,
    /// Canonical curves replaced by their shared positioned-grid polyline
    /// after a whole-partition topology validation.
    pub shared_curve_downgrades: usize,
}

#[derive(Clone, Copy, Debug)]
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

fn split_curve(segment: CurveSegment, amount: f32) -> (CurveSegment, CurveSegment) {
    let amount = amount.clamp(0.0, 1.0);
    match segment {
        CurveSegment::Line { start, end } => {
            let middle = interpolate_point(start, end, amount);
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
            let first_edge = interpolate_point(start, first, amount);
            let middle_edge = interpolate_point(first, second, amount);
            let last_edge = interpolate_point(second, end, amount);
            let first_face = interpolate_point(first_edge, middle_edge, amount);
            let last_face = interpolate_point(middle_edge, last_edge, amount);
            let middle = interpolate_point(first_face, last_face, amount);
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

fn curve_interval(segment: CurveSegment, start: f32, end: f32) -> CurveSegment {
    let start = start.clamp(0.0, 1.0);
    let end = end.clamp(start, 1.0);
    if start <= 1e-7 && end >= 1.0 - 1e-7 {
        return segment;
    }
    let (before_end, _) = split_curve(segment, end);
    if start <= 1e-7 {
        return before_end;
    }
    split_curve(before_end, start / end.max(1e-7)).1
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveCurveSpan {
    master_id: usize,
    curve: CurveSegment,
    start_parameter: f32,
    end_parameter: f32,
}

#[derive(Clone, Debug, Default)]
struct AdaptiveBoundaryGeometry {
    edge_spans: HashMap<EdgeKey, Vec<AdaptiveCurveSpan>>,
    vertex_positions: VertexPositions,
    optimal_polygons: usize,
    continuity_faired_masters: usize,
}

type TaggedCurve = (f32, f32, usize, CurveSegment);

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

fn is_canvas_vertex(vertex: u64, stride: usize, width: usize, height: usize) -> bool {
    let point = point_from_vertex(vertex, stride);
    point.x == 0.0 || point.y == 0.0 || point.x == width as f32 || point.y == height as f32
}

fn is_canvas_edge(first: u64, second: u64, stride: usize, width: usize, height: usize) -> bool {
    let first = point_from_vertex(first, stride);
    let second = point_from_vertex(second, stride);
    (first.x == 0.0 && second.x == 0.0)
        || (first.y == 0.0 && second.y == 0.0)
        || (first.x == width as f32 && second.x == width as f32)
        || (first.y == height as f32 && second.y == height as f32)
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

fn trace_topology_strand(
    start: u64,
    following: u64,
    continuation: &HashMap<(u64, u64), u64>,
    remaining: &mut HashSet<EdgeKey>,
) -> Vec<u64> {
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
    if strand.last() != Some(&start) {
        // The seed edge may be in the middle of an open strand.  Extend its
        // other direction too; otherwise HashSet iteration would split one
        // physical curve at an arbitrary point.
        let mut prefix = Vec::<u64>::new();
        previous = following;
        current = start;
        while let Some(&next) = continuation.get(&(current, previous)) {
            let edge = EdgeKey::new(current, next);
            if !remaining.remove(&edge) {
                break;
            }
            prefix.push(next);
            previous = current;
            current = next;
        }
        prefix.reverse();
        prefix.extend(strand);
        strand = prefix;
    }
    strand
}

fn trace_edge_chains(edges: &HashSet<EdgeKey>, forced_junctions: &HashSet<u64>) -> Vec<Vec<u64>> {
    let mut adjacency = HashMap::<u64, Vec<u64>>::new();
    for edge in edges {
        adjacency.entry(edge.0).or_default().push(edge.1);
        adjacency.entry(edge.1).or_default().push(edge.0);
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
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
    starts.sort_unstable();
    for start in starts {
        let neighbours = adjacency.get(&start).cloned().unwrap_or_default();
        for second in neighbours {
            if remaining.contains(&EdgeKey::new(start, second)) {
                trace(start, second, &mut remaining, &mut chains);
            }
        }
    }
    while let Some(&edge) = remaining.iter().min() {
        trace(edge.0, edge.1, &mut remaining, &mut chains);
    }
    chains
}

fn projected_to_segment(point: Point, start: Point, end: Point) -> Point {
    let direction = Point {
        x: end.x - start.x,
        y: end.y - start.y,
    };
    let squared = direction.x * direction.x + direction.y * direction.y;
    if squared <= 1e-8 {
        return start;
    }
    let parameter = (((point.x - start.x) * direction.x + (point.y - start.y) * direction.y)
        / squared)
        .clamp(0.0, 1.0);
    Point {
        x: start.x + parameter * direction.x,
        y: start.y + parameter * direction.y,
    }
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
    let (mut xx, mut xy, mut yy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for point in points {
        let dx = (point.x - centre.x) as f64;
        let dy = (point.y - centre.y) as f64;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut direction = Point {
        x: angle.cos() as f32,
        y: angle.sin() as f32,
    };
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
        directions.insert((
            (pair[1].x - pair[0].x).signum() as i8,
            (pair[1].y - pair[0].y).signum() as i8,
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
    parameters: [f32; 3],
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
    let mut tagged = |start: f32, end: f32, curve: CurveSegment| {
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
                (points.len() - 1) as f32,
                identifier,
                straight_cubic(adjusted[0], adjusted[1]),
            ));
            return (indices, curves);
        }
        let first_midpoint = interpolate_point(adjusted[0], adjusted[1], 0.5);
        let identifier = *next_master_id;
        *next_master_id += 1;
        curves.push((
            indices[0] as f32,
            0.5 * (indices[0] + indices[1]) as f32,
            identifier,
            straight_cubic(adjusted[0], first_midpoint),
        ));
        for position in 1..adjusted.len() - 1 {
            curves.extend(potrace_corner_curves(
                interpolate_point(adjusted[position - 1], adjusted[position], 0.5),
                adjusted[position],
                interpolate_point(adjusted[position], adjusted[position + 1], 0.5),
                [
                    0.5 * (indices[position - 1] + indices[position]) as f32,
                    indices[position] as f32,
                    0.5 * (indices[position] + indices[position + 1]) as f32,
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
            0.5 * (indices[last - 1] + indices[last]) as f32,
            indices[last] as f32,
            identifier,
            straight_cubic(last_midpoint, adjusted[last]),
        ));
        return (indices, curves);
    }

    let period = (points.len() - 1) as f32;
    for (position, &vertex) in adjusted.iter().enumerate() {
        let previous = (position + adjusted.len() - 1) % adjusted.len();
        let following = (position + 1) % adjusted.len();
        let mut previous_index = polygon_indices[previous] as f32;
        let vertex_index = polygon_indices[position] as f32;
        let mut following_index = polygon_indices[following] as f32;
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

fn regularize_strand_positions(
    strand: &[u64],
    stride: usize,
    proposals: &mut HashMap<u64, Vec<Point>>,
) {
    if strand.len() < 3 {
        return;
    }
    let closed = strand.first() == strand.last();
    let vertex_slice = if closed {
        &strand[..strand.len() - 1]
    } else {
        strand
    };
    let raw: Vec<Point> = vertex_slice
        .iter()
        .map(|&vertex| point_from_vertex(vertex, stride))
        .collect();
    let simplified = if closed {
        simplify_grid_closed(&raw, std::f32::consts::FRAC_1_SQRT_2)
    } else {
        simplify_grid_open(&raw, std::f32::consts::FRAC_1_SQRT_2)
    };
    if simplified.len() < 2 {
        return;
    }
    let segment_count = if closed {
        simplified.len()
    } else {
        simplified.len() - 1
    };
    for (offset, (&vertex, &point)) in vertex_slice.iter().zip(&raw).enumerate() {
        // Open strand endpoints are material junctions without a continuation
        // on one side. Keep them exact; another strand may attach there.
        if !closed && (offset == 0 || offset + 1 == raw.len()) {
            continue;
        }
        let target = (0..segment_count)
            .map(|index| {
                projected_to_segment(
                    point,
                    simplified[index],
                    simplified[(index + 1) % simplified.len()],
                )
            })
            .min_by(|first, second| point.distance(*first).total_cmp(&point.distance(*second)))
            .unwrap_or(point);
        let delta = Point {
            x: target.x - point.x,
            y: target.y - point.y,
        };
        let distance = (delta.x * delta.x + delta.y * delta.y).sqrt();
        if distance <= 1e-6 {
            continue;
        }
        let amount = (0.5 / distance).min(1.0);
        proposals.entry(vertex).or_default().push(Point {
            x: point.x + delta.x * amount,
            y: point.y + delta.y * amount,
        });
    }
}

fn boundary_topology(
    segmentation: &Segmentation,
    stride: usize,
) -> (VertexTangents, VertexPositions, Vec<Vec<u64>>, HashSet<u64>) {
    let mut remaining = HashSet::<EdgeKey>::new();
    for edges in pair_boundary_edges(segmentation, stride).into_values() {
        for edge in edges {
            remaining.insert(edge);
        }
    }
    let mut adjacency = HashMap::<u64, Vec<u64>>::new();
    for &edge in &remaining {
        adjacency.entry(edge.0).or_default().push(edge.1);
        adjacency.entry(edge.1).or_default().push(edge.0);
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }
    // At a three-way junction, preserve the continuation supported by a
    // closed material face before using straightness as a tie-breaker.  This
    // is the reference topology rule that prevents a background contour from
    // cutting across a roof/glass or outline/fill corner.
    let mut directed = vec![Vec::<GridEdge>::new(); segmentation.regions.len()];
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
                if neighbours[side] != Some(label as u32) {
                    directed[label].push(GridEdge {
                        start: vertices[side].0,
                        end: vertices[side].1,
                    });
                }
            }
        }
    }
    let mut votes = HashMap::<(u64, EdgeKey), f32>::new();
    for edges in &directed {
        for points in trace_region_vertex_loops(edges, stride) {
            if points.len() < 3 {
                continue;
            }
            let weight = ((points.len() as f32).sqrt() / 2.0).clamp(1.0, 12.0);
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
                - 2.0 * (a.x * b.x + a.y * b.y)
        };
        // The canvas perimeter is an immutable exterior half-edge.  If an
        // interior material interface reaches it, the two perimeter edges
        // continue along the frame and may never be paired with that interior
        // edge.  Mixing those continuations exposes a transparent triangular
        // hole even though every individual face path remains closed.
        let perimeter: Vec<u64> = neighbours
            .iter()
            .copied()
            .filter(|&neighbour| {
                is_canvas_edge(
                    vertex,
                    neighbour,
                    stride,
                    segmentation.width,
                    segmentation.height,
                )
            })
            .collect();
        let forced_perimeter = if perimeter.len() == 2 {
            Some((perimeter[0], perimeter[1]))
        } else {
            None
        };
        let remaining_neighbours: Vec<u64> = neighbours
            .iter()
            .copied()
            .filter(|neighbour| !perimeter.contains(neighbour))
            .collect();
        let pairing_neighbours = if forced_perimeter.is_some() {
            remaining_neighbours.as_slice()
        } else {
            neighbours.as_slice()
        };
        let mut pairings: Vec<(u64, u64)> = match pairing_neighbours {
            [first, second] => vec![(*first, *second)],
            [first, second, third] => [(*first, *second), (*first, *third), (*second, *third)]
                .into_iter()
                .max_by(|left, right| {
                    pairing_score(left.0, left.1).total_cmp(&pairing_score(right.0, right.1))
                })
                .into_iter()
                .collect(),
            [first, second, third, fourth] => {
                let alternatives = [
                    [(*first, *second), (*third, *fourth)],
                    [(*first, *third), (*second, *fourth)],
                    [(*first, *fourth), (*second, *third)],
                ];
                alternatives
                    .into_iter()
                    .max_by(|left, right| {
                        let score = |pairs: &[(u64, u64); 2]| {
                            pairing_score(pairs[0].0, pairs[0].1)
                                + pairing_score(pairs[1].0, pairs[1].1)
                        };
                        score(left).total_cmp(&score(right))
                    })
                    .unwrap_or_default()
                    .to_vec()
            }
            _ => {
                let mut unused: HashSet<u64> = neighbours.iter().copied().collect();
                let mut pairs = Vec::new();
                while unused.len() >= 2 {
                    let mut ordered: Vec<u64> = unused.iter().copied().collect();
                    ordered.sort_unstable();
                    let selected = ordered
                        .iter()
                        .enumerate()
                        .flat_map(|(index, &first)| {
                            ordered
                                .iter()
                                .skip(index + 1)
                                .map(move |&second| (first, second))
                        })
                        .max_by(|left, right| {
                            pairing_score(left.0, left.1)
                                .total_cmp(&pairing_score(right.0, right.1))
                        });
                    let Some((first, second)) = selected else {
                        break;
                    };
                    unused.remove(&first);
                    unused.remove(&second);
                    pairs.push((first, second));
                }
                pairs
            }
        };
        if let Some(pair) = forced_perimeter {
            pairings.push(pair);
        }
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

    // Trace the physical boundary independently of the labels on either
    // side.  A long line therefore remains one geometric observation even
    // where a third material begins and its RegionPair changes.
    let mut strands = Vec::<Vec<u64>>::new();
    while let Some(&edge) = remaining.iter().min() {
        let (start, following) = if !continuation.contains_key(&(edge.0, edge.1)) {
            (edge.0, edge.1)
        } else if !continuation.contains_key(&(edge.1, edge.0)) {
            (edge.1, edge.0)
        } else {
            (edge.0, edge.1)
        };
        strands.push(trace_topology_strand(
            start,
            following,
            &continuation,
            &mut remaining,
        ));
    }

    let mut proposals = HashMap::<u64, Vec<Point>>::new();
    for strand in &strands {
        regularize_strand_positions(strand, stride, &mut proposals);
    }
    let mut positions: VertexPositions = adjacency
        .keys()
        .map(|&vertex| (vertex, point_from_vertex(vertex, stride)))
        .collect();
    for (vertex, candidates) in proposals {
        let centre = point_from_vertex(vertex, stride);
        if is_canvas_vertex(vertex, stride, segmentation.width, segmentation.height) {
            positions.insert(vertex, centre);
            continue;
        }
        let proposed = Point {
            x: candidates.iter().map(|point| point.x).sum::<f32>() / candidates.len() as f32,
            y: candidates.iter().map(|point| point.y).sum::<f32>() / candidates.len() as f32,
        };
        let delta = Point {
            x: proposed.x - centre.x,
            y: proposed.y - centre.y,
        };
        let distance = (delta.x * delta.x + delta.y * delta.y).sqrt();
        // A full half-pixel move on both sides of a valid one-pixel face can
        // make its two interfaces coincide and erase the face.  Keep a
        // strict positive separation between neighbouring source grid lines;
        // the remaining 0.05 px is below the raster uncertainty but is a
        // topology invariant for every incident material.
        let maximum_shift = 0.45_f32;
        let amount = if distance > maximum_shift {
            maximum_shift / distance
        } else {
            1.0
        };
        positions.insert(
            vertex,
            Point {
                x: centre.x + delta.x * amount,
                y: centre.y + delta.y * amount,
            },
        );
    }
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
    let mut outgoing = HashMap::<u64, Vec<usize>>::new();
    for (index, edge) in edges.iter().enumerate() {
        outgoing.entry(edge.start).or_default().push(index);
    }
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    for initial in 0..edges.len() {
        if used[initial] {
            continue;
        }
        let start = edges[initial].start;
        let mut edge_index = initial;
        let mut vertices = vec![start];
        let mut safety = 0;
        loop {
            if used[edge_index] {
                break;
            }
            used[edge_index] = true;
            let edge = edges[edge_index];
            vertices.push(edge.end);
            if edge.end == start {
                break;
            }
            let incoming = direction(edge.start, edge.end, stride);
            let Some(candidates) = outgoing.get(&edge.end) else {
                break;
            };
            let priorities = [1_u8, 0, 3, 2];
            let mut selected = None;
            'choice: for priority in priorities {
                for &candidate in candidates {
                    if used[candidate] {
                        continue;
                    }
                    let next_direction =
                        direction(edges[candidate].start, edges[candidate].end, stride);
                    if (next_direction + 4 - incoming) % 4 == priority {
                        selected = Some(candidate);
                        break 'choice;
                    }
                }
            }
            let Some(next) = selected else {
                break;
            };
            edge_index = next;
            safety += 1;
            if safety > edges.len() {
                break;
            }
        }
        if vertices.len() >= 4 && vertices.last() == Some(&start) {
            vertices.pop();
            loops.push(vertices);
        }
    }
    loops
}

fn perpendicular_distance(point: Point, first: Point, last: Point) -> f32 {
    let dx = last.x - first.x;
    let dy = last.y - first.y;
    if dx.abs() + dy.abs() < 1e-8 {
        return point.distance(first);
    }
    ((dy * point.x - dx * point.y + last.x * first.y - last.y * first.x).abs())
        / (dx * dx + dy * dy).sqrt()
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
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0_f32);
    for pair in points.windows(2) {
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + pair[0].distance(pair[1]));
    }
    let length = *cumulative.last().unwrap_or(&0.0);
    if length <= 1e-6 {
        return vec![points[0]];
    }
    let count = ((length / spacing.max(0.1)).ceil() as usize + 1).max(2);
    let mut result = Vec::with_capacity(count);
    let mut segment = 0_usize;
    for index in 0..count {
        let position = length * index as f32 / (count - 1) as f32;
        while segment + 1 < cumulative.len() - 1 && cumulative[segment + 1] < position {
            segment += 1;
        }
        let span = (cumulative[segment + 1] - cumulative[segment]).max(1e-6);
        let amount = ((position - cumulative[segment]) / span).clamp(0.0, 1.0);
        result.push(Point {
            x: points[segment].x + amount * (points[segment + 1].x - points[segment].x),
            y: points[segment].y + amount * (points[segment + 1].y - points[segment].y),
        });
    }
    result
}

fn gaussian_fair_points(points: &[Point], sigma: f32) -> Vec<Point> {
    if points.len() < 5 || sigma <= 1e-3 {
        return points.to_vec();
    }
    let radius = (4.0 * sigma).ceil() as isize;
    let mut weights: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let value = offset as f32;
            (-0.5 * value * value / (sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total.max(1e-12);
    }
    (0..points.len())
        .map(|index| {
            let mut result = Point::default();
            for (kernel_index, &weight) in weights.iter().enumerate() {
                let source = (index as isize + kernel_index as isize - radius)
                    .clamp(0, points.len() as isize - 1) as usize;
                result.x += weight * points[source].x;
                result.y += weight * points[source].y;
            }
            result
        })
        .collect()
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
        for step in 0..=count {
            if index > 0 && step == 0 {
                continue;
            }
            result.push(cubic_point(curve, step as f32 / count as f32));
        }
    }
    result
}

fn fairing_candidate_segments(
    reference: &[Point],
    tolerance: f32,
    sigma: f32,
) -> Vec<CurveSegment> {
    let mut samples = resample_open_polyline(reference, 2.0);
    if samples.len() < 2 {
        return Vec::new();
    }
    let fairing_corners = persistent_open_corners(reference);
    let smoothed = gaussian_fair_points(&samples, sigma);
    for index in 0..samples.len() {
        let turn = signed_supported_turn(&samples, index, 2)
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
        if distance > tolerance {
            let amount = tolerance / distance.max(1e-6);
            samples[index] = Point {
                x: nearest.x + amount * (samples[index].x - nearest.x),
                y: nearest.y + amount * (samples[index].y - nearest.y),
            };
        }
    }
    samples[0] = reference[0];
    let last = samples.len() - 1;
    samples[last] = reference[reference.len() - 1];
    let mut anchors = simplify_open(&samples, tolerance.clamp(0.45, 1.0));
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
    for (_, corner) in fairing_corners {
        let index = samples
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first.distance(corner).total_cmp(&second.distance(corner))
            })
            .map(|value| value.0)
            .unwrap_or(0);
        ordered.push((index, corner));
    }
    ordered.sort_by_key(|value| value.0);
    ordered.dedup_by_key(|value| value.0);
    let anchors: Vec<Point> = ordered.into_iter().map(|value| value.1).collect();
    if anchors.len() < 2 {
        return Vec::new();
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
    curves
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
    let source_samples = resample_open_polyline(source, 0.25);
    nearest_sample_distances(&observations, rendered, maximum).is_some()
        && nearest_sample_distances(rendered, &source_samples, maximum).is_some()
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
    let travelled: f32 = source
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    let displacement = source[0].distance(source[source.len() - 1]);
    if travelled < 40.0 || (continuity_master && displacement < 32.0) {
        return baseline.to_vec();
    }
    let direct_support = travelled >= 64.0
        && displacement >= 48.0
        && displacement / travelled.max(1e-6) >= std::f32::consts::FRAC_1_SQRT_2;
    let reference = sample_curve_sequence(baseline, 0.35);
    let reference_samples = resample_open_polyline(&reference, 0.25);
    let source_corners = persistent_open_corners(source);
    let baseline_corner_error = source_corners
        .iter()
        .map(|(_, corner)| {
            reference_samples
                .iter()
                .map(|point| point.distance(*corner))
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0_f32, f32::max);
    let allowed_baseline = std::f32::consts::FRAC_1_SQRT_2 + 0.5;
    let allowed_source = std::f32::consts::SQRT_2;
    let mut best = baseline.to_vec();
    let mut best_error = f32::INFINITY;
    for sigma in [
        0.5_f32, 0.659, 0.869, 1.145, 1.510, 1.991, 2.626, 3.463, 4.568, 6.0,
    ] {
        let candidate = fairing_candidate_segments(
            &reference,
            std::f32::consts::FRAC_1_SQRT_2.max(0.35),
            sigma,
        );
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
        let candidate_samples = sample_curve_sequence(&candidate, 0.25);
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
        if !raster_boundary_supported(source, &candidate_samples, allowed_source) {
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
        if candidate.len() < best.len()
            || (candidate.len() == best.len() && mean_error < best_error)
        {
            best = candidate;
            best_error = mean_error;
        }
    }
    let sufficient_reduction = if continuity_master {
        best.len() < baseline.len()
    } else if direct_support {
        4 * best.len() <= 3 * baseline.len()
    } else {
        4 * best.len() <= baseline.len()
    };
    if sufficient_reduction {
        best
    } else {
        baseline.to_vec()
    }
}

fn parameterize_curves_by_source_arclength(
    source: &[Point],
    curves: Vec<CurveSegment>,
    next_master_id: &mut usize,
) -> Vec<(f32, f32, usize, CurveSegment)> {
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
        (index - 1) as f32 + (distance - start) / (end - start).max(1e-6)
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
            Point {
                x: inverse.powi(3) * start.x
                    + 3.0 * inverse.powi(2) * parameter * first.x
                    + 3.0 * inverse * parameter.powi(2) * second.x
                    + parameter.powi(3) * end.x,
                y: inverse.powi(3) * start.y
                    + 3.0 * inverse.powi(2) * parameter * first.y
                    + 3.0 * inverse * parameter.powi(2) * second.y
                    + parameter.powi(3) * end.y,
            }
        }
    }
}

fn pair_boundary_edges(
    segmentation: &Segmentation,
    stride: usize,
) -> HashMap<RegionPair, Vec<EdgeKey>> {
    let mut pairs = HashMap::<RegionPair, Vec<EdgeKey>>::new();
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

fn oriented_curve_interval(segment: CurveSegment, start: f32, end: f32) -> CurveSegment {
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
    match (at_end, interval) {
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
    }
}

fn fit_adaptive_boundary_geometry(
    segmentation: &Segmentation,
    stride: usize,
    strands: &[Vec<u64>],
) -> AdaptiveBoundaryGeometry {
    let mut edge_spans = HashMap::<EdgeKey, Vec<AdaptiveCurveSpan>>::new();
    let mut proposals = HashMap::<u64, Vec<(usize, Point)>>::new();
    let mut next_master_id = 0_usize;
    let mut optimal_polygons = 0_usize;
    let mut continuity_faired_masters = 0_usize;
    for strand in strands {
        if strand.len() < 2 {
            continue;
        }
        let closed = strand.len() > 2 && strand.first() == strand.last();
        let raw: Vec<Point> = strand
            .iter()
            .map(|&vertex| point_from_vertex(vertex, stride))
            .collect();
        let forced = HashSet::new();
        let (polygon, curves) =
            potrace_master_curves(&raw, closed, 0.5, 1.2, &mut next_master_id, &forced);
        if curves.is_empty() {
            continue;
        }
        optimal_polygons += usize::from(polygon.len() < raw.len());
        let period = strand.len() - 1;
        let mut expanded = Vec::<(f32, f32, usize, CurveSegment)>::new();
        if closed {
            for &(start, end, identifier, curve) in &curves {
                for shift in [-(period as f32), 0.0, period as f32] {
                    expanded.push((start + shift, end + shift, identifier, curve));
                }
            }
        } else {
            expanded.extend(curves.iter().copied());
        }
        expanded.sort_by(|first, second| first.0.total_cmp(&second.0));

        let point_at = |parameter: f32| {
            expanded
                .iter()
                .find_map(|&(start, end, _, curve)| {
                    (start - 1e-6 <= parameter && parameter <= end + 1e-6 && end > start + 1e-8)
                        .then(|| cubic_point(curve, (parameter - start) / (end - start)))
                })
                .unwrap_or_else(|| raw[(parameter.round() as usize) % period.max(1)])
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
                .push((weight, point_at(index as f32)));
        }
        if closed {
            proposals
                .entry(strand[0])
                .or_default()
                .push((weight, point_at(period as f32)));
        }

        for (index, vertices) in strand.windows(2).enumerate() {
            let mut pieces = Vec::<AdaptiveCurveSpan>::new();
            for &(start, end, identifier, curve) in &expanded {
                let overlap_start = (index as f32).max(start);
                let overlap_end = ((index + 1) as f32).min(end);
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

    // Port the Python continuity-class pass. Quantized Paint labels can
    // change many times along one visible contour; fitting each RegionPair
    // separately freezes those changes as staircase anchors. Trace the
    // boundary between coarse perceptual classes, fit it once, then slice the
    // same master back onto every original half-edge.
    let class_for_label = |label: i32| {
        if label < 0 {
            return -1;
        }
        let lab = segmentation.regions[label as usize].mean_lab;
        let chroma = lab.a.hypot(lab.b);
        if lab.l < 25.0 || chroma < 18.0 {
            0
        } else {
            let hue = lab.b.atan2(lab.a).rem_euclid(2.0 * std::f32::consts::PI);
            1 + (hue / (0.25 * std::f32::consts::PI)).floor() as i32
        }
    };
    let mut class_edges_by_pair = HashMap::<RegionPair, HashSet<EdgeKey>>::new();
    let mut class_adjacency = HashMap::<u64, HashSet<u64>>::new();
    for (pair, edges) in pair_boundary_edges(segmentation, stride) {
        let class_pair = RegionPair::new(class_for_label(pair.0), class_for_label(pair.1));
        if class_pair.0 == class_pair.1 {
            continue;
        }
        for edge in edges {
            class_edges_by_pair
                .entry(class_pair)
                .or_default()
                .insert(edge);
            class_adjacency.entry(edge.0).or_default().insert(edge.1);
            class_adjacency.entry(edge.1).or_default().insert(edge.0);
        }
    }
    let class_junctions: HashSet<u64> = class_adjacency
        .iter()
        .filter_map(|(&point, neighbours)| (neighbours.len() != 2).then_some(point))
        .collect();
    for edges in class_edges_by_pair.values() {
        for track in trace_edge_chains(edges, &class_junctions) {
            if track.len() < 40 || track.first() == track.last() {
                continue;
            }
            let corners = [0_usize, track.len() - 1];
            for corner_pair in corners.windows(2) {
                let corner_start = corner_pair[0];
                let corner_end = corner_pair[1];
                // This trace already ends at class junctions. Keeping those
                // shared endpoints in the fit is required until the complete
                // Python connected-master split (including its master-ID
                // ownership) is present; independently borrowing a neighbour
                // tangent here creates a thin cubic miter overshoot.
                let fit_start = corner_start;
                let fit_end = corner_end;
                if fit_end + 1 < fit_start + 40 {
                    continue;
                }
                let fitting_vertices = &track[fit_start..=fit_end];
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
                    ) {
                        let last = raw.len() - 1;
                        raw[last] = position;
                        end_tangent = Some(tangent);
                    }
                }
                if raw[0].distance(raw[raw.len() - 1]) < 32.0 {
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
                let mut fair = bounded_fairing_direct_shared_boundary(&raw, &baseline, false, true);
                if fair.len() < baseline.len() {
                    if let Some(tangent) = start_tangent {
                        fair[0] = align_tangent(fair[0], tangent, false);
                    }
                    if let Some(tangent) = end_tangent {
                        let last = fair.len() - 1;
                        fair[last] = align_tangent(fair[last], tangent, true);
                    }
                }
                // The bounded fairing routine already performs the same
                // bidirectional sqrt(2)-pixel raster corridor check as the
                // Python continuity pass.
                let fitted = if fair.len() < baseline.len() {
                    continuity_faired_masters += 1;
                    parameterize_curves_by_source_arclength(&raw, fair, &mut next_master_id)
                } else {
                    baseline_values
                };
                let correction_weight = 1_000_000 + raw.len();
                let point_at = |parameter: f32| {
                    fitted
                        .iter()
                        .find_map(|&(start, end, _, curve)| {
                            (start - 1e-6 <= parameter
                                && parameter <= end + 1e-6
                                && end > start + 1e-8)
                                .then(|| cubic_point(curve, (parameter - start) / (end - start)))
                        })
                        .unwrap_or_else(|| raw[parameter.round() as usize])
                };
                for (index, &vertex) in fitting_vertices.iter().enumerate() {
                    proposals
                        .entry(vertex)
                        .or_default()
                        .push((correction_weight, point_at(index as f32)));
                }
                for index in 0..fitting_vertices.len() - 1 {
                    let mut pieces = Vec::new();
                    for &(start, end, identifier, curve) in &fitted {
                        let overlap_start = (index as f32).max(start);
                        let overlap_end = ((index + 1) as f32).min(end);
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

    let all_edges = pair_boundary_edges(segmentation, stride);
    let vertices: HashSet<u64> = all_edges
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
        let mut position = Point {
            x: best.iter().map(|point| point.x).sum::<f32>() / best.len() as f32,
            y: best.iter().map(|point| point.y).sum::<f32>() / best.len() as f32,
        };
        let displacement = Point {
            x: position.x - origin.x,
            y: position.y - origin.y,
        };
        let distance = displacement.x.hypot(displacement.y);
        // Python uses min(0.5, max(0.25, corridor - 0.25)); the Paint
        // fitting corridor is 0.5, so the actual vertex adjustment is 0.25.
        let maximum_shift = 0.25_f32;
        if distance > maximum_shift {
            position = Point {
                x: origin.x + maximum_shift * displacement.x / distance,
                y: origin.y + maximum_shift * displacement.y / distance,
            };
        }
        vertex_positions.insert(vertex, position);
    }
    AdaptiveBoundaryGeometry {
        edge_spans,
        vertex_positions,
        optimal_polygons,
        continuity_faired_masters,
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
    let mut spans = Vec::<AdaptiveCurveSpan>::new();
    for &(start, end) in raw_edges {
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
        spans.extend(pieces);
    }
    let mut merged = Vec::<AdaptiveCurveSpan>::new();
    for span in spans {
        if let Some(previous) = merged.last_mut() {
            if previous.master_id == span.master_id
                && (previous.end_parameter - span.start_parameter).abs() <= 1e-5
            {
                previous.end_parameter = span.end_parameter;
                continue;
            }
        }
        merged.push(span);
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
    let mut curves: Vec<CurveSegment> = merged
        .into_iter()
        .map(|span| oriented_curve_interval(span.curve, span.start_parameter, span.end_parameter))
        .collect();
    if curves.is_empty() {
        curves = raw
            .windows(2)
            .map(|pair| CurveSegment::Line {
                start: pair[0],
                end: pair[1],
            })
            .collect();
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
    for curve in &mut curves {
        *curve = enforce_convex_cubic_controls(*curve);
    }
    let mut source: Vec<Point> = raw_edges
        .iter()
        .map(|edge| point_from_vertex(edge.0, stride))
        .collect();
    source.push(point_from_vertex(raw_edges[raw_edges.len() - 1].1, stride));
    let closed = source.len() > 2 && source.first() == source.last();
    curves = bounded_fairing_direct_shared_boundary(&source, &curves, closed, false);
    (raw, curves)
}

fn build_shared_chains(
    segmentation: &Segmentation,
    stride: usize,
) -> (
    Vec<SharedChain>,
    EdgeChainLookup,
    VertexPositions,
    usize,
    usize,
) {
    let (_, _, strands, junctions) = boundary_topology(segmentation, stride);
    let adaptive = fit_adaptive_boundary_geometry(segmentation, stride, &strands);
    let positions = adaptive.vertex_positions.clone();
    let mut edge_pairs = HashMap::<EdgeKey, RegionPair>::new();
    for (pair, edges) in pair_boundary_edges(segmentation, stride) {
        for edge in edges {
            edge_pairs.insert(edge, pair);
        }
    }
    let mut chains = Vec::<SharedChain>::new();
    let mut lookup = HashMap::<EdgeKey, (usize, u64, u64)>::new();
    for strand in strands {
        if strand.len() < 2 {
            continue;
        }
        let closed = strand.first() == strand.last();
        let mut raw_edges: Vec<(u64, u64)> = strand
            .windows(2)
            .map(|vertices| (vertices[0], vertices[1]))
            .collect();
        let mut pairs: Vec<RegionPair> = raw_edges
            .iter()
            .map(|&(first, second)| edge_pairs[&EdgeKey::new(first, second)])
            .collect();
        if raw_edges.is_empty() {
            continue;
        }

        if closed
            && pairs.iter().all(|pair| *pair == pairs[0])
            && raw_edges.iter().all(|edge| !junctions.contains(&edge.0))
        {
            let (_, segments) = adaptive_chain_curves(&raw_edges, &adaptive, stride);
            let points: Vec<Point> = segments.iter().map(|segment| segment.start()).collect();
            let chain_id = chains.len();
            for &(first, second) in &raw_edges {
                lookup.insert(EdgeKey::new(first, second), (chain_id, first, second));
            }
            chains.push(SharedChain {
                points,
                segments,
                closed: true,
            });
            continue;
        }

        // Break a multi-material closed strand at an actual pair transition.
        // It is then an open master trajectory whose endpoints are protected
        // topology junctions rather than an arbitrary raster vertex.
        if closed {
            let break_at = (0..pairs.len())
                .find(|&index| {
                    pairs[index] != pairs[(index + pairs.len() - 1) % pairs.len()]
                        || junctions.contains(&raw_edges[index].0)
                })
                .unwrap_or(0);
            raw_edges.rotate_left(break_at);
            pairs.rotate_left(break_at);
        }

        let mut run_start = 0_usize;
        while run_start < raw_edges.len() {
            let mut run_end = run_start + 1;
            while run_end < raw_edges.len()
                && pairs[run_end] == pairs[run_start]
                && !junctions.contains(&raw_edges[run_end].0)
            {
                run_end += 1;
            }
            let (_, segments) =
                adaptive_chain_curves(&raw_edges[run_start..run_end], &adaptive, stride);
            let mut points: Vec<Point> = segments.iter().map(|segment| segment.start()).collect();
            if let Some(last) = segments.last() {
                points.push(last.end());
            }
            let chain_id = chains.len();
            for &(first, second) in &raw_edges[run_start..run_end] {
                lookup.insert(EdgeKey::new(first, second), (chain_id, first, second));
            }
            chains.push(SharedChain {
                points,
                segments,
                closed: false,
            });
            run_start = run_end;
        }
    }
    (
        chains,
        lookup,
        positions,
        adaptive.optimal_polygons,
        adaptive.continuity_faired_masters,
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
    if all_segments.iter().enumerate().any(|(index, segment)| {
        segment
            .end()
            .distance(all_segments[(index + 1) % all_segments.len()].start())
            > 1e-3
    }) {
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

fn primitive_for(
    loops: &[Vec<Point>],
    region: usize,
    segmentation: &Segmentation,
) -> Option<Primitive> {
    let statistics = &segmentation.regions[region];
    let raster_width = statistics.max_x.saturating_sub(statistics.min_x);
    let raster_height = statistics.max_y.saturating_sub(statistics.min_y);
    if raster_width * raster_height == statistics.area {
        return Some(Primitive::Rect {
            x: statistics.min_x as f32,
            y: statistics.min_y as f32,
            width: raster_width as f32,
            height: raster_height as f32,
        });
    }
    if loops.len() != 1 {
        return None;
    }
    let points = &loops[0];
    if points.len() < 4 {
        return None;
    }
    let min_x = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max_x = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let min_y = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    let width = max_x - min_x;
    let height = max_y - min_y;
    let region_area = segmentation.regions[region].area;
    let corners = remove_collinear(points);
    if corners.len() == 4 && ((width * height) - region_area as f32).abs() <= 1.0 {
        return Some(Primitive::Rect {
            x: min_x,
            y: min_y,
            width,
            height,
        });
    }
    if points.len() < 8 || width < 4.0 || height < 4.0 {
        return None;
    }
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let rx = width * 0.5;
    let ry = height * 0.5;
    let residual = points
        .iter()
        .map(|p| (((p.x - cx) / rx).powi(2) + ((p.y - cy) / ry).powi(2) - 1.0).abs())
        .sum::<f32>()
        / points.len() as f32;
    let ellipse_area = std::f32::consts::PI * rx * ry;
    let area_error = (ellipse_area - region_area as f32).abs() / region_area.max(1) as f32;
    let mut mismatch = 0_usize;
    let mut union = 0_usize;
    for y in statistics.min_y.saturating_sub(1)..(statistics.max_y + 1).min(segmentation.height) {
        for x in statistics.min_x.saturating_sub(1)..(statistics.max_x + 1).min(segmentation.width)
        {
            let actual = segmentation.labels[y * segmentation.width + x] == region as u32;
            let predicted = (((x as f32 + 0.5 - cx) / rx).powi(2)
                + ((y as f32 + 0.5 - cy) / ry).powi(2))
                <= 1.0;
            mismatch += usize::from(actual != predicted);
            union += usize::from(actual || predicted);
        }
    }
    let mask_error = mismatch as f32 / union.max(1) as f32;
    if residual < 0.18 && area_error < 0.12 && mask_error <= 0.015 {
        if (rx - ry).abs() / rx.max(ry) < 0.035 {
            Some(Primitive::Circle {
                cx,
                cy,
                radius: (rx + ry) * 0.5,
            })
        } else {
            Some(Primitive::Ellipse { cx, cy, rx, ry })
        }
    } else {
        None
    }
}

pub fn build(segmentation: &Segmentation) -> (Vec<RegionGeometry>, GeometrySummary) {
    let count = segmentation.regions.len();
    let stride = segmentation.width + 1;
    let mut edges = vec![Vec::<GridEdge>::new(); count];
    let mut shared = 0_usize;
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
                if neighbours[side].is_some() {
                    shared += 1;
                }
            }
        }
    }
    let source_edges = edges.iter().map(Vec::len).sum();
    let (
        shared_chains,
        shared_lookup,
        positions,
        adaptive_optimal_polygons,
        continuity_faired_masters,
    ) = build_shared_chains(segmentation, stride);
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
    let shared_curve_downgrades = 0;
    let mut summary = GeometrySummary {
        regions: count,
        source_boundary_edges: source_edges,
        shared_boundary_edges: shared / 2,
        shared_curve_downgrades,
        adaptive_optimal_polygons,
        continuity_faired_masters,
        ..GeometrySummary::default()
    };
    let mut geometries = Vec::with_capacity(count);
    for (region, region_edges) in edges.iter().enumerate().take(count) {
        let traced = trace_region_vertex_loops(region_edges, stride);
        let mut loops = Vec::<Vec<Point>>::new();
        let mut data = String::new();
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
                data.push_str(&closed_path_data(
                    &points,
                    &mut summary.cubic_segments,
                    &mut summary.line_segments,
                ));
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
        let primitive = primitive_for(&loops, region, segmentation);
        match primitive {
            Some(Primitive::Rect { .. }) => summary.rectangles += 1,
            Some(Primitive::Circle { .. }) => summary.circles += 1,
            Some(Primitive::Ellipse { .. }) => summary.ellipses += 1,
            None => {}
        }
        geometries.push(RegionGeometry {
            region: region as u32,
            loops,
            path_data: data,
            primitive,
        });
    }
    (geometries, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb_to_lab;
    use crate::raster::Raster;
    use crate::segment::{RegionStats, Segmentation, SegmentationSummary};

    #[test]
    fn rectangular_region_becomes_rect() {
        let segmentation = Segmentation {
            width: 8,
            height: 6,
            labels: vec![0; 48],
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
        assert!(matches!(
            geometry[0].primitive,
            Some(Primitive::Rect { .. })
        ));
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
        let (chains, lookup, _, _, _) = build_shared_chains(&segmentation, stride);
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
            labels,
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
        let (chains, lookup, _, _, _) = build_shared_chains(&segmentation, stride);
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
    fn raster_supported_disc_becomes_circle() {
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
            labels,
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
        assert!(matches!(
            geometry[1].primitive,
            Some(Primitive::Circle { .. })
        ));
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
                labels,
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
