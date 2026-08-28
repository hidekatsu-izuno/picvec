//! Exact geometry normalization used by the serializer.
//!
//! Primitive recognition happens while region masks are still available, so
//! it is both faster and safer than reparsing a multi-megabyte SVG afterward.

use serde::Serialize;

use crate::geometry::{GeometrySummary, RegionGeometry};
use crate::gradient::Paint;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorPoint {
    pub x: f64,
    pub y: f64,
}

impl VectorPoint {
    fn distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SegmentKind {
    Line,
    Cubic,
    Arc,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    kind: SegmentKind,
    start: VectorPoint,
    end: VectorPoint,
    first: VectorPoint,
    second: VectorPoint,
    radius: f64,
    large_arc: u8,
    sweep: u8,
}

impl Segment {
    fn line(start: VectorPoint, end: VectorPoint) -> Self {
        Self {
            kind: SegmentKind::Line,
            start,
            end,
            first: start,
            second: end,
            radius: 0.0,
            large_arc: 0,
            sweep: 0,
        }
    }

    fn cubic(
        start: VectorPoint,
        first: VectorPoint,
        second: VectorPoint,
        end: VectorPoint,
    ) -> Self {
        Self {
            kind: SegmentKind::Cubic,
            start,
            end,
            first,
            second,
            radius: 0.0,
            large_arc: 0,
            sweep: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct Subpath {
    start: VectorPoint,
    segments: Vec<Segment>,
    closed: bool,
}

#[derive(Clone, Copy, Debug)]
struct ArcCandidate {
    segment: Segment,
    centre: VectorPoint,
    radius: f64,
    sweep_angle: f64,
    maximum_residual: f64,
}

#[derive(Clone, Debug)]
pub enum OptimizedElement {
    Path {
        data: String,
        bbox: Option<(f64, f64, f64, f64)>,
    },
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
    },
    Rect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    },
    Circle {
        cx: f64,
        cy: f64,
        radius: f64,
    },
}

#[derive(Clone, Debug, Default)]
pub struct PathOptimization {
    pub linear_cubics: usize,
    pub redundant_segments: usize,
    pub arc_segments: usize,
    pub merged_arcs: usize,
    pub maximum_arc_residual: f64,
}

#[derive(Clone, Copy, Debug)]
enum Token {
    Command(char),
    Number(f64),
}

fn path_tokens(value: &str) -> Option<Vec<Token>> {
    let bytes = value.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0_usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() || byte == b',' {
            index += 1;
            continue;
        }
        if byte.is_ascii_alphabetic() {
            tokens.push(Token::Command(byte as char));
            index += 1;
            continue;
        }
        let start = index;
        if matches!(bytes[index], b'+' | b'-') {
            index += 1;
        }
        let mut digits = false;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            digits = true;
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'.' {
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                digits = true;
                index += 1;
            }
        }
        if !digits {
            return None;
        }
        if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
            index += 1;
            if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
                index += 1;
            }
            let exponent_start = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            if index == exponent_start {
                return None;
            }
        }
        let number = value[start..index].parse::<f64>().ok()?;
        tokens.push(Token::Number(number));
    }
    Some(tokens)
}

fn parse_path(value: &str) -> Option<Vec<Subpath>> {
    let tokens = path_tokens(value)?;
    let mut index = 0_usize;
    let mut command = None::<char>;
    let mut current = VectorPoint::default();
    let mut start = None::<VectorPoint>;
    let mut segments = Vec::<Segment>::new();
    let mut subpaths = Vec::<Subpath>::new();
    let mut closed = false;
    let finish = |start: &mut Option<VectorPoint>,
                  segments: &mut Vec<Segment>,
                  closed: &mut bool,
                  subpaths: &mut Vec<Subpath>| {
        if let Some(value) = start.take() {
            subpaths.push(Subpath {
                start: value,
                segments: std::mem::take(segments),
                closed: *closed,
            });
        }
        *closed = false;
    };
    let number = |tokens: &[Token], index: &mut usize| -> Option<f64> {
        let Token::Number(value) = *tokens.get(*index)? else {
            return None;
        };
        *index += 1;
        Some(value)
    };
    while index < tokens.len() {
        if let Token::Command(value) = tokens[index] {
            command = Some(value);
            index += 1;
        }
        let value = command?;
        let upper = value.to_ascii_uppercase();
        let relative = value.is_ascii_lowercase();
        if upper == 'Z' {
            let origin = start?;
            current = origin;
            closed = true;
            finish(&mut start, &mut segments, &mut closed, &mut subpaths);
            command = None;
            continue;
        }
        let mut point = |index: &mut usize| -> Option<VectorPoint> {
            let mut point = VectorPoint {
                x: number(&tokens, index)?,
                y: number(&tokens, index)?,
            };
            if relative {
                point.x += current.x;
                point.y += current.y;
            }
            Some(point)
        };
        match upper {
            'M' => {
                let value = point(&mut index)?;
                if start.is_some() {
                    finish(&mut start, &mut segments, &mut closed, &mut subpaths);
                }
                start = Some(value);
                current = value;
                command = Some(if relative { 'l' } else { 'L' });
            }
            'L' => {
                let end = point(&mut index)?;
                start?;
                segments.push(Segment::line(current, end));
                current = end;
            }
            'H' => {
                let value = number(&tokens, &mut index)?;
                let end = VectorPoint {
                    x: if relative { current.x + value } else { value },
                    y: current.y,
                };
                start?;
                segments.push(Segment::line(current, end));
                current = end;
            }
            'V' => {
                let value = number(&tokens, &mut index)?;
                let end = VectorPoint {
                    x: current.x,
                    y: if relative { current.y + value } else { value },
                };
                start?;
                segments.push(Segment::line(current, end));
                current = end;
            }
            'C' => {
                let first = point(&mut index)?;
                let second = point(&mut index)?;
                let end = point(&mut index)?;
                start?;
                segments.push(Segment::cubic(current, first, second, end));
                current = end;
            }
            _ => return None,
        }
    }
    finish(&mut start, &mut segments, &mut closed, &mut subpaths);
    (!subpaths.is_empty()).then_some(subpaths)
}

fn cross(first: VectorPoint, second: VectorPoint) -> f64 {
    first.x * second.y - first.y * second.x
}

fn subtract(first: VectorPoint, second: VectorPoint) -> VectorPoint {
    VectorPoint {
        x: first.x - second.x,
        y: first.y - second.y,
    }
}

fn linear_cubic(segment: Segment, tolerance: f64) -> bool {
    if segment.kind != SegmentKind::Cubic {
        return false;
    }
    let chord = subtract(segment.end, segment.start);
    let length = chord.x.hypot(chord.y);
    if length <= tolerance {
        return segment.first.distance(segment.start) <= tolerance
            && segment.second.distance(segment.start) <= tolerance;
    }
    let first = subtract(segment.first, segment.start);
    let second = subtract(segment.second, segment.start);
    if cross(chord, first).abs().max(cross(chord, second).abs()) > tolerance * length {
        return false;
    }
    let denominator = length * length;
    let first_position = (first.x * chord.x + first.y * chord.y) / denominator;
    let second_position = (second.x * chord.x + second.y * chord.y) / denominator;
    -tolerance <= first_position
        && first_position <= second_position + tolerance
        && second_position + tolerance <= 1.0 + 2.0 * tolerance
}

fn between(first: VectorPoint, middle: VectorPoint, last: VectorPoint) -> bool {
    let chord = subtract(last, first);
    let squared = chord.x * chord.x + chord.y * chord.y;
    if squared <= 1e-18 {
        return false;
    }
    let offset = subtract(middle, first);
    let position = (offset.x * chord.x + offset.y * chord.y) / squared;
    (-1e-9..=1.0 + 1e-9).contains(&position)
}

fn canonicalize(subpath: Subpath) -> (Subpath, usize, usize) {
    let mut converted = 0_usize;
    let mut removed = 0_usize;
    let mut normalized = Vec::<Segment>::new();
    for segment in subpath.segments {
        let current = if linear_cubic(segment, 1e-6) {
            converted += 1;
            Segment::line(segment.start, segment.end)
        } else {
            segment
        };
        if current.kind == SegmentKind::Line && current.end.distance(current.start) <= 1e-9 {
            removed += 1;
            continue;
        }
        if let Some(previous) = normalized.last_mut() {
            if previous.kind == SegmentKind::Line && current.kind == SegmentKind::Line {
                let first_vector = subtract(previous.end, previous.start);
                let second_vector = subtract(current.end, previous.end);
                let scale = first_vector
                    .x
                    .hypot(first_vector.y)
                    .max(second_vector.x.hypot(second_vector.y))
                    .max(1.0);
                if cross(first_vector, second_vector).abs() <= 1e-8 * scale
                    && first_vector.x * second_vector.x + first_vector.y * second_vector.y >= -1e-9
                    && between(previous.start, previous.end, current.end)
                {
                    previous.end = current.end;
                    removed += 1;
                    continue;
                }
            }
        }
        normalized.push(current);
    }
    if subpath.closed
        && normalized
            .last()
            .map(|segment| {
                segment.kind == SegmentKind::Line && segment.end.distance(subpath.start) <= 1e-9
            })
            .unwrap_or(false)
    {
        normalized.pop();
        removed += 1;
    }
    (
        Subpath {
            segments: normalized,
            ..subpath
        },
        converted,
        removed,
    )
}

fn cubic_point(segment: Segment, amount: f64) -> VectorPoint {
    let inverse = 1.0 - amount;
    VectorPoint {
        x: inverse.powi(3) * segment.start.x
            + 3.0 * inverse.powi(2) * amount * segment.first.x
            + 3.0 * inverse * amount.powi(2) * segment.second.x
            + amount.powi(3) * segment.end.x,
        y: inverse.powi(3) * segment.start.y
            + 3.0 * inverse.powi(2) * amount * segment.first.y
            + 3.0 * inverse * amount.powi(2) * segment.second.y
            + amount.powi(3) * segment.end.y,
    }
}

fn sample_cubic(segment: Segment, count: usize) -> Vec<VectorPoint> {
    (0..count)
        .map(|index| cubic_point(segment, index as f64 / (count - 1).max(1) as f64))
        .collect()
}

fn circumcircle(
    first: VectorPoint,
    middle: VectorPoint,
    last: VectorPoint,
) -> Option<(VectorPoint, f64)> {
    let determinant = 2.0
        * (first.x * (middle.y - last.y)
            + middle.x * (last.y - first.y)
            + last.x * (first.y - middle.y));
    if determinant.abs() < 1e-10 {
        return None;
    }
    let first_norm = first.x * first.x + first.y * first.y;
    let middle_norm = middle.x * middle.x + middle.y * middle.y;
    let last_norm = last.x * last.x + last.y * last.y;
    let centre = VectorPoint {
        x: (first_norm * (middle.y - last.y)
            + middle_norm * (last.y - first.y)
            + last_norm * (first.y - middle.y))
            / determinant,
        y: (first_norm * (last.x - middle.x)
            + middle_norm * (first.x - last.x)
            + last_norm * (middle.x - first.x))
            / determinant,
    };
    Some((centre, first.distance(centre)))
}

fn unwrapped_angles(samples: &[VectorPoint], centre: VectorPoint) -> Vec<f64> {
    let mut result = Vec::with_capacity(samples.len());
    for &sample in samples {
        let mut angle = (sample.y - centre.y).atan2(sample.x - centre.x);
        if let Some(&previous) = result.last() {
            while angle - previous > std::f64::consts::PI {
                angle -= 2.0 * std::f64::consts::PI;
            }
            while angle - previous < -std::f64::consts::PI {
                angle += 2.0 * std::f64::consts::PI;
            }
        }
        result.push(angle);
    }
    result
}

fn arc_candidate(segment: Segment, maximum_residual: f64) -> Option<ArcCandidate> {
    if segment.kind != SegmentKind::Cubic || segment.start.distance(segment.end) < 2.0 {
        return None;
    }
    let samples = sample_cubic(segment, 33);
    let (centre, radius) = circumcircle(segment.start, samples[16], segment.end)?;
    if !(1.0..=10_000.0).contains(&radius) {
        return None;
    }
    let residual = samples
        .iter()
        .map(|&point| (point.distance(centre) - radius).abs())
        .fold(0.0_f64, f64::max);
    if residual > maximum_residual {
        return None;
    }
    let angles = unwrapped_angles(&samples, centre);
    let differences = angles.windows(2).map(|pair| pair[1] - pair[0]);
    let values = differences.collect::<Vec<_>>();
    if !values.iter().all(|&value| value >= -1e-7) && !values.iter().all(|&value| value <= 1e-7) {
        return None;
    }
    let sweep_angle = angles[angles.len() - 1] - angles[0];
    if sweep_angle.abs() < 3.0_f64.to_radians() || sweep_angle.abs() > 175.0_f64.to_radians() {
        return None;
    }
    Some(ArcCandidate {
        segment: Segment {
            kind: SegmentKind::Arc,
            radius,
            large_arc: u8::from(sweep_angle.abs() > std::f64::consts::PI),
            sweep: u8::from(sweep_angle > 0.0),
            ..segment
        },
        centre,
        radius,
        sweep_angle,
        maximum_residual: residual,
    })
}

fn arcify(subpath: Subpath, maximum_residual: f64) -> (Subpath, usize, usize, f64) {
    let mut output = Vec::<Segment>::new();
    let mut converted = 0_usize;
    let mut merged = 0_usize;
    let mut worst = 0.0_f64;
    let mut pending = None::<ArcCandidate>;
    let flush = |pending: &mut Option<ArcCandidate>, output: &mut Vec<Segment>| {
        if let Some(value) = pending.take() {
            output.push(value.segment);
        }
    };
    for segment in subpath.segments {
        let Some(candidate) = arc_candidate(segment, maximum_residual) else {
            flush(&mut pending, &mut output);
            output.push(segment);
            continue;
        };
        converted += 1;
        worst = worst.max(candidate.maximum_residual);
        let Some(previous) = pending else {
            pending = Some(candidate);
            continue;
        };
        let same_circle = previous.segment.sweep == candidate.segment.sweep
            && previous.centre.distance(candidate.centre) <= 0.002
            && (previous.radius - candidate.radius).abs() <= 0.002
            && previous.segment.end.distance(candidate.segment.start) <= 1e-7
            && (previous.sweep_angle + candidate.sweep_angle).abs() <= 175.0_f64.to_radians();
        if !same_circle {
            output.push(previous.segment);
            pending = Some(candidate);
            continue;
        }
        let sweep_angle = previous.sweep_angle + candidate.sweep_angle;
        let radius = 0.5 * (previous.radius + candidate.radius);
        pending = Some(ArcCandidate {
            segment: Segment {
                kind: SegmentKind::Arc,
                start: previous.segment.start,
                end: candidate.segment.end,
                radius,
                large_arc: u8::from(sweep_angle.abs() > std::f64::consts::PI),
                sweep: u8::from(sweep_angle > 0.0),
                ..previous.segment
            },
            centre: VectorPoint {
                x: 0.5 * (previous.centre.x + candidate.centre.x),
                y: 0.5 * (previous.centre.y + candidate.centre.y),
            },
            radius,
            sweep_angle,
            maximum_residual: previous.maximum_residual.max(candidate.maximum_residual),
        });
        merged += 1;
    }
    flush(&mut pending, &mut output);
    (
        Subpath {
            segments: output,
            ..subpath
        },
        converted,
        merged,
        worst,
    )
}

pub fn format_number(value: f64) -> String {
    let mut rounded = (value * 1_000_000.0).round() / 1_000_000.0;
    if rounded.abs() < 0.000_000_5 {
        rounded = 0.0;
    }
    if (rounded - rounded.round()).abs() < 1e-12 {
        format!("{rounded:.0}")
    } else {
        format!("{rounded:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn serialize_path(subpaths: &[Subpath]) -> String {
    let mut commands = Vec::<String>::new();
    for subpath in subpaths {
        commands.push(format!(
            "M {} {}",
            format_number(subpath.start.x),
            format_number(subpath.start.y)
        ));
        for segment in &subpath.segments {
            match segment.kind {
                SegmentKind::Line => commands.push(format!(
                    "L {} {}",
                    format_number(segment.end.x),
                    format_number(segment.end.y)
                )),
                SegmentKind::Cubic => commands.push(format!(
                    "C {} {} {} {} {} {}",
                    format_number(segment.first.x),
                    format_number(segment.first.y),
                    format_number(segment.second.x),
                    format_number(segment.second.y),
                    format_number(segment.end.x),
                    format_number(segment.end.y),
                )),
                SegmentKind::Arc => commands.push(format!(
                    "A {} {} 0 {} {} {} {}",
                    format_number(segment.radius),
                    format_number(segment.radius),
                    segment.large_arc,
                    segment.sweep,
                    format_number(segment.end.x),
                    format_number(segment.end.y),
                )),
            }
        }
        if subpath.closed {
            commands.push("Z".to_string());
        }
    }
    commands.join(" ")
}

fn rect_geometry(subpath: &Subpath) -> Option<(f64, f64, f64, f64)> {
    if !subpath.closed
        || subpath.segments.is_empty()
        || subpath
            .segments
            .iter()
            .any(|segment| segment.kind != SegmentKind::Line)
    {
        return None;
    }
    let mut vertices = vec![subpath.start];
    vertices.extend(subpath.segments.iter().map(|segment| segment.end));
    if vertices.last()?.distance(vertices[0]) <= 1e-8 {
        vertices.pop();
    }
    if vertices.len() != 4 {
        return None;
    }
    let round8 = |value: f64| (value * 100_000_000.0).round() / 100_000_000.0;
    let mut xs = vertices
        .iter()
        .map(|point| round8(point.x))
        .collect::<Vec<_>>();
    let mut ys = vertices
        .iter()
        .map(|point| round8(point.y))
        .collect::<Vec<_>>();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    xs.dedup();
    ys.dedup();
    if xs.len() != 2 || ys.len() != 2 {
        return None;
    }
    let mut corners = vertices
        .iter()
        .map(|point| (round8(point.x), round8(point.y)))
        .collect::<Vec<_>>();
    corners.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then_with(|| first.1.total_cmp(&second.1))
    });
    corners.dedup();
    let mut expected = vec![
        (xs[0], ys[0]),
        (xs[0], ys[1]),
        (xs[1], ys[0]),
        (xs[1], ys[1]),
    ];
    expected.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then_with(|| first.1.total_cmp(&second.1))
    });
    if corners != expected {
        return None;
    }
    for index in 0..vertices.len() {
        let first = vertices[index];
        let second = vertices[(index + 1) % vertices.len()];
        if (first.x - second.x).abs() > 1e-8 && (first.y - second.y).abs() > 1e-8 {
            return None;
        }
    }
    let width = xs[1] - xs[0];
    let height = ys[1] - ys[0];
    (width > 0.0 && height > 0.0).then_some((xs[0], ys[0], width, height))
}

fn solve_three_by_three(mut matrix: [[f64; 3]; 3], mut values: [f64; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let selected = (pivot..3).max_by(|&first, &second| {
            matrix[first][pivot]
                .abs()
                .total_cmp(&matrix[second][pivot].abs())
        })?;
        if matrix[selected][pivot].abs() <= 1e-12 {
            return None;
        }
        matrix.swap(pivot, selected);
        values.swap(pivot, selected);
        let scale = matrix[pivot][pivot];
        for column in pivot..3 {
            matrix[pivot][column] /= scale;
        }
        values[pivot] /= scale;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let scale = matrix[row][pivot];
            for column in pivot..3 {
                matrix[row][column] -= scale * matrix[pivot][column];
            }
            values[row] -= scale * values[pivot];
        }
    }
    Some(values)
}

fn fit_circle(subpath: &Subpath) -> Option<(f64, f64, f64, f64)> {
    if !subpath.closed
        || subpath.segments.len() != 4
        || subpath
            .segments
            .iter()
            .any(|segment| segment.kind != SegmentKind::Cubic)
        || subpath.segments.last()?.end.distance(subpath.start) > 1e-6
    {
        return None;
    }
    let samples = subpath
        .segments
        .iter()
        .flat_map(|&segment| sample_cubic(segment, 33).into_iter().take(32))
        .collect::<Vec<_>>();
    let mut normal = [[0.0_f64; 3]; 3];
    let mut values = [0.0_f64; 3];
    for sample in &samples {
        let row = [2.0 * sample.x, 2.0 * sample.y, 1.0];
        let value = sample.x * sample.x + sample.y * sample.y;
        for outer in 0..3 {
            values[outer] += row[outer] * value;
            for inner in 0..3 {
                normal[outer][inner] += row[outer] * row[inner];
            }
        }
    }
    let solution = solve_three_by_three(normal, values)?;
    let centre = VectorPoint {
        x: solution[0],
        y: solution[1],
    };
    let endpoint_radii = subpath
        .segments
        .iter()
        .map(|segment| segment.start.distance(centre))
        .collect::<Vec<_>>();
    let radius = endpoint_radii.iter().sum::<f64>() / endpoint_radii.len() as f64;
    let minimum = endpoint_radii.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = endpoint_radii
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if radius < 1.0 || maximum - minimum > 0.02 {
        return None;
    }
    let residual = samples
        .iter()
        .map(|&sample| (sample.distance(centre) - radius).abs())
        .fold(0.0_f64, f64::max);
    if residual > 0.15_f64.min((radius * 0.00035).max(0.02)) {
        return None;
    }
    let angles = unwrapped_angles(&samples, centre);
    let sweep = angles.last()? - angles.first()?;
    let sample_step = 2.0 * std::f64::consts::PI / samples.len() as f64;
    if (sweep.abs() + sample_step - 2.0 * std::f64::consts::PI).abs() > 3.0_f64.to_radians() {
        return None;
    }
    let differences = angles.windows(2).map(|pair| pair[1] - pair[0]);
    let values = differences.collect::<Vec<_>>();
    if !values.iter().all(|&value| value >= -1e-6) && !values.iter().all(|&value| value <= 1e-6) {
        return None;
    }
    Some((centre.x, centre.y, radius, residual))
}

fn path_bbox(subpaths: &[Subpath]) -> Option<(f64, f64, f64, f64)> {
    if subpaths.is_empty() || subpaths.iter().any(|subpath| !subpath.closed) {
        return None;
    }
    // Python's batching pass reparses the already optimized path with the
    // original M/L/H/V/C/Z-only parser.  An analytic A command therefore
    // makes that path ineligible for compound-path batching.
    if subpaths.iter().any(|subpath| {
        subpath
            .segments
            .iter()
            .any(|segment| segment.kind == SegmentKind::Arc)
    }) {
        return None;
    }
    let boxes = subpaths
        .iter()
        .map(|subpath| {
            let mut points = vec![subpath.start];
            for segment in &subpath.segments {
                if segment.kind == SegmentKind::Cubic {
                    points.extend([segment.first, segment.second]);
                }
                points.push(segment.end);
            }
            (
                points
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::INFINITY, f64::min),
                points
                    .iter()
                    .map(|point| point.y)
                    .fold(f64::INFINITY, f64::min),
                points
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::NEG_INFINITY, f64::max),
                points
                    .iter()
                    .map(|point| point.y)
                    .fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect::<Vec<_>>();
    for first in 0..boxes.len() {
        for second in first + 1..boxes.len() {
            // Overlapping subpath boxes may encode an even-odd hole.  Such a
            // path is deliberately ineligible for paint-order batching.
            if !separated_bboxes(boxes[first], boxes[second], 1.0) {
                return None;
            }
        }
    }
    Some((
        boxes
            .iter()
            .map(|bbox| bbox.0)
            .fold(f64::INFINITY, f64::min),
        boxes
            .iter()
            .map(|bbox| bbox.1)
            .fold(f64::INFINITY, f64::min),
        boxes
            .iter()
            .map(|bbox| bbox.2)
            .fold(f64::NEG_INFINITY, f64::max),
        boxes
            .iter()
            .map(|bbox| bbox.3)
            .fold(f64::NEG_INFINITY, f64::max),
    ))
}

pub fn optimize_path(
    value: &str,
    analytic_arcs: bool,
    stroke_only: bool,
) -> Option<(OptimizedElement, PathOptimization)> {
    let parsed = parse_path(value)?;
    let mut operations = PathOptimization::default();
    let mut subpaths = Vec::with_capacity(parsed.len());
    for subpath in parsed {
        let (subpath, converted, removed) = canonicalize(subpath);
        operations.linear_cubics += converted;
        operations.redundant_segments += removed;
        subpaths.push(subpath);
    }
    if analytic_arcs && subpaths.len() == 1 {
        if let Some((cx, cy, radius, _)) = fit_circle(&subpaths[0]) {
            return Some((OptimizedElement::Circle { cx, cy, radius }, operations));
        }
    }
    if analytic_arcs {
        let mut values = Vec::with_capacity(subpaths.len());
        for subpath in subpaths {
            let (subpath, converted, merged, worst) = arcify(subpath, 0.01);
            operations.arc_segments += converted;
            operations.merged_arcs += merged;
            operations.maximum_arc_residual = operations.maximum_arc_residual.max(worst);
            values.push(subpath);
        }
        subpaths = values;
    }
    if subpaths.len() == 1 {
        if let Some((x, y, width, height)) = rect_geometry(&subpaths[0]) {
            return Some((
                OptimizedElement::Rect {
                    x,
                    y,
                    width,
                    height,
                },
                operations,
            ));
        }
        if stroke_only
            && !subpaths[0].closed
            && subpaths[0].segments.len() == 1
            && subpaths[0].segments[0].kind == SegmentKind::Line
        {
            let segment = subpaths[0].segments[0];
            return Some((
                OptimizedElement::Line {
                    x1: segment.start.x,
                    y1: segment.start.y,
                    x2: segment.end.x,
                    y2: segment.end.y,
                },
                operations,
            ));
        }
    }
    let bbox = path_bbox(&subpaths);
    Some((
        OptimizedElement::Path {
            data: serialize_path(&subpaths),
            bbox,
        },
        operations,
    ))
}

pub fn separated_bboxes(
    first: (f64, f64, f64, f64),
    second: (f64, f64, f64, f64),
    padding: f64,
) -> bool {
    first.2 + padding < second.0 - padding
        || second.2 + padding < first.0 - padding
        || first.3 + padding < second.1 - padding
        || second.3 + padding < first.1 - padding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_compound_path_remains_batchable_but_nested_path_does_not() {
        let disjoint = parse_path("M0 0L4 0L4 4L0 4Z M10 0L14 0L14 4L10 4Z").unwrap();
        assert_eq!(path_bbox(&disjoint), Some((0.0, 0.0, 14.0, 4.0)));
        let nested = parse_path("M0 0L14 0L14 14L0 14Z M4 4L10 4L10 10L4 10Z").unwrap();
        assert!(path_bbox(&nested).is_none());
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OptimizationSummary {
    pub input_regions: usize,
    pub primitive_regions: usize,
    pub path_regions: usize,
    pub shared_gradient_regions: usize,
}

pub fn summarize(
    geometry: &[RegionGeometry],
    paints: &[Paint],
    report: &GeometrySummary,
) -> OptimizationSummary {
    let primitive_regions = report.rectangles + report.circles + report.ellipses;
    let mut shared = 0;
    for index in 0..paints.len() {
        if matches!(paints[index], Paint::Solid { .. }) {
            continue;
        }
        if paints[..index].iter().any(|paint| paint == &paints[index]) {
            shared += 1;
        }
    }
    OptimizationSummary {
        input_regions: geometry.len(),
        primitive_regions,
        path_regions: geometry.len().saturating_sub(primitive_regions),
        shared_gradient_regions: shared,
    }
}
