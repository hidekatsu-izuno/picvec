//! Source-supported lines, circular arcs and elliptical arcs.
//!
//! Endpoints belong to the shared graph and are never projected independently.
//! An open circle therefore has only one free parameter: its centre lies on
//! the chord's perpendicular bisector. A closed circle passes through its
//! storage anchor. Fits use equally spaced observations, not raster step counts.
//! Elliptical arcs use a conic with a similarity correction fixing both ends.

use super::{
    boundary_corridor_supported, normalized, persistent_open_corners, resample_open_polyline,
    sample_curve_sequence, CurveSegment, Point,
};

fn dot(a: Point, b: Point) -> f64 {
    a.x as f64 * b.x as f64 + a.y as f64 * b.y as f64
}

fn delta(a: Point, b: Point) -> Point {
    Point {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

fn tangent(curve: CurveSegment, end: bool) -> Point {
    normalized(match curve {
        CurveSegment::Line { start, end } => delta(end, start),
        CurveSegment::Cubic {
            start,
            first,
            second,
            end: last,
        } => {
            if end {
                delta(last, second)
            } else {
                delta(first, start)
            }
        }
    })
}

pub(super) fn supports_tangents(
    curves: &[CurveSegment],
    start: Option<Point>,
    end: Option<Point>,
) -> bool {
    [start.zip(curves.first()), end.zip(curves.last())]
        .into_iter()
        .enumerate()
        .all(|(i, constraint)| {
            constraint.is_none_or(|(expected, &curve)| {
                let expected = normalized(expected);
                dot(expected, expected) < 0.5 || dot(expected, tangent(curve, i == 1)) > 0.999_999
            })
        })
}

fn line(points: &[Point], tolerance: f32) -> Option<Vec<CurveSegment>> {
    let start = points[0];
    let end = *points.last()?;
    let length = start.distance(end);
    if length < 16.0 {
        return None;
    }
    let direction = normalized(delta(end, start));
    let mut previous = 0.0_f64;
    let mut squared_error = 0.0;
    for &point in points {
        let relative = delta(point, start);
        let along = dot(relative, direction);
        let error = (relative.x * direction.y - relative.y * direction.x).abs();
        // Reject backtracking and extensions beyond either end of the segment.
        if error > tolerance || along < previous - 0.25 || along > length as f64 + 0.25 {
            return None;
        }
        previous = previous.max(along);
        squared_error += (error as f64).powi(2);
    }
    if squared_error / points.len() as f64 > (0.55 * tolerance as f64).powi(2) {
        return None;
    }
    Some(vec![CurveSegment::Line { start, end }])
}

fn circle(points: &[Point], tolerance: f32) -> Option<Vec<CurveSegment>> {
    let start = points[0];
    let end = *points.last()?;
    let closed = start == end;
    let (cx, cy) = if closed {
        // |p-c|² = |anchor-c|² gives a two-variable linear least squares
        // system in coordinates relative to the fixed anchor.
        let (mut xx, mut xy, mut yy, mut xb, mut yb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &point in points {
            let x = point.x as f64 - start.x as f64;
            let y = point.y as f64 - start.y as f64;
            let b = 0.5 * (x * x + y * y);
            xx += x * x;
            xy += x * y;
            yy += y * y;
            xb += x * b;
            yb += y * b;
        }
        let determinant = xx * yy - xy * xy;
        if determinant <= 1e-8 * (xx + yy).powi(2) {
            return None;
        }
        (
            start.x as f64 + (xb * yy - yb * xy) / determinant,
            start.y as f64 + (yb * xx - xb * xy) / determinant,
        )
    } else {
        let chord = start.distance(end) as f64;
        if chord < 4.0 {
            return None;
        }
        let mx = 0.5 * (start.x as f64 + end.x as f64);
        let my = 0.5 * (start.y as f64 + end.y as f64);
        let nx = (start.y as f64 - end.y as f64) / chord;
        let ny = (end.x as f64 - start.x as f64) / chord;
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for &point in points {
            let x = point.x as f64 - mx;
            let y = point.y as f64 - my;
            let normal = x * nx + y * ny;
            numerator += normal * (x * x + y * y - chord * chord * 0.25);
            denominator += 2.0 * normal * normal;
        }
        if denominator < 1e-6 {
            return None;
        }
        let offset = numerator / denominator;
        (mx + offset * nx, my + offset * ny)
    };
    let radius = (start.x as f64 - cx).hypot(start.y as f64 - cy);
    if !radius.is_finite() || !(3.0..=100_000.0).contains(&radius) {
        return None;
    }
    let first_angle = (start.y as f64 - cy).atan2(start.x as f64 - cx);
    let mut previous = first_angle;
    let mut sweep = 0.0_f64;
    let mut travel = 0.0;
    let mut squared_error = 0.0;
    for &point in points {
        let x = point.x as f64 - cx;
        let y = point.y as f64 - cy;
        let error = (x.hypot(y) - radius).abs();
        if error > tolerance as f64 {
            return None;
        }
        squared_error += error * error;
        let angle = y.atan2(x);
        let step = (angle - previous + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        sweep += step;
        travel += step.abs();
        previous = angle;
    }
    // A short, almost straight arc has an ill-conditioned radius. Prefer a
    // line or the free curve there. Do not turn a retraced path into one arc.
    if sweep.abs() < 0.2
        || sweep.abs() > std::f64::consts::TAU + 1e-4
        || (travel - sweep.abs()) * radius > 0.5
        || radius * (1.0 - (0.5 * sweep.abs().min(std::f64::consts::PI)).cos())
            < 2.0 * tolerance as f64
        || squared_error / points.len() as f64 > (0.55 * tolerance as f64).powi(2)
    {
        return None;
    }
    if closed && (sweep.abs() - std::f64::consts::TAU).abs() > 1e-4 {
        return None;
    }

    // At most 45 degrees per cubic keeps the radial approximation well below
    // the serializer's 0.01 px arc-normalization tolerance at working scales.
    let count = (sweep.abs() / std::f64::consts::FRAC_PI_4).ceil() as usize;
    let step = sweep / count as f64;
    let handle = 4.0 / 3.0 * (step / 4.0).tan() * radius;
    let position = |angle: f64| Point {
        x: (cx + radius * angle.cos()) as f32,
        y: (cy + radius * angle.sin()) as f32,
    };
    let mut curves = Vec::with_capacity(count);
    let mut a = start;
    for i in 0..count {
        let angle = first_angle + i as f64 * step;
        let next = angle + step;
        let b = if i + 1 == count { end } else { position(next) };
        curves.push(CurveSegment::Cubic {
            start: a,
            first: Point {
                x: (a.x as f64 - handle * angle.sin()) as f32,
                y: (a.y as f64 + handle * angle.cos()) as f32,
            },
            second: Point {
                x: (b.x as f64 + handle * next.sin()) as f32,
                y: (b.y as f64 - handle * next.cos()) as f32,
            },
            end: b,
        });
        a = b;
    }
    Some(curves)
}

/// A circular cap has a supported straight cut as well as a curved rim.
/// Prefer this mixed model to rounding the cut into a complete ellipse.
pub(super) fn fit_closed_cap(source: &[Point], tolerance: f32) -> Option<Vec<CurveSegment>> {
    if source.first() != source.last() || source.len() < 40 || source.len() > 512 {
        return None;
    }
    let mut points = resample_open_polyline(source, 1.0);
    points.pop();
    let n = points.len();
    let minimum = 16.max(n / 5);
    let mut candidates = Vec::new();
    for first in (0..n).step_by(2) {
        for length in (minimum..=(n * 2 / 5)).rev() {
            let run: Vec<_> = (0..=length).map(|j| points[(first + j) % n]).collect();
            if let Some(line) = line(&run, 1.0) {
                candidates.push((length, first, line[0]));
                break;
            }
        }
    }
    candidates.sort_by_key(|&(length, first, _)| (std::cmp::Reverse(length), first));
    for (length, first, cut) in candidates.into_iter().take(12) {
        let arc: Vec<_> = (length..=n).map(|j| points[(first + j) % n]).collect();
        let mut observations = arc.clone();
        for i in 1..arc.len() - 1 {
            let support = 2.min(i).min(arc.len() - 1 - i);
            observations[i] = Point {
                x: arc[i - support..=i + support]
                    .iter()
                    .map(|p| p.x)
                    .sum::<f32>()
                    / (2 * support + 1) as f32,
                y: arc[i - support..=i + support]
                    .iter()
                    .map(|p| p.y)
                    .sum::<f32>()
                    / (2 * support + 1) as f32,
            };
        }
        let Some(mut curves) = circle(&observations, tolerance.min(1.25))
            .or_else(|| super::geometry_ellipse::fit_open(&observations, tolerance.min(1.25)))
        else {
            continue;
        };
        if !super::boundary_corridor_supported(&arc, &curves, tolerance.min(1.25)) {
            continue;
        }
        curves.push(cut);
        // Put the storage seam back near the first raster vertex. It is not
        // one of the two physical corners where the rim meets the cut.
        let (index, step, _) = curves
            .iter()
            .enumerate()
            .flat_map(|(i, &curve)| {
                (0..=256).map(move |j| {
                    (
                        i,
                        j,
                        super::cubic_point(curve, j as f32 / 256.0).distance(source[0]),
                    )
                })
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))?;
        let t = step as f64 / 256.0;
        let mut rotated = vec![super::curve_interval(curves[index], t, 1.0)];
        rotated.extend(curves[index + 1..].iter().copied());
        rotated.extend(curves[..index].iter().copied());
        rotated.push(super::curve_interval(curves[index], 0.0, t));
        rotated.retain(|c| c.start().distance(c.end()) > 1e-5);
        if super::boundary_corridor_supported(source, &rotated, tolerance.min(1.25)) {
            return Some(rotated);
        }
    }
    None
}

pub(super) fn fit(
    source: &[Point],
    tolerance: f32,
    start: Option<Point>,
    end: Option<Point>,
) -> Option<Vec<CurveSegment>> {
    if source.len() < 3 || tolerance < 0.25 {
        return None;
    }
    let length: f32 = source.windows(2).map(|p| p[0].distance(p[1])).sum();
    if length < 16.0 {
        return None;
    }
    let points = resample_open_polyline(source, 1.0);
    let candidate = line(&points, tolerance)
        .or_else(|| circle(&points, tolerance))
        .or_else(|| super::geometry_ellipse::fit_open(&points, tolerance))?;
    if !supports_tangents(&candidate, start, end)
        || !boundary_corridor_supported(source, &candidate, tolerance + 0.15)
    {
        return None;
    }
    Some(candidate)
}

/// Consolidate neighbouring fitted pieces before both incident faces reuse
/// the shared chain. A bounded lookahead avoids a quadratic partition search.
/// Source and baseline corridors prevent cumulative drift from earlier fits.
pub(super) fn regularize(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
) -> Vec<CurveSegment> {
    let analytic = regularize_analytic(source, baseline, tolerance, start_tangent, end_tangent);
    super::geometry_bezier::compact(source, &analytic, tolerance, start_tangent, end_tangent)
}

fn regularize_analytic(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
    start_tangent: Option<Point>,
    end_tangent: Option<Point>,
) -> Vec<CurveSegment> {
    if source.len() < 16 || baseline.is_empty() {
        return baseline.to_vec();
    }
    let mut anchored = source.to_vec();
    anchored[0] = baseline[0].start();
    *anchored.last_mut().unwrap() = baseline.last().unwrap().end();
    let reference = sample_curve_sequence(baseline, 0.5);
    if let Some(candidate) = fit(&anchored, tolerance.min(0.85), start_tangent, end_tangent) {
        if boundary_corridor_supported(&reference, &candidate, tolerance)
            && boundary_corridor_supported(source, &candidate, tolerance)
        {
            return candidate;
        }
    }
    let mut result = Vec::new();
    let mut index = 0;
    let mut changed = false;
    // Match existing curve joins to ordered source observations. Fitting the
    // already-smoothed cubics instead would preserve their local wobble and
    // reject an arc which the raster itself supports. Joins stay fixed, so
    // untouched neighbouring pieces retain exactly the same endpoints.
    let mut source_knots = vec![0];
    for curve in baseline.iter().take(baseline.len() - 1) {
        let start = *source_knots.last().unwrap();
        let nearest = super::nearest_point(&anchored[start..], curve.end()).0 + start;
        source_knots.push(nearest);
    }
    source_knots.push(anchored.len() - 1);
    let corners = persistent_open_corners(source);
    while index < baseline.len() {
        let remaining = baseline.len() - index;
        let mut accepted = None;
        // Try every join in the bounded window: powers of two alone can skip
        // the transition from an attached straight edge to a circular rim.
        for count in (1..=remaining.min(64)).rev() {
            let first = source_knots[index];
            let last = source_knots[index + count];
            if last < first + 2 {
                continue;
            }
            let pieces = &baseline[index..index + count];
            let mut observations = anchored[first..=last].to_vec();
            observations[0] = pieces[0].start();
            *observations.last_mut().unwrap() = pieces.last().unwrap().end();
            let start_constraint = (index == 0).then_some(start_tangent).flatten();
            let end_constraint = (index + count == baseline.len())
                .then_some(end_tangent)
                .flatten();
            let mut reference_samples = None;
            let candidate = fit(
                &observations,
                0.85_f32.min(tolerance),
                start_constraint,
                end_constraint,
            )
            .or_else(|| {
                // Retain the existing path for raster staircases whose RMS
                // fails the primitive estimator, although their already-fitted
                // line is supported by the source corridor. Both alternatives
                // undergo the same local source and corner checks below.
                let samples = sample_curve_sequence(pieces, 0.5);
                let candidate = fit(
                    &samples,
                    0.5_f32.min(tolerance),
                    start_constraint,
                    end_constraint,
                );
                reference_samples = Some(samples);
                candidate
            });
            let Some(candidate) = candidate else {
                continue;
            };
            // A circular model has three scalar parameters even when encoded
            // as several cubic pieces. It may replace one free cubic too.
            if candidate.len() > count + 1 {
                continue;
            }
            let samples = reference_samples.unwrap_or_else(|| sample_curve_sequence(pieces, 0.5));
            if !boundary_corridor_supported(&samples, &candidate, tolerance)
                || !boundary_corridor_supported(&source[first..=last], &candidate, tolerance)
            {
                continue;
            }
            let rendered = sample_curve_sequence(&candidate, 0.25);
            if corners.iter().any(|&(i, corner)| {
                i >= first
                    && i <= last
                    && super::nearest_point(&rendered, corner).1
                        > (super::nearest_point(&samples, corner).1 + 0.125).max(0.25)
            }) {
                continue;
            }
            accepted = Some((count, candidate));
            break;
        }
        if let Some((count, candidate)) = accepted {
            result.extend(candidate);
            index += count;
            changed = true;
        } else {
            result.push(baseline[index]);
            index += 1;
        }
    }
    if !changed
        || !boundary_corridor_supported(source, &result, tolerance)
        || !boundary_corridor_supported(&reference, &result, tolerance)
    {
        return baseline.to_vec();
    }
    // Keep supported corners, including corners inside a closed chain. The
    // fixed first/last graph anchor is already retained exactly by every fit.
    let samples = sample_curve_sequence(&result, 0.25);
    for (_, corner) in corners {
        let before = super::nearest_point(&reference, corner).1;
        if super::nearest_point(&samples, corner).1 > (before + 0.125).max(0.25) {
            return baseline.to_vec();
        }
    }
    result
}

#[cfg(test)]
include!("../tests/unit/geometry_primitives.rs");
