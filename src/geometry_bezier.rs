//! Fixed-complexity fitting of smooth shared boundaries.
//!
//! Fit one cubic, then two with a shared tangent and a searched interior knot.
//! Orthogonal reparameterization avoids treating raster arclength as Bezier t.

use super::{
    cubic_point, normalized, resample_open_polyline, sample_curve_sequence, CurveSegment, Point,
};

fn sub(a: Point, b: Point) -> Point {
    Point {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}
fn add(a: Point, b: Point) -> Point {
    Point {
        x: a.x + b.x,
        y: a.y + b.y,
    }
}
fn scale(a: Point, s: f32) -> Point {
    Point {
        x: a.x * s,
        y: a.y * s,
    }
}
fn dot(a: Point, b: Point) -> f32 {
    a.x * b.x + a.y * b.y
}

#[derive(Clone, Copy)]
struct Error {
    maximum: f32,
    rms: f32,
    worst: usize,
}

#[allow(clippy::needless_range_loop)]
fn solve(mut a: [[f64; 5]; 4], n: usize) -> Option<[f32; 4]> {
    for i in 0..n {
        let pivot = (i..n).max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))?;
        if a[pivot][i].abs() < 1e-10 {
            return None;
        }
        a.swap(i, pivot);
        let d = a[i][i];
        for k in i..=n {
            a[i][k] /= d;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = a[j][i];
            for k in i..=n {
                a[j][k] -= d * a[i][k];
            }
        }
    }
    Some(std::array::from_fn(|i| {
        if i < n {
            a[i][n] as f32
        } else {
            0.0
        }
    }))
}

fn estimate(
    points: &[Point],
    ts: &[f32],
    start: Option<Point>,
    end: Option<Point>,
    normal_model: Option<CurveSegment>,
) -> Option<CurveSegment> {
    let a = points[0];
    let b = *points.last()?;
    let mut basis = Vec::new();
    for (side, tangent) in [start, end].into_iter().enumerate() {
        if let Some(tangent) = tangent.filter(|p| dot(*p, *p) > 1e-12) {
            basis.push((
                side,
                scale(normalized(tangent), if side == 0 { 1.0 } else { -1.0 }),
            ));
        } else {
            basis.push((side, Point { x: 1.0, y: 0.0 }));
            basis.push((side, Point { x: 0.0, y: 1.0 }));
        }
    }
    let n = basis.len();
    let mut matrix = [[0.0; 5]; 4];
    for (&p, &t) in points.iter().zip(ts) {
        let u = 1.0 - t;
        let weights = [3.0 * u * u * t, 3.0 * u * t * t];
        let residual = sub(
            p,
            add(
                scale(a, u * u * u + weights[0]),
                scale(b, t * t * t + weights[1]),
            ),
        );
        let normal = normal_model.map(|c| {
            let d = normalized(derivatives(c, t).0);
            Point { x: -d.y, y: d.x }
        });
        let product = |a, b| normal.map_or_else(|| dot(a, b), |n| dot(a, n) * dot(b, n));
        for (i, &(side, v)) in basis.iter().enumerate() {
            let q = scale(v, weights[side]);
            for (j, &(other, w)) in basis.iter().enumerate() {
                matrix[i][j] += product(q, scale(w, weights[other])) as f64;
            }
            matrix[i][n] += product(q, residual) as f64;
        }
    }
    let values = solve(matrix, n)?;
    let mut controls = [a, b];
    for (i, &(side, v)) in basis.iter().enumerate() {
        controls[side] = add(controls[side], scale(v, values[i]));
    }
    let chord = a.distance(b);
    if controls[0].distance(a) > 3.0 * chord
        || controls[1].distance(b) > 3.0 * chord
        || start.is_some_and(|t| dot(sub(controls[0], a), t) <= 0.0)
        || end.is_some_and(|t| dot(sub(b, controls[1]), t) <= 0.0)
    {
        return None;
    }
    Some(CurveSegment::Cubic {
        start: a,
        first: controls[0],
        second: controls[1],
        end: b,
    })
}

fn derivatives(c: CurveSegment, t: f32) -> (Point, Point) {
    let CurveSegment::Cubic {
        start: a,
        first: b,
        second: c,
        end: d,
    } = c
    else {
        unreachable!()
    };
    let u = 1.0 - t;
    let first = add(
        add(scale(sub(b, a), 3.0 * u * u), scale(sub(c, b), 6.0 * u * t)),
        scale(sub(d, c), 3.0 * t * t),
    );
    let second = add(
        scale(add(sub(c, scale(b, 2.0)), a), 6.0 * u),
        scale(add(sub(d, scale(c, 2.0)), b), 6.0 * t),
    );
    (first, second)
}

fn reparameterize(points: &[Point], curve: CurveSegment, ts: &mut [f32]) {
    let previous = ts.to_vec();
    for i in 1..points.len() - 1 {
        let mut t = previous[i];
        for _ in 0..4 {
            let r = sub(cubic_point(curve, t), points[i]);
            let (d, dd) = derivatives(curve, t);
            let denominator = dot(d, d) + dot(r, dd);
            if denominator <= 1e-8 {
                break;
            }
            let next = (t - dot(r, d) / denominator).clamp(0.0, 1.0);
            if (next - t).abs() < 1e-6 {
                break;
            }
            t = next;
        }
        ts[i] = t.max(ts[i - 1]);
    }
}

fn one(
    points: &[Point],
    start: Option<Point>,
    end: Option<Point>,
) -> Option<(CurveSegment, Error)> {
    if points.len() < 4 {
        return None;
    }
    let mut ts = super::chord_parameters(points);
    let mut curve = estimate(points, &ts, start, end, None)?;
    for _ in 0..10 {
        reparameterize(points, curve, &mut ts);
        curve = estimate(points, &ts, start, end, None)?;
    }
    // Refine the normal-distance objective directly. Repeated full-vector
    // least squares alone converges slowly when the curve's speed differs
    // substantially from the source's arclength parameterization.
    for _ in 0..4 {
        reparameterize(points, curve, &mut ts);
        let Some(candidate) = estimate(points, &ts, start, end, Some(curve)) else {
            break;
        };
        let mut candidate_ts = ts.clone();
        reparameterize(points, candidate, &mut candidate_ts);
        let cost = |c, parameters: &[f32]| {
            points
                .iter()
                .zip(parameters)
                .map(|(&p, &t)| p.distance(cubic_point(c, t)).powi(2))
                .sum::<f32>()
        };
        if cost(candidate, &candidate_ts) >= cost(curve, &ts) {
            break;
        }
        curve = candidate;
        ts = candidate_ts;
    }
    reparameterize(points, curve, &mut ts);
    let mut maximum = 0.0;
    let mut squared = 0.0;
    let mut worst = 0;
    for (i, (&p, &t)) in points.iter().zip(&ts).enumerate() {
        let e = p.distance(cubic_point(curve, t));
        squared += e * e;
        if e > maximum {
            maximum = e;
            worst = i;
        }
    }
    Some((
        curve,
        Error {
            maximum,
            rms: (squared / points.len() as f32).sqrt(),
            worst,
        },
    ))
}

fn error(source: &[Point], curves: &[CurveSegment]) -> Error {
    let rendered = sample_curve_sequence(curves, 0.35);
    let mut maximum = 0.0;
    let mut squared = 0.0;
    let mut worst = 0;
    for (i, &p) in source.iter().enumerate() {
        let e = rendered
            .windows(2)
            .map(|pair| {
                let d = sub(pair[1], pair[0]);
                let t = (dot(sub(p, pair[0]), d) / dot(d, d).max(1e-12)).clamp(0.0, 1.0);
                p.distance(add(pair[0], scale(d, t)))
            })
            .fold(f32::INFINITY, f32::min);
        if e > maximum {
            maximum = e;
            worst = i;
        }
        squared += e * e;
    }
    Error {
        maximum,
        rms: (squared / source.len() as f32).sqrt(),
        worst,
    }
}

fn trace(source: &[Point], model: &str, metrics: Option<Error>, reason: &str) {
    #[cfg(feature = "diagnostics")]
    if let Some(path) = std::env::var_os("PICVEC_BEZIER_DIAGNOSTICS") {
        use std::io::Write;
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(
                file,
                "{}",
                serde_json::json!({"start":[source[0].x,source[0].y],"end":[source.last().unwrap().x,source.last().unwrap().y],"source_points":source.len(),"model":model,"maximum_error":metrics.map(|m|m.maximum),"rms_error":metrics.map(|m|m.rms),"error_measure":if reason=="source_maximum_screen" {"ordered_fit_residual"} else {"distance_to_curve_polyline"},"decision":reason})
            );
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (source, model, metrics, reason);
}

/// Test all validation gates before selecting a model. Diagnostic acceptance
/// here is local; the caller still validates the complete shared graph map.
pub(super) fn fit(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
    start: Option<Point>,
    end: Option<Point>,
) -> Option<Vec<CurveSegment>> {
    if baseline.len() < 2 || source.len() < 16 || source.first() == source.last() {
        return None;
    }
    let points = resample_open_polyline(source, 1.0);
    let a = points[0];
    let b = *points.last()?;
    let chord = a.distance(b);
    if chord < 16.0 || points.len() as f32 > chord * 1.8 + 2.0 {
        return None;
    }
    let start = start.filter(|t| dot(*t, *t) > 1e-12);
    let end = end.filter(|t| dot(*t, *t) > 1e-12);
    let direction = normalized(sub(b, a));
    let mut progress = 0.0_f32;
    for &point in &points {
        let along = dot(sub(point, a), direction);
        if along < progress - tolerance || along > chord + tolerance {
            trace(source, "interval", None, "backtracking");
            return None;
        }
        progress = progress.max(along);
    }
    let reference = sample_curve_sequence(baseline, 0.5);
    let corners = super::persistent_open_corners(source);
    let allowed = tolerance.min(1.25);
    let validate = |curves: &[CurveSegment], model: &str| {
        let metrics = error(&points, curves);
        let reason = if metrics.maximum > allowed {
            "source_maximum"
        } else if metrics.rms > allowed * 0.6 {
            "source_rms"
        } else if !super::geometry_primitives::supports_tangents(curves, start, end) {
            "endpoint_tangent"
        } else if !super::boundary_corridor_supported(source, curves, tolerance) {
            "source_corridor"
        } else if !super::boundary_corridor_supported(&reference, curves, tolerance) {
            "baseline_corridor"
        } else {
            let rendered = sample_curve_sequence(curves, 0.25);
            if corners.iter().any(|&(_, p)| {
                super::nearest_point(&rendered, p).1
                    > (super::nearest_point(&reference, p).1 + 0.125).max(0.25)
            }) {
                "protected_corner"
            } else {
                "supported"
            }
        };
        trace(source, model, Some(metrics), reason);
        reason == "supported"
    };
    let line = vec![CurveSegment::Line { start: a, end: b }];
    if validate(&line, "line") {
        return Some(line);
    }
    let single = one(&points, start, end);
    if let Some((c, _)) = single {
        if validate(&[c], "cubic_1") {
            return Some(vec![c]);
        }
    } else {
        trace(source, "cubic_1", None, "ill_conditioned_or_tangent");
    }
    if baseline.len() < 3 {
        return None;
    }
    let last = points.len() - 1;
    let mut knots: Vec<usize> = [0.25, 0.375, 0.5, 0.625, 0.75]
        .into_iter()
        .map(|t| (last as f32 * t).round() as usize)
        .collect();
    if let Some((_, m)) = single {
        knots.push(m.worst.clamp(last / 5, last * 4 / 5));
    }
    knots.sort_unstable();
    knots.dedup();
    let mut best: Option<(Vec<CurveSegment>, f32, usize)> = None;
    let mut best_trial = (f32::INFINITY, last / 2);
    for round in 0..3 {
        for &k in &knots {
            let support = 6.min(k).min(last - k);
            if support < 3 {
                continue;
            }
            let direction = normalized(sub(points[k + support], points[k - support]));
            let knot = scale(
                points[k - 2..=k + 2]
                    .iter()
                    .copied()
                    .fold(Point::default(), add),
                0.2,
            );
            let mut left = points[..=k].to_vec();
            let mut right = points[k..].to_vec();
            *left.last_mut().unwrap() = knot;
            right[0] = knot;
            let Some((l, le)) = one(&left, start, Some(direction)) else {
                continue;
            };
            let Some((r, re)) = one(&right, Some(direction), end) else {
                continue;
            };
            let curves = vec![l, r];
            let score =
                (le.rms * le.rms * k as f32 + re.rms * re.rms * (last - k) as f32) / last as f32;
            let trial_score = score + 0.1 * le.maximum.max(re.maximum).powi(2);
            if trial_score < best_trial.0 {
                best_trial = (trial_score, k);
            }
            if le.maximum.max(re.maximum) > allowed + 0.35 {
                trace(
                    source,
                    "cubic_2",
                    Some(Error {
                        maximum: le.maximum.max(re.maximum),
                        rms: score.sqrt(),
                        worst: k,
                    }),
                    "source_maximum_screen",
                );
                continue;
            }
            if best.as_ref().is_none_or(|b| score < b.1) && validate(&curves, "cubic_2") {
                best = Some((curves, score, k));
            }
        }
        if best_trial.0.is_finite() {
            let k = best_trial.1;
            let step = (last / (16 << round)).max(1);
            knots = vec![k.saturating_sub(step).max(3), (k + step).min(last - 3)];
        } else {
            break;
        }
    }
    if best.is_none() {
        trace(source, "cubic_2", None, "no_supported_knot");
    }
    best.map(|(curves, _, _)| curves)
}

/// Small intervals do not have enough raster observations for the broad
/// one/two-cubic search. Fit their existing curves instead, with a tighter
/// movement limit and fixed endpoint tangents. Source support remains required.
fn compact_pair(
    source: &[Point],
    pair: &[CurveSegment],
    tolerance: f32,
    start: Option<Point>,
    end: Option<Point>,
) -> Option<Vec<CurveSegment>> {
    let [left @ CurveSegment::Cubic { .. }, right @ CurveSegment::Cubic { .. }] = pair else {
        return None;
    };
    if left.end() != right.start()
        || source.len() < 2
        || !(2.0..=32.0).contains(&left.start().distance(right.end()))
    {
        return None;
    }
    let incoming = derivatives(*left, 1.0).0;
    let outgoing = derivatives(*right, 0.0).0;
    // A corner or a cusp is a meaningful node even on a tiny contour.
    if dot(normalized(incoming), normalized(outgoing)) < 0.98 {
        return None;
    }
    let reference = sample_curve_sequence(pair, 0.25);
    let length: f32 = reference.windows(2).map(|p| p[0].distance(p[1])).sum();
    if !(2.0..=32.0).contains(&length) {
        return None;
    }
    let first = derivatives(*left, 0.0).0;
    let last = derivatives(*right, 1.0).0;
    if dot(first, first) < 1e-12 || dot(last, last) < 1e-12 {
        return None;
    }
    let allowed = tolerance.min(0.20);
    let (curve, metrics) = one(&reference, Some(first), Some(last))?;
    if metrics.maximum > allowed || metrics.rms > allowed * 0.6 {
        return None;
    }
    let candidate = vec![curve];
    if !super::geometry_primitives::supports_tangents(&candidate, start, end)
        || !super::boundary_corridor_supported(&reference, &candidate, allowed)
        || !super::boundary_corridor_supported(source, &candidate, tolerance)
    {
        return None;
    }
    let rendered = sample_curve_sequence(&candidate, 0.25);
    if super::persistent_open_corners(source)
        .iter()
        .any(|&(_, p)| {
            super::nearest_point(&rendered, p).1
                > (super::nearest_point(&reference, p).1 + 0.125).max(0.25)
        })
    {
        return None;
    }
    trace(source, "short_cubic", Some(metrics), "supported");
    Some(candidate)
}

pub(super) fn compact(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
    start: Option<Point>,
    end: Option<Point>,
) -> Vec<CurveSegment> {
    if source.len() < 2 || baseline.len() < 2 {
        return baseline.to_vec();
    }
    // A short line can be a raster stair step within a longer smooth strand.
    // Allow mixed line/cubic intervals; fit() still preserves source corners,
    // shared endpoints, endpoint tangents and the bidirectional corridor.
    // Preserve exact analytic line/arc runs already handled by the primitive fitter.
    let mut protected: Vec<bool> = baseline
        .iter()
        .map(|c| matches!(c, CurveSegment::Line { .. }) && c.start().distance(c.end()) <= 2.0)
        .collect();
    for (i, pair) in baseline.windows(2).enumerate() {
        if super::geometry_primitives::fit(&sample_curve_sequence(pair, 0.75), 0.25, None, None)
            .is_some_and(|c| {
                super::boundary_corridor_supported(&sample_curve_sequence(pair, 0.25), &c, 0.025)
            })
        {
            protected[i] = true;
            protected[i + 1] = true;
        }
    }
    let mut knots = vec![0];
    for c in baseline.iter().take(baseline.len() - 1) {
        let first = *knots.last().unwrap();
        knots.push(first + super::nearest_point(&source[first..], c.end()).0);
    }
    knots.push(source.len() - 1);
    let mut result = Vec::new();
    let mut i = 0;
    while i < baseline.len() {
        let run = protected[i..]
            .iter()
            .position(|&p| p)
            .unwrap_or(baseline.len() - i);
        let mut counts = vec![run.min(12), 8, 4, 2];
        counts.retain(|&n| n >= 2 && n <= run);
        counts.dedup();
        let mut accepted = None;
        for n in counts {
            let first = knots[i];
            let last = knots[i + n];
            if last <= first {
                continue;
            }
            let mut observations = source[first..=last].to_vec();
            observations[0] = baseline[i].start();
            *observations.last_mut().unwrap() = baseline[i + n - 1].end();
            let candidate = fit(
                &observations,
                &baseline[i..i + n],
                tolerance,
                (i == 0).then_some(start).flatten(),
                (i + n == baseline.len()).then_some(end).flatten(),
            )
            .or_else(|| {
                (n == 2).then(|| {
                    compact_pair(
                        &observations,
                        &baseline[i..i + n],
                        tolerance,
                        (i == 0).then_some(start).flatten(),
                        (i + n == baseline.len()).then_some(end).flatten(),
                    )
                })?
            });
            if let Some(candidate) = candidate {
                if baseline[i..i + n]
                    .iter()
                    .any(|c| matches!(c, CurveSegment::Line { .. }))
                    && !super::boundary_corridor_supported(
                        &sample_curve_sequence(&baseline[i..i + n], 0.25),
                        &candidate,
                        0.25,
                    )
                {
                    continue;
                }
                accepted = Some((n, candidate));
                break;
            }
        }
        if let Some((n, candidate)) = accepted {
            #[cfg(feature = "diagnostics")]
            if std::env::var_os("PICVEC_BEZIER_DIAGNOSTICS").is_some() {
                trace(
                    &source[knots[i]..=knots[i + n]],
                    "interval",
                    Some(error(&source[knots[i]..=knots[i + n]], &candidate)),
                    "selected",
                );
            }
            result.extend(candidate);
            i += n;
        } else {
            result.push(baseline[i]);
            i += 1;
        }
    }
    if !super::boundary_corridor_supported(source, &result, tolerance)
        || !super::boundary_corridor_supported(
            &sample_curve_sequence(baseline, 0.5),
            &result,
            tolerance,
        )
    {
        trace(source, "assembled", None, "final_corridor");
        return baseline.to_vec();
    }
    result
}

pub(super) const CLOSED_CORRIDOR: f32 = 2.5;

fn simple_loop(points: &[Point]) -> bool {
    use std::collections::HashMap;
    let mut cells = HashMap::<(i32, i32), Vec<usize>>::new();
    let cross = |a: Point, b: Point, p: Point| {
        (b.x - a.x) as f64 * (p.y - a.y) as f64 - (b.y - a.y) as f64 * (p.x - a.x) as f64
    };
    for (i, edge) in points.windows(2).enumerate() {
        let [a, b] = [edge[0], edge[1]];
        if a.distance(b) < 1e-6 {
            return false;
        }
        for y in (a.y.min(b.y) / 4.0).floor() as i32..=(a.y.max(b.y) / 4.0).floor() as i32 {
            for x in (a.x.min(b.x) / 4.0).floor() as i32..=(a.x.max(b.x) / 4.0).floor() as i32 {
                let bucket = cells.entry((x, y)).or_default();
                for &j in bucket.iter() {
                    if j + 1 == i || (j == 0 && i + 2 == points.len()) {
                        continue;
                    }
                    let (c, d) = (points[j], points[j + 1]);
                    if a.x.min(b.x) <= c.x.max(d.x)
                        && c.x.min(d.x) <= a.x.max(b.x)
                        && a.y.min(b.y) <= c.y.max(d.y)
                        && c.y.min(d.y) <= a.y.max(b.y)
                        && cross(a, b, c) * cross(a, b, d) <= 0.0
                        && cross(c, d, a) * cross(c, d, b) <= 0.0
                    {
                        return false;
                    }
                }
                bucket.push(i);
            }
        }
    }
    true
}

/// Fit a complete smooth material loop before shading junctions split it.
/// The node budget is fixed; a complex silhouette keeps its existing model.
pub(super) fn fit_closed(source: &[Point], tolerance: f32) -> Option<Vec<CurveSegment>> {
    fit_closed_with_limit(source, tolerance, 8)
}

pub(super) fn fit_closed_with_limit(
    source: &[Point],
    tolerance: f32,
    maximum_segments: usize,
) -> Option<Vec<CurveSegment>> {
    if source.len() < 80
        || source.len() > 2400
        || source.first() != source.last()
        || !tolerance.is_finite()
        || tolerance <= 0.0
    {
        return None;
    }
    // Thin material bands must not disappear within the geometric budget.
    let perimeter: f32 = source.windows(2).map(|p| p[0].distance(p[1])).sum();
    if super::signed_area(source).abs() / perimeter < 3.0 {
        return None;
    }
    let mut points = resample_open_polyline(source, 1.0);
    points.pop();
    let count = points.len();
    if count < 64 {
        return None;
    }
    let turn = |i: usize, support: usize| {
        let a = sub(points[i], points[(i + count - support) % count]);
        let b = sub(points[(i + support) % count], points[i]);
        (a.x * b.y - a.y * b.x).atan2(dot(a, b))
    };
    let corners: Vec<_> = (0..count)
        .filter(|&i| {
            let local = turn(i, 2);
            let coarse = turn(i, 9);
            local.abs() > 65.0_f32.to_radians()
                && coarse.abs() > 45.0_f32.to_radians()
                && local * coarse > 0.0
                && [-2_isize, -1, 1, 2].into_iter().all(|offset| {
                    let j = (i as isize + offset).rem_euclid(count as isize) as usize;
                    let other = turn(j, 2);
                    if other.abs() != local.abs() || other * local <= 0.0 {
                        return other.abs() <= local.abs();
                    }
                    // A raster corner can have two equal turn maxima. Keeping
                    // both creates a one-edge cubic interval, which `one`
                    // cannot fit, and rejects the entire otherwise smooth loop.
                    // Prefer the stronger coarse turn, then a spatial tie-break
                    // independent of traversal direction and starting vertex.
                    coarse
                        .abs()
                        .total_cmp(&turn(j, 9).abs())
                        .then_with(|| points[j].x.total_cmp(&points[i].x))
                        .then_with(|| points[j].y.total_cmp(&points[i].y))
                        .is_gt()
                })
        })
        .collect();
    if corners.len() > 4 {
        trace(source, "closed", None, "too_many_corners");
        return None;
    }
    let smooth: Vec<_> = (0..count)
        .map(|i| {
            if corners.contains(&i) {
                return points[i];
            }
            scale(
                (0..5)
                    .map(|j| points[(i + count + j - 2) % count])
                    .fold(Point::default(), add),
                0.2,
            )
        })
        .collect();
    let tangent = |i: usize| {
        (!corners.contains(&i)).then(|| {
            normalized(sub(
                smooth[(i + 4) % count],
                smooth[(i + count - 4) % count],
            ))
        })
    };
    let mut knots = corners.clone();
    knots.push(0);
    if knots.len() < 2 {
        knots.push((1..count).max_by(|&a, &b| {
            points[a]
                .distance(points[0])
                .total_cmp(&points[b].distance(points[0]))
        })?);
    }
    knots.push(count);
    knots.sort_unstable();
    knots.dedup();
    let allowed = tolerance.min(2.0);
    loop {
        let mut curves = Vec::new();
        let mut worst = None::<(f32, usize)>;
        for pair in knots.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut observations: Vec<_> = (a..=b).map(|i| points[i % count]).collect();
            observations[0] = smooth[a % count];
            *observations.last_mut()? = smooth[b % count];
            let result = one(&observations, tangent(a % count), tangent(b % count));
            let (score, split) = if let Some((curve, e)) = result {
                curves.push(curve);
                (
                    (e.maximum / allowed).max(e.rms / (allowed * 0.6)),
                    a + e.worst,
                )
            } else {
                (f32::INFINITY, (a + b) / 2)
            };
            if score > 1.0 && worst.is_none_or(|w| score > w.0) {
                if b - a < 8 {
                    trace(source, "closed", None, "short_knot_interval");
                    return None;
                }
                worst = Some((score, split.clamp(a + 4, b - 4)));
            }
        }
        if let Some((_, split)) = worst {
            if knots.len() > maximum_segments {
                trace(source, "closed", None, "curve_budget");
                return None;
            }
            knots.push(split);
            knots.sort_unstable();
            continue;
        }
        let rendered = sample_curve_sequence(&curves, 0.5);
        if !super::boundary_corridor_supported(source, &curves, tolerance)
            || super::signed_area(source) * super::signed_area(&rendered) <= 0.0
            || !simple_loop(&rendered)
        {
            trace(source, "closed", None, "loop_validation");
            return None;
        }
        return Some(curves);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_adjacent_corner_turns_do_not_reject_the_window_loop() {
        // The same lower window corner occupies (683,513) and (683,514).
        // Treating its two 90-degree maxima as separate knots previously
        // rejected the whole contour, irrespective of the curve budget.
        let points: Vec<_> = include_str!("test-data/car-window-contour.txt")
            .lines()
            .map(|line| {
                let mut values = line.split_whitespace().map(|v| v.parse::<f32>().unwrap());
                Point {
                    x: values.next().unwrap(),
                    y: values.next().unwrap(),
                }
            })
            .collect();
        for reversed in [false, true] {
            let mut source = points.clone();
            if reversed {
                source.reverse();
            }
            let curves = fit_closed_with_limit(&source, CLOSED_CORRIDOR, 16)
                .unwrap_or_else(|| panic!("one smooth window contour, reversed={reversed}"));
            let mapping =
                super::super::geometry_mapping::map(&source, &curves, CLOSED_CORRIDOR, &mut 0)
                    .expect("every raster junction must retain an ordered correspondence");
            assert_eq!(mapping.edges.len() + 1, source.len());
            assert!(source
                .iter()
                .zip(mapping.positions)
                .all(|(a, b)| a.distance(b) <= CLOSED_CORRIDOR));
            assert!(simple_loop(&sample_curve_sequence(&curves, 0.25)));
        }
    }

    #[test]
    #[cfg(feature = "diagnostics")]
    fn complete_bands_use_few_curves_and_map_every_shading_junction() {
        let records: serde_json::Value =
            serde_json::from_str(include_str!("test-data/wifi-closed-contours.json")).unwrap();
        for (index, r) in records.as_array().unwrap().iter().enumerate() {
            if ![0, 3, 5].contains(&index) {
                continue;
            }
            let points: Vec<_> = r
                .as_array()
                .unwrap()
                .iter()
                .map(|p| Point {
                    x: p[0].as_f64().unwrap() as f32,
                    y: p[1].as_f64().unwrap() as f32,
                })
                .collect();
            let candidate = fit_closed(&points, CLOSED_CORRIDOR).expect("complete curved band");
            assert!(candidate.len() <= 8);
            assert_eq!(candidate[0].start(), candidate.last().unwrap().end());
            assert!(candidate.windows(2).all(|p| p[0].end() == p[1].start()));
            let mut master = 0;
            let mapping = super::super::geometry_mapping::map(
                &points,
                &candidate,
                CLOSED_CORRIDOR,
                &mut master,
            )
            .expect("ordered graph mapping");
            assert_eq!(mapping.positions.len(), points.len());
            assert!(mapping
                .positions
                .iter()
                .zip(&points)
                .all(|(a, b)| a.distance(*b) <= CLOSED_CORRIDOR));
        }
    }

    #[test]
    fn closed_fit_preserves_corners_and_rejects_thin_bands_and_crossings() {
        let rectangle = |height: f32| {
            let p = [
                Point { x: 0.0, y: 0.0 },
                Point { x: 80.0, y: 0.0 },
                Point { x: 80.0, y: height },
                Point { x: 0.0, y: height },
                Point { x: 0.0, y: 0.0 },
            ];
            resample_open_polyline(&p, 1.0)
        };
        assert!(fit_closed(&rectangle(2.0), CLOSED_CORRIDOR).is_none());
        let square = rectangle(80.0);
        if let Some(curves) = fit_closed(&square, CLOSED_CORRIDOR) {
            for corner in square.iter().step_by(80) {
                assert!(curves.iter().any(|c| c.start().distance(*corner) < 0.01));
            }
        }
        let crossing = [
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 0.0, y: 0.0 },
        ];
        assert!(!simple_loop(&crossing));
        assert!(simple_loop(&rectangle(10.0)));
    }
    fn baseline(points: &[Point]) -> Vec<CurveSegment> {
        let mut anchors: Vec<_> = points.iter().step_by(8).copied().collect();
        if anchors.last() != points.last() {
            anchors.push(*points.last().unwrap());
        }
        anchors
            .windows(2)
            .map(|p| super::super::straight_cubic(p[0], p[1]))
            .collect()
    }
    fn cubic(a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) -> CurveSegment {
        let p = |(x, y)| Point { x, y };
        CurveSegment::Cubic {
            start: p(a),
            first: p(b),
            second: p(c),
            end: p(d),
        }
    }
    #[test]
    fn mixed_straight_pieces_compact_without_rounding_a_corner() {
        let points = [
            Point { x: 0.0, y: 0.0 },
            Point { x: 8.0, y: 0.0 },
            Point { x: 16.0, y: 0.0 },
            Point { x: 24.0, y: 0.0 },
        ];
        let pieces = vec![
            CurveSegment::Line {
                start: points[0],
                end: points[1],
            },
            cubic((8.0, 0.0), (10.0, 0.25), (14.0, 0.25), (16.0, 0.0)),
            CurveSegment::Line {
                start: points[2],
                end: points[3],
            },
        ];
        let source = sample_curve_sequence(&pieces, 1.0);
        let result = compact(&source, &pieces, 1.0, None, None);
        assert_eq!(
            result,
            vec![CurveSegment::Line {
                start: points[0],
                end: points[3]
            }]
        );
        let corner = vec![
            CurveSegment::Line {
                start: points[0],
                end: points[3],
            },
            CurveSegment::Line {
                start: points[3],
                end: Point { x: 24.0, y: 24.0 },
            },
        ];
        let source = sample_curve_sequence(&corner, 1.0);
        assert_eq!(compact(&source, &corner, 1.0, None, None), corner);
    }

    #[test]
    fn short_s_curve_loses_a_node_without_losing_its_inflection_or_tangents() {
        // Exact halves of a cubic with a visible change of curvature. Neither
        // the old 16-pixel interval search nor analytic arc fitting covers it.
        let pair = [
            cubic((0.0, 0.0), (2.0, 0.0), (4.0, 0.75), (6.0, 1.5)),
            cubic((6.0, 1.5), (8.0, 2.25), (10.0, 3.0), (12.0, 3.0)),
        ];
        let source = sample_curve_sequence(&pair, 1.0);
        assert!(source.len() < 16);
        let result = compact(&source, &pair, 0.5, None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start(), pair[0].start());
        assert_eq!(result[0].end(), pair[1].end());
        assert!(super::super::boundary_corridor_supported(
            &sample_curve_sequence(&pair, 0.1),
            &result,
            0.20
        ));
        for (before, t) in [(pair[0], 0.0), (pair[1], 1.0)] {
            assert!(
                dot(
                    normalized(derivatives(before, t).0),
                    normalized(derivatives(result[0], t).0)
                ) > 0.999_999
            );
        }
        let curvature = |t| {
            let (d, dd) = derivatives(result[0], t);
            d.x * dd.y - d.y * dd.x
        };
        assert!(curvature(0.25) * curvature(0.75) < 0.0);
        let mut next_master = 0;
        assert!(
            super::super::geometry_mapping::map(&source, &result, 0.5, &mut next_master).is_some()
        );
    }

    #[test]
    fn short_pair_keeps_corners_and_rejects_unsupported_source() {
        let pair = [
            cubic((0.0, 0.0), (2.0, 0.0), (4.0, 0.75), (6.0, 1.5)),
            cubic((6.0, 1.5), (8.0, 2.25), (10.0, 3.0), (12.0, 3.0)),
        ];
        let mut source = sample_curve_sequence(&pair, 1.0);
        let middle = source.len() / 2;
        source[middle].y += 2.0;
        assert!(compact_pair(&source, &pair, 0.5, None, None).is_none());
        let corner = [
            pair[0],
            cubic((6.0, 1.5), (6.0, 3.5), (6.0, 5.5), (6.0, 7.5)),
        ];
        let source = sample_curve_sequence(&corner, 1.0);
        assert!(compact_pair(&source, &corner, 0.5, None, None).is_none());
        assert_eq!(compact(&source, &corner, 0.5, None, None), corner);
        let bump = [
            cubic((0.0, 0.0), (2.0, 0.0), (4.0, 1.5), (6.0, 1.5)),
            cubic((6.0, 1.5), (8.0, 1.5), (10.0, 0.0), (12.0, 0.0)),
        ];
        let source = sample_curve_sequence(&bump, 1.0);
        assert!(compact_pair(&source, &bump, 0.5, None, None).is_none());
    }
    #[test]
    fn single_cubic_uses_geometric_distance_with_nonuniform_observations() {
        let model = cubic((0.0, 0.0), (40.0, 1.0), (60.0, 21.0), (120.0, 20.0));
        let points: Vec<_> = (0..=240)
            .map(|i| cubic_point(model, (i as f32 / 240.0).powi(3)))
            .collect();
        let result = fit(&points, &baseline(&points), 1.0, None, None).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], CurveSegment::Cubic { .. }));
        assert_eq!(result[0].start(), points[0]);
        assert_eq!(result[0].end(), *points.last().unwrap());
        assert!(error(&points, &result).maximum < 0.3);
    }
    #[test]
    fn two_cubics_have_one_smooth_interior_knot() {
        let models = [
            cubic((0.0, 0.0), (40.0, 0.0), (30.0, 35.0), (60.0, 35.0)),
            cubic((60.0, 35.0), (90.0, 35.0), (90.0, -20.0), (140.0, -20.0)),
        ];
        let points = sample_curve_sequence(&models, 0.5);
        let result = fit(&points, &baseline(&points), 1.0, None, None).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].end(), result[1].start());
        let (left, _) = derivatives(result[0], 1.0);
        let (right, _) = derivatives(result[1], 0.0);
        assert!(dot(normalized(left), normalized(right)) > 0.999_999);
        assert!(error(&points, &result).maximum <= 1.0);
    }
    #[test]
    fn corners_and_retraced_intervals_are_not_smoothed_away() {
        let corner: Vec<_> = (0..=80)
            .map(|i| {
                if i <= 40 {
                    Point {
                        x: i as f32,
                        y: 0.0,
                    }
                } else {
                    Point {
                        x: 40.0,
                        y: (i - 40) as f32,
                    }
                }
            })
            .collect();
        assert!(fit(&corner, &baseline(&corner), 1.0, None, None).is_none());
        let retraced: Vec<_> = (0..=80)
            .map(|i| Point {
                x: if i <= 50 { i as f32 } else { 100.0 - i as f32 },
                y: 0.0,
            })
            .collect();
        assert!(fit(&retraced, &baseline(&retraced), 1.0, None, None).is_none());
    }
    #[test]
    #[cfg(feature = "diagnostics")]
    fn shoulder_contours_use_at_most_one_interior_knot_and_keep_graph_connections() {
        let records: serde_json::Value =
            serde_json::from_str(include_str!("test-data/man-shoulder-contours.json")).unwrap();
        let point = |p: &serde_json::Value| Point {
            x: p[0].as_f64().unwrap() as f32,
            y: p[1].as_f64().unwrap() as f32,
        };
        for (i, r) in records.as_array().unwrap().iter().enumerate() {
            let source: Vec<_> = r["source"].as_array().unwrap().iter().map(point).collect();
            let curves: Vec<_> = r["result"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| CurveSegment::Cubic {
                    start: point(&v[0]),
                    first: point(&v[1]),
                    second: point(&v[2]),
                    end: point(&v[3]),
                })
                .collect();
            let start = (!r["start_tangent"].is_null()).then(|| point(&r["start_tangent"]));
            let end = (!r["end_tangent"].is_null()).then(|| point(&r["end_tangent"]));
            let result = compact(
                &source,
                &curves,
                super::super::fairing_raster_corridor(),
                start,
                end,
            );
            assert!(result.len() <= [2, 1, 17][i]);
            assert_eq!(result[0].start(), curves[0].start());
            assert_eq!(result.last().unwrap().end(), curves.last().unwrap().end());
            assert!(result.windows(2).all(|p| p[0].end() == p[1].start()));
            assert!(super::super::geometry_primitives::supports_tangents(
                &result, start, end
            ));
            let mut next_master = 0;
            assert!(super::super::geometry_mapping::map(
                &source,
                &result,
                super::super::fairing_raster_corridor(),
                &mut next_master
            )
            .is_some());
        }
    }
}
