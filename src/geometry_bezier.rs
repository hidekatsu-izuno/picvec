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

pub(super) fn compact(
    source: &[Point],
    baseline: &[CurveSegment],
    tolerance: f32,
    start: Option<Point>,
    end: Option<Point>,
) -> Vec<CurveSegment> {
    if source.len() < 16 || baseline.len() < 2 {
        return baseline.to_vec();
    }
    // Existing lines and consecutive pieces of an analytic arc are already
    // compact models, even when the SVG encoder uses several cubic pieces.
    let mut protected: Vec<bool> = baseline
        .iter()
        .map(|c| matches!(c, CurveSegment::Line { .. }))
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
            if last < first + 15 {
                continue;
            }
            let mut observations = source[first..=last].to_vec();
            observations[0] = baseline[i].start();
            *observations.last_mut().unwrap() = baseline[i + n - 1].end();
            if let Some(candidate) = fit(
                &observations,
                &baseline[i..i + n],
                tolerance,
                (i == 0).then_some(start).flatten(),
                (i + n == baseline.len()).then_some(end).flatten(),
            ) {
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

#[cfg(test)]
mod tests {
    use super::*;
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
