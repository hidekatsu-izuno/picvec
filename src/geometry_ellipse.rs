//! Whole ellipses and fixed-endpoint elliptical arcs, before shared slicing.
//! A closed loop's storage anchor is not a geometric constraint: incident
//! faces share its projected graph nodes. Open arcs retain both graph ends.

use super::{resample_open_polyline, CurveSegment, Point};

#[derive(Clone, Copy, Debug)]
struct Ellipse {
    centre: Point,
    radii: [f64; 2],
    rotation: f64,
}

#[allow(clippy::needless_range_loop)]
fn solve(mut matrix: [[f64; 6]; 5]) -> Option<[f64; 5]> {
    for column in 0..5 {
        let pivot = (column..5)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() < 1e-9 {
            return None;
        }
        matrix.swap(column, pivot);
        let scale = matrix[column][column];
        for k in column..6 {
            matrix[column][k] /= scale;
        }
        for row in 0..5 {
            if row == column {
                continue;
            }
            let scale = matrix[row][column];
            for k in column..6 {
                matrix[row][k] -= scale * matrix[column][k];
            }
        }
    }
    Some(std::array::from_fn(|i| matrix[i][5]))
}

impl Ellipse {
    fn fit(points: &[Point]) -> Option<Self> {
        if points.len() < 12 || points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return None;
        }
        let centre = Point {
            x: points.iter().map(|p| p.x as f64).sum::<f64>() as f32 / points.len() as f32,
            y: points.iter().map(|p| p.y as f64).sum::<f64>() as f32 / points.len() as f32,
        };
        let scale = points
            .iter()
            .map(|p| p.distance(centre))
            .fold(0.0_f32, f32::max) as f64;
        if scale < 3.0 {
            return None;
        }
        // Centre and scale the algebraic system so large canvas coordinates
        // do not dominate the quadratic terms. Reject non-elliptic conics.
        let mut matrix = [[0.0; 6]; 5];
        for p in points {
            let x = (p.x as f64 - centre.x as f64) / scale;
            let y = (p.y as f64 - centre.y as f64) / scale;
            let row = [x * x, 2.0 * x * y, y * y, x, y];
            for i in 0..5 {
                for j in 0..5 {
                    matrix[i][j] += row[i] * row[j];
                }
                matrix[i][5] += row[i];
            }
        }
        let [a, b, c, d, e] = solve(matrix)?;
        let determinant = a * c - b * b;
        if a <= 0.0 || c <= 0.0 || determinant <= 1e-8 {
            return None;
        }
        let x = (b * e - c * d) / (2.0 * determinant);
        let y = (b * d - a * e) / (2.0 * determinant);
        let level = 1.0 + a * x * x + 2.0 * b * x * y + c * y * y;
        let spread = (a - c).hypot(2.0 * b);
        let eigenvalues = [0.5 * (a + c + spread), 0.5 * (a + c - spread)];
        let radii = eigenvalues.map(|v| scale * (level / v).sqrt());
        if radii
            .iter()
            .any(|r| !r.is_finite() || *r < 3.0 || *r > 100_000.0)
            || radii[1] / radii[0] > 8.0
        {
            return None;
        }
        Some(Self {
            centre: Point {
                x: (centre.x as f64 + scale * x) as f32,
                y: (centre.y as f64 + scale * y) as f32,
            },
            radii,
            rotation: 0.5 * (2.0 * b).atan2(a - c),
        })
    }

    fn local(self, p: Point) -> [f64; 2] {
        let (sin, cos) = self.rotation.sin_cos();
        let x = p.x as f64 - self.centre.x as f64;
        let y = p.y as f64 - self.centre.y as f64;
        [cos * x + sin * y, -sin * x + cos * y]
    }

    fn angle(self, p: Point) -> f64 {
        let [x, y] = self.local(p);
        (y / self.radii[1]).atan2(x / self.radii[0])
    }

    fn point(self, angle: f64) -> Point {
        let (sin, cos) = self.rotation.sin_cos();
        let x = self.radii[0] * angle.cos();
        let y = self.radii[1] * angle.sin();
        Point {
            x: (self.centre.x as f64 + cos * x - sin * y) as f32,
            y: (self.centre.y as f64 + sin * x + cos * y) as f32,
        }
    }

    fn derivative(self, angle: f64) -> [f64; 2] {
        let (sin, cos) = self.rotation.sin_cos();
        let x = -self.radii[0] * angle.sin();
        let y = self.radii[1] * angle.cos();
        [cos * x - sin * y, sin * x + cos * y]
    }

    fn curves(self, start: f64, sweep: f64) -> Vec<CurveSegment> {
        let count = (sweep.abs() / std::f64::consts::FRAC_PI_4).ceil() as usize;
        let step = sweep / count as f64;
        let handle = 4.0 / 3.0 * (step / 4.0).tan();
        let anchor = self.point(start);
        let mut a = anchor;
        (0..count)
            .map(|i| {
                let angle = start + i as f64 * step;
                let next = angle + step;
                let b = if i + 1 == count && (sweep.abs() - std::f64::consts::TAU).abs() < 1e-6 {
                    anchor
                } else {
                    self.point(next)
                };
                let da = self.derivative(angle);
                let db = self.derivative(next);
                let curve = CurveSegment::Cubic {
                    start: a,
                    first: Point {
                        x: (a.x as f64 + handle * da[0]) as f32,
                        y: (a.y as f64 + handle * da[1]) as f32,
                    },
                    second: Point {
                        x: (b.x as f64 - handle * db[0]) as f32,
                        y: (b.y as f64 - handle * db[1]) as f32,
                    },
                    end: b,
                };
                a = b;
                curve
            })
            .collect()
    }
}

/// A partial ellipse is useful for arches and shaded rims. A similarity
/// transform of the fitted conic fixes both shared endpoints without turning
/// its ends into independently fitted cubics. Validate again after that
/// correction; short or nearly closed chords are ill-conditioned.
pub(super) fn fit_open(source: &[Point], corridor: f32) -> Option<Vec<CurveSegment>> {
    if source.len() < 24 || source.first() == source.last() {
        return None;
    }
    let length: f32 = source.windows(2).map(|p| p[0].distance(p[1])).sum();
    if length < 32.0 || source[0].distance(*source.last()?) > 0.92 * length {
        return None;
    }
    let points = resample_open_polyline(source, 1.0);
    let mut ellipse = Ellipse::fit(&points)?;
    let start = source[0];
    let end = *source.last()?;
    let a = ellipse.point(ellipse.angle(start));
    let b = ellipse.point(ellipse.angle(end));
    let chord = a.distance(b);
    if chord < 16.0_f32.max(ellipse.radii[0] as f32 * 0.5) {
        return None;
    }
    let scale = start.distance(end) as f64 / chord as f64;
    if !(0.9..=1.1).contains(&scale) {
        return None;
    }
    let rotation =
        (end.y - start.y).atan2(end.x - start.x) as f64 - (b.y - a.y).atan2(b.x - a.x) as f64;
    let (sin, cos) = rotation.sin_cos();
    let x = (ellipse.centre.x - a.x) as f64;
    let y = (ellipse.centre.y - a.y) as f64;
    ellipse.centre = Point {
        x: (start.x as f64 + scale * (cos * x - sin * y)) as f32,
        y: (start.y as f64 + scale * (sin * x + cos * y)) as f32,
    };
    ellipse.radii.iter_mut().for_each(|r| *r *= scale);
    ellipse.rotation += rotation;
    let first = ellipse.angle(start);
    let mut previous = first;
    let (mut sweep, mut travel, mut error) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &point in &points {
        let angle = ellipse.angle(point);
        let distance = point.distance(ellipse.point(angle));
        if distance > corridor {
            return None;
        }
        error += (distance as f64).powi(2);
        let step = (angle - previous + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        sweep += step;
        travel += step.abs();
        previous = angle;
    }
    if sweep.abs() < std::f64::consts::FRAC_PI_2
        || sweep.abs() > 1.5 * std::f64::consts::PI
        || (travel - sweep.abs()) * ellipse.radii[0] > 0.5
        || error / points.len() as f64 > (0.55 * corridor as f64).powi(2)
    {
        return None;
    }
    let mut curves = ellipse.curves(first, sweep);
    // Remove floating-point endpoint drift only. The graph owns these knots.
    if let CurveSegment::Cubic {
        start: p, first, ..
    } = &mut curves[0]
    {
        first.x += start.x - p.x;
        first.y += start.y - p.y;
        *p = start;
    }
    if let CurveSegment::Cubic { end: p, second, .. } = curves.last_mut()? {
        second.x += end.x - p.x;
        second.y += end.y - p.y;
        *p = end;
    }
    super::raster_boundary_supported(
        source,
        &super::sample_curve_sequence(&curves, 0.25),
        corridor,
    )
    .then_some(curves)
}

/// A large, segmented rim carries more than one pixel of localization error
/// even when its complete shape is an ellipse. Allow two percent of its span,
/// capped at 4.5 working pixels; small contours keep the original bound.
pub(super) fn closed_corridor(source: &[Point], minimum: f32) -> f32 {
    let (mut lo, mut hi) = (
        Point {
            x: f32::INFINITY,
            y: f32::INFINITY,
        },
        Point {
            x: f32::NEG_INFINITY,
            y: f32::NEG_INFINITY,
        },
    );
    for p in source {
        lo.x = lo.x.min(p.x);
        lo.y = lo.y.min(p.y);
        hi.x = hi.x.max(p.x);
        hi.y = hi.y.max(p.y);
    }
    (0.02 * (hi.x - lo.x).max(hi.y - lo.y))
        .max(minimum)
        .min(4.5_f32.max(minimum))
}

pub(super) fn fit_closed(source: &[Point], corridor: f32) -> Option<Vec<CurveSegment>> {
    if source.len() < 24 || source.first() != source.last() {
        return None;
    }
    let points = resample_open_polyline(source, 1.0);
    let ellipse = Ellipse::fit(&points[..points.len() - 1])?;
    let mut previous = ellipse.angle(points[0]);
    let start = previous;
    let mut sweep = 0.0_f64;
    let mut travel = 0.0;
    let mut error = 0.0;
    for &point in &points {
        let angle = ellipse.angle(point);
        // Radial projection is a conservative distance bound. Followed by
        // the bidirectional raster corridor and ordered graph mapping.
        let distance = point.distance(ellipse.point(angle));
        if distance > corridor {
            return None;
        }
        error += (distance as f64).powi(2);
        let step = (angle - previous + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        sweep += step;
        travel += step.abs();
        previous = angle;
    }
    if (sweep.abs() - std::f64::consts::TAU).abs() > 1e-4
        || (travel - sweep.abs()) * ellipse.radii[0] > 2.0 * corridor as f64
        || error / points.len() as f64 > (0.5 * corridor as f64).powi(2)
    {
        return None;
    }
    // An absolute pixel corridor alone can round off a small authored square.
    // Check persistent corners cyclically, including across the storage seam.
    // Raster staircase turns that disappear at larger support are ignored.
    let count = points.len() - 1;
    let turn = |i: usize, support: usize| {
        let a = points[(i + count - support) % count];
        let p = points[i];
        let b = points[(i + support) % count];
        let dx = p.x - a.x;
        let dy = p.y - a.y;
        (dx * (b.y - p.y) - dy * (b.x - p.x)).atan2(dx * (b.x - p.x) + dy * (b.y - p.y))
    };
    for (i, &p) in points[..count].iter().enumerate() {
        let local = turn(i, 2);
        let coarse = turn(i, 4.min(count / 4));
        if local.abs() >= 80.0_f32.to_radians()
            && coarse.abs() >= 65.0_f32.to_radians()
            && local * coarse > 0.0
            && p.distance(ellipse.point(ellipse.angle(p))) > 0.5
        {
            return None;
        }
    }
    let curves = ellipse.curves(start, sweep.signum() * std::f64::consts::TAU);
    super::raster_boundary_supported(
        source,
        &super::sample_curve_sequence(&curves, 0.25),
        corridor,
    )
    .then_some(curves)
}

pub(super) fn align_stroke(
    source: &[Point],
    contour: &[Point],
    width: f32,
) -> Option<(Vec<Point>, String)> {
    let mut ellipse = Ellipse::fit(contour)?;
    let mut offsets: Vec<_> = source
        .iter()
        .map(|&p| {
            let local = ellipse.local(p);
            let sign =
                ((local[0] / ellipse.radii[0]).hypot(local[1] / ellipse.radii[1]) - 1.0).signum();
            sign * p.distance(ellipse.point(ellipse.angle(p))) as f64
        })
        .collect();
    offsets.sort_by(f64::total_cmp);
    let offset = *offsets.get(offsets.len() / 2)?;
    // Reuse the Paint boundary as the corresponding edge of a narrow stroke.
    // Its colour can still correct an inaccurate dark Paint gradient without
    // independently tracing the original raster bumps again.
    let inset = if offset.abs() < 0.25 {
        0.0
    } else {
        offset.signum() * width as f64 * 0.5
    };
    for radius in &mut ellipse.radii {
        *radius += inset;
        if *radius < 3.0 {
            return None;
        }
    }
    let start = ellipse.angle(*source.first()?);
    let mut previous = start;
    let mut sweep = 0.0_f64;
    let mut travel = 0.0;
    for &p in source {
        let angle = ellipse.angle(p);
        let step = (angle - previous + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        sweep += step;
        travel += step.abs();
        previous = angle;
    }
    if sweep.abs() < 0.05
        || sweep.abs() > std::f64::consts::TAU + 1e-4
        || (travel - sweep.abs()) * ellipse.radii[0] > 0.5
    {
        return None;
    }
    let curves = ellipse.curves(start, sweep);
    if !super::boundary_corridor_supported(source, &curves, super::closed_contour_corridor(contour))
    {
        return None;
    }
    Some((
        super::sample_curve_sequence(&curves, 0.5),
        super::structural_curve_path_data(&curves, source.first() == source.last()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_models_share_a_budget_independent_of_location_and_winding() {
        let data: Vec<[f32; 2]> =
            serde_json::from_str(include_str!("test-data/round-window-outline.json")).unwrap();
        for shift in [
            Point::default(),
            Point {
                x: 1200.0,
                y: -400.0,
            },
        ] {
            for reverse in [false, true] {
                let mut points: Vec<_> = data
                    .iter()
                    .map(|p| Point {
                        x: p[0] + shift.x,
                        y: p[1] + shift.y,
                    })
                    .collect();
                if reverse {
                    points.reverse();
                }
                let corridor =
                    closed_corridor(&points, super::super::geometry_bezier::CLOSED_CORRIDOR);
                assert!(fit_closed(&points, corridor).is_some());
            }
        }
    }

    #[test]
    fn open_rotated_ellipse_keeps_shared_endpoints_and_one_model() {
        let ellipse = Ellipse {
            centre: Point { x: 100.0, y: 90.0 },
            radii: [32.0, 65.0],
            rotation: 0.45,
        };
        for direction in [-1.0, 1.0] {
            let source: Vec<_> = (0..=240)
                .map(|i| {
                    let t = i as f64 / 240.0;
                    let mut p = ellipse.point(0.3 + direction * std::f64::consts::PI * t);
                    p.x += (0.1 * (t * 24.0 * std::f64::consts::PI).sin()) as f32;
                    p
                })
                .collect();
            let curves = super::super::geometry_primitives::fit(&source, 0.85, None, None)
                .expect("a noncircular arch must have an analytic model");
            assert_eq!(curves[0].start(), source[0]);
            assert_eq!(curves.last().unwrap().end(), *source.last().unwrap());
            for p in super::super::sample_curve_sequence(&curves, 0.25) {
                assert!(p.distance(ellipse.point(ellipse.angle(p))) < 0.3);
            }
            assert!(fit_open(&source[..20], 0.85).is_none());
            let mut retraced = source.clone();
            retraced.extend(source.iter().rev().skip(1));
            assert!(fit_open(&retraced, 0.85).is_none());
        }
    }

    #[test]
    fn large_ellipse_localization_budget_is_bounded_and_retains_dents() {
        let ellipse = Ellipse {
            centre: Point { x: 300.0, y: 300.0 },
            radii: [180.0, 210.0],
            rotation: 0.2,
        };
        let mut source: Vec<_> = (0..=720)
            .map(|i| {
                let a = i as f64 / 720.0 * std::f64::consts::TAU;
                let mut p = ellipse.point(a);
                p.x += (1.8 * (a * 20.0).sin()) as f32;
                p
            })
            .collect();
        source[720] = source[0];
        let corridor = closed_corridor(&source, super::super::fairing_raster_corridor());
        assert_eq!(corridor, 4.5);
        assert!(fit_closed(&source, corridor).is_some());
        for p in &mut source[170..190] {
            p.x += 12.0;
        }
        assert!(fit_closed(&source, corridor).is_none());
    }

    #[test]
    fn noisy_rotated_ellipses_keep_one_smooth_closed_model() {
        for rotation in [0.0, 0.7, 1.8] {
            for reverse in [false, true] {
                let ellipse = Ellipse {
                    centre: Point { x: 400.0, y: 260.0 },
                    radii: [24.0, 38.0],
                    rotation,
                };
                let mut source: Vec<_> = (0..=360)
                    .map(|i| {
                        let angle = 0.32 + i as f64 / 360.0 * std::f64::consts::TAU;
                        let mut p = ellipse.point(angle);
                        p.x = (p.x + 0.2 * (13.0 * angle).cos() as f32).round();
                        p.y = (p.y + 0.2 * (17.0 * angle).sin() as f32).round();
                        p
                    })
                    .collect();
                if reverse {
                    source.reverse();
                }
                let curves = fit_closed(&source, super::super::fairing_raster_corridor()).unwrap();
                assert_eq!(curves.len(), 8);
                assert_eq!(curves[0].start(), curves[7].end());
                for p in super::super::sample_curve_sequence(&curves, 0.25) {
                    assert!(p.distance(ellipse.point(ellipse.angle(p))) < 0.2);
                }
            }
        }
    }

    #[test]
    fn corners_dents_open_and_retraced_contours_stay_freeform() {
        let ellipse = Ellipse {
            centre: Point { x: 50.0, y: 50.0 },
            radii: [24.0, 38.0],
            rotation: 0.0,
        };
        let source: Vec<_> = (0..=360)
            .map(|i| ellipse.point(i as f64 / 360.0 * std::f64::consts::TAU))
            .collect();
        let mut closed = source.clone();
        closed[360] = closed[0];
        let mut dent = closed.clone();
        for p in &mut dent[40..70] {
            p.x -= 5.0;
        }
        let mut twice = closed.clone();
        twice.extend_from_slice(&closed[1..]);
        let rectangle: Vec<_> = (0..=160)
            .map(|i| match i % 160 {
                n if n < 40 => Point {
                    x: n as f32,
                    y: 0.0,
                },
                n if n < 80 => Point {
                    x: 40.0,
                    y: (n - 40) as f32,
                },
                n if n < 120 => Point {
                    x: (120 - n) as f32,
                    y: 40.0,
                },
                n => Point {
                    x: 0.0,
                    y: (160 - n) as f32,
                },
            })
            .collect();
        for points in [
            source[..300].to_vec(),
            dent,
            twice,
            rectangle,
            vec![Point::default(); 30],
        ] {
            assert!(fit_closed(&points, super::super::geometry_bezier::CLOSED_CORRIDOR).is_none());
        }
        for size in [6, 8, 10, 16] {
            let points: Vec<_> = (0..=4 * size)
                .map(|i| match i % (4 * size) {
                    n if n < size => Point {
                        x: n as f32,
                        y: 0.0,
                    },
                    n if n < 2 * size => Point {
                        x: size as f32,
                        y: (n - size) as f32,
                    },
                    n if n < 3 * size => Point {
                        x: (3 * size - n) as f32,
                        y: size as f32,
                    },
                    n => Point {
                        x: 0.0,
                        y: (4 * size - n) as f32,
                    },
                })
                .collect();
            assert!(
                fit_closed(&points, super::super::geometry_bezier::CLOSED_CORRIDOR).is_none(),
                "small square {size}"
            );
        }
    }
}
