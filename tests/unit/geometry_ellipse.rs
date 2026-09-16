mod tests {
    use super::*;

    #[test]
    fn closed_models_share_a_budget_independent_of_location_and_winding() {
        let data: Vec<[f32; 2]> =
            serde_json::from_str(include_str!("../data/round-window-outline.json")).unwrap();
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
