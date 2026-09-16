mod tests {
    use super::*;
    use crate::geometry::{fitted_structural_open_path_data, structural_curve_path_data};
    use crate::optimize::{optimize_path, OptimizedElement};

    #[test]
    #[cfg(feature = "diagnostics")]
    fn attached_disc_rim_becomes_arcs_without_rounding_the_attached_garment() {
        // Native-resolution continuity master from row 3, column 1 of
        // cliparts-6x6. The rim shares a contour with the suit, so this is
        // deliberately not a closed circle or an isolated circular arc.
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../data/man-disc-contour.json")).unwrap();
        let point = |v: &serde_json::Value| Point {
            x: v[0].as_f64().unwrap() as f32,
            y: v[1].as_f64().unwrap() as f32,
        };
        let source: Vec<_> = fixture["source"]
            .as_array()
            .unwrap()
            .iter()
            .map(point)
            .collect();
        let baseline: Vec<_> = fixture["baseline"]
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
        let result = regularize(
            &source,
            &baseline,
            super::super::fairing_raster_corridor(),
            None,
            None,
        );
        let rim_stats = |curves: &[CurveSegment]| {
            let rim: Vec<_> = curves
                .iter()
                .copied()
                .filter(|c| c.start().x > 510.0 && c.end().x > 510.0)
                .collect();
            let path = structural_curve_path_data(&rim, false);
            let (_, ops) = optimize_path(&path, true, true).unwrap();
            (rim.len(), ops.arc_segments)
        };
        let before = rim_stats(&baseline);
        let after = rim_stats(&result);
        assert!(after.0 < before.0);
        assert!(after.1 > before.1, "the rim must gain actual SVG arcs");
        assert!(
            result.windows(3).any(|pieces| {
                pieces
                    .iter()
                    .all(|c| c.start().x > 510.0 && c.end().x > 510.0)
                    && circle(&sample_curve_sequence(pieces, 0.25), 0.02)
                        .is_some_and(|arc| arc.len() >= 3)
            }),
            "the attached rim must contain a consistent arc spanning over 90 degrees"
        );
        assert_eq!(result[0].start(), baseline[0].start());
        assert_eq!(result.last().unwrap().end(), baseline.last().unwrap().end());
        assert!(result.windows(2).all(|p| p[0].end() == p[1].start()));
        let corridor = super::super::fairing_raster_corridor();
        assert!(boundary_corridor_supported(&source, &result, corridor));
        assert!(boundary_corridor_supported(
            &sample_curve_sequence(&baseline, 0.5),
            &result,
            corridor
        ));
        let reference = sample_curve_sequence(&baseline, 0.5);
        let rendered = sample_curve_sequence(&result, 0.25);
        for (_, corner) in persistent_open_corners(&source) {
            assert!(
                super::super::nearest_point(&rendered, corner).1
                    <= (super::super::nearest_point(&reference, corner).1 + 0.125).max(0.25)
            );
        }
        let mut next_master = 0;
        assert!(super::super::geometry_mapping::map(
            &source,
            &result,
            super::super::fairing_raster_corridor(),
            &mut next_master
        )
        .is_some());
        assert_ne!(result, baseline);
    }

    fn noisy_arc(start: f64, sweep: f64, closed: bool) -> Vec<Point> {
        let mut points: Vec<_> = (0..=360)
            .map(|i| {
                let t = i as f64 / 360.0;
                let angle = start + sweep * t;
                let radius = 60.0 + 0.25 * (t * 54.0 * std::f64::consts::PI).sin();
                Point {
                    x: (100.0 + radius * angle.cos()) as f32,
                    y: (90.0 + radius * angle.sin()) as f32,
                }
            })
            .collect();
        if closed {
            points[360] = points[0];
        }
        points
    }

    #[test]
    fn raster_staircase_becomes_one_line_with_fixed_endpoints() {
        for slope in [0.0_f32, 0.35, 1.5, -0.65] {
            let points: Vec<_> = (0..=200)
                .map(|x| Point {
                    x: x as f32,
                    y: (slope * x as f32).round(),
                })
                .collect();
            let curves = fit(&points, 0.85, None, None).expect("supported line");
            assert_eq!(
                curves,
                vec![CurveSegment::Line {
                    start: points[0],
                    end: *points.last().unwrap(),
                }]
            );
            let path = fitted_structural_open_path_data(&points, 0.85, 1.0);
            let (element, _) = optimize_path(&path, true, true).unwrap();
            assert!(matches!(element, OptimizedElement::Line { .. }));
        }
    }

    #[test]
    fn noisy_minor_major_and_closed_arcs_have_one_radius_and_export_as_arcs() {
        for (sweep, closed) in [(1.8, false), (-4.5, false), (std::f64::consts::TAU, true)] {
            for origin in [0.3, 2.8] {
                let points = noisy_arc(origin, sweep, closed);
                let curves = fit(&points, 0.65, None, None).expect("supported circular arc");
                assert_eq!(curves[0].start(), points[0]);
                assert_eq!(curves.last().unwrap().end(), *points.last().unwrap());
                for point in sample_curve_sequence(&curves, 0.25) {
                    assert!(((point.x - 100.0).hypot(point.y - 90.0) - 60.0).abs() < 0.1);
                }
                let path = structural_curve_path_data(&curves, closed);
                let (element, operations) = optimize_path(&path, true, true).unwrap();
                if closed {
                    assert!(matches!(element, OptimizedElement::Circle { .. }));
                    assert!(
                        dot(
                            tangent(curves[0], false),
                            tangent(*curves.last().unwrap(), true)
                        ) > 0.999_999
                    );
                } else {
                    assert!(
                        operations.arc_segments > 0,
                        "circle fit must reach SVG arc normalization"
                    );
                }
            }
        }
    }

    #[test]
    fn noncircular_shapes_corners_and_retraced_paths_are_not_single_primitives() {
        let ellipse: Vec<_> = noisy_arc(0.0, std::f64::consts::TAU, true)
            .into_iter()
            .map(|p| Point {
                x: p.x * 1.6,
                y: p.y,
            })
            .collect();
        let s_curve: Vec<_> = (0..=200)
            .map(|x| Point {
                x: x as f32,
                y: 15.0 * (x as f32 / 24.0).sin(),
            })
            .collect();
        let corner = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 40.0, y: 0.0 },
            Point { x: 40.0, y: 40.0 },
        ];
        let retraced = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 50.0, y: 0.0 },
            Point { x: 20.0, y: 0.0 },
        ];
        let mut double_circle = noisy_arc(0.0, std::f64::consts::TAU, true);
        double_circle.extend(
            noisy_arc(0.0, std::f64::consts::TAU, true)
                .into_iter()
                .skip(1),
        );
        for points in [ellipse, s_curve, corner, retraced, double_circle] {
            assert!(fit(&points, 0.85, None, None).is_none());
        }
    }

    #[test]
    fn graph_tangent_constraints_can_veto_a_geometric_fit() {
        let points: Vec<_> = (0..=80)
            .map(|x| Point {
                x: x as f32,
                y: 0.0,
            })
            .collect();
        let forward = Some(Point { x: 1.0, y: 0.0 });
        assert!(fit(&points, 0.65, forward, forward).is_some());
        assert!(fit(&points, 0.65, Some(Point { x: 1.0, y: 0.2 }), forward).is_none());
        assert!(fit(&points, 0.65, forward, Some(Point { x: -1.0, y: 0.0 })).is_none());
    }

    #[test]
    fn shared_piece_consolidation_does_not_accumulate_source_drift() {
        let source: Vec<_> = (0..=80)
            .map(|x| Point {
                x: x as f32,
                y: 0.0,
            })
            .collect();
        let baseline: Vec<_> = (0..8)
            .map(|i| {
                super::super::straight_cubic(
                    Point {
                        x: i as f32 * 10.0,
                        y: 0.0,
                    },
                    Point {
                        x: (i + 1) as f32 * 10.0,
                        y: 0.0,
                    },
                )
            })
            .collect();
        let fitted = regularize(&source, &baseline, 0.85, None, None);
        assert_eq!(fitted.len(), 1);
        assert_eq!(fitted[0].start(), source[0]);
        assert_eq!(fitted[0].end(), *source.last().unwrap());
        let unrelated: Vec<_> = source
            .iter()
            .map(|p| Point {
                x: p.x,
                y: p.y + 3.0,
            })
            .collect();
        assert_eq!(
            regularize(&unrelated, &baseline, 0.85, None, None),
            baseline
        );
    }

    #[test]
    fn tiny_and_degenerate_support_is_left_to_the_existing_fitter() {
        for points in [
            vec![],
            vec![Point::default(); 20],
            vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 2.0, y: 0.0 },
            ],
        ] {
            assert!(fit(&points, 0.65, None, None).is_none());
        }
    }
}
