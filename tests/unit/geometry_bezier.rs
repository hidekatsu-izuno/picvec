mod tests {
    use super::*;

    #[test]
    fn fixed_point_exit_matches_all_ten_iterations() {
        for count in [4, 17, 80, 257] {
            for shape in 0..4 {
                let points: Vec<_> = (0..count)
                    .map(|i| {
                        let x = i as f32 * 0.5;
                        Point {
                            x,
                            y: match shape {
                                0 => x * 0.2,
                                1 => (x * 0.07).sin() * 5.0,
                                2 => (x * 0.07).sin().round() * 5.0,
                                _ => (i % 7) as f32,
                            },
                        }
                    })
                    .collect();
                for tangent in [None, Some(Point { x: 1.0, y: 0.0 })] {
                    let expected = one_with_convergence::<false>(&points, tangent, tangent);
                    let actual = one(&points, tangent, tangent);
                    assert_eq!(actual.is_some(), expected.is_some());
                    if let (Some((a, ae)), Some((b, be))) = (actual, expected) {
                        assert_eq!(a, b);
                        assert_eq!(ae.maximum.to_bits(), be.maximum.to_bits());
                        assert_eq!(ae.rms.to_bits(), be.rms.to_bits());
                        assert_eq!(ae.worst, be.worst);
                    }
                }
            }
        }
    }

    #[test]
    fn car_hood_tonal_contour_can_use_fewer_curves_with_fixed_end_tangents() {
        let record: serde_json::Value =
            serde_json::from_str(include_str!("../data/car-hood-contour.json")).unwrap();
        let point = |v: &serde_json::Value| Point {
            x: v[0].as_f64().unwrap() as f32,
            y: v[1].as_f64().unwrap() as f32,
        };
        let mut source: Vec<_> = record["source"]
            .as_array()
            .unwrap()
            .iter()
            .map(point)
            .collect();
        let baseline: Vec<_> = record["curves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| CurveSegment::Cubic {
                start: point(&c["start"]),
                first: point(&c["first"]),
                second: point(&c["second"]),
                end: point(&c["end"]),
            })
            .collect();
        source[0] = baseline[0].start();
        *source.last_mut().unwrap() = baseline.last().unwrap().end();
        let result = fit_shading(&source, &baseline, 6.0).unwrap();
        let image = crate::raster::Raster::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png"),
            2000,
            4_000_000,
            64_000_000,
        )
        .unwrap();
        let accepted = super::super::fair_shading_contour(&image, &source, &baseline);
        assert!(
            accepted.len() <= 2,
            "the source-colour gate must allow the hood fit"
        );
        assert!(result.len() <= 2);
        assert_eq!(result[0].start(), baseline[0].start());
        assert_eq!(result.last().unwrap().end(), baseline.last().unwrap().end());
        assert!(super::super::geometry_primitives::supports_tangents(
            &result,
            Some(derivatives(baseline[0], 0.0).0),
            Some(derivatives(*baseline.last().unwrap(), 1.0).0)
        ));
    }

    #[test]
    fn equal_adjacent_corner_turns_do_not_reject_the_window_loop() {
        // The same lower window corner occupies (683,513) and (683,514).
        // Treating its two 90-degree maxima as separate knots previously
        // rejected the whole contour, irrespective of the curve budget.
        let points: Vec<_> = include_str!("../data/car-window-contour.txt")
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
            serde_json::from_str(include_str!("../data/wifi-closed-contours.json")).unwrap();
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
    fn partial_analytic_run_merges_up_to_a_real_corner() {
        let pieces = [
            CurveSegment::Line {
                start: Point { x: 0.0, y: 0.0 },
                end: Point { x: 20.0, y: 0.0 },
            },
            CurveSegment::Line {
                start: Point { x: 20.0, y: 0.0 },
                end: Point { x: 40.0, y: 0.0 },
            },
            CurveSegment::Line {
                start: Point { x: 40.0, y: 0.0 },
                end: Point { x: 60.0, y: 0.0 },
            },
            cubic((60.0, 0.0), (60.0, 20.0), (60.0, 40.0), (60.0, 60.0)),
        ];
        let source = sample_curve_sequence(&pieces, 0.5);
        let result = compact(&source, &pieces, 1.0, None, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].end(), pieces[2].end());
        assert_eq!(result[1], pieces[3]);
        assert!(error(&source, &result).maximum < 0.01);
    }

    #[test]
    fn three_cubics_merge_before_a_corner_without_a_new_endpoint_kink() {
        let model = cubic((0.0, 0.0), (40.0, 0.0), (60.0, 40.0), (100.0, 40.0));
        let mut pieces: Vec<_> = (0..3)
            .map(|i| super::super::curve_interval(model, i as f64 / 3.0, (i + 1) as f64 / 3.0))
            .collect();
        pieces.push(CurveSegment::Line {
            start: model.end(),
            end: Point { x: 100.0, y: 100.0 },
        });
        let source = sample_curve_sequence(&pieces, 0.5);
        let result = compact(&source, &pieces, 1.0, None, None);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0], CurveSegment::Cubic { .. }));
        assert_eq!(result[0].end(), model.end());
        assert_eq!(result[1], pieces[3]);
        assert!(error(&source, &result).maximum < 0.05);
        assert!(
            dot(
                normalized(derivatives(result[0], 1.0).0),
                normalized(derivatives(model, 1.0).0)
            ) > 0.999_999
        );
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
    fn remojii_spiral_arc_uses_one_cubic_with_shared_graph_mapping() {
        let record: serde_json::Value =
            serde_json::from_str(include_str!("../data/remojii-spiral-contour.json")).unwrap();
        let point = |p: &serde_json::Value| Point {
            x: p[0].as_f64().unwrap() as f32,
            y: p[1].as_f64().unwrap() as f32,
        };
        let source: Vec<_> = record["source"]
            .as_array()
            .unwrap()
            .iter()
            .map(point)
            .collect();
        let baseline: Vec<_> = record["result"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| CurveSegment::Cubic {
                start: point(&c[0]),
                first: point(&c[1]),
                second: point(&c[2]),
                end: point(&c[3]),
            })
            .collect();
        let tolerance = record["raster_corridor"].as_f64().unwrap() as f32;
        let result = compact(&source, &baseline, tolerance, None, None);
        assert_eq!(baseline.len(), 6);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], CurveSegment::Cubic { .. }));
        assert_eq!(result[0].start(), baseline[0].start());
        assert_eq!(result[0].end(), baseline.last().unwrap().end());
        assert!(error(&source, &result).maximum <= 1.25);
        assert!(super::super::geometry_mapping::map(&source, &result, tolerance, &mut 0).is_some());
    }

    #[test]
    fn tiny_analytic_pieces_do_not_block_a_source_supported_cubic() {
        let model = cubic((0.0, 0.0), (40.0, 1.0), (60.0, 21.0), (120.0, 20.0));
        let source: Vec<_> = (0..=240)
            .map(|i| cubic_point(model, i as f32 / 240.0))
            .collect();
        let pieces: Vec<_> = source
            .windows(2)
            .map(|p| CurveSegment::Line {
                start: p[0],
                end: p[1],
            })
            .collect();
        let result = compact(&source, &pieces, 1.0, None, None);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], CurveSegment::Cubic { .. }));
        assert!(error(&source, &result).maximum < 0.3);
        assert_eq!(result[0].start(), source[0]);
        assert_eq!(result[0].end(), *source.last().unwrap());
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
            serde_json::from_str(include_str!("../data/man-shoulder-contours.json")).unwrap();
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
