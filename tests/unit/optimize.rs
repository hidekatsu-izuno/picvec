mod tests {
    use super::*;

    fn test_curve() -> Segment {
        Segment::cubic(
            VectorPoint { x: 0.0, y: 0.0 },
            VectorPoint { x: 15.0, y: 30.0 },
            VectorPoint { x: 35.0, y: -20.0 },
            VectorPoint { x: 50.0, y: 5.0 },
        )
    }

    #[test]
    fn rounded_cubic_slices_recover_one_curve_in_both_directions() {
        let curve = test_curve();
        let pieces: Vec<_> = (0..24)
            .map(|i| cubic_interval(curve, i as f64 / 24.0, (i + 1) as f64 / 24.0))
            .collect();
        // Geometry arrives at the serializer rounded to three decimals.
        let round = |p: VectorPoint| VectorPoint {
            x: (p.x * 1000.0).round() / 1000.0,
            y: (p.y * 1000.0).round() / 1000.0,
        };
        for reverse in [false, true] {
            let mut segments: Vec<_> = pieces
                .iter()
                .map(|c| {
                    Segment::cubic(
                        round(c.start),
                        round(c.first),
                        round(c.second),
                        round(c.end),
                    )
                })
                .collect();
            if reverse {
                segments.reverse();
                for c in &mut segments {
                    *c = Segment::cubic(c.end, c.second, c.first, c.start);
                }
            }
            let mut path = Subpath {
                start: segments[0].start,
                segments,
                closed: false,
            };
            assert_eq!(consolidate_cubics(&mut path), 23);
            let result = path.segments[0];
            for i in 0..=1000 {
                let t = i as f64 / 1000.0;
                let expected = split_cubic(curve, if reverse { 1.0 - t } else { t }).0.end;
                assert!(split_cubic(result, t).0.end.distance(expected) < 0.1);
            }
        }
    }

    #[test]
    fn consolidation_keeps_corners_and_rejects_large_shape_changes() {
        let (left, right) = split_cubic(test_curve(), 0.5);
        for modified in [
            Segment {
                first: VectorPoint {
                    x: right.start.x,
                    y: right.start.y + 5.0,
                },
                ..right
            },
            Segment {
                second: VectorPoint {
                    x: right.second.x,
                    y: right.second.y + 5.0,
                },
                ..right
            },
            Segment {
                first: right.start,
                ..right
            },
        ] {
            let mut path = Subpath {
                start: left.start,
                segments: vec![left, modified],
                closed: false,
            };
            assert_eq!(consolidate_cubics(&mut path), 0);
        }
    }

    #[test]
    fn repeated_joins_remain_close_to_all_original_pieces() {
        let curve = test_curve();
        let pieces: Vec<_> = (0..48)
            .map(|i| {
                let mut piece = cubic_interval(curve, i as f64 / 48.0, (i + 1) as f64 / 48.0);
                let offset = (i as f64 * 0.6).sin() * 0.09;
                piece.first.y += offset;
                piece.second.y += offset;
                piece
            })
            .collect();
        let reference: Vec<_> = pieces
            .iter()
            .flat_map(|&c| (0..=200).map(move |i| split_cubic(c, i as f64 / 200.0).0.end))
            .collect();
        let mut path = Subpath {
            start: curve.start,
            segments: pieces,
            closed: false,
        };
        assert!(consolidate_cubics(&mut path) > 0);
        let result: Vec<_> = path
            .segments
            .iter()
            .flat_map(|&c| (0..=2000).map(move |i| split_cubic(c, i as f64 / 2000.0).0.end))
            .collect();
        for (from, to) in [(&reference, &result), (&result, &reference)] {
            assert!(from
                .iter()
                .all(|p| to.iter().any(|q| p.distance(*q) <= 0.11)));
        }
    }

    #[test]
    fn disjoint_compound_path_remains_batchable_but_nested_path_does_not() {
        let disjoint = parse_path("M0 0L4 0L4 4L0 4Z M10 0L14 0L14 4L10 4Z").unwrap();
        assert_eq!(path_bbox(&disjoint), Some((0.0, 0.0, 14.0, 4.0)));
        let nested = parse_path("M0 0L14 0L14 14L0 14Z M4 4L10 4L10 10L4 10Z").unwrap();
        assert!(path_bbox(&nested).is_none());
    }
}
