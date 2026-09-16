mod tests {
    use super::*;

    #[test]
    fn graph_nodes_follow_position_not_uniform_bezier_parameter() {
        let curve = CurveSegment::Cubic {
            start: Point { x: 0.0, y: 0.0 },
            first: Point { x: 0.1, y: 0.1 },
            second: Point { x: 0.2, y: 0.2 },
            end: Point { x: 20.0, y: 20.0 },
        };
        let source: Vec<_> = (0..=20)
            .map(|i| Point {
                x: i as f32,
                y: i as f32,
            })
            .collect();
        let mapping = map(&source, &[curve], 0.1, &mut 0).unwrap();
        assert!(mapping
            .positions
            .iter()
            .zip(&source)
            .all(|(&a, &b)| a.distance(b) < 0.01));
        assert!(mapping
            .edges
            .iter()
            .flatten()
            .all(|span| span.master_id == 0));
        assert_eq!(mapping.positions[0], source[0]);
        assert_eq!(*mapping.positions.last().unwrap(), *source.last().unwrap());
    }

    #[test]
    fn raster_backtracking_is_pooled_without_reversing_the_shared_master() {
        let source = [0.0, 5.0, 4.5, 10.0, 20.0].map(|x| Point { x, y: x });
        let curve = CurveSegment::Line {
            start: source[0],
            end: source[4],
        };
        let mapping = map(&source, &[curve], 0.5, &mut 0).unwrap();
        assert_eq!(mapping.positions[1], mapping.positions[2]);
        assert!(mapping
            .positions
            .windows(2)
            .all(|pair| pair[0].x <= pair[1].x));
        assert!(mapping
            .edges
            .iter()
            .flatten()
            .all(|span| span.start_parameter <= span.end_parameter));
        assert!(
            map(&source, &[curve], 0.1, &mut 0).is_none(),
            "an unsupported correspondence must reject the whole fit"
        );
    }
}
