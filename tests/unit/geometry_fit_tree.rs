mod tests {
    use super::*;

    #[test]
    fn shared_tree_matches_fresh_fits_in_both_tolerance_orders_and_budgets() {
        for count in [2, 3, 17, 129, 513] {
            for shape in 0..3 {
                let points: Vec<_> = (0..count)
                    .map(|i| {
                        let x = i as f32 * 0.25;
                        Point {
                            x,
                            y: match shape {
                                0 => (x * 0.5).sin() * 3.0,
                                1 => ((i / 5) % 3) as f32,
                                _ => 0.0,
                            },
                        }
                    })
                    .collect();
                let left = normalized(Point {
                    x: points[1].x - points[0].x,
                    y: points[1].y - points[0].y,
                });
                let last = points.len() - 1;
                let right = normalized(Point {
                    x: points[last - 1].x - points[last].x,
                    y: points[last - 1].y - points[last].y,
                });
                let mut tree = FitTree::new(&points, left, right);
                for tolerance in [1.25_f32, 0.75, 0.125, 1.0, 4.0, 0.25] {
                    for budget in [0, 1, 2, 7, 1000] {
                        let cached = tree.fit(tolerance * tolerance, budget);
                        let fresh = fit_cubic_with_budget(
                            &points,
                            left,
                            right,
                            tolerance * tolerance,
                            budget,
                        );
                        assert_eq!(
                            format!("{cached:?}"),
                            format!("{fresh:?}"),
                            "count={count} shape={shape} tolerance={tolerance} budget={budget}"
                        );
                    }
                }
            }
        }
    }
}
