//! Lazily shared split trees for fits with different error tolerances.
use super::*;

pub(super) struct FitTree<'a> {
    points: &'a [Point],
    left: Point,
    right: Point,
    observation: Option<(CurveSegment, f32, usize, Point)>,
    children: Option<Box<(FitTree<'a>, FitTree<'a>)>>,
}

impl<'a> FitTree<'a> {
    pub(super) fn new(points: &'a [Point], left: Point, right: Point) -> Self {
        Self {
            points,
            left,
            right,
            observation: None,
            children: None,
        }
    }

    pub(super) fn fit(
        &mut self,
        tolerance_squared: f32,
        budget: usize,
    ) -> Option<Vec<CurveSegment>> {
        if budget == 0 {
            return None;
        }
        if self.points.len() == 2 {
            return fit_cubic_with_budget(
                self.points,
                self.left,
                self.right,
                tolerance_squared,
                budget,
            );
        }
        let (curve, error, split, centre) = *self.observation.get_or_insert_with(|| {
            let points = self.points;
            let left_tangent = self.left;
            let right_tangent = self.right;
            let parameters = chord_parameters(points);
            let curve = least_squares_cubic(points, &parameters, left_tangent, right_tangent);
            let mut errors = Vec::with_capacity(points.len());
            let predicted = cubic_points(curve, &parameters);
            for (&point, predicted) in points.iter().zip(predicted) {
                let dx = predicted.x - point.x;
                let dy = predicted.y - point.y;
                errors.push(dx * dx + dy * dy);
            }
            let mut split = 0_usize;
            for index in 1..errors.len() {
                if errors[index] > errors[split] {
                    split = index;
                }
            }
            let error = errors[split];
            if split == 0 || split + 1 == points.len() {
                split = points.len() / 2;
            }
            let mut centre_tangent = normalized(Point {
                x: points[split - 1].x - points[split + 1].x,
                y: points[split - 1].y - points[split + 1].y,
            });
            if centre_tangent.x == 0.0 && centre_tangent.y == 0.0 {
                centre_tangent = normalized(Point {
                    x: points[split].x - points[split + 1].x,
                    y: points[split].y - points[split + 1].y,
                });
            }
            (curve, error, split, centre_tangent)
        });
        if error <= tolerance_squared {
            return Some(vec![curve]);
        }
        if budget < 2 {
            return None;
        }
        let children = self.children.get_or_insert_with(|| {
            Box::new((
                Self::new(&self.points[..=split], self.left, centre),
                Self::new(
                    &self.points[split..],
                    Point {
                        x: -centre.x,
                        y: -centre.y,
                    },
                    self.right,
                ),
            ))
        });
        let mut fitted = children.0.fit(tolerance_squared, budget - 1)?;
        let remaining = budget - fitted.len();
        fitted.extend(children.1.fit(tolerance_squared, remaining)?);
        Some(fitted)
    }
}

#[cfg(test)]
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
