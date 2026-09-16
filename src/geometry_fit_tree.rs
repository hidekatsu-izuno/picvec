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
            let predicted = cubic_points(curve, &parameters);
            let mut split = 0_usize;
            let mut error = 0.0;
            for (index, (&point, predicted)) in points.iter().zip(predicted).enumerate() {
                let dx = predicted.x - point.x;
                let dy = predicted.y - point.y;
                let squared = dx * dx + dy * dy;
                // Preserve the first maximum, including the original NaN behavior.
                if index == 0 || squared > error {
                    split = index;
                    error = squared;
                }
            }
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
include!("../tests/unit/geometry_fit_tree.rs");
