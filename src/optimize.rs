//! Exact geometry normalization used by the serializer.
//!
//! Primitive recognition happens while region masks are still available, so
//! it is both faster and safer than reparsing a multi-megabyte SVG afterward.

use serde::Serialize;

use crate::geometry::{GeometrySummary, RegionGeometry};
use crate::gradient::Paint;

#[derive(Clone, Debug, Default, Serialize)]
pub struct OptimizationSummary {
    pub input_regions: usize,
    pub primitive_regions: usize,
    pub path_regions: usize,
    pub shared_gradient_regions: usize,
}

pub fn summarize(
    geometry: &[RegionGeometry],
    paints: &[Paint],
    report: &GeometrySummary,
) -> OptimizationSummary {
    let primitive_regions = report.rectangles + report.circles + report.ellipses;
    let mut shared = 0;
    for index in 0..paints.len() {
        if matches!(paints[index], Paint::Solid { .. }) {
            continue;
        }
        if paints[..index].iter().any(|paint| paint == &paints[index]) {
            shared += 1;
        }
    }
    OptimizationSummary {
        input_regions: geometry.len(),
        primitive_regions,
        path_regions: geometry.len().saturating_sub(primitive_regions),
        shared_gradient_regions: shared,
    }
}
