use serde::Serialize;

use crate::geometry::Point;
use crate::raster::Raster;
use crate::structural::{select_missing_with_junctions, StructuralInk};

/// Resolve structural ownership against unexpanded paint. The SVG uses
/// overlap incorporated into shared fill contours, not auxiliary strokes.
#[derive(Clone, Debug)]
pub struct BoundaryOwnership {
    pub structural: StructuralInk,
    pub paint_overlap: f32,
    pub summary: BoundaryOwnershipSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BoundaryOwnershipSummary {
    pub structural_strokes: usize,
    pub paint_overlap: f32,
    pub overlap_is_underpaint: bool,
    pub overlap_is_fill_geometry: bool,
}

pub fn resolve(
    source: &Raster,
    unexpanded_paint_render: &Raster,
    candidates: &StructuralInk,
    paint_junctions: &[Point],
    requested_overlap: f32,
) -> BoundaryOwnership {
    let structural =
        select_missing_with_junctions(source, unexpanded_paint_render, candidates, paint_junctions);
    let paint_overlap = requested_overlap.max(0.0);
    BoundaryOwnership {
        summary: BoundaryOwnershipSummary {
            structural_strokes: structural.strokes.len(),
            paint_overlap,
            overlap_is_underpaint: false,
            overlap_is_fill_geometry: true,
        },
        structural,
        paint_overlap,
    }
}
