use serde::Serialize;

use crate::geometry::Point;
use crate::raster::Raster;
use crate::structural::{select_missing_with_junctions, StructuralInk};

/// One final decision for Paint interfaces, structural ink, and seam
/// underpaint.
///
/// The overlap stroke is deliberately not an owner: it is attached only
/// after source-supported structural ownership has been resolved against the
/// unexpanded Paint partition.  This prevents a renderer seam correction
/// from deleting an authored line while still ensuring the final SVG uses the
/// exact same structural owner selected here.
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
            overlap_is_underpaint: true,
        },
        structural,
        paint_overlap,
    }
}
