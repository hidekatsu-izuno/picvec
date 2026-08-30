use serde::Serialize;

use crate::segment::Segmentation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyCell {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub label: u32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct HierarchicalTopologySummary {
    pub nodes: usize,
    pub leaves: usize,
    pub unit_leaves: usize,
    pub represented_pixels: usize,
    pub compression_ratio: f32,
}

/// Exact non-uniform view of the dense ownership partition.
///
/// A leaf is retained only when every covered source pixel has one Paint
/// owner. Mixed cells are recursively split, so expanding the leaves always
/// reproduces `Segmentation::labels` without approximation. Downstream code
/// can therefore aggregate flat interiors by area while continuing to use
/// source-pixel cells along complex boundaries.
#[derive(Clone, Debug)]
pub struct HierarchicalTopology {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<TopologyCell>,
    region_leaf_counts: Vec<usize>,
    pub summary: HierarchicalTopologySummary,
}

impl HierarchicalTopology {
    pub fn build(segmentation: &Segmentation) -> Self {
        let (cells, nodes) = uniform_cells(
            &segmentation.labels,
            segmentation.width,
            segmentation.height,
        );
        let mut region_leaf_counts = vec![0_usize; segmentation.regions.len()];
        for cell in &cells {
            if let Some(count) = region_leaf_counts.get_mut(cell.label as usize) {
                *count += 1;
            }
        }
        let pixels = segmentation.width * segmentation.height;
        let leaves = cells.len();
        let summary = HierarchicalTopologySummary {
            nodes,
            leaves,
            unit_leaves: cells
                .iter()
                .filter(|cell| cell.width == 1 && cell.height == 1)
                .count(),
            represented_pixels: pixels,
            compression_ratio: pixels as f32 / leaves.max(1) as f32,
        };
        Self {
            width: segmentation.width,
            height: segmentation.height,
            cells,
            region_leaf_counts,
            summary,
        }
    }

    pub fn is_compatible(&self, segmentation: &Segmentation) -> bool {
        self.dimensions_match(segmentation)
            && self.cells.iter().all(|cell| {
                (cell.y..cell.y + cell.height).all(|y| {
                    (cell.x..cell.x + cell.width)
                        .all(|x| segmentation.labels[y * segmentation.width + x] == cell.label)
                })
            })
    }

    pub fn dimensions_match(&self, segmentation: &Segmentation) -> bool {
        self.width == segmentation.width && self.height == segmentation.height
    }

    /// Validate a Paint owner against the hierarchy before assigning its
    /// sample budget. The current quality gate keeps the full budget for
    /// every represented owner; lower budgets are intentionally not enabled
    /// because they changed SSIM on the heterogeneous regression image.
    pub fn paint_sample_budget(&self, label: usize, maximum: usize) -> usize {
        if self.region_leaf_counts.get(label).copied().unwrap_or(0) == 0 {
            64
        } else {
            maximum.max(64)
        }
    }
}

pub(crate) fn uniform_cells(
    labels: &[u32],
    width: usize,
    height: usize,
) -> (Vec<TopologyCell>, usize) {
    assert_eq!(labels.len(), width * height);
    let mut cells = Vec::new();
    let mut nodes = 0_usize;
    if width > 0 && height > 0 {
        split_cell(labels, 0, 0, width, height, width, &mut nodes, &mut cells);
    }
    (cells, nodes)
}

#[allow(clippy::too_many_arguments)]
fn split_cell(
    labels: &[u32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    nodes: &mut usize,
    cells: &mut Vec<TopologyCell>,
) {
    *nodes += 1;
    let label = labels[y * stride + x];
    let uniform = (y..y + height).all(|row| {
        labels[row * stride + x..row * stride + x + width]
            .iter()
            .all(|&v| v == label)
    });
    if uniform || (width == 1 && height == 1) {
        cells.push(TopologyCell {
            x,
            y,
            width,
            height,
            label,
        });
        return;
    }
    let left_width = width.div_ceil(2);
    let right_width = width - left_width;
    let top_height = height.div_ceil(2);
    let bottom_height = height - top_height;
    for (child_x, child_y, child_width, child_height) in [
        (x, y, left_width, top_height),
        (x + left_width, y, right_width, top_height),
        (x, y + top_height, left_width, bottom_height),
        (x + left_width, y + top_height, right_width, bottom_height),
    ] {
        if child_width > 0 && child_height > 0 {
            split_cell(
                labels,
                child_x,
                child_y,
                child_width,
                child_height,
                stride,
                nodes,
                cells,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb_to_lab;
    use crate::raster::Raster;
    use crate::segment::{RegionStats, SegmentationSummary};

    #[test]
    fn hierarchy_expands_to_the_exact_dense_partition() {
        let width = 8;
        let height = 6;
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| u32::from(x >= 5 || (x >= 3 && y >= 4))))
            .collect();
        let source = Raster::blank(width, height, [0.5; 3]);
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical: source,
            regions: (0..2)
                .map(|id| RegionStats {
                    id,
                    area: labels.iter().filter(|&&label| label == id).count(),
                    min_x: 0,
                    min_y: 0,
                    max_x: width,
                    max_y: height,
                    mean_rgb: [0.5; 3],
                    mean_lab: rgb_to_lab([0.5; 3]),
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let hierarchy = HierarchicalTopology::build(&segmentation);
        let mut expanded = vec![u32::MAX; labels.len()];
        for cell in &hierarchy.cells {
            for y in cell.y..cell.y + cell.height {
                expanded[y * width + cell.x..y * width + cell.x + cell.width].fill(cell.label);
            }
        }
        assert_eq!(expanded, labels);
        assert!(hierarchy.is_compatible(&segmentation));
        assert!(hierarchy.cells.len() < labels.len());
    }
}
