mod tests {
    use super::*;
    use crate::color::rgb_to_oklab;
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
                    mean_lab: rgb_to_oklab([0.5; 3]),
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
