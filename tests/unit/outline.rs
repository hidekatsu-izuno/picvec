mod tests {
    use super::*;
    use crate::{
        color::rgb_to_oklab,
        segment::{RegionStats, SegmentationSummary},
    };

    #[test]
    fn ellipse_inset_recovers_bounded_asymmetric_rims() {
        let profile = |shift: Point| {
            (0..64)
                .map(|i| {
                    let angle = i as f32 * std::f32::consts::TAU / 64.0;
                    let normal = Point {
                        x: angle.cos(),
                        y: angle.sin(),
                    };
                    (normal, 3.0 + normal.x * shift.x + normal.y * shift.y)
                })
                .collect::<Vec<_>>()
        };
        let expected = Point { x: -0.5, y: 0.6 };
        let (width, offset) = ellipse_inset(&profile(expected), 3.0);
        assert!((width - 3.0).abs() < 1e-4);
        assert!(offset.distance(expected) < 1e-4);
        assert_eq!(
            ellipse_inset(&profile(Point { x: 2.0, y: 0.0 }), 3.0),
            (3.0, Point::default())
        );
        assert_eq!(
            ellipse_inset(&[(Point { x: 1.0, y: 0.0 }, 3.0); 8], 3.0),
            (3.0, Point::default())
        );
    }

    fn disc(background: [f32; 3]) -> (Raster, Segmentation, Vec<Point>) {
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        for y in 0..96 {
            for x in 0..96 {
                let radius = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                let id = if radius >= 30.0 {
                    0
                } else if radius >= 28.0 {
                    1
                } else if x < 48 {
                    2
                } else {
                    3
                };
                labels.push(id);
                pixels.push(match id {
                    0 => background,
                    1 => [0.05, 0.1, 0.2],
                    2 => [0.4, 0.65, 0.9],
                    _ => [0.42, 0.67, 0.9],
                });
            }
        }
        let source = Raster::new(96, 96, pixels);
        let regions = (0..4)
            .map(|id| {
                let owned: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, label)| **label == id)
                    .map(|(i, _)| i)
                    .collect();
                let color = source.pixels[owned[0]];
                RegionStats {
                    id,
                    area: owned.len(),
                    min_x: 0,
                    min_y: 0,
                    max_x: 95,
                    max_y: 95,
                    mean_rgb: color,
                    mean_lab: rgb_to_oklab(color),
                }
            })
            .collect();
        let segmentation = Segmentation {
            width: 96,
            height: 96,
            labels,
            paint_keys: vec![0, 1, 2, 3],
            paint_samples: vec![true; 96 * 96],
            canonical: source.clone(),
            regions,
            summary: SegmentationSummary::default(),
        };
        let mut contour: Vec<_> = (0..360)
            .map(|i| {
                let theta = i as f32 * std::f32::consts::TAU / 360.0;
                Point {
                    x: 48.0 + 30.0 * theta.cos(),
                    y: 48.0 + 30.0 * theta.sin(),
                }
            })
            .collect();
        contour.push(contour[0]);
        (source, segmentation, contour)
    }

    #[test]
    fn parallel_contours_keep_serial_ownership_and_fallback_priority() {
        let (source, segmentation, points) = disc([1.0; 3]);
        let inner: Vec<_> = points
            .iter()
            .map(|p| Point {
                x: 48.0 + (p.x - 48.0) * 28.0 / 30.0,
                y: 48.0 + (p.y - 48.0) * 28.0 / 30.0,
            })
            .collect();
        let outer = crate::geometry::ClosedContour::from_points(&points);
        let inner = crate::geometry::ClosedContour::from_points(&inner);
        let mut fallback = outer.clone();
        fallback.fallback = Some(Box::new(inner.clone()));
        for contours in [
            vec![outer.clone(), inner.clone()],
            vec![inner.clone(), outer.clone()],
            vec![fallback.clone(), outer.clone(), fallback, inner],
        ] {
            let expected = propose_with_alignment(&source, None, &segmentation, &contours, false);
            assert!(!expected.is_empty());
            for count in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(count)
                    .build()
                    .unwrap();
                let actual = pool.install(|| propose(&source, None, &segmentation, &contours));
                assert_eq!(actual, expected, "threads={count}");
            }
        }
    }

    #[test]
    fn reconstructs_ink_outside_a_fill_contour_in_both_directions() {
        let (source, segmentation, outer) = disc([1.0; 3]);
        let inner: Vec<_> = outer
            .iter()
            .map(|p| Point {
                x: 48.0 + (p.x - 48.0) * 28.0 / 30.0,
                y: 48.0 + (p.y - 48.0) * 28.0 / 30.0,
            })
            .collect();
        for reverse in [false, true] {
            let mut points = inner.clone();
            if reverse {
                points.reverse();
            }
            let shift = source_outer_offset(&source, &points).unwrap();
            assert!((1.5..=2.5).contains(&shift), "outer offset: {shift}");
            let bands = propose(
                &source,
                None,
                &segmentation,
                &[crate::geometry::ClosedContour::from_points(&points)],
            );
            assert_eq!(bands.len(), 1, "reversed={reverse}");
            assert!(bands[0].regions.contains(&1));
            // The band owns the ink and both adjacent 1.25px fill collars.
            assert!((3.5..=5.5).contains(&bands[0].width));
        }
        assert!(source_outer_offset(&source, &outer).is_none());
        let (plain, _, _) = disc([0.0; 3]);
        assert!(source_outer_offset(&plain, &inner).is_none());
    }

    #[test]
    fn reconstructs_a_ridge_but_not_a_plain_brightness_edge() {
        let (source, segmentation, contour) = disc([1.0; 3]);
        let bands = propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)],
        );
        assert_eq!(bands.len(), 1);
        assert!((2.0..=4.5).contains(&bands[0].width));
        assert!(bands[0].regions.contains(&1));
        let (source, segmentation, contour) = disc([0.0; 3]);
        assert!(propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)]
        )
        .is_empty());
    }

    #[test]
    fn rendered_colour_changes_and_erased_detail_are_rejected() {
        let (source, segmentation, contour) = disc([1.0; 3]);
        let band = propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)],
        )
        .remove(0);
        assert!(band.supported_by_render(&source, &source, &source));
        let mut damaged = source.clone();
        for &(i, _) in &band.pixels {
            damaged.pixels[i] = [1.0, 0.0, 0.0];
        }
        assert!(!band.supported_by_render(&source, &source, &damaged));
        let mut gap_source = source.clone();
        for &(i, rim) in &band.pixels {
            if rim && i % 96 > 48 {
                gap_source.pixels[i] = [1.0; 3];
            }
        }
        assert!(!band.supported_by_render(&gap_source, &gap_source, &source));
    }
}
