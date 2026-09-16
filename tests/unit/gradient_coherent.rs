mod tests {
    use super::*;
    fn partition(source: &Raster, labels: Vec<u32>) -> Segmentation {
        let mut segmentation = Segmentation {
            width: source.width,
            height: source.height,
            labels: vec![0; source.pixels.len()],
            paint_keys: vec![0],
            paint_samples: vec![true; source.pixels.len()],
            canonical: source.clone(),
            regions: vec![],
            summary: Default::default(),
        };
        replace_source_supported_paint_labels(source, &mut segmentation, labels, 0);
        segmentation
    }
    #[test]
    fn shared_quantizer_owners_do_not_join_fields_across_an_occluding_line() {
        let mut source = Raster::blank(256, 128, [1.0; 3]);
        let mut labels = vec![0; 256 * 128];
        for y in 8..120 {
            for x in 8..248 {
                let t = if x < 128 {
                    (y - 8) as f32 / 111.0
                } else {
                    (x - 128) as f32 / 119.0
                };
                source.pixels[y * 256 + x] = [0.2 + 0.5 * t; 3];
                labels[y * 256 + x] = 1 + (t * 3.0).min(2.0) as u32;
            }
        }
        let mut boundary = source.clone();
        for y in 8..120 {
            for x in 127..130 {
                boundary.pixels[y * 256 + x] = [0.0; 3];
            }
        }
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &boundary, &mut segmentation, &Config::default());
        let left = segmentation.labels[20 * 256 + 64];
        let right = segmentation.labels[64 * 256 + 150];
        assert_ne!(left, right);
        assert_eq!(left, segmentation.labels[100 * 256 + 64]);
        assert_eq!(right, segmentation.labels[64 * 256 + 230]);
        for (x, y) in [(64, 20), (64, 100), (150, 64), (230, 64)] {
            let i = y * 256 + x;
            let paint = hints[segmentation.labels[i] as usize].as_ref().unwrap();
            assert!(
                delta_e_ok(
                    rgb_to_oklab(source.pixels[i]),
                    rgb_to_oklab(paint_at(paint, i, 256))
                ) < 1.0
            );
        }
    }

    #[test]
    fn quiet_bridge_cannot_erase_a_measured_material_interface() {
        let source = Raster::new(
            128,
            96,
            (0..128 * 96)
                .map(|i| {
                    let x = i % 128;
                    let y = i / 128;
                    let step = if y < 8 {
                        x as f32 / 127.0
                    } else if x < 64 {
                        0.0
                    } else {
                        1.0
                    };
                    [0.4 + 0.04 * step; 3]
                })
                .collect(),
        );
        let labels = (0..128 * 96)
            .map(|i| ((i / 128 / 16) * 8 + i % 128 / 16) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_ne!(
            segmentation.labels[48 * 128 + 63],
            segmentation.labels[48 * 128 + 64]
        );
    }
    #[test]
    fn a_single_fittable_face_is_not_cut_into_spatial_patches() {
        for noise in [0.0, 0.006] {
            let source = Raster::new(
                256,
                256,
                (0..256 * 256)
                    .map(|i| {
                        [0.3 + 0.3 * (i % 256) as f32 / 255.0
                            + noise * ((i / 256 % 7) as f32 - 3.0); 3]
                    })
                    .collect(),
            );
            let labels = (0..256 * 256)
                .map(|i| {
                    u32::from((32..224).contains(&(i % 256)) && (32..224).contains(&(i / 256)))
                })
                .collect();
            let mut segmentation = partition(&source, labels);
            let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
            assert!(hints[1].is_some());
            let protected = hints.iter().map(Option::is_some).collect::<Vec<_>>();
            crate::segment::split_adaptive_paint_patches_with_protected(
                &source,
                &source,
                &mut segmentation,
                &protected,
            );
            assert_eq!(segmentation.regions.len(), 2);
        }
    }

    #[test]
    fn a_local_highlight_is_refined_without_subdividing_its_field() {
        let source = Raster::new(
            256,
            256,
            (0..256 * 256)
                .map(|i| {
                    let x = (i % 256) as f32;
                    let y = (i / 256) as f32;
                    let r = ((x - 110.0) / 28.0).hypot((y - 120.0) / 28.0);
                    [0.3 + 0.3 * x / 255.0 + 0.2 * (-r * r).exp(); 3]
                })
                .collect(),
        );
        let labels = (0..256 * 256)
            .map(|i| u32::from((32..224).contains(&(i % 256)) && (32..224).contains(&(i / 256))))
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert!(hints[1].is_some());
        let protected = hints.iter().map(Option::is_some).collect::<Vec<_>>();
        crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut segmentation,
            &protected,
        );
        assert_eq!(segmentation.regions.len(), 2);
        let branches = crate::ridge::StrongRidgeBranches {
            dark: vec![false; 256 * 256],
            bright: vec![false; 256 * 256],
        };
        let (paints, _, _) = fit_all_without_topology(
            &hints,
            &source,
            &source,
            &segmentation,
            &branches,
            &Config::default(),
        );
        let centre = 120 * 256 + 110;
        let original = paint_at(hints[1].as_ref().unwrap(), centre, 256)[0];
        let corrected = paint_at(&paints[1], centre, 256)[0];
        let target = source.pixels[centre][0];
        assert!(
            (target - corrected).abs() < (target - original).abs() * 0.6,
            "{original} -> {corrected}, source {target}"
        );
    }

    #[test]
    fn blurred_step_inside_quantizer_faces_is_not_merged_away() {
        let source = Raster::new(
            256,
            128,
            (0..256 * 128)
                .map(|i| {
                    let x = (i % 256) as f32;
                    let step = ((x - 124.0) / 8.0).clamp(0.0, 1.0);
                    [0.4 + 0.04 * step; 3]
                })
                .collect(),
        );
        // The source step falls inside a face, not on a label interface.
        let labels = (0..256 * 128)
            .map(|i| ((i % 256 + 16) / 32) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_ne!(
            segmentation.labels[64 * 256 + 96],
            segmentation.labels[64 * 256 + 160]
        );
    }

    #[test]
    fn perforated_smooth_interior_does_not_create_extra_boundaries() {
        let source = Raster::new(
            128,
            128,
            (0..128 * 128)
                .map(|i| {
                    let spot = if i % 128 % 8 == 4 && i / 128 % 8 == 4 {
                        0.1
                    } else {
                        0.0
                    };
                    [0.3 + 0.3 * (i % 128) as f32 / 127.0 + spot; 3]
                })
                .collect(),
        );
        let labels = (0..128 * 128)
            .map(|i| {
                if i % 128 % 8 == 4 && i / 128 % 8 == 4 {
                    3
                } else {
                    (i % 128 / 43).min(2) as u32
                }
            })
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_eq!(hints.iter().flatten().count(), 1);
        assert_eq!(segmentation.regions.len(), 2);
        assert_ne!(
            segmentation.labels[4 * 128 + 4],
            segmentation.labels[4 * 128 + 3]
        );
    }
    #[test]
    fn continuous_ramp_replaces_artificial_grid_and_retains_thin_line() {
        for line in [false, true] {
            let source = Raster::new(
                96,
                64,
                (0..96 * 64)
                    .map(|i| {
                        [0.3 + 0.4 * (i % 96) as f32 / 95.0
                            + if line && i % 96 == 48 { 0.005 } else { 0.0 };
                            3]
                    })
                    .collect(),
            );
            let labels = (0..96 * 64)
                .map(|i| {
                    if line && i % 96 == 48 {
                        24
                    } else {
                        ((i / 96 / 16) * 6 + (i % 96 / 16)) as u32
                    }
                })
                .collect();
            let mut segmentation = partition(&source, labels);
            let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
            assert_eq!(hints.iter().flatten().count(), 1);
            assert_eq!(segmentation.regions.len(), if line { 2 } else { 1 });
            if line {
                assert_ne!(
                    segmentation.labels[32 * 96 + 48],
                    segmentation.labels[32 * 96 + 47]
                );
            }
        }
    }
    #[test]
    fn material_transition_separates_coherent_fields() {
        let source = Raster::new(
            128,
            64,
            (0..128 * 64)
                .map(|i| [if i % 128 < 64 { 0.2 } else { 0.7 } + 0.1 * (i % 64) as f32 / 63.0; 3])
                .collect(),
        );
        let labels = (0..128 * 64)
            .map(|i| ((i / 128 / 16) * 8 + i % 128 / 16) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_eq!(hints.iter().flatten().count(), 2);
        assert_ne!(
            segmentation.labels[32 * 128 + 63],
            segmentation.labels[32 * 128 + 64]
        );
    }
    #[test]
    fn fitted_field_is_not_subdivided_into_long_axis_patches() {
        let source = Raster::new(
            256,
            192,
            (0..256 * 192)
                .map(|i| [0.2 + 0.6 * (i % 256) as f32 / 255.0; 3])
                .collect(),
        );
        let labels = (0..256 * 192)
            .map(|i| (i % 256 / 86).min(2) as u32)
            .collect();
        let segmentation = partition(&source, labels);
        let mut split = segmentation.clone();
        let parents = crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut split,
            &[false, true, false],
        );
        assert_eq!(parents.iter().filter(|&&i| i == 1).count(), 1);
        let mut legacy = segmentation;
        let parents = crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut legacy,
            &[],
        );
        assert!(parents.iter().filter(|&&i| i == 1).count() > 1);
    }
    #[test]
    fn colour_knots_follow_a_non_tenth_highlight() {
        let input = vec![
            ColorStop {
                offset: 0.0,
                color: [0.2; 3],
            },
            ColorStop {
                offset: 0.173,
                color: [0.9; 3],
            },
            ColorStop {
                offset: 1.0,
                color: [0.3; 3],
            },
        ];
        let parameters = (0..256).map(|i| i as f32 / 255.0).collect::<Vec<_>>();
        let profile = Raster::new(
            256,
            1,
            parameters.iter().map(|&t| interpolate(&input, t)).collect(),
        );
        let fitted = profile_stops(&profile, &(0..256).collect::<Vec<_>>(), &parameters, 3);
        assert!((fitted[1].offset - 0.173).abs() < 0.006);
        let error = parameters
            .iter()
            .zip(&profile.pixels)
            .map(|(&t, p)| (interpolate(&fitted, t)[0] - p[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 0.01, "{error}");
    }
}
