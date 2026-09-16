fn compact_connected_dense(values: &[u32], width: usize, height: usize) -> (Vec<u32>, usize) {
    let mut union = UnionFind::new(values.len());
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if x + 1 < width && values[index] == values[index + 1] {
                union.union(index, index + 1);
            }
            if y + 1 < height && values[index] == values[index + width] {
                union.union(index, index + width);
            }
        }
    }
    let mut pixel_roots = Vec::with_capacity(values.len());
    let mut components = HashMap::<usize, (u32, usize)>::new();
    for (index, &palette) in values.iter().enumerate() {
        let root = union.find(index);
        pixel_roots.push(root);
        components
            .entry(root)
            .and_modify(|component| component.1 = component.1.min(index))
            .or_insert((palette, index));
    }
    // scipy.ndimage.label is invoked once per palette in ascending palette
    // order.  Within one palette it numbers components by their first
    // row-major sample.  Preserve that exact ordering because later passes
    // use stable label order to resolve otherwise equal candidates.
    let mut ordered: Vec<(usize, u32, usize)> = components
        .into_iter()
        .map(|(root, (palette, first))| (root, palette, first))
        .collect();
    ordered.sort_by_key(|&(_, palette, first)| (palette, first));
    let root_labels: HashMap<usize, u32> = ordered
        .iter()
        .enumerate()
        .map(|(label, &(root, _, _))| (root, label as u32))
        .collect();
    let labels = pixel_roots
        .into_iter()
        .map(|root| root_labels[&root])
        .collect();
    (labels, ordered.len())
}

mod tests {

    #[test]
    fn neutral_outline_against_blue_paint_is_not_a_small_colour_island() {
        let image = image::load_from_memory(include_bytes!("../data/cliparts-woman-outline.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            image.width() as usize,
            image.height() as usize,
            image
                .pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let roles = crate::edge::classify(&source);
        let partition = segment_with_paint_owned_lines(
            &source,
            &roles,
            &Config::default(),
            &roles.dark_boundary,
        );
        for (x, y) in [(62, 20), (37, 25)] {
            let rgb = partition.canonical.get(x, y);
            assert!(
                rgb[2] < 0.30 && (rgb[2] - rgb[0]).abs() < 0.12,
                "neutral outline was absorbed into blue paint at {x},{y}: {rgb:?}"
            );
        }
    }
    use super::*;
    use crate::edge::classify;

    #[test]
    fn car_bumper_shadow_is_refined_without_requiring_a_neighbouring_tone_seam() {
        // Native underpaint/ownership crop at (104,800), keeping the original
        // alignment of the eight-pixel source-validation cells.
        let image = image::load_from_memory(include_bytes!("../data/car-bumper-shadow.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            208,
            80,
            image
                .pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let labels: Vec<_> = include_bytes!("../data/car-bumper-shadow.labels")
            .iter()
            .map(|&v| v as u32)
            .collect();
        let pixels: Vec<_> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label == 1).then_some(i))
            .collect();
        assert!(crate::gradient::has_unresolved_local_shading(
            &source, &pixels
        ));
        let mut segmentation = Segmentation {
            width: 208,
            height: 80,
            regions: region_stats(&source, &labels, 2),
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; 208 * 80],
            canonical: source.clone(),
            summary: SegmentationSummary::default(),
        };
        split_adaptive_paint_patches(&source, &source, &mut segmentation);
        let owners: HashSet<_> = pixels.iter().map(|&i| segmentation.labels[i]).collect();
        assert!((2..=8).contains(&owners.len()), "shadow owners: {owners:?}");
        assert!(owners
            .iter()
            .all(|&owner| segmentation.paint_keys[owner as usize] == 1));
    }

    #[test]
    fn ambiguous_headlight_coverage_does_not_become_dark_ink() {
        // Native car headlight samples: both mixing spaces explain these
        // colours, but a minute residual difference used to pick black.
        for (value, light, dark) in [
            (
                [0.45277935, 0.5408499, 0.58111066],
                [0.65986395, 0.74419475, 0.7910196],
                [0.03210497, 0.032667756, 0.030182343],
            ),
            (
                [83.0 / 255.0, 98.0 / 255.0, 109.0 / 255.0],
                [0.5024745, 0.5916716, 0.6489957],
                [0.049291536, 0.047945917, 0.04566788],
            ),
        ] {
            assert!(partition_coverage_alpha(value, light, dark) > 0.5);
            assert!(partition_coverage_alpha(value, dark, light) < 0.5);
            assert!(partition_coverage_alpha(dark, light, dark) < 0.5);
            assert!(partition_coverage_alpha(light, light, dark) > 0.5);
        }
    }

    #[test]
    fn orphaned_coverage_does_not_inherit_a_remote_red_palette() {
        // A glass/trim coverage pixel was left alone after its former palette
        // neighbour moved. Masked Paint fitting will use this canonical colour.
        let mut source = Raster::blank(8, 8, [0.53, 0.73, 0.81]);
        source.pixels[27] = [76.0 / 255.0, 82.0 / 255.0, 91.0 / 255.0];
        let mut labels = vec![0; 64];
        labels[27] = 1;
        let mut quantized = Raster::blank(8, 8, [0.52, 0.72, 0.80]);
        quantized.pixels[27] = [188.0 / 255.0, 37.0 / 255.0, 36.0 / 255.0];
        let canonical =
            partition_canonical(&quantized, &labels, &region_stats(&source, &labels, 2));
        assert!(
            delta_e_ok(
                rgb_to_oklab(canonical.pixels[27]),
                rgb_to_oklab(source.pixels[27])
            ) < 0.01
        );
        assert!(
            delta_e_ok(
                rgb_to_oklab(canonical.pixels[0]),
                rgb_to_oklab(quantized.pixels[0])
            ) < 0.01
        );
    }

    #[test]
    fn source_highlight_is_not_quantized_into_its_incident_paint() {
        let mut colours = vec![
            Oklab {
                l: 88.0,
                a: 0.0,
                b: 0.0
            };
            100
        ];
        colours.extend(vec![
            Oklab {
                l: 94.0,
                a: 0.0,
                b: 0.0
            };
            8
        ]);
        let mut protected = vec![false; 100];
        protected.extend(vec![true; 8]);
        let (labels, palette, _) = build_palette(&colours, &protected, &Config::default());
        assert_ne!(labels[0], labels[100]);
        assert!(palette[labels[100] as usize].l > 93.0);
    }

    #[test]
    fn small_component_merge_keeps_a_shaded_bridge_and_an_authored_gap() {
        let config = Config::default();
        for reversed_palette in [false, true] {
            let mut palette = vec![
                Oklab {
                    l: 20.0,
                    a: 0.0,
                    b: 0.0,
                },
                Oklab {
                    l: 70.0,
                    a: 0.0,
                    b: 0.0,
                },
                Oklab {
                    l: 90.0,
                    a: 0.0,
                    b: 0.0,
                },
            ];
            for gap in [false, true] {
                let mut labels = vec![2; 49];
                labels[23] = 0;
                labels[25] = 0;
                if !gap {
                    labels[24] = 1;
                }
                if reversed_palette {
                    for label in &mut labels {
                        *label = 2 - *label;
                    }
                    palette.reverse();
                }
                let source = labels
                    .iter()
                    .map(|&l| palette[l as usize])
                    .collect::<Vec<_>>();
                merge_small_components(
                    &mut labels,
                    &palette,
                    &source,
                    7,
                    7,
                    &[8; 49],
                    8,
                    &config,
                    &[],
                    false,
                );
                let middle = palette[labels[24] as usize].l;
                if gap {
                    assert_eq!(middle, 90.0, "authored gap filled");
                } else {
                    assert!(middle < 80.0, "shaded bridge removed: {middle}");
                }
                if reversed_palette {
                    palette.reverse();
                }
            }
        }
    }

    #[test]
    fn tonal_line_support_requires_matching_incident_paints() {
        for bright in [false, true] {
            for different_paints in [false, true] {
                let source = (0..81)
                    .map(|i| {
                        let x = i % 9;
                        Oklab {
                            l: if x == 4 {
                                if bright {
                                    80.0
                                } else {
                                    20.0
                                }
                            } else if bright {
                                20.0
                            } else {
                                80.0
                            },
                            a: if different_paints && x > 4 { 30.0 } else { 0.0 },
                            b: 0.0,
                        }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    source_supported_tonal_line(&source, 40, 9, 9),
                    !different_paints
                );
            }
        }
        let shaded = (0..81)
            .map(|i| Oklab {
                l: if i % 9 == 4 {
                    30.0
                } else if i % 9 < 4 {
                    85.0
                } else {
                    90.0
                },
                a: 0.0,
                b: 0.0,
            })
            .collect::<Vec<_>>();
        assert!(source_supported_tonal_line(&shaded, 40, 9, 9));
    }

    #[test]
    fn tonal_connectivity_crosses_palette_shades_without_protecting_edges() {
        for bright in [false, true] {
            let palette = [25.0, 45.0, 90.0].map(|l| Oklab {
                l: if bright { 100.0 - l } else { l },
                a: 0.0,
                b: 0.0,
            });
            for vertical in [false, true] {
                let mut labels = vec![2; 25];
                let step = if vertical { 5 } else { 1 };
                labels[12] = 1;
                labels[12 - step] = 0;
                labels[12 + step] = 1;
                assert!(splits_tonal_connection(
                    &labels, &palette, 12, 5, 5, palette[2]
                ));
                // Recolouring into the darker incident shade keeps the line.
                assert!(!splits_tonal_connection(
                    &labels, &palette, 12, 5, 5, palette[0]
                ));
                labels[12 + step] = 2;
                assert!(!splits_tonal_connection(
                    &labels, &palette, 12, 5, 5, palette[2]
                ));
            }
            let mut labels = vec![2; 25];
            for y in 1..4 {
                for x in 1..4 {
                    labels[y * 5 + x] = 0;
                }
            }
            labels[12] = 1;
            assert!(!splits_tonal_connection(
                &labels, &palette, 12, 5, 5, palette[2]
            ));
        }
    }

    #[test]
    fn tonal_connectivity_keeps_the_weak_side_of_a_shaded_line() {
        for bright in [false, true] {
            let palette = [20.0, 30.0, 70.0, 90.0].map(|l| Oklab {
                l: if bright { 100.0 - l } else { l },
                a: 0.0,
                b: 0.0,
            });
            let mut labels = vec![3; 25];
            labels[11] = 0;
            labels[12] = 1;
            labels[13] = 2;
            assert!(splits_tonal_connection(
                &labels, &palette, 12, 5, 5, palette[3]
            ));
        }
    }

    #[test]
    fn indexed_component_seeds_preserve_dense_reassignment_order() {
        let config = Config::default();
        for seed in 0..24_usize {
            let (width, height) = (17, 13);
            let palette: Vec<Oklab> = (0..5)
                .map(|i| Oklab {
                    l: 45.0 + i as f32 * 2.0,
                    a: 2.0,
                    b: 1.0,
                })
                .collect();
            let original: Vec<u32> = (0..width * height)
                .map(|i| ((i * 17 + i * i * (seed + 1) + seed * 31) % 5) as u32)
                .collect();
            let source: Vec<Oklab> = original.iter().map(|&i| palette[i as usize]).collect();
            let mut dense = original.clone();
            let mut indexed = original;
            let local_area = vec![8; width * height];
            let a = merge_small_components(
                &mut dense,
                &palette,
                &source,
                width,
                height,
                &local_area,
                12,
                &config,
                &[],
                true,
            );
            let b = merge_small_components(
                &mut indexed,
                &palette,
                &source,
                width,
                height,
                &local_area,
                12,
                &config,
                &[],
                false,
            );
            assert_eq!(a, b, "seed={seed}");
            assert_eq!(dense, indexed, "seed={seed}");
        }
    }

    #[test]
    fn exact_values_split_final_owner_without_changing_its_paint_key() {
        let image = Raster::new(
            3,
            1,
            vec![[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.8, 0.7, 0.6]],
        );
        let mut segmentation = Segmentation {
            width: 3,
            height: 1,
            labels: vec![0, 0, 1],
            paint_keys: vec![7, 9],
            paint_samples: vec![true; 3],
            canonical: image.clone(),
            regions: region_stats(&image, &[0, 0, 1], 2),
            summary: SegmentationSummary {
                merged_regions: 2,
                ..SegmentationSummary::default()
            },
        };

        let (parents, values) = split_partition_by_values(&image, &mut segmentation, &[0, 1, 1]);

        assert_eq!(segmentation.labels, vec![0, 1, 2]);
        assert_eq!(segmentation.paint_keys, vec![7, 7, 9]);
        assert_eq!(parents, vec![0, 0, 1]);
        assert_eq!(values, vec![0, 1, 1]);
        assert_eq!(segmentation.summary.merged_regions, 3);
    }

    #[test]
    fn hierarchical_connected_components_match_dense_label_order() {
        for mut code in 0_u32..3_u32.pow(9) {
            let values: Vec<u32> = (0..9)
                .map(|_| {
                    let value = code % 3;
                    code /= 3;
                    value
                })
                .collect();
            assert_eq!(
                compact_connected(&values, 3, 3),
                compact_connected_dense(&values, 3, 3),
                "values={values:?}",
            );
        }
    }

    #[test]
    fn boundary_keeps_two_faces_separate() {
        let mut image = Raster::blank(24, 12, [0.1, 0.1, 0.1]);
        for y in 0..12 {
            for x in 12..24 {
                image.pixels[y * 24 + x] = [0.9, 0.1, 0.1];
            }
        }
        let roles = classify(&image);
        let result = segment(
            &image,
            &roles,
            &Config {
                segmentation_min_size: 2,
                ..Config::default()
            },
        );
        assert_ne!(result.labels[5 * 24 + 5], result.labels[5 * 24 + 18]);
    }

    #[test]
    fn spatially_separate_equal_colours_remain_separate_regions() {
        let mut image = Raster::blank(16, 8, [1.0, 1.0, 1.0]);
        image.pixels[2 * 16 + 2] = [0.0, 0.0, 0.0];
        image.pixels[2 * 16 + 13] = [0.0, 0.0, 0.0];
        let roles = classify(&image);
        let result = segment(
            &image,
            &roles,
            &Config {
                segmentation_min_size: 1,
                ..Config::default()
            },
        );
        assert_ne!(result.labels[2 * 16 + 2], result.labels[2 * 16 + 13]);
    }

    #[test]
    fn antialias_band_is_returned_to_its_two_parent_faces() {
        let width = 10;
        let height = 7;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                match x {
                    0..=2 => {}
                    3 => {
                        image.pixels[index] = [0.25; 3];
                        labels[index] = 1;
                    }
                    4 => {
                        image.pixels[index] = [0.75; 3];
                        labels[index] = 1;
                    }
                    _ => {
                        image.pixels[index] = [1.0; 3];
                        labels[index] = 2;
                    }
                }
            }
        }
        let roles = classify(&image);
        let correction = correct_antialias_partition(&image, &labels, 3, &roles);
        assert_eq!(correction.split_regions, 1);
        assert_ne!(correction.labels[3], correction.labels[4]);
        assert!(correction.paint_samples[3..=4].iter().all(|&value| !value));
    }

    #[test]
    fn expanded_parent_recovery_does_not_promote_distant_third_face() {
        let assignment_with_third_face_at = |third_y: usize| {
            let width = 18;
            let height = 12;
            let colours = [
                [0.91, 0.90, 0.89],
                [0.96, 0.34, 0.30],
                // Coverage between the first two faces, not a third ink
                // extremum (which must be retained regardless of topology).
                [0.935, 0.62, 0.595],
                [0.58, 0.05, 0.04],
            ];
            let mut labels = vec![0_u32; width * height];
            for y in 6..third_y {
                for x in 0..width {
                    labels[y * width + x] = 1;
                }
            }
            for y in third_y..height {
                for x in 0..width {
                    labels[y * width + x] = 3;
                }
            }
            let mut component = Vec::new();
            for y in 4..=5 {
                for x in 4..14 {
                    let index = y * width + x;
                    labels[index] = 2;
                    component.push(index);
                }
            }
            let image = Raster::new(
                width,
                height,
                labels
                    .iter()
                    .map(|&label| colours[label as usize])
                    .collect(),
            );
            let mut roles = classify(&image);
            roles.visible_ridge_centres.fill(false);
            let parent_lab = colours.map(rgb_to_oklab);
            boundary_sleeve_assignment(
                &image,
                &component,
                &labels,
                2,
                &[true, true, false, true],
                &parent_lab,
                &roles,
                width,
                height,
            )
        };

        let distant = assignment_with_third_face_at(9).unwrap();
        assert_eq!((distant.0, distant.1), (0, 1));
        assert!(assignment_with_third_face_at(8).is_none());
    }

    #[test]
    fn compact_coverage_next_to_a_fragmented_rim_is_not_a_third_face() {
        for protected in [false, true] {
            let (width, height) = (24, 160);
            let mut image =
                Raster::blank(width, height, [134.0 / 255.0, 185.0 / 255.0, 207.0 / 255.0]);
            let mut labels = vec![0_u32; width * height];
            for y in 0..height {
                for x in 0..6 {
                    let i = y * width + x;
                    labels[i] = 1;
                    image.pixels[i] = [0.75, 0.1, 0.1];
                }
                for x in 6..8 {
                    let i = y * width + x;
                    labels[i] = 2 + y as u32;
                    image.pixels[i] = [37.0 / 255.0, 23.0 / 255.0, 26.0 / 255.0];
                }
            }
            let index = 80 * width + 8;
            labels[index] = 162;
            image.pixels[index] = [76.0 / 255.0, 82.0 / 255.0, 91.0 / 255.0];
            let mut roles = classify(&image);
            for y in 0..height {
                for x in 6..8 {
                    roles.dark_boundary[y * width + x] = true;
                }
            }
            roles.dark_boundary[index] = true;
            roles.visible_ridge_centres[index] = protected;
            let result = correct_antialias_partition(&image, &labels, 163, &roles);
            if protected {
                assert_ne!(result.labels[index], result.labels[index - 1]);
                assert_ne!(result.labels[index], result.labels[index + 1]);
            } else {
                assert_eq!(result.labels[index], result.labels[index - 1]);
                assert!(!result.paint_samples[index]);
            }
        }
    }

    #[test]
    fn isolated_intermediate_antialias_pixel_is_absorbed_by_a_parent_face() {
        let width = 7;
        let height = 7;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = [1.0; 3];
                labels[index] = 2;
            }
        }
        let intermediate = 3 * width + 3;
        image.pixels[intermediate] = [0.47; 3];
        labels[intermediate] = 1;

        let roles = classify(&image);
        let correction = correct_antialias_partition(&image, &labels, 3, &roles);

        assert_eq!(
            correction.labels[intermediate],
            correction.labels[3 * width + 2]
        );
        assert!(!correction.paint_samples[intermediate]);
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn long_ink_edge_shoulder_and_undershoot_share_the_durable_outline() {
        for protected in [false, true] {
            let width = 64;
            let height = 64;
            let mut image = Raster::blank(width, height, [0.015; 3]);
            let mut labels = vec![0; width * height];
            for y in 0..height {
                for x in 35..width {
                    image.pixels[y * width + x] = [0.98; 3];
                    labels[y * width + x] = 1;
                }
            }
            for y in 16..48 {
                for x in 31..35 {
                    let i = y * width + x;
                    labels[i] = if x < 34 { 2 } else { 3 };
                    image.pixels[i] = if x < 34 { [0.03; 3] } else { [0.0; 3] };
                }
            }
            let mut roles = classify(&image);
            for y in 16..48 {
                for x in 31..35 {
                    let i = y * width + x;
                    roles.dark_boundary[i] = true;
                    roles.visible_ridge_centres[i] = protected;
                }
            }
            let result = correct_antialias_partition(&image, &labels, 4, &roles);
            for y in 16..48 {
                for x in 31..35 {
                    let i = y * width + x;
                    if protected {
                        assert_eq!(result.labels[i], labels[i]);
                    } else {
                        assert_eq!(result.labels[i], 0);
                        assert!(!result.paint_samples[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn intermediate_sleeve_beside_split_ink_is_coverage() {
        for (third, protected, expected_coverage) in [
            ([0.018, 0.023, 0.028], false, true),
            ([0.0001, 0.0002, 0.0003], false, true),
            ([0.8, 0.1, 0.1], false, false),
            ([0.018, 0.023, 0.028], true, false),
        ] {
            let width = 15;
            let height = 17;
            let mut image = Raster::blank(width, height, [0.02, 0.025, 0.03]);
            let mut labels = vec![0_u32; width * height];
            for y in 0..height {
                for x in 0..width {
                    let i = y * width + x;
                    if x >= 7 {
                        image.pixels[i] = [0.98, 0.96, 0.92];
                        labels[i] = 2;
                    } else if y >= 10 {
                        image.pixels[i] = third;
                        labels[i] = 3;
                    }
                }
            }
            for y in 4..13 {
                let i = y * width + 7;
                image.pixels[i] = [0.40, 0.40, 0.38];
                labels[i] = 1;
            }
            let mut roles = classify(&image);
            for y in 4..13 {
                let i = y * width + 7;
                roles.dark_boundary[i] = true;
                roles.visible_ridge_centres[i] = protected;
            }
            let correction = correct_antialias_partition(&image, &labels, 4, &roles);
            for y in 4..13 {
                let i = y * width + 7;
                assert_eq!(!correction.paint_samples[i], expected_coverage);
                if expected_coverage {
                    assert!(
                        correction.labels[i] == correction.labels[4 * width + 6]
                            || correction.labels[i] == correction.labels[i + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn short_and_dark_tailed_mixture_runs_are_not_independent_paint() {
        for length in [1, 4, 9] {
            let width = 17;
            let height = 17;
            let mut image = Raster::blank(width, height, [0.01; 3]);
            let mut labels = vec![0; width * height];
            for y in 0..height {
                for x in 8..width {
                    image.pixels[y * width + x] = [0.98; 3];
                    labels[y * width + x] = 2;
                }
            }
            for offset in 0..length {
                let i = (4 + offset) * width + 8;
                image.pixels[i] = [if length == 9 && offset >= 7 {
                    0.15
                } else {
                    0.45
                }; 3];
                labels[i] = 1;
            }
            let mut roles = classify(&image);
            for offset in 0..length {
                let i = (4 + offset) * width + 8;
                roles.dark_boundary[i] = true;
                roles.visible_ridge_centres[i] = false;
            }
            let correction = correct_antialias_partition(&image, &labels, 3, &roles);
            for offset in 0..length {
                let i = (4 + offset) * width + 8;
                assert!(
                    !correction.paint_samples[i],
                    "length={length}, offset={offset}"
                );
                assert_eq!(correction.labels[i], correction.labels[i - 1]);
            }
        }
    }

    #[test]
    fn isolated_colour_not_explained_by_its_neighbours_remains_paint() {
        let width = 7;
        let height = 7;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = [1.0; 3];
                labels[index] = 2;
            }
        }
        let distinct = 3 * width + 3;
        image.pixels[distinct] = [1.0, 0.0, 0.0];
        labels[distinct] = 1;

        let roles = classify(&image);
        let correction = correct_antialias_partition(&image, &labels, 3, &roles);

        assert_ne!(
            correction.labels[distinct],
            correction.labels[3 * width + 2]
        );
        assert_ne!(
            correction.labels[distinct],
            correction.labels[3 * width + 4]
        );
        assert!(correction.paint_samples[distinct]);
        assert_eq!(correction.split_regions, 0);
    }

    #[test]
    fn sleeve_coverage_is_independent_of_grid_phase_and_parent_label_order() {
        let width = 48;
        let height = 48;
        let foreground = [0.85, 0.15, 0.14];
        let background = [0.92; 3];
        for slope in [0.12_f32, 0.55, 1.2] {
            for phase in [0.1_f32, 0.45, 0.8] {
                for reversed in [false, true] {
                    let foreground_label = if reversed { 2 } else { 0 };
                    let background_label = 2 - foreground_label;
                    let mut image = Raster::blank(width, height, background);
                    let mut labels = vec![background_label; width * height];
                    let mut coverage = vec![0.0; labels.len()];
                    for y in 0..height {
                        for x in 0..width {
                            let index = y * width + x;
                            let amount =
                                ((16.0 + slope * (y as f32 - 24.0) + phase - x as f32 - 0.5)
                                    / (1.0 + slope * slope).sqrt()
                                    + 0.5)
                                    .clamp(0.0, 1.0);
                            coverage[index] = amount;
                            image.pixels[index] = std::array::from_fn(|c| {
                                amount * foreground[c] + (1.0 - amount) * background[c]
                            });
                            labels[index] = if amount == 1.0 {
                                foreground_label
                            } else if amount == 0.0 {
                                background_label
                            } else {
                                1
                            };
                        }
                    }
                    let mut roles = classify(&image);
                    roles.visible_ridge_centres.fill(false);
                    roles.dark_boundary.fill(false);
                    let corrected = correct_antialias_partition(&image, &labels, 3, &roles);
                    let first = corrected.labels[24 * width];
                    let second = corrected.labels[24 * width + width - 1];
                    assert_ne!(first, second);
                    for y in 4..height - 4 {
                        for x in 0..width {
                            let index = y * width + x;
                            if labels[index] == 1 && (coverage[index] - 0.5).abs() > 0.02 {
                                assert_eq!(
                                    corrected.labels[index],
                                    if coverage[index] > 0.5 { first } else { second },
                                    "slope={slope} phase={phase} reversed={reversed} at {x},{y}"
                                );
                                assert!(!corrected.paint_samples[index]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn elongated_monotonic_boundary_band_is_absorbed() {
        let width = 7;
        let height = 9;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = [1.0; 3];
                labels[index] = 2;
            }
        }
        for y in 2..=6 {
            let index = y * width + 3;
            image.pixels[index] = [0.47; 3];
            labels[index] = 1;
        }

        let roles = classify(&image);
        let correction = correct_antialias_partition(&image, &labels, 3, &roles);

        for y in 2..=6 {
            let index = y * width + 3;
            assert_ne!(correction.labels[index], 1);
            assert!(!correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn source_supported_medial_ridge_is_not_absorbed_as_a_sleeve() {
        let width = 7;
        let height = 9;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = [1.0; 3];
                labels[index] = 2;
            }
        }
        for y in 2..=6 {
            let index = y * width + 3;
            image.pixels[index] = [0.47; 3];
            labels[index] = 1;
        }
        let mut roles = classify(&image);
        for y in 2..=6 {
            roles.visible_ridge_centres[y * width + 3] = true;
        }

        let correction = correct_antialias_partition(&image, &labels, 3, &roles);
        let middle = 4 * width + 3;

        assert_eq!(correction.labels[middle], 1);
        assert!(correction.paint_samples[middle]);
        assert_eq!(correction.split_regions, 0);
    }

    #[test]
    fn fragmented_dark_outline_is_not_absorbed_as_a_boundary_sleeve() {
        let width = 7;
        let height = 9;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 6;
            }
        }
        for (offset, y) in (2..=6).enumerate() {
            let index = y * width + 3;
            image.pixels[index] = [0.03; 3];
            labels[index] = 1 + offset as u32;
        }
        let mut roles = classify(&image);
        assert!((2..=6).any(|y| roles.dark_boundary[y * width + 3]));
        // The colour-extremum guard must preserve the authored outline even
        // when the graph classifier has a local gap.
        roles.dark_boundary.fill(false);

        let correction = correct_antialias_partition(&image, &labels, 7, &roles);

        for y in 2..=6 {
            let index = y * width + 3;
            assert!((1..=5).contains(&correction.labels[index]));
            assert!(correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 0);
    }

    #[test]
    fn fragmented_dark_boundary_labels_share_one_paint_owner() {
        let width = 7;
        let height = 9;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 6;
            }
        }
        for (offset, y) in (2..=6).enumerate() {
            let index = y * width + 3;
            image.pixels[index] = [0.03; 3];
            labels[index] = 1 + offset as u32;
        }
        let mut roles = classify(&image);
        for y in 2..=6 {
            let index = y * width + 3;
            roles.dark_boundary[index] = true;
            roles.visible_ridge_centres[index] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 7, &roles);
        let outline = correction.labels[2 * width + 3];
        assert_ne!(outline, correction.labels[2 * width + 2]);
        assert_ne!(outline, correction.labels[2 * width + 4]);
        for y in 2..=6 {
            let index = y * width + 3;
            assert_eq!(correction.labels[index], outline);
            assert!(correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 4);
    }

    #[test]
    fn long_fragmented_dark_boundary_joins_a_durable_black_endpoint() {
        let width = 106;
        let height = 9;
        let light = [0.85, 0.90, 0.92];
        let dark = [0.03; 3];
        let mut image = Raster::blank(width, height, light);
        let mut labels = vec![0_u32; width * height];
        for (offset, x) in (2..=97).enumerate() {
            let index = 4 * width + x;
            image.pixels[index] = dark;
            labels[index] = 1 + offset as u32;
        }
        for y in 2..=6 {
            for x in 98..=102 {
                let index = y * width + x;
                image.pixels[index] = dark;
                labels[index] = 97;
            }
        }
        let mut roles = classify(&image);
        for x in 2..=97 {
            let index = 4 * width + x;
            roles.dark_boundary[index] = true;
            roles.visible_ridge_centres[index] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 98, &roles);
        let durable = correction.labels[4 * width + 98];
        for x in 2..=97 {
            assert_eq!(correction.labels[4 * width + x], durable);
        }
        assert_eq!(correction.split_regions, 96);
    }

    #[test]
    fn dark_outline_tail_keeps_coloured_ink_but_not_light_ringing_prefix() {
        let width = 7;
        let height = 31;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 4;
            }
        }
        for y in 2..=13 {
            let index = y * width + 3;
            image.pixels[index] = [0.86, 0.98, 1.0];
            labels[index] = 1;
        }
        for y in 14..=25 {
            let index = y * width + 3;
            image.pixels[index] = [0.50, 0.20, 0.20];
            labels[index] = 2;
        }
        for y in 26..=28 {
            let index = y * width + 3;
            image.pixels[index] = [0.03; 3];
            labels[index] = 3;
        }

        let mut roles = classify(&image);
        roles.dark_boundary.fill(false);
        for y in 2..=28 {
            roles.visible_ridge_centres[y * width + 3] = false;
        }
        for y in 26..=28 {
            roles.dark_boundary[y * width + 3] = true;
        }

        let correction = correct_antialias_partition(&image, &labels, 5, &roles);

        for y in 2..=13 {
            let index = y * width + 3;
            assert_eq!(correction.labels[index], correction.labels[y * width + 1]);
            assert!(!correction.paint_samples[index]);
        }
        for y in 14..=25 {
            let index = y * width + 3;
            assert_ne!(correction.labels[index], correction.labels[y * width + 1]);
            assert_ne!(correction.labels[index], correction.labels[y * width + 5]);
            assert!(correction.paint_samples[index]);
        }
        let outline_label = correction.labels[26 * width + 3];
        assert_ne!(outline_label, correction.labels[26 * width + 2]);
        assert_ne!(outline_label, correction.labels[26 * width + 4]);
        for y in 26..=28 {
            let index = y * width + 3;
            assert_eq!(correction.labels[index], outline_label);
            assert!(correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn adjacent_single_pixel_quantisation_labels_form_one_sleeve() {
        let width = 7;
        let height = 9;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 6;
            }
        }
        for (offset, y) in (2..=6).enumerate() {
            let index = y * width + 3;
            image.pixels[index] = [0.86, 0.98, 1.0];
            labels[index] = 1 + offset as u32;
        }
        let mut roles = classify(&image);
        for y in 2..=6 {
            roles.visible_ridge_centres[y * width + 3] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 7, &roles);

        for y in 2..=6 {
            let index = y * width + 3;
            assert!(matches!(correction.labels[index], 0 | 6));
            assert!(!correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 5);
    }

    #[test]
    fn repeated_four_pixel_sleeves_bridge_one_parent_pixel_gap() {
        let width = 26;
        let height = 12;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];

        for x in 0..width {
            let boundary_y = if x < 2 {
                8
            } else if x < 22 {
                8 - (x - 2) / 5
            } else {
                4
            };
            for y in boundary_y..height {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 5;
            }
        }
        for group in 0..4 {
            let y = 8 - group;
            let start_x = 2 + group * 5;
            for x in start_x..start_x + 4 {
                let index = y * width + x;
                image.pixels[index] = [0.86, 0.98, 1.0];
                labels[index] = 1 + group as u32;
            }
        }

        let mut roles = classify(&image);
        for group in 0..4 {
            let y = 8 - group;
            let start_x = 2 + group * 5;
            for x in start_x..start_x + 4 {
                roles.visible_ridge_centres[y * width + x] = false;
            }
        }

        let correction = correct_antialias_partition(&image, &labels, 6, &roles);

        for group in 0..4 {
            let y = 8 - group;
            let start_x = 2 + group * 5;
            for x in start_x..start_x + 4 {
                let index = y * width + x;
                assert!(matches!(correction.labels[index], 0 | 5));
                assert!(!correction.paint_samples[index]);
            }
        }
        assert_eq!(correction.split_regions, 4);
    }

    #[test]
    fn repeated_subpixel_sleeves_choose_source_matched_parent() {
        let width = 28;
        let height = 12;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 6..height {
            for x in 0..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 5;
            }
        }
        let runs = [(2, 1), (8, 2), (14, 3), (20, 4)];
        for (group, &(start_x, length)) in runs.iter().enumerate() {
            for x in start_x..start_x + length {
                let index = 6 * width + x;
                image.pixels[index] = [0.86, 0.98, 1.0];
                labels[index] = 1 + group as u32;
            }
        }

        let mut roles = classify(&image);
        for &(start_x, length) in &runs {
            for x in start_x..start_x + length {
                roles.visible_ridge_centres[6 * width + x] = false;
            }
            roles.dark_boundary[6 * width + start_x] = true;
        }

        let correction = correct_antialias_partition(&image, &labels, 6, &roles);

        for &(start_x, length) in &runs {
            for x in start_x..start_x + length {
                let index = 6 * width + x;
                assert_eq!(correction.labels[index], 0);
                assert!(!correction.paint_samples[index]);
            }
        }
        assert_eq!(correction.split_regions, 4);
    }

    #[test]
    fn thin_same_material_microface_returns_to_durable_owner() {
        let width = 9;
        let height = 9;
        let durable = [0.92, 0.24, 0.22];
        let mut image = Raster::blank(width, height, durable);
        let mut labels = vec![0_u32; width * height];
        for x in 2..=5 {
            let index = 4 * width + x;
            image.pixels[index] = [0.96, 0.28, 0.26];
            labels[index] = 1;
        }
        let mut roles = classify(&image);
        for x in 2..=5 {
            roles.visible_ridge_centres[4 * width + x] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 2, &roles);

        for x in 2..=5 {
            let index = 4 * width + x;
            assert_eq!(correction.labels[index], 0);
            assert!(!correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn substantial_chroma_shoulder_returns_to_lightness_matched_owner() {
        let width = 12;
        let height = 14;
        // This pair reproduces a saturated red transition where the native
        // samples differ little in lightness but slightly exceed the normal
        // same-material chroma tolerance.
        let durable = [0.778_05, 0.352_321, 0.363_366];
        let shoulder = [0.916_967, 0.347_291, 0.317_088];
        let error = delta_e_ok(rgb_to_oklab(durable), rgb_to_oklab(shoulder));
        assert!((6.2..=7.5).contains(&error));

        let mut image = Raster::blank(width, height, durable);
        let mut labels = vec![0_u32; width * height];
        for y in 3..=10 {
            for x in 5..=6 {
                let index = y * width + x;
                image.pixels[index] = shoulder;
                labels[index] = 1;
            }
        }
        let mut roles = classify(&image);
        for y in 3..=10 {
            for x in 5..=6 {
                roles.visible_ridge_centres[y * width + x] = false;
            }
        }

        let correction = correct_antialias_partition(&image, &labels, 2, &roles);

        for y in 3..=10 {
            for x in 5..=6 {
                let index = y * width + x;
                assert_eq!(correction.labels[index], 0);
                assert!(!correction.paint_samples[index]);
            }
        }
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn short_chroma_mark_keeps_independent_paint() {
        let width = 11;
        let height = 11;
        let durable = [0.778_05, 0.352_321, 0.363_366];
        let shoulder = [0.916_967, 0.347_291, 0.317_088];
        let mut image = Raster::blank(width, height, durable);
        let mut labels = vec![0_u32; width * height];
        for x in 2..=8 {
            let index = 5 * width + x;
            image.pixels[index] = shoulder;
            labels[index] = 1;
        }
        let mut roles = classify(&image);
        for x in 2..=8 {
            roles.visible_ridge_centres[5 * width + x] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 2, &roles);

        for x in 2..=8 {
            let index = 5 * width + x;
            assert_eq!(correction.labels[index], 1);
            assert!(correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 0);
    }

    #[test]
    fn ringing_shoulder_beside_fragmented_dark_outline_returns_to_face() {
        let width = 10;
        let height = 14;
        let face = [0.450_583, 0.340_051, 0.353_434];
        let shoulder = [0.510_028, 0.431_782, 0.453_044];
        let outline = [0.03; 3];
        let error = delta_e_ok(rgb_to_oklab(face), rgb_to_oklab(shoulder));
        assert!(error > 7.2 && error <= 12.0);

        let mut image = Raster::blank(width, height, face);
        let mut labels = vec![3_u32; width * height];
        for y in 1..=12 {
            let index = y * width;
            image.pixels[index] = outline;
            labels[index] = if y <= 6 { 1 } else { 2 };
        }
        for y in 3..=9 {
            let index = y * width + 1;
            image.pixels[index] = shoulder;
            labels[index] = 4;
        }
        image.pixels[6 * width + 2] = shoulder;
        labels[6 * width + 2] = 4;
        let mut roles = classify(&image);
        for y in 1..=12 {
            let index = y * width;
            roles.dark_boundary[index] = true;
            roles.visible_ridge_centres[index] = false;
        }
        for y in 3..=9 {
            let index = y * width + 1;
            roles.dark_boundary[index] = y == 6;
            roles.visible_ridge_centres[index] = false;
        }
        roles.visible_ridge_centres[6 * width + 2] = false;

        let correction = correct_antialias_partition(&image, &labels, 5, &roles);

        for y in 3..=9 {
            let index = y * width + 1;
            let expected = if y == 6 { index - 1 } else { index + 1 };
            assert_eq!(correction.labels[index], correction.labels[expected]);
            assert!(!correction.paint_samples[index]);
        }
        assert_eq!(
            correction.labels[6 * width + 2],
            correction.labels[6 * width + 3]
        );
        assert!(!correction.paint_samples[6 * width + 2]);
    }

    #[test]
    fn elongated_ringing_sleeve_is_returned_to_parent_faces() {
        let width = 7;
        let height = 9;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 2;
            }
        }
        for y in 2..=6 {
            let index = y * width + 3;
            // A sharpened-edge overshoot is deliberately outside the convex
            // colour segment between the two durable faces.
            image.pixels[index] = [0.86, 0.98, 1.0];
            labels[index] = 1;
        }

        let mut roles = classify(&image);
        for y in 2..=6 {
            let index = y * width + 3;
            roles.visible_ridge_centres[index] = false;
        }
        let correction = correct_antialias_partition(&image, &labels, 3, &roles);

        for y in 2..=6 {
            let index = y * width + 3;
            assert_ne!(correction.labels[index], 1);
            assert!(!correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 1);
    }

    #[test]
    fn bright_highlight_is_not_coverage_between_two_darker_faces() {
        for length in [1, 5] {
            let width = 7;
            let height = 9;
            let mut image = Raster::blank(width, height, [1.0, 0.20, 0.0]);
            let mut labels = vec![0_u32; width * height];
            for y in 0..height {
                for x in 3..width {
                    let index = y * width + x;
                    image.pixels[index] = [1.0, 0.38, 0.11];
                    labels[index] = 2;
                }
            }
            for y in 2..2 + length {
                image.pixels[y * width + 3] = [0.82, 1.0, 0.77];
                labels[y * width + 3] = 1;
            }
            let mut roles = classify(&image);
            roles.visible_ridge_centres.fill(false);
            let correction = correct_antialias_partition(&image, &labels, 3, &roles);
            for y in 2..2 + length {
                let index = y * width + 3;
                assert_ne!(correction.labels[index], correction.labels[y * width + 1]);
                assert_ne!(correction.labels[index], correction.labels[y * width + 5]);
                assert!(correction.paint_samples[index]);
            }
        }
    }

    #[test]
    fn coloured_ink_extremum_is_not_erased_when_detector_has_a_gap() {
        let width = 7;
        let height = 9;
        let first = [0.78, 0.91, 0.95];
        let second = [0.90, 0.22, 0.21];
        let mut image = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 3..width {
                let index = y * width + x;
                image.pixels[index] = second;
                labels[index] = 2;
            }
        }
        for y in 2..=6 {
            let index = y * width + 3;
            image.pixels[index] = [0.50, 0.20, 0.20];
            labels[index] = 1;
        }
        let mut roles = classify(&image);
        for y in 2..=6 {
            let index = y * width + 3;
            roles.dark_boundary[index] = false;
            roles.visible_ridge_centres[index] = false;
        }

        let correction = correct_antialias_partition(&image, &labels, 3, &roles);

        for y in 2..=6 {
            let index = y * width + 3;
            assert_ne!(correction.labels[index], correction.labels[y * width + 1]);
            assert_ne!(correction.labels[index], correction.labels[y * width + 5]);
            assert!(correction.paint_samples[index]);
        }
        assert_eq!(correction.split_regions, 0);
    }

    #[test]
    fn thin_paint_rejects_one_sided_erosion_before_commit() {
        let width = 10;
        let height = 5;
        let mut source = Raster::blank(width, height, [0.2; 3]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if x >= 6 {
                    labels[index] = 1;
                    source.pixels[index] = [0.9; 3];
                } else if x >= 4 {
                    labels[index] = 2;
                    source.pixels[index] = if x == 4 { [0.2; 3] } else { [0.0; 3] };
                }
            }
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => [0.2; 3],
                    1 => [0.9; 3],
                    _ => [0.0; 3],
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 3);
        let original = labels.clone();
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };
        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );
        assert_eq!(segmentation.labels, original);
        assert_eq!(segmentation.summary.thin_paint_preflight_rejected, 1);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 0);
    }

    #[test]
    fn thin_two_sided_transition_is_reassigned_atomically() {
        let width = 9;
        let height = 5;
        let first = [0.2; 3];
        let second = [0.9; 3];
        let transition = [0.5; 3];
        let mut source = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 5..width {
                labels[y * width + x] = 1;
                source.pixels[y * width + x] = second;
            }
        }
        for (x, colour) in [(3, first), (4, transition), (5, second)] {
            labels[2 * width + x] = 2;
            source.pixels[2 * width + x] = colour;
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => first,
                    1 => second,
                    _ => transition,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 3);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        for x in 3..=5 {
            let index = 2 * width + x;
            assert_ne!(segmentation.labels[index], 2);
            assert!(!segmentation.paint_samples[index]);
        }
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(segmentation.summary.thin_paint_preflight_rejected, 0);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 3);
    }

    #[test]
    fn unstable_coreless_residual_is_not_left_as_a_hard_face() {
        let width = 9;
        let height = 7;
        let upper = [0.18, 0.20, 0.23];
        let lower = [0.72, 0.18, 0.14];
        let overshoot = [1.0, 0.36, 0.25];
        let mut source = Raster::blank(width, height, upper);
        let mut labels = vec![0_u32; width * height];
        for y in 4..height {
            for x in 0..width {
                let index = y * width + x;
                labels[index] = 1;
                source.pixels[index] = lower;
            }
        }
        for x in 2..=6 {
            let index = 3 * width + x;
            labels[index] = 2;
            source.pixels[index] = match x {
                2 | 3 => upper,
                5 | 6 => lower,
                _ => overshoot,
            };
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => upper,
                    1 => lower,
                    _ => overshoot,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 3);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        for x in 2..=6 {
            let index = 3 * width + x;
            assert_ne!(segmentation.labels[index], 2);
            assert!(!segmentation.paint_samples[index]);
        }
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(segmentation.summary.thin_paint_preflight_rejected, 0);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 5);
    }

    #[test]
    fn subresolution_two_face_cap_does_not_become_an_independent_vector_face() {
        let width = 24;
        let height = 14;
        let first = [0.03, 0.04, 0.03];
        let second = [0.52, 0.82, 0.52];
        let cap = [0.08, 0.61, 0.08];
        let mut source = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 12..width {
                let index = y * width + x;
                source.pixels[index] = second;
                labels[index] = 1;
            }
        }
        // Three pixels wide gives this cap a four-connected core.  Its two
        // durable neighbours are nevertheless much larger, and the cap is a
        // clearly one-sided terminal sampling phase rather than a stable face.
        for y in 3..=8 {
            for x in 11..=13 {
                let index = y * width + x;
                source.pixels[index] = cap;
                labels[index] = 2;
            }
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => first,
                    1 => second,
                    _ => cap,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 3);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        let second_owner = segmentation.labels[7 * width + 18];
        for y in 3..=8 {
            for x in 11..=13 {
                let index = y * width + x;
                assert_eq!(segmentation.labels[index], second_owner);
                assert!(!segmentation.paint_samples[index]);
            }
        }
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 18);
    }

    #[test]
    fn gapped_microfacets_follow_the_same_anchored_boundary_island() {
        let width = 40;
        let height = 16;
        let first = [0.03, 0.04, 0.03];
        let second = [0.52, 0.82, 0.52];
        let phase = [0.08, 0.61, 0.08];
        let mut source = Raster::blank(width, height, first);
        let mut labels = vec![0_u32; width * height];
        for y in 8..height {
            for x in 0..width {
                let index = y * width + x;
                source.pixels[index] = second;
                labels[index] = 1;
            }
        }
        let fragments = [(2_usize, 16_usize), (20, 3), (25, 3)];
        for (fragment, &(start, length)) in fragments.iter().enumerate() {
            for x in start..start + length {
                let index = 8 * width + x;
                source.pixels[index] = phase;
                labels[index] = 2 + fragment as u32;
            }
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => first,
                    1 => second,
                    _ => phase,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 5);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: (0..5).collect(),
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        let second_owner = segmentation.labels[10 * width];
        for &(start, length) in &fragments {
            for x in start..start + length {
                let index = 8 * width + x;
                assert_eq!(segmentation.labels[index], second_owner);
                assert!(!segmentation.paint_samples[index]);
            }
        }
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 22);
    }

    #[test]
    fn repeated_boundary_phases_are_rejoined_without_hue_or_lightness_rules() {
        let width = 32;
        let height = 14;
        let carrier = [0.30, 0.55, 0.72];
        let opposite = [0.88, 0.35, 0.12];
        let phase = [0.32, 0.57, 0.74];
        let mut source = Raster::blank(width, height, carrier);
        let mut labels = vec![0_u32; width * height];
        for y in 7..height {
            for x in 0..width {
                let index = y * width + x;
                source.pixels[index] = opposite;
                labels[index] = 4;
            }
        }
        let fragments = [(2_usize, 7_usize), (12, 6), (22, 6)];
        for (group, &(start_x, length)) in fragments.iter().enumerate() {
            for x in start_x..start_x + length {
                let index = 7 * width + x;
                source.pixels[index] = phase;
                labels[index] = 1 + group as u32;
            }
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => carrier,
                    4 => opposite,
                    _ => phase,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 5);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: (0..5).collect(),
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        let durable_carrier = segmentation.labels[6 * width];
        for &(start_x, length) in &fragments {
            for x in start_x..start_x + length {
                let index = 7 * width + x;
                assert_eq!(segmentation.labels[index], durable_carrier);
                assert!(!segmentation.paint_samples[index]);
            }
        }
        assert_eq!(segmentation.summary.thin_paint_refined, 3);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 19);
    }

    #[test]
    fn diagonal_phase_family_uses_repeated_parent_contacts_collectively() {
        let width = 24;
        let height = 14;
        let first = [0.18, 0.22, 0.26];
        let second = [0.72, 0.76, 0.80];
        let phase = [0.66, 0.70, 0.74];
        let mut source = Raster::blank(width, height, second);
        let mut labels = vec![5_u32; width * height];
        for y in 0..7 {
            for x in 0..10 {
                let index = y * width + x;
                source.pixels[index] = first;
                labels[index] = 0;
            }
        }
        for (label, x) in (1_u32..=4).zip([4_usize, 7, 10, 13]) {
            let index = 7 * width + x;
            source.pixels[index] = phase;
            labels[index] = label;
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| match label {
                    0 => first,
                    5 => second,
                    _ => phase,
                })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 6);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: (0..6).collect(),
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        let second_owner = segmentation.labels[8 * width + 12];
        for x in [4_usize, 7, 10, 13] {
            let index = 7 * width + x;
            assert_eq!(segmentation.labels[index], second_owner);
            assert!(!segmentation.paint_samples[index]);
        }
        assert_eq!(segmentation.summary.thin_paint_refined, 4);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 4);
    }

    #[test]
    fn boundary_phase_cannot_discard_colour_when_its_matching_parent_is_screened() {
        let width = 24;
        let height = 14;
        let background = [0.92; 3];
        let red = [0.75, 0.08, 0.06];
        let mut source = Raster::blank(width, height, background);
        let mut labels = vec![0; width * height];
        for y in 7..height {
            for x in 0..width {
                let i = y * width + x;
                source.pixels[i] = red;
                labels[i] = 5;
            }
        }
        for (label, x) in (1..=4).zip([4, 7, 10, 13]) {
            labels[7 * width + x] = label;
        }
        // This equally red, coreless fragment screens the first phase
        // from the durable red face. Its only direct durable neighbour is
        // the background, but that does not make the phase background.
        for x in 3..=5 {
            labels[7 * width + x] = 6;
            labels[8 * width + x] = 6;
        }
        labels[7 * width + 4] = 1;
        let regions = region_stats(&source, &labels, 7);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: (0..7).collect(),
            paint_samples: vec![true; width * height],
            canonical: source.clone(),
            regions,
            summary: SegmentationSummary::default(),
        };
        refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );
        let tip = 7 * width + 4;
        assert_ne!(segmentation.labels[tip], segmentation.labels[0]);
        assert_eq!(segmentation.canonical.pixels[tip], red);
        // The unscreened phases can still rejoin their matching parent.
        for x in [7, 10, 13] {
            assert_eq!(
                segmentation.labels[7 * width + x],
                segmentation.labels[10 * width + x]
            );
        }
    }

    #[test]
    fn source_supported_thin_face_is_returned_to_structural_line_graph() {
        let width = 15;
        let height = 9;
        let face = [0.75, 0.42, 0.18];
        let line = [0.16, 0.30, 0.68];
        let mut source = Raster::blank(width, height, face);
        let mut labels = vec![0_u32; width * height];
        let mut structural_line = vec![false; width * height];
        for x in 4..=10 {
            let index = 4 * width + x;
            source.pixels[index] = line;
            labels[index] = 1;
            structural_line[index] = true;
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| if label == 0 { face } else { line })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 2);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let structural_ownership = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &structural_line,
        );

        for x in 4..=10 {
            let index = 4 * width + x;
            assert_eq!(segmentation.labels[index], 0);
            assert!(!segmentation.paint_samples[index]);
            assert!(structural_ownership[index]);
        }
        assert_eq!(segmentation.summary.thin_paint_refined, 1);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 7);
    }

    #[test]
    fn repeated_microdots_inside_one_face_remain_authored_paint() {
        let width = 32;
        let height = 14;
        let carrier = [0.74, 0.45, 0.67];
        let dot = [0.20, 0.78, 0.42];
        let mut source = Raster::blank(width, height, carrier);
        let mut labels = vec![0_u32; width * height];
        for group in 0..4 {
            let start_x = 2 + 7 * group;
            for x in start_x..start_x + 6 {
                let index = 7 * width + x;
                source.pixels[index] = dot;
                labels[index] = 1 + group as u32;
            }
        }
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| if label == 0 { carrier } else { dot })
                .collect(),
        );
        let regions = region_stats(&source, &labels, 5);
        let mut segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: (0..5).collect(),
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let _ = refine_thin_paint_ownership(
            &source,
            &mut segmentation,
            &vec![false; width * height],
            &vec![false; width * height],
        );

        for group in 0..4 {
            let start_x = 2 + 7 * group;
            for x in start_x..start_x + 6 {
                assert_ne!(segmentation.labels[7 * width + x], 0);
            }
        }
        assert_eq!(segmentation.summary.thin_paint_refined, 0);
        assert_eq!(segmentation.summary.thin_paint_reassigned_pixels, 0);
    }
}
