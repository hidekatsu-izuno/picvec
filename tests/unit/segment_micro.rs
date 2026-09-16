mod tests {
    use super::*;
    fn partition(image: &Raster, labels: Vec<u32>, count: usize) -> Segmentation {
        Segmentation {
            width: image.width,
            height: image.height,
            regions: region_stats(image, &labels, count),
            labels,
            paint_keys: (0..count as u32).collect(),
            paint_samples: vec![true; image.pixels.len()],
            canonical: image.clone(),
            summary: Default::default(),
        }
    }
    #[test]
    fn wiper_material_is_not_reassigned_to_glass_through_a_coverage_parent() {
        let image = image::load_from_memory(include_bytes!("../data/car-wiper-material.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            64,
            48,
            image
                .pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let labels = include_bytes!("../data/car-wiper-material.labels");
        let mut seg = partition(
            &source,
            labels.iter().map(|&v| v as u32).collect(),
            *labels.iter().max().unwrap() as usize + 1,
        );
        absorb_micro_regions(&source, &mut seg, None);
        let sample = 17 * 64 + 18;
        let owner = seg.labels[sample];
        let pixels: Vec<_> = seg
            .labels
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == owner).then_some(i))
            .collect();
        let material = colour_distribution(&source, &pixels);
        assert!(
            material[0][2] < 0.4,
            "wiper inherited glass paint: {material:?}"
        );
    }

    #[test]
    fn local_ownership_and_paint_evidence_are_identical_across_worker_counts() {
        for (png, labels) in [
            (
                &include_bytes!("../data/car-wiper-material.png")[..],
                &include_bytes!("../data/car-wiper-material.labels")[..],
            ),
            (
                &include_bytes!("../data/remojii-micro-boundary.png")[..],
                &include_bytes!("../data/remojii-micro-boundary.labels")[..],
            ),
            (
                &include_bytes!("../data/remojii-micro-junction.png")[..],
                &include_bytes!("../data/remojii-micro-junction.labels")[..],
            ),
        ] {
            let image = image::load_from_memory(png).unwrap().to_rgb8();
            let source = Raster::new(
                image.width() as usize,
                image.height() as usize,
                image
                    .pixels()
                    .map(|p| p.0.map(|v| v as f32 / 255.0))
                    .collect(),
            );
            let count = *labels.iter().max().unwrap() as usize + 1;
            let mut reference = None;
            for threads in [1, 2, 4] {
                let mut seg = partition(&source, labels.iter().map(|&v| v as u32).collect(), count);
                let merges = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| absorb_micro_regions(&source, &mut seg, None));
                assert!(merges > 0);
                let result = (
                    merges,
                    seg.labels,
                    seg.canonical.pixels,
                    seg.paint_samples,
                    seg.summary.micro_pixels_reassigned,
                );
                if let Some(expected) = &reference {
                    assert_eq!(&result, expected, "workers: {threads}");
                } else {
                    reference = Some(result);
                }
            }
        }
    }

    #[test]
    fn chained_unions_do_not_leave_fragments_under_retired_parent_labels() {
        let widths: Vec<usize> = (10..22).collect();
        let w = widths.iter().sum();
        let source = Raster::blank(w, 16, [0.02; 3]);
        let mut labels = vec![0; w * 16];
        let mut offset = 0;
        for (id, width) in widths.iter().enumerate() {
            for y in 0..16 {
                for x in offset..offset + width {
                    labels[y * w + x] = id as u32;
                }
            }
            offset += width;
        }
        let mut seg = partition(&source, labels, widths.len());
        absorb_micro_regions(&source, &mut seg, None);
        assert_eq!(seg.regions.len(), 1);
    }

    #[test]
    fn matching_source_material_is_not_limited_by_fragment_area() {
        for side in [16, 64, 128] {
            let w = side * 3;
            let mut source = Raster::blank(w, w, [0.0; 3]);
            let mut labels = vec![0; w * w];
            for y in 0..w {
                for x in 0..w {
                    source.pixels[y * w + x] = [((x * 17 + y * 23) % 5) as f32 / 255.0; 3];
                    if (side..side * 2).contains(&x) && (side..side * 2).contains(&y) {
                        labels[y * w + x] = 1;
                    }
                }
            }
            let mut seg = partition(&source, labels, 2);
            assert_eq!(absorb_micro_regions(&source, &mut seg, None), 1);
            assert_eq!(seg.regions.len(), 1, "fragment area {}", side * side);
            let center = (side + side / 2) * w + side + side / 2;
            assert!(
                seg.paint_samples[center],
                "material interiors remain fit evidence"
            );
            assert_eq!(seg.canonical.pixels[center], source.pixels[center]);
        }
    }

    #[test]
    fn large_low_contrast_faces_gradients_and_authored_opacity_are_preserved() {
        for mode in 0..4 {
            let w = 64;
            let mut source = Raster::blank(w, w, [0.5; 3]);
            let mut labels = vec![0; w * w];
            let mut alpha = vec![1.0; w * w];
            for y in 16..40 {
                for x in 16..40 {
                    labels[y * w + x] = 1;
                    source.pixels[y * w + x] = [0.5
                        + match mode {
                            0 | 3 => 5.0 / 255.0,
                            1 => (x - 16) as f32 / 23.0 * 4.0 / 255.0,
                            _ => 0.0,
                        }; 3];
                    if mode == 2 {
                        alpha[y * w + x] = 0.4;
                    }
                    if mode == 3 && (x == 16 || x == 39 || y == 16 || y == 39) {
                        source.pixels[y * w + x] = [0.5 + 2.5 / 255.0; 3];
                    }
                }
            }
            let matte = crate::chroma::AlphaMatte::new(w, w, alpha);
            let mut seg = partition(&source, labels, 2);
            assert_eq!(
                absorb_micro_regions(&source, &mut seg, Some(&matte)),
                0,
                "mode {mode}"
            );
            assert_eq!(seg.regions.len(), 2);
        }
    }

    #[test]
    fn nearby_paint_does_not_justify_replacing_an_island_with_an_unrelated_colour() {
        for full_coverage in [false, true] {
            let mut source = Raster::blank(32, 32, [0.02; 3]);
            let mut labels = vec![0; 1024];
            for y in 0..32 {
                for x in 16..32 {
                    source.pixels[y * 32 + x] = [0.9; 3];
                    labels[y * 32 + x] = 1;
                }
            }
            let terminal = 16 * 32 + 13;
            source.pixels[terminal] = [if full_coverage { 0.9 } else { 0.6 }; 3];
            labels[terminal] = 2;
            let mut seg = partition(&source, labels, 3);
            absorb_micro_regions(&source, &mut seg, None);
            // Neither the white paint three pixels away nor the incident
            // black paint can own this island without changing its material.
            assert_eq!(seg.regions[seg.labels[terminal] as usize].area, 1);
            assert_ne!(seg.labels[terminal], seg.labels[terminal - 1]);
            assert_ne!(seg.labels[terminal], seg.labels[16 * 32 + 16]);
        }
    }

    #[test]
    fn absorbs_local_boundary_mixtures_but_keeps_marks_lines_and_authored_alpha() {
        let (w, h) = (32, 32);
        let mut source = Raster::blank(w, h, [0.02; 3]);
        let mut labels = vec![0; w * h];
        let mut alpha = vec![1.0; w * h];
        for y in 0..h {
            for x in 16..w {
                source.pixels[y * w + x] = [0.7 + y as f32 * 0.008; 3];
                labels[y * w + x] = 1;
            }
        }
        let aa = 16 * w + 15;
        source.pixels[aa] = [0.42; 3];
        labels[aa] = 2;
        let dot = 8 * w + 8;
        source.pixels[dot] = [0.9, 0.05, 0.04];
        labels[dot] = 3;
        let highlight = 4 * w + 4;
        source.pixels[highlight] = [1.0; 3];
        labels[highlight] = 4;
        for y in 10..14 {
            source.pixels[y * w + 24] = [0.02; 3];
            labels[y * w + 24] = 5;
        }
        let translucent = 20 * w + 4;
        labels[translucent] = 6;
        alpha[translucent] = 0.4;
        let mut seg = partition(&source, labels, 7);
        let matte = crate::chroma::AlphaMatte::new(w, h, alpha);
        assert_eq!(absorb_micro_regions(&source, &mut seg, Some(&matte)), 1);
        assert!(seg.regions[seg.labels[aa] as usize].area > 4);
        assert!(!seg.paint_samples[aa]);
        for point in [dot, highlight, translucent, 11 * w + 24] {
            assert_ne!(seg.labels[point], seg.labels[point + 1]);
        }
    }
    #[test]
    fn remojii_reported_one_pixel_objects_are_absorbed_before_fitting() {
        for (png, labels) in [
            (
                &include_bytes!("../data/remojii-micro-boundary.png")[..],
                &include_bytes!("../data/remojii-micro-boundary.labels")[..],
            ),
            (
                &include_bytes!("../data/remojii-micro-junction.png")[..],
                &include_bytes!("../data/remojii-micro-junction.labels")[..],
            ),
        ] {
            let image = image::load_from_memory(png).unwrap().to_rgb8();
            let source = Raster::new(
                16,
                16,
                image
                    .pixels()
                    .map(|p| p.0.map(|v| v as f32 / 255.0))
                    .collect(),
            );
            let count = *labels.iter().max().unwrap() as usize + 1;
            let mut seg = partition(&source, labels.iter().map(|&v| v as u32).collect(), count);
            let sample = 8 * 16 + 8;
            assert_eq!(seg.regions[seg.labels[sample] as usize].area, 1);
            assert!(absorb_micro_regions(&source, &mut seg, None) > 0);
            assert!(seg.regions[seg.labels[sample] as usize].area > 4);
            assert!(!seg.paint_samples[sample]);
        }
    }
    #[test]
    fn three_colour_junction_is_coverage_but_a_fourth_colour_mark_is_not() {
        for mark in [false, true] {
            let (w, h) = (24, 24);
            let colors = [[0.01; 3], [0.4, 0.1, 0.55], [0.9, 0.55, 0.1]];
            let mut labels = vec![0; w * h];
            let mut source = Raster::blank(w, h, colors[0]);
            for y in 0..h {
                for x in 12..w {
                    let p = if y < 12 { 1 } else { 2 };
                    labels[y * w + x] = p as u32;
                    source.pixels[y * w + x] = colors[p];
                }
            }
            let i = 12 * w + 11;
            labels[i] = 3;
            source.pixels[i] = if mark {
                [0.0, 1.0, 0.0]
            } else {
                std::array::from_fn(|c| {
                    colors[0][c] * 0.3 + colors[1][c] * 0.3 + colors[2][c] * 0.4
                })
            };
            // All three incident faces touch this one-pixel junction.
            labels[i - w] = 1;
            source.pixels[i - w] = colors[1];
            let mut seg = partition(&source, labels, 4);
            let count = absorb_micro_regions(&source, &mut seg, None);
            assert_eq!(count, usize::from(!mark));
            assert_eq!(seg.regions[seg.labels[i] as usize].area == 1, mark);
        }
    }
    #[test]
    fn reported_dark_fragments_use_the_observed_ink_range() {
        for (png, labels) in [
            (
                &include_bytes!("../data/remojii-ink-noise.png")[..],
                &include_bytes!("../data/remojii-ink-noise.labels")[..],
            ),
            (
                &include_bytes!("../data/remojii-ink-tip.png")[..],
                &include_bytes!("../data/remojii-ink-tip.labels")[..],
            ),
        ] {
            let image = image::load_from_memory(png).unwrap().to_rgb8();
            let source = Raster::new(
                16,
                16,
                image
                    .pixels()
                    .map(|p| p.0.map(|v| v as f32 / 255.0))
                    .collect(),
            );
            let count = *labels.iter().max().unwrap() as usize + 1;
            let mut seg = partition(&source, labels.iter().map(|&v| v as u32).collect(), count);
            let sample = 8 * 16 + 8;
            assert!(absorb_micro_regions(&source, &mut seg, None) > 0);
            assert!(!seg.paint_samples[sample]);
        }
    }

    #[test]
    fn same_material_patch_merges_without_erasing_a_contrasting_dot() {
        let (w, h) = (32, 32);
        let mut source = Raster::blank(w, h, [0.5; 3]);
        let mut labels = vec![0; w * h];
        for y in 8..16 {
            for x in 8..16 {
                source.pixels[y * w + x] = [0.505; 3];
                labels[y * w + x] = 1;
            }
        }
        let dot = 24 * w + 24;
        source.pixels[dot] = [0.8; 3];
        labels[dot] = 2;
        let mut seg = partition(&source, labels, 3);
        assert_eq!(absorb_micro_regions(&source, &mut seg, None), 1);
        assert_eq!(seg.labels[12 * w + 12], seg.labels[0]);
        assert_ne!(seg.labels[dot], seg.labels[0]);
    }

    #[test]
    fn long_coverage_fringe_merges_but_a_material_patch_keeps_its_interior() {
        for patch in [false, true] {
            let (w, h) = (48, 48);
            let mut source = Raster::blank(w, h, [0.0; 3]);
            let mut labels = vec![0; w * h];
            for y in 0..h {
                for x in 24..w {
                    source.pixels[y * w + x] = [0.9; 3];
                    labels[y * w + x] = 1;
                }
            }
            for y in 8..32 {
                for x in if patch { 22..25 } else { 23..24 } {
                    source.pixels[y * w + x] = [0.4; 3];
                    labels[y * w + x] = 2;
                }
            }
            let mut seg = partition(&source, labels, 3);
            assert_eq!(
                absorb_micro_regions(&source, &mut seg, None),
                usize::from(!patch)
            );
            assert_eq!(seg.paint_samples[16 * w + 23], patch);
        }
    }

    #[test]
    fn four_pixel_coverage_strip_is_not_mistaken_for_an_independent_line() {
        let (w, h) = (24, 24);
        let mut source = Raster::blank(w, h, [0.0; 3]);
        let mut labels = vec![0; w * h];
        for y in 0..h {
            for x in 12..w {
                source.pixels[y * w + x] = [0.9; 3];
                labels[y * w + x] = 1;
            }
        }
        for y in 10..14 {
            source.pixels[y * w + 11] = [0.4; 3];
            labels[y * w + 11] = 2;
        }
        let mut seg = partition(&source, labels, 3);
        assert_eq!(absorb_micro_regions(&source, &mut seg, None), 1);
        for y in 10..14 {
            assert!(seg.regions[seg.labels[y * w + 11] as usize].area > 4);
        }
    }
    #[test]
    fn a_fragmented_outline_does_not_need_a_global_three_by_three_core() {
        let (w, h) = (24, 24);
        let mut source = Raster::blank(w, h, [0.9, 0.88, 0.85]);
        let mut labels = vec![0; w * h];
        for y in 0..h {
            for x in 0..12 {
                labels[y * w + x] = 1;
                source.pixels[y * w + x] = [0.01; 3];
            }
        }
        // The incident dark parent is itself a three-pixel boundary fragment.
        for y in 11..14 {
            labels[y * w + 11] = 2;
        }
        let aa = 12 * w + 12;
        labels[aa] = 3;
        source.pixels[aa] = [0.43, 0.42, 0.40];
        let mut seg = partition(&source, labels, 4);
        absorb_micro_regions(&source, &mut seg, None);
        assert!(seg.regions[seg.labels[aa] as usize].area > 4);
        assert!(!seg.paint_samples[aa]);
    }
}
