/// Turn the soft keyed boundary into vector ownership while removing backing
/// contamination from the retained side.  Pixels below the half-coverage
/// crossing become uniform backing; pixels above it are unmixed with the
/// standard compositing equation before ordinary colour segmentation.
pub(crate) fn separate_foreground(image: &Raster, matte: &AlphaMatte, backing: [f32; 3]) -> Raster {
    if image.pixels.len() != matte.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(matte.iter())
            .map(|(&pixel, alpha)| {
                let alpha = alpha.clamp(0.0, 1.0);
                if alpha < BACKGROUND_OWNERSHIP_ALPHA {
                    backing
                } else {
                    [0, 1, 2].map(|channel| {
                        ((pixel[channel] - backing[channel] * (1.0 - alpha)) / alpha.max(1e-6))
                            .clamp(0.0, 1.0)
                    })
                }
            })
            .collect(),
    )
}

mod tests {
    use super::*;

    #[test]
    fn alpha_resize_ignores_hidden_rgb_and_preserves_unscaled_samples() {
        let matte = AlphaMatte::from_u8(
            128,
            64,
            (0..128 * 64)
                .map(|i| if i % 128 < 64 { 255 } else { 0 })
                .collect(),
        );
        let foreground = Raster::blank(128, 64, [1.0, 0.0, 0.0]);
        let mut hidden = foreground.clone();
        for (i, pixel) in hidden.pixels.iter_mut().enumerate() {
            if matte.get(i) == 0.0 {
                *pixel = [0.0, 0.0, 1.0];
            }
        }
        let (first, _) = resize_source_alpha(&foreground, &matte, 64);
        let (second, _) = resize_source_alpha(&hidden, &matte, 64);
        assert_eq!(first.pixels, second.pixels);
        assert_eq!(first.get(63, 16), [1.0, 0.0, 0.0]);
        let (unchanged, alpha) = resize_source_alpha(&hidden, &matte, 128);
        assert_eq!(unchanged.pixels, hidden.pixels);
        assert_eq!(
            alpha.iter().collect::<Vec<_>>(),
            matte.iter().collect::<Vec<_>>()
        );
        let clear = AlphaMatte::from_u8(128, 64, vec![0; 128 * 64]);
        let (image, alpha) = resize_source_alpha(&hidden, &clear, 64);
        assert!(image.pixels.iter().all(|p| *p == [0.0; 3]));
        assert!(alpha.iter().all(|a| a == 0.0));
    }

    #[test]
    fn detection_accepts_six_keys_but_not_white_or_black() {
        for corner in KEY_CORNERS {
            let image = Raster::blank(8, 8, corner);
            assert_eq!(detect(&image).unwrap().corner, corner);
        }
        assert!(detect(&Raster::blank(8, 8, [1.0; 3])).is_none());
        assert!(detect(&Raster::blank(8, 8, [0.0; 3])).is_none());
    }

    #[test]
    fn detection_looks_past_a_narrow_neutral_outer_frame() {
        let mut image = Raster::blank(512, 384, [0.0, 1.0, 0.0]);
        for y in 0..image.height {
            for x in 0..image.width {
                if x < 3 || y < 3 || x + 3 >= image.width || y + 3 >= image.height {
                    image.pixels[y * image.width + x] = [0.9; 3];
                }
            }
        }
        // An unrelated saturated colour in the interior must not become the
        // key merely because the exact perimeter is neutral.
        for y in 96..288 {
            for x in 128..384 {
                image.pixels[y * image.width + x] = [1.0, 0.0, 0.0];
            }
        }
        let key = detect(&image).expect("green backing behind the trim");
        assert_eq!(key.corner, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn colour_difference_matte_removes_antialias_key_contamination() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        // Half-covered neutral gray and black over green.
        let mut source = Raster::blank(16, 16, key.sampled);
        source.pixels[0] = [0.25, 0.75, 0.25];
        source.pixels[1] = [0.0, 0.5, 0.0];
        let matte = pull_matte(&source, key);
        assert!((matte.get(0) - 0.5).abs() <= 0.5 / 65_535.0);
        assert!((matte.get(1) - 0.5).abs() <= 0.5 / 65_535.0);
    }

    #[test]
    fn green_key_does_not_remove_yellow_or_cyan_foreground() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        let source = Raster::new(2, 1, vec![[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]);
        assert_eq!(
            pull_matte(&source, key).iter().collect::<Vec<_>>(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn same_hue_opaque_objects_keep_colour_and_antialias_coverage() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let foreground = corner.map(|channel| if channel > 0.5 { 0.78 } else { 0.16 });
            let mut source = Raster::blank(32, 32, corner);
            for y in 5..27 {
                for x in 5..27 {
                    let alpha = if (6..26).contains(&x) && (6..26).contains(&y) {
                        1.0
                    } else {
                        0.4
                    };
                    source.pixels[y * 32 + x] =
                        [0, 1, 2].map(|c| foreground[c] * alpha + corner[c] * (1.0 - alpha));
                }
            }
            // A small isolated exact-key patch has no broad background seed
            // and is now preserved, just like a lamp inside a dark housing.
            for y in 14..18 {
                for x in 14..18 {
                    source.pixels[y * 32 + x] = corner;
                }
            }
            let matte = pull_matte(&source, key);
            let separated = separate_foreground(&source, &matte, corner);
            for y in 6..26 {
                for x in 6..26 {
                    let i = y * 32 + x;
                    if (14..18).contains(&x) && (14..18).contains(&y) {
                        assert_eq!(matte.get(i), 1.0);
                    } else {
                        assert!(
                            (matte.get(i) - 1.0).abs() < 1e-4,
                            "lost {corner:?} foreground at ({x}, {y})"
                        );
                        assert!(separated.pixels[i]
                            .iter()
                            .zip(foreground)
                            .all(|(actual, expected)| (actual - expected).abs() < 1e-4));
                    }
                }
            }
            assert!((matte.get(5 * 32 + 16) - 0.4).abs() < 1e-4);
            assert_eq!(matte.get(0), 0.0);
        }
    }

    #[test]
    fn background_colour_variation_does_not_create_opaque_islands() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let mut source = Raster::blank(40, 40, corner);
            // Broad backing variations must not become opaque just because
            // erosion leaves an interior. These include the reported green
            // gap samples (0,242,0) and (0,239,0), plus channel contamination.
            for (channel_shift, contamination) in [
                (13.0, 0.0),
                (16.0, 0.0),
                (32.0, 16.0),
                (40.0, 16.0),
                (85.0, 0.0),
                (85.0, 12.0),
                (100.0, 8.0),
            ] {
                for y in 4..36 {
                    for x in 4..36 {
                        source.pixels[y * 40 + x] = corner.map(|channel| {
                            if channel > 0.5 {
                                1.0 - channel_shift / 255.0
                            } else {
                                contamination / 255.0
                            }
                        });
                    }
                }
                let matte = pull_matte(&source, key);
                assert!(matte.iter().all(|alpha| alpha < BACKGROUND_OWNERSHIP_ALPHA));
                assert!(separate_foreground(&source, &matte, corner)
                    .pixels
                    .iter()
                    .all(|pixel| *pixel == corner));
            }
        }
    }

    #[test]
    fn isolated_shaded_key_colour_is_preserved_without_a_broad_seed() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let source = Raster::blank(16, 16, corner.map(|v| v * 0.35));
            let matte = pull_matte(&source, key);
            assert!(matte.iter().all(|a| a == 1.0));
            let separated = separate_foreground(&source, &matte, corner);
            assert_eq!(separated.pixels, source.pixels);
        }
    }

    #[test]
    fn connected_key_shadow_requires_a_core_and_limits_fringe_propagation() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let mut source = Raster::blank(40, 28, corner);
            for y in 2..26 {
                for x in 4..32 {
                    source.pixels[y * 40 + x] = [0.0; 3];
                }
            }
            let shadow = corner.map(|v| v * 0.35);
            // A thick shaded patch extends across the black silhouette into
            // clear backing; its attached one-pixel fringe extends inward.
            for y in 6..13 {
                for x in 26..35 {
                    source.pixels[y * 40 + x] = shadow;
                }
            }
            for x in 10..26 {
                source.pixels[9 * 40 + x] = shadow;
            }
            // An isolated thick mark has no clear-background neighbour.
            for y in 17..23 {
                for x in 10..16 {
                    source.pixels[y * 40 + x] = shadow;
                }
            }
            let matte = pull_matte(&source, key);
            assert_eq!(matte.get(9 * 40 + 30), 0.0);
            assert!((matte.get(9 * 40 + 12) - 0.65).abs() < 1e-4);
            assert_eq!(matte.get(20 * 40 + 12), 1.0);
            assert_eq!(matte.get(14 * 40 + 24), 1.0);
        }
    }

    #[test]
    fn server_rack_lamps_keep_opaque_colour_at_native_and_reduced_sizes() {
        let image = image::load_from_memory(include_bytes!("../data/server-rack-key.png"))
            .expect("server rack fixture");
        let original = Raster::from_dynamic(&image);
        for maximum in [836, 418] {
            let source = original.resize_max(maximum);
            let key = detect(&source).expect("outer green background");
            let matte = pull_matte(&source, key);
            let scale = source.width as f32 / original.width as f32;
            assert_eq!(matte.get(10 * source.width + 10), 0.0);
            for cy in [196, 343, 490, 636] {
                for y in cy - 12..=cy + 12 {
                    for x in 485..=502 {
                        let (x, y) = ((x as f32 * scale) as usize, (y as f32 * scale) as usize);
                        assert_eq!(
                            matte.get(y * source.width + x),
                            1.0,
                            "lamp lost coverage at ({x}, {y}), size {maximum}"
                        );
                    }
                }
            }
            let separated = separate_foreground(&source, &matte, key.sampled);
            let i = (343.0 * scale) as usize * source.width + (492.0 * scale) as usize;
            assert_eq!(separated.pixels[i], source.pixels[i]);
        }
    }

    #[test]
    fn broad_key_patches_seed_removal_but_isolated_small_patches_do_not() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let mut source = Raster::blank(64, 48, [0.0; 3]);
            for y in 4..20 {
                for x in 4..20 {
                    source.pixels[y * 64 + x] = corner;
                }
                for x in 20..28 {
                    source.pixels[y * 64 + x] = corner.map(|c| c * 0.35);
                }
            }
            for y in 30..34 {
                for x in 42..46 {
                    source.pixels[y * 64 + x] = corner;
                }
                for x in 46..50 {
                    source.pixels[y * 64 + x] = corner.map(|c| c * 0.35);
                }
            }
            let matte = pull_matte(&source, key);
            assert_eq!(matte.get(12 * 64 + 12), 0.0);
            assert_eq!(matte.get(12 * 64 + 24), 0.0);
            assert_eq!(matte.get(32 * 64 + 44), 1.0);
            assert_eq!(matte.get(32 * 64 + 48), 1.0);
            assert_eq!(matte.get(24 * 64 + 32), 1.0);
        }
    }

    #[test]
    fn background_does_not_jump_a_one_pixel_outline_into_a_small_lamp() {
        for corner in KEY_CORNERS {
            let key = ChromaKey {
                corner,
                sampled: corner,
                border_coverage: 1.0,
            };
            let mut source = Raster::blank(32, 32, corner);
            for y in 10..16 {
                for x in 10..16 {
                    source.pixels[y * 32 + x] = [0.0; 3];
                }
            }
            for y in 11..15 {
                for x in 11..15 {
                    source.pixels[y * 32 + x] = corner.map(|v| v * if x < 13 { 1.0 } else { 0.35 });
                }
            }
            let matte = pull_matte(&source, key);
            assert_eq!(matte.get(8 * 32 + 12), 0.0);
            for y in 10..16 {
                for x in 10..16 {
                    assert_eq!(matte.get(y * 32 + x), 1.0);
                }
            }
        }
    }

    #[test]
    fn enclosed_background_seed_requirement_is_capped_on_large_canvases() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        let mut source = Raster::blank(1024, 1024, [0.0; 3]);
        for y in 400..416 {
            for x in 400..416 {
                source.pixels[y * 1024 + x] = key.sampled;
            }
        }
        let matte = pull_matte(&source, key);
        assert_eq!(matte.get(408 * 1024 + 408), 0.0);
        assert_eq!(matte.get(390 * 1024 + 408), 1.0);
    }

    #[test]
    fn shaded_key_gap_keeps_background_ownership() {
        let image = image::load_from_memory(include_bytes!("../data/shaded-key-gap.png"))
            .expect("shaded backing fixture");
        let source = Raster::from_dynamic(&image);
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        let matte = pull_matte(&source, key);
        for (x, y) in [(44, 115), (45, 116), (46, 116), (47, 116)] {
            assert!(
                matte.get(y * source.width + x) < BACKGROUND_OWNERSHIP_ALPHA,
                "shaded gap became foreground at ({x}, {y})"
            );
        }
        for (x, y) in [(40, 108), (45, 132), (42, 129)] {
            assert_eq!(
                matte.get(y * source.width + x),
                0.0,
                "connected shadow survived at ({x}, {y})"
            );
        }
        for (x, y) in [(20, 100), (80, 90)] {
            assert_eq!(
                matte.get(y * source.width + x),
                1.0,
                "blue clothing lost coverage at ({x}, {y})"
            );
        }
    }

    #[test]
    fn source_alpha_is_composited_over_a_saturated_temporary_backing() {
        let image = Raster::new(2, 1, vec![[1.0, 0.0, 0.0], [0.5; 3]]);
        let matte = AlphaMatte::new(2, 1, vec![0.0, 0.5]);
        let composed = composite_over(&image, &matte, [0.0, 1.0, 0.0]);
        assert_eq!(composed.pixels[0], [0.0, 1.0, 0.0]);
        assert_eq!(composed.pixels[1], [0.25, 0.75, 0.25]);
    }

    #[test]
    fn source_alpha_is_quantized_to_four_uniform_levels() {
        let matte = AlphaMatte::new(8, 1, vec![0.0, 0.10, 0.17, 0.49, 0.50, 0.82, 0.84, 1.0]);
        let quantized = matte.quantized_2bit();
        assert_eq!(
            quantized.iter().collect::<Vec<_>>(),
            vec![
                0.0,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                2.0 / 3.0,
                1.0,
                1.0
            ]
        );
        assert_eq!(quantized.storage_bytes(), 2);
        assert_eq!(matte.quantized_levels(), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn narrow_source_antialias_shoulders_are_not_vector_alpha_regions() {
        let matte = AlphaMatte::new(
            5,
            3,
            [0.0, 0.25, 0.75, 1.0, 1.0]
                .into_iter()
                .cycle()
                .take(15)
                .collect(),
        );
        assert_eq!(
            matte.vectorized_levels(),
            [0, 0, 3, 3, 3]
                .into_iter()
                .cycle()
                .take(15)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn authored_translucent_area_is_retained() {
        let mut values = vec![0.0; 25];
        for y in 1..4 {
            for x in 1..4 {
                values[y * 5 + x] = 0.25;
            }
        }
        let matte = AlphaMatte::new(5, 5, values);
        let levels = matte.vectorized_levels();
        for y in 1..4 {
            for x in 1..4 {
                assert_eq!(levels[y * 5 + x], 1);
            }
        }
    }

    #[test]
    fn partial_line_connected_to_opaque_junction_is_not_an_entire_shoulder() {
        let mut values = vec![0.0; 64 * 64];
        for y in 2..62 {
            for x in 31..33 {
                values[y * 64 + x] = 0.8;
            }
        }
        values[32 * 64 + 31] = 1.0;
        let levels = AlphaMatte::new(64, 64, values).vectorized_levels();
        for y in 2..62 {
            assert!(
                levels[y * 64 + 31] > 0 && levels[y * 64 + 32] > 0,
                "erased line at y={y}"
            );
        }
    }

    #[test]
    fn broad_partial_alpha_transition_keeps_a_durable_core() {
        let matte = AlphaMatte::new(
            5,
            5,
            [0.0, 0.25, 0.25, 0.25, 1.0]
                .into_iter()
                .cycle()
                .take(25)
                .collect(),
        );
        let levels = matte.vectorized_levels();
        assert_eq!(levels[2 * 5 + 2], 1);
        assert!(levels.contains(&1));
    }

    #[test]
    fn source_alpha_isoline_uses_the_interpolated_subpixel_crossing() {
        let matte = AlphaMatte::new(2, 2, vec![0.0, 0.75, 0.0, 0.75]);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);

        // Pixel centres are x=0.5 and x=1.5. Alpha=0.5 crosses two thirds of
        // the way between them, at x=7/6, rather than at their grid edge x=1.
        let expected = 7.0 / 6.0;
        let crossings = contours[0]
            .iter()
            .filter(|point| (point.x - expected).abs() < 1e-5)
            .count();
        assert_eq!(crossings, 2, "contour was snapped back to pixel edges");
    }

    #[test]
    fn opaque_alpha_isoline_covers_all_four_canvas_corners() {
        let matte = AlphaMatte::new(2, 2, vec![1.0; 4]);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);
        for corner in [
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 0.0, y: 2.0 },
        ] {
            assert!(contours[0].contains(&corner), "missing corner {corner:?}");
        }
    }

    #[test]
    fn source_alpha_rgb_is_retained_for_every_nonzero_coverage_sample() {
        let image = Raster::new(
            3,
            1,
            vec![[0.8, 0.1, 0.2], [0.1, 0.8, 0.2], [0.1, 0.2, 0.8]],
        );
        let matte = AlphaMatte::new(3, 1, vec![0.0, 0.1, 1.0]);
        let prepared = prepare_source_alpha(&image, &matte);
        assert_eq!(prepared.pixels[0], image.pixels[1]);
        assert_eq!(prepared.pixels[1], image.pixels[1]);
        assert_eq!(prepared.pixels[2], image.pixels[2]);
    }

    #[test]
    fn foreground_side_is_unmixed_and_background_side_is_normalized() {
        let image = Raster::new(2, 1, vec![[0.3, 0.7, 0.3], [0.2, 0.8, 0.2]]);
        let matte = AlphaMatte::new(2, 1, vec![0.6, 0.4]);
        let separated = separate_foreground(&image, &matte, [0.0, 1.0, 0.0]);
        assert!(separated.pixels[0]
            .into_iter()
            .all(|channel| (channel - 0.5).abs() < 1e-6));
        assert_eq!(separated.pixels[1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn structural_stroke_on_background_side_is_rejected() {
        let matte = AlphaMatte::new(3, 1, vec![0.1, 0.2, 0.9]);
        assert!(!matte.retains_stroke(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
        ]));
        assert!(matte.retains_stroke(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
        ]));
    }

    #[test]
    fn disconnected_clear_labels_are_all_background() {
        let matte = AlphaMatte::new(5, 1, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let removed = background_regions(&[0, 0, 1, 2, 2], 3, &matte);
        assert_eq!(removed, vec![true, false, true]);
    }

    #[test]
    fn background_region_tolerates_many_antialiased_boundary_samples() {
        let matte = AlphaMatte::new(
            10,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.20, 0.30],
        );
        assert_eq!(background_regions(&[0; 10], 1, &matte), vec![true]);

        let foreground = AlphaMatte::new(
            10,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(background_regions(&[0; 10], 1, &foreground), vec![false]);
    }
}

impl AlphaMatte {
    pub(crate) fn new(width: usize, height: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), width * height);
        Self {
            width,
            height,
            values: AlphaValues::Float(values),
        }
    }

    fn storage_bytes(&self) -> usize {
        match &self.values {
            #[cfg(test)]
            AlphaValues::Float(values) => std::mem::size_of_val(values.as_slice()),
            AlphaValues::Unorm16(values) => std::mem::size_of_val(values.as_slice()),
            AlphaValues::Byte(values) => values.len(),
            AlphaValues::Packed2 { bytes, .. } => bytes.len(),
        }
    }

    /// Quantize exact source coverage to the four values representable by a
    /// two-bit alpha channel: 0, 1/3, 2/3, and 1.
    pub(crate) fn quantized_2bit(&self) -> Self {
        if matches!(self.values, AlphaValues::Packed2 { .. }) {
            return self.clone();
        }
        Self::packed_2bit_from_levels(
            self.width,
            self.height,
            (0..self.len()).map(|index| self.quantized_level_at(index)),
        )
    }

    pub(crate) fn quantized_levels(&self) -> Vec<u8> {
        (0..self.len())
            .map(|index| self.quantized_level_at(index))
            .collect()
    }
}
