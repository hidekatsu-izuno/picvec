mod tests {

    #[test]
    fn local_coverage_rim_is_measured_inside_the_silhouette_without_a_mask() {
        let (w, h) = (192, 192);
        let mut colours = Vec::new();
        let mut values = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 + 0.5 - 96.0).powi(2) + (y as f32 + 0.5 - 96.0).powi(2)).sqrt();
                let alpha = (85.5 - d).clamp(0.0, 1.0);
                let inner = (84.8 - d).clamp(0.0, 1.0);
                values.push(alpha);
                colours.push([0.9, 0.7, 0.2].map(|c| c * inner / alpha.max(1e-6)));
            }
        }
        let source = Raster::new(w, h, colours);
        let matte = crate::chroma::AlphaMatte::new(w, h, values);
        let local = crate::alpha_coverage::detect_components(&matte).unwrap();
        let strokes = recover_local_coverage_boundary(&source, &matte, &local);
        let ink: Vec<_> = strokes
            .iter()
            .filter(|s| s.role == "alpha-boundary-stroke")
            .collect();
        assert!(!ink.is_empty());
        for stroke in ink {
            assert!(stroke.width < 1.8);
            for p in &stroke.points {
                let outer =
                    ((p.x - 96.0).powi(2) + (p.y - 96.0).powi(2)).sqrt() + stroke.width * 0.5;
                assert!(
                    outer < 85.75,
                    "rim extended outside its silhouette: {outer}"
                );
            }
        }
    }

    use super::*;

    #[test]
    fn window_bottom_ridge_uses_its_own_core_instead_of_the_whole_trim() {
        let input = image::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png"),
        )
        .unwrap();
        let source = Raster::from_dynamic(&input);
        let (a, b, width) = refine_isolated_ridge(
            &source,
            Point {
                x: 815.252,
                y: 503.51,
            },
            Point {
                x: 749.156,
                y: 513.332,
            },
            4.886,
        )
        .expect("the source contains a narrow, continuous lower-window line");
        assert!((1.0..2.5).contains(&width), "measured width: {width}");
        let y = a.y + (780.5 - a.x) * (b.y - a.y) / (b.x - a.x);
        assert!(
            (509.0..510.0).contains(&y),
            "line missed the source core: {y}"
        );
    }

    #[test]
    fn isolated_ridge_refinement_preserves_a_wide_band_and_rejects_a_step() {
        for step in [false, true] {
            let pixels = (0..48)
                .flat_map(|y| {
                    (0..128).map(move |_| {
                        if (18..24).contains(&y) || (step && y >= 24) {
                            [0.02; 3]
                        } else {
                            [0.7; 3]
                        }
                    })
                })
                .collect();
            let source = Raster::new(128, 48, pixels);
            assert!(refine_isolated_ridge(
                &source,
                Point { x: 8.5, y: 21.0 },
                Point { x: 119.5, y: 21.0 },
                6.0,
            )
            .is_none());
        }
    }

    #[test]
    fn dotted_ink_keeps_its_gaps_and_ignores_invisible_black() {
        for hidden_black in [false, true] {
            let mut source = Raster::blank(96, 40, [1.0; 3]);
            let mut alpha = vec![1.0; 96 * 40];
            for x in 8..88 {
                if x % 4 < 2 {
                    source.pixels[20 * 96 + x] = [0.0; 3];
                }
                if hidden_black {
                    for y in 17..20 {
                        source.pixels[y * 96 + x] = [0.0; 3];
                        alpha[y * 96 + x] = 0.0;
                    }
                }
            }
            let matte = crate::chroma::AlphaMatte::new(96, 40, alpha);
            let stroke = StructuralStroke {
                points: vec![Point { x: 8.5, y: 20.5 }, Point { x: 87.5, y: 20.5 }],
                path_data: None,
                precise_points: None,
                color: [0.0; 3],
                width: 4.0,
                role: "ridge-on-boundary",
                width_samples: Vec::new(),
            };
            let restored = refine_interrupted(&source, &stroke, Some(&matte)).unwrap();
            assert!(!restored.is_empty());
            assert!(restored.iter().all(|s| s.width <= 1.01));
            assert!(restored
                .iter()
                .flat_map(|s| &s.points)
                .all(|p| (p.y - 20.5).abs() < 0.01));
            for x in (10..86).step_by(4) {
                let gap = x as f32 + 1.0;
                assert!(
                    restored
                        .iter()
                        .all(|s| !(s.points[0].x < gap && s.points[1].x > gap)),
                    "closed source gap at {gap}"
                );
            }
        }
    }

    #[test]
    fn insufficient_profile_contrast_does_not_turn_a_continuous_line_into_dots() {
        let width = 96;
        let height = 40;
        let mut source = Raster::new(
            width,
            height,
            (0..width * height)
                .map(|i| {
                    // The background crosses the profile contrast gate,
                    // while the black line itself remains continuous.
                    let tone = if (i % width) % 12 < 6 { 0.13 } else { 0.18 };
                    [tone; 3]
                })
                .collect(),
        );
        for x in 8..88 {
            source.pixels[20 * width + x] = [0.0; 3];
        }
        let stroke = StructuralStroke {
            points: vec![Point { x: 10.5, y: 20.5 }, Point { x: 85.5, y: 20.5 }],
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 1.8,
            role: "ridge",
            width_samples: Vec::new(),
        };
        assert!(refine_interrupted(&source, &stroke, None).is_none());
    }

    #[test]
    fn a_continuous_thin_line_is_not_replaced_by_dashes() {
        let mut source = Raster::blank(96, 40, [1.0; 3]);
        for x in 8..88 {
            source.pixels[20 * 96 + x] = [0.0; 3];
        }
        let stroke = StructuralStroke {
            points: vec![Point { x: 10.5, y: 20.5 }, Point { x: 85.5, y: 20.5 }],
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 1.2,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        };
        assert!(refine_interrupted(&source, &stroke, None).is_none());
    }

    fn edge() -> SourceEdge {
        SourceEdge {
            points: (6..58).map(|x| [x as f64 + 0.5, 20.5]).collect(),
            width: 3.2,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        }
    }

    fn wide_band(phase: f32, gap: bool, wobble: bool) -> (Raster, SourceEdge) {
        let mut image = Raster::blank(240, 80, [1.0; 3]);
        for y in 0..80 {
            for x in 0..240 {
                let d = y as f32 + 0.5 - (40.0 + phase);
                let width = 12.0
                    + if wobble {
                        0.3 * (x as f32 * 0.09).sin()
                    } else {
                        0.0
                    };
                let coverage = if !(12..228).contains(&x) || (gap && (116..124).contains(&x)) {
                    0.0
                } else {
                    (0.5 * width + 0.5 - d.abs()).clamp(0.0, 1.0)
                };
                let paint = if d < 0.0 {
                    [0.9, 0.8, 0.7]
                } else {
                    [0.6, 0.8, 1.0]
                };
                image.pixels[y * 240 + x] = paint.map(|v| 0.1 * coverage + v * (1.0 - coverage));
            }
        }
        let seed = SourceEdge {
            // A detector's 1.2px inset lies near the outside of a 12px band.
            points: (12..228)
                .map(|x| [x as f64 + 0.5, (34.5 + phase) as f64])
                .collect(),
            width: 1.2,
            role: "dark-boundary",
            width_samples: Vec::new(),
        };
        (image, seed)
    }

    fn render_recovered(
        width: usize,
        height: usize,
        recovered: &Recovery,
    ) -> resvg::tiny_skia::Pixmap {
        let mut ink = super::super::StructuralInk::empty();
        ink.strokes = recovered.strokes.clone();
        let (svg, _) = crate::svg::serialize(width, height, &[], &[], &ink, 0.0, true);
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut pixels = resvg::tiny_skia::Pixmap::new(width as u32, height as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixels.as_mut(),
        );
        pixels
    }

    #[test]
    fn outside_seeds_recover_one_complete_band_at_different_pixel_phases() {
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let (image, seed) = wide_band(phase, false, false);
            let mut opposite = seed.clone();
            for p in &mut opposite.points {
                p[1] += 11.0;
            }
            let recovered = recover(&image, &[seed, opposite], &[]);
            assert_eq!(recovered.strokes.len(), 1, "phase {phase}");
            let stroke = &recovered.strokes[0];
            assert!((stroke.width - 12.0).abs() < 0.15);
            assert!(stroke
                .points
                .iter()
                .all(|p| (p.y - 40.0 - phase).abs() < 0.15));
            assert!(stroke.path_data.as_ref().unwrap().contains(" L"));
            for x in 20..220 {
                for y in 35..45 {
                    assert!(recovered.mask[y * 240 + x], "unremoved core at {x}, {y}");
                }
            }
        }
    }

    #[test]
    fn partial_outline_intervals_keep_the_connected_region_in_paint() {
        for bright in [false, true] {
            let (mut image, seed) = wide_band(0.25, false, false);
            if bright {
                for pixel in &mut image.pixels {
                    *pixel = pixel.map(|c| 1.0 - c);
                }
            }
            assert_eq!(
                recover(&image, std::slice::from_ref(&seed), &[])
                    .strokes
                    .len(),
                1
            );
            let partial = SourceEdge {
                points: seed.points[20..seed.points.len() - 20].to_vec(),
                ..seed
            };
            let rejected = recover(&image, &[partial], &[]);
            assert!(rejected.strokes.is_empty());
            assert!(rejected.updates.is_empty());
            assert!(rejected.mask.iter().all(|&owned| !owned));
        }
    }

    #[test]
    fn connected_branches_and_short_width_outliers_stay_in_paint() {
        for bright in [false, true] {
            for branch in [false, true] {
                let (mut image, seed) = wide_band(0.0, false, false);
                if branch {
                    // A branch is part of the same ink region even when a
                    // detector reports just the long horizontal interval.
                    for y in 42..66 {
                        for x in 118..124 {
                            image.pixels[y * image.width + x] = [0.1; 3];
                        }
                    }
                } else {
                    // Fewer than 10% of sections widen. The previous
                    // percentile gate discarded these observations.
                    for y in 32..48 {
                        for x in 214..224 {
                            image.pixels[y * image.width + x] = [0.1; 3];
                        }
                    }
                }
                if bright {
                    for pixel in &mut image.pixels {
                        *pixel = pixel.map(|c| 1.0 - c);
                    }
                }
                let recovered = recover(&image, &[seed], &[]);
                assert!(
                    recovered.strokes.is_empty(),
                    "bright={bright}, branch={branch}"
                );
                assert!(recovered.updates.is_empty());
            }
        }
    }

    #[test]
    fn recovered_width_is_uniform_and_original_paint_is_removed_under_it() {
        let (image, seed) = wide_band(0.25, false, true);
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        let raster = render_recovered(240, 80, &recovered);
        let mut widths = Vec::new();
        let mut paint = image.clone();
        for &(i, color) in &recovered.updates {
            paint.pixels[i] = color;
        }
        for x in 20..220 {
            widths.push(
                (20..60)
                    .map(|y| raster.pixels()[y * 240 + x].alpha() as f32 / 255.0)
                    .sum::<f32>(),
            );
            for y in 36..44 {
                let expected = if (y as f32 + 0.5) < 40.25 {
                    [0.9, 0.8, 0.7]
                } else {
                    [0.6, 0.8, 1.0]
                };
                assert!(
                    distance(paint.pixels[y * 240 + x], expected) < 1e-5,
                    "source ink remains under the fitted stroke"
                );
            }
        }
        let min = widths.iter().copied().fold(f32::INFINITY, f32::min);
        let max = widths.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max - min < 0.05,
            "rendered width still wobbles: {min}..{max}"
        );
        assert!((0.5 * (min + max) - 12.0).abs() < 0.2);
    }

    #[test]
    fn mildly_shaded_core_uses_interior_colour_instead_of_the_extremum() {
        let (mut image, seed) = wide_band(0.0, false, false);
        for y in 0..80 {
            let d = y as f32 + 0.5 - 40.0;
            let coverage = (6.5 - d.abs()).clamp(0.0, 1.0);
            for x in 0..240 {
                image.pixels[y * 240 + x] =
                    image.pixels[y * 240 + x].map(|v| v + 0.005 * d * coverage);
            }
        }
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        for c in recovered.strokes[0].color {
            assert!((c - 0.1).abs() < 0.005, "biased core colour {c}");
        }
    }

    #[test]
    fn bright_band_between_different_paints_uses_the_same_joint_model() {
        let (mut image, seed) = wide_band(0.5, false, false);
        for color in &mut image.pixels {
            *color = color.map(|v| 1.0 - v);
        }
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        assert!(recovered.strokes[0]
            .color
            .iter()
            .all(|v| (*v - 0.9).abs() < 1e-5));
        assert!((recovered.strokes[0].width - 12.0).abs() < 0.15);
        let mut roles = crate::edge::classify(&image);
        let (paint, ink) = super::super::analyse(&image, &mut roles);
        // Here the detector fragments at the caps. The complete explicit
        // seed above is recoverable; partial detector runs must keep Paint.
        assert_eq!(ink.summary.recovered_boundary_strokes, 0);
        let middle = 40 * image.width + 120;
        assert_eq!(paint.pixels[middle], image.pixels[middle]);
        assert!(!ink.paint_ownership_mask[middle]);
    }

    #[test]
    fn closed_circular_band_has_one_width_and_no_artificial_storage_gap() {
        let mut image = Raster::blank(160, 160, [0.8; 3]);
        for y in 0..160 {
            for x in 0..160 {
                let r = (x as f32 + 0.5 - 80.0).hypot(y as f32 + 0.5 - 80.0);
                let coverage = (4.5 - (r - 50.0).abs()).clamp(0.0, 1.0);
                let paint = if r < 50.0 { 0.6 } else { 0.8 };
                image.pixels[y * 160 + x] = [0.1 * coverage + paint * (1.0 - coverage); 3];
            }
        }
        let mut points: Vec<_> = (0..360)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 360.0;
                [80.0 + 46.5 * a.cos(), 80.0 + 46.5 * a.sin()]
            })
            .collect();
        points.push(points[0]);
        let seed = SourceEdge {
            points,
            width: 1.2,
            role: "dark-boundary",
            width_samples: Vec::new(),
        };
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        let stroke = &recovered.strokes[0];
        assert!((stroke.width - 8.0).abs() < 0.2);
        for bright in [false, true] {
            let mut source = image.clone();
            if bright {
                for pixel in &mut source.pixels {
                    *pixel = pixel.map(|c| 1.0 - c);
                }
            }
            let (_, detected) = super::super::analyse(&source, &mut crate::edge::classify(&source));
            assert!(
                detected.summary.recovered_boundary_strokes > 0,
                "complete circles should be detected automatically, bright={bright}"
            );
        }
        assert!(stroke.path_data.as_ref().unwrap().ends_with(" Z"));
        let raster = render_recovered(160, 160, &recovered);
        for degrees in 0..360 {
            let a = degrees as f32 * std::f32::consts::TAU / 360.0;
            let x = (80.0 + 50.0 * a.cos()).floor() as usize;
            let y = (80.0 + 50.0 * a.sin()).floor() as usize;
            assert!(raster.pixels()[y * 160 + x].alpha() > 250);
            assert!(recovered.mask[y * 160 + x]);
        }
    }

    #[test]
    fn short_detector_fragments_join_only_over_source_supported_ink() {
        let (image, seed) = wide_band(0.0, false, false);
        let fragments: Vec<_> = seed
            .points
            .chunks(54)
            .map(|points| SourceEdge {
                points: points.to_vec(),
                ..seed.clone()
            })
            .collect();
        let recovered = recover(&image, &fragments, &[]);
        assert_eq!(
            recovered.strokes.len(),
            1,
            "four short fragments should become one editable stroke"
        );
        assert!(recovered.strokes[0].points.len() >= seed.points.len());
        let mut broken = image.clone();
        for y in 0..80 {
            broken.pixels[y * 240 + 120] = if y < 40 {
                [0.9, 0.8, 0.7]
            } else {
                [0.6, 0.8, 1.0]
            };
        }
        let recovered = recover(&broken, &[seed], &[]);
        assert_eq!(
            recovered.strokes.len(),
            2,
            "a one-pixel source gap must veto joining"
        );
        let raster = render_recovered(240, 80, &recovered);
        for y in 32..48 {
            assert_eq!(raster.pixels()[y * 240 + 120].alpha(), 0);
        }
    }

    #[test]
    fn interval_caps_do_not_bridge_an_authored_gap() {
        let (image, seed) = wide_band(0.0, true, false);
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 2);
        let raster = render_recovered(240, 80, &recovered);
        for y in 32..48 {
            for x in 116..124 {
                assert_eq!(
                    raster.pixels()[y * 240 + x].alpha(),
                    0,
                    "invented ink in the gap at {x}, {y}"
                );
                assert!(!recovered.mask[y * 240 + x]);
            }
        }
    }

    #[test]
    fn cactus_dotted_rim_is_recovered_at_the_transparent_left_edge() {
        let input = image::load_from_memory(include_bytes!("../data/cactus-circle.png"))
            .unwrap()
            .to_rgba8();
        let source = Raster::new(
            input.width() as usize,
            input.height() as usize,
            input
                .pixels()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ]
                })
                .collect(),
        );
        let matte = crate::chroma::AlphaMatte::from_u8(
            source.width,
            source.height,
            input.pixels().map(|p| p[3]).collect(),
        );
        let source = crate::chroma::prepare_source_alpha(&source, &matte);
        let strokes = recover_alpha_boundary(&source, &matte);
        for y in [41.0, 44.0] {
            assert!(
                strokes.iter().any(|s| s.role == "alpha-boundary-stroke"
                    && luma(s.color) < 0.6
                    && s.points.iter().any(|p| p.x < 19.0 && (p.y - y).abs() < 1.0)),
                "source dots disappeared near left edge y={y}"
            );
        }
    }

    #[test]
    fn alpha_rim_uses_continuous_mask_curves_and_keeps_black_silhouettes() {
        let width = 96;
        for mode in 0..6 {
            let has_rim = mode % 3 != 0;
            let mut image = Raster::blank(width, width, [0.0; 3]);
            let mut alpha = vec![0.0; width * width];
            for y in 0..width {
                for x in 0..width {
                    let radius = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                    if radius < 38.0 {
                        alpha[y * width + x] = 1.0;
                        let angle = (y as f32 + 0.5 - 48.0).atan2(x as f32 + 0.5 - 48.0);
                        let band = if has_rim && !(mode % 3 == 2 && (0.3..1.1).contains(&angle)) {
                            ((radius - 36.5) / 1.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        image.pixels[y * width + x] = [0.3 + 0.7 * band; 3];
                    }
                }
            }
            if mode >= 3 {
                for pixel in &mut image.pixels {
                    *pixel = pixel.map(|c| 1.0 - c);
                }
            }
            let matte = crate::chroma::AlphaMatte::new(width, width, alpha);
            let source = crate::chroma::prepare_source_alpha(&image, &matte);
            let recovered = recover_alpha_boundary(&source, &matte);
            if !has_rim {
                assert!(recovered.is_empty());
                continue;
            }
            assert!(!recovered.is_empty());
            let mut covered = [false; 72];
            for stroke in &recovered {
                assert!(stroke.path_data.as_ref().unwrap().contains(" C"));
                for pair in stroke.points.windows(2) {
                    for i in 0..=8 {
                        let t = i as f32 / 8.0;
                        let x = pair[0].x * (1.0 - t) + pair[1].x * t - 48.0;
                        let y = pair[0].y * (1.0 - t) + pair[1].y * t - 48.0;
                        let angle = y.atan2(x).rem_euclid(std::f32::consts::TAU);
                        covered[(angle / std::f32::consts::TAU * 72.0) as usize % 72] = true;
                    }
                }
            }
            if mode % 3 == 2 {
                assert!(!covered[7], "an intentional rim gap was bridged");
                assert!(covered[36], "the supported opposite rim disappeared");
                continue;
            }
            assert!(
                covered.iter().all(|&b| b),
                "alpha rim has angular gaps: {covered:?}"
            );
        }
    }

    #[test]
    fn paired_band_recovers_width_and_both_incident_colors() {
        let mut image = Raster::blank(64, 40, [1.0; 3]);
        let paints = [[0.9, 0.8, 0.7], [0.6, 0.8, 1.0]];
        for y in 0..image.height {
            for x in 0..image.width {
                let d = y as f32 + 0.5 - 20.5;
                let coverage = if (6..58).contains(&x) {
                    (2.1 - d.abs()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let paint = paints[usize::from(d >= 0.0)];
                image.pixels[y * 64 + x] = paint.map(|v| 0.1 * coverage + v * (1.0 - coverage));
            }
        }
        let recovered = recover(&image, &[edge(), edge()], &[]);
        assert_eq!(
            recovered.strokes.len(),
            1,
            "duplicate paired edges must share one owner"
        );
        assert!((recovered.strokes[0].width - 3.2).abs() < 0.15);
        for &(i, c) in &recovered.updates {
            let side = usize::from(i / 64 >= 20);
            assert!(
                distance(c, paints[side]) < 1e-5,
                "incident colors must not be averaged"
            );
        }
        assert!(recovered.mask[20 * 64 + 30]);
        let mut roles = crate::edge::classify(&image);
        let (_, ink) = super::super::analyse(&image, &mut roles);
        assert!(
            ink.summary.recovered_boundary_strokes > 0,
            "the detector must feed the model"
        );
    }

    #[test]
    fn steps_and_diffuse_shadows_remain_paint_owned() {
        for shadow in [false, true] {
            let mut image = Raster::blank(64, 40, [1.0; 3]);
            for y in 0..40 {
                let value = if shadow {
                    1.0 - 0.8 * (-0.5 * ((y as f32 - 20.0) / 4.0).powi(2)).exp()
                } else if y < 21 {
                    0.1
                } else {
                    1.0
                };
                for x in 0..64 {
                    image.pixels[y * 64 + x] = [value; 3];
                }
            }
            for bright in [false, true] {
                if bright {
                    for color in &mut image.pixels {
                        *color = color.map(|v| 1.0 - v);
                    }
                }
                assert!(
                    recover(&image, &[edge()], &[]).strokes.is_empty(),
                    "shadow={shadow}, bright={bright}"
                );
            }
        }
    }

    #[test]
    fn diagonal_band_has_continuous_underpaint_ownership() {
        let mut image = Raster::blank(64, 64, [1.0; 3]);
        for y in 0..64 {
            for x in 0..64 {
                let d = (y as f32 - x as f32) * std::f32::consts::FRAC_1_SQRT_2;
                let coverage = if (28..=98).contains(&(x + y)) {
                    (2.1 - d.abs()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                image.pixels[y * 64 + x] = [1.0 - 0.9 * coverage; 3];
            }
        }
        let mut diagonal = edge();
        // Sample just inside the antialiased cap; its retained Paint fits
        // within the bounded collar around the complete stroke model.
        diagonal.points = (15..49).map(|x| [x as f64 + 0.5; 2]).collect();
        let recovered = recover(&image, &[diagonal], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        for x in 20..44 {
            for y in x - 1..=x + 1 {
                assert!(recovered.mask[y * 64 + x], "unowned ink at ({x}, {y})");
            }
        }
    }
}
