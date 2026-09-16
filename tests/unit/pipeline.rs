use rayon::prelude::*;

mod tests {
    use super::*;
    use crate::geometry::fitted_alpha_contour_path_data;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn refinement_quality_does_not_reward_a_higher_resolution_copy() {
        let body = r##"<path d="M12 80L62 12L116 80Z" fill="#1762ba"/><circle cx="64" cy="52" r="17" fill="#fbc950" fill-opacity="0.55"/>"##;
        let fine = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96">{body}</svg>"#
        );
        let coarse = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12" viewBox="0 0 128 96">{body}</svg>"#
        );
        let background = [0.9, 0.95, 1.0];
        let source = render_svg_document_on(&fine, 128, 96, background).unwrap();
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 128,
            height: 96,
        };
        let coarse_preview = render_svg_document_on(&coarse, 16, 12, background).unwrap();
        let old_baseline = perceptual_score(&source, whole, &coarse_preview, whole);
        let comparison = compare_refinement(
            &source,
            None,
            &parse_svg_document(&coarse).unwrap(),
            &parse_svg_document(&fine).unwrap(),
            whole,
            whole,
            background,
        )
        .unwrap();
        assert!(
            old_baseline.combined > Config::default().adaptive_min_perceptual_gain,
            "coarse interpolation should reproduce the false gain: {old_baseline:?}"
        );
        assert!(comparison.boundary_matches);
        assert_eq!(comparison.baseline.combined, 0.0);
        assert_eq!(comparison.refined.combined, 0.0);
        assert!(!comparison.improves(&Config::default()));
    }

    #[test]
    fn refinement_quality_keeps_real_missing_line_improvements() {
        let base = r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12"/>"#;
        let fine = r#"<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96"><path d="M12 47H116V49H12Z" fill="black"/></svg>"#;
        let source = render_svg_document_on(fine, 128, 96, [1.0; 3]).unwrap();
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 128,
            height: 96,
        };
        let comparison = compare_refinement(
            &source,
            None,
            &parse_svg_document(base).unwrap(),
            &parse_svg_document(fine).unwrap(),
            whole,
            whole,
            [1.0; 3],
        )
        .unwrap();
        assert_eq!(comparison.refined.combined, 0.0);
        assert_eq!(comparison.baseline.missing_edge_fraction, 1.0);
        assert!(comparison.improves(&Config::default()));
    }

    #[test]
    fn refinement_quality_aligns_offset_crops_and_both_scale_axes() {
        let body = r##"<path d="M12 80L62 12L116 80Z" fill="#1762ba" fill-opacity="0.6"/><path d="M18 40H110" fill="none" stroke="#c82030" stroke-width="2"/>"##;
        let fine = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96">{body}</svg>"#
        );
        let coarse = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12" viewBox="0 0 128 96">{body}</svg>"#
        );
        // Nonzero source origin; the child uses different X/Y processing scales.
        let child = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="30" height="11" viewBox="28 20 60 44" preserveAspectRatio="none">{body}</svg>"#
        );
        let expanded = SourceRect {
            x: 28,
            y: 20,
            width: 60,
            height: 44,
        };
        let core = SourceRect {
            x: 32,
            y: 24,
            width: 52,
            height: 36,
        };
        for background in [[0.0; 3], [1.0; 3]] {
            let source = render_svg_document_on(&fine, 128, 96, background).unwrap();
            let comparison = compare_refinement(
                &source,
                None,
                &parse_svg_document(&coarse).unwrap(),
                &parse_svg_document(&child).unwrap(),
                core,
                expanded,
                background,
            )
            .unwrap();
            assert!(comparison.boundary_matches);
            assert_eq!(comparison.core, core);
            assert_eq!(comparison.baseline.combined, 0.0);
            assert_eq!(comparison.refined.combined, 0.0);
            assert!(!comparison.improves(&Config::default()));
        }
    }

    #[test]
    fn refinement_quality_still_rejects_a_changed_crop_join() {
        let base = r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12"/>"#;
        let child = r#"<svg xmlns="http://www.w3.org/2000/svg" width="60" height="44"><path d="M0 0H60V44H0Z" fill="black"/></svg>"#;
        let source = Raster::blank(128, 96, [0.0; 3]);
        let expanded = SourceRect {
            x: 28,
            y: 20,
            width: 60,
            height: 44,
        };
        let core = SourceRect {
            x: 32,
            y: 24,
            width: 52,
            height: 36,
        };
        let comparison = compare_refinement(
            &source,
            None,
            &parse_svg_document(base).unwrap(),
            &parse_svg_document(child).unwrap(),
            core,
            expanded,
            [1.0; 3],
        )
        .unwrap();
        assert!(comparison.gain() > 1.0);
        assert!(!comparison.boundary_matches);
        assert!(!comparison.improves(&Config::default()));
    }

    #[test]
    #[ignore = "source-resolution scanned drawing regression; run explicitly"]
    fn scanned_annotations_retain_ink_through_the_common_pipeline() {
        let image = image::load_from_memory(include_bytes!("../data/booster-annotations.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            image.width() as usize,
            image.height() as usize,
            image
                .pixels()
                .map(|p| p.0.map(|c| c as f32 / 255.0))
                .collect(),
        );
        let config = Config {
            adaptive_refinement: false,
            ..Config::default()
        };
        let core = vectorize_processing(source.clone(), None, false, [1.0; 3], &config).unwrap();
        assert!(!core.labels.is_empty());
        let rendered =
            render_svg_document_on(&core.document, source.width, source.height, [1.0; 3]).unwrap();
        let (mut ink_error, mut ink, mut dark, mut missing) = (0.0, 0usize, 0usize, 0usize);
        for (a, b) in source.pixels.iter().zip(&rendered.pixels) {
            if a[0] < 240.0 / 255.0 {
                ink += 1;
                ink_error += (a[0] - b[0]).abs();
            }
            if a[0] < 128.0 / 255.0 {
                dark += 1;
                missing += usize::from(b[0] > 192.0 / 255.0);
            }
        }
        assert!(
            ink_error / (ink.max(1) as f32) < 0.2,
            "ink error {}",
            ink_error / ink.max(1) as f32
        );
        assert!(
            (missing as f32) / (dark.max(1) as f32) < 0.12,
            "missing {missing}/{dark}"
        );
    }

    #[test]
    fn neutral_and_coloured_parallel_lines_use_normal_region_geometry() {
        let config = Config {
            adaptive_refinement: false,
            ..Config::default()
        };
        for ink in [[0.0; 3], [0.6, 0.0, 0.0]] {
            let mut source = Raster::blank(128, 64, [1.0; 3]);
            for y in [20, 28] {
                for yy in y..y + 3 {
                    for x in 12..116 {
                        source.pixels[yy * 128 + x] = ink;
                    }
                }
            }
            let core = vectorize_processing(source, None, false, [1.0; 3], &config).unwrap();
            assert!(!core.labels.is_empty());
            assert!(!core.document.contains("<mask") && !core.document.contains("<image"));
            let rendered = render_svg_document_on(&core.document, 128, 64, [1.0; 3]).unwrap();
            for x in 20..108 {
                assert!(rendered.pixels[21 * 128 + x][1] < 0.3);
                assert!(rendered.pixels[29 * 128 + x][1] < 0.3);
                assert!(rendered.pixels[25 * 128 + x][1] > 0.9);
            }
        }
    }

    #[test]
    fn variable_opacity_elsewhere_does_not_erase_opaque_ink() {
        let (w, h) = (88, 64);
        let mut source = Raster::blank(w, h, [1.0; 3]);
        let mut opacity = vec![1.0; w * h];
        for y in 8..56 {
            for x in 6..30 {
                source.pixels[y * w + x] = [0.1, 0.4, 0.8];
                opacity[y * w + x] = 0.3 + 0.4 * (x - 6) as f32 / 23.0;
            }
        }
        for y in 10..54 {
            for x in 63..65 {
                source.pixels[y * w + x] = [0.0; 3];
            }
        }
        let matte = AlphaMatte::new(w, h, opacity);
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                vectorize_processing(
                    source,
                    Some(&matte),
                    true,
                    [1.0; 3],
                    &Config {
                        adaptive_refinement: false,
                        ..Config::default()
                    },
                )
                .unwrap()
            });
        assert!(!result.document.contains("<mask"));
        let tree = parse_svg_document(&result.document).unwrap();
        let mut image = resvg::tiny_skia::Pixmap::new(w as u32, h as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut image.as_mut(),
        );
        for y in 14..50 {
            assert!(
                (62..66).any(|x| image.pixels()[y * w + x].red() < 50),
                "opaque ink lost at row {y}"
            );
        }
        for y in 12..52 {
            for x in 10..26 {
                let actual = image.pixels()[y * w + x];
                assert!(
                    (actual.alpha() as f32 / 255.0 - matte.get(y * w + x)).abs() < 4.0 / 255.0,
                    "authored opacity changed at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn reported_remojii_top_objects_join_adjacent_faces_in_the_complete_pipeline() {
        // These are the native support pixels of objects 2..=50, top down,
        // from the user's 665770-byte SVG. Full-frame context is necessary:
        // cropping splits the incident paint owners and changes their sizes.
        #[derive(serde::Deserialize)]
        struct ReportedObject {
            rank: usize,
            pixels: Vec<usize>,
        }
        let objects: Vec<ReportedObject> =
            serde_json::from_str(include_str!("../data/remojii-top-objects.json")).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("remojii.png");
        fs::write(
            &input,
            include_bytes!("../data/remojii-top-objects-source.png"),
        )
        .unwrap();
        let config = Config {
            adaptive_refinement: false,
            rayon_threads: 2,
            ..Config::default()
        };
        let (decoded, alpha) = SourceRaster::load_with_alpha(
            &input,
            config.maximum_input_dimension,
            config.maximum_input_pixels,
            config.maximum_decode_bytes,
        )
        .unwrap();
        let matte = AlphaMatte::from_u8(decoded.width, decoded.height, alpha.unwrap());
        let backing = chroma::select_alpha_backing(&decoded, &matte);
        let source = chroma::prepare_compact_source_alpha(&decoded, &matte);
        let (processing, alpha) = resize_processing(&source, Some(&matte), true, 1254);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let result = pool
            .install(|| vectorize_processing(processing, alpha.as_ref(), true, backing, &config))
            .unwrap();
        assert!(result.paint_order.line_regions > 0);
        assert!(result.paint_order.changed_ranks > 0);
        assert!(result.paint_order.accepted, "{:?}", result.paint_order);
        #[derive(serde::Deserialize)]
        struct ReportedInk {
            pixels: Vec<usize>,
            neighbor_pixel: usize,
        }
        let reported: Vec<ReportedInk> = serde_json::from_str(include_str!(
            "../data/remojii-reported-material-regions.json"
        ))
        .unwrap();
        for (case, ink) in reported.iter().enumerate() {
            let ink_owner = result.labels[ink.neighbor_pixel];
            assert!(
                ink.pixels.iter().all(|&i| result.labels[i] == ink_owner),
                "reported material case {case} must join its adjacent face"
            );
        }
        let mut areas = std::collections::HashMap::<u32, usize>::new();
        for &label in &result.labels {
            *areas.entry(label).or_default() += 1;
        }
        // These two reported faces formerly emitted the same outline four
        // and six times for weak residual colour corrections.
        fn count_face_paths(group: &resvg::usvg::Group, x: f32, y: f32) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => count_face_paths(group, x, y),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        usize::from(
                            (b.x() - x).abs() < 2.0
                                && (b.y() - y).abs() < 2.0
                                && b.width() < 45.0
                                && b.height() < 45.0,
                        )
                    }
                    _ => 0,
                })
                .sum()
        }
        let tree = parse_svg_document(&result.document).unwrap();
        fn shell_contours(group: &resvg::usvg::Group) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => shell_contours(group),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        if (b.x() - 155.977).abs() < 2.0 && (b.y() - 63.639).abs() < 2.0 {
                            assert_eq!(
                                path.data()
                                    .segments()
                                    .filter(|s| matches!(
                                        s,
                                        resvg::tiny_skia::PathSegment::MoveTo(_)
                                    ))
                                    .count(),
                                1,
                                "covered shell pattern must not remain as holes in the lower paint"
                            );
                            1
                        } else {
                            0
                        }
                    }
                    _ => 0,
                })
                .sum()
        }
        assert_eq!(shell_contours(tree.root()), 1);
        assert!(result.geometry.covered_holes_removed > 0);

        fn shadow_copies(
            group: &resvg::usvg::Group,
            x: f32,
            y: f32,
            width: f32,
            height: f32,
        ) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => shadow_copies(group, x, y, width, height),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        usize::from(
                            (b.x() - x).abs() < 2.0
                                && (b.y() - y).abs() < 2.0
                                && b.width() < width
                                && b.height() < height,
                        )
                    }
                    _ => 0,
                })
                .sum()
        }
        assert_eq!(
            shadow_copies(tree.root(), 368.57, 994.89, 65.0, 80.0),
            1,
            "the book shadow must have one paint, not repeated residual geometry"
        );

        assert_eq!(
            shadow_copies(tree.root(), 733.65, 639.28, 120.0, 60.0),
            1,
            "the image-right eyelid shadow must have one paint"
        );

        fn assert_paint_has_no_auxiliary_strokes(group: &resvg::usvg::Group) {
            for node in group.children() {
                match node {
                    resvg::usvg::Node::Group(group) => assert_paint_has_no_auxiliary_strokes(group),
                    resvg::usvg::Node::Path(path) => assert!(
                        path.stroke().is_none(),
                        "paint face emitted an auxiliary seam stroke"
                    ),
                    _ => {}
                }
            }
        }
        let paint_group = tree
            .root()
            .children()
            .iter()
            .find_map(|n| match n {
                resvg::usvg::Node::Group(g) if g.id() == "paint-layer" => Some(g),
                _ => None,
            })
            .unwrap();
        assert_paint_has_no_auxiliary_strokes(paint_group);
        for (x, y) in [(863.93, 407.78), (925.97, 825.06)] {
            assert_eq!(
                count_face_paths(tree.root(), x, y),
                1,
                "reported face at {x},{y} retained duplicate correction layers"
            );
        }
        let fragments: Vec<Vec<usize>> =
            serde_json::from_str(include_str!("../data/remojii-third-ninth-regions.json")).unwrap();
        for pixels in fragments {
            let owner = result.labels[pixels[0]];
            assert!(
                pixels.iter().all(|&i| result.labels[i] == owner) && areas[&owner] > pixels.len(),
                "reported third/ninth fragment retained an independent owner"
            );
        }
        let sixth: Vec<usize> =
            serde_json::from_str(include_str!("../data/remojii-sixth-black-region.json")).unwrap();
        assert!(
            sixth
                .iter()
                .all(|&i| areas[&result.labels[i]] > sixth.len()),
            "sixth black face still has an independent owner"
        );
        for object in objects {
            assert!(
                object
                    .pixels
                    .iter()
                    .all(|&i| areas[&result.labels[i]] > object.pixels.len()),
                "reported object {} still has its own small owner",
                object.rank
            );
        }
        assert!(!result.document.contains("<mask"));
        assert!(!result.document.contains("fill-opacity=\"0\""));
    }

    #[test]
    fn remojii_alpha_boundary_has_few_nodes_along_the_upper_left_rim() {
        let input = image::load_from_memory(include_bytes!("../data/remojii-rim-alpha.png"))
            .unwrap()
            .to_luma8();
        let matte = AlphaMatte::from_u8(
            input.width() as usize,
            input.height() as usize,
            input.into_raw(),
        );
        let path = matte
            .isocontours(0.5)
            .iter()
            .map(|contour| fitted_alpha_contour_path_data(contour))
            .collect::<String>();
        let mut tokens = path.split_whitespace();
        let mut rim_nodes = 0;
        while let Some(command) = tokens.next() {
            let count = match command {
                "M" | "L" => 2,
                "C" => 6,
                "Z" => 0,
                _ => panic!("unexpected {command}"),
            };
            let values: Vec<f32> = tokens
                .by_ref()
                .take(count)
                .map(|s| s.parse().unwrap())
                .collect();
            if count > 0 {
                let (x, y) = (values[count - 2], values[count - 1]);
                rim_nodes += usize::from((80.0..420.0).contains(&x) && (50.0..350.0).contains(&y));
            }
        }
        assert!(
            rim_nodes <= 24,
            "smooth upper-left rim retained {rim_nodes} nodes"
        );
    }

    #[test]
    fn alpha_weighted_resize_preserves_visible_colour_in_base_and_crops() {
        let source = SourceRaster::from_rgb8_fn(128, 64, |i| {
            if i % 2 == 0 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            }
        });
        let matte = AlphaMatte::from_u8(
            128,
            64,
            (0..128 * 64)
                .map(|i| if i % 2 == 0 { 255 } else { 1 })
                .collect(),
        );
        let crop = source.crop(0, 0, 128, 64);
        let (base, alpha) = resize_processing(&source, Some(&matte), true, 64);
        let (child, _) = resize_processing(&crop, Some(&matte), true, 64);
        for image in [&base, &child] {
            let rgb = image.get(32, 16);
            assert!((rgb[0] - 255.0 / 256.0).abs() < 0.001, "{rgb:?}");
            assert!((rgb[2] - 1.0 / 256.0).abs() < 0.001, "{rgb:?}");
        }
        assert!((alpha.as_ref().unwrap().get(16 * 64 + 32) - 128.0 / 255.0).abs() < 0.005);
        let core =
            vectorize_processing(base, alpha.as_ref(), true, [1.0; 3], &Config::default()).unwrap();
        let tree = parse_svg_document(&core.document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let pixel = pixmap.pixels()[16 * 64 + 32];
        assert!(pixel.alpha() > 0);
        assert!(f32::from(pixel.red()) / f32::from(pixel.alpha()) > 0.98);
        assert!(f32::from(pixel.blue()) / f32::from(pixel.alpha()) < 0.02);
    }

    #[test]
    fn small_inputs_select_native_dimensions_without_a_probe() {
        let source = Raster::blank(1024, 1024, [0.5; 3]);
        let config = Config::default();
        let selection = select_dimension(&source, &config);
        assert_eq!(
            selection.selected_dimension,
            estimate_dimension(&source, &config).selected_dimension
        );
        assert_eq!(selection.probe_width, 0);
        let limited = Config {
            maximum_dimension: 64,
            ..config
        };
        assert_eq!(select_dimension(&source, &limited).selected_dimension, 64);
    }

    #[test]
    fn native_vector_join_does_not_inherit_coarse_preview_blur() {
        let document = r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M0 8H32V8.5H0Z"/><circle cx="16" cy="16" r="5"/></svg>"#;
        let tree = parse_svg_document(document).unwrap();
        let coarse = render_svg_document_on(document, 32, 32, [1.0; 3]).unwrap();
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 128,
            height: 128,
        };
        let expanded = SourceRect {
            x: 32,
            y: 28,
            width: 64,
            height: 64,
        };
        let core = SourceRect {
            x: 40,
            y: 36,
            width: 48,
            height: 52,
        };
        let native = render_svg_tree_on(
            &tree,
            64,
            64,
            resvg::tiny_skia::Transform::from_row(4.0, 0.0, 0.0, 4.0, -32.0, -28.0),
            [1.0; 3],
        )
        .unwrap();
        assert!(!crate::adaptive::refinement_boundary_matches(
            &coarse, &native, whole, whole, core, expanded
        ));
        assert!(crate::adaptive::refinement_boundary_matches(
            &native, &native, expanded, whole, core, expanded
        ));
        let mut damaged = native.clone();
        damaged.pixels[(core.y - expanded.y) * 64 + 25] = [0.0; 3];
        assert!(!crate::adaptive::refinement_boundary_matches(
            &native, &damaged, expanded, whole, core, expanded
        ));
    }

    #[test]
    fn factored_separator_stays_behind_foreground_and_leaves_background_clear() {
        let mut document: Document = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect x="14" y="6" width="4" height="20" fill="#202020"/></svg>"##.into();
        crate::separators::prepend(
            &mut document,
            &[crate::separators::Separator {
                rect: [0.0, 32.0, 128.0, 6.0],
                color: [1.0; 3],
                opacity: 1.0,
            }],
            [0.25, 0.25],
        );
        let raster = render_svg_document_on(&document, 32, 32, [0.0, 1.0, 0.0]).unwrap();
        assert_eq!(raster.get(0, 8), [1.0; 3]);
        assert_eq!(raster.get(16, 8), [32.0 / 255.0; 3]);
        assert_eq!(raster.get(0, 0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn exact_final_paint_merge_compacts_adjacent_equal_owners() {
        let source = Raster::blank(2, 1, [0.25, 0.5, 0.75]);
        let mut segmentation = Segmentation {
            width: 2,
            height: 1,
            labels: vec![0, 1],
            paint_keys: vec![0, 1],
            paint_samples: vec![true; 2],
            canonical: source.clone(),
            regions: vec![
                crate::segment::RegionStats {
                    id: 0,
                    area: 1,
                    min_x: 0,
                    min_y: 0,
                    max_x: 1,
                    max_y: 1,
                    mean_rgb: [0.25, 0.5, 0.75],
                    mean_lab: rgb_to_oklab([0.25, 0.5, 0.75]),
                },
                crate::segment::RegionStats {
                    id: 1,
                    area: 1,
                    min_x: 1,
                    min_y: 0,
                    max_x: 2,
                    max_y: 1,
                    mean_rgb: [0.25, 0.5, 0.75],
                    mean_lab: rgb_to_oklab([0.25, 0.5, 0.75]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let mut paints = vec![
            Paint::Solid {
                color: [0.25, 0.5, 0.75],
            },
            // Different fitting values, identical serialized #4080bf.
            Paint::Solid {
                color: [0.2501, 0.5001, 0.7501],
            },
        ];
        assert_eq!(
            merge_exact_final_paints(&source, &mut segmentation, &mut paints),
            1
        );
        assert_eq!(segmentation.labels, vec![0, 0]);
        assert_eq!(segmentation.regions.len(), 1);
        assert_eq!(paints.len(), 1);
    }

    #[test]
    fn complexity_probe_distinguishes_sparse_and_dense_edges() {
        let width = 128;
        let height = 128;
        let flat = Raster::blank(width, height, [0.5; 3]);
        let mut split = flat.clone();
        for y in 0..height {
            for x in width / 2..width {
                split.pixels[y * width + x] = [0.9; 3];
            }
        }
        let mut tiled = flat.clone();
        for y in 0..height {
            for x in 0..width {
                if (x / 4 + y / 4) % 2 == 0 {
                    tiled.pixels[y * width + x] = [0.9; 3];
                }
            }
        }
        let config = Config {
            maximum_dimension: 128,
            auto_minimum_dimension: 64,
            auto_maximum_dimension: 128,
            ..Config::default()
        };
        let flat_probe = estimate_dimension(&flat, &config);
        let split_probe = estimate_dimension(&split, &config);
        let tiled_probe = estimate_dimension(&tiled, &config);
        assert_eq!(flat_probe.edge_density, 0.0);
        assert!(split_probe.edge_density > flat_probe.edge_density);
        assert!(tiled_probe.edge_density > split_probe.edge_density);
        assert!(tiled_probe.complexity > split_probe.complexity);
        assert!(split_probe.complexity > flat_probe.complexity);
    }

    #[test]
    fn complexity_probe_honours_the_general_maximum_dimension() {
        let image = Raster::blank(640, 480, [0.5; 3]);
        let config = Config {
            maximum_dimension: 192,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            ..Config::default()
        };
        assert_eq!(estimate_dimension(&image, &config).selected_dimension, 192);
    }

    #[test]
    fn refinement_queue_preserves_order_and_limits_nested_jobs() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for limit in [1, 2, 4, 8, 10] {
                    let active = AtomicUsize::new(0);
                    let peak = AtomicUsize::new(0);
                    let tasks: Vec<_> = (0usize..37).collect();
                    let outcomes = bounded_map(&tasks, limit, |&i| {
                        let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        let sum: usize = (0..1024).into_par_iter().map(|j| i + j).sum();
                        active.fetch_sub(1, Ordering::SeqCst);
                        sum
                    });
                    assert_eq!(
                        outcomes,
                        tasks
                            .iter()
                            .map(|i| i * 1024 + 1023 * 1024 / 2)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(active.load(Ordering::SeqCst), 0);
                    assert!(peak.load(Ordering::SeqCst) <= limit);
                }
            });
        }
    }

    #[test]
    fn default_worker_count_uses_half_the_cpus_capped_at_ten() {
        assert_eq!(default_execution_thread_count(1), 1);
        assert_eq!(default_execution_thread_count(2), 1);
        assert_eq!(default_execution_thread_count(3), 1);
        assert_eq!(default_execution_thread_count(4), 2);
        assert_eq!(default_execution_thread_count(6), 3);
        assert_eq!(default_execution_thread_count(8), 4);
        assert_eq!(default_execution_thread_count(20), 10);
        assert_eq!(default_execution_thread_count(64), 10);
    }

    #[test]
    fn conversion_writes_only_the_named_svg() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("picvec-contract-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("chosen-name.svg");
        let mut raster = Raster::blank(32, 24, [0.92, 0.92, 0.96]);
        for y in 5..19 {
            for x in 6..26 {
                let amount = (x - 6) as f32 / 19.0;
                raster.pixels[y * 32 + x] = [0.15 + 0.65 * amount, 0.12, 0.72 - 0.4 * amount];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.quality.is_none());
        assert!(!summary.source_alpha.detected);
        assert!(!summary.chroma_key.enabled);
        assert!(summary.adaptive_refinement.enabled);
        assert_eq!(summary.adaptive_refinement.accepted_regions, 0);
        assert_eq!(summary.adaptive_refinement.source_scale, 1.0);
        let files: HashSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files,
            HashSet::from(["input.png".to_string(), "chosen-name.svg".to_string()])
        );
        let document = fs::read_to_string(&output).unwrap();
        assert!(document.starts_with("<?xml"));
        assert!(document.contains("<svg"));
        assert!(!document.contains("silhouette\""));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn chroma_key_removes_outer_and_enclosed_background_but_keeps_white_subject() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut raster = Raster::blank(64, 64, [0.0, 1.0, 0.0]);
        // An opaque white foreground detail verifies that white is not
        // confused with the keyed background.
        for y in 3..13 {
            for x in 3..13 {
                raster.pixels[y * 64 + x] = [1.0; 3];
            }
        }
        // Red ring around a disconnected island of keyed background.
        for y in 14..50 {
            for x in 14..50 {
                raster.pixels[y * 64 + x] = [1.0, 0.0, 0.0];
            }
        }
        // One-pixel 50% coverage shoulder, as produced by raster
        // antialiasing of red over the green backing.
        for position in 14..50 {
            raster.pixels[14 * 64 + position] = [0.5, 0.5, 0.0];
            raster.pixels[49 * 64 + position] = [0.5, 0.5, 0.0];
            raster.pixels[position * 64 + 14] = [0.5, 0.5, 0.0];
            raster.pixels[position * 64 + 49] = [0.5, 0.5, 0.0];
        }
        for y in 26..38 {
            for x in 26..38 {
                raster.pixels[y * 64 + x] = [0.0, 1.0, 0.0];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                remove_chroma_key_background: true,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.chroma_key.enabled);
        assert!(summary.chroma_key.detected);
        assert_eq!(summary.chroma_key.key_color, Some([0, 255, 0]));
        assert!(summary.chroma_key.removed_regions >= 2);

        let document = fs::read_to_string(&output).unwrap();
        for (_, suffix) in document.match_indices('#') {
            let Some(hex) = suffix.get(1..7) else {
                continue;
            };
            let Ok(color) = u32::from_str_radix(hex, 16) else {
                continue;
            };
            let red = (color >> 16) & 0xff;
            let green = (color >> 8) & 0xff;
            let blue = color & 0xff;
            assert!(
                green < 200 || red >= 80 || blue >= 80,
                "key-coloured antialias paint leaked into SVG: #{hex}"
            );
        }
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let alpha = |x: usize, y: usize| pixmap.pixels()[y * 64 + x].alpha();
        assert_eq!(alpha(2, 2), 0, "outer key should be transparent");
        assert!(
            alpha(8, 8) > 240,
            "white foreground should remain opaque: {}",
            alpha(8, 8)
        );
        assert!(alpha(18, 18) > 240, "red foreground should remain opaque");
        assert_eq!(alpha(32, 32), 0, "enclosed key should be transparent");

        let opaque_output = directory.path().join("opaque.svg");
        let opaque_summary = vectorize(
            &input,
            &opaque_output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(!opaque_summary.source_alpha.detected);
        assert!(!opaque_summary.chroma_key.enabled);
        let opaque_document = fs::read_to_string(&opaque_output).unwrap();
        let opaque_tree = parse_svg_document(&opaque_document).unwrap();
        let mut opaque_pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &opaque_tree,
            resvg::tiny_skia::Transform::identity(),
            &mut opaque_pixmap.as_mut(),
        );
        assert!(
            opaque_pixmap.pixels()[2 * 64 + 2].alpha() > 240,
            "opaque chroma input must remain opaque without the option"
        );
    }

    #[test]
    fn source_alpha_is_removed_without_the_chroma_option() {
        use image::{ImageBuffer, Rgba};

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut image = ImageBuffer::from_pixel(64, 64, Rgba([17_u8, 31, 47, 0]));
        // Opaque black and white details must both survive regardless of the
        // temporary saturated backing selected for RGB vectorization.
        for y in 4..14 {
            for x in 4..14 {
                image.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        for y in 16..50 {
            for x in 16..50 {
                image.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        for y in 27..39 {
            for x in 27..39 {
                image.put_pixel(x, y, Rgba([17, 31, 47, 0]));
            }
        }
        image.save(&input).unwrap();

        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.source_alpha.detected);
        assert!(summary.source_alpha.temporary_backing_color.is_some());
        assert_eq!(summary.source_alpha.quantization_bits, 8);
        assert_eq!(summary.source_alpha.mask_paths, 0);
        assert!(summary.source_alpha.removed_regions >= 2);
        assert!(!summary.chroma_key.enabled);

        let document = fs::read_to_string(&output).unwrap();
        assert!(!document.contains("source-alpha-clip"));
        assert!(!document.contains("source-alpha-mask"));
        assert!(!document.contains("<mask"));
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let alpha = |x: usize, y: usize| pixmap.pixels()[y * 64 + x].alpha();
        assert_eq!(alpha(2, 2), 0);
        assert!(alpha(8, 8) > 240, "opaque white should remain");
        assert!(alpha(20, 20) > 240, "opaque black should remain");
        assert_eq!(
            alpha(32, 32),
            0,
            "enclosed source alpha should remain clear"
        );
    }

    #[test]
    fn shaded_shoulder_has_no_spurs_at_paint_junctions() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("shoulder.png");
        let output = directory.path().join("shoulder.svg");
        fs::write(&input, include_bytes!("../data/shoulder-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 802,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2508,
            3208,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        let mut positions = Vec::new();
        for x in 195..239 {
            let values: Vec<_> = (345 * 4..385 * 4)
                .map(|y| rendered.get(x * 4, y)[1])
                .collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len()).find(|&i| values[i] > 0.4).unwrap();
            let t = (0.4 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push((crossing as f32 - 1.0 + t) * 0.25);
        }
        let curvature: Vec<_> = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .collect();
        assert!(
            curvature.iter().all(|&v| v < 0.6),
            "rim spikes: {curvature:?}"
        );
        assert!(curvature.iter().sum::<f32>() / (curvature.len() as f32) < 0.12);
    }

    #[test]
    fn round_window_control_does_not_reintroduce_polygonal_outer_ink() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("window.png");
        let output = directory.path().join("window.svg");
        fs::write(&input, include_bytes!("../data/round-window-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 705,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2820,
            2752,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // These exterior samples were dark protrusions of the polygonal rim.
        // Check serialized Paint plus residual strokes, not just ellipse fits.
        for (x, y) in [(135, 100), (170, 116), (174, 112)] {
            let p = rendered.get(x * 4, y * 4);
            assert!(p[2] > 0.5 && p[0] < 0.2, "outer rim at {x},{y}: {p:?}");
        }
        let fill = rendered.get(157 * 4, 98 * 4);
        assert!(fill[0] > 0.9 && fill[1] > 0.5 && fill[2] < 0.2);
    }

    #[test]
    fn third_round_button_has_a_smooth_inner_rim_in_the_final_svg() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("buttons.png");
        let output = directory.path().join("buttons.svg");
        fs::write(&input, include_bytes!("../data/round-buttons-source.png")).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 718,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.svg.outline_bands >= 3);
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2872,
            2764,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // Replacing a band's interior must retain the original exterior
        // coverage. This blue backdrop sample previously became a white gap.
        let backdrop = rendered.get(794, 348);
        assert!(
            backdrop[0] < 0.5 && backdrop[2] < 0.85,
            "missing band underpaint: {backdrop:?}"
        );
        let mut positions = Vec::new();
        for x in 208..233 {
            let values: Vec<_> = (68 * 4..98 * 4)
                .map(|y| rendered.get(x * 4, y)[1])
                .collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len())
                .find(|&i| values[i] > 0.55)
                .unwrap();
            let t = (0.55 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push(68.0 + (crossing as f32 - 1.0 + t) * 0.25);
        }
        let curvature: Vec<_> = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .collect();
        assert!(curvature.iter().sum::<f32>() / (curvature.len() as f32) < 0.12);
        assert!(curvature.iter().all(|&v| v < 0.4));
    }

    #[test]
    fn pie_highlight_does_not_acquire_an_orange_notch() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("pie.png");
        let output = directory.path().join("pie.svg");
        fs::write(&input, include_bytes!("../data/pie-highlight-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 710,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let rendered = render_svg_document_on(&document, 564, 710, [1.0; 3]).unwrap();
        // These bright native samples were assigned to the orange interior
        // during antialias cleanup. Check the final emitted image, not just
        // the intermediate partition or the number of simplified contours.
        for y in [462, 463] {
            let pixel = rendered.get(114, y);
            assert!(
                pixel[1] > 0.8 && pixel[2] > 0.6,
                "highlight overwritten at (114,{y}): {pixel:?}"
            );
        }
    }

    #[test]
    fn wifi_inner_rim_is_visibly_smooth_after_final_serialization() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("wifi.png");
        let output = directory.path().join("wifi.svg");
        fs::write(&input, include_bytes!("../data/wifi-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 310,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        // Seam underpaint can make an extra outline band unnecessary. Check
        // the rendered rim's colour, contrast and roughness below instead of
        // requiring a particular number of reconstruction primitives.
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            1240,
            1060,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // Locate the rim in the source after removing its green key. A fixed
        // y=203 dark-colour assertion instead darkens the source's cyan fill
        // when a different perceptual partition moves the fitted boundary.
        let source_image = Raster::from_dynamic(
            &image::load_from_memory(include_bytes!("../data/wifi-source.png")).unwrap(),
        );
        let key = chroma::detect(&source_image).unwrap();
        let matte = chroma::pull_matte(&source_image, key);
        let foreground = chroma::separate_foreground(&source_image, &matte, key.sampled);
        let reference = chroma::composite_over(&foreground, &matte, [1.0; 3]);
        let luminance = |p: [f32; 3]| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722;
        let mut rim_colour_error = 0.0;
        for x in 148..=154 {
            let y = (200..=204)
                .min_by(|&a, &b| {
                    luminance(reference.get(x, a)).total_cmp(&luminance(reference.get(x, b)))
                })
                .unwrap();
            let expected = reference.get(x, y);
            // Allow at most one source pixel of subpixel boundary movement.
            let colour_error =
                |pixel| crate::color::delta_e_ok(rgb_to_oklab(expected), rgb_to_oklab(pixel));
            let actual = ((y - 1) * 4..=(y + 1) * 4)
                .map(|row| rendered.get(x * 4, row))
                .min_by(|a, b| colour_error(*a).total_cmp(&colour_error(*b)))
                .unwrap();
            let error = colour_error(actual);
            assert!(
                error < 12.0,
                "source rim colour at {x},{y}: {error}, {actual:?}"
            );
            rim_colour_error += error;
            let source_contrast = luminance(reference.get(x, 205)) - luminance(expected);
            // Colour matching may select a partially covered rim pixel. Use
            // the actual darkest core, not that antialias sample, for contrast.
            let core_luminance = ((y - 1) * 4..=(y + 1) * 4)
                .map(|row| luminance(rendered.get(x * 4, row)))
                .fold(f32::INFINITY, f32::min);
            let rendered_contrast = luminance(rendered.get(x * 4, 205 * 4)) - core_luminance;
            assert!(
                rendered_contrast >= source_contrast * 0.75,
                "dot rim contrast at {x}: {rendered_contrast} vs source {source_contrast}"
            );
        }
        assert!(
            rim_colour_error / 7.0 < 6.0,
            "mean rim colour error: {}",
            rim_colour_error / 7.0
        );
        // Track the visible inner edge of the upper side of the middle arc.
        // The previous final SVG measured 0.093 px of second-difference
        // roughness here at 4x; checking master counts did not detect it.
        let mut positions = Vec::new();
        for x in 95..206 {
            let expected = 205.0 - (130.0_f32.powi(2) - (x as f32 - 150.0).powi(2)).sqrt();
            let a = ((expected - 12.0) * 4.0) as usize;
            let b = ((expected + 12.0) * 4.0) as usize;
            let values: Vec<_> = (a..b).map(|y| luminance(rendered.get(x * 4, y))).collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len())
                .find(|&i| values[i] > 0.42)
                .expect("unbroken visible rim");
            let t = (0.42 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push((a as f32 + crossing as f32 - 1.0 + t) * 0.25);
        }
        let roughness = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .sum::<f32>()
            / (positions.len() - 2) as f32;
        assert!(roughness < 0.08, "visible inner rim roughness: {roughness}");
        let native = render_svg_document_on(&document, 310, 265, [1.0; 3]).unwrap();
        let source = image::load_from_memory(include_bytes!("../data/wifi-source.png"))
            .unwrap()
            .to_rgb8();
        let mut error = 0.0;
        let mut count = 0;
        for (i, pixel) in source.pixels().enumerate() {
            if pixel[2] as i16 > pixel[1] as i16 + 5 && pixel[2] as i16 > pixel[0] as i16 + 5 {
                error += crate::color::delta_e_ok(
                    rgb_to_oklab(pixel.0.map(|v| v as f32 / 255.0)),
                    rgb_to_oklab(native.pixels[i]),
                );
                count += 1;
            }
        }
        assert!(
            error / (count as f32) < 4.0,
            "foreground colour error: {}",
            error / count as f32
        );
    }

    #[test]
    fn outline_contours_keep_line_arc_and_branch_connections() {
        let directory = tempfile::tempdir().unwrap();
        for gap in [false, true] {
            // An eight-pixel outline separates two colours, turns through
            // circular corners and joins a tapered branch. It must keep its
            // filled contour even where a uniform centre-line would fit.
            let source_svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="192" height="144">
                <path fill="#3399cc" d="M0 0H192V144H0Z"/>
                <path fill="#000000" d="M48 24H144A32 32 0 0 1 176 56V88A32 32 0 0 1 144 120H48A32 32 0 0 1 16 88V56A32 32 0 0 1 48 24Z"/>
                <path fill="#eec488" d="M48 32H144A24 24 0 0 1 168 56V88A24 24 0 0 1 144 112H48A24 24 0 0 1 24 88V56A24 24 0 0 1 48 32Z"/>
                <path fill="#000000" d="M76 116H86L110 140H106Z"/>
                {}</svg>"##,
                if gap {
                    r##"<path fill="#3399cc" d="M88 22H94V34H88Z"/>"##
                } else {
                    ""
                }
            );
            let source_tree = parse_svg_document(&source_svg).unwrap();
            let mut source = resvg::tiny_skia::Pixmap::new(192, 144).unwrap();
            resvg::render(
                &source_tree,
                resvg::tiny_skia::Transform::identity(),
                &mut source.as_mut(),
            );
            let input = directory.path().join("outline.png");
            let output = directory.path().join("outline.svg");
            source.save_png(&input).unwrap();
            let summary = vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 192,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            assert_eq!(summary.structural.recovered_boundary_strokes, 0);
            let document = fs::read_to_string(&output).unwrap();
            let paint = document
                .split("id=\"paint-layer\"")
                .nth(1)
                .unwrap()
                .split("</g>")
                .next()
                .unwrap();
            assert!(
                paint.contains('L'),
                "straight spans must remain in Paint paths"
            );
            if !gap {
                assert!(
                    paint.contains('A'),
                    "supported circular spans must remain in Paint paths"
                );
            }
            let tree = parse_svg_document(&document).unwrap();
            let mut rendered = resvg::tiny_skia::Pixmap::new(192, 144).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut rendered.as_mut(),
            );
            // Inspect the whole dark interior, including both straight/arc
            // joins and the branch junction. Ignore only the AA fringe.
            for y in 2..142 {
                for x in 2..190 {
                    let core = (y - 1..=y + 1).all(|py| {
                        (x - 1..=x + 1).all(|px| source.pixels()[py * 192 + px].red() < 20)
                    });
                    if core {
                        let p = rendered.pixels()[y * 192 + x];
                        assert!(
                            p.red().max(p.green()).max(p.blue()) < 100,
                            "broken outline at {x},{y}, gap={gap}: {p:?}"
                        );
                    }
                }
            }
            if gap {
                assert!(
                    rendered.pixels()[28 * 192 + 91].blue() > 150,
                    "the authored gap must stay open"
                );
            }
        }
    }

    #[test]
    fn native_alpha_white_rim_is_continuous_after_rendering() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let mut image = ImageBuffer::from_pixel(96, 96, Rgba([255_u8, 255, 255, 0]));
        for y in 0..96 {
            for x in 0..96 {
                let r = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                if r < 38.0 {
                    let grey =
                        (255.0 * (0.3 + 0.7 * ((r - 36.5) / 1.0).clamp(0.0, 1.0))).round() as u8;
                    image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
                }
            }
        }
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        image.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 96,
                auto_dimension: false,
                adaptive_refinement: false,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.structural.recovered_alpha_boundary_strokes > 0);
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for degrees in 0..360 {
            let angle = degrees as f32 * std::f32::consts::PI / 180.0;
            let mut peak = 0.0_f32;
            for j in 0..13 {
                let r = 36.0 + j as f32 * 0.25;
                let x = 47.5 + r * angle.cos();
                let y = 47.5 + r * angle.sin();
                let ix = x as usize;
                let iy = y as usize;
                let tx = x - ix as f32;
                let ty = y - iy as f32;
                let value: f32 = [
                    (ix, iy, (1.0 - tx) * (1.0 - ty)),
                    (ix + 1, iy, tx * (1.0 - ty)),
                    (ix, iy + 1, (1.0 - tx) * ty),
                    (ix + 1, iy + 1, tx * ty),
                ]
                .into_iter()
                .map(|(x, y, w)| {
                    let p = pixmap.pixels()[y * 96 + x];
                    w * (f32::from(p.red()) - 0.3 * f32::from(p.alpha()))
                })
                .sum();
                peak = peak.max(value);
            }
            assert!(peak > 30.0, "broken white rim at {degrees} degrees: {peak}");
        }
    }

    #[test]
    fn curved_highlight_remains_visible_beside_a_dark_seam() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        for transparent in [false, true] {
            let mut image = ImageBuffer::from_pixel(
                192,
                192,
                Rgba([255_u8, 255, 255, if transparent { 0 } else { 255 }]),
            );
            for y in 12..180 {
                for x in 12..180 {
                    let grey = 205 + (x / 8) as u8;
                    image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
                }
                let centre = (96.0 + 25.0 * (y as f32 / 28.0).sin()).round() as u32;
                for x in centre - 4..centre {
                    image.put_pixel(x, y, Rgba([100, 100, 100, 255]));
                }
                for x in centre..centre + 3 {
                    image.put_pixel(x, y, Rgba([252, 252, 252, 255]));
                }
            }
            let input = directory.path().join("input.png");
            let output = directory.path().join("output.svg");
            image.save(&input).unwrap();
            vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 192,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(192, 192).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in 20..172 {
                let centre = (96.0 + 25.0 * (y as f32 / 28.0).sin()).round() as usize;
                let peak = pixmap.pixels()[y * 192 + centre..y * 192 + centre + 4]
                    .iter()
                    .map(|p| p.red())
                    .max()
                    .unwrap();
                // Require at least half the local source contrast after curve antialiasing.
                let background = 205 + ((centre + 1) / 8) as u8;
                assert!(
                    f32::from(peak) >= 0.5 * f32::from(252_u16 + u16::from(background)),
                    "lost highlight at y={y}: {peak}, transparent={transparent}"
                );
            }
        }
    }

    #[test]
    fn transparent_greyscale_edges_do_not_invent_colours() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let config = Config {
            maximum_dimension: 96,
            auto_dimension: false,
            adaptive_refinement: false,
            rayon_threads: 1,
            ..Config::default()
        };
        let mut renders = Vec::new();
        for hidden in [[255, 0, 255, 0], [0, 255, 0, 0]] {
            let mut image = ImageBuffer::from_pixel(96, 96, Rgba(hidden));
            for y in 0..96 {
                for x in 0..96 {
                    let d = ((x as f32 - 48.0).powi(2) + (y as f32 - 48.0).powi(2)).sqrt();
                    let alpha = ((34.5 - d).clamp(0.0, 1.0) * 255.0) as u8;
                    if alpha > 0 {
                        let grey = if (45..49).contains(&x) {
                            250
                        } else {
                            110 + x as u8
                        };
                        image.put_pixel(x, y, Rgba([grey, grey, grey, alpha]));
                    }
                }
            }
            let input = directory.path().join("input.png");
            let output = directory.path().join("output.svg");
            image.save(&input).unwrap();
            vectorize(&input, &output, &config).unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for p in pixmap.pixels() {
                let channels = [p.red(), p.green(), p.blue()];
                assert!(
                    channels.iter().max().unwrap() - channels.iter().min().unwrap() <= 1,
                    "invented colour: {p:?}"
                );
            }
            renders.push(pixmap.take());
        }
        assert_eq!(
            renders[0], renders[1],
            "hidden RGB changed the visible result"
        );
    }

    #[test]
    fn translucent_black_grid_stays_neutral_and_connected() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("grid.png");
        let output = directory.path().join("grid.svg");
        let mut image = ImageBuffer::from_pixel(96, 96, Rgba([255_u8, 0, 255, 0]));
        // Other Paint fits must not leak the normalization backing into the
        // masked grid's underpaint, even when the image contains gradients.
        for y in 12..84 {
            for x in 6..34 {
                let grey = (90 + 4 * x) as u8;
                image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
            }
        }
        for y in 12..40 {
            for x in 65..88 {
                image.put_pixel(x, y, Rgba([0, 255, 0, 255]));
            }
        }
        for y in 3..93 {
            image.put_pixel(47, y, Rgba([0, 0, 0, 210]));
            image.put_pixel(48, y, Rgba([0, 0, 0, 204]));
        }
        for x in 3..93 {
            image.put_pixel(x, 47, Rgba([0, 0, 0, 210]));
            image.put_pixel(x, 48, Rgba([0, 0, 0, 204]));
        }
        image.put_pixel(47, 47, Rgba([0, 0, 0, 255]));
        // A real break must remain a break.
        for y in 70..76 {
            for x in 47..49 {
                image.put_pixel(x, y, Rgba([255, 0, 255, 0]));
            }
        }
        image.save(&input).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 96,
                auto_dimension: false,
                adaptive_refinement: false,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for y in 6..90 {
            if (69..77).contains(&y) {
                continue;
            }
            let pixels = &pixmap.pixels()[y * 96 + 46..y * 96 + 50];
            assert!(pixels.iter().any(|p| p.alpha() > 64), "grid gap at y={y}");
            for p in pixels.iter().filter(|p| p.alpha() > 32) {
                assert!(
                    p.red().max(p.green()).max(p.blue()) <= 3,
                    "tinted black grid at y={y}: {p:?}"
                );
            }
        }
        assert_eq!(pixmap.pixels()[73 * 96 + 47].alpha(), 0);
    }

    #[test]
    #[ignore = "renders the full-size car sample"]
    fn car_headlight_boundary_has_no_dark_spurs() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("car.svg");
        vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1254, 1254).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        // The AA samples at both tips used to acquire the near-black Paint
        // of the adjacent rim. Inspect the final composite, not just labels.
        for (x, y) in [(170, 637), (281, 593)] {
            let expected = source.get_pixel(x, y).0;
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()];
            assert!(
                expected
                    .into_iter()
                    .zip(actual)
                    .all(|(a, b)| a.abs_diff(b) <= 64),
                "headlight spur at ({x}, {y}): source {expected:?}, SVG {actual:?}"
            );
        }
        // The bumper highlight brightens towards the lower trim. The former
        // radial focus reversed this slope in the final rendered SVG.
        let upper = pixmap.pixels()[739 * 1254 + 216];
        let lower = pixmap.pixels()[745 * 1254 + 216];
        assert!(
            lower.green() >= upper.green() + 3,
            "bumper gradient reversed: {} -> {}",
            upper.green(),
            lower.green()
        );
        assert!(lower.green().abs_diff(source.get_pixel(216, 745)[1]) <= 24);
        // Residual corrections along the window rim must not spread a dark
        // patch into otherwise smooth blue glass.
        for (x, y) in [
            (664, 403),
            (664, 405),
            (670, 403),
            (690, 510),
            (735, 504),
            (819, 491),
        ] {
            let expected = source.get_pixel(x, y).0.map(|v| v as f32 / 255.0);
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()].map(|v| v as f32 / 255.0);
            let error = crate::color::delta_e_ok(rgb_to_oklab(expected), rgb_to_oklab(actual));
            assert!(
                error <= 3.0,
                "unsupported window shadow at {x},{y}: {error}"
            );
        }
        // Residual overlays must not reverse the smooth source ramps under
        // the mirror or above the front wheel into circular dark/light spots.
        for ((x0, y0), (x1, y1)) in [((628, 550), (628, 556)), ((380, 637), (386, 637))] {
            let lightness = |x: u32, y: u32| {
                let p = pixmap.pixels()[(y * 1254 + x) as usize];
                rgb_to_oklab([p.red(), p.green(), p.blue()].map(|c| c as f32 / 255.0)).l
            };
            let source_lightness =
                |x, y| rgb_to_oklab(source.get_pixel(x, y).0.map(|c| c as f32 / 255.0)).l;
            assert!(source_lightness(x1, y1) > source_lightness(x0, y0));
            assert!(
                lightness(x1, y1) >= lightness(x0, y0) - 0.5,
                "circular residual reversed the highlight at {x0},{y0} -> {x1},{y1}"
            );
        }
        // Smooth body highlights must not acquire nested quantizer bands.
        // Check both the rear quarter and the door, including the left bend.
        // resvg renders 672 / 1053 such steps in the original sample. Keep
        // substantial headroom for antialiasing while requiring a large drop.
        for (name, x0, y0, x1, y1, maximum_jumps, maximum_mean_error) in [
            ("rear", 950, 475, 1120, 524, 250, 2.8),
            ("door", 558, 543, 840, 610, 300, 4.1),
        ] {
            let mut jumps = 0;
            let mut colour_error = 0_u64;
            let mut colour_samples = 0_u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let reference = source.get_pixel(x, y).0;
                    if reference[0] < 220 || !(36..175).contains(&reference[1]) {
                        continue;
                    }
                    let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
                    let actual = [pixel.red(), pixel.green(), pixel.blue()];
                    colour_error += (0..3)
                        .map(|c| u64::from(actual[c].abs_diff(reference[c])))
                        .sum::<u64>();
                    colour_samples += 3;
                    for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                        let neighbour = source.get_pixel(nx, ny).0;
                        let rendered = pixmap.pixels()[(ny * 1254 + nx) as usize];
                        let rendered = [rendered.red(), rendered.green(), rendered.blue()];
                        let source_step = (0..3)
                            .map(|c| reference[c].abs_diff(neighbour[c]))
                            .max()
                            .unwrap();
                        let output_step = (0..3)
                            .map(|c| actual[c].abs_diff(rendered[c]))
                            .max()
                            .unwrap();
                        if source_step <= 3 && output_step > source_step + 8 {
                            jumps += 1;
                        }
                    }
                }
            }
            assert!(
                jumps <= maximum_jumps,
                "nested {name} highlight bands: {jumps}"
            );
            let mean = colour_error as f64 / colour_samples.max(1) as f64;
            assert!(
                mean <= maximum_mean_error,
                "{name} highlight colour error: {mean}"
            );
        }
        // Coverage repair must retain the previously restored rim reflection.
        for (x, y) in [(365, 762), (386, 738)] {
            let pixel = pixmap.pixels()[y * 1254 + x];
            assert!(
                pixel.red() > 120 && pixel.green() > 120 && pixel.blue() > 120,
                "wheel highlight disappeared at ({x}, {y})"
            );
        }
    }

    #[test]
    #[ignore = "renders the full-size car sample"]
    fn car_front_fender_highlight_keeps_local_shading() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("car.svg");
        let summary = vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            (summary.processing_width, summary.processing_height),
            (1254, 1254)
        );
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1254, 1254).unwrap();
        // Match the opaque source/browser background instead of measuring
        // premultiplied RGB against transparent black at antialiased edges.
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let mut error = 0_u64;
        let mut channels = 0_u64;
        // The curved highlight above the front wheel, excluding the lamp.
        // Keep the existing colour-error gate, including antialiased boundaries.
        // The reproducible pre-OKLab HEAD measures 3.737 with this renderer.
        for y in 565..645 {
            for x in 320..530 {
                let reference = source.get_pixel(x, y).0;
                if reference[0] <= 190 || reference[1] >= 160 {
                    continue;
                }
                let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
                for (expected, actual) in
                    reference
                        .into_iter()
                        .zip([pixel.red(), pixel.green(), pixel.blue()])
                {
                    error += u64::from(expected.abs_diff(actual));
                    channels += 1;
                }
            }
        }
        // Previously reported fender/bumper artifacts and the lower window
        // ridge: evaluate the final
        // composite as well as the field fit, so antialias seams are included.
        for (x, y) in [(335, 655), (528, 592), (286, 752), (780, 506), (780, 509)] {
            let expected = source.get_pixel(x, y).0;
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()];
            let worst = expected
                .into_iter()
                .zip(actual)
                .map(|(a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(
                worst <= 10,
                "false colour at ({x},{y}): {actual:?} != {expected:?}"
            );
        }
        // A good interior colour fit does not guarantee a smooth boundary.
        // Track the native upper wheel-arch step at subpixel precision and
        // measure its turning separately from the shading above it.
        let mut edge_positions = [Vec::new(), Vec::new()];
        for x in 360..491 {
            let y = (620..694)
                .min_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[1]) - i16::from(source.get_pixel(x, y)[1])
                })
                .unwrap();
            let mut weights = [0.0_f64; 2];
            let mut moments = [0.0_f64; 2];
            for offset in -4_i32..5 {
                let row = (y as i32 + offset) as u32;
                let a = pixmap.pixels()[(row * 1254 + x) as usize].green();
                let b = pixmap.pixels()[((row + 1) * 1254 + x) as usize].green();
                let steps = [
                    source.get_pixel(x, row)[1].saturating_sub(source.get_pixel(x, row + 1)[1]),
                    a.saturating_sub(b),
                ];
                for (i, step) in steps.into_iter().enumerate() {
                    weights[i] += f64::from(step);
                    moments[i] += f64::from(step) * (f64::from(row) + 0.5);
                }
            }
            for i in 0..2 {
                assert!(weights[i] > 0.0);
                edge_positions[i].push(moments[i] / weights[i]);
            }
        }
        let roughness = edge_positions[1]
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .sum::<f64>()
            / (edge_positions[1].len() - 2) as f64;
        let displacement = edge_positions[0]
            .iter()
            .zip(&edge_positions[1])
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / edge_positions[0].len() as f64;
        assert!(
            roughness < 0.16,
            "wheel-arch contour roughness: {roughness} (pre-OKLab: 0.149; regression: 0.180)"
        );
        assert!(
            displacement < 0.4,
            "wheel-arch contour moved {displacement}px from source"
        );
        assert!(channels > 40_000);
        let mean_error = error as f64 / channels as f64;
        assert!(
            mean_error < 3.65,
            "front fender highlight error: {mean_error}, regions: {}, output: {}",
            summary.geometry.regions,
            directory.keep().display()
        );
        // Inspect both sides of the narrow window rim in the final SVG.
        // The old fill-only contour fit hid a broken/stepped outer ink edge,
        // and pixel colour repairs introduced a 0.81px kink on the inner one.
        let enlarged = render_svg_tree_on(
            &tree,
            5016,
            5016,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        let mut rim_edges = [Vec::new(), Vec::new()];
        for x in 775..885 {
            let glass_edge = (332..364)
                .max_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[2]) - i16::from(source.get_pixel(x, y)[2])
                })
                .unwrap();
            let core = (glass_edge - 3..=glass_edge)
                .min_by_key(|&y| source.get_pixel(x, y).0.into_iter().max().unwrap())
                .unwrap();
            let start = (core - 2) * 4;
            let levels: Vec<_> = (start..start + 20)
                .map(|y| {
                    enlarged
                        .get(x as usize * 4 + 2, y as usize)
                        .into_iter()
                        .fold(0.0_f32, f32::max)
                })
                .collect();
            let threshold = 30.0 / 255.0;
            let first = levels
                .iter()
                .position(|&v| v < threshold)
                .expect("window ink gap");
            let last = levels.iter().rposition(|&v| v < threshold).unwrap();
            assert!(
                first > 0 && last + 1 < levels.len(),
                "rim left its source corridor at {x}"
            );
            for (edge, (a, b)) in [(first - 1, first), (last, last + 1)]
                .into_iter()
                .enumerate()
            {
                let t = (threshold - levels[a]) / (levels[b] - levels[a]);
                rim_edges[edge].push((start as f32 + a as f32 + 0.5 + t) * 0.25);
            }
        }
        for (edge, positions) in rim_edges.iter().enumerate() {
            let turns: Vec<_> = positions
                .windows(3)
                .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
                .collect();
            assert!(
                turns.iter().all(|&v| v < 0.4),
                "window rim {edge} kink: {turns:?}"
            );
            if edge == 1 {
                let roughness = turns.iter().sum::<f32>() / turns.len() as f32;
                assert!(roughness < 0.08, "window inner rim roughness: {roughness}");
            }
        }
        // The neighbouring grey trim has a separate boundary. Checking the
        // glass-facing edge alone missed its former >1px staircase.
        let mut grey_edges = [Vec::new(), Vec::new(), Vec::new()];
        for x in 775..885 {
            let y = (332..364)
                .max_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[2]) - i16::from(source.get_pixel(x, y)[2])
                })
                .unwrap();
            let start = (y - 11) * 4;
            let levels: Vec<_> = (start..(y - 3) * 4)
                .map(|row| {
                    let rgb = enlarged.get(x as usize * 4 + 2, row as usize);
                    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                })
                .collect();
            for (edge, threshold) in [40.0_f32, 50.0, 60.0].into_iter().enumerate() {
                let threshold = threshold / 255.0;
                let i = levels
                    .windows(2)
                    .position(|p| p[0] < threshold && p[1] >= threshold)
                    .unwrap_or_else(|| panic!("grey trim edge missing at {x}"));
                let t = (threshold - levels[i]) / (levels[i + 1] - levels[i]);
                grey_edges[edge].push((start as f32 + i as f32 + 0.5 + t) * 0.25);
            }
        }
        for positions in grey_edges {
            let turns: Vec<_> = positions
                .windows(3)
                .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
                .collect();
            assert!(turns.iter().all(|&v| v < 0.55), "grey trim kink: {turns:?}");
            let mean = turns.iter().sum::<f32>() / turns.len() as f32;
            assert!(mean < 0.09, "grey trim roughness: {mean}");
        }
    }

    #[test]
    #[ignore = "renders the full-size cliparts sample"]
    fn cliparts_highlights_preserve_colour_and_authored_transparency() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/cliparts.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("cliparts.svg");
        vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let source = image::open(input).unwrap().to_rgba8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1600, 1200).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for (name, (x0, y0, x1, y1), limit) in [
            ("penguin highlight", (945, 457, 975, 470), 7.0),
            ("flask liquid", (1380, 160, 1430, 215), 2.0),
            // Before source-supported ink refinement: 35.54 and 6.52.
            ("dotted oval upper edge", (500, 309, 530, 320), 8.0),
            ("dotted oval", (468, 310, 546, 368), 4.0),
        ] {
            let mut error = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let reference = source.get_pixel(x, y).0;
                    let alpha = reference[3] as f64 / 255.0;
                    let pixel = pixmap.pixels()[(y * 1600 + x) as usize];
                    for (c, actual) in [pixel.red(), pixel.green(), pixel.blue()]
                        .into_iter()
                        .enumerate()
                    {
                        error += (reference[c] as f64 * alpha + 255.0 * (1.0 - alpha)
                            - actual as f64)
                            .abs();
                    }
                }
            }
            let mean = error / (3 * (x1 - x0) * (y1 - y0)) as f64;
            assert!(mean < limit, "{name}: mean RGB error {mean}");
        }
    }

    #[test]
    #[ignore = "renders the full-size cliparts sample"]
    fn cliparts_penguin_inner_foot_outlines_remain_continuous() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/cliparts.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("cliparts.svg");
        let summary = vectorize(&input, &output, &Config::default()).unwrap();
        assert_eq!(
            (summary.processing_width, summary.processing_height),
            (1600, 1200)
        );
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1600, 1200).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        // Follow the source's dark centre on each side of the white gap.
        // Permit one pixel of fitting displacement, but never a missing row.
        for (left, right) in [(989, 998), (1000, 1018)] {
            for y in 685..723 {
                let x = (left..right)
                    .min_by_key(|&x| *source.get_pixel(x, y).0.iter().max().unwrap())
                    .unwrap();
                assert!(
                    (x - 1..=x + 1).any(|sample_x| {
                        let p = pixmap.pixels()[(y * 1600 + sample_x) as usize];
                        p.red().max(p.green()).max(p.blue()) <= 128
                    }),
                    "penguin foot outline lost at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn tonal_details_survive_rendering_with_authored_gaps() {
        use image::{ImageBuffer, Rgb};

        for (background, detail) in [
            ([238_u8, 238, 242], [228_u8, 228, 232]),
            ([24, 24, 28], [32, 32, 36]),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let input = directory.path().join("faint-seam.png");
            let output = directory.path().join("faint-seam.svg");
            let mut image = ImageBuffer::from_pixel(96, 96, Rgb(background));
            for y in 8..88 {
                if (43..53).contains(&y) {
                    continue;
                }
                for x in 46..50 {
                    image.put_pixel(x, y, Rgb(detail));
                }
            }
            image.save(&input).unwrap();
            vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 96,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in (12..39).chain(57..84) {
                let wall = pixmap.pixels()[y * 96 + 40].red();
                let seam = pixmap.pixels()[y * 96 + 48].red();
                assert!(
                    wall.abs_diff(seam) >= 4,
                    "faint seam lost at {y}: wall={wall}, seam={seam}"
                );
            }
            let wall = pixmap.pixels()[48 * 96 + 40].red();
            let gap = pixmap.pixels()[48 * 96 + 48].red();
            assert!(wall.abs_diff(gap) <= 2, "authored seam gap was filled");
        }
    }

    #[test]
    fn face_alpha_preserves_foreground_equal_to_temporary_backing() {
        use image::{ImageBuffer, Rgba};

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut image = ImageBuffer::from_pixel(64, 64, Rgba([17_u8, 31, 47, 0]));

        // Cyan dominates covered pixels, making red the farthest temporary
        // normalization colour. The disconnected opaque red square must not
        // merge with that normalized transparent RGB and disappear.
        for y in 16..56 {
            for x in 16..56 {
                image.put_pixel(x, y, Rgba([0, 255, 255, 255]));
            }
        }
        for y in 4..12 {
            for x in 4..12 {
                image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        // Broad authored translucent areas exercise the two intermediate
        // alpha levels independently of boundary antialiasing.
        for y in 22..30 {
            for x in 22..30 {
                image.put_pixel(x, y, Rgba([0, 255, 0, 80]));
            }
        }
        for y in 38..46 {
            for x in 38..46 {
                image.put_pixel(x, y, Rgba([0, 0, 255, 180]));
            }
        }
        image.save(&input).unwrap();

        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            summary.source_alpha.temporary_backing_color,
            Some([255, 0, 0])
        );
        assert_eq!(summary.source_alpha.quantization_bits, 8);
        assert_eq!(summary.source_alpha.mask_paths, 0);

        let document = fs::read_to_string(&output).unwrap();
        assert!(document.contains("fill-opacity=\"0.314\""));
        assert!(document.contains("fill-opacity=\"0.706\""));
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let pixel = |x: usize, y: usize| pixmap.pixels()[y * 64 + x];
        assert_eq!(pixel(2, 2).alpha(), 0);
        assert!(pixel(8, 8).alpha() > 240);
        assert!(pixel(8, 8).red() > 240);
        assert!(pixel(8, 8).green() < 10);
        assert!((75..=95).contains(&pixel(26, 26).alpha()));
        assert!((160..=180).contains(&pixel(42, 42).alpha()));
    }

    #[test]
    fn authored_alpha_ramp_is_not_reduced_to_four_bands() {
        use image::{ImageBuffer, Rgba};
        for (low, span) in [(16.0, 224.0), (220.0, 32.0)] {
            let directory = tempfile::tempdir().unwrap();
            let input = directory.path().join("ramp.png");
            let output = directory.path().join("ramp.svg");
            let mut image = ImageBuffer::from_pixel(128, 128, Rgba([0_u8, 0, 255, 0]));
            for y in 16..112 {
                for x in 16..112 {
                    image.put_pixel(
                        x,
                        y,
                        Rgba([
                            0,
                            0,
                            255,
                            (low + span * (y - 16) as f32 / 95.0).round() as u8,
                        ]),
                    );
                }
            }
            image.save(&input).unwrap();
            let summary = vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 128,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 4,
                    ..Config::default()
                },
            )
            .unwrap();
            assert_eq!(summary.source_alpha.quantization_bits, 8);
            // Each coverage layer now has a separate seam stroke underneath
            // the fills; allow eight layers plus their eight underpass paths.
            assert!(summary.source_alpha.mask_paths <= 16);
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(128, 128).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in 24..104 {
                let expected = image.get_pixel(64, y).0[3];
                let actual = pixmap.pixels()[y as usize * 128 + 64].alpha();
                assert!(
                    expected.abs_diff(actual) <= 5,
                    "y={y}: alpha {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn source_antialias_coverage_builds_one_opaque_vector_silhouette() {
        let matte = AlphaMatte::new(
            5,
            3,
            [0.0, 0.25, 0.75, 1.0, 1.0]
                .into_iter()
                .cycle()
                .take(15)
                .collect(),
        );
        let path = matte
            .isocontours(0.5)
            .iter()
            .map(|contour| fitted_alpha_contour_path_data(contour))
            .collect::<String>();
        assert!(!path.is_empty());
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn quality_metrics_require_explicit_opt_in() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "picvec-quality-contract-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("output.svg");
        let mut raster = Raster::blank(32, 24, [0.9, 0.9, 0.9]);
        for y in 6..18 {
            for x in 8..24 {
                raster.pixels[y * 32 + x] = [0.2, 0.4, 0.75];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                compute_quality_metrics: true,
                ..Config::default()
            },
        )
        .unwrap();
        let quality = summary.quality.unwrap();
        assert!(quality.delta_e_ok_mean.is_finite());
        assert!(quality.delta_e_ok_p90.is_finite());
        assert!(quality.delta_e_ok_p99.is_finite());
        assert!(quality.global_ssim.is_finite());
        assert!(quality.local_ssim.is_finite());
        assert_eq!((quality.width, quality.height), (32, 24));
        assert_eq!(quality.local_ssim_window, 7);
        assert_eq!(quality.comparison_background, Some([1.0; 3]));
        assert!(!quality.worst_tiles.is_empty());
        let without_metrics = directory.join("without-metrics.svg");
        vectorize(
            &input,
            &without_metrics,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            fs::read(&output).unwrap(),
            fs::read(without_metrics).unwrap()
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn concurrent_conversions_to_one_output_remain_atomic() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "picvec-concurrent-contract-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("shared.svg");
        let mut raster = Raster::blank(32, 24, [0.88, 0.91, 0.95]);
        for y in 4..20 {
            for x in 5..27 {
                raster.pixels[y * 32 + x] = if x < 16 {
                    [0.16, 0.24, 0.72]
                } else {
                    [0.78, 0.18, 0.25]
                };
            }
        }
        raster.save(&input).unwrap();

        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let input = input.clone();
                let output = output.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    vectorize(
                        &input,
                        &output,
                        &Config {
                            segmentation_min_size: 2,
                            minimum_gradient_area: 8,
                            rayon_threads: 1,
                            ..Config::default()
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        let document = fs::read_to_string(&output).unwrap();
        assert!(document.starts_with("<?xml"));
        assert!(document.contains("<svg"));
        let files: HashSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files,
            HashSet::from(["input.png".to_string(), "shared.svg".to_string()])
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
