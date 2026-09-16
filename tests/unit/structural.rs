fn select_residual_components_reference(
    represented: &[bool],
    source_lines: &[bool],
    width: usize,
    height: usize,
) -> (Vec<bool>, Vec<bool>) {
    let mut selected = vec![false; source_lines.len()];
    let mut measured = vec![false; source_lines.len()];
    for component in connected_components(source_lines, width, height) {
        let mut missing = vec![false; source_lines.len()];
        let mut missing_area = 0_usize;
        for &index in &component {
            if !represented[index] {
                missing[index] = true;
                missing_area += 1;
            }
        }
        if missing_area < 3
            || (missing_area < 8 && missing_area as f32 / (component.len().max(1) as f32) < 0.12)
        {
            continue;
        }
        let expanded = dilate(&missing, width, height, 1);
        let component_residual: Vec<usize> = component
            .iter()
            .copied()
            .filter(|&index| expanded[index])
            .collect();
        for &index in &component_residual {
            measured[index] = true;
        }
        if component_residual.len() as f32 / component.len().max(1) as f32 >= 0.75 {
            for &index in &component {
                selected[index] = true;
            }
        } else {
            for index in component_residual {
                selected[index] = true;
            }
        }
    }
    remove_small_components(&mut selected, width, height, 3);
    remove_small_components(&mut measured, width, height, 3);
    (selected, measured)
}

mod tests {
    #[test]
    fn paint_owned_transparency_never_enters_residual_line_fitting() {
        let mut source = super::Raster::blank(64, 48, [0.8; 3]);
        for x in 8..56 {
            source.pixels[24 * 64 + x] = [0.02; 3];
        }
        let mut roles = crate::edge::classify(&source);
        let protected = vec![true; source.pixels.len()];
        let (paint, mut candidates) =
            super::analyse_with_protection(&source, &mut roles, Some(&protected));
        assert!(roles.visible_ridge_graph.is_empty());
        assert!(roles.dark_boundary_graph.is_empty());
        assert!(candidates.strokes.is_empty());
        candidates.release_to_paint(&protected, source.width);
        assert!(candidates.role_line_mask.iter().all(|&v| !v));
        assert!(candidates.legacy_line_mask.iter().all(|&v| !v));
        assert!(candidates.visible_ridge_coverage.iter().all(|&v| !v));
        assert_eq!(paint.pixels, source.pixels);
        let selected = super::select_missing(&source, &paint, &candidates);
        assert!(selected.strokes.is_empty());
    }
    #[test]
    fn local_residual_dilation_matches_full_image_components() {
        let mut state = 17u64;
        for (width, height) in [(1, 1), (1, 41), (47, 1), (17, 29), (63, 49)] {
            for density in [0, 1, 3, 7, 10] {
                for _ in 0..5 {
                    let mut random = || {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (state >> 32) as usize
                    };
                    let source: Vec<_> = (0..width * height)
                        .map(|_| random() % 10 < density)
                        .collect();
                    let represented: Vec<_> =
                        (0..width * height).map(|_| random() % 4 != 0).collect();
                    assert_eq!(
                        select_residual_components(&represented, &source, width, height),
                        select_residual_components_reference(&represented, &source, width, height),
                        "{width}x{height}, density={density}"
                    );
                }
            }
        }
        // A diagonal contact joins a component but must not enter the axial
        // radius-one dilation. Include separate neighbours and edge pixels.
        let (w, h) = (19, 17);
        let mut source = vec![false; w * h];
        for i in 0..17 {
            source[i * w + i] = true;
        }
        for x in 0..19 {
            source[x] = true;
            source[16 * w + x] = true;
        }
        for period in 2..9 {
            let represented: Vec<_> = (0..w * h).map(|i| i % period != 0).collect();
            assert_eq!(
                select_residual_components(&represented, &source, w, h),
                select_residual_components_reference(&represented, &source, w, h)
            );
        }
    }

    use super::*;

    #[test]
    fn a_bright_rim_does_not_join_the_adjacent_dark_trim() {
        let first = StructuralStroke {
            points: vec![Point { x: 4.5, y: 8.5 }, Point { x: 14.5, y: 8.5 }],
            path_data: None,
            precise_points: None,
            color: [0.9; 3],
            width: 1.5,
            role: "bright-ridge-on-boundary",
            width_samples: Vec::new(),
        };
        let mut second = StructuralStroke {
            points: vec![Point { x: 16.5, y: 8.5 }, Point { x: 26.5, y: 8.5 }],
            role: "ridge",
            color: [0.1; 3],
            ..first.clone()
        };
        let support = vec![true; 32 * 16];
        assert_eq!(
            connect_graph_edges(
                vec![first.clone(), second.clone()],
                &support,
                32,
                16,
                6.0,
                15.0
            )
            .len(),
            2
        );
        second.role = "bright-ridge-on-boundary";
        second.color = first.color;
        assert_eq!(
            connect_graph_edges(vec![first, second], &support, 32, 16, 6.0, 15.0).len(),
            1
        );
    }

    #[test]
    fn car_wheel_highlight_is_centred_on_its_bright_source_ridge() {
        let raster = |bytes: &[u8]| {
            let image = image::load_from_memory(bytes).unwrap().to_rgb8();
            Raster::new(
                image.width() as usize,
                image.height() as usize,
                image
                    .pixels()
                    .map(|p| p.0.map(|c| c as f32 / 255.0))
                    .collect(),
            )
        };
        let source = raster(include_bytes!("../data/car-wheel-highlight.png"));
        let paint = raster(include_bytes!("../data/car-wheel-underpaint.png"));
        let record: serde_json::Value =
            serde_json::from_str(include_str!("../data/car-wheel-ridge.json")).unwrap();
        let mut stroke = StructuralStroke {
            points: record["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| Point {
                    x: p[0].as_f64().unwrap() as f32,
                    y: p[1].as_f64().unwrap() as f32,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: std::array::from_fn(|c| record["color"][c].as_f64().unwrap() as f32),
            width: record["width"].as_f64().unwrap() as f32,
            role: "bright-ridge-on-boundary",
            width_samples: Vec::new(),
        };
        refine_bright_ridge(
            &mut stroke,
            &oklab_pixels(&source),
            source.width,
            source.height,
        );
        stroke.color = sample_graph_color(&source, &stroke);
        stroke.path_data = Some(fitted_structural_open_path_data_with_tangents(
            &stroke.points,
            0.35,
            0.45,
            None,
            None,
        ));
        assert!(
            stroke.color[0] > 0.7,
            "a bright rim must not inherit the dark side colour"
        );
        assert!(stroke_overlay_improves_paint(&stroke, &source, &paint));
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="45" height="64"><path d="{}" fill="none" stroke="white" stroke-width="{}" stroke-linecap="round"/></svg>"#,
            stroke.path_data.as_ref().unwrap(),
            stroke.width
        );
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut mask = resvg::tiny_skia::Pixmap::new(45, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut mask.as_mut(),
        );
        assert!(
            mask.pixels()[17 * 45 + 25].alpha() > 120,
            "the removed highlight must be painted back at its source position"
        );
        let path = stroke.path_data.unwrap();
        assert_eq!(
            path.matches('C').count() + path.matches('A').count(),
            1,
            "the smooth rim interval should use one curve: {path}"
        );
    }

    #[test]
    fn a_tapered_paint_tip_is_not_retraced_as_a_round_capped_line() {
        let render = |content: &str| {
            let document = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path fill="#a156ac" d="M0 0H32V32H0Z"/>{content}</svg>"##
            );
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            let mut pixels = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixels.as_mut(),
            );
            Raster::new(
                32,
                32,
                pixels
                    .pixels()
                    .iter()
                    .map(|p| {
                        [
                            p.red() as f32 / 255.0,
                            p.green() as f32 / 255.0,
                            p.blue() as f32 / 255.0,
                        ]
                    })
                    .collect(),
            )
        };
        let stroke = StructuralStroke {
            points: vec![Point { x: 15.0, y: 17.0 }, Point { x: 10.0, y: 24.0 }],
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 4.0,
            role: "legacy-structural",
            width_samples: vec![(4.0, 8)],
        };
        let source = render(r##"<path fill="#000000" d="M10 24L20 3L28 9Z"/>"##);
        let paint = render(r##"<path fill="#080808" d="M10 24L20 3L28 9Z"/>"##);
        // The centreline gets darker, but the full stroke thickens the tip
        // and adds a round cap outside the authored silhouette.
        assert!(!stroke_overlay_improves_paint(&stroke, &source, &paint));
        let actual_line = render(
            r##"<path fill="none" stroke="#000000" stroke-width="4" stroke-linecap="round" d="M15 17L10 24"/>"##,
        );
        assert!(stroke_overlay_improves_paint(
            &stroke,
            &actual_line,
            &render("")
        ));
        let bent = StructuralStroke {
            points: vec![
                Point { x: 20.0, y: 16.0 },
                Point { x: 19.5, y: 13.5 },
                Point { x: 8.5, y: 19.5 },
            ],
            path_data: Some("M20 16C19.8 15.2 19.7 14.3 19.5 13.5L8.5 19.5".into()),
            width: 1.0,
            ..stroke
        };
        let rim =
            render(r##"<path fill="none" stroke="#000000" stroke-width="2" d="M5 22L25 12"/>"##);
        let rim_paint =
            render(r##"<path fill="none" stroke="#080808" stroke-width="2" d="M5 22L25 12"/>"##);
        assert!(!stroke_overlay_improves_paint(&bent, &rim, &rim_paint));
        let authored = render(
            r##"<path fill="none" stroke="#000000" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" d="M20 16C19.8 15.2 19.7 14.3 19.5 13.5L8.5 19.5"/>"##,
        );
        assert!(stroke_overlay_improves_paint(&bent, &authored, &render("")));
    }

    #[test]
    fn serialized_interior_colour_patch_preserves_the_stroke_silhouette() {
        let render = |ink: &StructuralInk| {
            let (document, _) = crate::svg::serialize(24, 24, &[], &[], ink, 0.0, true);
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(24, 24).unwrap();
            pixmap.fill(resvg::tiny_skia::Color::WHITE);
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            Raster::new(
                24,
                24,
                pixmap
                    .pixels()
                    .iter()
                    .map(|p| {
                        [
                            p.red() as f32 / 255.0,
                            p.green() as f32 / 255.0,
                            p.blue() as f32 / 255.0,
                        ]
                    })
                    .collect(),
            )
        };
        let mut ink = StructuralInk::empty();
        ink.strokes.push(StructuralStroke {
            points: vec![Point { x: 3.5, y: 12.0 }, Point { x: 20.5, y: 12.0 }],
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 4.0,
            role: "ridge",
            width_samples: vec![(4.0, 2)],
        });
        let before = render(&ink);
        let mut source = before.clone();
        for x in (8..14).chain(17..19) {
            source.pixels[12 * 24 + x] = [0.9, 0.05, 0.1];
        }
        let paint = Raster::blank(24, 24, [1.0; 3]);
        let mut probe = ink.clone();
        probe.strokes[0].color = [1.0; 3];
        let white = render(&probe);
        ink.color_patches = crate::ink_color::propose(&source, &paint, &before, &white, &before);
        assert_eq!(ink.color_patches.len(), 2);
        let (_, summary) = crate::svg::serialize(24, 24, &[], &[], &ink, 0.0, true);
        assert_eq!(
            summary.structural_color_patches, 1,
            "same-colour interiors must share one path"
        );
        let after = render(&ink);
        assert!(ink.color_patches[0].improves(&source, &before, &after));
        for i in 0..24 * 24 {
            if i / 24 == 12 && ((8..14).contains(&(i % 24)) || (17..19).contains(&(i % 24))) {
                assert!(after.pixels[i][0] > 0.8 && after.pixels[i][1] < 0.1);
            } else {
                assert_eq!(
                    after.pixels[i], before.pixels[i],
                    "stroke changed outside recovered interior at {i}"
                );
            }
        }
        ink.color_patches[0].path = "M8 0h6v24h-6z".to_string();
        let clipped = render(&ink);
        for i in 0..24 * 24 {
            if before.pixels[i] == [1.0; 3] {
                assert_eq!(
                    clipped.pixels[i], [1.0; 3],
                    "patch escaped the original ink at {i}"
                );
            }
        }
    }

    #[test]
    fn car_mirror_terminal_joins_crossing_ink_but_not_an_authored_gap() {
        let image = image::load_from_memory(include_bytes!("../data/car-mirror-junction.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            128,
            128,
            image
                .pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../data/car-mirror-junction.json")).unwrap();
        let strokes: Vec<_> = data
            .as_array()
            .unwrap()
            .iter()
            .map(|s| StructuralStroke {
                points: s["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| Point {
                        x: p[0].as_f64().unwrap() as f32,
                        y: p[1].as_f64().unwrap() as f32,
                    })
                    .collect(),
                path_data: None,
                precise_points: None,
                color: std::array::from_fn(|i| s["color"][i].as_f64().unwrap() as f32),
                width: s["width"].as_f64().unwrap() as f32,
                role: if s["role"] == "ridge" {
                    "ridge"
                } else {
                    "ridge-on-boundary"
                },
                width_samples: Vec::new(),
            })
            .collect();
        let before = strokes[0].points[0];
        for gap in [false, true] {
            let mut source = source.clone();
            if gap {
                for y in 62..66 {
                    for x in 50..63 {
                        source.pixels[y * 128 + x] = [0.22; 3];
                    }
                }
            }
            let mut candidate = strokes.clone();
            extend_graph_to_crossing_ink(&mut candidate, &oklab_pixels(&source), 128, 128);
            if gap {
                assert_eq!(
                    candidate[0].points[0], before,
                    "must not bridge a real source gap"
                );
            } else {
                let target = candidate[0].points[0];
                assert!(
                    target.y < 59.0 && target.y > 56.0,
                    "missed the mirror rim: {target:?}"
                );
                assert!((target.x - 56.2).abs() < 0.5);
                assert_eq!(
                    candidate[1].points, strokes[1].points,
                    "receiving rim must stay fixed"
                );
            }
        }
        let mut isolated = vec![strokes[0].clone()];
        extend_graph_to_crossing_ink(&mut isolated, &oklab_pixels(&source), 128, 128);
        assert_eq!(
            isolated[0].points, strokes[0].points,
            "source evidence alone is not a crossing stroke"
        );
    }

    #[test]
    fn terminal_ridge_joins_paint_covering_its_drawn_width() {
        for stroke_width in [2.0_f32, 4.0, 6.0] {
            let (width, height) = (48, 32);
            let background = [0.65; 3];
            let ink = [0.03; 3];
            let mut source = Raster::blank(width, height, background);
            let mut paint = source.clone();
            let half = (stroke_width * 0.5) as usize;
            for y in 16 - half..=16 + half {
                for x in 10..width {
                    source.pixels[y * width + x] = ink;
                    if x >= 32 {
                        paint.pixels[y * width + x] = ink;
                    }
                }
            }
            let mut strokes = vec![StructuralStroke {
                points: vec![Point { x: 10.5, y: 16.5 }, Point { x: 28.5, y: 16.5 }],
                path_data: None,
                precise_points: None,
                color: ink,
                width: stroke_width,
                role: "ridge",
                width_samples: Vec::new(),
            }];
            extend_graph_to_dark_paint(
                &mut strokes,
                &oklab_pixels(&source),
                &oklab_pixels(&paint),
                width,
                height,
            );
            assert!(
                strokes[0].points.last().unwrap().x >= 32.0,
                "receiving ink need only cover the drawn width {stroke_width}"
            );
        }
    }

    #[test]
    fn terminal_ridge_does_not_stop_across_an_unpainted_sliver() {
        for sliver_row in [15, 18] {
            for source_gap in [false, true] {
                let (width, height) = (48, 32);
                let background = [0.65; 3];
                let ink = [0.03; 3];
                let mut source = Raster::blank(width, height, background);
                for y in 0..height {
                    for x in 28..width {
                        source.pixels[y * width + x] = ink;
                    }
                }
                for x in 10..28 {
                    source.pixels[16 * width + x] = ink;
                }
                let mut paint = source.clone();
                // The old three samples (centre and two shoulders) all see
                // ink, while a one-pixel Paint hole lies between them.
                for x in 28..35 {
                    paint.pixels[sliver_row * width + x] = background;
                }
                if source_gap {
                    for x in 30..33 {
                        source.pixels[16 * width + x] = background;
                    }
                }
                let end = Point { x: 28.5, y: 16.5 };
                let mut strokes = vec![StructuralStroke {
                    points: vec![Point { x: 10.5, y: 16.5 }, end],
                    path_data: None,
                    precise_points: None,
                    color: ink,
                    width: 4.0,
                    role: "ridge",
                    width_samples: Vec::new(),
                }];
                extend_graph_to_dark_paint(
                    &mut strokes,
                    &oklab_pixels(&source),
                    &oklab_pixels(&paint),
                    width,
                    height,
                );
                let actual = *strokes[0].points.last().unwrap();
                if source_gap {
                    assert_eq!(actual.x, end.x, "must not bridge authored gaps");
                } else {
                    assert!(
                        actual.x >= 35.0,
                        "must overlap solid receiving Paint: {actual:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn terminal_ridge_reaches_dark_paint_only_with_continuous_source_ink() {
        for gap in [false, true] {
            let width = 48;
            let height = 32;
            let background = [0.65, 0.1, 0.1];
            let ink = [0.03; 3];
            let mut paint = Raster::blank(width, height, background);
            for y in 0..height {
                for x in 32..width {
                    paint.pixels[y * width + x] = ink;
                }
            }
            let mut source = paint.clone();
            for x in 10..32 {
                if !gap || x < 29 {
                    source.pixels[16 * width + x] = ink;
                }
            }
            let original_end = Point { x: 28.5, y: 16.5 };
            let mut strokes = vec![StructuralStroke {
                points: vec![Point { x: 10.5, y: 16.5 }, original_end],
                path_data: None,
                precise_points: None,
                color: ink,
                width: 1.2,
                role: "ridge",
                width_samples: Vec::new(),
            }];
            extend_graph_to_dark_paint(
                &mut strokes,
                &oklab_pixels(&source),
                &oklab_pixels(&paint),
                width,
                height,
            );
            let end = *strokes[0].points.last().unwrap();
            if gap {
                assert_eq!(end, original_end, "bridged an intentional source gap");
            } else {
                assert!(
                    end.x >= 32.5 && end.x <= 33.0,
                    "missed the receiving Paint: {end:?}"
                );
                assert_eq!(end.y, 16.5);
            }
            assert_eq!(strokes[0].points[0], Point { x: 10.5, y: 16.5 });
        }
    }

    #[test]
    fn straight_stroke_is_validated_after_restoring_shared_endpoints() {
        // A long source-aligned run with a junction displaced by 2.6 px,
        // as at the car's side-window corner. The unconstrained fit passes,
        // but anchoring its end would tilt the whole run off the source.
        let mut points = (0..=100)
            .map(|x| Point {
                x: x as f32,
                y: 20.0,
            })
            .collect::<Vec<_>>();
        points[100].y -= 2.6;
        let tolerance = std::f32::consts::FRAC_1_SQRT_2;
        assert!(straight_graph_line(&points, tolerance, 4.0, false, false).is_some());
        assert!(straight_graph_line(&points, tolerance, 4.0, false, true).is_none());
        points.reverse();
        assert!(straight_graph_line(&points, tolerance, 4.0, true, false).is_none());
        // A truly straight run still emits one line and retains both nodes.
        for point in &mut points {
            point.y = 20.0 - point.x * 0.15;
        }
        let (start, end) = straight_graph_line(&points, tolerance, 4.0, true, true).unwrap();
        assert_eq!(start, points[0]);
        assert_eq!(end, points[100]);
    }

    #[test]
    fn dotted_continuations_require_source_marks_inside_the_gap() {
        for missing_source in [false, true] {
            let mut source = Raster::blank(96, 40, [1.0; 3]);
            for x in 8..88 {
                if x % 4 < 2 && !(missing_source && (32..48).contains(&x)) {
                    source.pixels[20 * 96 + x] = [0.0; 3];
                }
            }
            let mut ink = StructuralInk::empty();
            for (a, b) in [(8.5, 30.5), (48.5, 85.5)] {
                ink.strokes.push(StructuralStroke {
                    points: vec![Point { x: a, y: 20.5 }, Point { x: b, y: 20.5 }],
                    path_data: None,
                    precise_points: None,
                    color: [0.0; 3],
                    width: 1.4,
                    role: "ridge-on-boundary",
                    width_samples: Vec::new(),
                });
            }
            ink.refine_interrupted_strokes(&source, None);
            let bridge = ink
                .strokes
                .iter()
                .any(|s| s.points.iter().any(|p| (35.0..45.0).contains(&p.x)));
            assert_eq!(bridge, !missing_source, "source gap must remain empty");
        }
    }

    #[test]
    fn partial_ridge_detection_keeps_a_connected_silhouette_in_paint() {
        let mut source = Raster::blank(64, 64, [1.0; 3]);
        for y in 8..56 {
            for x in 8..32 {
                source.pixels[y * 64 + x] = [0.0; 3];
            }
        }
        for y in 30..32 {
            for x in 32..56 {
                source.pixels[y * 64 + x] = [0.0; 3];
            }
        }
        let mut roles = crate::edge::classify(&source);
        // Model a detector that recognizes only the middle of the thin arm.
        roles.visible_ridge_coverage.fill(false);
        roles.dark_boundary_graph.clear();
        roles.band_boundary_graph.clear();
        for y in 30..32 {
            for x in 38..44 {
                roles.visible_ridge_coverage[y * 64 + x] = true;
            }
        }
        let (paint, ink) = analyse(&source, &mut roles);
        let residual = ink.residual_source_line_mask();
        for y in 30..32 {
            for x in 32..56 {
                let i = y * 64 + x;
                assert_eq!(
                    paint.pixels[i], source.pixels[i],
                    "silhouette removed at ({x}, {y})"
                );
                assert!(!ink.paint_ownership_mask[i]);
                assert!(!residual[i]);
            }
        }
    }

    fn horizontal_boundary_stroke(last_x: usize) -> StructuralStroke {
        StructuralStroke {
            points: (2..=last_x)
                .map(|x| Point {
                    x: x as f32 + 0.5,
                    y: 4.5,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: [0.02; 3],
            width: 2.0,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        }
    }

    fn dark_boundary_profile(light_side: f32) -> Vec<Oklab> {
        let width = 18;
        let height = 9;
        let mut values = vec![Oklab::default(); width * height];
        for y in 0..height {
            let lightness = if y < 4 {
                light_side
            } else if y == 4 {
                5.0
            } else {
                22.0
            };
            for x in 0..width {
                values[y * width + x] = Oklab {
                    l: lightness,
                    a: 0.0,
                    b: 0.0,
                };
            }
        }
        values
    }

    fn graph_stroke(points: &[(f32, f32)]) -> StructuralStroke {
        StructuralStroke {
            points: points.iter().map(|&(x, y)| Point { x, y }).collect(),
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 1.0,
            role: "legacy-structural",
            width_samples: Vec::new(),
        }
    }

    #[test]
    fn ellipse_paint_suppresses_only_complete_already_painted_retraces() {
        let width = 64;
        let paint = Raster::new(
            width,
            width,
            (0..width * width)
                .map(|i| {
                    let radius =
                        ((i % width) as f32 + 0.5 - 32.0).hypot((i / width) as f32 + 0.5 - 32.0);
                    if (18.0..=22.0).contains(&radius) {
                        [0.0; 3]
                    } else {
                        [1.0; 3]
                    }
                })
                .collect(),
        );
        let circle = |radius: f32| -> Vec<Point> {
            (0..=180)
                .map(|i| {
                    let a = i as f32 / 180.0 * std::f32::consts::TAU;
                    Point {
                        x: 32.0 + radius * a.cos(),
                        y: 32.0 + radius * a.sin(),
                    }
                })
                .collect()
        };
        let contours = vec![circle(22.0)];
        let mut retrace = graph_stroke(&[]);
        retrace.points = circle(20.0);
        retrace.color = [0.0; 3];
        retrace.width = 3.0;
        let mut branch = retrace.clone();
        branch.points.push(Point { x: 32.0, y: 32.0 });
        let mut other_colour = retrace.clone();
        other_colour.color = [0.0, 0.0, 1.0];
        let mut transferred = retrace.clone();
        transferred.role = "boundary-stroke";
        let mut ink = StructuralInk::empty();
        ink.strokes = vec![retrace.clone(), branch, other_colour, transferred];
        ink.retain_missing_from_ellipse_paint(&paint, &contours, &[]);
        assert_eq!(ink.strokes.len(), 3);
        assert_eq!(ink.summary.suppressed_ellipse_retraces, 1);
        assert_eq!(ink.summary.aligned_ellipse_strokes, 1);
        assert_eq!(ink.summary.recovered_boundary_strokes, 1);
        let coloured = ink
            .strokes
            .iter()
            .find(|s| s.color == [0.0, 0.0, 1.0])
            .unwrap();
        assert_eq!(coloured.width, 3.0);
        assert!(coloured.path_data.is_some());
        for p in &coloured.points {
            assert!(((p.x - 32.0).hypot(p.y - 32.0) - 20.5).abs() < 0.01);
        }

        ink.strokes = vec![retrace.clone()];
        ink.retain_missing_from_ellipse_paint(
            &Raster::blank(width, width, [1.0; 3]),
            &contours,
            &[],
        );
        assert_eq!(ink.strokes.len(), 1, "missing ink must still be restored");
        ink.strokes = vec![retrace];
        ink.retain_missing_from_ellipse_paint(&paint, &contours, &vec![true; width * width]);
        assert_eq!(
            ink.strokes.len(),
            1,
            "transferred source ink must retain its owner"
        );
    }

    #[test]
    fn junction_grouping_selects_the_straight_through_pair() {
        let strokes = vec![
            graph_stroke(&[(-4.0, 0.0), (-2.0, 0.0), (0.0, 0.0)]),
            graph_stroke(&[(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]),
            graph_stroke(&[(0.0, 0.0), (0.0, 2.0), (0.0, 4.0)]),
        ];
        let tangents = graph_continuation_tangents(&strokes);
        assert_eq!(tangents.get(&(0, false)), Some(&Point { x: 1.0, y: 0.0 }));
        assert_eq!(tangents.get(&(1, true)), Some(&Point { x: 1.0, y: 0.0 }));
        assert!(!tangents.contains_key(&(2, true)));
    }

    #[test]
    fn source_profile_refines_a_pixel_centred_skeleton_without_moving_endpoints() {
        let mut source = Raster::blank(9, 9, [1.0; 3]);
        for x in 1..8 {
            source.pixels[3 * 9 + x] = [0.0; 3];
            source.pixels[4 * 9 + x] = [0.45; 3];
        }
        let stroke = graph_stroke(&(1..8).map(|x| (x as f32 + 0.5, 4.5)).collect::<Vec<_>>());
        let refined = refine_stroke_centerline(&stroke, &oklab_pixels(&source), 9, 9);
        assert_eq!(refined[0], stroke.points[0]);
        assert_eq!(
            refined[refined.len() - 1],
            stroke.points[stroke.points.len() - 1]
        );
        assert!(refined[3].y < 4.25, "refined={refined:?}");
        assert!(refined[3].y >= 4.0);
    }

    #[test]
    fn spatial_point_pairs_match_dense_lexicographic_scan() {
        let points = vec![
            Point { x: -4.0, y: 0.0 },
            Point { x: 0.0, y: 0.0 },
            Point { x: 3.0, y: 4.0 },
            Point { x: 4.99, y: 0.0 },
            Point { x: 5.01, y: 0.0 },
            Point { x: 12.0, y: 8.0 },
        ];
        let radius = 5.0;
        let dense = (0..points.len())
            .flat_map(|first| {
                let points = &points;
                (first + 1..points.len()).filter_map(move |second| {
                    (points[first].distance(points[second]) <= radius).then_some((first, second))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(nearby_point_pairs(&points, radius), dense);
    }

    #[test]
    fn short_dark_boundary_undershoot_stays_paint_owned() {
        let source_lab = dark_boundary_profile(18.0);
        assert!(paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(10),
            &source_lab,
            18,
            9,
        ));
    }

    #[test]
    fn ordinary_or_long_boundary_ridge_remains_structural() {
        let light_incident_face = dark_boundary_profile(70.0);
        assert!(!paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(10),
            &light_incident_face,
            18,
            9,
        ));

        let dark_incident_faces = dark_boundary_profile(18.0);
        assert!(!paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(14),
            &dark_incident_faces,
            18,
            9,
        ));
    }

    #[test]
    fn bright_boundary_ridge_uses_positive_profile_polarity() {
        let width = 18;
        let height = 9;
        let mut source = Raster::blank(width, height, [0.18; 3]);
        for x in 4..14 {
            source.pixels[4 * width + x] = [0.92; 3];
        }
        let rendered = Raster::blank(width, height, [0.18; 3]);
        let stroke = StructuralStroke {
            points: (4..14)
                .map(|x| Point {
                    x: x as f32 + 0.5,
                    y: 4.5,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: [0.92; 3],
            width: 1.2,
            role: "bright-ridge-on-boundary",
            width_samples: Vec::new(),
        };
        for role in ["ridge", "ridge-on-boundary"] {
            let mut untyped = stroke.clone();
            untyped.role = role;
            assert_eq!(
                classify_boundary_role(&untyped, &oklab_pixels(&source), width, height),
                "bright-ridge-on-boundary"
            );
            let inverse = Raster::new(
                width,
                height,
                source.pixels.iter().map(|p| p.map(|c| 1.0 - c)).collect(),
            );
            assert_eq!(
                classify_boundary_role(&untyped, &oklab_pixels(&inverse), width, height),
                role
            );
        }
        let (missing, supported) = boundary_profile_flags(
            &stroke,
            &oklab_pixels(&source),
            &oklab_pixels(&rendered),
            width,
            height,
            5.0,
        );

        assert!(missing.iter().filter(|&&value| value).count() >= 8);
        assert!(supported.iter().all(|&value| value));
        assert!(sample_graph_color(&source, &stroke)[0] > 0.8);
    }

    #[test]
    fn thinning_keeps_a_line_connected() {
        let mut mask = vec![false; 32 * 16];
        for y in 6..10 {
            for x in 3..29 {
                mask[y * 32 + x] = true;
            }
        }
        let thin = skeletonize(&mask, 32, 16);
        assert!(thin.iter().filter(|&&v| v).count() >= 20);
        assert!(thin.iter().filter(|&&v| v).count() < 35);
    }

    #[test]
    fn underpaint_uses_one_incident_face_without_mixing_sides() {
        let mut source = Raster::blank(5, 3, [1.0, 0.0, 0.0]);
        for y in 0..3 {
            for x in 3..5 {
                source.pixels[y * 5 + x] = [0.0, 0.0, 1.0];
            }
        }
        let mut mask = vec![false; 15];
        for y in 0..3 {
            mask[y * 5 + 2] = true;
        }
        let result = nearest_underpaint(&source, &mask);
        for y in 0..3 {
            let colour = result.pixels[y * 5 + 2];
            assert!(colour == [1.0, 0.0, 0.0] || colour == [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn structural_antialias_shoulder_is_returned_to_paint() {
        let mut source = Raster::blank(7, 5, [1.0, 1.0, 1.0]);
        let mut structural = vec![false; 35];
        for y in 1..4 {
            structural[y * 7 + 3] = true;
            source.pixels[y * 7 + 3] = [0.0, 0.0, 0.0];
            source.pixels[y * 7 + 4] = [0.5, 0.5, 0.5];
        }
        let mut underpaint = nearest_underpaint(&source, &structural);
        let selected = unmix_structural_antialias(&source, &mut underpaint, &structural);
        assert!(selected.iter().filter(|&&value| value).count() >= 3);
        for y in 1..4 {
            assert!(underpaint.pixels[y * 7 + 4][0] > 0.9);
        }
    }
}
