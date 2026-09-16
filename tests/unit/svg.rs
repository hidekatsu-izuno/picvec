fn batch_equal_paint_paths(elements: &mut [Option<PaintElement>], summary: &mut SvgSummary) {
    batch_equal_paint_paths_impl(elements, summary, &mut Vec::new());
}

mod tests {
    use super::*;

    fn path(data: &str, bbox: (f64, f64, f64, f64), attributes: &str) -> Option<PaintElement> {
        Some(PaintElement {
            geometry: OptimizedElement::Path {
                data: data.to_string(),
                bbox: Some(bbox),
            },
            attributes: attrs([("fill", attributes.into())]),
            batchable: true,
        })
    }

    #[test]
    fn hole_trials_preserve_serialization_across_geometry_and_alpha_changes() {
        use crate::geometry::Point;
        let gradient = Paint::Radial {
            origin: crate::gradient::RadialOrigin::Fitted,
            center: Point { x: 16.0, y: 16.0 },
            radius: Point { x: 12.0, y: 8.0 },
            rotation: 0.3,
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.1, 0.2, 0.3],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.7, 0.8, 0.9],
                },
            ],
        };
        let overlay = PaintOverlay {
            paint: Box::new(gradient.clone()),
            opacity_stops: vec![
                OpacityStop {
                    offset: 0.0,
                    opacity: 0.65,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let mut transparent_overlay = overlay.clone();
        for stop in &mut transparent_overlay.opacity_stops {
            stop.opacity = 0.0;
        }
        let paints = vec![
            gradient.clone(),
            gradient.clone(),
            Paint::Layered {
                base: Box::new(gradient.clone()),
                overlays: vec![overlay, transparent_overlay],
            },
            Paint::Solid {
                color: [0.2, 0.3, 0.4],
            },
            gradient,
        ];
        let mut geometry: Vec<_> = (0..paints.len())
            .map(|i| RegionGeometry {
                region: i as u32,
                loops: vec![],
                path_data: format!("M{} 0h24v24h-24Z", i * 32),
                occlusion_path_data: Some(format!(
                    "M{} 0h24v24h-24Z M{} 8h8v8h-8Z",
                    i * 32,
                    i * 32 + 8
                )),
                covered_hole_paths: vec![],
                primitive: None,
            })
            .collect();
        let excluded = [false, false, false, false, true];
        let alpha = crate::face_alpha::FaceAlpha {
            fields: vec![Paint::Solid { color: [0.5; 3] }; paints.len()],
            ink_opacity: 0.5,
            bands: vec![],
            composite_layers: vec![],
            source_fields: vec![],
        };
        for outlined in [false, true] {
            let mut structural = StructuralInk::empty();
            if outlined {
                structural.outlines.push(crate::outline::OutlineBand {
                    outer: "M0 0h160v32h-160Z".into(),
                    inner: "M1 1h158v30H1Z".into(),
                    underpaint: Paint::Solid { color: [0.8; 3] },
                    inner_underpaint: None,
                    patches: vec![],
                    regions: [1, 3].into_iter().collect(),
                    hidden: [3].into_iter().collect(),
                    boundary_underpaint: [1].into_iter().collect(),
                    contour: vec![],
                    width: 1.0,
                    pixels: vec![],
                });
            }
            for alpha in [None, Some(&alpha)] {
                let mut cache = GeometryCache::default();
                let mut serialize = hole_serializer(
                    160,
                    32,
                    &paints,
                    &structural,
                    0.3,
                    &excluded,
                    alpha,
                    &mut cache,
                );
                // Trials change paths, revert an earlier edit, change primitive
                // type and drawing order, and omit geometry entirely.
                for trial in 0..6 {
                    geometry[0].occlusion_path_data = Some(
                        match trial {
                            0 | 3 => "M0 0h24v24H0Z M8 8h8v8H8Z",
                            1 | 4 => "M0 0h24v24H0Z",
                            _ => "",
                        }
                        .into(),
                    );
                    geometry[2].primitive = (trial == 2).then_some(Primitive::Rect {
                        x: 64.0,
                        y: 0.0,
                        width: 24.0,
                        height: 24.0,
                    });
                    let mut g = geometry.clone();
                    if trial == 4 {
                        g.reverse();
                    }
                    if trial == 5 {
                        g.clear();
                    }
                    let expected = serialize_filtered_with_alpha(
                        160,
                        32,
                        &g,
                        &paints,
                        &structural,
                        0.3,
                        true,
                        &excluded,
                        alpha,
                    );
                    let actual = serialize(&g);
                    assert_eq!(actual.0, expected.0, "outlined={outlined}, trial={trial}");
                    assert_eq!(
                        serde_json::to_value(actual.1).unwrap(),
                        serde_json::to_value(expected.1).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn cached_stroke_geometry_and_outlines_keep_cap_and_width_context() {
        let mut cache = GeometryCache::default();
        for path in [
            "M0 0L24 0L24 24Z",
            "M1.25 2.75C13 2.75 13 19 25 19",
            "M0 0L0 0",
        ] {
            for _ in 0..2 {
                assert_eq!(
                    format!("{:?}", cache.optimize(path)),
                    format!("{:?}", optimize_path(path, true, false))
                );
                assert_eq!(
                    format!("{:?}", cache.optimize_stroke(path)),
                    format!("{:?}", optimize_path(path, true, true))
                );
                for width in [0.13, 1.0, 5.75] {
                    for butt in [false, true] {
                        assert_eq!(
                            cache.stroke_outline(path, width, butt),
                            stroke_outline(path, width, butt)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_face_contours_fill_without_closing_oriented_holes() {
        // Two expanded lobes of the same face overlap at x=15..17. The
        // counterclockwise inner loop is a real hole and must stay transparent.
        let geometry = RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: "M2 2H17V30H2Z M15 2H30V30H15Z M6 10V22H12V10Z".into(),
            occlusion_path_data: None,
            covered_hole_paths: vec![],
            primitive: None,
        };
        for opacity in [1.0, 0.5] {
            let alpha = crate::face_alpha::FaceAlpha {
                fields: vec![Paint::Solid {
                    color: [opacity; 3],
                }],
                ink_opacity: 0.0,
                bands: vec![],
                composite_layers: vec![],
                source_fields: vec![],
            };
            let (document, _) = serialize_filtered_with_alpha(
                32,
                32,
                std::slice::from_ref(&geometry),
                &[Paint::Solid {
                    color: [1.0, 0.0, 0.0],
                }],
                &StructuralInk::empty(),
                0.3,
                false,
                &[false],
                (opacity < 1.0).then_some(&alpha),
            );
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            for scale in [1.0, 3.3, 8.0] {
                let size = (32.0_f32 * scale).ceil() as u32;
                let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size).unwrap();
                resvg::render(
                    &tree,
                    resvg::tiny_skia::Transform::from_scale(scale, scale),
                    &mut pixmap.as_mut(),
                );
                for (x, y, filled) in [
                    (16.0, 16.0, true),
                    (24.0, 16.0, true),
                    (9.0, 16.0, false),
                    (0.0, 16.0, false),
                ] {
                    let pixel = pixmap
                        .pixel((x * scale) as u32, (y * scale) as u32)
                        .unwrap();
                    let expected = if filled {
                        (opacity * 255.0).round() as u8
                    } else {
                        0
                    };
                    assert!(
                        pixel.alpha().abs_diff(expected) <= 1,
                        "scale={scale}, opacity={opacity}, x={x}: {pixel:?}"
                    );
                    assert_eq!(pixel.red(), pixel.alpha());
                    assert_eq!(pixel.green(), 0);
                    assert_eq!(pixel.blue(), 0);
                }
            }
        }
    }

    #[test]
    fn zero_alpha_objects_and_overlays_are_not_serialized() {
        let geometry = RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: "M0 0H20V20H0Z".into(),
            occlusion_path_data: None,
            covered_hole_paths: Vec::new(),
            primitive: None,
        };
        let paints = vec![Paint::Layered {
            base: Box::new(Paint::Solid {
                color: [1.0, 0.0, 0.0],
            }),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Solid {
                    color: [0.0, 0.0, 1.0],
                }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 0.0,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        }];
        let mask = crate::face_alpha::FaceAlpha {
            fields: vec![Paint::Solid { color: [0.0; 3] }],
            ink_opacity: 0.0,
            bands: vec![],
            composite_layers: vec![],
            source_fields: vec![],
        };
        let (transparent, report) = serialize_filtered_with_alpha(
            24,
            24,
            std::slice::from_ref(&geometry),
            &paints,
            &StructuralInk::empty(),
            0.0,
            false,
            &[false],
            Some(&mask),
        );
        assert!(!transparent.contains("<path"));
        assert_eq!(report.path_elements, 0);
        let (opaque, report) = serialize_filtered_with_alpha(
            24,
            24,
            &[geometry],
            &paints,
            &StructuralInk::empty(),
            0.0,
            false,
            &[false],
            None,
        );
        assert_eq!(report.path_elements + report.rect_elements, 1);
        assert!(!opaque.contains("#0000ff"));
    }

    #[test]
    fn authored_alpha_boundary_does_not_add_a_bright_fringe() {
        for reversed in [false, true] {
            let mut geometries = vec![
                RegionGeometry {
                    region: 0,
                    loops: vec![],
                    path_data: "M0 0H20V20H0Z".into(),
                    occlusion_path_data: None,
                    covered_hole_paths: Vec::new(),
                    primitive: None,
                },
                RegionGeometry {
                    region: 1,
                    loops: vec![],
                    path_data: "M20 0H40V20H20Z".into(),
                    occlusion_path_data: None,
                    covered_hole_paths: Vec::new(),
                    primitive: None,
                },
            ];
            if reversed {
                geometries.reverse();
            }
            let alpha = crate::face_alpha::FaceAlpha {
                fields: vec![
                    Paint::Solid {
                        color: [170.0 / 255.0; 3],
                    },
                    Paint::Solid {
                        color: [204.0 / 255.0; 3],
                    },
                ],
                ink_opacity: 0.0,
                bands: vec![],
                composite_layers: vec![],
                source_fields: vec![],
            };
            let (document, _) = serialize_filtered_with_alpha(
                40,
                20,
                &geometries,
                &[
                    Paint::Solid { color: [1.0; 3] },
                    Paint::Solid {
                        color: [213.0 / 255.0; 3],
                    },
                ],
                &StructuralInk::empty(),
                0.0,
                false,
                &[false, false],
                Some(&alpha),
            );
            assert!(!document.contains("<mask"));
            assert!(document.contains("fill-opacity="));
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            for background in [
                resvg::tiny_skia::Color::BLACK,
                resvg::tiny_skia::Color::WHITE,
            ] {
                let mut pixmap = resvg::tiny_skia::Pixmap::new(320, 160).unwrap();
                pixmap.fill(background);
                resvg::render(
                    &tree,
                    resvg::tiny_skia::Transform::from_scale(8.0, 8.0),
                    &mut pixmap.as_mut(),
                );
                for x in 144..176 {
                    let expected = if background == resvg::tiny_skia::Color::BLACK {
                        170
                    } else if x < 160 {
                        255
                    } else {
                        221
                    };
                    let actual = pixmap.pixel(x, 80).unwrap().red();
                    assert!(
                        (actual as i32 - expected).abs() <= 1,
                        "reversed={reversed}, x={x}: {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotated_radial_gradient_renders_its_major_axis_and_has_a_distinct_key() {
        let mut paint = Paint::Radial {
            origin: crate::gradient::RadialOrigin::Fitted,
            center: crate::geometry::Point { x: 16.0, y: 16.0 },
            radius: crate::geometry::Point { x: 12.0, y: 4.0 },
            rotation: std::f32::consts::FRAC_PI_4,
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.0; 3],
                },
                ColorStop {
                    offset: 1.0,
                    color: [1.0; 3],
                },
            ],
        };
        let key = paint_key(&paint).unwrap();
        let mut ids = HashMap::new();
        let mut definitions = Elements::new();
        register_gradient(
            &paint,
            None,
            key.clone(),
            &mut ids,
            &mut definitions,
            &mut SvgSummary::default(),
        );
        let document=format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"32\" height=\"32\"><defs>{definitions}</defs><rect width=\"32\" height=\"32\" fill=\"url(#{})\"/></svg>",ids[&key]);
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        assert!(pixmap.pixel(22, 22).unwrap().red() < 210);
        assert!(pixmap.pixel(22, 9).unwrap().red() > 245);
        if let Paint::Radial { rotation, .. } = &mut paint {
            *rotation = 0.0;
        }
        assert_ne!(paint_key(&paint).unwrap(), key);
        let unrotated_key = paint_key(&paint).unwrap();
        if let Paint::Radial { rotation, .. } = &mut paint {
            *rotation = 0.0004;
        }
        assert_ne!(paint_key(&paint).unwrap(), unrotated_key);
    }

    #[test]
    fn shared_edge_does_not_step_when_paint_order_changes() {
        let mut elements = Vec::new();
        for (x, y, width, height, color) in [
            (0.0, 0.0, 20.0, 10.25, [0.02; 3]),
            (0.0, 10.25, 40.0, 9.75, [1.0; 3]),
            (20.0, 0.0, 20.0, 10.25, [0.03; 3]),
        ] {
            append_paint_elements(
                &mut elements,
                OptimizedElement::Rect {
                    x,
                    y,
                    width,
                    height,
                },
                &Paint::Solid { color },
                &HashMap::new(),
                0.3,
            );
        }
        let mut body = Elements::new();
        write_paint_elements(&mut body, &elements, &mut SvgSummary::default());
        let document = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40\" height=\"20\">{body}</svg>"
        );
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(320, 160).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_scale(8.0, 8.0),
            &mut pixmap.as_mut(),
        );
        for x in [80, 240] {
            assert!(pixmap.pixel(x, 80).unwrap().red() < 30);
            assert!(pixmap.pixel(x, 83).unwrap().red() > 245);
            for y in 79..85 {
                assert_eq!(pixmap.pixel(x, y).unwrap().alpha(), 255);
            }
        }
    }

    #[test]
    fn batch_plan_reuses_decisions_but_keeps_current_paths_and_paint_barriers() {
        let mut cache = GeometryCache::default();
        for version in [0, 1, 2, 3, 4, 0, 1] {
            let mut elements = vec![
                path("M0 0H4V4H0Z", (0.0, 0.0, 4.0, 4.0), "red"),
                path("M20 0H24V4H20Z", (20.0, 0.0, 24.0, 4.0), "blue"),
                path(
                    if version == 1 {
                        "M40 0H44V4H40Z M41 1V2H42V1Z"
                    } else {
                        "M40 0H44V4H40Z"
                    },
                    (40.0, 0.0, 44.0, 4.0),
                    "red",
                ),
            ];
            if version == 2 {
                elements[1].as_mut().unwrap().batchable = false;
            }
            if version == 3 {
                elements[2].as_mut().unwrap().attributes = attrs([("fill", "blue".into())]);
            }
            if version == 4 {
                elements[1].as_mut().unwrap().geometry = OptimizedElement::Rect {
                    x: 20.0,
                    y: 0.0,
                    width: 4.0,
                    height: 4.0,
                };
            }
            let mut reference = elements.clone();
            let mut expected = SvgSummary::default();
            let mut actual = SvgSummary::default();
            batch_equal_paint_paths(&mut reference, &mut expected);
            cache.batch(&mut elements, &mut actual);
            let mut a = Elements::new();
            let mut b = Elements::new();
            write_paint_elements(&mut a, &reference, &mut expected);
            write_paint_elements(&mut b, &elements, &mut actual);
            assert_eq!(a, b);
            assert_eq!(
                serde_json::to_value(expected).unwrap(),
                serde_json::to_value(actual).unwrap()
            );
        }
    }

    #[test]
    fn equal_paint_crosses_only_spatially_disjoint_intervening_elements() {
        let mut elements = vec![
            path("M0 0L4 0L4 4L0 4Z", (0.0, 0.0, 4.0, 4.0), "red"),
            path("M20 0L24 0L24 4L20 4Z", (20.0, 0.0, 24.0, 4.0), "blue"),
            path("M40 0L44 0L44 4L40 4Z", (40.0, 0.0, 44.0, 4.0), "red"),
        ];
        let mut summary = SvgSummary::default();
        batch_equal_paint_paths(&mut elements, &mut summary);
        assert!(elements[2].is_none());
        assert_eq!(summary.paint_paths_merged, 1);

        let mut blocked = vec![
            path("M0 0L4 0L4 4L0 4Z", (0.0, 0.0, 4.0, 4.0), "red"),
            path("M10 0L16 0L16 4L10 4Z", (10.0, 0.0, 16.0, 4.0), "blue"),
            path("M8 0L12 0L12 4L8 4Z", (8.0, 0.0, 12.0, 4.0), "red"),
        ];
        let mut summary = SvgSummary::default();
        batch_equal_paint_paths(&mut blocked, &mut summary);
        assert!(blocked[2].is_some());
        assert_eq!(summary.paint_paths_merged, 0);
    }

    #[test]
    fn layered_paint_emits_ordered_transparent_gradient_components() {
        let overlay = PaintOverlay {
            paint: Box::new(Paint::Radial {
                rotation: 0.0,
                origin: crate::gradient::RadialOrigin::Fitted,
                center: crate::geometry::Point { x: 5.0, y: 5.0 },
                radius: crate::geometry::Point { x: 4.0, y: 3.0 },
                stops: vec![
                    ColorStop {
                        offset: 0.0,
                        color: [1.0, 0.0, 0.0],
                    },
                    ColorStop {
                        offset: 1.0,
                        color: [1.0, 0.0, 0.0],
                    },
                ],
            }),
            opacity_stops: vec![
                OpacityStop {
                    offset: 0.0,
                    opacity: 0.7,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let mut ids = HashMap::new();
        let mut definitions = Elements::new();
        let mut summary = SvgSummary::default();
        register_gradient(
            &overlay.paint,
            Some(&overlay.opacity_stops),
            overlay_key(&overlay),
            &mut ids,
            &mut definitions,
            &mut summary,
        );
        assert!(definitions.to_string().contains("stop-opacity=\"0.7\""));
        assert!(definitions.to_string().contains("stop-opacity=\"0\""));

        let paint = Paint::Layered {
            base: Box::new(Paint::Solid {
                color: [0.2, 0.3, 0.4],
            }),
            overlays: vec![overlay],
        };
        let mut elements = Vec::new();
        append_paint_elements(
            &mut elements,
            OptimizedElement::Path {
                data: "M0 0L10 0L10 10Z".to_string(),
                bbox: Some((0.0, 0.0, 10.0, 10.0)),
            },
            &paint,
            &ids,
            0.2,
        );
        assert_eq!(elements.len(), 2);
        assert!(elements.iter().flatten().all(|element| !element.batchable));
    }
}
