fn gradient_error(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    stops: &[ColorStop],
) -> ErrorStats {
    assert_eq!(samples.len(), parameters.len());
    if samples.is_empty() {
        return ErrorStats {
            mean: 0.0,
            percentile: 0.0,
        };
    }
    let references = samples
        .iter()
        .map(|&index| source.pixels[index])
        .collect::<Vec<_>>();
    let rendered = parameters
        .iter()
        .map(|&parameter| interpolate(stops, parameter))
        .collect::<Vec<_>>();
    let reference_lab = preprocess_color_values(references);
    let rendered_lab = preprocess_color_values(rendered);
    let errors = delta_e_ok_pairs(&reference_lab, &rendered_lab);
    ErrorStats {
        mean: numpy_sum_f32(&errors) / errors.len() as f32,
        percentile: percentile(errors, 0.90),
    }
}

/// Remove a final Paint interface only when both its geometry and colour are
/// unsupported by the native source.
///
/// This is intentionally downstream of Paint fitting.  Quantizer labels are
/// topology owners, so mere colour similarity is insufficient: every
/// connected interface must be below a native-source JND, the two emitted
/// Paints must already agree at that interface, and one editable Paint made
/// from Office-compatible gradient components must preserve the measured
/// error on both incident faces.  The
/// accepted labels are compacted before shared geometry is rebuilt, which
/// removes the obsolete master curve instead of hiding it with an overlay.
pub(crate) fn merge_source_supported_paints(
    source: &Raster,
    boundary_source: &Raster,
    segmentation: &mut Segmentation,
    paints: &mut Vec<Paint>,
    config: &Config,
) -> SupportedPaintMergeReport {
    let source_labs = oklab_pixels(source);
    let boundary_labs = oklab_pixels(boundary_source);
    let boundaries = smooth_paint_boundaries(&boundary_labs, segmentation, 2, true);
    merge_source_supported_paints_with_evidence(
        source,
        boundary_source,
        segmentation,
        paints,
        config,
        PaintEvidence {
            source_labs,
            boundary_labs,
            boundaries,
        },
    )
}

mod tests {
    #[test]
    fn zero_opacity_residual_validation_matches_full_sample_checks() {
        let width = 80;
        let height = 64;
        let source = Raster::new(
            width,
            height,
            (0..width * height)
                .map(|i| {
                    let x = (i % width) as f32;
                    let y = (i / width) as f32;
                    let field = 0.4
                        + x / 400.0
                        + 0.12 * (-((x - 30.0).powi(2) + (y - 29.0).powi(2)) / 180.0).exp();
                    [field, field * 0.97, field * 0.92]
                })
                .collect(),
        );
        let validation: Vec<_> = (0..width * height).collect();
        let samples = sampled_indices(&validation, 256);
        for preserve_shape in [false, true] {
            for layers in [1, 3] {
                let base = Paint::Solid {
                    color: [0.5, 0.48, 0.46],
                };
                let reference = fit_residual_paint_impl(
                    &source,
                    &samples,
                    &validation,
                    bounds(&validation, width),
                    base.clone(),
                    layers,
                    preserve_shape,
                    false,
                );
                let candidate = fit_residual_paint_impl(
                    &source,
                    &samples,
                    &validation,
                    bounds(&validation, width),
                    base,
                    layers,
                    preserve_shape,
                    true,
                );
                assert_eq!(candidate.0, reference.0);
                assert_eq!(format!("{:?}", candidate.1), format!("{:?}", reference.1));
            }
        }
    }

    #[test]
    fn orthogonal_colour_and_opacity_are_not_forced_onto_one_axis() {
        let (w, h) = (32, 32);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| [0.2 + 0.6 * (i % w) as f32 / 31.0; 3])
                .collect(),
        );
        let alpha = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| [0.2 + 0.6 * (i / w) as f32 / 31.0; 3])
                .collect(),
        );
        let pixels: Vec<_> = (0..w * h).collect();
        let colour = fit_alpha_field(&source, &pixels).unwrap();
        let field = fit_alpha_field(&alpha, &pixels).unwrap();
        assert!(fit_alpha_on_paint(&alpha, &pixels, &colour).is_none());
        assert!(fit_colour_on_alpha(&source, &pixels, &colour, &field).is_none());
    }

    #[test]
    fn penguin_highlight_can_share_an_rgba_coordinate() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../data/cliparts-highlight-field.json")).unwrap();
        let image = image::load_from_memory(include_bytes!("../data/cliparts-highlight-field.png"))
            .unwrap()
            .to_rgba8();
        let (w, h) = image.dimensions();
        let source = Raster::new(
            w as usize,
            h as usize,
            image
                .pixels()
                .map(|p| [p[0], p[1], p[2]].map(|v| v as f32 / 255.0))
                .collect(),
        );
        let alpha = Raster::new(
            w as usize,
            h as usize,
            image.pixels().map(|p| [p[3] as f32 / 255.0; 3]).collect(),
        );
        let point = |name: &str| Point {
            x: data[name][0].as_f64().unwrap() as f32,
            y: data[name][1].as_f64().unwrap() as f32,
        };
        let original = Paint::Radial {
            origin: RadialOrigin::Fitted,
            center: point("center"),
            radius: point("radius"),
            rotation: data["rotation"].as_f64().unwrap() as f32,
            stops: data["stops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| ColorStop {
                    offset: s["offset"].as_f64().unwrap(),
                    color: std::array::from_fn(|c| s["color"][c].as_f64().unwrap()),
                })
                .collect(),
        };
        let pixels: Vec<_> = data["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect();
        assert!(fit_alpha_on_paint(&alpha, &pixels, &original).is_none());
        let field = fit_alpha_field(&alpha, &pixels).expect("continuous highlight opacity");
        let colour = fit_colour_on_alpha(&source, &pixels, &original, &field)
            .expect("continuous highlight colour on opacity coordinate");
        assert!(fit_alpha_on_paint(&alpha, &pixels, &colour).is_some());
        assert!(
            paint_rgb_mse(&source, &pixels, &colour).sqrt()
                <= paint_rgb_mse(&source, &pixels, &original).sqrt() + 1.0 / 255.0
        );
    }

    #[test]
    fn merge_keeps_a_local_shadow_even_when_global_error_is_small() {
        let (w, h) = (96, 64);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let x = (i % w) as f32;
                    let y = (i / w) as f32;
                    let shadow = 0.16 * (-((x - 48.0).powi(2) + (y - 32.0).powi(2)) / 100.0).exp();
                    [0.65 - shadow, 0.08, 0.09]
                })
                .collect(),
        );
        let pixels: Vec<_> = (0..w * h).collect();
        let flattened = Paint::Solid {
            color: [0.65, 0.08, 0.09],
        };
        assert!(paint_stats(&source, &pixels, &flattened).mean < 1.0);
        assert!(!merge_preserves_local_shading(
            &source, &pixels, &flattened, 2.7
        ));
        let uniform = Raster::new(w, h, vec![[0.65, 0.08, 0.09]; w * h]);
        assert!(merge_preserves_local_shading(
            &uniform, &pixels, &flattened, 2.7
        ));
    }

    #[test]
    fn tilted_highlight_band_keeps_its_radial_gradient() {
        let (w, h) = (100, 80);
        let (center, radius, rotation) =
            (Point { x: 52.0, y: 68.0 }, Point { x: 90.0, y: 25.0 }, -0.6);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let t = rotated_radial_parameter(i, w, center, radius, rotation);
                    [0.85 - 0.08 * t, 0.55 - 0.12 * t, 0.5 - 0.1 * t]
                })
                .collect(),
        );
        let samples: Vec<_> = (0..w * h)
            .filter(|&i| {
                (0.9..1.2).contains(&rotated_radial_parameter(i, w, center, radius, rotation))
            })
            .collect();
        let (paint, stats) = office_gradient_candidate_with_labs(
            &source,
            &samples
                .iter()
                .map(|&i| rgb_to_oklab(source.pixels[i]))
                .collect::<Vec<_>>(),
            &samples,
            bounds(&samples, w),
            5,
        )
        .unwrap();
        assert!(
            stats.mean < 0.1,
            "tilted band flattened: {stats:?} {paint:?}"
        );
        assert!(matches!(paint, Paint::Radial { .. }));
    }

    #[test]
    fn merged_material_recovers_shading_without_reintroducing_overlay_faces() {
        let (w, h) = (64, 48);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let v = 0.2 + 0.3 * (i / w) as f32 / (h - 1) as f32;
                    [0.7 + v * 0.3, v, v * 0.8]
                })
                .collect(),
        );
        let mut seg = two_face_segmentation(&source);
        seg.paint_samples.fill(false);
        let mut paints = vec![
            Paint::Solid {
                color: [0.8, 0.35, 0.28]
            };
            2
        ];
        assert_eq!(restore_interior_shading(&source, &seg, &mut paints), 2);
        for x in [16, 48] {
            for y in [8, 24, 40] {
                let i = y * w + x;
                assert!(
                    delta_e_ok(
                        rgb_to_oklab(source.pixels[i]),
                        rgb_to_oklab(paint_at(&paints[seg.labels[i] as usize], i, w))
                    ) < 0.5
                );
            }
        }
        assert!(paints
            .iter()
            .all(|p| matches!(p, Paint::Linear { .. } | Paint::Radial { .. })));
    }

    #[test]
    fn residual_layers_require_more_than_a_small_local_colour_correction() {
        let samples: Vec<_> = (0..48 * 32).collect();
        for contrast in [0.015_f32, 0.25] {
            let base = Paint::Solid { color: [0.5; 3] };
            let source = Raster::new(
                48,
                32,
                samples
                    .iter()
                    .map(|&i| {
                        let radius = (((i % 48) as f32 - 24.0).powi(2)
                            + ((i / 48) as f32 - 16.0).powi(2))
                        .sqrt()
                            / 8.0;
                        [0.5 + contrast * (1.0 - radius).max(0.0); 3]
                    })
                    .collect(),
            );
            let (paint, _) =
                fit_layered_residual_paint(&source, &samples, bounds(&samples, 48), base, 3);
            if contrast < 0.02 {
                assert!(
                    !matches!(paint, Paint::Layered { .. }),
                    "minor colour correction created duplicate geometry"
                );
            } else {
                assert!(
                    paint_at(&paint, 16 * 48 + 24, 48)[0] > 0.7
                        && paint_at(&paint, 0, 48)[0] < 0.55,
                    "a visible local highlight must remain"
                );
            }
        }
    }

    #[test]
    fn nonlinear_residual_profile_reuses_one_gradient_instead_of_layering() {
        let (w, h) = (48, 40);
        let samples: Vec<_> = (0..w * h).collect();
        for degrees in [23.0_f32, 73.0] {
            let angle = degrees.to_radians();
            let (start, end) = fitted_linear_geometry(&samples, w, (angle.cos(), angle.sin()));
            let field = Paint::Linear {
                preset: LinearPreset::Fitted,
                start,
                end,
                stops: [0.78, 0.68, 0.83, 0.68, 0.78]
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| ColorStop {
                        offset: i as f64 / 4.0,
                        color: [value; 3],
                    })
                    .collect(),
            };
            let source = Raster::new(
                w,
                h,
                samples.iter().map(|&i| paint_at(&field, i, w)).collect(),
            );
            let layered = Paint::Layered {
                base: Box::new(Paint::Solid { color: [0.75; 3] }),
                overlays: vec![PaintOverlay {
                    paint: Box::new(field),
                    opacity_stops: vec![
                        OpacityStop {
                            offset: 0.0,
                            opacity: 1.0,
                        },
                        OpacityStop {
                            offset: 1.0,
                            opacity: 1.0,
                        },
                    ],
                }],
            };
            let single = refit_single_residual_field(&source, &samples, &samples, &layered)
                .expect("nonlinear profile needs a better axis fit, not another path");
            assert!(!matches!(single, Paint::Layered { .. }));
            assert!(samples.iter().all(|&i| delta_e_ok(
                rgb_to_oklab(source.pixels[i]),
                rgb_to_oklab(paint_at(&single, i, w))
            ) <= 1.5));
        }
    }

    #[test]
    fn weak_dark_teal_residual_does_not_start_an_overlay_chain() {
        let (w, h) = (48, 48);
        let samples: Vec<_> = (0..w * h).collect();
        let colour = [1.0 / 255.0, 68.0 / 255.0, 54.0 / 255.0];
        let source = Raster::new(
            w,
            h,
            samples
                .iter()
                .map(|&i| {
                    let radius =
                        (((i % w) as f32 - 24.0).powi(2) + ((i / w) as f32 - 24.0).powi(2)).sqrt()
                            / 6.0;
                    let t = (1.0 - radius).max(0.0);
                    [
                        colour[0],
                        colour[1] + t * 10.0 / 255.0,
                        colour[2] + t * 8.0 / 255.0,
                    ]
                })
                .collect(),
        );
        let (paint, _) = fit_layered_residual_paint(
            &source,
            &samples,
            bounds(&samples, w),
            Paint::Solid { color: colour },
            3,
        );
        assert!(
            !matches!(paint, Paint::Layered { .. }),
            "weak residual must be rejected before creating a layered paint"
        );
    }

    #[test]
    fn dark_edge_outliers_do_not_cast_a_shadow_into_supported_paint() {
        let blue = [0.65, 0.83, 0.91];
        let mut source = Raster::blank(80, 40, blue);
        let samples: Vec<_> = (0..80 * 40).collect();
        for y in 0..2 {
            for x in 14..19 {
                source.pixels[y * 80 + x] = [0.2, 0.25, 0.28];
            }
        }
        // Also cover a base with a modest colour bias: those pixels still
        // must not be sacrificed to explain a few edge outliers.
        for offset in [0.0_f32, -0.05] {
            let color = blue.map(|c| c + offset);
            let baseline_error = delta_e_ok(rgb_to_oklab(blue), rgb_to_oklab(color));
            let base = Paint::Solid { color };
            let (paint, _) =
                fit_layered_residual_paint(&source, &samples, bounds(&samples, 80), base, 8);
            for y in 2..12 {
                for x in 8..26 {
                    let error = delta_e_ok(
                        rgb_to_oklab(blue),
                        rgb_to_oklab(paint_at(&paint, y * 80 + x, 80)),
                    );
                    assert!(
                        error <= baseline_error + 1.0,
                        "edge outliers created a shadow at {x},{y}: {baseline_error} -> {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_residual_fit_checks_pixels_between_observations() {
        let blue = [0.65, 0.83, 0.91];
        let mut source = Raster::blank(80, 40, blue);
        for y in 0..2 {
            for x in 14..19 {
                source.pixels[y * 80 + x] = [0.2, 0.25, 0.28];
            }
        }
        let validation: Vec<_> = (0..80 * 40).collect();
        // Retain the boundary residual but omit its healthy neighbourhood
        // from the optimization observations, as a sparse merge fit can do.
        let samples: Vec<_> = validation
            .iter()
            .copied()
            .filter(|&i| {
                let (x, y) = (i % 80, i / 80);
                (y < 2 && (14..19).contains(&x)) || (x >= 40 && i % 7 == 0)
            })
            .collect();
        let (paint, _) = fit_layered_residual_paint_validated(
            &source,
            &samples,
            &validation,
            bounds(&validation, 80),
            Paint::Solid { color: blue },
            3,
        );
        for y in 2..12 {
            for x in 8..26 {
                let error = delta_e_ok(
                    rgb_to_oklab(blue),
                    rgb_to_oklab(paint_at(&paint, y * 80 + x, 80)),
                );
                assert!(error <= 1.0, "unobserved shadow at {x},{y}: {error}");
            }
        }
    }

    #[test]
    fn residual_corrections_do_not_add_circular_shading_to_a_ramp() {
        let samples: Vec<_> = (0..80 * 40).collect();
        let source = Raster::new(
            80,
            40,
            samples
                .iter()
                .map(|&i| {
                    let t = (i / 80) as f32 / 39.0;
                    [0.95, 0.1 + 0.4 * t, 0.1 + 0.3 * t]
                })
                .collect(),
        );
        let base = Paint::Linear {
            preset: LinearPreset::Fitted,
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 0.0, y: 39.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.95, 0.14, 0.14],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.95, 0.36, 0.32],
                },
            ],
        };
        let (paint, _) = fit_residual_paint(
            &source,
            &samples,
            &samples,
            bounds(&samples, 80),
            base,
            3,
            true,
        );
        for y in 0..40 {
            for x in 0..74 {
                let i = y * 80 + x;
                let step = delta_e_ok(
                    rgb_to_oklab(paint_at(&paint, i, 80)),
                    rgb_to_oklab(paint_at(&paint, i + 6, 80)),
                );
                assert!(
                    step <= 0.501,
                    "a circular correction bent the ramp at {x},{y}: {step}"
                );
            }
        }
    }

    #[test]
    fn supported_local_shadow_survives_residual_validation() {
        let base = Paint::Solid {
            color: [0.65, 0.83, 0.91],
        };
        let reference = Paint::Layered {
            base: Box::new(base.clone()),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Radial {
                    origin: RadialOrigin::Fitted,
                    center: Point { x: 39.0, y: 19.0 },
                    radius: Point { x: 18.96, y: 9.36 },
                    rotation: 0.0,
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: [0.2, 0.3, 0.4],
                        },
                        ColorStop {
                            offset: 1.0,
                            color: [0.2, 0.3, 0.4],
                        },
                    ],
                }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 0.55,
                    },
                    OpacityStop {
                        offset: 0.55,
                        opacity: 0.1925,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        };
        let samples: Vec<_> = (0..80 * 40).collect();
        let source = Raster::new(
            80,
            40,
            samples
                .iter()
                .map(|&i| paint_at(&reference, i, 80))
                .collect(),
        );
        let (paint, _) = fit_residual_paint(
            &source,
            &samples,
            &samples,
            bounds(&samples, 80),
            base,
            8,
            true,
        );
        let center = 19 * 80 + 39;
        assert!(
            paint_at(&paint, center, 80)[0] < 0.5,
            "the real shadow disappeared"
        );
        let target = rgb_to_oklab(source.pixels[center]);
        let before = delta_e_ok(target, rgb_to_oklab([0.65, 0.83, 0.91]));
        let after = delta_e_ok(target, rgb_to_oklab(paint_at(&paint, center, 80)));
        assert!(
            after < before * 0.5,
            "real shadow colour was not recovered: {before} -> {after}"
        );
    }

    #[test]
    fn car_bumper_highlight_keeps_brightening_towards_the_trim() {
        let input = image::load_from_memory(include_bytes!("../data/car-bumper-highlight.png"))
            .unwrap()
            .to_rgb8();
        let mask = image::load_from_memory(include_bytes!("../data/car-bumper-highlight-mask.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            input.width() as usize,
            input.height() as usize,
            input
                .pixels()
                .map(|p| p.0.map(|c| c as f32 / 255.0))
                .collect(),
        );
        let pixels: Vec<_> = mask
            .pixels()
            .enumerate()
            .filter_map(|(i, p)| (p[0] != 0).then_some(i))
            .collect();
        let samples: Vec<_> = mask
            .pixels()
            .enumerate()
            .filter_map(|(i, p)| (p[0] != 0 && p[1] != 0).then_some(i))
            .collect();
        let mean = mean_color(&source, &samples);
        let (paint, _, _) = fit_region_samples(
            1175,
            &source,
            &oklab_pixels(&source),
            &samples,
            pixels.len(),
            bounds(&pixels, source.width),
            mean,
            rgb_to_oklab(mean),
            false,
            true,
            &Config::default(),
        );

        for x in [216_usize, 220] {
            let top = (739 - 715) * source.width + x - 176;
            let bottom = (745 - 715) * source.width + x - 176;
            for (a, b) in [(739, 742), (742, 745)] {
                let a = (a - 715) * source.width + x - 176;
                let b = (b - 715) * source.width + x - 176;
                assert!(
                    rgb_to_oklab(paint_at(&paint, b, source.width)).l + 0.5
                        >= rgb_to_oklab(paint_at(&paint, a, source.width)).l,
                    "local gradient reversal at {x}"
                );
            }
            assert!(
                paint_at(&paint, bottom, source.width)[1] >= paint_at(&paint, top, source.width)[1],
                "highlight fades towards the trim at x={x}"
            );
        }
    }

    #[test]
    fn flat_chroma_quantile_roundoff_does_not_invert_stop_bounds() {
        let neutral = Oklab {
            l: 60.0,
            a: 2.980_232_2e-6,
            b: 1.490_116_1e-6,
        };
        for count in 2..128 {
            let labs = vec![neutral; count];
            let samples = (0..count).collect::<Vec<_>>();
            let parameters = (0..count)
                .map(|i| i as f32 / (count - 1) as f32)
                .collect::<Vec<_>>();
            let stops = legacy_stops(&labs, &samples, &parameters);
            let expected = oklab_values_to_rgb(&[neutral])[0];
            for stop in stops {
                for (actual, expected) in stop.color.into_iter().zip(expected) {
                    assert!((actual - f64::from(expected)).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn a_narrow_observed_profile_keeps_its_slope_without_fabricated_samples() {
        let parameters = (0..64)
            .map(|i| 0.8 + 0.005 * i as f32 / 63.0)
            .collect::<Vec<_>>();
        let source = Raster::new(
            64,
            1,
            parameters
                .iter()
                .map(|&t| [0.2 + 0.6 * (t - 0.8) / 0.005; 3])
                .collect(),
        );
        let stops = fitted_stops(
            &source,
            &(0..64).collect::<Vec<_>>(),
            &parameters,
            &[0.0, 1.0],
        );
        for (i, &t) in parameters.iter().enumerate() {
            let predicted = interpolate(&stops, t);
            assert!(
                (predicted[0] - source.pixels[i][0]).abs() < 1e-4,
                "an unobserved interval flattened the local gradient at {i}: {predicted:?}"
            );
        }
    }

    #[test]
    fn local_mean_preserves_a_large_flat_faces_colour() {
        let color = [233.0 / 255.0; 3];
        let source = Raster::blank(1024, 1024, color);
        let indices = (0..source.pixels.len()).collect::<Vec<_>>();
        let fitted = mean_color_f64(&source, &indices);
        for (actual, expected) in fitted.into_iter().zip(color) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn irregular_stop_spacing_does_not_bend_a_linear_ramp() {
        let parameters = [0.0, 0.1, 0.2, 0.7, 1.0];
        let source = Raster::new(
            5,
            1,
            parameters.iter().map(|&t| [0.2 + 0.6 * t; 3]).collect(),
        );
        let stops = fitted_stops_direct(
            &source,
            &[0, 1, 2, 3, 4],
            &parameters,
            &[0.0, 0.1, 0.2, 1.0],
        );
        for stop in stops {
            let expected = 0.2 + 0.6 * stop.offset;
            assert!(
                (stop.color[0] - expected).abs() < 1e-5,
                "a linear ramp acquired a bend at {}: {} instead of {expected}",
                stop.offset,
                stop.color[0]
            );
        }
    }

    #[test]
    fn spatial_slivers_use_their_own_colour_instead_of_the_parent_palette() {
        check_local_sliver_colours(true);
    }

    #[test]
    fn unsplit_slivers_use_valid_native_samples_instead_of_a_global_palette_colour() {
        check_local_sliver_colours(false);
    }

    fn check_local_sliver_colours(spatial_siblings: bool) {
        let w = 32;
        let source = Raster::new(
            w,
            w,
            (0..w * w)
                .map(|i| {
                    [
                        0.98,
                        0.30 + (i % w) as f32 * 0.002,
                        0.28 + (i / w) as f32 * 0.002,
                    ]
                })
                .collect(),
        );
        let mut segmentation = two_face_segmentation(&source);
        let mut labels = vec![0; w * w];
        let slivers = [
            vec![20 * w + 20],
            vec![24 * w + 24, 24 * w + 25, 25 * w + 25],
        ];
        for (k, pixels) in slivers.iter().enumerate() {
            for &i in pixels {
                labels[i] = k as u32 + 1;
            }
        }
        replace_source_supported_paint_labels(&source, &mut segmentation, labels, 0);
        segmentation.paint_keys = if spatial_siblings {
            vec![0, 1, 1]
        } else {
            vec![0, 1, 2]
        };
        segmentation.canonical.pixels.fill([0.95, 0.25, 0.23]);
        let branches = crate::ridge::StrongRidgeBranches {
            dark: vec![false; w * w],
            bright: vec![false; w * w],
        };
        let (paints, _, _) = fit_all_without_topology(
            &[None, None, None],
            &source,
            &source,
            &segmentation,
            &branches,
            &Config::default(),
        );
        for (k, pixels) in slivers.iter().enumerate() {
            for &i in pixels {
                let fitted = paint_at(&paints[k + 1], i, w);
                for (a, b) in fitted.iter().zip(source.pixels[i]) {
                    assert!(
                        (a - b).abs() < 2.0 / 255.0,
                        "sliver {k}: {fitted:?} != {:?}",
                        source.pixels[i]
                    );
                }
            }
        }
    }

    use super::*;

    #[test]
    fn sparse_outline_fields_keep_the_color_error_gate() {
        let source = Raster::new(4, 1, vec![[0.2, 0.3, 0.4]; 4]);
        for count in 1..=4 {
            let samples: Vec<_> = (0..count).collect();
            assert!(matches!(
                fit_outline_field(&source, &samples, 5.0),
                Some(Paint::Solid { .. })
            ));
        }
        assert!(fit_outline_field(&source, &[], 5.0).is_none());
        let contrasted = Raster::new(2, 1, vec![[0.0; 3], [1.0; 3]]);
        assert!(fit_outline_field(&contrasted, &[0, 1], 5.0).is_none());
    }

    #[test]
    fn primary_gate_keeps_smooth_local_highlights() {
        let source = Raster::new(
            128,
            128,
            (0..128 * 128)
                .map(|i| {
                    let r =
                        (((i % 128) as f32 - 35.0) / 12.0).hypot(((i / 128) as f32 - 24.0) / 8.0);
                    [0.5 + 0.3 * (-r * r).exp(); 3]
                })
                .collect(),
        );
        let samples = (0..128 * 128).collect::<Vec<_>>();
        assert!(
            primary_gradient_coherence(&source, &samples, bounds(&samples, 128), false, 256) > 0.95
        );
    }

    #[test]
    fn seam_adjustment_cannot_hide_local_colour_loss_in_a_group_average() {
        let source = Raster::new(
            128,
            32,
            (0..128 * 32)
                .map(|i| [0.3 + 0.4 * (i % 128) as f32 / 127.0; 3])
                .collect(),
        );
        let samples = (0..128 * 32).collect::<Vec<_>>();
        let before = Paint::Solid { color: [0.3; 3] };
        let after = Paint::Solid { color: [0.5; 3] };
        assert!(
            paint_rgb_mse(&source, &samples, &after) < paint_rgb_mse(&source, &samples, &before)
        );
        assert!(!preserves_local_shading(&source, &samples, &before, &after));
        let exact = differential_linear_paint(&source, &samples).unwrap();
        assert!(preserves_local_shading(&source, &samples, &before, &exact));
    }

    #[test]
    fn narrow_isocolour_band_keeps_the_source_slope() {
        let source = Raster::new(
            128,
            128,
            (0..128 * 128)
                .map(|i| [0.2 + 0.001 * (i % 128 + i / 128) as f32; 3])
                .collect(),
        );
        let samples = (0..128 * 128)
            .filter(|i| (100..108).contains(&(i % 128 + i / 128)))
            .collect::<Vec<_>>();
        let labs = oklab_pixels(&source);
        let mean = mean_color(&source, &samples);
        let (paint, _, _) = fit_region_samples(
            0,
            &source,
            &labs,
            &samples,
            samples.len(),
            bounds(&samples, 128),
            mean,
            rgb_to_oklab(mean),
            false,
            true,
            &Config::default(),
        );
        assert!(matches!(paint, Paint::Linear { .. }), "{paint:?}");
        assert!(paint_rgb_mse(&source, &samples, &paint) < 1e-7);
        // A real material step is never included in the fitting halo.
        let mut stepped = source.clone();
        for i in 0..128 * 128 {
            if i % 128 >= 64 {
                stepped.pixels[i] = [0.8; 3];
            }
        }
        let support = smooth_fit_support(&stepped, &[60 * 128 + 62; 32]);
        assert!(support.iter().all(|i| i % 128 < 64));
    }

    #[test]
    fn radial_normals_recover_a_highlight_centre_outside_the_face() {
        let source = Raster::new(
            128,
            128,
            (0..128 * 128)
                .map(|i| {
                    let r = (((i % 128) as f32 - 64.0) / 110.0)
                        .hypot(((i / 128) as f32 - 150.0) / 130.0);
                    [0.3 + 0.3 * r; 3]
                })
                .collect(),
        );
        let samples = (0..128 * 128)
            .filter(|i| (20..80).contains(&(i / 128)) && (20..108).contains(&(i % 128)))
            .collect::<Vec<_>>();
        let (center, radius) = fitted_radial_geometry(&source, &samples).unwrap();
        assert!((center.x - 64.0).abs() < 0.5, "{center:?}");
        assert!((center.y - 150.0).abs() < 0.5, "{center:?}");
        assert!((radius.x / radius.y - 110.0 / 130.0).abs() < 0.01);
    }

    #[test]
    fn selected_median_matches_sorted_bits() {
        for length in 1..1025 {
            let mut values = (0..length)
                .map(|i| ((i * 73 + length * 31) % 101) as f32 - 50.0)
                .collect::<Vec<_>>();
            if length >= 6 {
                values[..6].copy_from_slice(&[
                    -0.0,
                    0.0,
                    f32::MIN,
                    f32::MAX,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                ]);
            }
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            let middle = length / 2;
            let expected = if length % 2 == 0 {
                0.5 * (sorted[middle - 1] + sorted[middle])
            } else {
                sorted[middle]
            };
            assert_eq!(median(&mut values).to_bits(), expected.to_bits());
        }
        for mut values in [
            vec![-0.0, 0.0],
            vec![0.0, -0.0],
            vec![f32::NAN],
            vec![1.0, f32::NAN, 2.0],
        ] {
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            let middle = sorted.len() / 2;
            let expected = if sorted.len() % 2 == 0 {
                0.5 * (sorted[middle - 1] + sorted[middle])
            } else {
                sorted[middle]
            };
            assert_eq!(median(&mut values).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn reused_merge_errors_preserve_all_gate_outcomes() {
        // Independent reference: compute face statistics, then render both
        // paints again for the combined gate, as before this optimization.
        fn reference(
            labs: &[Oklab],
            width: usize,
            faces: [(&[usize], &Paint); 2],
            candidate: &Paint,
        ) -> u8 {
            let mut baseline_errors = Vec::new();
            let mut candidate_errors = Vec::new();
            for (samples, baseline) in faces {
                let before = paint_stats_against_labs(labs, samples, width, baseline);
                let after = paint_stats_against_labs(labs, samples, width, candidate);
                if after.mean > before.mean + 0.30 || after.percentile > before.percentile + 0.75 {
                    return 1;
                }
                baseline_errors.extend(errors_for_indices(labs, samples, width, baseline));
                candidate_errors.extend(errors_for_indices(labs, samples, width, candidate));
            }
            let before = numpy_sum_f32(&baseline_errors) / baseline_errors.len().max(1) as f32;
            let after = numpy_sum_f32(&candidate_errors) / candidate_errors.len().max(1) as f32;
            if after > before + 0.01
                || percentile(candidate_errors, 0.90) > percentile(baseline_errors, 0.90) + 0.04
            {
                2
            } else {
                0
            }
        }
        let source = Raster::new(
            32,
            32,
            (0..1024)
                .map(|i| [0.3 + (i % 32) as f32 / 200.0; 3])
                .collect(),
        );
        let labs = oklab_pixels(&source);
        let mut outcomes = [0usize; 3];
        for length in [0, 1, 7, 32, 127, 256, 511] {
            let left: Vec<_> = (0..length).collect();
            let right: Vec<_> = (512..512 + length).rev().collect();
            for offset in [-0.01, 0.0, 0.0001, 0.001, 0.002, 0.01, 0.1] {
                let baseline = Paint::Linear {
                    preset: LinearPreset::LeftToRight,
                    start: Point { x: 0.0, y: 0.0 },
                    end: Point { x: 31.0, y: 0.0 },
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: [0.3; 3],
                        },
                        ColorStop {
                            offset: 1.0,
                            color: [0.455; 3],
                        },
                    ],
                };
                let mut candidate = baseline.clone();
                if let Paint::Linear { stops, .. } = &mut candidate {
                    for stop in stops {
                        for c in &mut stop.color {
                            *c += offset;
                        }
                    }
                }
                let faces = [(&left[..], &baseline), (&right[..], &baseline)];
                let expected = reference(&labs, 32, faces, &candidate);
                assert_eq!(
                    supported_merge_error_gate(&labs, 32, faces, &candidate),
                    expected
                );
                outcomes[expected as usize] += 1;
            }
        }
        assert!(outcomes.iter().all(|&count| count > 0), "{outcomes:?}");
    }

    #[test]
    fn cached_merge_observations_keep_both_error_statistics_exact() {
        let source = Raster::new(
            8,
            8,
            (0..64)
                .map(|i| [i as f32 / 64.0, (i % 7) as f32 / 7.0, 0.3])
                .collect(),
        );
        for samples in [vec![], vec![0], (0..64).step_by(3).collect()] {
            let cached = MergePaintSamples::new(&source, &samples);
            let paint = Paint::Solid {
                color: [0.2, 0.5, 0.8],
            };
            let expected = paint_stats(&source, &samples, &paint);
            let actual = cached.paint_stats(&paint);
            assert_eq!(actual.mean.to_bits(), expected.mean.to_bits());
            assert_eq!(actual.percentile.to_bits(), expected.percentile.to_bits());
            let stops = vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.1, 0.3, 0.7],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.9, 0.6, 0.2],
                },
            ];
            let parameters: Vec<_> = samples.iter().map(|&i| i as f32 / 63.0).collect();
            let expected = gradient_error(&source, &samples, &parameters, &stops);
            let actual = cached.gradient_error(&parameters, &stops);
            assert_eq!(actual.mean.to_bits(), expected.mean.to_bits());
            assert_eq!(actual.percentile.to_bits(), expected.percentile.to_bits());
        }
    }

    #[test]
    fn cached_solid_lab_preserves_paint_error_exactly() {
        let source = Raster::new(
            17,
            1,
            (0..17)
                .map(|index| {
                    let amount = index as f32 / 16.0;
                    [
                        0.08 + 0.81 * amount,
                        0.74 - 0.51 * amount,
                        0.19 + 0.37 * amount,
                    ]
                })
                .collect(),
        );
        let source_labs = oklab_pixels(&source);
        let samples = (0..17).collect::<Vec<_>>();
        let solid = [0.31, 0.57, 0.83];
        let rendered_lab = oklab_pixels(&Raster::new(1, 1, vec![solid]))[0];
        let previous = paint_error_against_labs(&source_labs, &samples, |_| solid);
        let cached = constant_paint_error_for_labs(&source_labs, rendered_lab);
        assert_eq!(cached.mean.to_bits(), previous.mean.to_bits());
        assert_eq!(cached.percentile.to_bits(), previous.percentile.to_bits());
    }

    fn two_face_segmentation(source: &Raster) -> Segmentation {
        let labels = (0..source.height)
            .flat_map(|_| (0..source.width).map(|x| u32::from(x >= source.width / 2)))
            .collect::<Vec<_>>();
        let half = source.width / 2;
        let area = half * source.height;
        Segmentation {
            width: source.width,
            height: source.height,
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; source.width * source.height],
            canonical: source.clone(),
            regions: vec![
                crate::segment::RegionStats {
                    id: 0,
                    area,
                    min_x: 0,
                    min_y: 0,
                    max_x: half,
                    max_y: source.height,
                    mean_rgb: source.pixels[0],
                    mean_lab: rgb_to_oklab(source.pixels[0]),
                },
                crate::segment::RegionStats {
                    id: 1,
                    area,
                    min_x: half,
                    min_y: 0,
                    max_x: source.width,
                    max_y: source.height,
                    mean_rgb: source.pixels[half],
                    mean_lab: rgb_to_oklab(source.pixels[half]),
                },
            ],
            summary: crate::segment::SegmentationSummary::default(),
        }
    }

    #[test]
    fn curved_car_highlight_keeps_colour_beyond_the_bright_ridge() {
        // A crop of the native underpaint and its pre-fit ownership. Mask
        // channels are face membership, valid Paint samples, and effective
        // bright-ridge membership.
        // The ridge covers a majority of the face but
        // omits the curved right tip where the old fit invented a shadow.
        let input = image::load_from_memory(include_bytes!("../data/car-highlight-source.png"))
            .unwrap()
            .to_rgb8();
        let mask = image::load_from_memory(include_bytes!("../data/car-highlight-mask.png"))
            .unwrap()
            .to_rgb8();
        let w = input.width() as usize;
        let h = input.height() as usize;
        let source = Raster::new(
            w,
            h,
            input
                .pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let mut segmentation = two_face_segmentation(&source);
        segmentation.labels = mask.pixels().map(|p| u32::from(p[0] > 0)).collect();
        segmentation.paint_samples = mask.pixels().map(|p| p[1] > 0).collect();
        let highlight = [0.96415824, 0.5070586, 0.467012];
        segmentation.canonical = Raster::new(
            w,
            h,
            segmentation
                .labels
                .iter()
                .map(|&l| if l == 1 { highlight } else { [0.3; 3] })
                .collect(),
        );
        for label in 0..2 {
            let indices = segmentation
                .labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| (l == label as u32).then_some(i))
                .collect::<Vec<_>>();
            let b = bounds(&indices, w);
            let region = &mut segmentation.regions[label];
            region.area = indices.len();
            region.min_x = b.min_x as usize;
            region.max_x = b.max_x as usize + 1;
            region.min_y = b.min_y as usize;
            region.max_y = b.max_y as usize + 1;
        }
        let branches = crate::ridge::StrongRidgeBranches {
            dark: vec![false; w * h],
            bright: mask.pixels().map(|p| p[2] > 0).collect(),
        };
        let (paints, _, _) = fit_all_without_topology(
            &[Some(Paint::Solid { color: [0.3; 3] }), None],
            &source,
            &source,
            &segmentation,
            &branches,
            &Config::default(),
        );
        let tip = segmentation
            .labels
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| {
                (l == 1 && i % w >= 78 && segmentation.paint_samples[i]).then_some(i)
            })
            .collect::<Vec<_>>();
        assert!(tip.len() > 100);
        let error = paint_stats_against_labs(&oklab_pixels(&source), &tip, w, &paints[1]);
        assert!(
            error.mean < 5.0,
            "highlight tip lost its source colour: {error:?}, {:?}",
            paints[1]
        );
    }

    #[test]
    fn smooth_highlight_samples_survive_a_quantized_ridge_mask() {
        let source = Raster::new(
            128,
            64,
            (0..128 * 64)
                .map(|i| {
                    let r =
                        (((i % 128) as f32 - 90.0) / 12.0).hypot(((i / 128) as f32 - 32.0) / 12.0);
                    [0.5 + 0.3 * (-r * r).exp(); 3]
                })
                .collect(),
        );
        let mut segmentation = two_face_segmentation(&source);
        segmentation.canonical = Raster::blank(128, 64, [0.5; 3]);
        for (i, sample) in segmentation.paint_samples.iter_mut().enumerate() {
            *sample = source.pixels[i][0] < 0.505;
        }
        let branches = crate::ridge::StrongRidgeBranches {
            dark: vec![false; 128 * 64],
            bright: vec![false; 128 * 64],
        };
        let (paints, _, _) = fit_all_without_topology(
            &[None, None],
            &source,
            &source,
            &segmentation,
            &branches,
            &Config::default(),
        );
        assert!(
            paint_at(&paints[1], 32 * 128 + 90, 128)[0] > 0.72,
            "{:?}",
            paints[1]
        );
        let mut ink = source.clone();
        ink.pixels[32 * 128 + 90] = [1.0; 3];
        assert!(!smooth_native_paint_sample(&source, &ink, 32 * 128 + 90));
    }

    #[test]
    fn fitted_fields_reconcile_smooth_seams_but_keep_blurred_material_steps() {
        for material_step in [false, true] {
            let source = Raster::new(
                128,
                128,
                (0..128 * 128)
                    .map(|i| {
                        let x = (i % 128) as f32;
                        let step = if material_step {
                            0.1 * ((x - 58.0) / 12.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        [0.3 + 0.3 * x / 127.0 + step; 3]
                    })
                    .collect(),
            );
            let labels = (0..128 * 128)
                .map(|i| {
                    let x = i % 128;
                    let y = i / 128;
                    if !(8..120).contains(&x) || !(8..120).contains(&y) {
                        0
                    } else if x < 64 {
                        1
                    } else {
                        2
                    }
                })
                .collect();
            let mut segmentation = two_face_segmentation(&source);
            replace_source_supported_paint_labels(&source, &mut segmentation, labels, 0);
            let field = |bias: f64| Paint::Linear {
                preset: LinearPreset::Fitted,
                start: Point { x: 0.0, y: 0.0 },
                end: Point { x: 127.0, y: 0.0 },
                stops: vec![
                    ColorStop {
                        offset: 0.0,
                        color: [0.3 + bias; 3],
                    },
                    ColorStop {
                        offset: 1.0,
                        color: [0.6 + bias; 3],
                    },
                ],
            };
            let hints = vec![
                None,
                Some(field(-0.02)),
                Some(field(if material_step { 0.12 } else { 0.02 })),
            ];
            let seam = [Point { x: 63.5, y: 64.0 }];
            let before = seam_errors_at_points(
                hints[1].as_ref().unwrap(),
                hints[2].as_ref().unwrap(),
                &seam,
            )[0];
            let (paints, _, _) = fit_all_without_topology(
                &hints,
                &source,
                &source,
                &segmentation,
                &crate::ridge::StrongRidgeBranches {
                    dark: vec![false; 128 * 128],
                    bright: vec![false; 128 * 128],
                },
                &Config::default(),
            );
            let after = seam_errors_at_points(&paints[1], &paints[2], &seam)[0];
            if material_step {
                assert!((after - before).abs() < 1e-5);
            } else {
                assert!(after < 0.5 && after < before * 0.2, "{before} -> {after}");
            }
        }
    }

    #[test]
    fn interpolation_hits_endpoints() {
        let stops = vec![
            ColorStop {
                offset: 0.0,
                color: [0.0; 3],
            },
            ColorStop {
                offset: 1.0,
                color: [1.0; 3],
            },
        ];
        assert_eq!(interpolate(&stops, 0.0), [0.0; 3]);
        assert_eq!(interpolate(&stops, 1.0), [1.0; 3]);
    }

    #[test]
    fn primary_gate_separates_coherent_gradient_from_unstructured_texture() {
        let width = 16;
        let height = 16;
        let coherent = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|_| {
                    (0..width).map(|x| {
                        let value = x as f32 / (width - 1) as f32;
                        [value, 0.25 + 0.5 * value, 1.0 - value]
                    })
                })
                .collect(),
        );
        let texture = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let value = if (x + y).is_multiple_of(2) { 0.2 } else { 0.8 };
                        [value, 1.0 - value, value]
                    })
                })
                .collect(),
        );
        let indices = (0..width * height).collect::<Vec<_>>();
        let region_bounds = bounds(&indices, width);
        let coherent_score =
            primary_gradient_coherence(&coherent, &indices, region_bounds, false, 64);
        let texture_score =
            primary_gradient_coherence(&texture, &indices, region_bounds, false, 64);
        assert!(coherent_score > 0.95, "{coherent_score}");
        assert!(texture_score < 0.06, "{texture_score}");
    }

    #[test]
    fn office_fit_uses_the_source_angle_for_a_rotated_linear_ramp() {
        let width = 31;
        let height = 29;
        let source_direction = (0.26_f32, 0.97_f32);
        let maximum_projection =
            (width - 1) as f32 * source_direction.0 + (height - 1) as f32 * source_direction.1;
        let source = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let parameter = (x as f32 * source_direction.0
                            + y as f32 * source_direction.1)
                            / maximum_projection;
                        [
                            0.18 + 0.52 * parameter,
                            0.32 + 0.36 * parameter,
                            0.78 - 0.31 * parameter,
                        ]
                    })
                })
                .collect(),
        );
        let samples = (0..width * height).collect::<Vec<_>>();
        let source_labs = oklab_pixels(&source);
        let (paint, _) =
            office_gradient_candidate(&source, &source_labs, &samples, bounds(&samples, width), 5)
                .expect("a coherent ramp must produce a gradient candidate");
        let Paint::Linear {
            preset: LinearPreset::Fitted,
            start,
            end,
            ..
        } = paint
        else {
            panic!("a non-cardinal ramp must retain its continuously fitted angle");
        };
        let fitted = (end.x - start.x, end.y - start.y);
        let fitted_length = fitted.0.hypot(fitted.1);
        let source_length = source_direction.0.hypot(source_direction.1);
        let alignment = (fitted.0 * source_direction.0 + fitted.1 * source_direction.1).abs()
            / (fitted_length * source_length);
        assert!(alignment > 0.995, "angle alignment was {alignment}");
    }

    #[test]
    fn gradient_discontinuity_separates_a_ramp_from_a_step() {
        let lab = |lightness| Oklab {
            l: lightness,
            a: 12.0,
            b: -7.0,
        };
        assert!(lab_gradient_discontinuity(lab(20.0), lab(25.0), lab(30.0), lab(35.0)) < 1e-6);
        assert!(lab_gradient_discontinuity(lab(20.0), lab(20.0), lab(40.0), lab(40.0)) > 0.99);
    }

    #[test]
    fn smooth_boundary_detection_accepts_a_steep_continuous_ramp() {
        let pixels = (0..8)
            .flat_map(|_| {
                (0..12).map(|x| {
                    let value = 0.15 + 0.70 * x as f32 / 11.0;
                    [value; 3]
                })
            })
            .collect();
        let source = Raster::new(12, 8, pixels);
        let segmentation = two_face_segmentation(&source);
        let boundaries = smooth_paint_boundaries(&oklab_pixels(&source), &segmentation, 8, false);
        assert_eq!(boundaries.len(), 1);
        assert!(boundaries[0].median_delta_e > 3.0);
        assert!(boundary_has_continuous_gradient(&boundaries[0]));
    }

    #[test]
    fn blurred_material_step_is_not_a_continuous_shading_boundary() {
        for blurred_step in [false, true] {
            let source = Raster::new(
                128,
                64,
                (0..128 * 64)
                    .map(|i| {
                        let x = (i % 128) as f32;
                        let parameter = if blurred_step {
                            ((x - 58.0) / 12.0).clamp(0.0, 1.0)
                        } else {
                            x / 127.0
                        };
                        [0.4 + 0.10 * parameter; 3]
                    })
                    .collect(),
            );
            let segmentation = two_face_segmentation(&source);
            let mut boundaries =
                smooth_paint_boundaries(&oklab_pixels(&source), &segmentation, 8, true);
            assert_eq!(boundaries.len(), 1);
            measure_boundary_material_step(
                &oklab_pixels(&source),
                &segmentation,
                &mut boundaries[0],
            );
            assert_eq!(boundary_has_material_step(&boundaries[0]), blurred_step);
            assert_eq!(boundary_is_smooth(&boundaries[0]), !blurred_step);
        }
    }

    #[test]
    fn smooth_boundary_detection_rejects_a_material_step() {
        let mut pixels = vec![[0.2; 3]; 12 * 8];
        for y in 0..8 {
            for x in 6..12 {
                pixels[y * 12 + x] = [0.8; 3];
            }
        }
        let source = Raster::new(12, 8, pixels);
        let segmentation = two_face_segmentation(&source);
        assert!(
            smooth_paint_boundaries(&oklab_pixels(&source), &segmentation, 8, false).is_empty()
        );
    }

    fn coupling_boundary(length: usize, median_delta_e: f32) -> CouplingBoundary {
        CouplingBoundary {
            boundary: SmoothPaintBoundary {
                left: 0,
                right: 1,
                points: Vec::new(),
                length,
                median_delta_e,
                percentile_delta_e: median_delta_e + 1.0,
                gradient_sample_fraction: 1.0,
                median_gradient_discontinuity: 0.2,
                percentile_gradient_discontinuity: 0.3,
                material_step_fraction: 0.0,
            },
            seam_p90: 4.0,
            same_paint_key: false,
        }
    }

    #[test]
    fn continuous_ramp_priority_accounts_for_visible_boundary_length() {
        let short = coupling_boundary(16, 6.0);
        let long = coupling_boundary(256, 6.0);
        assert!(coupling_boundary_priority(&long) > coupling_boundary_priority(&short));

        let flat_short = coupling_boundary(16, 0.5);
        let flat_long = coupling_boundary(256, 0.5);
        assert_eq!(
            coupling_boundary_priority(&flat_short),
            coupling_boundary_priority(&flat_long)
        );
    }

    #[test]
    fn native_continuity_gate_keeps_interior_colour_regression_sub_jnd() {
        let config = Config::default();
        let native = coupling_regression_limits(true, false, &config);
        assert_eq!(native, (0.05, 0.20));

        let patch = coupling_regression_limits(true, true, &config);
        assert!(patch.0 > native.0);
        assert!(patch.1 > native.1);
    }

    #[test]
    fn source_supported_merge_removes_only_an_unsupported_paint_interface() {
        let source = Raster::blank(8, 8, [0.5; 3]);
        let mut segmentation = two_face_segmentation(&source);
        let mut paints = vec![
            Paint::Solid { color: [0.498; 3] },
            Paint::Solid { color: [0.502; 3] },
        ];
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config::default(),
        );
        assert_eq!(report.merges, 1);
        assert_eq!(report.boundary_edges_removed, 8);
        assert_eq!(segmentation.regions.len(), 1);
        assert_eq!(paints, vec![Paint::Solid { color: [0.5; 3] }]);
    }

    #[test]
    fn merge_template_preserves_fitted_direction_and_radial_focus() {
        let samples = (0..64 * 24).collect::<Vec<_>>();
        for radial in [false, true] {
            let center = Point { x: 10.0, y: 5.0 };
            let radius = Point { x: 20.0, y: 10.0 };
            let source = Raster::new(
                64,
                24,
                samples
                    .iter()
                    .map(|&i| {
                        let t = if radial {
                            rotated_radial_parameter(i, 64, center, radius, 0.4) * 0.2
                        } else {
                            ((i % 64) as f32 + 2.0 * (i / 64) as f32) / 160.0
                        };
                        [0.1 + t; 3]
                    })
                    .collect(),
            );
            let solid = Paint::Solid {
                color: mean_color(&source, &samples),
            };
            let template = if radial {
                Paint::Radial {
                    rotation: 0.4,
                    origin: RadialOrigin::Fitted,
                    center,
                    radius,
                    stops: Vec::new(),
                }
            } else {
                Paint::Linear {
                    preset: LinearPreset::Fitted,
                    start: Point { x: 0.0, y: 0.0 },
                    end: Point { x: 10.0, y: 20.0 },
                    stops: Vec::new(),
                }
            };
            let (paint, stats) = fit_like_merge_paint(
                &source,
                &samples,
                bounds(&samples, 64),
                &template,
                &solid,
                paint_stats(&source, &samples, &solid),
            );
            assert!(stats.mean < 0.01, "radial={radial}, mean={}", stats.mean);
            match paint {
                Paint::Linear { start, end, .. } => {
                    assert!(((end.y - start.y) / (end.x - start.x) - 2.0).abs() < 1e-5)
                }
                Paint::Radial {
                    center: actual,
                    radius: actual_radius,
                    rotation,
                    ..
                } => {
                    assert_eq!(actual, center);
                    assert_eq!(rotation, 0.4);
                    assert!((actual_radius.x / actual_radius.y - 2.0).abs() < 1e-5);
                }
                _ => panic!("gradient model lost"),
            }
        }
    }

    #[test]
    fn source_supported_merge_keeps_a_faint_one_pixel_line() {
        let source = Raster::new(
            64,
            16,
            (0..64 * 16)
                .map(|i| if i % 64 == 31 { [0.78; 3] } else { [0.8; 3] })
                .collect(),
        );
        let mut segmentation = two_face_segmentation(&source);
        let labels = (0..64 * 16)
            .map(|i| {
                if i % 64 < 31 {
                    0
                } else if i % 64 == 31 {
                    1
                } else {
                    2
                }
            })
            .collect();
        replace_source_supported_paint_labels(&source, &mut segmentation, labels, 0);
        let mut paints = vec![
            Paint::Solid { color: [0.8; 3] },
            Paint::Solid { color: [0.78; 3] },
            Paint::Solid { color: [0.8; 3] },
        ];
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config {
                paint_merge_passes: 8,
                ..Config::default()
            },
        );
        assert_eq!(report.merges, 0);
        assert_ne!(segmentation.labels[30], segmentation.labels[31]);
        assert_ne!(segmentation.labels[31], segmentation.labels[32]);
    }

    #[test]
    fn source_supported_merge_joins_a_chain_of_ramp_fragments() {
        let source = Raster::new(
            64,
            12,
            (0..64 * 12)
                .map(|i| [0.2 + 0.6 * (i % 64) as f32 / 63.0; 3])
                .collect(),
        );
        let mut segmentation = two_face_segmentation(&source);
        let labels = (0..64 * 12).map(|i| ((i % 64) / 16) as u32).collect();
        replace_source_supported_paint_labels(&source, &mut segmentation, labels, 0);
        let mut paints = (0..4)
            .map(|i| {
                let low = i * 16;
                let high = low + 15;
                Paint::Linear {
                    preset: LinearPreset::LeftToRight,
                    start: Point {
                        x: low as f32,
                        y: 0.0,
                    },
                    end: Point {
                        x: high as f32,
                        y: 0.0,
                    },
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: source.pixels[low].map(f64::from),
                        },
                        ColorStop {
                            offset: 1.0,
                            color: source.pixels[high].map(f64::from),
                        },
                    ],
                }
            })
            .collect::<Vec<_>>();
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config {
                paint_merge_passes: 8,
                ..Config::default()
            },
        );
        assert_eq!(report.merges, 3);
        assert_eq!(segmentation.regions.len(), 1);
        assert!(
            paint_stats(
                &source,
                &(0..source.pixels.len()).collect::<Vec<_>>(),
                &paints[0]
            )
            .mean
                < 0.01
        );
    }

    #[test]
    fn source_supported_merge_preserves_a_native_material_transition() {
        let mut pixels = vec![[0.2; 3]; 8 * 8];
        for y in 0..8 {
            for x in 4..8 {
                pixels[y * 8 + x] = [0.8; 3];
            }
        }
        let source = Raster::new(8, 8, pixels);
        let mut segmentation = two_face_segmentation(&source);
        let mut paints = vec![
            Paint::Solid { color: [0.2; 3] },
            Paint::Solid { color: [0.8; 3] },
        ];
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config::default(),
        );
        assert_eq!(report, SupportedPaintMergeReport::default());
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(paints.len(), 2);
    }

    #[test]
    fn removing_a_false_spot_preserves_the_supported_shading_and_owners() {
        let base = Paint::Linear {
            preset: LinearPreset::Fitted,
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 0.0, y: 31.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.7, 0.2, 0.2],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.95, 0.6, 0.5],
                },
            ],
        };
        let supported = PaintOverlay {
            paint: Box::new(Paint::Linear {
                preset: LinearPreset::Fitted,
                start: Point { x: 0.0, y: 0.0 },
                end: Point { x: 63.0, y: 0.0 },
                stops: vec![
                    ColorStop {
                        offset: 0.0,
                        color: [0.9, 0.8, 0.7],
                    },
                    ColorStop {
                        offset: 1.0,
                        color: [0.9, 0.8, 0.7],
                    },
                ],
            }),
            opacity_stops: vec![
                OpacityStop {
                    offset: 0.0,
                    opacity: 0.5,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let reference = Paint::Layered {
            base: Box::new(base.clone()),
            overlays: vec![supported.clone()],
        };
        let source = Raster::new(
            64,
            32,
            (0..64 * 32).map(|i| paint_at(&reference, i, 64)).collect(),
        );
        let segmentation = two_face_segmentation(&source);
        let labels = segmentation.labels.clone();
        let false_spot = PaintOverlay {
            paint: Box::new(Paint::Radial {
                origin: RadialOrigin::Fitted,
                center: Point { x: 16.0, y: 16.0 },
                radius: Point { x: 8.0, y: 8.0 },
                rotation: 0.0,
                stops: vec![
                    ColorStop {
                        offset: 0.0,
                        color: [0.1; 3],
                    },
                    ColorStop {
                        offset: 1.0,
                        color: [0.1; 3],
                    },
                ],
            }),
            opacity_stops: vec![
                OpacityStop {
                    offset: 0.0,
                    opacity: 0.6,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let mut paints = vec![
            Paint::Layered {
                base: Box::new(base),
                overlays: vec![supported, false_spot],
            },
            reference,
        ];
        assert_eq!(
            refine_residual_shapes(&source, &segmentation, &mut paints),
            1
        );
        assert_eq!(segmentation.labels, labels);
        assert_eq!(paints.len(), 2);
        for (i, &label) in labels.iter().enumerate() {
            let error = delta_e_ok(
                rgb_to_oklab(source.pixels[i]),
                rgb_to_oklab(paint_at(&paints[label as usize], i, 64)),
            );
            assert!(error < 0.5, "supported shading was lost at {i}: {error}");
        }
    }

    #[test]
    fn redundant_layered_field_becomes_one_paint() {
        let source = Raster::new(32, 16, vec![[0.4; 3]; 32 * 16]);
        let segmentation = two_face_segmentation(&source);
        let layered = Paint::Layered {
            base: Box::new(Paint::Solid { color: [0.4; 3] }),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Solid { color: [0.4; 3] }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 0.8,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        };
        let mut paints = vec![layered.clone(), layered];
        assert_eq!(
            simplify_layered_paints(&source, &segmentation, &mut paints),
            2
        );
        assert!(paints.iter().all(|p| !matches!(p, Paint::Layered { .. })));
        for (i, &owner) in segmentation.labels.iter().enumerate() {
            assert!(
                delta_e_ok(
                    rgb_to_oklab(source.pixels[i]),
                    rgb_to_oklab(paint_at(&paints[owner as usize], i, 32))
                ) < 0.01
            );
        }
    }

    #[test]
    fn layered_simplification_keeps_separate_local_highlights() {
        let layered = Paint::Layered {
            base: Box::new(Paint::Solid { color: [0.1; 3] }),
            overlays: [Point { x: 5.0, y: 5.0 }, Point { x: 11.0, y: 24.0 }]
                .into_iter()
                .map(|center| PaintOverlay {
                    paint: Box::new(Paint::Radial {
                        rotation: 0.0,
                        origin: RadialOrigin::Fitted,
                        center,
                        radius: Point { x: 4.0, y: 4.0 },
                        stops: vec![
                            ColorStop {
                                offset: 0.0,
                                color: [0.9; 3],
                            },
                            ColorStop {
                                offset: 1.0,
                                color: [0.9; 3],
                            },
                        ],
                    }),
                    opacity_stops: vec![
                        OpacityStop {
                            offset: 0.0,
                            opacity: 1.0,
                        },
                        OpacityStop {
                            offset: 1.0,
                            opacity: 0.0,
                        },
                    ],
                })
                .collect(),
        };
        let source = Raster::new(
            32,
            32,
            (0..1024).map(|i| paint_at(&layered, i, 32)).collect(),
        );
        let segmentation = two_face_segmentation(&source);
        let mut paints = vec![layered, Paint::Solid { color: [0.1; 3] }];
        assert_eq!(
            simplify_layered_paints(&source, &segmentation, &mut paints),
            0
        );
        assert!(matches!(paints[0], Paint::Layered { .. }));
    }

    #[test]
    fn layered_paint_fades_without_an_internal_boundary() {
        let paint = Paint::Layered {
            base: Box::new(Paint::Solid { color: [0.0; 3] }),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Radial {
                    rotation: 0.0,
                    origin: RadialOrigin::Fitted,
                    center: Point { x: 1.0, y: 0.0 },
                    radius: Point { x: 1.0, y: 1.0 },
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
                        opacity: 1.0,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        };
        assert_eq!(paint_at(&paint, 1, 3), [1.0, 0.0, 0.0]);
        assert_eq!(paint_at(&paint, 0, 3), [0.0; 3]);
    }

    #[test]
    fn small_region_gradient_requires_strict_measured_gain() {
        let selected = ErrorStats {
            mean: 10.0,
            percentile: 6.0,
        };
        let minimum_improvement = 0.25 * 2.3;
        assert!(gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 7.5,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 8.0,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 7.5,
                percentile: 6.1,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 3.0,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            16,
            64,
        ));
    }

    #[test]
    fn normal_region_uses_one_gain_gate_for_every_gradient_model() {
        let solid = ErrorStats {
            mean: 2.4,
            percentile: 3.8,
        };
        let gradient = ErrorStats {
            mean: 1.9,
            percentile: 3.3,
        };
        assert!(gradient_gain_is_sufficient(
            solid,
            gradient,
            false,
            0.25 * 2.3,
            14_000,
            64,
        ));
    }
}
