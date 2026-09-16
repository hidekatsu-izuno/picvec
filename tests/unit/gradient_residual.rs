mod tests {
    use super::*;

    #[test]
    fn bounded_mse_preserves_pairwise_rounding_and_strict_thresholds() {
        let source = Raster::blank(2048, 1, [0.3, 0.6, 0.9]);
        let predict = |i: usize| {
            [
                ((i * 17) % 257) as f32 / 256.0,
                0.5,
                ((i * 7) % 31) as f32 / 30.0,
            ]
        };
        for count in [0, 1, 2, 3, 7, 42, 43, 85, 128, 129, 513, 2048] {
            let samples: Vec<_> = (0..count).collect();
            let expected = paint_rgb_mse_with(&source, &samples, predict);
            for limit in [
                0.0,
                f32::from_bits(expected.to_bits().saturating_sub(1)),
                expected,
                f32::from_bits(expected.to_bits() + 1),
                0.05,
                f32::INFINITY,
            ] {
                let actual = mse_below(&source, &samples, predict, limit);
                assert_eq!(
                    actual.map(f32::to_bits),
                    (expected < limit).then_some(expected.to_bits()),
                    "count={count}, limit={limit}"
                );
            }
        }
    }

    #[test]
    fn bounded_mse_stops_before_predicting_a_proven_losing_suffix() {
        let source = Raster::blank(1024, 1, [0.0; 3]);
        let samples: Vec<_> = (0..1024).collect();
        let mut calls = 0;
        let result = mse_below(
            &source,
            &samples,
            |_| {
                calls += 1;
                [1.0; 3]
            },
            0.001,
        );
        assert!(result.is_none());
        assert!(calls < 128, "evaluated {calls} samples");
    }

    #[test]
    fn shared_overlay_parameter_keeps_colour_and_opacity_exact() {
        let stops = vec![
            ColorStop {
                offset: 0.0,
                color: [0.13, 0.92, 0.23],
            },
            ColorStop {
                offset: 0.4,
                color: [0.7, 0.02, 0.58],
            },
            ColorStop {
                offset: 1.0,
                color: [0.21, 0.18, 0.92],
            },
        ];
        for paint in [
            Paint::Linear {
                preset: LinearPreset::Fitted,
                start: Point { x: 4.0, y: 8.0 },
                end: Point { x: 18.0, y: 5.0 },
                stops: stops.clone(),
            },
            Paint::Radial {
                origin: RadialOrigin::Fitted,
                center: Point { x: 7.0, y: 9.0 },
                radius: Point { x: 12.0, y: 3.0 },
                rotation: 0.7,
                stops,
            },
            Paint::Solid {
                color: [0.2, 0.4, 0.8],
            },
        ] {
            let overlay = PaintOverlay {
                paint: Box::new(paint),
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
            };
            for i in 0..32 * 24 {
                let parameter = match overlay.paint.as_ref() {
                    Paint::Linear { start, end, .. } => linear_parameter(i, 32, *start, *end),
                    Paint::Radial {
                        center,
                        radius,
                        rotation,
                        ..
                    } => rotated_radial_parameter(i, 32, *center, *radius, *rotation),
                    _ => 0.0,
                };
                let expected = (
                    paint_at(&overlay.paint, i, 32),
                    interpolate_opacity(&overlay.opacity_stops, parameter),
                );
                assert_eq!(sample_overlay(&overlay, i, 32), expected);
            }
        }
    }

    #[test]
    fn lazy_slope_observations_match_uncached_checks() {
        for width in [1, 2, 6, 17] {
            let source = Raster::new(
                width,
                13,
                (0..width * 13)
                    .map(|i| [((i * 17) % 127) as f32 / 127.0; 3])
                    .collect(),
            );
            let pixels = (0..width * 13).filter(|i| i % 7 != 0).collect::<Vec<_>>();
            let observed = pixels.iter().copied().collect::<HashSet<_>>();
            let base = Paint::Solid { color: [0.4; 3] };
            let cache = SlopeSupport::new(&source, &base, &pixels);
            let difference = |a: Oklab, b: Oklab| Oklab {
                l: b.l - a.l,
                a: b.a - a.a,
                b: b.b - a.b,
            };
            for color in [0.1, 0.8, 0.4, 0.1] {
                let candidate = Paint::Solid { color: [color; 3] };
                let expected = pixels.iter().all(|&i| {
                    [i + 6, i + 6 * width].into_iter().all(|j| {
                        if (j == i + 6 && i % width + 6 >= width) || !observed.contains(&j) {
                            return true;
                        }
                        residual_slope_supported(
                            difference(
                                rgb_to_oklab(source.pixels[i]),
                                rgb_to_oklab(source.pixels[j]),
                            ),
                            difference(
                                rgb_to_oklab(paint_at(&base, i, width)),
                                rgb_to_oklab(paint_at(&base, j, width)),
                            ),
                            difference(
                                rgb_to_oklab(paint_at(&candidate, i, width)),
                                rgb_to_oklab(paint_at(&candidate, j, width)),
                            ),
                        )
                    })
                });
                assert_eq!(cache.accepts(&candidate), expected);
                let expected = pixels.iter().all(|&i| {
                    let target = rgb_to_oklab(source.pixels[i]);
                    delta_e_ok(target, rgb_to_oklab(paint_at(&candidate, i, width)))
                        <= delta_e_ok(target, rgb_to_oklab(paint_at(&base, i, width))) + 1.0
                });
                assert_eq!(cache.preserves_error(&candidate, 1.0), expected);
            }
        }
    }

    #[test]
    fn cached_subsets_match_complete_paint_evaluation_bit_for_bit() {
        let source = Raster::new(
            17,
            13,
            (0..221)
                .map(|i| {
                    [
                        ((i * 13) % 251) as f32 / 255.0,
                        ((i * 37) % 241) as f32 / 255.0,
                        0.7,
                    ]
                })
                .collect(),
        );
        let samples = (0..221).step_by(3).collect::<Vec<_>>();
        let base = Paint::Solid {
            color: [0.2, 0.3, 0.4],
        };
        let overlays = (0..5)
            .map(|k| PaintOverlay {
                paint: Box::new(if k % 2 == 0 {
                    Paint::Linear {
                        preset: LinearPreset::Fitted,
                        start: Point { x: -2.0, y: 1.0 },
                        end: Point { x: 17.0, y: 11.0 },
                        stops: vec![
                            ColorStop {
                                offset: 0.0,
                                color: [0.1, 0.9, 0.7],
                            },
                            ColorStop {
                                offset: 1.0,
                                color: [0.9, 0.3, 0.5],
                            },
                        ],
                    }
                } else {
                    Paint::Radial {
                        origin: RadialOrigin::Fitted,
                        rotation: 0.3,
                        center: Point { x: 8.0, y: 5.0 },
                        radius: Point { x: 9.0, y: 7.0 },
                        stops: vec![
                            ColorStop {
                                offset: 0.0,
                                color: [0.9, 0.1, 0.2],
                            },
                            ColorStop {
                                offset: 1.0,
                                color: [0.1, 0.7, 0.8],
                            },
                        ],
                    }
                }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 0.1 * k as f64,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.9,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let mut cached = SubsetSamples::new(&source, &samples, &base, &overlays);
        for mask in 0..1 << overlays.len() {
            let paint = Paint::Layered {
                base: Box::new(base.clone()),
                overlays: overlays
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, layer)| layer.clone())
                    .collect(),
            };
            assert_eq!(
                cached.mse(mask).to_bits(),
                paint_rgb_mse(&source, &samples, &paint).to_bits()
            );
        }
    }
}
