mod tests {
    use super::*;

    #[test]
    fn perceptual_sampling_detects_every_phase_of_periodic_thin_lines() {
        let (width, height) = (1024, 1024);
        let whole = SourceRect {
            x: 0,
            y: 0,
            width,
            height,
        };
        let blank = Raster::blank(width, height, [1.0; 3]);
        // The previous 6-pixel grid missed phase 3 completely, including edges.
        for vertical in [false, true] {
            for phase in 0..6 {
                let source = Raster::new(
                    width,
                    height,
                    (0..width * height)
                        .map(|i| {
                            let coordinate = if vertical { i % width } else { i / width };
                            if coordinate % 6 == phase {
                                [0.0; 3]
                            } else {
                                [1.0; 3]
                            }
                        })
                        .collect(),
                );
                let score = perceptual_score(&source, whole, &blank, whole);
                assert!(
                    score.mean_delta_e > 10.0,
                    "phase {phase}, vertical {vertical}: {score:?}"
                );
                assert_eq!(score.missing_edge_fraction, 1.0);
                let exact = perceptual_score(&source, whole, &source, whole);
                assert_eq!(exact.combined, 0.0);
            }
        }
    }

    #[test]
    fn edge_evidence_finds_an_isolated_dot_outside_colour_samples() {
        let (width, height) = (512, 512);
        let whole = SourceRect {
            x: 0,
            y: 0,
            width,
            height,
        };
        let blank = Raster::blank(width, height, [1.0; 3]);
        let mut source = blank.clone();
        // Pick a location explicitly outside the dispersed colour sample.
        let start = (200 * width + 200) / 8 * 8;
        let selected = (dispersed_sample(start / 8) % 8) as usize;
        source.pixels[start + (selected + 3) % 8] = [0.0; 3];
        let score = perceptual_score(&source, whole, &blank, whole);
        assert_eq!(score.mean_delta_e, 0.0);
        assert_eq!(score.missing_edge_fraction, 1.0);
        assert!(score.combined >= 3.0);
        assert_eq!(
            perceptual_score(&source, whole, &blank, whole).combined,
            score.combined
        );
    }

    #[test]
    fn perceptual_sampling_handles_offset_and_one_pixel_wide_regions() {
        let mut source = Raster::blank(9, 19, [1.0; 3]);
        source.pixels[10 * 9 + 4] = [0.0; 3];
        for region in [
            SourceRect {
                x: 4,
                y: 5,
                width: 1,
                height: 10,
            },
            SourceRect {
                x: 3,
                y: 10,
                width: 3,
                height: 1,
            },
        ] {
            let crop = source.crop(region.x, region.y, region.width, region.height);
            assert_eq!(
                perceptual_score(&source, region, &crop, region).combined,
                0.0
            );
            let blank = Raster::blank(region.width, region.height, [1.0; 3]);
            assert!(perceptual_score(&source, region, &blank, region).combined > 0.0);
        }
    }

    #[test]
    fn refinement_namespaces_reused_soft_geometry() {
        let child = "<svg><defs/><g id=\"soft-source\"/><use href=\"#soft-source\" filter=\"url(#blur)\"/></svg>";
        let mut nested = Document::from(child);
        nested.root_mut().namespace_ids("child-");
        assert!(nested.contains("href=\"#child-soft-source\""));
        assert!(nested.contains("id=\"child-soft-source\""));
        assert!(nested.contains("url(#child-blur)"));
    }

    #[test]
    fn thin_spanning_separator_does_not_hide_a_touching_figure() {
        let (width, height) = (320, 240);
        let mut support = vec![false; width * height];
        for y in 80..82 {
            for x in 0..width {
                support[y * width + x] = true;
            }
        }
        for y in 74..160 {
            for x in 110..180 {
                support[y * width + x] = true;
            }
        }
        let regions = object_regions(&support, width, height, 140);
        assert_eq!(regions.len(), 1);
        let r = regions[0];
        assert!(r.x < 110 && r.y < 74 && r.x + r.width > 180 && r.y + r.height > 160);
        assert!(support[80 * width], "planning must not erase source ink");
        for y in 82..90 {
            for x in 0..width {
                support[y * width + x] = true;
            }
        }
        assert!(
            object_regions(&support, width, height, 140).is_empty(),
            "a thick connected object must not be split as a separator"
        );
    }

    #[test]
    fn measured_refinement_rate_replaces_the_partition_estimate() {
        let mut candidate = RefinementCandidate {
            core: SourceRect {
                x: 0,
                y: 0,
                width: 550,
                height: 650,
            },
            baseline: PerceptualScore::default(),
            model_cost: 1.0,
            priority: 1.0,
        };
        let rate = candidate.measured_rate(5.8, candidate.core.area(), 1_080_000);
        assert!(
            rate > 1.0,
            "useful native shading must fit the byte-rate budget"
        );
        candidate.model_cost = 4.3;
        assert_eq!(
            candidate.measured_rate(5.8, candidate.core.area(), 1_080_000),
            rate
        );
        assert!(
            rate / candidate.model_cost.sqrt() < 1.0,
            "the former double charge rejected the same measured result"
        );
        assert!(
            candidate.measured_rate(5.8, candidate.core.area(), 3_000_000) < 1.0,
            "expensive detail must still pay for its actual bytes"
        );
        assert_eq!(
            candidate.measured_rate(5.8, 2 * candidate.core.area(), 2_160_000),
            rate
        );
    }

    #[test]
    #[ignore = "full-size sample regression"]
    fn clipart_sheet_refines_whole_figures_with_and_without_keying() {
        let path = std::path::Path::new("sample/input/cliparts-6x6.png");
        let (source, _) = crate::raster::SourceRaster::load_with_alpha(
            path,
            32768,
            32_000_000,
            512 * 1024 * 1024,
        )
        .unwrap();
        let key = crate::chroma::detect(&source).unwrap();
        let matte = crate::chroma::pull_matte(&source, key);
        let separated = crate::chroma::separate_compact_foreground(&source, &matte, key.sampled);
        let (bands, cleaned) = crate::separators::extract(&separated, &matte, 1400);
        assert!(
            bands
                .iter()
                .any(|band| band.rect[1] > 3330.0 && band.rect[1] < 3350.0),
            "the separator touching the lock must have one source model"
        );
        let cleaned = cleaned.unwrap();
        for alpha in [Some(&cleaned), None] {
            let support = foreground_support(&source, alpha);
            let regions = object_regions(&support, source.width, source.height, 1400);
            assert!(
                regions.len() >= 24,
                "keyed={}: too few intact figures",
                alpha.is_some()
            );
            if alpha.is_some() {
                // Opaque support also contains faint, connected grid debris;
                // keying separates these three figures from that network.
                for (x, y) in [
                    (2880, 2900),
                    (3700, 2100),
                    (430, 2100),
                    (1250, 3700),
                    (2060, 3670),
                ] {
                    assert!(
                        regions.iter().any(|r| r.x <= x
                            && x < r.x + r.width
                            && r.y <= y
                            && y < r.y + r.height),
                        "missing keyed figure at {x}, {y}"
                    );
                }
                assert!(
                    regions.iter().any(|r| r.x < 1100
                        && r.x + r.width > 1450
                        && r.y < 3337
                        && r.y + r.height > 3970),
                    "the lock must retain its cap above the separator"
                );
            }
            // All three cylinders and their common shading must share one fit.
            assert_eq!(
                regions
                    .iter()
                    .filter(|r| r.x <= 1000
                        && r.x + r.width >= 1510
                        && r.y <= 2590
                        && r.y + r.height >= 3240)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn connected_figures_are_not_cut_with_opaque_or_transparent_backgrounds() {
        for transparent in [false, true] {
            for background in [[1.0; 3], [0.0, 1.0, 0.0]] {
                let mut source = Raster::blank(400, 240, background);
                let mut alpha = vec![0.0; 400 * 240];
                for y in 90..150 {
                    for x in 80..320 {
                        source.pixels[y * 400 + x] = [x as f32 / 400.0, 0.1, 0.2];
                        alpha[y * 400 + x] = 1.0;
                    }
                }
                let matte = AlphaMatte::new(400, 240, alpha);
                let support = foreground_support(&source, transparent.then_some(&matte));
                let regions = object_regions(&support, 400, 240, 300);
                assert_eq!(regions.len(), 1);
                assert!(regions[0].x < 80 && regions[0].x + regions[0].width > 320);
                // The old grid would bisect this shape at x=200. A tighter
                // size budget must retain the whole base object, not tile it.
                assert!(object_regions(&support, 400, 240, 160).is_empty());
            }
        }
    }

    #[test]
    fn unknown_background_keeps_one_global_model() {
        let mut source = Raster::blank(400, 240, [0.0; 3]);
        for (i, pixel) in source.pixels.iter_mut().enumerate() {
            *pixel = [(i % 400) as f32 / 400.0, (i / 400) as f32 / 240.0, 0.5];
        }
        let support = foreground_support(&source, None);
        assert!(support.iter().all(|v| *v));
        assert!(object_regions(&support, 400, 240, 300).is_empty());
        assert_eq!(object_regions(&support, 400, 240, 400).len(), 1);
        let coarse = Raster::blank(100, 60, [1.0; 3]);
        assert!(
            plan_candidates(&source, None, &coarse, &vec![0; 100 * 60], 300, 64, 0.75).is_empty()
        );
    }

    #[test]
    fn nested_details_share_one_replacement_region() {
        let mut support = vec![false; 240 * 240];
        for y in 50..190 {
            for x in 50..190 {
                support[y * 240 + x] = x == 50
                    || x == 189
                    || y == 50
                    || y == 189
                    || ((100..140).contains(&x) && (100..140).contains(&y));
            }
        }
        assert_eq!(object_regions(&support, 240, 240, 220).len(), 1);
    }

    #[test]
    fn nearby_oversized_separator_leaves_a_whole_figure_candidate() {
        let mut support = vec![false; 400 * 240];
        // This grid line exceeds the crop limit and stays in the base.
        for x in 0..400 {
            support[40 * 400 + x] = true;
        }
        // A separate figure has a narrow but valid background gap above it.
        for y in 54..180 {
            for x in 100..220 {
                support[y * 400 + x] = true;
            }
        }
        let regions = object_regions(&support, 400, 240, 200);
        assert_eq!(regions.len(), 1);
        let r = regions[0];
        assert!(r.x < 100 && r.x + r.width > 220);
        assert!(r.y > 42 && r.y < 54 && r.y + r.height > 180);
        // A genuinely connected figure must still stay with the large grid.
        for y in 40..54 {
            support[y * 400 + 160] = true;
        }
        assert!(object_regions(&support, 400, 240, 200).is_empty());
    }

    #[test]
    fn oversized_connected_content_uses_the_same_measured_candidate_model() {
        for ink in [[0.0; 3], [0.7, 0.1, 0.2]] {
            let mut source = Raster::blank(320, 160, [1.0; 3]);
            for y in 40..120 {
                for x in 20..300 {
                    source.pixels[y * 320 + x] = ink;
                }
            }
            let coarse = Raster::blank(80, 40, [1.0; 3]);
            let labels = vec![0; 80 * 40];
            let candidates = plan_candidates(&source, None, &coarse, &labels, 140, 64, 0.75);
            assert_eq!(candidates.len(), 1);
            assert_eq!(
                candidates[0].core,
                SourceRect {
                    x: 0,
                    y: 0,
                    width: 320,
                    height: 160
                }
            );
            assert!(candidates[0].baseline.combined > 0.75);
            assert!(plan_candidates(&source, None, &source, &labels, 140, 64, 0.75).is_empty());
            assert!(plan_candidates(&source, None, &coarse, &labels, 140, 0, 0.75).is_empty());
        }
    }

    #[test]
    fn dense_visible_detail_reaches_measured_refinement_evaluation() {
        let mut source = Raster::blank(160, 160, [1.0; 3]);
        for y in 32..128 {
            for x in 32..128 {
                source.pixels[y * 160 + x] = if (x / 4 + y / 4) % 2 == 0 {
                    [0.2; 3]
                } else {
                    [0.4; 3]
                };
            }
        }
        let base = Raster::blank(40, 40, [0.3; 3]);
        let labels: Vec<_> = (0..1600).map(|i| i as u32).collect();
        let config = crate::Config::default();
        let candidates = plan_candidates(&source, None, &base, &labels, 160, 64, 0.75);
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert!(candidate.priority >= config.adaptive_min_predicted_rate);
        assert!(
            candidate.baseline.combined / candidate.model_cost < config.adaptive_min_predicted_rate
        );
    }

    #[test]
    fn rejected_border_can_move_only_through_fully_transparent_source() {
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 12,
            height: 12,
        };
        let core = SourceRect {
            x: 1,
            y: 1,
            width: 10,
            height: 10,
        };
        let base = Raster::new(12, 12, vec![[0.0, 1.0, 0.0]; 144]);
        let mut child = base.clone();
        child.pixels[12 + 5] = [1.0, 1.0, 1.0];
        let mut alpha = vec![0; 144];
        alpha[6 * 12 + 6] = 255;
        let matte = AlphaMatte::from_u8(12, 12, alpha.clone());
        assert_eq!(
            matching_refinement_core(&base, &child, whole, whole, core, whole, Some(&matte)),
            Some(SourceRect {
                x: 2,
                y: 2,
                width: 8,
                height: 8
            })
        );
        assert!(matching_refinement_core(&base, &child, whole, whole, core, whole, None).is_none());
        for coverage in [1, 255] {
            alpha[12 + 5] = coverage;
            let matte = AlphaMatte::from_u8(12, 12, alpha.clone());
            assert_eq!(
                matching_refinement_core(&base, &child, whole, whole, core, whole, Some(&matte)),
                Some(whole)
            );
        }
        alpha[5] = 255;
        let matte = AlphaMatte::from_u8(12, 12, alpha);
        assert!(
            matching_refinement_core(&base, &child, whole, whole, core, whole, Some(&matte))
                .is_none()
        );
    }

    #[test]
    fn fitted_border_discontinuities_are_rejected() {
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let core = SourceRect {
            x: 20,
            y: 20,
            width: 60,
            height: 60,
        };
        let base = Raster::blank(100, 100, [1.0; 3]);
        let mut child = Raster::blank(60, 60, [1.0; 3]);
        child.pixels[30 * 60 + 30] = [0.0; 3];
        assert!(refinement_boundary_matches(
            &base, &child, whole, whole, core, core
        ));
        child.pixels[30 * 60] = [0.9; 3];
        assert!(!refinement_boundary_matches(
            &base, &child, whole, whole, core, core
        ));
    }

    #[test]
    fn full_replacement_discards_base_for_opaque_and_transparent_sources() {
        let base = "<svg width=\"10\" height=\"10\"><defs/><path id=\"obsolete\"/></svg>";
        let mut patches: Vec<_> = (0..2)
            .map(|i| EmbeddedRefinement {
                core: SourceRect {
                    x: 5 * i,
                    y: 0,
                    width: 5,
                    height: 10,
                },
                expanded: SourceRect {
                    x: 5 * i,
                    y: 0,
                    width: 5,
                    height: 10,
                },
                document: "<svg><path id=\"replacement\"/></svg>".into(),
                processing_width: 5,
                processing_height: 10,
            })
            .collect();
        let full = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (10, 10),
            &patches,
            true,
        )
        .unwrap();
        assert!(!full.contains("obsolete"));
        assert!(full.contains("lod-0-replacement") && full.contains("lod-1-replacement"));
        let opaque = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (10, 10),
            &patches,
            false,
        )
        .unwrap();
        assert_eq!(opaque, full);
        patches[1].expanded.x = 4;
        patches[1].core.x = 4; // The summed area still matches, but there is overlap and a hole.
        let partial = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (10, 10),
            &patches,
            true,
        )
        .unwrap();
        assert!(partial.contains("obsolete"));
    }

    #[test]
    fn rate_score_rewards_an_explained_small_edge() {
        let mut source = Raster::blank(32, 32, [1.0; 3]);
        for y in 12..20 {
            for x in 12..20 {
                source.pixels[y * 32 + x] = [0.0; 3];
            }
        }
        let flat = Raster::blank(8, 8, [1.0; 3]);
        let exact = source.clone();
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        };
        assert!(
            perceptual_score(&source, whole, &flat, whole).combined
                > perceptual_score(&source, whole, &exact, whole).combined + 1.0
        );
    }

    #[test]
    fn model_cost_penalizes_uncompressible_partitions() {
        let region = SourceRect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        };
        let flat = vec![0_u32; 32 * 32];
        let checkerboard = (0..32 * 32)
            .map(|index| ((index % 32 + index / 32) % 2) as u32)
            .collect::<Vec<_>>();
        let flat_cost = local_model_cost(region, (32, 32), &flat, (32, 32));
        let checkerboard_cost = local_model_cost(region, (32, 32), &checkerboard, (32, 32));
        assert!(checkerboard_cost > flat_cost + 7.0);
    }

    #[test]
    fn nested_documents_receive_unique_ids_and_source_mapping() {
        let base = "<?xml version=\"1.0\"?><svg width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\"/></svg>";
        let child = "<svg width=\"8\" height=\"8\"><defs><linearGradient id=\"paint-0\"/></defs><path fill=\"url(#paint-0)\"/></svg>";
        let result = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (100, 100),
            &[EmbeddedRefinement {
                core: SourceRect {
                    x: 20,
                    y: 30,
                    width: 40,
                    height: 40,
                },
                expanded: SourceRect {
                    x: 10,
                    y: 20,
                    width: 60,
                    height: 60,
                },
                document: Document::from(child.to_string()),
                processing_width: 60,
                processing_height: 60,
            }],
            false,
        )
        .unwrap();
        assert!(result.contains("x=\"2\" y=\"3\" width=\"4\" height=\"4\""));
        assert!(result.contains("viewBox=\"10 10 40 40\""));
        assert!(result.contains("id=\"lod-0-paint-0\""));
        assert!(result.contains("url(#lod-0-paint-0)"));
    }

    #[test]
    fn transparent_refinement_clips_its_core_out_of_the_base() {
        let base = "<svg width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\"/></svg>";
        let child = "<svg width=\"4\" height=\"4\"></svg>";
        let result = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (10, 10),
            &[EmbeddedRefinement {
                core: SourceRect {
                    x: 2,
                    y: 3,
                    width: 4,
                    height: 4,
                },
                expanded: SourceRect {
                    x: 2,
                    y: 3,
                    width: 4,
                    height: 4,
                },
                document: Document::from(child.to_string()),
                processing_width: 4,
                processing_height: 4,
            }],
            true,
        )
        .unwrap();
        assert!(result.contains("id=\"adaptive-base-clip\""));
        assert!(result.contains("M2 0H6V3H2Z"));
        assert!(!result.contains("<mask"));
        assert!(result.contains("<g clip-path=\"url(#adaptive-base-clip)\">"));
    }
    #[test]
    fn overlapping_transparent_refinements_do_not_restore_the_base() {
        let base = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>"#;
        let child = r#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>"#;
        let refinements: Vec<_> = [2, 4]
            .into_iter()
            .map(|x| {
                let core = SourceRect {
                    x,
                    y: 3,
                    width: 4,
                    height: 4,
                };
                EmbeddedRefinement {
                    core,
                    expanded: core,
                    document: child.into(),
                    processing_width: 4,
                    processing_height: 4,
                }
            })
            .collect();
        let svg = compose_refinements(
            &Document::from((base).to_string()),
            (10, 10),
            (10, 10),
            &refinements,
            true,
        )
        .unwrap();
        assert!(!svg.contains("<mask"));
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(10, 10).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        assert_eq!(pixmap.pixel(1, 4).unwrap().alpha(), 255);
        for x in 2..8 {
            assert_eq!(
                pixmap.pixel(x, 4).unwrap().alpha(),
                0,
                "base leaked at x={x}"
            );
        }
    }
}
