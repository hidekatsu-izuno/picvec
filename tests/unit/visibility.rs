fn visibility_witnesses(
    draws: &[Draw<'_>],
    tiles: &[Vec<usize>],
    width: usize,
    height: usize,
    columns: usize,
) -> Vec<Vec<usize>> {
    visibility_evidence(
        draws,
        tiles,
        width,
        height,
        columns,
        &vec![false; draws.len()],
    )
    .witnesses
}

#[path = "../reference/visibility_reference.rs"]
mod reference;

mod tests {
    use super::*;

    fn layered_scene() -> String {
        let mut svg = String::from(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80"><defs><linearGradient id="g"><stop stop-color="#315a91" stop-opacity="0.3"/><stop offset="1" stop-color="#ef9021" stop-opacity="0.8"/></linearGradient><clipPath id="c"><circle cx="63.5" cy="35" r="29"/></clipPath></defs><rect id="fixed" width="160" height="80" fill="#254a81"/><rect x="3" y="3" width="12" height="12" fill="red"/><rect x="2" y="2" width="20" height="20" fill="#254a81"/><rect x="2" y="2" width="20" height="20" fill="#254a81"/><g clip-path="url(#c)"><rect x="10" y="4" width="110" height="70" fill="url(#g)"/></g>"##,
        );
        for i in 0..36 {
            let x = (i * 17 % 150) as f32 + 0.13;
            let y = (i * 13 % 68) as f32 + 0.37;
            svg.push_str(&format!(r##"<rect x="{x}" y="{y}" width="19.8" height="10.6" fill="#985c31" fill-opacity="0.17"/><path d="M{x} {y}l14 6" fill="none" stroke="#1246ab" stroke-opacity="0.31" stroke-width="0.13"/>"##));
        }
        svg.push_str("</svg>");
        svg
    }

    #[test]
    fn every_mixed_counterfactual_witness_matches_an_independent_omission() {
        let svg = layered_scene().replace("</svg>", r##"<defs><filter id="blur"><feGaussianBlur stdDeviation="1.3"/></filter></defs><g filter="url(#blur)"><rect x="13" y="9" width="77" height="35" fill="#892" fill-opacity=".3"/></g><g style="mix-blend-mode:multiply"><rect x="55" y="32" width="44" height="31" fill="#498" fill-opacity=".2"/></g></svg>"##);
        let (svg, _) = annotate(&Document::from((&svg).to_string()));
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
        let mut draws = Vec::new();
        collect(tree.root(), &mut draws);
        let members: Vec<_> = (0..draws.len()).collect();
        let tiles = vec![members.clone(); 6];
        let witnesses = visibility_witnesses(&draws, &tiles, 160, 80, 3);
        let active = vec![true; draws.len()];
        assert!(witnesses.iter().filter(|p| !p.is_empty()).count() > 8);
        for (i, points) in witnesses.iter().enumerate() {
            for &pixel in points {
                let x = pixel % 160 / TILE * TILE;
                let y = pixel / 160 / TILE * TILE;
                let w = TILE.min(160 - x);
                let h = TILE.min(80 - y);
                let before = reference::draw_tile(&draws, &members, &active, None, x, y, w, h, 1);
                let after = reference::draw_tile(&draws, &members, &active, Some(i), x, y, w, h, 1);
                let k = ((pixel / 160 - y) * w + pixel % 160 - x) * 4;
                assert_ne!(
                    &before.data()[k..k + 4],
                    &after.data()[k..k + 4],
                    "draw {i}, pixel {pixel}"
                );
            }
        }
    }

    #[test]
    fn distant_clipped_patches_and_filtered_spill_match_full_reference() {
        for filter in ["", " filter=\"url(#f)\""] {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="128"><defs><clipPath id="c"><rect width="256" height="128"/></clipPath><filter id="f" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="3"/></filter></defs><rect width="256" height="128" fill="white"/><path d="M125 62H132V67H125Z" fill="red"/><g clip-path="url(#c)"><path d="M2 2H4V4H2Z" fill="blue"/><path d="M244 119H247V123H244Z" fill="green"/><g{filter}><path d="M62 63H63V64H62Z" fill="black"/></g></g></svg>"##
            );
            assert_eq!(
                prune(&Document::from((&svg).to_string()), 256, 128),
                reference::prune(&Document::from((&svg).to_string()), 256, 128)
            );
        }
    }

    #[test]
    fn clip_tile_membership_matches_original_omission_renderer() {
        for transform in ["", " transform=\"translate(0.25 0.5)\""] {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="192" height="128"><defs><clipPath id="c"><path d="M65.1 62.7H80.3V65.2H65.1Z"/></clipPath><linearGradient id="g"><stop stop-color="red" stop-opacity="0.2"/><stop offset="1" stop-color="blue" stop-opacity="0.8"/></linearGradient></defs><rect width="192" height="128" fill="white"/><g{transform}><rect width="192" height="128" fill="url(#g)" clip-path="url(#c)"/><rect width="192" height="128" fill="url(#g)" clip-path="url(#c)"/></g><path d="M64 62H85V63H64Z" fill="white"/></svg>"##
            );
            assert_eq!(
                prune(&Document::from((&svg).to_string()), 192, 128),
                reference::prune(&Document::from((&svg).to_string()), 192, 128)
            );
        }
    }

    #[test]
    fn a_general_removal_invalidates_a_later_prefix_noop() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect id="fixed" width="64" height="64" fill="white"/><rect x="8" y="8" width="16" height="16" fill="red"/><rect x="8" y="8" width="16" height="16" fill="red"/></svg>"##;
        let expected = reference::prune(&Document::from((svg).to_string()), 64, 64);
        assert_eq!(expected.1.shapes, 1);
        assert_eq!(prune(&Document::from((svg).to_string()), 64, 64), expected);
    }

    #[test]
    fn removing_a_covered_draw_invalidates_the_covering_draws_witness() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect id="fixed" width="64" height="64" fill="white"/><rect x="8" y="8" width="16" height="16" fill="red"/><rect x="8" y="8" width="16" height="16" fill="white"/></svg>"##;
        let expected = reference::prune(&Document::from((svg).to_string()), 64, 64);
        assert_eq!(expected.1.shapes, 2);
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    assert_eq!(prune(&Document::from((svg).to_string()), 64, 64), expected);
                });
        }
    }

    #[test]
    fn indexed_presence_matches_recursive_lookup_even_inside_isolated_groups() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><defs><clipPath id="c"><circle cx="32" cy="32" r="20"/></clipPath></defs><rect id="picvec-visibility-0" width="10" height="10"/><g clip-path="url(#c)"><rect id="picvec-visibility-1" width="64" height="64"/></g><path id="picvec-visibility-2" d="M0 0H20"/><rect id="picvec-visibility-03" x="3" width="5" height="5"/><rect id="picvec-visibility-+4" x="4" width="5" height="5"/><rect id="picvec-visibility-99" width="5" height="5"/></svg>"##;
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let indexed = missing_candidates(tree.root(), 8);
        let scanned: Vec<_> = (0..8)
            .map(|i| tree.node_by_id(&format!("{PREFIX}{i}")).is_none())
            .collect();
        assert_eq!(indexed, scanned);
        assert!(!indexed[0] && !indexed[1]);
        assert!(indexed[3] && indexed[4]);
    }

    #[test]
    fn incremental_framebuffers_match_full_redraw_at_every_candidate() {
        let (svg, _) = annotate(&Document::from((&layered_scene()).to_string()));
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
        let mut draws = Vec::new();
        collect(tree.root(), &mut draws);
        let members: Vec<_> = (0..draws.len()).collect();
        for scale in [1, 4, 8] {
            for (x, y, w, h) in [(0, 0, 64, 64), (64, 0, 64, 64), (128, 64, 32, 16)] {
                let view = TileView { x, y, w, h, scale };
                let mut active = vec![true; draws.len()];
                let mut raster = TileRaster::new(view, &draws, &members, &active, 0);
                for i in 0..draws.len() {
                    let before =
                        reference::draw_tile(&draws, &members, &active, None, x, y, w, h, scale);
                    let after =
                        reference::draw_tile(&draws, &members, &active, Some(i), x, y, w, h, scale);
                    assert_eq!(raster.baseline.data(), before.data());
                    let unchanged = raster.unchanged_without(view, &draws, &members, &active, i);
                    assert_eq!(unchanged, before.data() == after.data());
                    let prefix = reference::draw_tile(
                        &draws,
                        &members[..i],
                        &active,
                        None,
                        x,
                        y,
                        w,
                        h,
                        scale,
                    );
                    assert_eq!(raster.prefix.data(), prefix.data());
                    // Include consecutive removals, then advance through only
                    // the retained nodes on the next visit to this tile.
                    if unchanged {
                        active[i] = false;
                    }
                }
            }
        }
    }

    #[test]
    fn lru_evicts_only_the_cold_entry_and_stays_within_the_pixmap_budget() {
        let mut cache = TileCache::new(3);
        let create = || TileRaster {
            baseline: Pixmap::new(256, 256).unwrap(),
            prefix: Pixmap::new(256, 256).unwrap(),
            next: 0,
            layers: HashMap::new(),
        };
        for i in 0..3 {
            cache.get_or_insert((i, 4), create);
        }
        cache.get_or_insert((0, 4), || panic!("hot entry was lost"));
        cache.get_or_insert((3, 4), create);
        assert_eq!(cache.entries.len(), 3);
        assert!(!cache.entries.contains_key(&(1, 4)));
        for key in [(0, 4), (2, 4), (3, 4)] {
            cache.get_or_insert(key, || panic!("unrelated entry was evicted"));
        }
        for i in 4..140 {
            cache.get_or_insert((i, 4), create);
        }
        let bytes: usize = cache
            .entries
            .values()
            .map(|v| v.raster.baseline.data().len() + v.raster.prefix.data().len())
            .sum();
        assert_eq!(bytes, cache.capacity * 2 * 256 * 256 * 4);
        assert!(CACHE_TILES * 2 * 256 * 256 * 4 <= 32 * 1024 * 1024);
    }

    #[test]
    fn eviction_and_transparency_preserve_the_reference_removal_decisions() {
        let width = (CACHE_TILES + 5) * TILE;
        let mut wide = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="72"><rect id="fixed" width="{width}" height="72" fill="#125641"/>"##
        );
        // Cross the cache limit, then return to tiles with retained and
        // removed candidates at both scales, including partial edge tiles.
        for pass in 0..3 {
            for i in 0..CACHE_TILES + 5 {
                let x = i * TILE + 1;
                let opacity = if pass == 2 { 0.2 } else { 1.0 };
                if pass == 0 {
                    // Fully covered interior mark; the fractional outer rims
                    // themselves must stay because their AA accumulates.
                    wide.push_str(&format!(
                        r##"<rect x="{}" y="10" width="4" height="4" fill="#ff0077"/>"##,
                        x + 5
                    ));
                }
                wide.push_str(&format!(r##"<rect x="{x}" y="3.25" width="66" height="64.5" fill="#192581" fill-opacity="{opacity}"/>"##));
            }
        }
        wide.push_str("</svg>");
        for (svg, w, h) in [(layered_scene(), 160, 80), (wide, width, 72)] {
            let expected = reference::prune(&Document::from((&svg).to_string()), w, h);
            assert!(expected.1.shapes > 0, "fixture {w}x{h} needs covered marks");
            for capacity in [1, 3, CACHE_TILES] {
                let actual = prune_cached(&Document::from((&svg).to_string()), w, h, capacity);
                assert_eq!(actual, expected);
            }
            for scale in [1, 4] {
                assert_eq!(image(&svg, scale), image(&expected.0, scale));
            }
        }
    }

    #[test]
    #[ignore = "stage benchmark; set PICVEC_VISIBILITY_BENCH_INPUTS and PICVEC_VISIBILITY_BENCH_OUTPUT"]
    fn benchmark_emitted_svgs_against_full_redraw_reference() {
        let paths = std::env::var_os("PICVEC_VISIBILITY_BENCH_INPUTS").unwrap();
        let output = std::env::var_os("PICVEC_VISIBILITY_BENCH_OUTPUT").unwrap();
        let repeats = std::env::var("PICVEC_VISIBILITY_BENCH_REPEATS")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(repeats > 0);
        let mut rows = Vec::new();
        for path in std::env::split_paths(&paths) {
            let svg = std::fs::read_to_string(&path).unwrap();
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
            let (w, h) = (
                tree.size().width().ceil() as usize,
                tree.size().height().ceil() as usize,
            );
            drop(tree);
            let measure = |reference_version| {
                let start = crate::time::Instant::now();
                let result = if reference_version {
                    reference::prune(&Document::from((&svg).to_string()), w, h)
                } else {
                    prune(&Document::from((&svg).to_string()), w, h)
                };
                (result, start.elapsed().as_secs_f64())
            };
            if std::env::var_os("PICVEC_VISIBILITY_BENCH_CANDIDATE_ONLY").is_some() {
                // Profiling only; this mode makes no reference-equivalence claim.
                for repeat in 0..repeats {
                    let (candidate, seconds) = measure(false);
                    std::fs::write(
                        std::path::Path::new(&output).with_extension("svg"),
                        &candidate.0,
                    )
                    .unwrap();
                    rows.push(serde_json::json!({"input": path, "repeat": repeat, "candidate_seconds": seconds, "reference_compared": false, "removed_shapes": candidate.1.shapes}));
                    std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
                    eprintln!("visibility profiling {}: {seconds:.3}s", path.display());
                }
                continue;
            }
            for repeat in 0..repeats {
                let (baseline, candidate) = if repeat % 2 == 0 {
                    (measure(true), measure(false))
                } else {
                    let candidate = measure(false);
                    (measure(true), candidate)
                };
                assert_eq!(baseline.0, candidate.0);
                eprintln!(
                    "visibility {} trial {}: {:.3}s -> {:.3}s",
                    path.display(),
                    repeat + 1,
                    baseline.1,
                    candidate.1
                );
                rows.push(serde_json::json!({"input": path, "repeat": repeat, "baseline_seconds": baseline.1, "candidate_seconds": candidate.1, "input_bytes": svg.len(), "output_bytes": candidate.0.0.len(), "removed_shapes": candidate.0.1.shapes, "outputs_equal": true}));
                std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
            }
        }
    }

    #[test]
    fn geometric_covers_preserve_holes_alpha_and_subpixel_exposure() {
        let cases = [
            // Opaque cubic encloses the mark with a wide margin.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M4 32C4 4 60 4 60 32C60 60 4 60 4 32Z"/>"##,
                true,
            ),
            // A curved notch enters the query despite the broad outer bounds.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M2 2H62V62H2V40C50 40 50 24 2 24Z"/>"##,
                false,
            ),
            // Opposite winding makes a nonzero-rule hole too.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M2 2H62V62H2Z M30 30V34H34V30Z"/>"##,
                false,
            ),
            // A hole inside the queried bounds defeats boundary-only containment.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" fill-rule="evenodd" d="M2 2H62V62H2Z M30 30H34V34H30Z"/>"##,
                false,
            ),
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><rect x="2" y="2" width="60" height="60" fill="blue" fill-opacity=".5"/>"##,
                false,
            ),
            // A broad object with only a subpixel strip exposed.
            (
                r##"<rect x="8" y="8" width="40.1" height="40" fill="red"/><rect x="2" y="2" width="46" height="60" fill="blue"/>"##,
                false,
            ),
            // Implicit closure and a transformed quadratic boundary.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path transform="translate(2 2)" fill="blue" d="M0 0H60V60H0 Q-5 30 0 0"/>"##,
                true,
            ),
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><defs><clipPath id="c"><rect width="30" height="64"/></clipPath></defs><rect width="64" height="64" fill="blue" clip-path="url(#c)"/>"##,
                false,
            ),
            // Shared boundaries are excluded even when both fills are opaque.
            (
                r##"<rect x="8" y="8" width="40" height="40" fill="red"/><rect x="8" y="8" width="40" height="40" fill="blue"/>"##,
                false,
            ),
        ];
        for (body, expected_cover) in cases {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">{body}</svg>"##
            );
            let document = Document::from(svg.clone());
            let (annotated, _) = annotate(&document);
            let tree = usvg::Tree::from_str(&annotated, &usvg::Options::default()).unwrap();
            let mut draws = Vec::new();
            collect(tree.root(), &mut draws);
            let covered = coverage::Cover::new(draws.last().unwrap().node)
                .is_some_and(|cover| cover.contains(draws[0].bounds));
            assert_eq!(covered, expected_cover, "{body}");
            let baseline = prune_impl(&document, 64, 64, CACHE_TILES, false);
            let candidate = prune_impl(&document, 64, 64, CACHE_TILES, true);
            assert_eq!(baseline, candidate, "{body}");
            for scale in [1, 4, 8, 16] {
                assert!(
                    image(&svg, scale) == image(&candidate.0, scale),
                    "RGBA differs at scale {scale}: {body}"
                );
            }
        }
    }

    #[test]
    fn unresolved_subpixel_exposure_retains_the_legacy_fallback() {
        // Characterize a pre-existing limitation, not a geometry shortcut:
        // 1x/4x equality does not imply equality at arbitrary SVG zoom levels.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect x="8" y="8" width="40.01" height="40" fill="red"/><rect x="2" y="2" width="46" height="60" fill="blue"/></svg>"##;
        let document = Document::from(svg);
        let baseline = prune_impl(&document, 64, 64, CACHE_TILES, false);
        let candidate = prune_impl(&document, 64, 64, CACHE_TILES, true);
        assert_eq!(baseline, candidate);
        assert_eq!(candidate.1.shapes, 1);
        for scale in [1, 4, 8] {
            assert!(image(svg, scale) == image(&candidate.0, scale));
        }
        assert!(image(svg, 16) != image(&candidate.0, 16));
    }

    #[test]
    #[ignore = "prototype A/B benchmark; set PICVEC_VISIBILITY_BENCH_INPUTS and PICVEC_VISIBILITY_BENCH_OUTPUT"]
    fn benchmark_geometric_coverage() {
        let paths = std::env::var_os("PICVEC_VISIBILITY_BENCH_INPUTS").unwrap();
        let output = std::env::var_os("PICVEC_VISIBILITY_BENCH_OUTPUT").unwrap();
        let repeats = std::env::var("PICVEC_VISIBILITY_BENCH_REPEATS")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3);
        let mut rows = Vec::new();
        for path in std::env::split_paths(&paths) {
            let document = Document::from(std::fs::read_to_string(&path).unwrap());
            let tree = usvg::Tree::from_str(&document, &usvg::Options::default()).unwrap();
            let (w, h) = (
                tree.size().width().ceil() as usize,
                tree.size().height().ceil() as usize,
            );
            drop(tree);
            let measure = |geometric| {
                let start = crate::time::Instant::now();
                let result = prune_impl(&document, w, h, CACHE_TILES, geometric);
                (result, start.elapsed().as_secs_f64())
            };
            // Warm both implementations; alternate order in measured trials.
            assert_eq!(measure(false).0, measure(true).0);
            for repeat in 0..repeats {
                let (baseline, candidate) = if repeat % 2 == 0 {
                    (measure(false), measure(true))
                } else {
                    let candidate = measure(true);
                    (measure(false), candidate)
                };
                assert_eq!(baseline.0, candidate.0, "{}", path.display());
                eprintln!(
                    "geometry {} trial {}: {:.3}s -> {:.3}s",
                    path.display(),
                    repeat + 1,
                    baseline.1,
                    candidate.1
                );
                rows.push(
                    serde_json::json!({"input": path, "repeat": repeat, "width": w, "height": h,
                    "baseline_seconds": baseline.1, "candidate_seconds": candidate.1,
                    "removed_shapes": candidate.0.1.shapes, "outputs_equal": true}),
                );
                std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
            }
        }
    }

    fn image(svg: &str, scale: u32) -> Vec<u8> {
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let mut p = Pixmap::new(
            tree.size().width() as u32 * scale,
            tree.size().height() as u32 * scale,
        )
        .unwrap();
        resvg::render(
            &tree,
            Transform::from_scale(scale as f32, scale as f32),
            &mut p.as_mut(),
        );
        p.take()
    }
    #[test]
    fn removes_hidden_faces_and_redundant_lines_but_keeps_partial_overlap() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><g fill="red"><rect x="5" y="5" width="5" height="5"/><rect width="20" height="20"/><path d="M5 5L15 15" fill="none" stroke="red" stroke-width="2"/><rect x="15" y="15" width="10" height="10" fill="blue"/></g></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 2);
        assert_eq!(report.strokes, 1);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn tile_boundaries_and_multiple_removals_preserve_the_complete_render() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="32"><rect width="256" height="32" fill="red"/><rect x="60.2" y="4" width="120" height="16" fill="red"/><path d="M10 15.3H200" fill="none" stroke="red" stroke-width="2"/><rect x="128" y="16" width="80" height="16" fill="blue"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 256, 32);
        assert_eq!(report.shapes, 2);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn zero_area_paint_is_removed_but_its_stroked_counterpart_is_kept() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M2 2H20" fill="red"/><path d="M2 2H20" fill="none" stroke="red"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 1);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
    #[test]
    fn preserves_alpha_accumulation_gradients_and_subpixel_lines() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect x="10" width="10" height="25" fill="url(#g)"/><path d="M1 1L30 30" fill="none" stroke="black" stroke-width="0.1"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(output, Document::from(svg));
    }
    #[test]
    fn preserves_clip_context_and_referenced_geometry() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><clipPath id="c"><rect width="10" height="10"/></clipPath></defs><rect id="base" width="32" height="32" fill="red"/><g clip-path="url(#c)"><rect width="32" height="32" fill="blue"/></g></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
}
