pub(crate) fn build_with_source(
    segmentation: &Segmentation,
    topology: &HierarchicalTopology,
    reference: &Raster,
) -> (Vec<RegionGeometry>, GeometrySummary) {
    assert_eq!(
        (segmentation.width, segmentation.height),
        (reference.width, reference.height)
    );
    build_internal(
        segmentation,
        Some(topology),
        Some(reference),
        None,
        0.0,
        &[],
        None,
        &[],
    )
}

pub(crate) fn build_with_source_alpha(
    segmentation: &Segmentation,
    topology: &HierarchicalTopology,
    reference: &Raster,
    matte: &crate::chroma::AlphaMatte,
) -> (Vec<RegionGeometry>, GeometrySummary) {
    assert_eq!(
        (segmentation.width, segmentation.height),
        (matte.width, matte.height)
    );
    assert_eq!(
        (reference.width, reference.height),
        (matte.width, matte.height)
    );
    build_internal(
        segmentation,
        Some(topology),
        Some(reference),
        Some(matte),
        0.0,
        &[],
        None,
        &[],
    )
}

mod tests {
    use super::*;

    #[test]
    fn shared_preparation_does_not_retain_overlap_or_order_from_previous_builds() {
        let (width, height) = (32, 24);
        let labels: Vec<u32> = (0..width * height)
            .map(|i| {
                let boundary = 12 + i / width / 4;
                if i % width < boundary {
                    0
                } else if i % width < boundary + 3 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let colours = [[0.2; 3], [0.05; 3], [0.8; 3]];
        let source = Raster::new(
            width,
            height,
            labels.iter().map(|&id| colours[id as usize]).collect(),
        );
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            canonical: source.clone(),
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; labels.len()],
            regions: (0..3)
                .map(|id| RegionStats {
                    id,
                    area: labels.iter().filter(|&&label| label == id).count(),
                    min_x: 0,
                    min_y: 0,
                    max_x: width,
                    max_y: height,
                    mean_rgb: colours[id as usize],
                    mean_lab: rgb_to_oklab(colours[id as usize]),
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let topology = HierarchicalTopology::build(&segmentation);
        let matte = crate::chroma::AlphaMatte::from_u8(
            width,
            height,
            labels
                .iter()
                .map(|&id| if id == 2 { 128 } else { 255 })
                .collect(),
        );
        for alpha in [None, Some(&matte)] {
            let prepared =
                PreparedGeometry::new(&segmentation, Some(&topology), Some(&source), alpha, &[]);
            for (overlap, opaque, order) in [
                (0.6, [true; 3], Some([2, 0, 1])),
                (0.0, [true, true, false], Some([0, 1, 2])),
                (0.6, [true; 3], None),
                (0.6, [true; 3], Some([2, 0, 1])),
            ] {
                let order = order.as_ref().map(|r| r.as_slice());
                let reused = prepared.build(overlap, &opaque, order);
                let fresh = build_internal(
                    &segmentation,
                    Some(&topology),
                    Some(&source),
                    alpha,
                    overlap,
                    &opaque,
                    order,
                    &[],
                );
                assert_eq!(format!("{reused:?}"), format!("{fresh:?}"));
            }
        }
    }

    #[test]
    fn incident_chain_index_matches_the_complete_ownership_scan() {
        let owners: Vec<BTreeSet<usize>> = (0..64)
            .map(|mask| (0..6).filter(|&owner| mask & (1 << owner) != 0).collect())
            .collect();
        let indexed = chains_by_owner(&owners, 6);
        for (owner, chains) in indexed.iter().enumerate() {
            let scanned: Vec<usize> = owners
                .iter()
                .enumerate()
                .filter(|(_, set)| set.contains(&owner))
                .map(|(chain, _)| chain)
                .collect();
            assert_eq!(*chains, scanned);
        }
    }

    #[test]
    fn segment_budget_preserves_every_competitive_cubic_fit() {
        for count in [2, 17, 129, 513] {
            for amplitude in [0.0_f32, 1.0, 12.0] {
                let points: Vec<Point> = (0..count)
                    .map(|i| Point {
                        x: i as f32 * 0.35,
                        y: amplitude * ((i as f32 * 0.17).sin() + 0.2 * (i as f32 * 1.3).cos()),
                    })
                    .collect();
                let left = normalized(Point {
                    x: points[1].x - points[0].x,
                    y: points[1].y - points[0].y,
                });
                let right = normalized(Point {
                    x: points[count - 2].x - points[count - 1].x,
                    y: points[count - 2].y - points[count - 1].y,
                });
                for tolerance in [0.1_f32, 0.75, 1.25] {
                    let complete = fit_cubic_recursive(&points, left, right, tolerance * tolerance);
                    for budget in [
                        0,
                        1,
                        complete.len().saturating_sub(1),
                        complete.len(),
                        complete.len() + 3,
                    ] {
                        let limited = fit_cubic_with_budget(
                            &points,
                            left,
                            right,
                            tolerance * tolerance,
                            budget,
                        );
                        if complete.len() > budget {
                            assert!(limited.is_none());
                        } else {
                            assert_eq!(limited.unwrap(), complete);
                        }
                    }
                }
            }
        }
    }

    use crate::color::rgb_to_oklab;
    use crate::segment::{RegionStats, Segmentation, SegmentationSummary};

    #[test]
    fn expanded_hook_boundary_does_not_open_a_transparent_puncture() {
        use resvg::{
            tiny_skia::{Pixmap, Transform},
            usvg::{Options, Tree},
        };
        let p = |x, y| Point { x, y };
        let hook = CurveSegment::Cubic {
            start: p(20.0, 10.0),
            first: p(26.8, 10.0),
            second: p(6.8, 18.8),
            end: p(26.9, 14.75),
        };
        let line = |start, end| CurveSegment::Line { start, end };
        let lower = vec![
            line(p(0.0, 10.0), hook.start()),
            hook,
            line(hook.end(), p(40.0, 15.0)),
            line(p(40.0, 15.0), p(40.0, 30.0)),
            line(p(40.0, 30.0), p(0.0, 30.0)),
            line(p(0.0, 30.0), p(0.0, 10.0)),
        ];
        let upper = vec![
            line(p(0.0, 0.0), p(40.0, 0.0)),
            line(p(40.0, 0.0), p(40.0, 15.0)),
            line(p(40.0, 15.0), hook.end()),
            hook.reversed(),
            line(hook.start(), p(0.0, 10.0)),
            line(p(0.0, 10.0), p(0.0, 0.0)),
        ];
        let flags = vec![true, true, true, false, false, false];
        let (mut expanded, flags, _) = prepare_offset_segments(lower, flags, vec![false; 6]);
        expand_hidden_edges(&mut expanded, &flags, 0.6);
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="30"><path d="{}"/><path d="{}"/></svg>"#,
            structural_curve_path_data(&expanded, true),
            structural_curve_path_data(&upper, true)
        );
        let tree = Tree::from_str(&svg, &Options::default()).unwrap();
        let mut pix = Pixmap::new(160, 120).unwrap();
        resvg::render(&tree, Transform::from_scale(4.0, 4.0), &mut pix.as_mut());
        for y in 4..116 {
            for x in 40..128 {
                assert!(
                    pix.pixels()[y * 160 + x].alpha() >= 250,
                    "puncture at {},{}",
                    x as f32 / 4.0,
                    y as f32 / 4.0
                );
            }
        }
    }

    #[test]
    fn source_alpha_simplifies_the_rgb_rim_as_well_as_the_mask() {
        let input = image::load_from_memory(include_bytes!("../data/remojii-rim-alpha.png"))
            .unwrap()
            .resize(384, 384, image::imageops::FilterType::Triangle)
            .to_luma8();
        let (width, height) = (input.width() as usize, input.height() as usize);
        let matte = crate::chroma::AlphaMatte::from_u8(width, height, input.into_raw());
        let labels: Vec<u32> = (0..width * height)
            .map(|i| u32::from(matte.get(i) > 0.0))
            .collect();
        let reference = Raster::blank(width, height, [0.0; 3]);
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; labels.len()],
            canonical: reference.clone(),
            regions: (0..2)
                .map(|id| RegionStats {
                    id,
                    area: labels.iter().filter(|&&label| label == id).count(),
                    min_x: 0,
                    min_y: 0,
                    max_x: width,
                    max_y: height,
                    mean_rgb: [0.0; 3],
                    mean_lab: rgb_to_oklab([0.0; 3]),
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let topology = HierarchicalTopology::build(&segmentation);
        let (before, _) = build_with_source(&segmentation, &topology, &reference);
        let (after, _) = build_with_source_alpha(&segmentation, &topology, &reference, &matte);
        let commands = |geometries: &[RegionGeometry]| {
            geometries
                .iter()
                .find(|g| g.region == 1)
                .unwrap()
                .path_data
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .count()
        };
        let (old, new) = (commands(&before), commands(&after));
        assert!(
            new * 4 < old * 3,
            "RGB silhouette still has {new} commands (previously {old})"
        );
        assert_eq!(
            before.iter().find(|g| g.region == 1).unwrap().loops.len(),
            after.iter().find(|g| g.region == 1).unwrap().loops.len(),
            "the silhouette must retain disconnected pieces and transparent holes"
        );
    }

    #[test]
    fn a_highlight_edge_stays_continuous_inside_one_material_class() {
        let width = 64;
        let colours = [[0.60; 3], [0.80; 3], [0.82; 3], [0.70; 3]];
        let labels: Vec<u32> = (0..width * width)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                if y >= 54 {
                    3
                } else if x <= y + 4 {
                    0
                } else {
                    1 + (y / 10 % 2) as u32
                }
            })
            .collect();
        let raster = Raster::new(
            width,
            width,
            labels.iter().map(|&l| colours[l as usize]).collect(),
        );
        let segmentation = Segmentation {
            width,
            height: width,
            labels: labels.clone(),
            paint_keys: (0..4).collect(),
            paint_samples: vec![true; width * width],
            canonical: raster,
            regions: colours
                .iter()
                .enumerate()
                .map(|(id, &colour)| {
                    let pixels: Vec<_> = labels
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &l)| (l as usize == id).then_some(i))
                        .collect();
                    RegionStats {
                        id: id as u32,
                        area: pixels.len(),
                        min_x: pixels.iter().map(|i| i % width).min().unwrap(),
                        min_y: pixels.iter().map(|i| i / width).min().unwrap(),
                        max_x: pixels.iter().map(|i| i % width + 1).max().unwrap(),
                        max_y: pixels.iter().map(|i| i / width + 1).max().unwrap(),
                        mean_rgb: colour,
                        mean_lab: rgb_to_oklab(colour),
                    }
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pairs = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, ..) =
            build_shared_chains(&segmentation, None, &[], stride, &edges, &pairs);
        let mut checked = HashSet::new();
        let mut points = 0;
        let mut maximum_error = 0.0_f32;
        for (pair, edges) in &pairs {
            if pair.0 != 0 || !(1..=2).contains(&pair.1) {
                continue;
            }
            for edge in edges {
                let chain = lookup[edge].0;
                if !checked.insert(chain) {
                    continue;
                }
                for p in sample_curve_sequence(&chains[chain].segments, 0.5) {
                    if (8.0..48.0).contains(&p.y) {
                        maximum_error = maximum_error.max((p.x - p.y - 4.5).abs());
                        points += 1;
                    }
                }
            }
        }
        assert!(points > 60);
        assert!(
            maximum_error < 0.3,
            "junction displacement: {maximum_error}"
        );
    }

    #[test]
    fn shading_fairing_requires_source_colour_support() {
        let points: Vec<_> = (4..60)
            .map(|x| Point {
                x: x as f32,
                y: 24.0 + 2.0 * (x as f32 * 0.3).sin(),
            })
            .collect();
        let before: Vec<_> = points
            .windows(2)
            .map(|p| CurveSegment::Line {
                start: p[0],
                end: p[1],
            })
            .collect();
        let after = [CurveSegment::Line {
            start: points[0],
            end: *points.last().unwrap(),
        }];
        let smooth = Raster::new(
            64,
            48,
            (0..64 * 48)
                .map(|i| [0.3 + (i / 64) as f32 * 0.001; 3])
                .collect(),
        );
        assert!(source_supports_shading_fit(&smooth, &before, &after));
        let hard = Raster::new(
            64,
            48,
            (0..64 * 48)
                .map(|i| {
                    let border = 24.0 + 2.0 * (((i % 64) as f32 + 0.5) * 0.3).sin();
                    if (i / 64) as f32 + 0.5 < border {
                        [0.1; 3]
                    } else {
                        [0.9; 3]
                    }
                })
                .collect(),
        );
        assert!(!source_supports_shading_fit(&hard, &before, &after));
    }

    #[test]
    fn offset_band_maps_the_displacement_at_the_window_corner() {
        let points: Vec<_> = include_str!("../data/car-window-inset-contour.txt")
            .lines()
            .map(|line| {
                let mut values = line.split_whitespace().map(|v| v.parse::<f32>().unwrap());
                Point {
                    x: values.next().unwrap(),
                    y: values.next().unwrap(),
                }
            })
            .collect();
        let curves =
            geometry_bezier::fit_closed(&points, geometry_bezier::CLOSED_CORRIDOR).unwrap();
        let model = ClosedContour {
            points: sample_curve_sequence(&curves, 0.75),
            curves,
            is_ellipse: false,
            fallback: None,
        };
        let outer = offset_outline(&model, 2.405_499_5).unwrap();
        let inner = offset_outline(&outer, -3.25).unwrap();
        let band = outline_between(&outer, &inner)
            .expect("the measured window collar must retain its corner correspondence");
        assert_eq!(band.outer_edges.len(), band.inner_edges.len());
        for (a, b) in [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)] {
            assert!(!band.patch(a, b).is_empty());
        }
    }

    #[test]
    fn changing_alpha_winding_preserves_the_fitted_silhouette() {
        let input = image::load_from_memory(include_bytes!("../data/cube-alpha.png"))
            .unwrap()
            .to_luma8();
        let (width, height) = input.dimensions();
        let matte =
            crate::chroma::AlphaMatte::from_u8(width as usize, height as usize, input.into_raw());
        for contour in matte.isocontours(0.5) {
            let render = |winding, scale| {
                let path = oriented_alpha_contour_path_data(&contour, winding);
                let svg = format!(
                    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><path d=\"{path}\" fill=\"red\" fill-opacity=\"0.5\"/></svg>"
                );
                let tree = resvg::usvg::Tree::from_str(&svg, &Default::default()).unwrap();
                let mut image =
                    resvg::tiny_skia::Pixmap::new(width * scale, height * scale).unwrap();
                resvg::render(
                    &tree,
                    resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
                    &mut image.as_mut(),
                );
                image
            };
            for scale in [1, 4] {
                let before = render(1.0, scale);
                let after = render(-1.0, scale);
                assert!(before
                    .data()
                    .iter()
                    .zip(after.data())
                    .all(|(&a, &b)| a.abs_diff(b) <= 1));
            }
        }
    }

    #[test]
    fn cube_alpha_contour_does_not_restore_corner_spurs() {
        let input = image::load_from_memory(include_bytes!("../data/cube-alpha.png"))
            .unwrap()
            .to_luma8();
        let matte = crate::chroma::AlphaMatte::from_u8(
            input.width() as usize,
            input.height() as usize,
            input.into_raw(),
        );
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);
        let (_, curves) = fit_alpha_contour(&contours[0]);
        let samples = sample_curve_sequence(&curves, 0.1);
        let left = samples.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let bottom = samples
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(left > 18.0, "left spur survived: {left}");
        assert!(bottom < 387.0, "bottom spur survived: {bottom}");
    }

    #[test]
    fn corner_regularization_preserves_narrow_crescent_tips() {
        let contour = include_str!("../data/narrow-crescent-contour.txt")
            .lines()
            .map(|line| {
                let mut values = line.split_whitespace().map(|v| v.parse::<f32>().unwrap());
                Point {
                    x: values.next().unwrap(),
                    y: values.next().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        // Neither placement nor the direction of contour traversal may turn
        // a genuine narrow tip into a removable corner excursion.
        for reflected in [false, true] {
            for reversed in [false, true] {
                let mut raw = contour
                    .iter()
                    .map(|p| Point {
                        x: 100.0 + if reflected { -p.x } else { p.x },
                        y: 100.0 + p.y,
                    })
                    .collect::<Vec<_>>();
                if reversed {
                    raw.reverse();
                }
                let result = regularize_short_corner_excursions(
                    &raw,
                    &vec![RegionPair::new(0, 1); raw.len() - 1],
                    &HashSet::new(),
                    256,
                    0.5,
                    None,
                );
                assert!(
                    result.changed.is_empty(),
                    "crescent tip removed: reflected={reflected}, reversed={reversed}"
                );
            }
        }
    }

    #[test]
    fn source_supported_hooks_survive_excursion_removal() {
        let contour = include_str!("../data/hooked-contour.txt")
            .lines()
            .map(|line| {
                let values = line
                    .split_whitespace()
                    .map(|v| v.parse::<f32>().unwrap())
                    .collect::<Vec<_>>();
                Point {
                    x: values[0],
                    y: values[1],
                }
            })
            .collect::<Vec<_>>();
        let width = 96;
        let height = 96;
        for reflected in [false, true] {
            for reversed in [false, true] {
                for inverted in [false, true] {
                    let mut raw = contour
                        .iter()
                        .map(|p| Point {
                            x: 40.0 + if reflected { -p.x } else { p.x },
                            y: 20.0 + p.y,
                        })
                        .collect::<Vec<_>>();
                    if reversed {
                        raw.reverse();
                    }
                    let pairs = vec![RegionPair::new(0, 1); raw.len() - 1];
                    let without_source = regularize_short_corner_excursions(
                        &raw,
                        &pairs,
                        &HashSet::new(),
                        width + 1,
                        0.5,
                        None,
                    );
                    assert!(
                        !without_source.changed.is_empty(),
                        "fixture must exercise excursion removal"
                    );
                    let rasterize = |points: &[Point]| -> Vec<u32> {
                        (0..width * height)
                            .map(|index| {
                                let x = (index % width) as f32 + 0.5;
                                let y = (index / width) as f32 + 0.5;
                                let mut inside = false;
                                for i in 0..points.len() {
                                    let a = points[i];
                                    let b = points[(i + 1) % points.len()];
                                    if (a.y > y) != (b.y > y)
                                        && x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y)
                                    {
                                        inside = !inside;
                                    }
                                }
                                u32::from(inside)
                            })
                            .collect()
                    };
                    let colours = if inverted {
                        [[0.05; 3], [0.95; 3]]
                    } else {
                        [[0.95; 3], [0.05; 3]]
                    };
                    let labels = rasterize(&raw);
                    let canonical = Raster::new(
                        width,
                        height,
                        labels.iter().map(|&i| colours[i as usize]).collect(),
                    );
                    let regions = (0..2)
                        .map(|i| RegionStats {
                            id: i as u32,
                            area: labels.iter().filter(|&&v| v == i as u32).count(),
                            min_x: 0,
                            min_y: 0,
                            max_x: width,
                            max_y: height,
                            mean_rgb: colours[i],
                            mean_lab: rgb_to_oklab(colours[i]),
                        })
                        .collect();
                    let segmentation = Segmentation {
                        width,
                        height,
                        labels,
                        paint_keys: vec![0, 1],
                        paint_samples: vec![true; width * height],
                        canonical: canonical.clone(),
                        regions,
                        summary: SegmentationSummary::default(),
                    };
                    let authored = regularize_short_corner_excursions(
                        &raw,
                        &pairs,
                        &HashSet::new(),
                        width + 1,
                        0.5,
                        Some((&segmentation, &canonical)),
                    );
                    assert!(authored.changed.is_empty(), "authored hook removed: reflected={reflected}, reversed={reversed}, inverted={inverted}");
                    let corrected = Raster::new(
                        width,
                        height,
                        rasterize(&without_source.points)
                            .iter()
                            .map(|&i| colours[i as usize])
                            .collect(),
                    );
                    let spurious = regularize_short_corner_excursions(
                        &raw,
                        &pairs,
                        &HashSet::new(),
                        width + 1,
                        0.5,
                        Some((&segmentation, &corrected)),
                    );
                    assert!(!spurious.changed.is_empty(), "unsupported spur retained");
                }
            }
        }
    }

    #[test]
    fn cube_bottom_corner_excursion_is_regularized() {
        let raw = include_str!("../data/cube-bottom-corner.txt")
            .lines()
            .map(|line| {
                let mut values = line.split_whitespace().map(|v| v.parse::<f32>().unwrap());
                Point {
                    x: values.next().unwrap(),
                    y: values.next().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        let result = regularize_short_corner_excursions(
            &raw,
            &vec![RegionPair::new(0, 1); raw.len() - 1],
            &HashSet::new(),
            1601,
            0.5,
            None,
        );
        assert!(!result.corners.is_empty(), "cube spike survived");
        assert!(result.points.iter().all(|point| point.y < 387.0));
        assert_eq!(result.points.first(), raw.first());
        assert_eq!(result.points.last(), raw.last());
    }

    #[test]
    fn sorted_fairing_search_matches_exhaustive_hypot_and_ties() {
        let mut reference: Vec<Point> = (0..180)
            .map(|i| Point {
                x: ((i * 37) % 31) as f32 - 15.0,
                y: ((i * 13) % 47) as f32 - 23.0,
            })
            .collect();
        reference.extend([Point { x: -1.0, y: 0.0 }, Point { x: 1.0, y: 0.0 }]);
        let mut order: Vec<_> = (0..reference.len()).collect();
        order.sort_unstable_by(|&a, &b| reference[a].x.total_cmp(&reference[b].x).then(a.cmp(&b)));
        for i in 0..1000 {
            let x = (i % 37) as f64 * 1.31 - 24.0;
            let y = (i / 37) as f64 * 2.17 - 30.0;
            let expected = reference.iter().copied().min_by(|a, b| {
                (x - a.x as f64)
                    .hypot(y - a.y as f64)
                    .total_cmp(&(x - b.x as f64).hypot(y - b.y as f64))
            });
            assert_eq!(nearest_fairing_sample(&reference, &order, x, y), expected);
        }
        let ties = [Point { x: 1.0, y: 0.0 }, Point { x: -1.0, y: 0.0 }];
        assert_eq!(
            nearest_fairing_sample(&ties, &[1, 0], 0.0, 0.0),
            Some(ties[0])
        );
        assert_eq!(nearest_fairing_sample(&[], &[], 0.0, 0.0), None);
    }

    #[test]
    fn corridor_queries_match_exhaustive_distances() {
        let reference: Vec<Point> = (0..200)
            .map(|i| Point {
                x: (i % 20) as f32 * 0.31 - 3.0,
                y: (i / 20) as f32 * 0.27 - 1.5,
            })
            .collect();
        for maximum in [0.0, 0.25, 0.5, std::f32::consts::SQRT_2, 2.0] {
            for i in 0..300 {
                let p = Point {
                    x: (i % 30) as f32 * 0.3 - 4.0,
                    y: (i / 30) as f32 * 0.4 - 2.0,
                };
                let nearest = reference
                    .iter()
                    .map(|&q| p.distance(q))
                    .fold(f32::INFINITY, f32::min);
                assert_eq!(
                    samples_within_corridor(&[p], &reference, maximum),
                    nearest <= maximum
                );
                assert_eq!(
                    nearest_sample_distances(&[p], &reference, maximum),
                    (nearest <= maximum).then_some((nearest, nearest))
                );
            }
        }
        let mut queries = reference.clone();
        assert!(samples_within_corridor(&queries, &reference, 0.5));
        queries.push(Point { x: 100.0, y: 100.0 });
        assert!(!samples_within_corridor(&queries, &reference, 0.5));
        let p = Point { x: -0.25, y: 0.0 };
        let q = Point { x: 0.25, y: 0.0 };
        assert!(samples_within_corridor(&[p], &[q], 0.5));
        assert!(!samples_within_corridor(
            &[p],
            &[q],
            f32::from_bits(0.5_f32.to_bits() - 1)
        ));
        assert!(!samples_within_corridor(&[], &[q], 1.0));
        assert!(!samples_within_corridor(&[p], &[], 1.0));
    }

    #[test]
    fn alpha_fairing_retains_rectangle_corners_and_small_islands() {
        for (left, top, right, bottom) in [(20, 24, 76, 72), (40, 40, 41, 41)] {
            let values = (0..96 * 96)
                .map(|index| {
                    let (x, y) = (index % 96, index / 96);
                    if (left..right).contains(&x) && (top..bottom).contains(&y) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let matte = crate::chroma::AlphaMatte::new(96, 96, values);
            let contours = matte.isocontours(0.5);
            assert_eq!(contours.len(), 1);
            let (_, curves) = fit_alpha_contour(&contours[0]);
            let samples = sample_curve_sequence(&curves, 0.1);
            for (x, y) in [(left, top), (right, top), (right, bottom), (left, bottom)] {
                let corner = Point {
                    x: x as f32,
                    y: y as f32,
                };
                let distance = samples
                    .iter()
                    .map(|point| point.distance(corner))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    distance < 0.8,
                    "corner rounded away: {corner:?}, distance={distance}"
                );
            }
            let area = samples
                .windows(2)
                .map(|p| p[0].x * p[1].y - p[1].x * p[0].y)
                .sum::<f32>()
                .abs()
                * 0.5;
            assert!(
                area > 0.3 * ((right - left) * (bottom - top)) as f32,
                "island collapsed"
            );
        }
    }

    #[test]
    fn raster_alpha_circle_has_smooth_turning_without_losing_its_shape() {
        let size = 192;
        let radius = 80.0;
        let values = (0..size * size)
            .map(|index| {
                let x = (index % size) as f32 + 0.5 - 96.0;
                let y = (index / size) as f32 + 0.5 - 96.0;
                if x.hypot(y) < radius {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let matte = crate::chroma::AlphaMatte::new(size, size, values);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);
        let (_, curves) = fit_alpha_contour(&contours[0]);
        let samples = sample_curve_sequence(&curves, 0.5);
        for point in &samples {
            assert!(
                ((point.x - 96.0).hypot(point.y - 96.0) - radius).abs() < 0.9,
                "smoothed silhouette moved too far: {point:?}"
            );
        }
        let directions = samples
            .windows(2)
            .filter(|pair| pair[0].distance(pair[1]) > 1e-4)
            .map(|pair| {
                normalized(Point {
                    x: pair[1].x - pair[0].x,
                    y: pair[1].y - pair[0].y,
                })
            })
            .collect::<Vec<_>>();
        let total_turn: f32 = directions
            .iter()
            .zip(directions.iter().cycle().skip(1))
            .map(|(a, b)| (a.x * b.y - a.y * b.x).atan2(a.x * b.x + a.y * b.y).abs())
            .sum();
        assert!(
            total_turn < 8.0,
            "raster ripples remain: total turn {total_turn}"
        );
    }

    #[test]
    fn smooth_closed_fit_has_no_corner_at_its_storage_origin() {
        let mut points = (0..48)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / 48.0;
                Point {
                    x: 20.0 + 12.0 * angle.cos(),
                    y: 20.0 + 7.0 * angle.sin(),
                }
            })
            .collect::<Vec<_>>();
        points.push(points[0]);

        let curves = fit_shared_boundary_candidate(
            &points,
            true,
            std::f32::consts::FRAC_1_SQRT_2,
            1.5,
            70.0,
            &HashSet::new(),
            None,
            None,
        );

        let first = curves.first().copied().expect("closed curve");
        let last = curves.last().copied().expect("closed curve");
        assert_eq!(first.start(), points[0]);
        assert_eq!(last.end(), points[0]);
        let outgoing = match first {
            CurveSegment::Cubic { start, first, .. } => normalized(Point {
                x: first.x - start.x,
                y: first.y - start.y,
            }),
            CurveSegment::Line { start, end } => normalized(Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }),
        };
        let incoming = match last {
            CurveSegment::Cubic { second, end, .. } => normalized(Point {
                x: end.x - second.x,
                y: end.y - second.y,
            }),
            CurveSegment::Line { start, end } => normalized(Point {
                x: end.x - start.x,
                y: end.y - start.y,
            }),
        };

        assert!(incoming.x * outgoing.x + incoming.y * outgoing.y > 0.999_999);
        assert!((incoming.x * outgoing.y - incoming.y * outgoing.x).abs() < 1e-4);
    }

    #[test]
    fn structural_endpoint_constraint_preserves_anchor_and_sets_g1_direction() {
        let mut curves = vec![CurveSegment::Cubic {
            start: Point { x: 0.0, y: 0.0 },
            first: Point { x: 1.0, y: 1.0 },
            second: Point { x: 3.0, y: 1.0 },
            end: Point { x: 4.0, y: 0.0 },
        }];
        constrain_structural_endpoint_tangents(
            &mut curves,
            Some(Point { x: 1.0, y: 0.0 }),
            Some(Point { x: 1.0, y: 0.0 }),
        );
        let CurveSegment::Cubic {
            start,
            first,
            second,
            end,
        } = curves[0]
        else {
            panic!("expected cubic");
        };
        assert_eq!(start, Point { x: 0.0, y: 0.0 });
        assert_eq!(end, Point { x: 4.0, y: 0.0 });
        assert_eq!(first.y, start.y);
        assert_eq!(second.y, end.y);
        assert!(first.x > start.x);
        assert!(second.x < end.x);
    }

    #[test]
    fn connected_curve_endpoint_stays_inside_shared_vertex_budget() {
        let stride = 64;
        let start = vertex_id(10, 10, stride);
        let end = vertex_id(11, 10, stride);
        let edge = EdgeKey::new(start, end);
        let spans = HashMap::from([(
            edge,
            vec![AdaptiveCurveSpan {
                master_id: 0,
                curve: straight_cubic(Point { x: 0.0, y: 10.0 }, Point { x: 30.0, y: 10.0 }),
                start_parameter: 0.5,
                end_parameter: 0.75,
            }],
        )]);

        let (at_start, _) =
            connected_adaptive_edge_geometry(&spans, &HashSet::new(), start, end, false, stride)
                .unwrap();
        let (at_end, _) =
            connected_adaptive_edge_geometry(&spans, &HashSet::new(), start, end, true, stride)
                .unwrap();

        assert_eq!(at_start, Point { x: 10.25, y: 10.0 });
        assert_eq!(at_end, Point { x: 11.25, y: 10.0 });
    }

    #[test]
    fn potrace_default_corner_threshold_keeps_a_compact_cusp() {
        let mut master_id = 0;
        let smooth = potrace_corner_curves(
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 3.0 },
            Point { x: 1.0, y: 0.0 },
            [0.0, 1.0, 2.0],
            1.2,
            &mut master_id,
        );
        let sharp = potrace_corner_curves(
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 3.0 },
            Point { x: 1.0, y: 0.0 },
            [0.0, 1.0, 2.0],
            1.0,
            &mut master_id,
        );

        assert_eq!(smooth.len(), 1);
        assert_eq!(sharp.len(), 2);
        assert_eq!(sharp[0].3.end(), Point { x: 0.0, y: 3.0 });
        assert_eq!(sharp[1].3.start(), Point { x: 0.0, y: 3.0 });
    }

    #[test]
    fn adjacent_masters_share_their_raster_vertex_exactly() {
        let stride = 64;
        let first = vertex_id(10, 10, stride);
        let joint = vertex_id(11, 10, stride);
        let last = vertex_id(12, 10, stride);
        let mut geometry = AdaptiveBoundaryGeometry::default();
        geometry.edge_spans.insert(
            EdgeKey::new(first, joint),
            vec![AdaptiveCurveSpan {
                master_id: 1,
                curve: straight_cubic(Point { x: 10.0, y: 10.0 }, Point { x: 11.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );
        geometry.edge_spans.insert(
            EdgeKey::new(joint, last),
            vec![AdaptiveCurveSpan {
                master_id: 2,
                curve: straight_cubic(Point { x: 8.5, y: 10.0 }, Point { x: 12.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );

        let (_, curves) =
            adaptive_chain_curves(&[(first, joint), (joint, last)], &geometry, stride);

        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].end(), Point { x: 11.0, y: 10.0 });
        assert_eq!(curves[1].start(), Point { x: 11.0, y: 10.0 });
    }

    #[test]
    fn distant_master_handoff_does_not_translate_a_control_handle() {
        let stride = 64;
        let first = vertex_id(10, 10, stride);
        let joint = vertex_id(11, 10, stride);
        let last = vertex_id(12, 10, stride);
        let mut geometry = AdaptiveBoundaryGeometry::default();
        geometry.edge_spans.insert(
            EdgeKey::new(first, joint),
            vec![AdaptiveCurveSpan {
                master_id: 1,
                curve: straight_cubic(Point { x: 10.0, y: 10.0 }, Point { x: 11.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );
        geometry.edge_spans.insert(
            EdgeKey::new(joint, last),
            vec![AdaptiveCurveSpan {
                master_id: 2,
                curve: straight_cubic(Point { x: 15.0, y: 10.0 }, Point { x: 12.0, y: 10.0 }),
                start_parameter: 0.0,
                end_parameter: 1.0,
            }],
        );

        let (_, curves) =
            adaptive_chain_curves(&[(first, joint), (joint, last)], &geometry, stride);

        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].end(), Point { x: 11.0, y: 10.0 });
        assert_eq!(curves[1].start(), Point { x: 15.0, y: 10.0 });
    }

    #[test]
    fn dominant_shared_edge_connects_shades_without_merging_distinct_materials() {
        let shades = [
            Oklab {
                l: 4.0,
                a: 13.5,
                b: -17.2,
            },
            Oklab {
                l: 6.6,
                a: 22.0,
                b: -31.8,
            },
            Oklab {
                l: 80.5,
                a: -12.8,
                b: -27.0,
            },
        ];
        assert!(delta_e_ok(shades[0], shades[1]) > 6.9);
        for order in [[0, 1, 2], [2, 1, 0]] {
            let labs: Vec<_> = order.iter().map(|&i| shades[i]).collect();
            let adjacency: Vec<_> = (0..3)
                .map(|a| (0..3).filter(|&b| a != b).map(|b| (b, 8)).collect())
                .collect();
            let classes = contextual_continuity_classes(&labs, &adjacency, &[true; 3]);
            let index = |label| order.iter().position(|&i| i == label).unwrap();
            assert_eq!(classes[index(0)], classes[index(1)]);
            assert_ne!(classes[index(0)], classes[index(2)]);
        }
        let adjacency = vec![
            HashMap::from([(1, 8)]),
            HashMap::from([(0, 8)]),
            HashMap::new(),
        ];
        let separate = contextual_continuity_classes(&shades, &adjacency, &[true; 3]);
        assert_ne!(separate[0], separate[1], "no shared contrasting edge");
        let mut different_lightness = shades;
        different_lightness[1].l = 25.0;
        let adjacency = vec![
            HashMap::from([(1, 8), (2, 8)]),
            HashMap::from([(0, 8), (2, 8)]),
            HashMap::from([(0, 8), (1, 8)]),
        ];
        let separate = contextual_continuity_classes(&different_lightness, &adjacency, &[true; 3]);
        assert_ne!(
            separate[0], separate[1],
            "a real lightness edge must remain separate"
        );
    }

    #[test]
    fn window_ink_antialias_shades_keep_one_continuity_class() {
        // Adjacent source samples at (842,336) and (842,337). The old
        // CIEDE2000 distance was 6.15; their scaled OKLab distance is 10.34.
        // Reusing 6.9 therefore splits the same dark rim after migration.
        let labs: Vec<_> = [[18.0, 0.0, 0.0], [26.0, 23.0, 24.0], [79.0, 85.0, 89.0]]
            .map(|rgb| crate::color::rgb_to_oklab(rgb.map(|v| v / 255.0)))
            .into_iter()
            .collect();
        let adjacency = vec![
            HashMap::from([(1, 8)]),
            HashMap::from([(0, 8), (2, 8)]),
            HashMap::from([(1, 8)]),
        ];
        let classes = contextual_continuity_classes(&labs, &adjacency, &[true; 3]);
        assert_eq!(classes[0], classes[1]);
        assert_ne!(
            classes[1], classes[2],
            "the adjacent grey trim is a separate face"
        );
    }

    #[test]
    fn perceptually_close_adjacent_patch_shares_continuity_class() {
        let labs = vec![
            Oklab {
                l: 20.0,
                a: 0.0,
                b: 0.0,
            },
            Oklab {
                l: 74.8,
                a: -10.8,
                b: -15.4,
            },
            Oklab {
                l: 76.0,
                a: -10.0,
                b: -14.5,
            },
            Oklab {
                l: 100.0,
                a: 0.0,
                b: 0.0,
            },
        ];
        let adjacency = vec![
            HashMap::from([(2, 8)]),
            HashMap::from([(2, 20), (3, 8)]),
            HashMap::from([(0, 8), (1, 20)]),
            HashMap::from([(1, 8)]),
        ];

        let classes = contextual_continuity_classes(&labs, &adjacency, &[true, true, false, true]);

        assert_eq!(classes[2], classes[1]);
        assert_ne!(classes[0], classes[1]);
        assert_ne!(classes[3], classes[1]);
    }

    #[test]
    fn disconnected_or_visibly_distinct_neutral_faces_keep_separate_classes() {
        let labs = vec![
            Oklab {
                l: 20.0,
                a: 0.0,
                b: 0.0,
            },
            Oklab {
                l: 35.0,
                a: 1.0,
                b: -1.0,
            },
            Oklab {
                l: 72.0,
                a: -18.0,
                b: -22.0,
            },
            Oklab {
                l: 36.0,
                a: 0.0,
                b: 0.0,
            },
        ];
        let adjacency = vec![
            HashMap::from([(1, 12)]),
            HashMap::from([(0, 12), (2, 24)]),
            HashMap::from([(1, 24)]),
            HashMap::new(),
        ];

        let classes = contextual_continuity_classes(&labs, &adjacency, &[true; 4]);

        assert_ne!(classes[0], classes[1]);
        assert_ne!(classes[1], classes[2]);
        assert_ne!(classes[1], classes[3]);
    }

    #[test]
    fn coreless_line_cap_does_not_split_supported_surface_contour() {
        let labs = vec![
            Oklab {
                l: 92.0,
                a: 0.0,
                b: 0.0,
            },
            Oklab {
                l: 54.0,
                a: 62.0,
                b: 38.0,
            },
            Oklab {
                l: 59.0,
                a: 58.0,
                b: 34.0,
            },
            Oklab {
                l: 18.0,
                a: 20.0,
                b: 12.0,
            },
        ];
        // Quantisation can split the surface into more neighbours than the
        // exterior, reversing the contact majority without changing material.
        for exterior_contact in [2, 20] {
            let adjacency = vec![
                HashMap::from([(3, exterior_contact)]),
                HashMap::from([(3, 5)]),
                HashMap::from([(3, 4)]),
                HashMap::from([(0, exterior_contact), (1, 5), (2, 4)]),
            ];

            let classes =
                contextual_continuity_classes(&labs, &adjacency, &[true, true, true, false]);

            assert_eq!(classes[1], classes[2]);
            assert_eq!(classes[3], classes[1]);
            assert_ne!(classes[3], classes[0]);
        }
    }

    #[test]
    fn continuity_follows_colour_through_fragments_before_accepting_exterior() {
        let colours = [
            [0.92, 0.92, 0.92], // exterior
            [0.88, 0.13, 0.12], // durable surface
            [0.83, 0.12, 0.11], // outer contour fragment
            [0.86, 0.13, 0.12], // intervening fragment
            [0.05, 0.05, 0.05], // independent durable ink
        ];
        for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]] {
            let mut labs = vec![Oklab::default(); colours.len()];
            let mut interior = vec![false; colours.len()];
            let mut adjacency = vec![HashMap::new(); colours.len()];
            for (original, &label) in order.iter().enumerate() {
                labs[label] = rgb_to_oklab(colours[original]);
                interior[label] = matches!(original, 0 | 1 | 4);
            }
            for (a, b, contact) in [(0, 2, 40), (2, 3, 4), (3, 1, 3), (3, 4, 8)] {
                adjacency[order[a]].insert(order[b], contact);
                adjacency[order[b]].insert(order[a], contact);
            }
            let classes = contextual_continuity_classes(&labs, &adjacency, &interior);
            assert_eq!(classes[order[2]], classes[order[1]]);
            assert_eq!(classes[order[3]], classes[order[1]]);
            assert_ne!(classes[order[0]], classes[order[1]]);
            assert_ne!(classes[order[4]], classes[order[1]]);
        }
    }

    #[test]
    fn closed_continuity_split_preserves_every_boundary_edge() {
        let stride = 64;
        let mut track = Vec::new();
        for x in 0..=48 {
            track.push(vertex_id(x, 0, stride));
        }
        for y in 1..=20 {
            track.push(vertex_id(48, y, stride));
        }
        for x in (0..48).rev() {
            track.push(vertex_id(x, 20, stride));
        }
        for y in (1..20).rev() {
            track.push(vertex_id(0, y, stride));
        }
        track.push(track[0]);

        let arcs = split_closed_continuity_track(&track, stride);

        assert!(arcs.len() >= 2);
        assert!(arcs
            .iter()
            .all(|arc| arc.len() >= 2 && arc[0] != arc[arc.len() - 1]));
        assert_eq!(
            arcs.iter().map(|arc| arc.len() - 1).sum::<usize>(),
            track.len() - 1
        );
        assert!(arcs
            .iter()
            .any(|arc| is_shallow_continuity_arc(arc, stride)));
    }

    #[test]
    fn shaded_diagonal_continuity_master_keeps_analytic_lines() {
        let width = 128;
        let colours = [[0.95; 3], [0.8, 0.1, 0.1], [0.82, 0.11, 0.1]];
        let labels: Vec<u32> = (0..width * width)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                if x < y {
                    0
                } else {
                    1 + (y / 16 % 2) as u32
                }
            })
            .collect();
        let regions = colours
            .iter()
            .enumerate()
            .map(|(id, &rgb)| RegionStats {
                id: id as u32,
                area: labels.iter().filter(|&&label| label == id as u32).count(),
                min_x: 0,
                min_y: 0,
                max_x: width,
                max_y: width,
                mean_rgb: rgb,
                mean_lab: rgb_to_oklab(rgb),
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height: width,
            canonical: Raster::new(
                width,
                width,
                labels.iter().map(|&v| colours[v as usize]).collect(),
            ),
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * width],
            regions,
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pairs = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, ..) =
            build_shared_chains(&segmentation, None, &[], stride, &edges, &pairs);
        let mut checked = HashSet::new();
        let mut lines = 0;
        for (&pair, edges) in &pairs {
            if pair.0 != 0 || pair.1 < 1 {
                continue;
            }
            for edge in edges {
                let id = lookup[edge].0;
                if !checked.insert(id) {
                    continue;
                }
                for curve in &chains[id].segments {
                    let middle = interpolate_point(curve.start(), curve.end(), 0.5);
                    if !(20.0..108.0).contains(&middle.y) {
                        continue;
                    }
                    assert!(matches!(curve, CurveSegment::Line { .. }), "{curve:?}");
                    assert!((middle.x - middle.y + 0.5).abs() < 0.75);
                    lines += 1;
                }
            }
        }
        assert!(lines >= 4, "must exercise multiple Paint pairs");
        let (_, summary) = build(&segmentation);
        assert_eq!(summary.shared_loop_fallbacks, 0);
        assert_eq!(summary.shared_curve_downgrades, 0);
    }

    #[test]
    fn remojii_chain_cap_preserves_the_cut_across_paint_pairs() {
        let coordinates: Vec<[f32; 2]> =
            serde_json::from_str(include_str!("../data/remojii-chain-contour.json")).unwrap();
        let source: Vec<_> = coordinates.iter().map(|&[x, y]| Point { x, y }).collect();
        let corridor = closed_contour_corridor(&source);
        let ellipse = geometry_primitives::fit_closed_cap(&source, 1.25)
            .expect("cut rim must have a cap model");
        assert!(corridor > fairing_raster_corridor());
        assert!(ellipse
            .iter()
            .any(|c| matches!(c, CurveSegment::Line { .. })));
        assert!(ellipse
            .iter()
            .any(|c| matches!(c, CurveSegment::Cubic { .. })));
        let width = 60;
        let height = 52;
        let colours = [[0.02; 3], [0.8, 0.5, 0.12], [0.025; 3]];
        let labels: Vec<u32> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 + 0.5;
                let y = (i / width) as f32 + 0.5;
                let inside = source
                    .windows(2)
                    .filter(|edge| {
                        let [a, b] = [edge[0], edge[1]];
                        (a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x
                    })
                    .count()
                    % 2
                    == 1;
                if inside {
                    1
                } else if y > 33.0 {
                    2
                } else {
                    0
                }
            })
            .collect();
        let regions = colours
            .iter()
            .enumerate()
            .map(|(label, &rgb)| {
                let pixels: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v == label as u32).then_some(i))
                    .collect();
                RegionStats {
                    id: label as u32,
                    area: pixels.len(),
                    min_x: pixels.iter().map(|i| i % width).min().unwrap(),
                    min_y: pixels.iter().map(|i| i / width).min().unwrap(),
                    max_x: pixels.iter().map(|i| i % width + 1).max().unwrap(),
                    max_y: pixels.iter().map(|i| i / width + 1).max().unwrap(),
                    mean_rgb: rgb,
                    mean_lab: rgb_to_oklab(rgb),
                }
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            canonical: Raster::new(
                width,
                height,
                labels.iter().map(|&v| colours[v as usize]).collect(),
            ),
            labels,
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            regions,
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pairs = pair_boundary_edges(&segmentation, stride, None);
        let (_, _, strands, junctions) = boundary_topology(stride, &edges, &pairs);
        let adaptive = fit_adaptive_boundary_geometry(
            &segmentation,
            None,
            &[],
            stride,
            &strands,
            &junctions,
            &pairs,
        );
        let master = adaptive
            .closed_contours
            .iter()
            .find(|c| !c.is_ellipse)
            .unwrap();
        let reference = sample_curve_sequence(&master.curves, 0.1);
        let (chains, lookup, ..) =
            build_shared_chains(&segmentation, None, &[], stride, &edges, &pairs);
        let mut checked = HashSet::new();
        for (pair, edges) in &pairs {
            if pair.0 != 1 && pair.1 != 1 {
                continue;
            }
            for edge in edges {
                let id = lookup[edge].0;
                if !checked.insert(id) {
                    continue;
                }
                for point in sample_curve_sequence(&chains[id].segments, 0.25) {
                    let distance = reference
                        .windows(2)
                        .map(|edge| point_segment_distance(point, edge[0], edge[1]))
                        .fold(f32::INFINITY, f32::min);
                    assert!(distance < 0.01,
                        "accepted ellipse slice reverted at a paint junction: {point:?}, error {distance}");
                }
            }
        }
        assert!(checked.len() >= 2);
    }

    #[test]
    fn shaded_ellipse_ring_reuses_whole_inner_and_outer_contours() {
        let width = 128;
        let colours = [[0.2, 0.4, 0.7], [0.02; 3], [0.9, 0.1, 0.1], [1.0, 0.4, 0.3]];
        for rotation in [0.0_f32, 0.65] {
            let (sin, cos) = rotation.sin_cos();
            let radial = |p: Point| {
                let x = p.x - 63.4;
                let y = p.y - 62.8;
                ((cos * x + sin * y) / 30.0).hypot((-sin * x + cos * y) / 40.0)
            };
            let labels: Vec<_> = (0..width * width)
                .map(|i| {
                    let p = Point {
                        x: (i % width) as f32 + 0.5,
                        y: (i / width) as f32 + 0.5,
                    };
                    let radius = radial(p);
                    if radius > 1.0 {
                        0
                    } else if radius > 0.88 {
                        1
                    } else if p.y < 57.0 {
                        3
                    } else {
                        2
                    }
                })
                .collect();
            let regions = colours
                .iter()
                .enumerate()
                .map(|(label, &rgb)| {
                    let pixels: Vec<_> = labels
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| (v == label as u32).then_some(i))
                        .collect();
                    RegionStats {
                        id: label as u32,
                        area: pixels.len(),
                        min_x: pixels.iter().map(|i| i % width).min().unwrap(),
                        min_y: pixels.iter().map(|i| i / width).min().unwrap(),
                        max_x: pixels.iter().map(|i| i % width + 1).max().unwrap(),
                        max_y: pixels.iter().map(|i| i / width + 1).max().unwrap(),
                        mean_rgb: rgb,
                        mean_lab: rgb_to_oklab(rgb),
                    }
                })
                .collect();
            let canonical = Raster::new(
                width,
                width,
                labels.iter().map(|&v| colours[v as usize]).collect(),
            );
            let segmentation = Segmentation {
                width,
                height: width,
                labels,
                paint_keys: vec![0, 1, 2, 3],
                paint_samples: vec![true; width * width],
                canonical,
                regions,
                summary: SegmentationSummary::default(),
            };
            let stride = width + 1;
            let (edges, _) = region_boundary_edges(&segmentation, stride, None);
            let pairs = pair_boundary_edges(&segmentation, stride, None);
            let (chains, lookup, ..) =
                build_shared_chains(&segmentation, None, &[], stride, &edges, &pairs);
            let mut checked = HashSet::new();
            for (&pair, edges) in &pairs {
                if pair.0 != 1 && pair.1 != 1 {
                    continue;
                }
                let expected = if pair.0 == 0 { 1.0 } else { 0.88 };
                for edge in edges {
                    let chain_id = lookup[edge].0;
                    if !checked.insert(chain_id) {
                        continue;
                    }
                    for point in sample_curve_sequence(&chains[chain_id].segments, 0.25) {
                        assert!(
                            (radial(point) - expected).abs() * 40.0 < 0.25,
                            "whole ellipse lost at a shade junction: {rotation} {pair:?} {point:?}"
                        );
                    }
                }
            }
            assert!(
                checked.len() >= 3,
                "inner contour must span different paint pairs"
            );
            let (_, summary) = build(&segmentation);
            assert_eq!(summary.fitted_ellipse_contours, 2, "{summary:?}");
            assert_eq!(summary.shared_loop_fallbacks, 0, "{summary:?}");
            assert_eq!(summary.shared_curve_downgrades, 0, "{summary:?}");
        }
    }

    #[test]
    fn fragmented_diagonal_silhouette_does_not_export_grid_steps() {
        let width = 96;
        let colours = [
            [0.92; 3],
            [0.88, 0.14, 0.12],
            [0.80, 0.12, 0.10],
            [0.81, 0.12, 0.10],
            [0.86, 0.14, 0.12],
            [1.0, 0.45, 0.40],
            [0.95, 0.42, 0.38],
            [0.96, 0.42, 0.38],
            [0.98, 0.44, 0.39],
        ];
        for rotated in [false, true] {
            let mut labels = vec![0_u32; width * width];
            for y in 8..80 {
                let mut left = 15 + (80 - y) / 2;
                // Quantized coverage changes which side owns one sample;
                // the one-pixel excursion is not a designed polygon corner.
                if (35..47).contains(&y) && y % 7 == 0 {
                    left -= 1;
                }
                for x in left..84 {
                    let mut label = if x < left + 2 {
                        2 + (y / 3 % 2) as u32
                    } else if x < left + 4 {
                        4
                    } else {
                        1
                    };
                    // This durable shaded band cuts the diagonal material
                    // boundary into an open track shorter than forty edges.
                    if (35..47).contains(&y) {
                        label += 4;
                    }
                    let index = if rotated {
                        (width - 1 - x) * width + y
                    } else {
                        y * width + x
                    };
                    labels[index] = label;
                }
            }
            let regions = colours
                .iter()
                .enumerate()
                .map(|(label, &colour)| {
                    let pixels: Vec<_> = labels
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &owner)| (owner as usize == label).then_some(index))
                        .collect();
                    RegionStats {
                        id: label as u32,
                        area: pixels.len(),
                        min_x: pixels.iter().map(|index| index % width).min().unwrap(),
                        min_y: pixels.iter().map(|index| index / width).min().unwrap(),
                        max_x: pixels.iter().map(|index| index % width + 1).max().unwrap(),
                        max_y: pixels.iter().map(|index| index / width + 1).max().unwrap(),
                        mean_rgb: colour,
                        mean_lab: rgb_to_oklab(colour),
                    }
                })
                .collect();
            let canonical = Raster::new(
                width,
                width,
                labels
                    .iter()
                    .map(|&label| colours[label as usize])
                    .collect(),
            );
            let segmentation = Segmentation {
                width,
                height: width,
                labels,
                paint_keys: (0..colours.len() as u32).collect(),
                paint_samples: vec![true; width * width],
                canonical,
                regions,
                summary: SegmentationSummary::default(),
            };
            let stride = width + 1;
            let (edges, _) = region_boundary_edges(&segmentation, stride, None);
            let pairs = pair_boundary_edges(&segmentation, stride, None);
            let (chains, lookup, ..) =
                build_shared_chains(&segmentation, None, &[], stride, &edges, &pairs);
            let mut checked = HashSet::new();
            let mut diagonal_segments = 0;
            for (&pair, edges) in &pairs {
                // Include the parallel interior colour boundary: keeping the
                // silhouette smooth must not restore grid fallback next to it.
                if pair.0 != 0 && pair != RegionPair::new(1, 4) && pair != RegionPair::new(5, 8) {
                    continue;
                }
                for edge in edges {
                    let chain = lookup[edge].0;
                    if !checked.insert(chain) {
                        continue;
                    }
                    for curve in &chains[chain].segments {
                        let restore = |p: Point| {
                            if rotated {
                                Point {
                                    x: width as f32 - p.y,
                                    y: p.x,
                                }
                            } else {
                                p
                            }
                        };
                        let a = restore(curve.start());
                        let b = restore(curve.end());
                        let middle = interpolate_point(a, b, 0.5);
                        if !(20.0..68.0).contains(&middle.y) || middle.x > 50.0 {
                            continue;
                        }
                        diagonal_segments += 1;
                        assert!(
                            (a.x - b.x).abs() > 1e-3 && (a.y - b.y).abs() > 1e-3,
                            "raster step in rotated={rotated}: {curve:?}"
                        );
                        let mut backward = [0.0_f32; 2];
                        let direction = (b.y - a.y).signum();
                        for pair in sample_curve_sequence(&[*curve], 0.1).windows(2) {
                            let first = restore(pair[0]);
                            let second = restore(pair[1]);
                            backward[0] += ((second.x - first.x) * direction).max(0.0);
                            backward[1] += (-(second.y - first.y) * direction).max(0.0);
                        }
                        // Subpixel cubic approximation is allowed, but pulling
                        // a shared knot toward a pixel corner causes a visible
                        // backward excursion substantially larger than this.
                        assert!(
                            pair.0 != 0 || backward.into_iter().all(|value| value < 0.1),
                            "backward hook in rotated={rotated}: {curve:?}"
                        );
                    }
                }
            }
            assert!(diagonal_segments > 0);
            let (_, summary) = build(&segmentation);
            assert_eq!(summary.shared_loop_fallbacks, 0);
            assert_eq!(summary.shared_loop_discontinuities, 0);
        }
    }

    #[test]
    fn closed_material_contour_is_fitted_across_quantized_face_transitions() {
        let width = 96;
        let height = 72;
        let centre_x = 48.0_f32;
        let centre_y = 36.0_f32;
        let colours = [
            [0.90, 0.10, 0.08],
            [0.025, 0.025, 0.025],
            [0.06, 0.06, 0.06],
            [0.11, 0.11, 0.11],
        ];
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = (x as f32 + 0.5 - centre_x) / 32.0;
                    let dy = (y as f32 + 0.5 - centre_y) / 24.0;
                    if dx * dx + dy * dy > 1.0 {
                        0
                    } else if y < 30 {
                        1
                    } else if y < 43 {
                        2
                    } else {
                        3
                    }
                })
            })
            .collect();
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| colours[label as usize])
                .collect(),
        );
        let regions = (0..colours.len())
            .map(|label| {
                let pixels: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &owner)| (owner as usize == label).then_some(index))
                    .collect();
                RegionStats {
                    id: label as u32,
                    area: pixels.len(),
                    min_x: pixels.iter().map(|index| index % width).min().unwrap(),
                    min_y: pixels.iter().map(|index| index / width).min().unwrap(),
                    max_x: pixels.iter().map(|index| index % width + 1).max().unwrap(),
                    max_y: pixels.iter().map(|index| index / width + 1).max().unwrap(),
                    mean_rgb: colours[label],
                    mean_lab: rgb_to_oklab(colours[label]),
                }
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels,
            paint_keys: vec![0, 1, 2, 3],
            paint_samples: vec![true; width * height],
            canonical,
            regions,
            summary: SegmentationSummary::default(),
        };

        let (_, summary) = build(&segmentation);

        assert!(summary.continuity_faired_masters > 0, "{summary:?}");
        assert_eq!(summary.shared_loop_fallbacks, 0, "{summary:?}");
        assert_eq!(summary.shared_loop_discontinuities, 0, "{summary:?}");
    }

    #[test]
    fn small_rectangular_insets_keep_observed_corners() {
        for (w, h, x0, y0) in [(6, 8, 8, 7), (8, 6, 9, 8), (10, 12, 5, 6)] {
            let width = 32;
            let height = 32;
            let colours = [[0.8; 3], [0.05; 3]];
            let labels: Vec<u32> = (0..width * height)
                .map(|i| {
                    let (x, y) = (i % width, i / width);
                    u32::from(x >= x0 && x < x0 + w && y >= y0 && y < y0 + h)
                })
                .collect();
            let regions = (0..2)
                .map(|label| {
                    let pixels: Vec<_> = labels
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| (v == label as u32).then_some(i))
                        .collect();
                    RegionStats {
                        id: label as u32,
                        area: pixels.len(),
                        min_x: pixels.iter().map(|i| i % width).min().unwrap(),
                        min_y: pixels.iter().map(|i| i / width).min().unwrap(),
                        max_x: pixels.iter().map(|i| i % width + 1).max().unwrap(),
                        max_y: pixels.iter().map(|i| i / width + 1).max().unwrap(),
                        mean_rgb: colours[label],
                        mean_lab: rgb_to_oklab(colours[label]),
                    }
                })
                .collect();
            let segmentation = Segmentation {
                width,
                height,
                canonical: Raster::new(
                    width,
                    height,
                    labels.iter().map(|&v| colours[v as usize]).collect(),
                ),
                labels,
                regions,
                paint_keys: vec![0, 1],
                paint_samples: vec![true; width * height],
                summary: SegmentationSummary::default(),
            };
            let (geometry, summary) = build(&segmentation);
            let inset = geometry.iter().find(|g| g.region == 1).unwrap();
            for (x, y) in [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)] {
                let corner = Point {
                    x: x as f32,
                    y: y as f32,
                };
                let distance = inset
                    .loops
                    .iter()
                    .flatten()
                    .map(|p| p.distance(corner))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    distance < 0.35,
                    "lost observed corner {corner:?}: {distance}; {}",
                    inset.path_data
                );
            }
            assert_eq!(summary.shared_loop_discontinuities, 0);
        }
    }

    #[test]
    fn hidden_edge_expansion_preserves_visible_edges_and_junctions() {
        let points = [
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
        ];
        let original: Vec<_> = (0..4)
            .map(|i| CurveSegment::Line {
                start: points[i],
                end: points[(i + 1) % 4],
            })
            .collect();
        let mut expanded = original.clone();
        expand_hidden_edges(&mut expanded, &[false, true, false, false], 0.3);
        for i in [0, 2, 3] {
            assert_eq!(expanded[i], original[i]);
        }
        assert_eq!(expanded[1].start(), points[1]);
        assert_eq!(expanded[1].end(), points[2]);
        let CurveSegment::Cubic { first, second, .. } = expanded[1] else {
            panic!("hidden edge did not gain overlap");
        };
        assert!((first.x - 10.3).abs() < 1e-5 && (second.x - 10.3).abs() < 1e-5);
        let mut unchanged = original.clone();
        expand_hidden_edges(&mut unchanged, &[false; 4], 0.3);
        assert_eq!(unchanged, original);
    }

    #[test]
    fn rectangular_region_remains_shared_path_before_final_optimization() {
        let segmentation = Segmentation {
            width: 8,
            height: 6,
            labels: vec![0; 48],
            paint_keys: vec![0],
            paint_samples: vec![true; 48],
            canonical: Raster::blank(8, 6, [0.0; 3]),
            regions: vec![RegionStats {
                id: 0,
                area: 48,
                min_x: 0,
                min_y: 0,
                max_x: 8,
                max_y: 6,
                mean_rgb: [0.0; 3],
                mean_lab: rgb_to_oklab([0.0; 3]),
            }],
            summary: SegmentationSummary {
                initial_regions: 1,
                merged_regions: 1,
                effective_minimum_area: 1,
                local_minimum_area: 1,
                local_median_area: 1,
                local_maximum_area: 1,
                ..SegmentationSummary::default()
            },
        };
        let (geometry, _) = build(&segmentation);
        assert!(geometry[0].primitive.is_none());
        assert!(!geometry[0].path_data.is_empty());
    }

    #[test]
    fn only_holes_owned_entirely_by_later_faces_are_removed() {
        let polygon = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 1.0, y: 4.0 },
            Point { x: 4.0, y: 4.0 },
            Point { x: 4.0, y: 1.0 },
        ];
        let mut labels = vec![0_u32; 25];
        for y in 1..4 {
            for x in 1..4 {
                labels[y * 5 + x] = 1;
            }
        }
        let segmentation = Segmentation {
            width: 5,
            height: 5,
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; 25],
            canonical: Raster::blank(5, 5, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: 16,
                    min_x: 0,
                    min_y: 0,
                    max_x: 5,
                    max_y: 5,
                    mean_rgb: [0.0; 3],
                    mean_lab: rgb_to_oklab([0.0; 3]),
                },
                RegionStats {
                    id: 1,
                    area: 9,
                    min_x: 1,
                    min_y: 1,
                    max_x: 4,
                    max_y: 4,
                    mean_rgb: [1.0; 3],
                    mean_lab: rgb_to_oklab([1.0; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        assert!(hole_is_covered_by_later_regions(
            &polygon,
            0,
            &segmentation,
            &[0, 1],
            &[]
        ));
        assert!(!hole_is_covered_by_later_regions(
            &polygon,
            0,
            &segmentation,
            &[1, 0],
            &[]
        ));
        let (geometry, summary) = build(&segmentation);
        let outer = geometry.iter().find(|item| item.region == 0).unwrap();
        assert_eq!(summary.covered_holes_removed, 1);
        assert!(outer.occlusion_path_data.is_some());
        assert!(outer.occlusion_path_data.as_ref().unwrap().len() < outer.path_data.len());
        let mut multiple = segmentation.clone();
        multiple.labels[2 * 5 + 2] = 2;
        assert!(hole_is_covered_by_later_regions(
            &polygon,
            0,
            &multiple,
            &[0, 1, 2],
            &[true; 3]
        ));
        assert!(!hole_is_covered_by_later_regions(
            &polygon,
            0,
            &multiple,
            &[0, 1, 2],
            &[true, true, false]
        ));
        assert!(!hole_is_covered_by_later_regions(
            &polygon,
            0,
            &multiple,
            &[1, 2, 0],
            &[true; 3]
        ));
    }

    #[test]
    fn closed_simplification_keeps_canvas_corner() {
        let mut raw = Vec::<Point>::new();
        for x in 0..=32 {
            raw.push(Point {
                x: x as f32,
                y: 0.0,
            });
        }
        for y in 1..=12 {
            raw.push(Point {
                x: 32.0,
                y: y as f32,
            });
        }
        raw.extend([
            Point { x: 20.0, y: 9.0 },
            Point { x: 8.0, y: 6.0 },
            Point { x: 0.0, y: 3.0 },
        ]);
        let corner = Point { x: 32.0, y: 0.0 };
        let simplified = preserve_closed_points(
            simplify_grid_closed(&raw, std::f32::consts::FRAC_1_SQRT_2),
            &raw,
            &[corner],
        );
        assert!(simplified.contains(&corner));
    }

    #[test]
    fn bounded_fairing_removes_raster_staircase_without_leaving_source_corridor() {
        let mut raw = vec![Point { x: 0.0, y: 0.0 }];
        let mut y = 0.0_f32;
        for x in 0..96 {
            raw.push(Point {
                x: (x + 1) as f32,
                y,
            });
            if (x + 1) % 4 == 0 {
                y += 1.0;
                raw.push(Point {
                    x: (x + 1) as f32,
                    y,
                });
            }
        }
        let baseline = simplify_open(&raw, 0.55);
        let fair = bounded_fairing_open(&raw, 0.55);
        assert!(
            fair.len() < baseline.len(),
            "fair={} baseline={}",
            fair.len(),
            baseline.len()
        );

        let source = resample_open_polyline(&raw, 0.25);
        let candidate = sample_open_catmull(&fair, 0.25);
        let maximum = std::f32::consts::SQRT_2;
        assert!(nearest_sample_distances(&source, &candidate, maximum).is_some());
        assert!(nearest_sample_distances(&candidate, &source, maximum).is_some());
    }

    #[test]
    fn bounded_fairing_preserves_a_persistent_right_angle() {
        let mut raw: Vec<Point> = (0..=32)
            .map(|x| Point {
                x: x as f32,
                y: 0.0,
            })
            .collect();
        raw.extend((1..=32).map(|y| Point {
            x: 32.0,
            y: y as f32,
        }));
        let fair = bounded_fairing_open(&raw, 0.55);
        let corner = Point { x: 32.0, y: 0.0 };
        let corner_error = sample_open_catmull(&fair, 0.1)
            .into_iter()
            .map(|point| point.distance(corner))
            .fold(f32::INFINITY, f32::min);
        assert!(corner_error <= 0.25, "corner error={corner_error}");
    }

    #[test]
    fn short_cornerless_boundary_can_reject_a_one_pixel_phase_reversal() {
        let mut raw = vec![Point { x: 746.0, y: 574.0 }];
        for (target_x, target_y) in [
            (750.0, 574.0),
            (750.0, 572.0),
            (755.0, 572.0),
            (755.0, 571.0),
            (760.0, 571.0),
            (760.0, 570.0),
            (763.0, 570.0),
            (763.0, 571.0),
            (767.0, 571.0),
            (767.0, 570.0),
            (773.0, 570.0),
            (773.0, 569.0),
            (778.0, 569.0),
            (778.0, 568.0),
            (784.0, 568.0),
            (784.0, 567.0),
            (790.0, 567.0),
            (790.0, 566.0),
            (793.0, 566.0),
        ] {
            while (raw.last().unwrap().x - target_x).abs() > 1e-6 {
                let previous = *raw.last().unwrap();
                raw.push(Point {
                    x: previous.x + (target_x - previous.x).signum(),
                    y: previous.y,
                });
            }
            while (raw.last().unwrap().y - target_y).abs() > 1e-6 {
                let previous = *raw.last().unwrap();
                raw.push(Point {
                    x: previous.x,
                    y: previous.y + (target_y - previous.y).signum(),
                });
            }
        }
        let mut master_id = 0;
        let (_, tagged) =
            potrace_master_curves(&raw, false, 0.5, 1.2, &mut master_id, &HashSet::new());
        let baseline: Vec<CurveSegment> = tagged.iter().map(|value| value.3).collect();
        let catmull = bounded_fairing_shared_boundary(&raw, &baseline, 0.75);
        let least = least_squares_fairing_shared_boundary(&raw, &baseline);
        let fair = bounded_fairing_direct_shared_boundary(&raw, &baseline, false, false);

        assert!(persistent_open_corners(&raw).is_empty());
        assert!(
            fair.len() < baseline.len(),
            "fair={} baseline={} catmull={} least={}",
            fair.len(),
            baseline.len(),
            catmull.len(),
            least.len()
        );
        assert!(raster_boundary_supported(
            &raw,
            &sample_curve_sequence(&fair, 0.25),
            std::f32::consts::SQRT_2 + 0.25,
        ));
    }

    #[test]
    fn adjacent_faces_reuse_one_exact_reversed_boundary() {
        let width = 6;
        let height = 4;
        let labels: Vec<u32> = (0..height)
            .flat_map(|_| (0..width).map(|x| u32::from(x >= 3)))
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: 12,
                    min_x: 0,
                    min_y: 0,
                    max_x: 3,
                    max_y: 4,
                    mean_rgb: [0.5; 3],
                    mean_lab: rgb_to_oklab([0.5; 3]),
                },
                RegionStats {
                    id: 1,
                    area: 12,
                    min_x: 3,
                    min_y: 0,
                    max_x: 6,
                    max_y: 4,
                    mean_rgb: [0.5; 3],
                    mean_lab: rgb_to_oklab([0.5; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (directed_edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pair_edges = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, ..) = build_shared_chains(
            &segmentation,
            None,
            &[],
            stride,
            &directed_edges,
            &pair_edges,
        );
        let chain_ids: Vec<usize> = (0..height)
            .map(|y| lookup[&EdgeKey::new(vertex_id(3, y, stride), vertex_id(3, y + 1, stride))].0)
            .collect();
        assert!(chain_ids.iter().all(|&value| value == chain_ids[0]));
        let forward = oriented_segments(&chains[chain_ids[0]], true);
        let restored: Vec<CurveSegment> = oriented_segments(&chains[chain_ids[0]], false)
            .into_iter()
            .rev()
            .map(CurveSegment::reversed)
            .collect();
        assert_eq!(forward, restored);
    }

    #[test]
    fn diagonal_grid_staircase_is_one_continuous_line() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 1.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 3.0, y: 2.0 },
            Point { x: 3.0, y: 3.0 },
        ];
        assert_eq!(
            simplify_grid_open(&points, std::f32::consts::FRAC_1_SQRT_2),
            vec![points[0], points[points.len() - 1]]
        );
        // The production shared-boundary path uses the Potrace dynamic
        // program, whose midpoint observations must make the same
        // minimum-description choice without preserving turn vertices.
        assert_eq!(potrace_optimal_polygon(&points, 0.5), vec![0, 6]);
    }

    #[test]
    fn material_transition_slices_one_continuous_master_boundary() {
        let width = 7;
        let height = 7;
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| if x < 3 { 0 } else { 1 + u32::from(y >= 3) }))
            .collect();
        let areas = [3 * height, 4 * 3, 4 * 4];
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1, 2],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: (0..3)
                .map(|id| RegionStats {
                    id: id as u32,
                    area: areas[id],
                    min_x: if id == 0 { 0 } else { 3 },
                    min_y: if id == 2 { 3 } else { 0 },
                    max_x: if id == 0 { 3 } else { width },
                    max_y: if id == 1 { 3 } else { height },
                    mean_rgb: [id as f32 * 0.3; 3],
                    mean_lab: rgb_to_oklab([id as f32 * 0.3; 3]),
                })
                .collect(),
            summary: SegmentationSummary::default(),
        };
        let stride = width + 1;
        let (directed_edges, _) = region_boundary_edges(&segmentation, stride, None);
        let pair_edges = pair_boundary_edges(&segmentation, stride, None);
        let (chains, lookup, ..) = build_shared_chains(
            &segmentation,
            None,
            &[],
            stride,
            &directed_edges,
            &pair_edges,
        );
        let upper = lookup[&EdgeKey::new(vertex_id(3, 2, stride), vertex_id(3, 3, stride))].0;
        let lower = lookup[&EdgeKey::new(vertex_id(3, 3, stride), vertex_id(3, 4, stride))].0;
        assert_ne!(upper, lower);
        let endpoints = |chain: &SharedChain| {
            [
                chain.segments.first().unwrap().start(),
                chain.segments.last().unwrap().end(),
            ]
        };
        let junction = endpoints(&chains[upper])
            .into_iter()
            .find(|first| {
                endpoints(&chains[lower])
                    .into_iter()
                    .any(|second| first.distance(second) <= 1e-5)
            })
            .expect("material runs share one adjusted topology vertex");
        let tangent_away = |chain: &SharedChain| {
            chain.segments.iter().find_map(|segment| match *segment {
                CurveSegment::Line { start, end } if start.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: end.x - start.x,
                        y: end.y - start.y,
                    })
                }
                CurveSegment::Line { start, end } if end.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: start.x - end.x,
                        y: start.y - end.y,
                    })
                }
                CurveSegment::Cubic { start, first, .. } if start.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: first.x - start.x,
                        y: first.y - start.y,
                    })
                }
                CurveSegment::Cubic { second, end, .. } if end.distance(junction) <= 1e-5 => {
                    Some(Point {
                        x: second.x - end.x,
                        y: second.y - end.y,
                    })
                }
                _ => None,
            })
        };
        let first = tangent_away(&chains[upper]).expect("upper chain reaches junction");
        let second = tangent_away(&chains[lower]).expect("lower chain reaches junction");
        let cross = first.x * second.y - first.y * second.x;
        let dot = first.x * second.x + first.y * second.y;
        assert!(cross.abs() <= 1e-5);
        assert!(dot < 0.0);
        let (_, summary) = build(&segmentation);
        assert_eq!(summary.shared_loop_fallbacks, 0, "{summary:?}");
    }

    #[test]
    fn nearby_colours_keep_a_closed_paint_contour_across_dotted_ink() {
        let (width, height) = (90, 70);
        let colours = [[0.96, 0.80, 0.14], [1.0, 0.92, 0.0], [0.35, 0.28, 0.02]];
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = (x as f32 + 0.5 - 45.0) / 32.0;
                    let dy = (y as f32 + 0.5 - 35.0) / 23.0;
                    let radius = dx.hypot(dy);
                    if radius > 1.0 {
                        0
                    } else if radius > 0.97 && (dy.atan2(dx) * 24.0).sin() > 0.0 {
                        2
                    } else {
                        1
                    }
                })
            })
            .collect();
        let canonical = Raster::new(
            width,
            height,
            labels
                .iter()
                .map(|&label| colours[label as usize])
                .collect(),
        );
        let regions = (0..3)
            .map(|id| RegionStats {
                id,
                area: labels.iter().filter(|&&label| label == id).count(),
                min_x: 0,
                min_y: 0,
                max_x: width,
                max_y: height,
                mean_rgb: colours[id as usize],
                mean_lab: rgb_to_oklab(colours[id as usize]),
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels,
            canonical,
            regions,
            paint_keys: (0..3).collect(),
            paint_samples: vec![true; width * height],
            summary: SegmentationSummary::default(),
        };
        let (_, summary) = build(&segmentation);
        assert!(
            summary.paint_ellipse_contours.iter().any(|contour| {
                let bounds = [
                    contour.iter().map(|p| p.x).fold(f32::INFINITY, f32::min),
                    contour.iter().map(|p| p.y).fold(f32::INFINITY, f32::min),
                    contour
                        .iter()
                        .map(|p| p.x)
                        .fold(f32::NEG_INFINITY, f32::max),
                    contour
                        .iter()
                        .map(|p| p.y)
                        .fold(f32::NEG_INFINITY, f32::max),
                ];
                (bounds[0] - 13.0).abs() < 1.5
                    && (bounds[1] - 12.0).abs() < 1.5
                    && (bounds[2] - 77.0).abs() < 1.5
                    && (bounds[3] - 58.0).abs() < 1.5
            }),
            "a tonal class must not erase the oval: {summary:?}"
        );
        assert_eq!(summary.shared_loop_fallbacks, 0);
    }
    #[test]
    fn invisible_same_rgb_faces_do_not_erase_the_silhouette_master() {
        let (width, height) = (80, 80);
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let inside =
                        (x as f32 - 39.5).powi(2) + (y as f32 - 39.5).powi(2) < 30.0_f32.powi(2);
                    if inside {
                        1 + ((y / 4) % 4) as u32
                    } else {
                        0
                    }
                })
            })
            .collect();
        // Coverage splitting changes opacity but intentionally retains RGB.
        let canonical = Raster::blank(width, height, [0.05, 0.0, 0.43]);
        let regions = (0..5)
            .map(|id| RegionStats {
                id,
                area: labels.iter().filter(|&&label| label == id).count(),
                min_x: 0,
                min_y: 0,
                max_x: width,
                max_y: height,
                mean_rgb: [0.05, 0.0, 0.43],
                mean_lab: rgb_to_oklab([0.05, 0.0, 0.43]),
            })
            .collect();
        let segmentation = Segmentation {
            width,
            height,
            labels,
            canonical,
            regions,
            paint_keys: (0..5).collect(),
            paint_samples: vec![true; width * height],
            summary: SegmentationSummary::default(),
        };
        let (_, before) = build_internal(&segmentation, None, None, None, 0.0, &[], None, &[]);
        let (_, after) = build_internal(
            &segmentation,
            None,
            None,
            None,
            0.0,
            &[],
            None,
            &[true, false, false, false, false],
        );
        assert!(
            after.continuity_faired_masters > before.continuity_faired_masters,
            "the exterior must retain a shared silhouette: {before:?} / {after:?}"
        );
        assert_eq!(after.shared_loop_fallbacks, 0);
    }
    #[test]
    fn raster_supported_disc_remains_shared_path_before_final_optimization() {
        let width = 31;
        let height = 31;
        let centre = (15.5_f32, 15.5_f32);
        let radius = 9.5_f32;
        let labels: Vec<u32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = x as f32 + 0.5 - centre.0;
                    let dy = y as f32 + 0.5 - centre.1;
                    u32::from(dx * dx + dy * dy <= radius * radius)
                })
            })
            .collect();
        let area = labels.iter().filter(|&&label| label == 1).count();
        let segmentation = Segmentation {
            width,
            height,
            labels: labels.clone(),
            paint_keys: vec![0, 1],
            paint_samples: vec![true; width * height],
            canonical: Raster::blank(width, height, [0.0; 3]),
            regions: vec![
                RegionStats {
                    id: 0,
                    area: width * height - area,
                    min_x: 0,
                    min_y: 0,
                    max_x: width,
                    max_y: height,
                    mean_rgb: [1.0; 3],
                    mean_lab: rgb_to_oklab([1.0; 3]),
                },
                RegionStats {
                    id: 1,
                    area,
                    min_x: 6,
                    min_y: 6,
                    max_x: 25,
                    max_y: 25,
                    mean_rgb: [0.0; 3],
                    mean_lab: rgb_to_oklab([0.0; 3]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let (geometry, _) = build(&segmentation);
        assert!(geometry[1].primitive.is_none());
        assert!(!geometry[1].path_data.is_empty());
    }

    #[test]
    fn every_binary_three_by_three_partition_has_closed_faces() {
        let width = 3;
        let height = 3;
        for bits in 1_u16..(1_u16 << 9) - 1 {
            let labels: Vec<u32> = (0..9)
                .map(|index| u32::from(bits & (1 << index) != 0))
                .collect();
            let first_area = labels.iter().filter(|&&label| label == 0).count();
            let second_area = labels.len() - first_area;
            let segmentation = Segmentation {
                width,
                height,
                labels: labels.clone(),
                paint_keys: vec![0, 1],
                paint_samples: vec![true; width * height],
                canonical: Raster::blank(width, height, [0.0; 3]),
                regions: [first_area, second_area]
                    .into_iter()
                    .enumerate()
                    .map(|(id, area)| RegionStats {
                        id: id as u32,
                        area,
                        min_x: 0,
                        min_y: 0,
                        max_x: width,
                        max_y: height,
                        mean_rgb: [id as f32; 3],
                        mean_lab: rgb_to_oklab([id as f32; 3]),
                    })
                    .collect(),
                summary: SegmentationSummary::default(),
            };
            let (geometry, _) = build(&segmentation);
            assert!(
                geometry.iter().all(|face| !face.loops.is_empty()),
                "partition {bits:09b} lost faces {:?}",
                geometry
                    .iter()
                    .filter(|face| face.loops.is_empty())
                    .map(|face| face.region)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn every_ternary_three_by_three_partition_uses_shared_loops() {
        let width = 3;
        let height = 3;
        for mut code in 0_u32..3_u32.pow(9) {
            let mut labels = Vec::with_capacity(9);
            for _ in 0..9 {
                labels.push(code % 3);
                code /= 3;
            }
            let mut remap = [u32::MAX; 3];
            let mut next = 0_u32;
            for label in &mut labels {
                if remap[*label as usize] == u32::MAX {
                    remap[*label as usize] = next;
                    next += 1;
                }
                *label = remap[*label as usize];
            }
            let all_connected = (0..next).all(|id| {
                let Some(start) = labels.iter().position(|&label| label == id) else {
                    return false;
                };
                let mut seen = HashSet::from([start]);
                let mut queue = vec![start];
                while let Some(index) = queue.pop() {
                    let x = index % width;
                    let y = index / width;
                    for neighbour in [
                        (x > 0).then(|| index - 1),
                        (x + 1 < width).then(|| index + 1),
                        (y > 0).then(|| index - width),
                        (y + 1 < height).then(|| index + width),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if labels[neighbour] == id && seen.insert(neighbour) {
                            queue.push(neighbour);
                        }
                    }
                }
                seen.len() == labels.iter().filter(|&&label| label == id).count()
            });
            if !all_connected {
                continue;
            }
            let regions = (0..next)
                .map(|id| {
                    let area = labels.iter().filter(|&&label| label == id).count();
                    RegionStats {
                        id,
                        area,
                        min_x: 0,
                        min_y: 0,
                        max_x: width,
                        max_y: height,
                        mean_rgb: [id as f32 * 0.3; 3],
                        mean_lab: rgb_to_oklab([id as f32 * 0.3; 3]),
                    }
                })
                .collect();
            let segmentation = Segmentation {
                width,
                height,
                labels: labels.clone(),
                paint_keys: (0..next).collect(),
                paint_samples: vec![true; width * height],
                canonical: Raster::blank(width, height, [0.0; 3]),
                regions,
                summary: SegmentationSummary::default(),
            };
            let (_, summary) = build(&segmentation);
            assert_eq!(
                summary.shared_loop_fallbacks, 0,
                "labels={labels:?}, summary={summary:?}"
            );
        }
    }
}

impl ClosedContour {
    pub fn from_points(points: &[Point]) -> Self {
        let curves = geometry_bezier::fit_closed(points, geometry_bezier::CLOSED_CORRIDOR).unwrap();
        Self {
            points: sample_curve_sequence(&curves, 0.75),
            curves,
            is_ellipse: false,
            fallback: None,
        }
    }
}
