/// Raster validation is independent of line classification. Both renderings
/// are compared in premultiplied RGBA, on black and white simultaneously.
/// At 4x, changes outside a two-source-pixel boundary corridor are forbidden.
pub(crate) fn validate(
    before: &Document,
    after: &Document,
    source: &Raster,
    matte: Option<&crate::chroma::AlphaMatte>,
    labels: &[u32],
    summary: &mut Summary,
) -> bool {
    validate_cached(before, after, source, matte, labels, summary, &mut None)
}

mod tests {

    #[test]
    fn parallel_bands_match_original_validation_and_rejection_order() {
        let source = Raster::blank(137, 151, [0.5; 3]);
        let labels: Vec<u32> = (0..137 * 151).map(|i| (i % 137 >= 65) as u32).collect();
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="137" height="151"><rect width="137" height="151" fill="#808080"/><path d="M64.8 0L65.2 0L65.2 151L64.8 151Z" fill="#111" fill-opacity=".4"/></svg>"##;
        let changed = [
            before.to_owned(),
            before.replace("#808080", "#818181"),
            before.replace("#808080", "#eeeeee"),
            before.replace("65.2", "68.2"),
            before.replace(".4", ".2"),
        ];
        for after in changed {
            let mut expected = Summary::default();
            let result = validate_reference(before, &after, &source, None, &labels, &mut expected);
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut actual = Summary::default();
                let accepted = pool.install(|| {
                    validate(
                        &Document::from((before).to_string()),
                        &Document::from((&after).to_string()),
                        &source,
                        None,
                        &labels,
                        &mut actual,
                    )
                });
                assert_eq!(accepted, result);
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            }
        }
    }

    #[test]
    fn clipped_bands_match_full_renderer_with_translucent_and_thin_paint() {
        let source = Raster::blank(137, 193, [0.5; 3]);
        let labels: Vec<u32> = (0..137 * 193).map(|i| (i % 137 >= 65) as u32).collect();
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="137" height="193"><defs><clipPath id="c"><path d="M1 1H135V191H1Z"/></clipPath></defs><rect width="137" height="193" fill="#808080"/><g clip-path="url(#c)"><path d="M2 3H20V12H2Z M110 179H130V188H110Z" fill="#555" fill-opacity=".4"/><path d="M3 96H133" fill="none" stroke="#123" stroke-width=".3" stroke-opacity=".6"/></g></svg>"##;
        for after in [
            before.to_owned(),
            before.replace("#808080", "#818181"),
            before.replace("#808080", "#eeeeee"),
            before.replace(".4", ".2"),
            before.replace("M3 96H133", "M3 97H133"),
        ] {
            let mut expected = Summary::default();
            let result = validate_reference(before, &after, &source, None, &labels, &mut expected);
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut actual = Summary::default();
                let accepted = pool.install(|| {
                    validate(
                        &Document::from((before).to_string()),
                        &Document::from((&after).to_string()),
                        &source,
                        None,
                        &labels,
                        &mut actual,
                    )
                });
                assert_eq!(accepted, result);
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            }
        }
    }

    use super::*;
    #[test]
    fn graph_removes_weak_cycle_and_preserves_alpha_barrier() {
        let (r, c) = order_graph(&[0, 1, 2], &[], vec![(0, 1, 3.0), (1, 2, 2.0), (2, 0, 1.0)]);
        assert_eq!(r, vec![0, 1, 2]);
        assert_eq!(c, 1);
        let (r, c) = order_graph(&[0, 1, 2], &[(0, 1), (1, 2)], vec![(2, 0, 9.0)]);
        assert_eq!(r, vec![0, 1, 2]);
        assert_eq!(c, 1);
    }
    #[test]
    fn recognises_a_connected_frame_but_not_a_flat_dark_field() {
        use crate::segment::{RegionStats, SegmentationSummary};
        for (frame, opaque) in [(true, true), (false, true), (true, false)] {
            let w = 48;
            let mut labels = vec![0u32; w * w];
            for y in 4..44 {
                for x in 4..44 {
                    labels[y * w + x] = if frame && (8..40).contains(&x) && (8..40).contains(&y) {
                        2
                    } else {
                        1
                    };
                }
            }
            let colors = [[0.8; 3], [0.02; 3], [0.95; 3]];
            let source = Raster::new(w, w, labels.iter().map(|&i| colors[i as usize]).collect());
            let seg = Segmentation {
                width: w,
                height: w,
                regions: (0..3)
                    .map(|id| RegionStats {
                        id,
                        area: labels.iter().filter(|&&v| v == id).count(),
                        min_x: 0,
                        min_y: 0,
                        max_x: w,
                        max_y: w,
                        mean_rgb: colors[id as usize],
                        mean_lab: rgb_to_oklab(colors[id as usize]),
                    })
                    .collect(),
                labels,
                canonical: source.clone(),
                paint_keys: vec![0, 1, 2],
                paint_samples: vec![true; w * w],
                summary: SegmentationSummary::default(),
            };
            let proposal = propose(
                &source,
                &seg,
                &vec![false; w * w],
                &[true, opaque, true],
                &[0, 1, 2],
            );
            if frame && opaque {
                assert!(proposal.lines[1], "connected frame was missed");
                assert!(
                    proposal.ranks[1] > proposal.ranks[2],
                    "frame must cover its inner paint"
                );
            } else {
                assert!(
                    !proposal.lines[1],
                    "flat field/translucent frame must not be promoted"
                );
                assert_eq!(proposal.ranks, vec![0, 1, 2]);
            }
        }
    }

    #[test]
    fn validates_boundary_alignment_but_rejects_erasing_a_thin_line() {
        let svg = |x: f32| {
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#fff"/><rect x="{x}" width="2" height="32" fill="#000"/></svg>"##
            )
        };
        let labels: Vec<u32> = (0..1024)
            .map(|i| u32::from((15..17).contains(&(i % 32))))
            .collect();
        let source = Raster::new(
            32,
            32,
            labels
                .iter()
                .map(|&l| [if l == 1 { 0.0 } else { 1.0 }; 3])
                .collect(),
        );
        assert!(validate(
            &Document::from((&svg(15.3)).to_string()),
            &Document::from((&svg(15.0)).to_string()),
            &source,
            None,
            &labels,
            &mut Summary::default()
        ));
        let erased = svg(15.0).replace("#000", "#fff");
        assert!(!validate(
            &Document::from((&svg(15.0)).to_string()),
            &Document::from((&erased).to_string()),
            &source,
            None,
            &labels,
            &mut Summary::default()
        ));
    }

    #[test]
    fn rejects_subpixel_pinholes_at_opaque_paint_boundaries() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path fill="#808080" fill-rule="evenodd" d="M0 0H32V32H0Z"/></svg>"##;
        let after = before.replace("H0Z", "H0Z M16.25 16.25h0.25v0.25h-0.25Z");
        let labels: Vec<u32> = (0..1024).map(|i| u32::from(i % 32 >= 16)).collect();
        let source = Raster::blank(32, 32, [128.0 / 255.0; 3]);
        let mut summary = Summary::default();
        assert!(!validate(
            &Document::from(before.to_owned()),
            &Document::from(after.clone()),
            &source,
            None,
            &labels,
            &mut summary,
        ));
        assert_eq!(
            summary.rejection.as_deref(),
            Some("opaque coverage lost at 4x")
        );
        let mut coverage = vec![1.0; 1024];
        coverage[16 * 32 + 16] = 0.0;
        let matte = crate::chroma::AlphaMatte::new(32, 32, coverage);
        assert!(validate(
            &Document::from(before.to_owned()),
            &Document::from(after),
            &source,
            Some(&matte),
            &labels,
            &mut Summary::default(),
        ));
    }

    #[test]
    fn wide_render_bands_keep_the_original_local_error_budget() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="512" height="64"><rect width="512" height="64" fill="#808080"/><rect x="448" width="64" height="64" fill="#a0a0a0"/></svg>"##;
        let after = before
            .replace(r#"x="448""#, r#"x="0""#)
            .replace("#a0a0a0", "#909090");
        let mut report = Summary::default();
        assert!(!validate(
            &Document::from((before).to_string()),
            &Document::from((&after).to_string()),
            &Raster::blank(512, 64, [128.0 / 255.0; 3]),
            None,
            &vec![0; 512 * 64],
            &mut report
        ));
        assert_eq!(
            report.rejection.as_deref(),
            Some("tile source error increased")
        );
    }

    #[test]
    fn rejects_interior_overpaint_in_render_validation() {
        let a = r##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path fill="#000" d="M0 0H16V16H0Z"/></svg>"##;
        let b = a.replace("#000", "#fff");
        assert!(!validate(
            &Document::from((a).to_string()),
            &Document::from((&b).to_string()),
            &Raster::blank(16, 16, [0.0; 3]),
            None,
            &vec![0; 256],
            &mut Summary::default()
        ));
    }
}

fn validate_reference(
    before: &str,
    after: &str,
    source: &Raster,
    matte: Option<&crate::chroma::AlphaMatte>,
    labels: &[u32],
    summary: &mut Summary,
) -> bool {
    use resvg::{
        tiny_skia::{Pixmap, Transform},
        usvg::{Options, Tree},
    };
    let (Ok(a), Ok(b)) = (
        Tree::from_str(before, &Options::default()),
        Tree::from_str(after, &Options::default()),
    ) else {
        summary.rejection = Some("SVG parse failed".into());
        return false;
    };
    let (w, h) = (source.width, source.height);
    let mut boundary = vec![false; w * h];
    for i in 0..w * h {
        let (x, y) = (i % w, i / w);
        boundary[i] = [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        .any(|j| labels[i] != labels[j]);
    }
    let boundary = crate::edge::dilate_square(&boundary, w, h, 2);
    let mut increase = 0.0f64;
    for scale in [1usize, 4] {
        for y in (0..h).step_by(64) {
            // Render horizontal bands: traversing the complete SVG once
            // per 64-pixel column repeats expensive clip/gradient setup.
            // The same native and 4x pixels are still checked below.
            let x = 0;
            let tw = w - x;
            let th = (h - y).min(64);
            let mut pa = Pixmap::new((tw * scale) as u32, (th * scale) as u32).unwrap();
            let mut pb = pa.clone();
            let transform = Transform::from_row(
                scale as f32,
                0.0,
                0.0,
                scale as f32,
                -((x * scale) as f32),
                -((y * scale) as f32),
            );
            resvg::render(&a, transform, &mut pa.as_mut());
            resvg::render(&b, transform, &mut pb.as_mut());
            let mut tile_increase = vec![0.0_f64; w.div_ceil(64)];
            for (k, (p, q)) in pa.pixels().iter().zip(pb.pixels()).enumerate() {
                if p == q {
                    continue;
                }
                let i = (y + k / (tw * scale) / scale) * w + x + k % (tw * scale) / scale;
                let pv = [p.red(), p.green(), p.blue(), p.alpha()];
                let qv = [q.red(), q.green(), q.blue(), q.alpha()];
                if scale == 4 {
                    if pv[3] >= 250 && qv[3] < 128 && matte.is_none_or(|m| m.get(i) >= 1.0) {
                        summary.rejection = Some("opaque coverage lost at 4x".into());
                        return false;
                    }
                    if !boundary[i] && pv.iter().zip(qv).any(|(&p, q)| p.abs_diff(q) > 2) {
                        summary.rejection = Some("interior changed at 4x".into());
                        return false;
                    }
                    continue;
                }
                let alpha = matte.map_or(1.0, |m| m.get(i)) as f64;
                let error = |v: [u8; 4]| {
                    let av = v[3] as f64 / 255.0;
                    let mut e = (av - alpha).abs();
                    for c in 0..3 {
                        let target = source.pixels[i][c] as f64 * alpha;
                        let actual = v[c] as f64 / 255.0;
                        e += (actual - target).abs()
                            + (actual + 1.0 - av - target - 1.0 + alpha).abs();
                    }
                    e / 7.0
                };
                let loss = error(qv) - error(pv);
                if !boundary[i] && loss > 32.0 / 255.0 {
                    summary.rejection = Some(format!(
                        "local source error at {},{}: {:.2} codes",
                        i % w,
                        i / w,
                        loss * 255.0
                    ));
                    return false;
                }
                tile_increase[(i % w) / 64] += loss;
            }
            if scale == 1 {
                for (column, loss) in tile_increase.into_iter().enumerate() {
                    increase += loss;
                    // Rendering uses a wide band, but the original 64x64
                    // local error budgets must not be diluted across it.
                    let area = (w - column * 64).min(64) * th;
                    if loss / area as f64 > 2.0 / 255.0 {
                        summary.rejection = Some("tile source error increased".into());
                        return false;
                    }
                }
            }
        }
    }
    summary.source_error_increase = (increase / (w * h).max(1) as f64) as f32;
    if summary.source_error_increase > 0.25 / 255.0 {
        summary.rejection = Some("source error increased".into());
        return false;
    }
    summary.accepted = true;
    true
}
