//! Source-supported ordering of filled stroke shapes. This preserves fills:
//! it neither generates duplicate strokes nor infers semantic object depth.
use crate::{
    color::{delta_e_ok, rgb_to_oklab},
    raster::Raster,
    segment::Segmentation,
};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub line_regions: usize,
    pub proposed_constraints: usize,
    pub conflicting_constraints: usize,
    pub changed_ranks: usize,
    pub accepted: bool,
    pub source_error_increase: f32,
    pub rejection: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct Evidence {
    area: usize,
    perimeter: usize,
    medial_points: usize,
    supported_points: usize,
    width_low: f32,
    width_median: f32,
    width_high: f32,
    source_line_fraction: f32,
    score: f32,
}

pub(crate) struct Proposal {
    pub ranks: Vec<usize>,
    pub lines: Vec<bool>,
    pub evidence: Vec<Evidence>,
    pub summary: Summary,
}

pub(crate) fn propose(
    source: &Raster,
    seg: &Segmentation,
    source_lines: &[bool],
    opaque: &[bool],
    baseline: &[usize],
) -> Proposal {
    let (w, h, n) = (seg.width, seg.height, seg.regions.len());
    let labels = &seg.labels;
    // Chamfer distance to any other owner; source-line evidence is retained
    // independently so a broad branched outline is not rejected by its area.
    let mut distance = vec![u32::MAX / 4; labels.len()];
    let mut perimeter = vec![0usize; n];
    let mut line_pixels = vec![0usize; n];
    let mut contacts = BTreeSet::new();
    for i in 0..labels.len() {
        let (x, y, id) = (i % w, i / w, labels[i] as usize);
        line_pixels[id] += usize::from(source_lines.get(i).copied().unwrap_or(false));
        for j in [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ] {
            if j.is_none_or(|j| labels[j] != labels[i]) {
                perimeter[id] += 1;
                distance[i] = 1;
                if let Some(j) = j {
                    contacts.insert((id.min(labels[j] as usize), id.max(labels[j] as usize)));
                }
            }
        }
    }
    for i in 0..labels.len() {
        for j in [(i % w > 0).then(|| i - 1), (i >= w).then(|| i - w)]
            .into_iter()
            .flatten()
        {
            if labels[j] == labels[i] {
                distance[i] = distance[i].min(distance[j] + 1);
            }
        }
    }
    for i in (0..labels.len()).rev() {
        for j in [
            (i % w + 1 < w).then(|| i + 1),
            (i + w < labels.len()).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        {
            if labels[j] == labels[i] {
                distance[i] = distance[i].min(distance[j] + 1);
            }
        }
    }
    let mut widths = vec![Vec::new(); n];
    let mut supported = vec![0usize; n];
    let mut medials = vec![0usize; n];
    let mut pairs = vec![BTreeSet::new(); n];
    let labs: Vec<_> = source.pixels.iter().copied().map(rgb_to_oklab).collect();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let id = labels[i] as usize;
            if !opaque.get(id).copied().unwrap_or(false) || distance[i] < 2 {
                continue;
            }
            if [i - 1, i + 1, i - w, i + w]
                .iter()
                .any(|&j| distance[j] > distance[i] && labels[j] == labels[i])
            {
                continue;
            }
            medials[id] += 1;
            let limit = (distance[i] as usize * 3 + 3).min(w.max(h));
            let mut best = None;
            for (dx, dy) in [(1isize, 0isize), (0, 1)] {
                let mut ends = Vec::new();
                for sign in [-1isize, 1] {
                    for step in 1..=limit {
                        let xx = x as isize + dx * sign * step as isize;
                        let yy = y as isize + dy * sign * step as isize;
                        if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                            break;
                        }
                        let j = yy as usize * w + xx as usize;
                        if labels[j] != labels[i] {
                            // Step one pixel into the adjacent material to
                            // avoid treating its AA mixture as the paint colour.
                            let kx = xx + dx * sign;
                            let ky = yy + dy * sign;
                            let k = if kx >= 0 && ky >= 0 && kx < w as isize && ky < h as isize {
                                let k = ky as usize * w + kx as usize;
                                if labels[k] == labels[j] {
                                    k
                                } else {
                                    j
                                }
                            } else {
                                j
                            };
                            ends.push((step, j, k));
                            break;
                        }
                    }
                }
                if ends.len() == 2 {
                    let span = ends[0].0 + ends[1].0;
                    if best.as_ref().is_none_or(|&(s, _, _)| span < s) {
                        best = Some((span, ends[0], ends[1]));
                    }
                }
            }
            let Some((span, (_, a, ac), (_, b, bc))) = best else {
                continue;
            };
            if !opaque[labels[a] as usize] || !opaque[labels[b] as usize] {
                continue;
            }
            let c = labs[i];
            let ca = labs[ac];
            let cb = labs[bc];
            let da = delta_e_ok(c, ca);
            let db = delta_e_ok(c, cb);
            let dot = (ca.l - c.l) * (cb.l - c.l)
                + (ca.a - c.a) * (cb.a - c.a)
                + (ca.b - c.b) * (cb.b - c.b);
            if da >= 6.0 && db >= 6.0 && dot >= 0.25 * da * db {
                supported[id] += 1;
                widths[id].push(span as f32);
                pairs[id].insert(labels[a] as usize);
                pairs[id].insert(labels[b] as usize);
            }
        }
    }
    let mut scores = vec![0.0f32; n];
    let mut lines = vec![false; n];
    for id in 0..n {
        let area = seg.regions[id].area as f32;
        let p = perimeter[id] as f32;
        let ws = &mut widths[id];
        ws.sort_by(f32::total_cmp);
        if ws.len() < 6 || p * p < 48.0 * area {
            continue;
        }
        let lo = ws[(ws.len() - 1) / 10];
        let hi = ws[(ws.len() - 1) * 9 / 10];
        let median = ws[ws.len() / 2];
        let support = supported[id] as f32 / medials[id].max(1) as f32;
        let legacy = line_pixels[id] as f32 / area.max(1.0);
        // Branches may widen a line, but an arbitrarily broad flat field
        // cannot become a line merely because it has a high-contrast border.
        if area < median * median * 20.0
            || hi > lo * 4.0
            || (supported[id] as f32) < median * 2.0
            || support < 0.25
            || (legacy < 0.02 && p * p < 80.0 * area)
        {
            continue;
        }
        lines[id] = true;
        scores[id] = support + legacy.min(1.0) + (supported[id] as f32).ln_1p() * 0.05;
    }
    let mut fixed = Vec::new();
    // Authored translucent faces retain their relative compositing order.
    for id in 0..n {
        if opaque.get(id).copied().unwrap_or(false) {
            continue;
        }
        for other in 0..n {
            if id == other {
                continue;
            }
            fixed.push(if baseline[id] < baseline[other] {
                (id, other)
            } else {
                (other, id)
            });
        }
    }
    let mut constraints = Vec::new();
    for (a, b) in contacts {
        if !opaque[a] || !opaque[b] || lines[a] == lines[b] {
            continue;
        }
        let (paint, line) = if lines[a] { (b, a) } else { (a, b) };
        // Require opposite-side source evidence for this particular neighbour.
        if pairs[line].contains(&paint) {
            constraints.push((paint, line, scores[line]));
        }
    }
    let proposed_constraints = constraints.len();
    let (ranks, conflicts) = order_graph(baseline, &fixed, constraints);
    let changed_ranks = ranks.iter().zip(baseline).filter(|(a, b)| a != b).count();
    let evidence = (0..n)
        .map(|id| {
            let ws = &widths[id];
            Evidence {
                area: seg.regions[id].area,
                perimeter: perimeter[id],
                medial_points: medials[id],
                supported_points: supported[id],
                width_low: if ws.is_empty() {
                    0.0
                } else {
                    ws[(ws.len() - 1) / 10]
                },
                width_median: if ws.is_empty() { 0.0 } else { ws[ws.len() / 2] },
                width_high: if ws.is_empty() {
                    0.0
                } else {
                    ws[(ws.len() - 1) * 9 / 10]
                },
                source_line_fraction: line_pixels[id] as f32 / seg.regions[id].area.max(1) as f32,
                score: scores[id],
            }
        })
        .collect();
    Proposal {
        ranks,
        evidence,
        lines: lines.clone(),
        summary: Summary {
            line_regions: lines.iter().filter(|&&v| v).count(),
            proposed_constraints,
            conflicting_constraints: conflicts,
            changed_ranks,
            ..Summary::default()
        },
    }
}

/// Edges run from lower to upper. Fixed alpha constraints precede proposals;
/// weaker proposals that close a cycle are discarded deterministically.
fn order_graph(
    baseline: &[usize],
    fixed: &[(usize, usize)],
    mut candidates: Vec<(usize, usize, f32)>,
) -> (Vec<usize>, usize) {
    let n = baseline.len();
    let mut edges = vec![BTreeSet::new(); n];
    for &(a, b) in fixed {
        edges[a].insert(b);
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let mut conflicts = 0;
    for (a, b, _) in candidates {
        let mut stack = vec![b];
        let mut seen = vec![false; n];
        let mut cycle = false;
        while let Some(v) = stack.pop() {
            if v == a {
                cycle = true;
                break;
            }
            if seen[v] {
                continue;
            }
            seen[v] = true;
            stack.extend(edges[v].iter().copied());
        }
        if cycle {
            conflicts += 1;
        } else {
            edges[a].insert(b);
        }
    }
    let mut degree = vec![0; n];
    for es in &edges {
        for &v in es {
            degree[v] += 1;
        }
    }
    let mut ready: BTreeSet<_> = (0..n)
        .filter(|&i| degree[i] == 0)
        .map(|i| (baseline[i], i))
        .collect();
    let mut ranks = vec![0; n];
    let mut rank = 0;
    while let Some((_, i)) = ready.pop_first() {
        ranks[i] = rank;
        rank += 1;
        for &j in &edges[i] {
            degree[j] -= 1;
            if degree[j] == 0 {
                ready.insert((baseline[j], j));
            }
        }
    }
    assert_eq!(rank, n, "fixed paint order must be acyclic");
    (ranks, conflicts)
}

/// Raster validation is independent of line classification. Both renderings
/// are compared in premultiplied RGBA, on black and white simultaneously.
/// At 4x, changes outside a two-source-pixel boundary corridor are forbidden.
pub(crate) fn validate(
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

#[cfg(test)]
mod tests {
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
            &svg(15.3),
            &svg(15.0),
            &source,
            None,
            &labels,
            &mut Summary::default()
        ));
        let erased = svg(15.0).replace("#000", "#fff");
        assert!(!validate(
            &svg(15.0),
            &erased,
            &source,
            None,
            &labels,
            &mut Summary::default()
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
            before,
            &after,
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
            a,
            &b,
            &Raster::blank(16, 16, [0.0; 3]),
            None,
            &vec![0; 256],
            &mut Summary::default()
        ));
    }
}
