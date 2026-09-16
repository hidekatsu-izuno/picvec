//! Source-supported ordering of filled stroke shapes. This preserves fills:
//! it neither generates duplicate strokes nor infers semantic object depth.
use crate::svg_document::Document;
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

#[cfg(feature = "diagnostics")]
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
    #[cfg(any(test, feature = "diagnostics"))]
    pub lines: Vec<bool>,
    #[cfg(feature = "diagnostics")]
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
    #[cfg(feature = "diagnostics")]
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
        #[cfg(feature = "diagnostics")]
        evidence,
        #[cfg(any(test, feature = "diagnostics"))]
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

/// Keep unchanged draw operations available to subsequent covered-hole trials.
/// Cache keys include geometry, paint definitions and inherited SVG context.
pub(crate) fn validate_cached(
    before: &Document,
    after: &Document,
    source: &Raster,
    matte: Option<&crate::chroma::AlphaMatte>,
    labels: &[u32],
    summary: &mut Summary,
    fragments: &mut Option<crate::svg_fragments::Cache>,
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
    // Complex geometric clips are expensive to set up in every empty band.
    // Reuse the existing fragment renderer only when there are multiple bands
    // and clips to cull; ordinary small/unclipped documents keep the direct
    // renderer. Unsupported SVG contexts also keep the direct renderer.
    *fragments = None;
    let scenes = (h > 128
        && (before.root().contains_attribute("clip-path")
            || after.root().contains_attribute("clip-path")))
    .then(|| {
        let cache = crate::svg_fragments::Cache::new(before)?;
        let scenes = (cache.scene(before)?, cache.scene(after)?);
        *fragments = Some(cache);
        Some(scenes)
    })
    .flatten();

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
    use rayon::prelude::*;
    let rows: Vec<_> = (0..h).step_by(64).collect();
    let batch_size = rayon::current_num_threads().clamp(1, 4);
    let mut increase = 0.0f64;
    for scale in [1usize, 4] {
        for batch in rows.chunks(batch_size) {
            // Bands are independent. Preserve the old row/column reduction
            // and first rejection, with at most four pairs of live pixmaps.
            let evaluated: Vec<Result<Vec<f64>, String>> = batch
                .par_iter()
                .map(|&y| {
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
                        0.0,
                        -((y * scale) as f32),
                    );
                    if let Some((a, b)) = &scenes {
                        a.render(scale, y, &mut pa);
                        b.render(scale, y, &mut pb);
                    } else {
                        resvg::render(&a, transform, &mut pa.as_mut());
                        resvg::render(&b, transform, &mut pb.as_mut());
                    }
                    let mut tile_increase = vec![0.0_f64; w.div_ceil(64)];
                    for (k, (p, q)) in pa.pixels().iter().zip(pb.pixels()).enumerate() {
                        if p == q {
                            continue;
                        }
                        let i = (y + k / (tw * scale) / scale) * w + x + k % (tw * scale) / scale;
                        let pv = [p.red(), p.green(), p.blue(), p.alpha()];
                        let qv = [q.red(), q.green(), q.blue(), q.alpha()];
                        if scale == 4 {
                            // Boundary alignment may change RGB, but must not
                            // punch a hole through previously opaque coverage.
                            // A tiny gap can disappear in native-size averages.
                            if pv[3] >= 250 && qv[3] < 128 && matte.is_none_or(|m| m.get(i) >= 1.0)
                            {
                                return Err("opaque coverage lost at 4x".to_owned());
                            }
                            if !boundary[i] && pv.iter().zip(qv).any(|(&p, q)| p.abs_diff(q) > 2) {
                                return Err("interior changed at 4x".to_owned());
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
                            return Err(format!(
                                "local source error at {},{}: {:.2} codes",
                                i % w,
                                i / w,
                                loss * 255.0
                            ));
                        }
                        tile_increase[(i % w) / 64] += loss;
                    }
                    Ok(tile_increase)
                })
                .collect();
            for (&y, result) in batch.iter().zip(evaluated) {
                let losses = match result {
                    Ok(losses) => losses,
                    Err(reason) => {
                        summary.rejection = Some(reason);
                        return false;
                    }
                };
                if scale == 1 {
                    let th = (h - y).min(64);
                    for (column, loss) in losses.into_iter().enumerate() {
                        increase += loss;
                        let area = (w - column * 64).min(64) * th;
                        if loss / area as f64 > 2.0 / 255.0 {
                            summary.rejection = Some("tile source error increased".into());
                            return false;
                        }
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
include!("../tests/unit/paint_order.rs");
