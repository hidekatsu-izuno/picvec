//! Source-resolution, rate-distortion driven refinement support.
//!
//! The base vectorizer remains the only image model.  This module identifies
//! source regions where rerunning that same model at a finer pyramid level is
//! likely to pay for its additional SVG representation cost, and composes the
//! accepted refinements back into the base document.

use std::collections::HashSet;

use rayon::prelude::*;
use serde::Serialize;

use crate::chroma::AlphaMatte;
use crate::color::{delta_e2000_pairs, rgb_to_lab, Lab};
use crate::raster::{percentile, Raster, RasterSource};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl SourceRect {
    pub fn area(self) -> usize {
        self.width * self.height
    }

    pub fn expanded(self, margin: usize, image_width: usize, image_height: usize) -> Self {
        let x = self.x.saturating_sub(margin);
        let y = self.y.saturating_sub(margin);
        let right = (self.x + self.width + margin).min(image_width);
        let bottom = (self.y + self.height + margin).min(image_height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PerceptualScore {
    pub mean_delta_e: f32,
    pub p90_delta_e: f32,
    pub missing_edge_fraction: f32,
    pub combined: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct RefinementCandidate {
    pub core: SourceRect,
    pub baseline: PerceptualScore,
    pub model_cost: f32,
    pub priority: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct EmbeddedRefinement {
    pub core: SourceRect,
    pub expanded: SourceRect,
    pub document: String,
    pub processing_width: usize,
    pub processing_height: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AdaptiveRefinementSummary {
    pub enabled: bool,
    pub source_scale: f32,
    pub proposed_regions: usize,
    pub candidate_regions: usize,
    pub prefiltered_for_complexity: usize,
    pub evaluated_regions: usize,
    pub parallel_jobs: usize,
    pub accepted_regions: usize,
    pub rejected_for_quality: usize,
    pub rejected_for_complexity: usize,
    pub baseline_mean_delta_e: f32,
    pub refined_mean_delta_e: f32,
    pub estimated_global_delta_e_reduction: f32,
    pub added_svg_bytes: usize,
}

/// Own each connected foreground object as a whole. Independent crop fits
/// cannot agree on a gradient or a stroke that a rectangular grid cuts in two.
/// Keep oversized objects in the base instead of creating such a discontinuity.
fn object_regions(
    support: &[bool],
    width: usize,
    height: usize,
    maximum_dimension: usize,
) -> Vec<SourceRect> {
    let mut pending = support.to_vec();
    let mut stack = Vec::new();
    let mut regions = Vec::new();
    for seed in 0..pending.len() {
        if !pending[seed] {
            continue;
        }
        pending[seed] = false;
        stack.push(seed);
        let (mut left, mut top, mut right, mut bottom) = (width, height, 0, 0);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % width, i / width);
            left = left.min(x);
            top = top.min(y);
            right = right.max(x + 1);
            bottom = bottom.max(y + 1);
            for py in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for px in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbour = py * width + px;
                    if pending[neighbour] {
                        pending[neighbour] = false;
                        stack.push(neighbour);
                    }
                }
            }
        }
        if right - left > maximum_dimension || bottom - top > maximum_dimension {
            continue;
        }
        regions.push(
            SourceRect {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top,
            }
            .expanded(16, width, height),
        );
    }
    // Disconnected details inside/near another silhouette must share its fit.
    // Recheck after every union: a union's rectangle can enclose a third object.
    let mut merged = Vec::<SourceRect>::new();
    for mut region in regions {
        let mut i = 0;
        while i < merged.len() {
            let other = merged[i];
            if region.x < other.x + other.width
                && other.x < region.x + region.width
                && region.y < other.y + other.height
                && other.y < region.y + region.height
            {
                let right = (region.x + region.width).max(other.x + other.width);
                let bottom = (region.y + region.height).max(other.y + other.height);
                region.x = region.x.min(other.x);
                region.y = region.y.min(other.y);
                region.width = right - region.x;
                region.height = bottom - region.y;
                merged.swap_remove(i);
                i = 0;
            } else {
                i += 1;
            }
        }
        merged.push(region);
    }
    merged.retain(|r| {
        if r.width < 64
            || r.height < 64
            || r.width > maximum_dimension
            || r.height > maximum_dimension
        {
            return false;
        }
        // In particular, a large background grid may surround many small
        // objects. Leave it in the base and never cut it with a replacement.
        let clear = |x: usize, y: usize| !support[y * width + x];
        (r.x..r.x + r.width).all(|x| {
            (r.y == 0 || clear(x, r.y))
                && (r.y + r.height == height || clear(x, r.y + r.height - 1))
        }) && (r.y..r.y + r.height).all(|y| {
            (r.x == 0 || clear(r.x, y)) && (r.x + r.width == width || clear(r.x + r.width - 1, y))
        })
    });
    merged.sort_by_key(|r| (r.y, r.x));
    merged
}

/// A flat border colour can provide separation evidence even for opaque
/// images. This mask is only for planning; it does not remove that background.
fn foreground_support<S: RasterSource + ?Sized>(
    source: &S,
    matte: Option<&AlphaMatte>,
) -> Vec<bool> {
    let (width, height) = (source.width(), source.height());
    if let Some(matte) = matte {
        return (0..matte.len())
            .into_par_iter()
            .map(|i| matte.get(i) >= 1.0 / 16.0)
            .collect();
    }
    let band = (width.min(height) / 32).clamp(1, 64);
    let stride = (width.max(height) / 1024).max(1);
    let mut histogram = std::collections::BTreeMap::<[u8; 3], usize>::new();
    let mut samples = Vec::new();
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            if x >= band && y >= band && x + band < width && y + band < height {
                continue;
            }
            let rgb = source.get(x, y);
            let bin = rgb.map(|v| (v.clamp(0.0, 1.0) * 31.0).round() as u8);
            *histogram.entry(bin).or_default() += 1;
            samples.push((bin, rgb));
        }
    }
    let Some((&bin, &count)) = histogram.iter().max_by_key(|(_, count)| *count) else {
        return vec![true; width * height];
    };
    if count * 2 < samples.len() {
        return vec![true; width * height];
    }
    let background: [f32; 3] = std::array::from_fn(|c| {
        let mut values: Vec<_> = samples
            .iter()
            .filter(|(b, _)| *b == bin)
            .map(|(_, rgb)| rgb[c])
            .collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    });
    let flat = |rgb: [f32; 3]| {
        rgb.iter()
            .zip(background)
            .all(|(&a, b)| (a - b).abs() <= 3.0 / 255.0)
    };
    if samples.iter().filter(|(_, rgb)| flat(*rgb)).count() * 2 < samples.len() {
        return vec![true; width * height];
    }
    (0..width * height)
        .into_par_iter()
        .map(|i| !flat(source.get(i % width, i / width)))
        .collect()
}

fn mapped_sample(
    raster: &Raster,
    represented_source: SourceRect,
    source_x: f32,
    source_y: f32,
) -> [f32; 3] {
    let local_x = source_x - represented_source.x as f32;
    let local_y = source_y - represented_source.y as f32;
    let x = (local_x + 0.5) * raster.width as f32 / represented_source.width.max(1) as f32 - 0.5;
    let y = (local_y + 0.5) * raster.height as f32 / represented_source.height.max(1) as f32 - 0.5;
    raster.sample_bilinear(x, y)
}

/// A background separator in the source must also remain a separator after
/// fitting. Reject a crop whose border disagrees with the retained base.
pub(crate) fn refinement_boundary_matches(
    base: &Raster,
    child: &Raster,
    whole: SourceRect,
    core: SourceRect,
    expanded: SourceRect,
) -> bool {
    let matches = |x: usize, y: usize| {
        mapped_sample(base, whole, x as f32, y as f32)
            .iter()
            .zip(mapped_sample(child, expanded, x as f32, y as f32))
            .all(|(&a, b)| (a - b).abs() <= 2.0 / 255.0)
    };
    for inset in 0..2.min(core.width).min(core.height) {
        if !(core.x..core.x + core.width).all(|x| {
            (core.y == 0 || matches(x, core.y + inset))
                && (core.y + core.height == whole.height
                    || matches(x, core.y + core.height - 1 - inset))
        }) || !(core.y..core.y + core.height).all(|y| {
            (core.x == 0 || matches(core.x + inset, y))
                && (core.x + core.width == whole.width
                    || matches(core.x + core.width - 1 - inset, y))
        }) {
            return false;
        }
    }
    true
}

/// Compare one source-space region with a raster that represents a possibly
/// larger source rectangle.  The local tail prevents small icon details from
/// disappearing into a large flat background; only source edges not present
/// in the candidate add an edge penalty.
pub(crate) fn perceptual_score<S: RasterSource + ?Sized>(
    source: &S,
    region: SourceRect,
    candidate: &Raster,
    candidate_source: SourceRect,
) -> PerceptualScore {
    const MAXIMUM_SAMPLES: usize = 32_768;
    let step = ((region.area().max(1) as f64 / MAXIMUM_SAMPLES as f64)
        .sqrt()
        .ceil() as usize)
        .max(1);
    let sample_capacity = region.area().div_ceil(step * step);
    let mut source_samples = Vec::<Lab>::with_capacity(sample_capacity);
    let mut represented_samples = Vec::<Lab>::with_capacity(sample_capacity);
    let mut source_edge_starts = Vec::<Lab>::with_capacity(sample_capacity * 2);
    let mut source_edge_ends = Vec::<Lab>::with_capacity(sample_capacity * 2);
    let mut represented_edge_starts = Vec::<Lab>::with_capacity(sample_capacity * 2);
    let mut represented_edge_ends = Vec::<(usize, usize)>::with_capacity(sample_capacity * 2);
    for y in (region.y..region.y + region.height).step_by(step) {
        for x in (region.x..region.x + region.width).step_by(step) {
            let source_pixel = source.get(x, y);
            let represented = mapped_sample(candidate, candidate_source, x as f32, y as f32);
            let source_lab = rgb_to_lab(source_pixel);
            let represented_lab = rgb_to_lab(represented);
            source_samples.push(source_lab);
            represented_samples.push(represented_lab);
            for (following_x, following_y) in [
                ((x + 1).min(region.x + region.width - 1), y),
                (x, (y + 1).min(region.y + region.height - 1)),
            ] {
                if following_x == x && following_y == y {
                    continue;
                }
                source_edge_starts.push(source_lab);
                source_edge_ends.push(rgb_to_lab(source.get(following_x, following_y)));
                represented_edge_starts.push(represented_lab);
                represented_edge_ends.push((following_x, following_y));
            }
        }
    }
    let deltas = delta_e2000_pairs(&source_samples, &represented_samples);
    if deltas.is_empty() {
        return PerceptualScore::default();
    }

    // CIEDE2000 contains several elementary functions. Evaluate all source
    // edges in contiguous SIMD batches, then evaluate only the rendered edges
    // whose source counterparts are visible. This retains the exact sampling
    // and thresholds while avoiding tens of thousands of one-element SIMD
    // allocations per refinement region.
    let source_edges = delta_e2000_pairs(&source_edge_starts, &source_edge_ends);
    let mut visible_source_edges = Vec::<f32>::new();
    let mut visible_represented_starts = Vec::<Lab>::new();
    let mut visible_represented_ends = Vec::<Lab>::new();
    for ((&source_edge, &represented_start), &(following_x, following_y)) in source_edges
        .iter()
        .zip(&represented_edge_starts)
        .zip(&represented_edge_ends)
    {
        if source_edge < 6.0 {
            continue;
        }
        let represented_following = mapped_sample(
            candidate,
            candidate_source,
            following_x as f32,
            following_y as f32,
        );
        visible_source_edges.push(source_edge);
        visible_represented_starts.push(represented_start);
        visible_represented_ends.push(rgb_to_lab(represented_following));
    }
    let represented_edges =
        delta_e2000_pairs(&visible_represented_starts, &visible_represented_ends);
    let edge_samples = visible_source_edges.len();
    let missing_edges = represented_edges
        .iter()
        .zip(&visible_source_edges)
        .filter(|(represented_edge, source_edge)| **represented_edge < 0.55 * **source_edge)
        .count();
    let mean_delta_e = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let p90_delta_e = percentile(deltas, 0.90);
    let missing_edge_fraction = missing_edges as f32 / edge_samples.max(1) as f32;
    let combined = 0.45 * mean_delta_e + 0.40 * p90_delta_e + 3.0 * missing_edge_fraction;
    PerceptualScore {
        mean_delta_e,
        p90_delta_e,
        missing_edge_fraction,
        combined,
    }
}

fn local_model_cost(
    region: SourceRect,
    source_dimensions: (usize, usize),
    labels: &[u32],
    label_dimensions: (usize, usize),
) -> f32 {
    let (source_width, source_height) = source_dimensions;
    let (width, height) = label_dimensions;
    if labels.len() != width * height || width == 0 || height == 0 {
        return 1.0;
    }
    let x0 = (region.x * width / source_width.max(1)).min(width - 1);
    let y0 = (region.y * height / source_height.max(1)).min(height - 1);
    let x1 = ((region.x + region.width) * width)
        .div_ceil(source_width.max(1))
        .clamp(x0 + 1, width);
    let y1 = ((region.y + region.height) * height)
        .div_ceil(source_height.max(1))
        .clamp(y0 + 1, height);
    let mut owners = HashSet::<u32>::new();
    let mut transitions = 0_usize;
    let mut comparisons = 0_usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            owners.insert(labels[index]);
            if x + 1 < x1 {
                transitions += usize::from(labels[index] != labels[index + 1]);
                comparisons += 1;
            }
            if y + 1 < y1 {
                transitions += usize::from(labels[index] != labels[index + width]);
                comparisons += 1;
            }
        }
    }
    let pixels = (x1 - x0) * (y1 - y0);
    let boundary_density = transitions as f32 / comparisons.max(1) as f32;
    let region_density = owners.len() as f32 * 10_000.0 / pixels.max(1) as f32;
    1.0 + 8.0 * boundary_density + 0.015 * region_density
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_candidates<S: RasterSource + ?Sized>(
    source: &S,
    matte: Option<&AlphaMatte>,
    base: &Raster,
    labels: &[u32],
    tile_dimension: usize,
    maximum_candidates: usize,
    minimum_error: f32,
) -> Vec<RefinementCandidate> {
    let whole = SourceRect {
        x: 0,
        y: 0,
        width: source.width(),
        height: source.height(),
    };
    let support = foreground_support(source, matte);
    let regions = object_regions(&support, source.width(), source.height(), tile_dimension);
    let mut candidates = regions
        .into_iter()
        .filter_map(|core| {
            let baseline = perceptual_score(source, core, base, whole);
            if baseline.combined < minimum_error {
                return None;
            }
            let model_cost = local_model_cost(
                core,
                (source.width(), source.height()),
                labels,
                (base.width, base.height),
            );
            Some(RefinementCandidate {
                core,
                baseline,
                model_cost,
                priority: baseline.combined / model_cost.max(1e-6),
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.priority.total_cmp(&left.priority));
    candidates.truncate(maximum_candidates);
    candidates
}

fn number(value: f32) -> String {
    let rounded = (value * 1000.0).round() / 1000.0;
    if (rounded - rounded.round()).abs() < 1e-5 {
        format!("{rounded:.0}")
    } else {
        format!("{rounded:.3}").trim_end_matches('0').to_string()
    }
}

fn inner_svg(document: &str, prefix: &str) -> Result<String> {
    let svg = document
        .find("<svg")
        .ok_or_else(|| -> Error { "adaptive child SVG has no root element".into() })?;
    let body = document[svg..]
        .find('>')
        .map(|offset| svg + offset + 1)
        .ok_or_else(|| -> Error { "adaptive child SVG has an incomplete root element".into() })?;
    let end = document
        .rfind("</svg>")
        .ok_or_else(|| -> Error { "adaptive child SVG has no closing root element".into() })?;
    if body > end {
        return Err("adaptive child SVG root is malformed".into());
    }
    Ok(document[body..end]
        .replace("id=\"", &format!("id=\"{prefix}"))
        .replace("url(#", &format!("url(#{prefix}")))
}

pub(crate) fn compose_refinements(
    base_document: &str,
    base_dimensions: (usize, usize),
    source_dimensions: (usize, usize),
    refinements: &[EmbeddedRefinement],
    replace_base: bool,
) -> Result<String> {
    if refinements.is_empty() {
        return Ok(base_document.to_string());
    }
    let close = base_document
        .rfind("</svg>")
        .ok_or_else(|| -> Error { "base SVG has no closing root element".into() })?;
    let (base_width, base_height) = base_dimensions;
    let (source_width, source_height) = source_dimensions;
    let mut layer = String::from("<g id=\"adaptive-refinement-layer\">");
    for (index, refinement) in refinements.iter().enumerate() {
        let prefix = format!("lod-{index}-");
        let body = inner_svg(&refinement.document, &prefix)?;
        let x = refinement.core.x as f32 * base_width as f32 / source_width as f32;
        let y = refinement.core.y as f32 * base_height as f32 / source_height as f32;
        let width = refinement.core.width as f32 * base_width as f32 / source_width as f32;
        let height = refinement.core.height as f32 * base_height as f32 / source_height as f32;
        let child_scale_x =
            refinement.processing_width as f32 / refinement.expanded.width.max(1) as f32;
        let child_scale_y =
            refinement.processing_height as f32 / refinement.expanded.height.max(1) as f32;
        let view_x = (refinement.core.x - refinement.expanded.x) as f32 * child_scale_x;
        let view_y = (refinement.core.y - refinement.expanded.y) as f32 * child_scale_y;
        let view_width = refinement.core.width as f32 * child_scale_x;
        let view_height = refinement.core.height as f32 * child_scale_y;
        layer.push_str(&format!(
            "<svg data-adaptive-refinement=\"{}\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\" preserveAspectRatio=\"none\" overflow=\"hidden\">{}</svg>",
            index,
            number(x),
            number(y),
            number(width),
            number(height),
            number(view_x),
            number(view_y),
            number(view_width),
            number(view_height),
            body,
        ));
    }
    layer.push_str("</g>");
    let mut document = String::with_capacity(close + layer.len() + 512);
    if replace_base && refinements_cover_canvas(refinements, source_dimensions) {
        // No base element is visible. Keeping its paths and gradient definitions
        // would still charge parsing, storage and mask-rendering costs.
        let root = base_document
            .find("<svg")
            .ok_or("base SVG has no root element")?;
        let body = base_document[root..]
            .find('>')
            .ok_or("base SVG has an incomplete root element")?
            + root
            + 1;
        document.push_str(&base_document[..body]);
    } else if replace_base {
        // A transparent refinement must replace, rather than merely cover,
        // the coarse content in its core.  Mask those disjoint rectangles out
        // of the base first; otherwise a finer, smaller silhouette would
        // leave the coarse silhouette visible underneath it.
        let root = base_document
            .find("<svg")
            .ok_or_else(|| -> Error { "base SVG has no root element".into() })?;
        let body = base_document[root..]
            .find('>')
            .map(|offset| root + offset + 1)
            .ok_or_else(|| -> Error { "base SVG has an incomplete root element".into() })?;
        document.push_str(&base_document[..body]);
        document.push_str("<defs><mask id=\"adaptive-base-mask\" maskUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"");
        document.push_str(&base_width.to_string());
        document.push_str("\" height=\"");
        document.push_str(&base_height.to_string());
        document.push_str("\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
        for refinement in refinements {
            let x = refinement.core.x as f32 * base_width as f32 / source_width as f32;
            let y = refinement.core.y as f32 * base_height as f32 / source_height as f32;
            let width = refinement.core.width as f32 * base_width as f32 / source_width as f32;
            let height = refinement.core.height as f32 * base_height as f32 / source_height as f32;
            document.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"black\"/>",
                number(x),
                number(y),
                number(width),
                number(height),
            ));
        }
        document.push_str("</mask></defs><g mask=\"url(#adaptive-base-mask)\">");
        document.push_str(&base_document[body..close]);
        document.push_str("</g>");
    } else {
        document.push_str(&base_document[..close]);
    }
    document.push_str(&layer);
    document.push_str(&base_document[close..]);
    Ok(document)
}

pub(crate) fn refinements_cover_canvas(
    refinements: &[EmbeddedRefinement],
    dimensions: (usize, usize),
) -> bool {
    let (width, height) = dimensions;
    let mut area = 0_usize;
    for (i, refinement) in refinements.iter().enumerate() {
        let a = refinement.core;
        if a.width == 0
            || a.height == 0
            || a.x.saturating_add(a.width) > width
            || a.y.saturating_add(a.height) > height
        {
            return false;
        }
        if refinements[..i].iter().any(|r| {
            let b = r.core;
            a.x < b.x + b.width
                && b.x < a.x + a.width
                && a.y < b.y + b.height
                && b.y < a.y + a.height
        }) {
            return false;
        }
        area = area.saturating_add(a.area());
    }
    area > 0 && area == width.saturating_mul(height)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        for alpha in [Some(&matte), None] {
            let support = foreground_support(&source, alpha);
            let regions = object_regions(&support, source.width, source.height, 1400);
            assert!(
                regions.len() >= 24,
                "keyed={}: too few intact figures",
                alpha.is_some()
            );
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
            &base, &child, whole, core, core
        ));
        child.pixels[30 * 60] = [0.9; 3];
        assert!(!refinement_boundary_matches(
            &base, &child, whole, core, core
        ));
    }

    #[test]
    fn full_transparent_replacement_discards_base_but_partial_does_not() {
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
        let full = compose_refinements(base, (10, 10), (10, 10), &patches, true).unwrap();
        assert!(!full.contains("obsolete"));
        assert!(full.contains("lod-0-replacement") && full.contains("lod-1-replacement"));
        patches[1].expanded.x = 4;
        patches[1].core.x = 4; // The summed area still matches, but there is overlap and a hole.
        let partial = compose_refinements(base, (10, 10), (10, 10), &patches, true).unwrap();
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
            base,
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
                document: child.to_string(),
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
    fn transparent_refinement_masks_its_core_out_of_the_base() {
        let base = "<svg width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\"/></svg>";
        let child = "<svg width=\"4\" height=\"4\"></svg>";
        let result = compose_refinements(
            base,
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
                document: child.to_string(),
                processing_width: 4,
                processing_height: 4,
            }],
            true,
        )
        .unwrap();
        assert!(result.contains("id=\"adaptive-base-mask\""));
        assert!(result.contains("<rect x=\"2\" y=\"3\" width=\"4\" height=\"4\" fill=\"black\"/>"));
        assert!(result.contains("<g mask=\"url(#adaptive-base-mask)\">"));
    }
}
