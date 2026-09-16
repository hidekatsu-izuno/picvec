//! Source-resolution, rate-distortion driven refinement support.
//!
//! The base vectorizer remains the only image model.  This module identifies
//! source regions where rerunning that same model at a finer pyramid level is
//! likely to pay for its additional SVG representation cost, and composes the
//! accepted refinements back into the base document.

use crate::svg_document::attrs;
use crate::svg_document::{Document, Elements};
use std::collections::HashSet;

use rayon::prelude::*;
use serde::Serialize;

use crate::chroma::AlphaMatte;
use crate::color::{delta_e_ok_pairs, rgb_to_oklab, Oklab};
use crate::raster::{percentile, Raster, RasterSource};
use crate::Result;

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
    #[cfg(any(test, feature = "diagnostics"))]
    pub baseline: PerceptualScore,
    #[cfg(any(test, feature = "diagnostics"))]
    pub model_cost: f32,
    pub priority: f32,
}

impl RefinementCandidate {
    /// Once encoded, bytes replace the partition-cost estimate. Charging
    /// both would penalize fragmented shading twice, even when its measured
    /// quality gain per byte is better than another candidate's.
    pub(crate) fn measured_rate(&self, gain: f32, area: usize, bytes: usize) -> f32 {
        gain * area as f32 / bytes.max(1) as f32
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EmbeddedRefinement {
    pub core: SourceRect,
    pub expanded: SourceRect,
    pub document: Document,
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
    /// Subset of complexity rejections caused by the global byte budget.
    pub rejected_for_budget: usize,
    /// Error of the coarse planning preview, not a source-resolution render.
    pub baseline_mean_delta_e: f32,
    /// Planning error minus estimated gains; not measured completed-SVG quality.
    pub refined_mean_delta_e: f32,
    /// Area-weighted source-resolution gains of the accepted replacements.
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
    const PADDING: usize = 16;
    // A thin canvas-spanning separator is background structure, even when
    // it touches a figure. Ignore it for grouping only; the source, matte,
    // and final rendered-join validation still retain every one of its pixels.
    let original_support = support;
    let support = grouping_support_without_separators(support, width, height, maximum_dimension);
    let support = support.as_slice();
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
        regions.push(SourceRect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        });
    }
    // Disconnected details inside/near another silhouette must share its fit.
    // Recheck after every union: a union's rectangle can enclose a third object.
    let mut merged = Vec::<SourceRect>::new();
    for mut region in regions {
        let mut i = 0;
        while i < merged.len() {
            let other = merged[i];
            let padded = region.expanded(PADDING, width, height);
            let other_padded = other.expanded(PADDING, width, height);
            if padded.x < other_padded.x + other_padded.width
                && other_padded.x < padded.x + padded.width
                && padded.y < other_padded.y + other_padded.height
                && other_padded.y < padded.y + padded.height
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
    // An excluded large component (for example a sheet's separator grid)
    // can lie inside the usual padding without touching the figure itself.
    // Place the crop in the middle of the available background gap instead
    // of discarding the whole figure or cutting the neighbouring component.
    let crossed_separators: Vec<bool> = merged
        .iter()
        .map(|r| {
            let separator =
                |x: usize, y: usize| original_support[y * width + x] && !support[y * width + x];
            (r.x..r.x + r.width).any(|x| separator(x, r.y) || separator(x, r.y + r.height - 1))
                || (r.y..r.y + r.height)
                    .any(|y| separator(r.x, y) || separator(r.x + r.width - 1, y))
        })
        .collect();
    for (region, &crosses) in merged.iter_mut().zip(&crossed_separators) {
        // Nearby separators still constrain padding. Only a separator that
        // actually crosses this figure's bounds may enter its native crop.
        let support = if crosses { support } else { original_support };
        let mut padding = PADDING;
        for distance in 1..=2 * PADDING {
            let ring = region.expanded(distance, width, height);
            let occupied = |x: usize, y: usize| support[y * width + x];
            if (ring.x..ring.x + ring.width).any(|x| {
                (ring.y < region.y && occupied(x, ring.y))
                    || (ring.y + ring.height > region.y + region.height
                        && occupied(x, ring.y + ring.height - 1))
            }) || (ring.y..ring.y + ring.height).any(|y| {
                (ring.x < region.x && occupied(ring.x, y))
                    || (ring.x + ring.width > region.x + region.width
                        && occupied(ring.x + ring.width - 1, y))
            }) {
                padding = (distance - 1) / 2;
                break;
            }
        }
        *region = region.expanded(padding, width, height);
    }
    let mut region_index = 0;
    merged.retain(|r| {
        let support = if crossed_separators[region_index] {
            support
        } else {
            original_support
        };
        region_index += 1;
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

fn grouping_support_without_separators(
    support: &[bool],
    width: usize,
    height: usize,
    maximum_dimension: usize,
) -> Vec<bool> {
    let mut result = support.to_vec();
    let max_band = (width.min(height) / 256).clamp(2, 16);
    for vertical in [false, true] {
        let (along, across) = if vertical {
            (height, width)
        } else {
            (width, height)
        };
        if along <= maximum_dimension {
            continue;
        }
        let counts: Vec<usize> = (0..across)
            .map(|i| {
                (0..along)
                    .filter(|&j| {
                        let (x, y) = if vertical { (i, j) } else { (j, i) };
                        support[y * width + x]
                    })
                    .count()
            })
            .collect();
        let mut i = 0;
        while i < across {
            if counts[i] * 5 < along * 4 {
                i += 1;
                continue;
            }
            let start = i;
            while i < across && counts[i] * 5 >= along * 4 {
                i += 1;
            }
            if i - start > max_band {
                continue;
            }
            // Remove the antialias shoulders from grouping too. Leaving a
            // one-pixel shoulder breaks a long grid into short components
            // which then spuriously merge neighbouring figures together.
            let shoulder = (max_band / 4).clamp(1, 3);
            for k in start.saturating_sub(shoulder)..(i + shoulder).min(across) {
                for j in 0..along {
                    let (x, y) = if vertical { (k, j) } else { (j, k) };
                    result[y * width + x] = false;
                }
            }
        }
    }
    result
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
    base_source: SourceRect,
    whole: SourceRect,
    core: SourceRect,
    expanded: SourceRect,
) -> bool {
    let matches = |x: usize, y: usize| {
        mapped_sample(base, base_source, x as f32, y as f32)
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

/// Move a rejected replacement boundary inward only through source pixels
/// that are fully transparent, or outward through the same background support
/// used by planning. The rendered join retains the existing tolerance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matching_refinement_core(
    base: &Raster,
    child: &Raster,
    base_source: SourceRect,
    whole: SourceRect,
    mut core: SourceRect,
    expanded: SourceRect,
    matte: Option<&crate::chroma::AlphaMatte>,
) -> Option<SourceRect> {
    let original = core;
    for inset in 0..=4 {
        if refinement_boundary_matches(base, child, base_source, whole, core, expanded) {
            return Some(core);
        }
        let matte = matte?;
        if inset == 4 || core.width <= 2 || core.height <= 2 {
            break;
        }
        let clear = |x: usize, y: usize| matte.get(y * whole.width + x) <= 0.0;
        if !(core.x..core.x + core.width)
            .all(|x| clear(x, core.y) && clear(x, core.y + core.height - 1))
            || !(core.y..core.y + core.height)
                .all(|y| clear(core.x, y) && clear(core.x + core.width - 1, y))
        {
            break;
        }
        core = SourceRect {
            x: core.x + 1,
            y: core.y + 1,
            width: core.width - 2,
            height: core.height - 2,
        };
    }
    // A faint source halo cannot be trimmed. Include it instead, staying
    // inside the already fitted child and outside foreign foreground.
    let matte = matte?;
    core = original;
    for _ in 0..4 {
        let larger = core.expanded(1, whole.width, whole.height);
        if larger == core
            || larger.x < expanded.x
            || larger.y < expanded.y
            || larger.x + larger.width > expanded.x + expanded.width
            || larger.y + larger.height > expanded.y + expanded.height
        {
            break;
        }
        let background = |x: usize, y: usize| matte.get(y * whole.width + x) < 1.0 / 16.0;
        if !(larger.x..larger.x + larger.width)
            .all(|x| background(x, larger.y) && background(x, larger.y + larger.height - 1))
            || !(larger.y..larger.y + larger.height)
                .all(|y| background(larger.x, y) && background(larger.x + larger.width - 1, y))
        {
            break;
        }
        core = larger;
        if refinement_boundary_matches(base, child, base_source, whole, core, expanded) {
            return Some(core);
        }
    }
    None
}

// SplitMix64's fixed integer mixer makes sampling reproducible across runs,
// worker counts and platforms, without a periodic pixel-grid phase.
fn dispersed_sample(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
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
    // Stratify the flattened source region, rather than always visiting the
    // same phase of a two-dimensional grid. Each stratum contributes one
    // deterministic, dispersed colour sample. Small regions remain exhaustive.
    let stride = region.area().div_ceil(MAXIMUM_SAMPLES).max(1);
    let sample_capacity = region.area().div_ceil(stride);
    let mut source_samples = Vec::<Oklab>::with_capacity(sample_capacity);
    let mut represented_samples = Vec::<Oklab>::with_capacity(sample_capacity);
    let mut source_edge_starts = Vec::<Oklab>::with_capacity(sample_capacity * 2);
    let mut source_edge_ends = Vec::<Oklab>::with_capacity(sample_capacity * 2);
    let mut represented_edge_starts = Vec::<Oklab>::with_capacity(sample_capacity * 2);
    let mut represented_edge_ends = Vec::<(usize, usize)>::with_capacity(sample_capacity * 2);
    let position = |index: usize| {
        (
            region.x + index % region.width,
            region.y + index / region.width,
        )
    };
    for start in (0..region.area()).step_by(stride) {
        let end = (start + stride).min(region.area());
        let index = start + (dispersed_sample(start / stride) % (end - start) as u64) as usize;
        let (x, y) = position(index);
        source_samples.push(rgb_to_oklab(source.get(x, y)));
        represented_samples.push(rgb_to_oklab(mapped_sample(
            candidate,
            candidate_source,
            x as f32,
            y as f32,
        )));

        // Discover edge evidence independently of the colour samples. A thin
        // line or isolated dot must not vanish simply because it lies between
        // sample points. Scan native neighbours with cheap RGB differences;
        // keep the strongest horizontal and vertical edge in each stratum.
        // OKLab still decides visibility below, and candidate pixels play no
        // part in selecting source evidence. Scratch and perceptual work stay
        // bounded even for very large source regions.
        let mut edges = [None; 2];
        let mut strengths = [0.0_f32; 2];
        for index in start..end {
            let (x, y) = position(index);
            let rgb = source.get(x, y);
            for (axis, following) in [
                (x + 1 < region.x + region.width).then_some((x + 1, y)),
                (y + 1 < region.y + region.height).then_some((x, y + 1)),
            ]
            .into_iter()
            .enumerate()
            {
                let Some((xx, yy)) = following else { continue };
                let next = source.get(xx, yy);
                let strength = (0..3).map(|c| (rgb[c] - next[c]).powi(2)).sum::<f32>();
                if strength > strengths[axis] {
                    strengths[axis] = strength;
                    edges[axis] = Some((x, y, xx, yy, rgb, next));
                }
            }
        }
        for (x, y, xx, yy, rgb, next) in edges.into_iter().flatten() {
            source_edge_starts.push(rgb_to_oklab(rgb));
            source_edge_ends.push(rgb_to_oklab(next));
            represented_edge_starts.push(rgb_to_oklab(mapped_sample(
                candidate,
                candidate_source,
                x as f32,
                y as f32,
            )));
            represented_edge_ends.push((xx, yy));
        }
    }
    let deltas = delta_e_ok_pairs(&source_samples, &represented_samples);
    if deltas.is_empty() {
        return PerceptualScore::default();
    }

    // Evaluate source edges together, then only the rendered counterparts
    // whose source edges are visible.
    let source_edges = delta_e_ok_pairs(&source_edge_starts, &source_edge_ends);
    let mut visible_source_edges = Vec::<f32>::new();
    let mut visible_represented_starts = Vec::<Oklab>::new();
    let mut visible_represented_ends = Vec::<Oklab>::new();
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
        visible_represented_ends.push(rgb_to_oklab(represented_following));
    }
    let represented_edges =
        delta_e_ok_pairs(&visible_represented_starts, &visible_represented_ends);
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
    let mut regions = object_regions(&support, source.width(), source.height(), tile_dimension);
    if regions.is_empty()
        && source.width().max(source.height()) > tile_dimension
        && support.iter().any(|&foreground| !foreground)
    {
        // A separated oversized object can use a complete source model with
        // no crop joins. With no separation evidence (all-foreground support),
        // keep the selected global model instead of unconditionally rerunning
        // a photograph at its entire input resolution. Oversized illustrations
        // still pass the usual measured source-error and byte-cost gates.
        regions.push(whole);
    }
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
                #[cfg(any(test, feature = "diagnostics"))]
                baseline,
                #[cfg(any(test, feature = "diagnostics"))]
                model_cost,
                // Estimate cost only until the actual SVG byte count is
                // available. A linear charge rejects dense repeated details
                // before their measured gain and representation cost are known.
                priority: baseline.combined / model_cost.sqrt().max(1e-6),
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

pub(crate) fn compose_refinements(
    base_document: &Document,
    base_dimensions: (usize, usize),
    source_dimensions: (usize, usize),
    refinements: &[EmbeddedRefinement],
    replace_base: bool,
) -> Result<Document> {
    if refinements.is_empty() {
        return Ok(base_document.clone());
    }
    let (base_width, base_height) = base_dimensions;
    let (source_width, source_height) = source_dimensions;
    let mut layer = Elements::new();
    layer.open("g", attrs([("id", "adaptive-refinement-layer".into())]));
    for (index, refinement) in refinements.iter().enumerate() {
        let prefix = format!("lod-{index}-");
        let mut body = Elements::new();
        body.roots = refinement.document.root().children.clone();
        for node in &mut body.roots {
            node.namespace_ids(&prefix);
        }
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
        layer.open(
            "svg",
            attrs([
                ("data-adaptive-refinement", (index).to_string()),
                ("x", (number(x)).to_string()),
                ("y", (number(y)).to_string()),
                ("width", (number(width)).to_string()),
                ("height", (number(height)).to_string()),
                (
                    "viewBox",
                    format!(
                        "{0} {1} {2} {3}",
                        number(view_x),
                        number(view_y),
                        number(view_width),
                        number(view_height)
                    ),
                ),
                ("preserveAspectRatio", "none".into()),
                ("overflow", "hidden".into()),
            ]),
        );
        layer.append(body);
        layer.close();
    }
    layer.close();
    let mut document = base_document.clone();
    if refinements_cover_canvas(refinements, source_dimensions) {
        document.root_mut().children.clear();
    } else if replace_base {
        // Build the uncovered strips. Subtracting a union this way also
        // handles overlapping refinement cores without restoring their overlap.
        let mut xs = vec![0, source_width];
        for refinement in refinements {
            xs.push(refinement.core.x.min(source_width));
            xs.push((refinement.core.x + refinement.core.width).min(source_width));
        }
        xs.sort_unstable();
        xs.dedup();
        let mut clip = String::new();
        for span in xs.windows(2) {
            let mut covered: Vec<_> = refinements
                .iter()
                .filter(|r| r.core.x < span[1] && r.core.x + r.core.width > span[0])
                .map(|r| {
                    (
                        r.core.y.min(source_height),
                        (r.core.y + r.core.height).min(source_height),
                    )
                })
                .collect();
            covered.sort_unstable();
            covered.push((source_height, source_height));
            let mut cursor = 0;
            for (top, bottom) in covered {
                if top > cursor {
                    let x0 = span[0] as f32 * base_width as f32 / source_width as f32;
                    let x1 = span[1] as f32 * base_width as f32 / source_width as f32;
                    let y0 = cursor as f32 * base_height as f32 / source_height as f32;
                    let y1 = top as f32 * base_height as f32 / source_height as f32;
                    clip.push_str(&format!(
                        "M{} {}H{}V{}H{}Z",
                        number(x0),
                        number(y0),
                        number(x1),
                        number(y1),
                        number(x0)
                    ));
                }
                cursor = cursor.max(bottom);
            }
        }
        let mut children = Elements::new();
        children.open("defs", vec![]);
        children.open("clipPath", attrs([("id", "adaptive-base-clip".into())]));
        children.leaf(
            "path",
            attrs([("d", clip.to_string()), ("clip-rule", "nonzero".into())]),
        );
        children.close();
        children.close();
        let mut base = Elements::new();
        base.roots = std::mem::take(&mut document.root_mut().children);
        children.append(base.wrap(
            "g",
            attrs([("clip-path", "url(#adaptive-base-clip)".into())]),
        ));
        document.root_mut().children = children.roots;
    }
    document.root_mut().children.extend(layer.roots);
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
include!("../tests/unit/adaptive.rs");
