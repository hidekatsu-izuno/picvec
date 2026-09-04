//! Source-resolution, rate-distortion driven refinement support.
//!
//! The base vectorizer remains the only image model.  This module identifies
//! source regions where rerunning that same model at a finer pyramid level is
//! likely to pay for its additional SVG representation cost, and composes the
//! accepted refinements back into the base document.

use std::collections::HashSet;

use serde::Serialize;

use crate::color::{delta_e2000, rgb_to_lab};
use crate::raster::{percentile, Raster};
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

/// Partition source space into balanced rectangular leaves.  Unlike a fixed
/// grid size, balanced leaves avoid a narrow remainder strip whose local
/// statistics and vector cost are not comparable with the other candidates.
pub(crate) fn balanced_regions(
    width: usize,
    height: usize,
    maximum_dimension: usize,
) -> Vec<SourceRect> {
    let maximum = maximum_dimension.max(1);
    let columns = width.div_ceil(maximum).max(1);
    let rows = height.div_ceil(maximum).max(1);
    let mut regions = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let y0 = row * height / rows;
        let y1 = (row + 1) * height / rows;
        for column in 0..columns {
            let x0 = column * width / columns;
            let x1 = (column + 1) * width / columns;
            regions.push(SourceRect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    regions
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

/// Compare one source-space region with a raster that represents a possibly
/// larger source rectangle.  The local tail prevents small icon details from
/// disappearing into a large flat background; only source edges not present
/// in the candidate add an edge penalty.
pub(crate) fn perceptual_score(
    source: &Raster,
    region: SourceRect,
    candidate: &Raster,
    candidate_source: SourceRect,
) -> PerceptualScore {
    const MAXIMUM_SAMPLES: usize = 32_768;
    let step = ((region.area().max(1) as f64 / MAXIMUM_SAMPLES as f64)
        .sqrt()
        .ceil() as usize)
        .max(1);
    let mut deltas = Vec::with_capacity(region.area().div_ceil(step * step));
    let mut edge_samples = 0_usize;
    let mut missing_edges = 0_usize;
    for y in (region.y..region.y + region.height).step_by(step) {
        for x in (region.x..region.x + region.width).step_by(step) {
            let source_pixel = source.get(x, y);
            let represented = mapped_sample(candidate, candidate_source, x as f32, y as f32);
            deltas.push(delta_e2000(
                rgb_to_lab(source_pixel),
                rgb_to_lab(represented),
            ));
            for (following_x, following_y) in [
                ((x + 1).min(region.x + region.width - 1), y),
                (x, (y + 1).min(region.y + region.height - 1)),
            ] {
                if following_x == x && following_y == y {
                    continue;
                }
                let source_edge = delta_e2000(
                    rgb_to_lab(source_pixel),
                    rgb_to_lab(source.get(following_x, following_y)),
                );
                if source_edge < 6.0 {
                    continue;
                }
                edge_samples += 1;
                let represented_following = mapped_sample(
                    candidate,
                    candidate_source,
                    following_x as f32,
                    following_y as f32,
                );
                let represented_edge =
                    delta_e2000(rgb_to_lab(represented), rgb_to_lab(represented_following));
                if represented_edge < 0.55 * source_edge {
                    missing_edges += 1;
                }
            }
        }
    }
    if deltas.is_empty() {
        return PerceptualScore::default();
    }
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
pub(crate) fn plan_candidates(
    source: &Raster,
    base: &Raster,
    labels: &[u32],
    tile_dimension: usize,
    maximum_candidates: usize,
    minimum_error: f32,
) -> Vec<RefinementCandidate> {
    let whole = SourceRect {
        x: 0,
        y: 0,
        width: source.width,
        height: source.height,
    };
    let mut candidates = balanced_regions(source.width, source.height, tile_dimension)
        .into_iter()
        .filter_map(|core| {
            let baseline = perceptual_score(source, core, base, whole);
            if baseline.combined < minimum_error {
                return None;
            }
            let model_cost = local_model_cost(
                core,
                (source.width, source.height),
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
    if replace_base {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_regions_cover_image_once_without_thin_remainder() {
        let regions = balanced_regions(5016, 5016, 1400);
        assert_eq!(regions.len(), 16);
        assert!(regions.iter().all(|region| region.width == 1254));
        let area = regions.iter().map(|region| region.area()).sum::<usize>();
        assert_eq!(area, 5016 * 5016);
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
