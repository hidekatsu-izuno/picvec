use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(test)]
use rayon::prelude::*;
use serde::Serialize;
use tempfile::{Builder as TemporaryFileBuilder, NamedTempFile};

use std::collections::HashMap;

use crate::adaptive::{
    compose_refinements, matching_refinement_core, perceptual_score, plan_candidates,
    refinements_cover_canvas, AdaptiveRefinementSummary, EmbeddedRefinement, SourceRect,
};
use crate::chroma::{self, AlphaMatte, AlphaTransparencySummary, ChromaKeySummary};
use crate::color::{rgb_to_oklab, Oklab};
use crate::config::Config;
use crate::edge::{classify, dilate, dilate_square, perceptual_smooth, EdgeSummary};
use crate::geometry::GeometrySummary;
use crate::gradient::{
    fit_all_without_topology, merge_partition, merge_source_supported_paints_with_evidence,
    refresh_summary, GradientSummary, Paint,
};
use crate::hierarchy::{HierarchicalTopology, HierarchicalTopologySummary};
use crate::metrics::QualityMetrics;
use crate::optimize::{summarize as optimization_summary, OptimizationSummary};
use crate::ownership::{resolve as resolve_boundary_ownership, BoundaryOwnershipSummary};
use crate::raster::{Raster, RasterSource, SourceRaster};
use crate::segment::{
    refine_thin_paint_ownership, regularize_boundaries, replace_final_exact_paint_labels,
    Segmentation, SegmentationSummary,
};
use crate::structural::{
    analyse_with_protection as analyse_structural, StructuralInk, StructuralSummary,
};
use crate::svg::{
    serialize_filtered_with_alpha_cached as serialize_svg, GeometryCache, SvgSummary,
};
use crate::union_find::UnionFind;
use crate::{Error, Result};

const MINIMUM_AUTOMATIC_TARGET_PIXELS: f32 = 1_200_000.0;
const MAXIMUM_AUTOMATIC_TARGET_PIXELS: f32 = 2_000_000.0;

#[derive(Clone, Debug, Default, Serialize)]
pub struct ComplexityProbe {
    pub probe_width: usize,
    pub probe_height: usize,
    pub probe_region_count: usize,
    pub probe_region_density: f32,
    pub edge_density: f32,
    pub normalized_region_density: f32,
    pub normalized_edge_density: f32,
    pub complexity: f32,
    pub target_pixels: f32,
    pub selected_dimension: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub input_width: usize,
    pub input_height: usize,
    pub processing_width: usize,
    pub processing_height: usize,
    pub output: PathBuf,
    pub elapsed_seconds: f64,
    pub execution_threads: usize,
    pub complexity: ComplexityProbe,
    pub source_alpha: AlphaTransparencySummary,
    pub chroma_key: ChromaKeySummary,
    pub adaptive_refinement: AdaptiveRefinementSummary,
    pub hierarchical_topology: HierarchicalTopologySummary,
    pub edge_roles: EdgeSummary,
    pub segmentation: SegmentationSummary,
    pub structural: StructuralSummary,
    pub ownership: BoundaryOwnershipSummary,
    pub paint_order: crate::paint_order::Summary,
    pub gradients: GradientSummary,
    pub geometry: GeometrySummary,
    pub optimization: OptimizationSummary,
    pub svg: SvgSummary,
    /// Report-only metrics, present only when explicitly requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualityMetrics>,
}

#[cfg(feature = "diagnostics")]
fn report_progress(config: &Config, stage: &str, started: Instant, checkpoint: &mut Instant) {
    if !config.retain_diagnostics {
        return;
    }
    let now = Instant::now();
    eprintln!(
        "picvec stage {stage}: {:.3}s (total {:.3}s)",
        now.duration_since(*checkpoint).as_secs_f64(),
        now.duration_since(started).as_secs_f64(),
    );
    *checkpoint = now;
}

#[cfg(not(feature = "diagnostics"))]
fn report_progress(_config: &Config, _stage: &str, _started: Instant, _checkpoint: &mut Instant) {}

#[cfg(feature = "diagnostics")]
fn save_pipeline_diagnostic(name: &str, image: &Raster) {
    let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") else {
        return;
    };
    let path = PathBuf::from(format!("{prefix}-{name}.png"));
    let _ = image.save(&path);
    let mut bytes = Vec::with_capacity(image.pixels.len() * 3 * 4);
    for pixel in &image.pixels {
        for &channel in pixel {
            bytes.extend_from_slice(&channel.to_le_bytes());
        }
    }
    let raw_path = PathBuf::from(format!(
        "{prefix}-{name}-{}x{}.f32le",
        image.width, image.height
    ));
    let _ = fs::write(raw_path, bytes);
}

#[cfg(not(feature = "diagnostics"))]
fn save_pipeline_diagnostic(_name: &str, _image: &Raster) {}

fn merge_exact_final_paints(
    source: &Raster,
    segmentation: &mut Segmentation,
    paints: &mut Vec<Paint>,
) -> usize {
    let count = segmentation.regions.len();
    if count < 2 || paints.len() != count {
        return 0;
    }
    // Decide equivalence in the representation we will actually serialize.
    // Otherwise sub-byte fit noise creates separate contours and seam strokes
    // even though both faces have exactly the same SVG paint.
    let identities: Vec<_> = paints.iter().map(crate::svg::appearance_key).collect();
    let mut owners = UnionFind::new(count);
    let mut accepted = 0_usize;
    for y in 0..segmentation.height {
        for x in 0..segmentation.width {
            let index = y * segmentation.width + x;
            let current = segmentation.labels[index] as usize;
            for neighbour in [
                (x + 1 < segmentation.width).then_some(index + 1),
                (y + 1 < segmentation.height).then_some(index + segmentation.width),
            ]
            .into_iter()
            .flatten()
            {
                let following = segmentation.labels[neighbour] as usize;
                if current == following
                    || !(paints[current] == paints[following]
                        || identities[current]
                            .as_ref()
                            .is_some_and(|key| Some(key) == identities[following].as_ref()))
                {
                    continue;
                }
                let first = owners.find(current);
                let second = owners.find(following);
                if first != second {
                    owners.union(first, second);
                    accepted += 1;
                }
            }
        }
    }
    if accepted == 0 {
        return 0;
    }
    let roots: Vec<usize> = (0..count).map(|label| owners.find(label)).collect();
    let mut unique = roots.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut representative = vec![usize::MAX; count];
    for (label, &root) in roots.iter().enumerate() {
        representative[root] = representative[root].min(label);
    }
    let merged_paints = unique
        .iter()
        .map(|&root| paints[representative[root]].clone())
        .collect::<Vec<_>>();
    let labels = segmentation
        .labels
        .iter()
        .map(|&label| roots[label as usize] as u32)
        .collect::<Vec<_>>();
    replace_final_exact_paint_labels(source, segmentation, labels, accepted);
    *paints = merged_paints;
    accepted
}
#[cfg(feature = "diagnostics")]
fn save_label_diagnostic(name: &str, labels: &[u32], width: usize, height: usize) {
    let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") else {
        return;
    };
    let mut bytes = Vec::with_capacity(labels.len() * 4);
    for &label in labels {
        bytes.extend_from_slice(&label.to_le_bytes());
    }
    let path = PathBuf::from(format!("{prefix}-{name}-{width}x{height}.u32le"));
    let _ = fs::write(path, bytes);
}

#[cfg(not(feature = "diagnostics"))]
fn save_label_diagnostic(_name: &str, _labels: &[u32], _width: usize, _height: usize) {}

#[cfg(feature = "diagnostics")]
fn save_mask_diagnostic(name: &str, mask: &[bool], width: usize, height: usize) {
    let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") else {
        return;
    };
    let values: Vec<u8> = mask.iter().map(|&value| u8::from(value)).collect();
    let path = PathBuf::from(format!("{prefix}-{name}-{width}x{height}.u8"));
    let _ = fs::write(path, values);
}

#[cfg(not(feature = "diagnostics"))]
fn save_mask_diagnostic(_name: &str, _mask: &[bool], _width: usize, _height: usize) {}

fn select_dimension<R: RasterSource + ?Sized>(image: &R, config: &Config) -> ComplexityProbe {
    let (_, maximum) = config.automatic_dimension_bounds();
    if image.width().max(image.height()) <= maximum as usize
        && (image.width() * image.height()) as f32 <= MINIMUM_AUTOMATIC_TARGET_PIXELS
    {
        ComplexityProbe {
            selected_dimension: image.width().max(image.height()) as u32,
            target_pixels: MINIMUM_AUTOMATIC_TARGET_PIXELS,
            ..ComplexityProbe::default()
        }
    } else {
        estimate_dimension(image, config)
    }
}

fn resize_processing<R: RasterSource + ?Sized>(
    source: &R,
    matte: Option<&AlphaMatte>,
    source_alpha: bool,
    maximum: u32,
) -> (Raster, Option<AlphaMatte>) {
    if source_alpha {
        let (image, matte) = chroma::resize_source_alpha(
            source,
            matte.expect("source alpha requires a matte"),
            maximum,
        );
        (image, Some(matte))
    } else {
        let image = source.resize_max(maximum);
        let matte = matte.map(|matte| matte.resized(image.width, image.height));
        (image, matte)
    }
}

fn estimate_dimension<R: RasterSource + ?Sized>(image: &R, config: &Config) -> ComplexityProbe {
    let probe_max = image.width().max(image.height()).min(1024) as u32;
    let probe = image.resize_max(probe_max);
    let lab: Vec<Oklab> = probe.pixels.iter().copied().map(rgb_to_oklab).collect();
    let at = |x: isize, y: isize| {
        let px = x.clamp(0, probe.width.saturating_sub(1) as isize) as usize;
        let py = y.clamp(0, probe.height.saturating_sub(1) as isize) as usize;
        lab[py * probe.width + px]
    };
    let mut magnitude = Vec::with_capacity(lab.len());
    for y in 0..probe.height as isize {
        for x in 0..probe.width as isize {
            let mut energy = 0.0_f32;
            for channel in 0..3 {
                let value = |dx: isize, dy: isize| {
                    let sample = at(x + dx, y + dy);
                    match channel {
                        0 => sample.l,
                        1 => sample.a,
                        _ => sample.b,
                    }
                };
                let gx = -value(-1, -1) + value(1, -1) - 2.0 * value(-1, 0) + 2.0 * value(1, 0)
                    - value(-1, 1)
                    + value(1, 1);
                let gy = -value(-1, -1) - 2.0 * value(0, -1) - value(1, -1)
                    + value(-1, 1)
                    + 2.0 * value(0, 1)
                    + value(1, 1);
                energy += gx * gx + gy * gy;
            }
            magnitude.push(energy.sqrt());
        }
    }
    // A percentile threshold followed by counting values above that same
    // percentile is almost constant by construction. A fixed perceptual
    // Sobel response instead measures how much of the probe contains a
    // visible transition. A response of eight is approximately a two-DeltaE
    // step after the Sobel kernel's fourfold gain.
    const EDGE_MAGNITUDE_THRESHOLD: f32 = 8.0;
    let edge_density = magnitude
        .iter()
        .filter(|&&value| value >= EDGE_MAGNITUDE_THRESHOLD)
        .count() as f32
        / magnitude.len().max(1) as f32;

    let coarse: Vec<(i16, i16, i16)> = lab
        .iter()
        .map(|value| {
            (
                (value.l / 4.0).round() as i16,
                (value.a / 4.0).round() as i16,
                (value.b / 4.0).round() as i16,
            )
        })
        .collect();
    let mut ids = HashMap::<(i16, i16, i16), u32>::new();
    let mut palette = Vec::with_capacity(coarse.len());
    for key in coarse {
        let following = ids.len() as u32;
        palette.push(*ids.entry(key).or_insert(following));
    }
    let mut visited = vec![false; palette.len()];
    let mut probe_region_count = 0_usize;
    let mut stack = Vec::<usize>::new();
    for start in 0..palette.len() {
        if visited[start] {
            continue;
        }
        probe_region_count += 1;
        visited[start] = true;
        stack.push(start);
        while let Some(index) = stack.pop() {
            let x = index % probe.width;
            let y = index / probe.width;
            for neighbour in [
                (x > 0).then(|| index - 1),
                (x + 1 < probe.width).then(|| index + 1),
                (y > 0).then(|| index - probe.width),
                (y + 1 < probe.height).then(|| index + probe.width),
            ]
            .into_iter()
            .flatten()
            {
                if !visited[neighbour] && palette[neighbour] == palette[start] {
                    visited[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
    }
    let probe_region_density =
        probe_region_count as f32 / probe.pixels.len().max(1) as f32 * 1_000_000.0;
    let normalized_region_density = (probe_region_density / 60_000.0).clamp(0.0, 1.0);
    let normalized_edge_density = (edge_density / 0.18).clamp(0.0, 1.0);
    let complexity = 0.65 * normalized_region_density + 0.35 * normalized_edge_density;
    let target_pixels = MINIMUM_AUTOMATIC_TARGET_PIXELS
        + (MAXIMUM_AUTOMATIC_TARGET_PIXELS - MINIMUM_AUTOMATIC_TARGET_PIXELS) * complexity;
    let source_pixels = (image.width() * image.height()).max(1) as f32;
    let estimated =
        image.width().max(image.height()) as f32 * (target_pixels / source_pixels).sqrt();
    let (automatic_minimum, automatic_maximum) = config.automatic_dimension_bounds();
    let selected = estimated
        .round()
        .clamp(automatic_minimum as f32, automatic_maximum as f32)
        .min(image.width().max(image.height()) as f32) as u32;
    ComplexityProbe {
        probe_width: probe.width,
        probe_height: probe.height,
        probe_region_count,
        probe_region_density,
        edge_density,
        normalized_region_density,
        normalized_edge_density,
        complexity,
        target_pixels,
        selected_dimension: selected,
    }
}

fn output_parent(output: &Path) -> &Path {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn temporary_svg(output: &Path, purpose: &str) -> Result<NamedTempFile> {
    let parent = output_parent(output);
    TemporaryFileBuilder::new()
        .prefix(&format!(".picvec-{purpose}-"))
        .suffix(".svg")
        .tempfile_in(parent)
        .map_err(|error| -> Error {
            format!(
                "could not create a temporary SVG next to {}: {error}",
                output.display()
            )
            .into()
        })
}

#[allow(clippy::too_many_arguments)]
fn render_svg_preview(
    dimensions: (usize, usize),
    paint_layer: (
        &[crate::geometry::RegionGeometry],
        &[crate::gradient::Paint],
    ),
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
    excluded_regions: &[bool],
    face_alpha: Option<&crate::face_alpha::FaceAlpha>,
    background: [f32; 3],
    geometry_cache: &mut GeometryCache,
) -> Result<Raster> {
    let (width, height) = dimensions;
    let (geometry, paints) = paint_layer;
    let (document, _) = serialize_svg(
        width,
        height,
        geometry,
        paints,
        structural,
        paint_overlap,
        final_geometry,
        excluded_regions,
        face_alpha,
        geometry_cache,
    );
    render_svg_document_on(&document, width, height, background)
}

fn render_svg_document_on(
    document: &str,
    width: usize,
    height: usize,
    background: [f32; 3],
) -> Result<Raster> {
    let tree = parse_svg_document(document)?;
    render_svg_tree_on(
        &tree,
        width,
        height,
        resvg::tiny_skia::Transform::identity(),
        background,
    )
}

fn render_svg_tree_on(
    tree: &resvg::usvg::Tree,
    width: usize,
    height: usize,
    transform: resvg::tiny_skia::Transform,
    background: [f32; 3],
) -> Result<Raster> {
    let width = u32::try_from(width)
        .map_err(|_| -> Error { "SVG preview width exceeds the renderer limit".into() })?;
    let height = u32::try_from(height)
        .map_err(|_| -> Error { "SVG preview height exceeds the renderer limit".into() })?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| -> Error { "could not allocate the SVG preview raster".into() })?;
    resvg::render(tree, transform, &mut pixmap.as_mut());
    let pixels = pixmap
        .pixels()
        .iter()
        .copied()
        .map(|pixel| {
            let inverse_alpha = (255 - pixel.alpha()) as f32 / 255.0;
            [
                pixel.red() as f32 / 255.0 + inverse_alpha * background[0],
                pixel.green() as f32 / 255.0 + inverse_alpha * background[1],
                pixel.blue() as f32 / 255.0 + inverse_alpha * background[2],
            ]
        })
        .collect();
    Ok(Raster::new(width as usize, height as usize, pixels))
}

fn parse_svg_document(document: &str) -> Result<resvg::usvg::Tree> {
    resvg::usvg::Tree::from_data(document.as_bytes(), &resvg::usvg::Options::default()).map_err(
        |error| -> Error { format!("could not parse the generated SVG preview: {error}").into() },
    )
}

/// Convert one raster into exactly the SVG path requested by the caller.
/// No source copy, rendered PNG, or JSON sidecar is produced.
pub fn vectorize(input: &Path, output: &Path, config: &Config) -> Result<Summary> {
    config.validate()?;
    if output
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| !value.eq_ignore_ascii_case("svg"))
        .unwrap_or(true)
    {
        return Err("output path must end in .svg".into());
    }
    if !input.is_file() {
        return Err(format!(
            "input raster does not exist or is not a file: {}",
            input.display()
        )
        .into());
    }
    let threads = execution_thread_count(config);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| -> Error { format!("could not create Rayon pool: {error}").into() })?;
    pool.install(|| vectorize_inner(input, output, config, threads))
}

fn default_execution_thread_count(cpu_count: usize) -> usize {
    (cpu_count / 2).clamp(1, 10)
}

fn execution_thread_count(config: &Config) -> usize {
    let logical = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if config.rayon_threads > 0 {
        return config.rayon_threads.min(logical).max(1);
    }
    default_execution_thread_count(logical)
}

#[cfg(target_os = "linux")]
fn available_memory_bytes() -> Option<usize> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            if fields.next()? != "MemAvailable:" {
                return None;
            }
            fields.next()?.parse::<usize>().ok()?.checked_mul(1024)
        })
}

#[cfg(not(target_os = "linux"))]
fn available_memory_bytes() -> Option<usize> {
    None
}

fn adaptive_parallel_jobs(
    candidates: &[crate::adaptive::RefinementCandidate],
    image_dimensions: (usize, usize),
    execution_threads: usize,
) -> usize {
    const MAXIMUM_JOBS: usize = 10;
    const ESTIMATED_WORKING_BYTES_PER_PIXEL: usize = 320;
    let maximum_memory_budget = usize::try_from(4_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX);
    let (width, height) = image_dimensions;
    let largest_job_pixels = candidates
        .iter()
        .map(|candidate| {
            let margin = (candidate.core.width.min(candidate.core.height) / 64).clamp(8, 24);
            candidate.core.expanded(margin, width, height).area()
        })
        .max()
        .unwrap_or(1);
    let estimated_job_bytes = largest_job_pixels.saturating_mul(ESTIMATED_WORKING_BYTES_PER_PIXEL);
    let memory_jobs = available_memory_bytes()
        .map(|available| {
            (available / 2)
                .min(maximum_memory_budget)
                .checked_div(estimated_job_bytes.max(1))
                .unwrap_or(1)
        })
        .unwrap_or(4);
    candidates
        .len()
        .min(execution_threads)
        .min(MAXIMUM_JOBS)
        .min(memory_jobs.max(1))
        .max(1)
}

/// Keep a bounded number of independent jobs in flight without waiting for
/// the slowest member of a batch. Commit results in their original order.
fn bounded_map<T: Sync, U: Send, F: Fn(&T) -> U + Sync>(
    tasks: &[T],
    limit: usize,
    evaluate: F,
) -> Vec<U> {
    if tasks.is_empty() {
        return Vec::new();
    }
    if limit <= 1 {
        return tasks.iter().map(evaluate).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let outputs = std::sync::Mutex::new((0..tasks.len()).map(|_| None).collect::<Vec<Option<U>>>());
    rayon::scope(|scope| {
        for _ in 0..limit.min(tasks.len()) {
            let (next, outputs, evaluate) = (&next, &outputs, &evaluate);
            scope.spawn(move |_| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(task) = tasks.get(i) else {
                    break;
                };
                let outcome = evaluate(task);
                outputs.lock().unwrap()[i] = Some(outcome);
            });
        }
    });
    outputs
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect()
}

struct EvaluatedRefinement {
    source_alpha_bits: u8,
    embedded: EmbeddedRefinement,
    svg: SvgSummary,
    baseline_mean: f32,
    refined_mean: f32,
    rate: f32,
}

enum RefinementOutcome {
    NotFiner,
    QualityRejected,
    ComplexityRejected,
    Accepted(Box<EvaluatedRefinement>),
}

/// Apply a single rate-distortion model to every input.  No photo/illustration
/// classifier is involved: a region is refined only when the same vectorizer
/// explains source-resolution evidence sufficiently better per added SVG
/// byte.  High-frequency photographic texture therefore competes on exactly
/// the same terms as a small, cheaply representable icon feature.
#[allow(clippy::too_many_arguments)]
fn adaptively_refine(
    vector_source: Option<&SourceRaster>,
    reference_source: Option<&SourceRaster>,
    source_matte: Option<&AlphaMatte>,
    source_alpha: bool,
    input_dimensions: (usize, usize),
    core: &mut CoreVectorization,
    config: &Config,
    execution_threads: usize,
) -> Result<AdaptiveRefinementSummary> {
    let (input_width, input_height) = input_dimensions;
    let source_scale = (input_width as f32 / core.processing_reference.width.max(1) as f32)
        .max(input_height as f32 / core.processing_reference.height.max(1) as f32);
    let mut summary = AdaptiveRefinementSummary {
        enabled: config.adaptive_refinement,
        source_scale,
        ..AdaptiveRefinementSummary::default()
    };
    if !config.adaptive_refinement || source_scale < config.adaptive_min_source_scale {
        return Ok(summary);
    }

    let vector_source = vector_source.ok_or_else(|| -> Error {
        "adaptive source raster was released before refinement".into()
    })?;
    let reference_source = reference_source.ok_or_else(|| -> Error {
        "adaptive reference raster was released before refinement".into()
    })?;
    let base_render = render_svg_document_on(
        &core.document,
        core.processing_reference.width,
        core.processing_reference.height,
        core.preview_background,
    )?;
    let whole = SourceRect {
        x: 0,
        y: 0,
        width: input_width,
        height: input_height,
    };
    let baseline_whole = perceptual_score(reference_source, whole, &base_render, whole);
    summary.baseline_mean_delta_e = baseline_whole.mean_delta_e;
    summary.refined_mean_delta_e = baseline_whole.mean_delta_e;

    let mut candidates = plan_candidates(
        reference_source,
        source_matte,
        &base_render,
        &core.labels,
        config.adaptive_tile_dimension as usize,
        config.adaptive_max_patches,
        config.adaptive_min_perceptual_gain,
    );
    summary.proposed_regions = candidates.len();
    #[cfg(feature = "diagnostics")]
    if config.retain_diagnostics {
        for candidate in &candidates {
            eprintln!(
                "picvec adaptive candidate {} {} {} {}: error={:.4} model_cost={:.4} priority={:.4}",
                candidate.core.x,
                candidate.core.y,
                candidate.core.width,
                candidate.core.height,
                candidate.baseline.combined,
                candidate.model_cost,
                candidate.priority,
            );
        }
    }
    // Estimate the available gain using a square-root partition
    // charge. Actual SVG bytes are
    // unknown here, so this is only a prefilter, not a bound on the final
    // rate. Dense repeated details must get a chance to demonstrate their
    // measured gain and representation cost.
    let predicted_rate_threshold =
        config.adaptive_min_predicted_rate * config.adaptive_complexity_penalty.sqrt();
    candidates.retain(|candidate| candidate.priority >= predicted_rate_threshold);
    summary.prefiltered_for_complexity = summary.proposed_regions - candidates.len();
    summary.candidate_regions = candidates.len();
    if candidates.is_empty() {
        return Ok(summary);
    }

    let mut child_config = config.clone();
    child_config.adaptive_refinement = false;
    child_config.compute_quality_metrics = false;
    child_config.retain_diagnostics = config.retain_diagnostics;
    let base_scale = 1.0 / source_scale;
    let mut evaluated = Vec::<EvaluatedRefinement>::new();
    // Keep enough independent jobs in flight to cover serial geometry and
    // segmentation stages, while estimating their dense working sets before
    // committing memory. The cap prevents a transiently high MemAvailable
    // value from turning a large source into an unbounded allocation burst.
    let parallel_jobs =
        adaptive_parallel_jobs(&candidates, (input_width, input_height), execution_threads);
    summary.parallel_jobs = parallel_jobs;
    // Compare the actual vector join at source resolution. Upsampling the
    // coarse preview invents an antialias halo across otherwise clear gaps.
    // Parse once; render only the small child bounds, not a full-size sheet.
    let base_tree = parse_svg_document(&core.document)?;
    let outcomes = bounded_map(&candidates, parallel_jobs,
            |candidate| -> Result<RefinementOutcome> {
                let margin = (candidate.core.width.min(candidate.core.height) / 64).clamp(8, 24);
                let expanded = candidate.core.expanded(margin, input_width, input_height);
                let crop =
                    vector_source.crop(expanded.x, expanded.y, expanded.width, expanded.height);
                let crop_matte = source_matte.map(|matte| {
                    matte.crop(expanded.x, expanded.y, expanded.width, expanded.height)
                });
                let probe = if expanded.width.max(expanded.height)
                    > config.adaptive_tile_dimension as usize
                {
                    // A connected object cannot be tiled without cutting its
                    // strokes/gradients. Evaluate its complete source model;
                    // the usual measured quality and byte-cost gates apply.
                    ComplexityProbe {
                        selected_dimension: expanded.width.max(expanded.height) as u32,
                        ..ComplexityProbe::default()
                    }
                } else if child_config.auto_dimension {
                    select_dimension(&crop, &child_config)
                } else {
                    ComplexityProbe {
                        selected_dimension: child_config
                            .maximum_dimension
                            .min(crop.width.max(crop.height) as u32),
                        ..ComplexityProbe::default()
                    }
                };
                let (processing, processing_matte) = resize_processing(
                    &crop,
                    crop_matte.as_ref(),
                    source_alpha,
                    probe.selected_dimension.max(64),
                );
                let local_scale = (processing.width as f32 / expanded.width.max(1) as f32)
                    .min(processing.height as f32 / expanded.height.max(1) as f32);
                if local_scale <= 1.1 * base_scale {
                    return Ok(RefinementOutcome::NotFiner);
                }
                #[cfg(feature = "diagnostics")]
                if config.retain_diagnostics {
                    eprintln!("picvec adaptive processing: source {} {} {} {}, processing {} {}", expanded.x, expanded.y, expanded.width, expanded.height, processing.width, processing.height);
                }
                let child = vectorize_processing(
                    processing,
                    processing_matte.as_ref(),
                    source_alpha,
                    core.preview_background,
                    &child_config,
                )?;
                let child_render = render_svg_document_on(
                    &child.document,
                    child.processing_reference.width,
                    child.processing_reference.height,
                    core.preview_background,
                )?;
                let boundary_base = render_svg_tree_on(
                    &base_tree,
                    expanded.width,
                    expanded.height,
                    resvg::tiny_skia::Transform::from_row(
                        input_width as f32 / base_tree.size().width(), 0.0,
                        0.0, input_height as f32 / base_tree.size().height(),
                        -(expanded.x as f32), -(expanded.y as f32),
                    ),
                    core.preview_background,
                )?;
                let matched_core = matching_refinement_core(
                    &boundary_base,
                    &child_render,
                    expanded,
                    whole,
                    candidate.core,
                    expanded,
                    source_matte,
                );
                let core = matched_core.unwrap_or(candidate.core);
                let baseline = if core == candidate.core { candidate.baseline }
                    else { perceptual_score(reference_source, core, &base_render, whole) };
                let refined = perceptual_score(reference_source, core, &child_render, expanded);
                let boundary_matches = matched_core.is_some();
                #[cfg(feature = "diagnostics")]
                if config.retain_diagnostics {
                    eprintln!(
                        "picvec adaptive evaluated {} {} {} {}: baseline={:?} refined={:?} boundary_matches={} bytes={}",
                        core.x,
                        core.y,
                        core.width,
                        core.height,
                        baseline,
                        refined,
                        boundary_matches,
                        child.svg.bytes,
                    );
                }
                if !boundary_matches {
                    return Ok(RefinementOutcome::QualityRejected);
                }
                let combined_gain = baseline.combined - refined.combined;
                if combined_gain < config.adaptive_min_perceptual_gain
                    || refined.p90_delta_e > baseline.p90_delta_e + 0.25
                    || refined.missing_edge_fraction
                        > baseline.missing_edge_fraction + 0.025
                {
                    return Ok(RefinementOutcome::QualityRejected);
                }
                let rate = candidate.measured_rate(combined_gain, core.area(), child.svg.bytes);
                if rate < config.adaptive_complexity_penalty {
                    return Ok(RefinementOutcome::ComplexityRejected);
                }
                Ok(RefinementOutcome::Accepted(Box::new(EvaluatedRefinement {
                    embedded: EmbeddedRefinement {
                        core,
                        expanded,
                        document: child.document,
                        processing_width: child.processing_reference.width,
                        processing_height: child.processing_reference.height,
                    },
                    svg: child.svg,
                    source_alpha_bits: child.source_alpha_bits,
                    baseline_mean: baseline.mean_delta_e,
                    refined_mean: refined.mean_delta_e,
                    rate,
                })))
            })
            .into_iter().collect::<Result<Vec<_>>>()?;
    for outcome in outcomes {
        match outcome {
            RefinementOutcome::NotFiner => summary.rejected_for_quality += 1,
            RefinementOutcome::QualityRejected => {
                summary.evaluated_regions += 1;
                summary.rejected_for_quality += 1;
            }
            RefinementOutcome::ComplexityRejected => {
                summary.evaluated_regions += 1;
                summary.rejected_for_complexity += 1;
            }
            RefinementOutcome::Accepted(refinement) => {
                summary.evaluated_regions += 1;
                evaluated.push(*refinement);
            }
        }
    }

    // Rank candidates deterministically by efficiency. An explicitly configured
    // byte budget retains the best candidates first.
    evaluated.sort_by(|left, right| {
        right
            .rate
            .total_cmp(&left.rate)
            .then_with(|| left.embedded.core.y.cmp(&right.embedded.core.y))
            .then_with(|| left.embedded.core.x.cmp(&right.embedded.core.x))
    });
    let mut accepted = Vec::<EmbeddedRefinement>::new();
    let mut refinement_svg = SvgSummary::default();
    for refinement in evaluated {
        if config.adaptive_svg_budget_bytes != 0
            && summary.added_svg_bytes.saturating_add(refinement.svg.bytes)
                > config.adaptive_svg_budget_bytes
        {
            summary.rejected_for_complexity += 1;
            summary.rejected_for_budget += 1;
            #[cfg(feature = "diagnostics")]
            if config.retain_diagnostics {
                eprintln!(
                    "picvec adaptive budget skipped {} {} {} {}: bytes={} used={} limit={}",
                    refinement.embedded.core.x,
                    refinement.embedded.core.y,
                    refinement.embedded.core.width,
                    refinement.embedded.core.height,
                    refinement.svg.bytes,
                    summary.added_svg_bytes,
                    config.adaptive_svg_budget_bytes
                );
            }
            continue;
        }
        let area_weight = refinement.embedded.core.area() as f32 / whole.area().max(1) as f32;
        summary.estimated_global_delta_e_reduction +=
            (refinement.baseline_mean - refinement.refined_mean).max(0.0) * area_weight;
        summary.added_svg_bytes += refinement.svg.bytes;
        refinement_svg.add_elements_from(&refinement.svg);
        core.source_alpha_bits = core.source_alpha_bits.max(refinement.source_alpha_bits);
        accepted.push(refinement.embedded);
    }
    summary.accepted_regions = accepted.len();
    summary.refined_mean_delta_e =
        (summary.baseline_mean_delta_e - summary.estimated_global_delta_e_reduction).max(0.0);
    if accepted.is_empty() {
        return Ok(summary);
    }
    if refinements_cover_canvas(&accepted, input_dimensions) {
        core.svg = SvgSummary::default();
    }
    core.svg.add_elements_from(&refinement_svg);
    core.document = compose_refinements(
        &core.document,
        (
            core.processing_reference.width,
            core.processing_reference.height,
        ),
        input_dimensions,
        &accepted,
        source_matte.is_some(),
    )?;
    core.svg.bytes = core.document.len();
    // Parse the composed document even when report-only quality metrics are
    // disabled. This turns any namespace/viewBox integration defect into an
    // atomic conversion failure instead of writing a malformed SVG.
    if config.compute_quality_metrics {
        let final_render = render_svg_document_on(
            &core.document,
            core.processing_reference.width,
            core.processing_reference.height,
            core.preview_background,
        )?;
        #[cfg(feature = "diagnostics")]
        {
            core.quality = Some(crate::metrics::compare(
                &core.processing_reference,
                &final_render,
            ));
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = final_render;
    } else {
        parse_svg_document(&core.document)?;
    }
    Ok(summary)
}

fn vectorize_inner(
    input: &Path,
    output: &Path,
    config: &Config,
    execution_threads: usize,
) -> Result<Summary> {
    fs::create_dir_all(output_parent(output))?;
    let started = Instant::now();
    let (decoded, decoded_alpha) = SourceRaster::load_with_alpha(
        input,
        config.maximum_input_dimension,
        config.maximum_input_pixels,
        config.maximum_decode_bytes,
    )?;
    let input_width = decoded.width;
    let input_height = decoded.height;
    let source_has_alpha = decoded_alpha.is_some();
    let (
        mut source,
        mut source_reference,
        mut input_matte,
        preview_background,
        detected_key,
        alpha_backing,
    ) = if let Some(alpha) = decoded_alpha {
        let matte = AlphaMatte::from_u8(input_width, input_height, alpha);
        let backing = chroma::select_alpha_backing(&decoded, &matte);
        let source = chroma::prepare_compact_source_alpha(&decoded, &matte);
        let reference = chroma::composite_source_over(&source, &matte, backing);
        (source, reference, Some(matte), backing, None, Some(backing))
    } else if let Some(key) = config
        .remove_chroma_key_background
        .then(|| chroma::detect(&decoded))
        .flatten()
    {
        let matte = chroma::pull_matte(&decoded, key);
        let separated = chroma::separate_compact_foreground(&decoded, &matte, key.sampled);
        (
            separated.clone(),
            separated,
            Some(matte),
            key.sampled,
            Some(key),
            None,
        )
    } else {
        (decoded.clone(), decoded, None, [1.0; 3], None, None)
    };
    let complexity = if config.auto_dimension {
        select_dimension(&source_reference, config)
    } else {
        ComplexityProbe {
            selected_dimension: config
                .maximum_dimension
                .min(source.width.max(source.height) as u32),
            ..ComplexityProbe::default()
        }
    };
    let mut separators = Vec::new();
    let mut separator_quality_reference = None;
    if config.adaptive_refinement && detected_key.is_some() {
        if let Some(matte) = &input_matte {
            let (bands, cleaned) =
                crate::separators::extract(&source, matte, config.adaptive_tile_dimension as usize);
            if let Some(cleaned) = cleaned {
                if config.compute_quality_metrics {
                    separator_quality_reference =
                        Some(source_reference.resize_max(complexity.selected_dimension.max(64)));
                }
                // Keep the keyed-source convention for hidden colours. The
                // pale band returns as one vector backdrop after refinement;
                // opaque foreground at its crossings remains in this matte.
                source = SourceRaster::from_unorm16_fn(input_width, input_height, |i| {
                    if cleaned.get(i) == 0.0 {
                        preview_background
                    } else {
                        source.get(i % input_width, i / input_width)
                    }
                });
                source_reference = source.clone();
                input_matte = Some(cleaned);
                separators = bands;
            }
        }
    }
    let (processing, processing_matte) = resize_processing(
        &source,
        input_matte.as_ref(),
        source_has_alpha,
        complexity.selected_dimension.max(64),
    );
    let processing_width = processing.width;
    let processing_height = processing.height;
    let source_scale = (input_width as f32 / processing_width.max(1) as f32)
        .max(input_height as f32 / processing_height.max(1) as f32);
    let retain_adaptive_source =
        config.adaptive_refinement && source_scale >= config.adaptive_min_source_scale;
    let adaptive_source = retain_adaptive_source.then_some(source);
    let adaptive_reference = retain_adaptive_source.then_some(source_reference);
    let adaptive_matte = if retain_adaptive_source {
        input_matte
    } else {
        None
    };
    let mut core = vectorize_processing(
        processing,
        processing_matte.as_ref(),
        source_has_alpha,
        preview_background,
        config,
    )?;
    let adaptive_refinement = adaptively_refine(
        adaptive_source.as_ref(),
        adaptive_reference.as_ref(),
        adaptive_matte.as_ref(),
        source_has_alpha,
        (input_width, input_height),
        &mut core,
        config,
        execution_threads,
    )?;
    crate::separators::prepend(
        &mut core.document,
        &separators,
        [
            processing_width as f32 / input_width as f32,
            processing_height as f32 / input_height as f32,
        ],
    );
    core.svg.rect_elements += separators.len();
    core.svg.bytes = core.document.len();
    if let Some(reference) = separator_quality_reference {
        let rendered = render_svg_document_on(
            &core.document,
            processing_width,
            processing_height,
            preview_background,
        )?;
        core.quality = Some(crate::metrics::compare(&reference, &rendered));
    }
    let to_u8 =
        |color: [f32; 3]| color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8);
    let source_alpha = AlphaTransparencySummary {
        detected: source_has_alpha,
        temporary_backing_color: alpha_backing.map(to_u8),
        quantization_bits: if source_has_alpha {
            core.source_alpha_bits
        } else {
            0
        },
        mask_paths: if source_has_alpha {
            core.svg.alpha_mask_paths
        } else {
            0
        },
        removed_regions: if source_has_alpha {
            core.removed_background_regions
        } else {
            0
        },
    };
    let chroma_key = detected_key
        .map(|key| key.summary(true, core.removed_background_regions))
        .unwrap_or(ChromaKeySummary {
            enabled: config.remove_chroma_key_background,
            ..ChromaKeySummary::default()
        });
    (core.svg.objects, core.svg.path_subpaths) = crate::svg::document_counts(&core.document)?;
    let temporary = temporary_svg(output, "output")?;
    fs::write(temporary.path(), core.document.as_bytes())?;
    temporary
        .persist(output)
        .map_err(|error| -> Error { error.error.into() })?;
    Ok(Summary {
        input_width,
        input_height,
        processing_width,
        processing_height,
        output: output.to_path_buf(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        execution_threads,
        complexity,
        source_alpha,
        chroma_key,
        adaptive_refinement,
        hierarchical_topology: core.hierarchical_topology,
        edge_roles: core.edge_roles,
        segmentation: core.segmentation,
        structural: core.structural,
        ownership: core.ownership,
        paint_order: core.paint_order,
        gradients: core.gradients,
        geometry: core.geometry,
        optimization: core.optimization,
        svg: core.svg,
        quality: core.quality,
    })
}

struct CoreVectorization {
    document: String,
    processing_reference: Raster,
    labels: Vec<u32>,
    removed_background_regions: usize,
    source_alpha_bits: u8,
    preview_background: [f32; 3],
    hierarchical_topology: HierarchicalTopologySummary,
    edge_roles: EdgeSummary,
    segmentation: SegmentationSummary,
    structural: StructuralSummary,
    ownership: BoundaryOwnershipSummary,
    paint_order: crate::paint_order::Summary,
    gradients: GradientSummary,
    geometry: GeometrySummary,
    optimization: OptimizationSummary,
    svg: SvgSummary,
    quality: Option<QualityMetrics>,
}

/// Run the complete vector model for an already selected processing raster.
/// Keeping this independent of file I/O lets adaptive refinement run the same
/// model on source-resolution regions instead of maintaining a second,
/// content-specific vectorizer.
fn vectorize_processing(
    processing: Raster,
    chroma_matte: Option<&AlphaMatte>,
    source_alpha: bool,
    preview_background: [f32; 3],
    config: &Config,
) -> Result<CoreVectorization> {
    let started = Instant::now();
    let mut checkpoint = started;
    save_pipeline_diagnostic("source", &processing);
    let processing_reference = if source_alpha {
        let matte = chroma_matte.ok_or_else(|| -> Error {
            "source alpha vectorization requires an alpha matte".into()
        })?;
        chroma::composite_over(&processing, matte, preview_background)
    } else {
        processing.clone()
    };
    // Extract coupled neutral fields before RGB segmentation and alpha-code
    // partitioning. Keep the original composite above for quality validation.
    let mut processing = processing;
    let (colour_matte, source_fields) = chroma_matte
        .filter(|_| source_alpha)
        .map(|m| crate::colour_fields::extract(&mut processing, m, preview_background))
        .map(|(m, p)| (Some(m), p))
        .unwrap_or_default();
    let chroma_matte = colour_matte.as_ref().or(chroma_matte);
    let (remaining_matte, composite_layers) = chroma_matte
        .filter(|_| source_alpha)
        .map(|matte| crate::neutral_fields::extract(&mut processing, matte, preview_background))
        .map(|(matte, layers)| (Some(matte), layers))
        .unwrap_or_default();
    let chroma_matte = remaining_matte.as_ref().or(chroma_matte);
    let coverage = chroma_matte
        .filter(|_| source_alpha)
        .and_then(crate::alpha_coverage::detect);
    let local_coverage = chroma_matte
        .filter(|_| source_alpha && coverage.is_none())
        .and_then(crate::alpha_coverage::detect_components);
    let processing = match (&coverage, chroma_matte) {
        (Some(coverage), Some(matte)) => coverage.extend_colour(&processing, matte),
        _ => processing,
    };
    save_pipeline_diagnostic("source-reference", &processing_reference);
    report_progress(config, "load-resize", started, &mut checkpoint);
    let mut roles = classify(&processing);
    save_mask_diagnostic(
        "edge-boundary",
        &roles.boundary,
        processing.width,
        processing.height,
    );
    save_mask_diagnostic(
        "edge-visible-ridge-centres",
        &roles.visible_ridge_centres,
        processing.width,
        processing.height,
    );
    save_mask_diagnostic(
        "edge-visible-ridge-coverage",
        &roles.visible_ridge_coverage,
        processing.width,
        processing.height,
    );
    save_mask_diagnostic(
        "edge-dark-boundary-coverage",
        &roles.dark_boundary,
        processing.width,
        processing.height,
    );
    save_mask_diagnostic(
        "edge-shading",
        &roles.shading,
        processing.width,
        processing.height,
    );
    save_mask_diagnostic(
        "edge-face-barrier",
        &roles.face_barrier,
        processing.width,
        processing.height,
    );
    report_progress(config, "edge-roles", started, &mut checkpoint);
    let variable_opacity = source_alpha
        && crate::face_alpha::uniform_opacity(chroma_matte.unwrap(), coverage.as_ref()).is_none();
    // Keep chromatic filled features in RGBA Paint. Only opaque,
    // low-chroma ink may enter RGB stroke fitting in a mixed-opacity scene;
    // otherwise a lip or iris can be reduced to an unrelated centreline.
    let mut rgba_ink_centres = vec![false; processing.pixels.len()];
    if variable_opacity {
        for graph in [
            &roles.visible_ridge_graph,
            &roles.dark_boundary_graph,
            &roles.band_boundary_graph,
        ] {
            for edge in graph {
                let indices: Vec<_> = edge
                    .points
                    .iter()
                    .map(|p| {
                        let x = (p[0] - 0.5)
                            .round()
                            .clamp(0.0, (processing.width - 1) as f64)
                            as usize;
                        let y = (p[1] - 0.5)
                            .round()
                            .clamp(0.0, (processing.height - 1) as f64)
                            as usize;
                        y * processing.width + x
                    })
                    .collect();
                let core: Vec<_> = indices
                    .iter()
                    .filter(|&&i| chroma_matte.unwrap().get(i) >= 1.0)
                    .map(|&i| rgb_to_oklab(processing.pixels[i]))
                    .collect();
                if core.len() < 3 {
                    continue;
                }
                let lightness = crate::raster::percentile(core.iter().map(|c| c.l).collect(), 0.5);
                let chroma =
                    crate::raster::percentile(core.iter().map(|c| c.a.hypot(c.b)).collect(), 0.75);
                if lightness <= 75.0 && chroma <= 10.0 {
                    // AA endpoints may pick up chroma from adjacent paint.
                    // Classify the supported graph as a whole, not each endpoint.
                    for i in indices {
                        rgba_ink_centres[i] = true;
                    }
                }
            }
        }
    }
    let paint_owned_alpha = chroma_matte.filter(|_| source_alpha).map(|matte| {
        let clear: Vec<_> = (0..matte.len()).map(|i| matte.get(i) <= 0.0).collect();
        let edge = dilate_square(&clear, processing.width, processing.height, 2);
        (0..matte.len())
            .map(|i| {
                if edge[i] || matte.get(i) < 1.0 {
                    return true;
                }
                if !variable_opacity || rgba_ink_centres[i] {
                    return false;
                }
                let colour = rgb_to_oklab(processing.pixels[i]);
                colour.a.hypot(colour.b) > 10.0
            })
            .collect::<Vec<_>>()
    });
    let paint_ridge_coverage = roles.visible_ridge_coverage.clone();
    let paint_ridge_centres = roles.visible_ridge_centres.clone();
    let paint_dark_boundary = roles.dark_boundary.clone();
    let (mut paint_reference, mut structural_candidates) =
        analyse_structural(&processing, &mut roles, paint_owned_alpha.as_deref());
    if let Some(protected) = &paint_owned_alpha {
        structural_candidates.release_to_paint(protected, processing.width);
        for (i, &paint_owned) in protected.iter().enumerate() {
            if paint_owned {
                paint_reference.pixels[i] = processing.pixels[i];
                // Paint-owned ink still needs its source evidence during
                // segmentation, even though no structural graph owns it.
                roles.visible_ridge_coverage[i] = paint_ridge_coverage[i];
                roles.visible_ridge_centres[i] = paint_ridge_centres[i];
                structural_candidates.source_line_mask[i] |= paint_ridge_coverage[i];
            }
        }
    }
    save_mask_diagnostic(
        "paint-ownership",
        &structural_candidates.paint_ownership_mask,
        processing.width,
        processing.height,
    );
    save_pipeline_diagnostic("underpaint", &paint_reference);
    report_progress(config, "structural-analysis", started, &mut checkpoint);
    let smoothed = perceptual_smooth(&paint_reference, config);
    save_pipeline_diagnostic("smoothed", &smoothed);
    #[cfg(feature = "diagnostics")]
    if std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some() {
        let smoothed_lab = Raster::new(
            smoothed.width,
            smoothed.height,
            smoothed
                .pixels
                .iter()
                .copied()
                .map(|pixel| {
                    let lab = rgb_to_oklab(pixel);
                    [lab.l, lab.a, lab.b]
                })
                .collect(),
        );
        save_pipeline_diagnostic("smoothed-lab", &smoothed_lab);
    }
    report_progress(config, "perceptual-smoothing", started, &mut checkpoint);
    let paint_owned_lines: Vec<_> = (0..processing.pixels.len())
        .map(|i| {
            variable_opacity
                && !structural_candidates.paint_ownership_mask[i]
                && paint_dark_boundary[i]
        })
        .collect();
    let mut segmentation = crate::segment::segment_with_paint_owned_lines(
        &smoothed,
        &roles,
        config,
        &paint_owned_lines,
    );
    // Once alpha is classified as coverage of a uniform-opacity material,
    // 253/254/255 edge samples are not distinct paint opacities. Testing
    // their raw alpha here blocks RGB material merging at those edges.
    // Native support is split explicitly by face_alpha::prepare below.
    let material_matte = chroma_matte.filter(|_| source_alpha && coverage.is_none());
    crate::segment::absorb_micro_regions(&paint_reference, &mut segmentation, material_matte);
    save_mask_diagnostic(
        "paint-samples",
        &segmentation.paint_samples,
        processing.width,
        processing.height,
    );
    save_label_diagnostic(
        "segmented-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(config, "segmentation", started, &mut checkpoint);
    // The regularizer sees the same underpainted geometry reference as Paint.
    // Only a measured material boundary is restored from the source. Passing
    // the source wholesale would freeze the medial ridge that was removed on
    // purpose and preserve its raster staircase in the face partition.
    let mut geometry_edge_reference = paint_reference.clone();
    let material_barrier =
        dilate_square(&roles.face_barrier, processing.width, processing.height, 1);
    for (index, &barrier) in material_barrier.iter().enumerate() {
        if barrier && !structural_candidates.boundary_stroke_mask[index] {
            geometry_edge_reference.pixels[index] = processing.pixels[index];
        }
    }
    save_pipeline_diagnostic("geometry-edge-reference", &geometry_edge_reference);
    regularize_boundaries(
        &smoothed,
        &geometry_edge_reference,
        &mut segmentation,
        &roles,
        &structural_candidates.source_line_mask,
        config,
    );
    save_label_diagnostic(
        "regularized-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    save_pipeline_diagnostic("regularized-canonical", &segmentation.canonical);
    report_progress(config, "boundary-regularization", started, &mut checkpoint);
    merge_partition(
        &paint_reference,
        &geometry_edge_reference,
        &mut segmentation,
        &roles,
        config,
    );
    save_pipeline_diagnostic("pre-thin-canonical", &segmentation.canonical);
    save_label_diagnostic(
        "merged-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(config, "paint-aware-merge", started, &mut checkpoint);
    let thin_protection: Vec<_> = structural_candidates
        .paint_ownership_mask
        .iter()
        .zip(&paint_owned_lines)
        .map(|(&structural, &paint)| structural || paint)
        .collect();
    let refined_structural_ownership = refine_thin_paint_ownership(
        &paint_reference,
        &mut segmentation,
        &thin_protection,
        &structural_candidates.residual_source_line_mask(),
    );
    crate::segment::absorb_micro_regions(&paint_reference, &mut segmentation, material_matte);
    save_label_diagnostic(
        "thin-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(config, "thin-paint-ownership", started, &mut checkpoint);
    let ridge_analysis = crate::ridge::analyze(&segmentation.canonical);
    segmentation.paint_samples = crate::ridge::adjust_paint_samples_from_analysis(
        &paint_reference,
        &segmentation.paint_samples,
        &ridge_analysis,
    );
    let refined_structural_exclusion = dilate(
        &refined_structural_ownership,
        segmentation.width,
        segmentation.height,
        1,
    );
    for (sample, &structural) in segmentation
        .paint_samples
        .iter_mut()
        .zip(&refined_structural_exclusion)
    {
        if structural {
            *sample = false;
        }
    }
    let strong_branches =
        crate::ridge::strong_branches_from_analysis(&segmentation.canonical, &ridge_analysis);
    drop(ridge_analysis);
    save_mask_diagnostic(
        "fitted-paint-samples",
        &segmentation.paint_samples,
        processing.width,
        processing.height,
    );
    // Recover continuous source fields before quantizer bands are subdivided.
    // A validated field keeps one paint owner; only unresolved faces need
    // local patches and the more expensive per-face model search.
    let coherent_hints = crate::gradient::reconstruct_coherent_domains(
        &paint_reference,
        &processing,
        &mut segmentation,
        config,
    );
    let protected = coherent_hints
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let parents = crate::segment::split_adaptive_paint_patches_with_protected(
        &paint_reference,
        &processing,
        &mut segmentation,
        &protected,
    );
    let coherent_hints = parents
        .iter()
        .map(|&i| coherent_hints[i].clone())
        .collect::<Vec<_>>();
    save_label_diagnostic(
        "final-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(
        config,
        "paint-topology-preservation",
        started,
        &mut checkpoint,
    );
    let (mut paints, mut gradient_report, paint_evidence) = fit_all_without_topology(
        &coherent_hints,
        &paint_reference,
        &processing,
        &segmentation,
        &strong_branches,
        config,
    );
    report_progress(config, "paint-fitting", started, &mut checkpoint);
    let supported_paint_merges = merge_source_supported_paints_with_evidence(
        &paint_reference,
        &processing,
        &mut segmentation,
        &mut paints,
        config,
        paint_evidence,
    );
    report_progress(
        config,
        "source-supported-paint-merge",
        started,
        &mut checkpoint,
    );
    let mut exact_paint_merges =
        merge_exact_final_paints(&paint_reference, &mut segmentation, &mut paints);
    report_progress(config, "exact-paint-merge", started, &mut checkpoint);
    let simplified_layers =
        crate::gradient::simplify_layered_paints(&paint_reference, &segmentation, &mut paints);
    report_progress(
        config,
        "layered-paint-simplification",
        started,
        &mut checkpoint,
    );
    // Simplification validates its replacement against the source above.
    // Check the slope support of any remaining layered fields here.
    let refined_shapes =
        crate::gradient::refine_residual_shapes(&paint_reference, &segmentation, &mut paints);
    report_progress(
        config,
        "residual-shape-refinement",
        started,
        &mut checkpoint,
    );
    let restored_shading =
        crate::gradient::restore_interior_shading(&paint_reference, &segmentation, &mut paints);
    report_progress(config, "interior-shading", started, &mut checkpoint);
    // Refit fields can become identical only after residual simplification.
    // Merge their ownership before generating any new shared contours.
    if simplified_layers > 0 || refined_shapes > 0 || restored_shading > 0 {
        exact_paint_merges +=
            merge_exact_final_paints(&paint_reference, &mut segmentation, &mut paints);
    }
    if supported_paint_merges.merges > 0
        || exact_paint_merges > 0
        || simplified_layers > 0
        || refined_shapes > 0
        || restored_shading > 0
    {
        refresh_summary(&mut gradient_report, &paints);
    }

    if source_alpha
        && crate::alpha_paint::consolidate(
            chroma_matte.unwrap(),
            &processing,
            &segmentation.labels,
            &mut paints,
        ) > 0
    {
        merge_exact_final_paints(&paint_reference, &mut segmentation, &mut paints);
        refresh_summary(&mut gradient_report, &paints);
    }

    let mut face_alpha = source_alpha.then(|| {
        crate::face_alpha::prepare(
            &processing,
            chroma_matte.unwrap(),
            coverage.as_ref(),
            local_coverage.as_ref(),
            &mut segmentation,
            &mut paints,
        )
    });
    if let Some(alpha) = &mut face_alpha {
        alpha.composite_layers = composite_layers;
        alpha.source_fields = source_fields;
    }
    refresh_summary(&mut gradient_report, &paints);
    save_label_diagnostic(
        "paint-merged-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        save_pipeline_diagnostic(
            "fitted-paint-fields",
            &Raster::new(
                segmentation.width,
                segmentation.height,
                segmentation
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(i, &label)| {
                        crate::gradient::paint_at(&paints[label as usize], i, segmentation.width)
                    })
                    .collect(),
            ),
        );
        let _ = fs::write(
            format!("{prefix}-fitted-paints.txt"),
            format!("{paints:#?}"),
        );
    }
    gradient_report.source_supported_paint_merges = supported_paint_merges.merges;
    gradient_report.source_supported_boundary_edges_removed =
        supported_paint_merges.boundary_edges_removed;
    let mut excluded_regions = if let Some(alpha) = &face_alpha {
        alpha
            .fields
            .iter()
            .map(|p| matches!(p, Paint::Solid { color } if color[0] <= 0.0))
            .collect()
    } else {
        chroma_matte
            .map(|matte| chroma::background_regions(&segmentation.labels, paints.len(), matte))
            .unwrap_or_default()
    };
    let removed_background_regions = excluded_regions.iter().filter(|&&removed| removed).count();
    report_progress(config, "face-alpha", started, &mut checkpoint);
    let topology = HierarchicalTopology::build(&segmentation);
    report_progress(config, "hierarchical-topology", started, &mut checkpoint);
    let overlap_opaque: Vec<bool> = (0..paints.len())
        .map(|i| {
            !excluded_regions.get(i).copied().unwrap_or(false)
                && face_alpha.as_ref().is_none_or(
                    |alpha| matches!(&alpha.fields[i], Paint::Solid { color } if color[0] >= 0.99),
                )
        })
        .collect();
    let baseline_order = crate::geometry::paint_order_ranks(&segmentation);
    let mut order_proposal = crate::paint_order::propose(
        &processing,
        &segmentation,
        &structural_candidates.source_line_mask,
        &overlap_opaque,
        &baseline_order,
    );
    // Finalize ordering before deciding and fitting hidden overlap.
    let prepared_geometry = crate::geometry::PreparedGeometry::new(
        &segmentation,
        Some(&topology),
        Some(&geometry_edge_reference),
        if source_alpha { chroma_matte } else { None },
        if variable_opacity {
            &excluded_regions
        } else {
            &[]
        },
    );
    let (mut geometry, mut geometry_report) = prepared_geometry.build(
        config.shared_boundary_overlap,
        &overlap_opaque,
        (order_proposal.summary.changed_ranks > 0).then_some(order_proposal.ranks.as_slice()),
    );
    let prepared_geometry = (order_proposal.summary.changed_ranks > 0).then_some(prepared_geometry);
    if let Some(alpha) = &mut face_alpha {
        for (band, colour) in std::mem::take(&mut alpha.bands) {
            let region = paints.len() as u32;
            paints.push(Paint::Solid { color: colour });
            alpha.fields.push(Paint::Solid {
                color: [band.opacity; 3],
            });
            excluded_regions.push(false);
            geometry.push(crate::geometry::RegionGeometry {
                region,
                loops: vec![],
                path_data: band.path_data,
                occlusion_path_data: None,
                covered_hole_paths: Vec::new(),
                primitive: None,
            });
        }
    }
    report_progress(config, "shared-geometry", started, &mut checkpoint);
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        for (name, overlap) in [
            ("paint-no-overlap", 0.0),
            ("paint-overlap", config.shared_boundary_overlap),
        ] {
            let (document, _) = serialize_svg(
                processing.width,
                processing.height,
                &geometry,
                &paints,
                &StructuralInk::empty(),
                overlap,
                false,
                &excluded_regions,
                face_alpha.as_ref(),
                &mut GeometryCache::default(),
            );
            let _ = fs::write(format!("{prefix}-{name}.svg"), document);
        }
    }
    // Resolve source ownership against the exact shared Paint partition.
    // Overlap is deliberately absent here: it seals seams in the final fill
    // contours and is not an authored Paint or structural owner. Native alpha is absent from this
    // comparison: both references must describe straight RGB. A preview
    // backdrop or alpha contour must not become a candidate ink colour.
    let mut geometry_cache = GeometryCache::default();
    let paint_render = render_svg_preview(
        (processing.width, processing.height),
        (&geometry, &paints),
        &StructuralInk::empty(),
        0.0,
        false,
        if source_alpha { &[] } else { &excluded_regions },
        if source_alpha {
            None
        } else {
            face_alpha.as_ref()
        },
        if source_alpha {
            [1.0; 3]
        } else {
            preview_background
        },
        &mut geometry_cache,
    )?;
    report_progress(config, "paint-preview", started, &mut checkpoint);
    let mut structural_checkpoint = Instant::now();
    let mut report_structural = |name: &str| {
        if cfg!(feature = "diagnostics") && config.retain_diagnostics {
            eprintln!(
                "picvec structural substage {name}: {:.3}s",
                structural_checkpoint.elapsed().as_secs_f64()
            );
        }
        structural_checkpoint = Instant::now();
    };
    let optimization = optimization_summary(&geometry, &paints, &geometry_report);
    let mut ownership = resolve_boundary_ownership(
        if source_alpha {
            &processing
        } else {
            &processing_reference
        },
        &paint_render,
        &structural_candidates,
        &geometry_report.paint_junctions,
        config.shared_boundary_overlap,
    );
    ownership.structural.retain_missing_from_ellipse_paint(
        &paint_render,
        &geometry_report.paint_ellipse_contours,
        &structural_candidates.paint_ownership_mask,
    );
    if source_alpha {
        ownership.structural.retain_source_alpha_supported_strokes(
            &processing,
            chroma_matte.expect("source alpha requires a matte"),
        );
        if crate::face_alpha::uniform_opacity(chroma_matte.unwrap(), coverage.as_ref()).is_some()
            && coverage
                .as_ref()
                .and_then(|c| c.exterior_colour(&processing, chroma_matte.unwrap()))
                .is_none()
        {
            ownership.structural.recover_alpha_boundary(
                &processing,
                chroma_matte.expect("source alpha requires a matte"),
            );
        } else if let Some(local) = &local_coverage {
            ownership.structural.recover_local_coverage_boundary(
                &processing,
                chroma_matte.unwrap(),
                local,
            );
        }
    }
    if let Some(matte) = chroma_matte.filter(|_| !source_alpha) {
        ownership
            .structural
            .retain_strokes(|stroke| matte.retains_stroke(&stroke.points));
    }
    ownership
        .structural
        .refine_interrupted_strokes(&processing, chroma_matte.filter(|_| source_alpha));
    report_structural("ownership");
    let outlines = if source_alpha {
        Vec::new()
    } else {
        crate::outline::propose(
            &processing,
            chroma_matte.filter(|_| source_alpha),
            &segmentation,
            &geometry_report.paint_closed_contours,
        )
    };
    report_structural("outline-proposals");
    if !outlines.is_empty() {
        let before = render_svg_preview(
            (processing.width, processing.height),
            (&geometry, &paints),
            &ownership.structural,
            ownership.paint_overlap,
            excluded_regions.iter().all(|&excluded| !excluded),
            &excluded_regions,
            face_alpha.as_ref(),
            preview_background,
            &mut geometry_cache,
        )?;
        ownership.structural.outlines = outlines;
        let after = render_svg_preview(
            (processing.width, processing.height),
            (&geometry, &paints),
            &ownership.structural,
            ownership.paint_overlap,
            excluded_regions.iter().all(|&excluded| !excluded),
            &excluded_regions,
            face_alpha.as_ref(),
            preview_background,
            &mut geometry_cache,
        )?;
        ownership
            .structural
            .outlines
            .retain(|band| band.supported_by_render(&processing_reference, &before, &after));
    }
    report_structural("outline-validation");
    if !source_alpha && !ownership.structural.strokes.is_empty() {
        let mut preview = |ink: &StructuralInk| {
            render_svg_preview(
                (processing.width, processing.height),
                (&geometry, &paints),
                ink,
                ownership.paint_overlap,
                excluded_regions.iter().all(|&excluded| !excluded),
                &excluded_regions,
                face_alpha.as_ref(),
                preview_background,
                &mut geometry_cache,
            )
        };
        let before = preview(&ownership.structural)?;
        let mut probe = ownership.structural.clone();
        for stroke in &mut probe.strokes {
            stroke.color = [1.0; 3];
        }
        let white = preview(&probe)?;
        for stroke in &mut probe.strokes {
            stroke.color = [0.0; 3];
        }
        let black = preview(&probe)?;
        ownership.structural.color_patches = crate::ink_color::propose(
            &processing_reference,
            &paint_render,
            &before,
            &white,
            &black,
        );
        if !ownership.structural.color_patches.is_empty() {
            let after = preview(&ownership.structural)?;
            ownership
                .structural
                .color_patches
                .retain(|patch| patch.improves(&processing_reference, &before, &after));
        }
    }
    if source_alpha {
        ownership.structural.outlines.clear();
        ownership.structural.color_patches.clear();
        if face_alpha.as_ref().is_some_and(|a| a.ink_opacity <= 0.0) {
            ownership.structural.strokes.clear();
        }
    }
    report_structural("colour-patches");
    ownership.summary.structural_strokes = ownership.structural.strokes.len();
    report_progress(config, "structural-selection", started, &mut checkpoint);
    let ownership_summary = ownership.summary.clone();
    let paint_overlap = ownership.paint_overlap;
    let structural = ownership.structural;
    let (mut document, mut svg_report) = serialize_svg(
        processing.width,
        processing.height,
        &geometry,
        &paints,
        &structural,
        paint_overlap,
        excluded_regions.iter().all(|&excluded| !excluded),
        &excluded_regions,
        face_alpha.as_ref(),
        &mut geometry_cache,
    );
    if order_proposal.summary.changed_ranks > 0 {
        // The old ordering is only a validation reference, never an input to
        // the ordered geometry's expansion decisions.
        let (mut reference_geometry, reference_geometry_report) = prepared_geometry
            .as_ref()
            .unwrap()
            .build(config.shared_boundary_overlap, &overlap_opaque, None);
        reference_geometry.extend(
            geometry
                .iter()
                .filter(|g| g.region as usize >= segmentation.regions.len())
                .cloned(),
        );
        let (reference, report) = serialize_svg(
            processing.width,
            processing.height,
            &reference_geometry,
            &paints,
            &structural,
            paint_overlap,
            excluded_regions.iter().all(|&excluded| !excluded),
            &excluded_regions,
            face_alpha.as_ref(),
            &mut geometry_cache,
        );
        #[cfg(feature = "diagnostics")]
        if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
            let _ = fs::write(format!("{prefix}-order-candidate.svg"), &document);
            let _ = fs::write(format!("{prefix}-order-baseline.svg"), &reference);
        }
        if !crate::paint_order::validate(
            &reference,
            &document,
            &processing,
            chroma_matte,
            &segmentation.labels,
            &mut order_proposal.summary,
        ) {
            document = reference;
            svg_report = report;
            geometry = reference_geometry;
            geometry_report = reference_geometry_report;
        }
        report_progress(config, "paint-order-validation", started, &mut checkpoint);
    }
    drop(prepared_geometry);
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let _ = fs::write(
            format!("{prefix}-paint-order.json"),
            serde_json::to_vec(&serde_json::json!({
                "baseline_ranks": baseline_order, "proposed_ranks": order_proposal.ranks,
                "line_regions": order_proposal.lines, "evidence": order_proposal.evidence, "summary": order_proposal.summary
            }))
            .unwrap(),
        );
    }
    if paint_overlap > 0.0 && geometry.iter().any(|g| !g.covered_hole_paths.is_empty()) {
        // Use the same authored-opacity field as SVG fills. The coverage model
        // has already classified isolated 253/254 samples as raster coverage,
        // so reusing raw alpha here would contradict that earlier decision.
        let material_alpha = face_alpha.as_ref().map(|alpha| {
            AlphaMatte::from_u8(
                processing.width,
                processing.height,
                segmentation
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(i, &label)| {
                        (crate::gradient::paint_at(
                            &alpha.fields[label as usize],
                            i,
                            processing.width,
                        )[0] * 255.0)
                            .round()
                            .clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            )
        });
        let (filled, report, removed) = crate::occlusion::simplify(
            &mut geometry,
            (document, svg_report),
            &segmentation.labels,
            processing.width,
            material_alpha.as_ref().or(chroma_matte),
            crate::svg::hole_serializer(
                processing.width,
                processing.height,
                &paints,
                &structural,
                paint_overlap,
                &excluded_regions,
                face_alpha.as_ref(),
                &mut geometry_cache,
            ),
        );
        document = filled;
        svg_report = report;
        geometry_report.covered_holes_removed += removed;
        report_progress(
            config,
            "covered-hole-simplification",
            started,
            &mut checkpoint,
        );
    }
    drop(geometry_cache);
    // Use a neutral comparison backing; the chroma diagnostic backing is
    // deliberately saturated and must not veto grayscale source evidence.
    let soft_reference = if source_alpha {
        chroma::composite_over(&processing, chroma_matte.unwrap(), [1.0; 3])
    } else {
        processing.clone()
    };
    let document = crate::soft_edges::refine(
        &document,
        &soft_reference,
        chroma_matte.filter(|_| source_alpha),
        |document| render_svg_document_on(document, processing.width, processing.height, [1.0; 3]),
    )?;
    let (document, removed) =
        crate::visibility::prune(&document, processing.width, processing.height);
    svg_report.path_elements = svg_report.path_elements.saturating_sub(removed.paths);
    svg_report.rect_elements = svg_report.rect_elements.saturating_sub(removed.rects);
    svg_report.circle_elements = svg_report.circle_elements.saturating_sub(removed.circles);
    svg_report.ellipse_elements = svg_report.ellipse_elements.saturating_sub(removed.ellipses);
    svg_report.line_elements = svg_report.line_elements.saturating_sub(removed.lines);
    svg_report.invisible_elements_removed = removed.shapes;
    svg_report.invisible_strokes_removed = removed.strokes;
    svg_report.structural_strokes = svg_report.structural_strokes.saturating_sub(removed.ink);
    svg_report.bytes = document.len();
    #[cfg(feature = "diagnostics")]
    let quality = if config.compute_quality_metrics {
        let final_render = render_svg_document_on(
            &document,
            processing.width,
            processing.height,
            preview_background,
        )?;
        Some(crate::metrics::compare(
            &processing_reference,
            &final_render,
        ))
    } else {
        None
    };

    #[cfg(not(feature = "diagnostics"))]
    let quality = None;

    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let _ = fs::write(format!("{prefix}-final.svg"), &document);
    }
    report_progress(config, "final-svg", started, &mut checkpoint);
    Ok(CoreVectorization {
        source_alpha_bits: if source_alpha {
            8
        } else {
            chroma::SOURCE_ALPHA_QUANTIZATION_BITS
        },
        document,
        processing_reference,
        labels: segmentation.labels,
        removed_background_regions,
        preview_background,
        hierarchical_topology: topology.summary,
        edge_roles: roles.summary,
        segmentation: segmentation.summary,
        structural: structural.summary,
        ownership: ownership_summary,
        paint_order: order_proposal.summary,
        gradients: gradient_report,
        geometry: geometry_report,
        optimization,
        svg: svg_report,
        quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::fitted_alpha_contour_path_data;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    #[ignore = "source-resolution scanned drawing regression; run explicitly"]
    fn scanned_annotations_retain_ink_through_the_common_pipeline() {
        let image = image::load_from_memory(include_bytes!("test-data/booster-annotations.png"))
            .unwrap()
            .to_rgb8();
        let source = Raster::new(
            image.width() as usize,
            image.height() as usize,
            image
                .pixels()
                .map(|p| p.0.map(|c| c as f32 / 255.0))
                .collect(),
        );
        let config = Config {
            adaptive_refinement: false,
            ..Config::default()
        };
        let core = vectorize_processing(source.clone(), None, false, [1.0; 3], &config).unwrap();
        assert!(!core.labels.is_empty());
        let rendered =
            render_svg_document_on(&core.document, source.width, source.height, [1.0; 3]).unwrap();
        let (mut ink_error, mut ink, mut dark, mut missing) = (0.0, 0usize, 0usize, 0usize);
        for (a, b) in source.pixels.iter().zip(&rendered.pixels) {
            if a[0] < 240.0 / 255.0 {
                ink += 1;
                ink_error += (a[0] - b[0]).abs();
            }
            if a[0] < 128.0 / 255.0 {
                dark += 1;
                missing += usize::from(b[0] > 192.0 / 255.0);
            }
        }
        assert!(
            ink_error / (ink.max(1) as f32) < 0.2,
            "ink error {}",
            ink_error / ink.max(1) as f32
        );
        assert!(
            (missing as f32) / (dark.max(1) as f32) < 0.12,
            "missing {missing}/{dark}"
        );
    }

    #[test]
    fn neutral_and_coloured_parallel_lines_use_normal_region_geometry() {
        let config = Config {
            adaptive_refinement: false,
            ..Config::default()
        };
        for ink in [[0.0; 3], [0.6, 0.0, 0.0]] {
            let mut source = Raster::blank(128, 64, [1.0; 3]);
            for y in [20, 28] {
                for yy in y..y + 3 {
                    for x in 12..116 {
                        source.pixels[yy * 128 + x] = ink;
                    }
                }
            }
            let core = vectorize_processing(source, None, false, [1.0; 3], &config).unwrap();
            assert!(!core.labels.is_empty());
            assert!(!core.document.contains("<mask") && !core.document.contains("<image"));
            let rendered = render_svg_document_on(&core.document, 128, 64, [1.0; 3]).unwrap();
            for x in 20..108 {
                assert!(rendered.pixels[21 * 128 + x][1] < 0.3);
                assert!(rendered.pixels[29 * 128 + x][1] < 0.3);
                assert!(rendered.pixels[25 * 128 + x][1] > 0.9);
            }
        }
    }

    #[test]
    fn variable_opacity_elsewhere_does_not_erase_opaque_ink() {
        let (w, h) = (88, 64);
        let mut source = Raster::blank(w, h, [1.0; 3]);
        let mut opacity = vec![1.0; w * h];
        for y in 8..56 {
            for x in 6..30 {
                source.pixels[y * w + x] = [0.1, 0.4, 0.8];
                opacity[y * w + x] = 0.3 + 0.4 * (x - 6) as f32 / 23.0;
            }
        }
        for y in 10..54 {
            for x in 63..65 {
                source.pixels[y * w + x] = [0.0; 3];
            }
        }
        let matte = AlphaMatte::new(w, h, opacity);
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                vectorize_processing(
                    source,
                    Some(&matte),
                    true,
                    [1.0; 3],
                    &Config {
                        adaptive_refinement: false,
                        ..Config::default()
                    },
                )
                .unwrap()
            });
        assert!(!result.document.contains("<mask"));
        let tree = parse_svg_document(&result.document).unwrap();
        let mut image = resvg::tiny_skia::Pixmap::new(w as u32, h as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut image.as_mut(),
        );
        for y in 14..50 {
            assert!(
                (62..66).any(|x| image.pixels()[y * w + x].red() < 50),
                "opaque ink lost at row {y}"
            );
        }
        for y in 12..52 {
            for x in 10..26 {
                let actual = image.pixels()[y * w + x];
                assert!(
                    (actual.alpha() as f32 / 255.0 - matte.get(y * w + x)).abs() < 4.0 / 255.0,
                    "authored opacity changed at {x},{y}"
                );
            }
        }
    }

    #[test]
    fn reported_remojii_top_objects_join_adjacent_faces_in_the_complete_pipeline() {
        // These are the native support pixels of objects 2..=50, top down,
        // from the user's 665770-byte SVG. Full-frame context is necessary:
        // cropping splits the incident paint owners and changes their sizes.
        #[derive(serde::Deserialize)]
        struct ReportedObject {
            rank: usize,
            pixels: Vec<usize>,
        }
        let objects: Vec<ReportedObject> =
            serde_json::from_str(include_str!("test-data/remojii-top-objects.json")).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("remojii.png");
        fs::write(
            &input,
            include_bytes!("test-data/remojii-top-objects-source.png"),
        )
        .unwrap();
        let config = Config {
            adaptive_refinement: false,
            rayon_threads: 2,
            ..Config::default()
        };
        let (decoded, alpha) = SourceRaster::load_with_alpha(
            &input,
            config.maximum_input_dimension,
            config.maximum_input_pixels,
            config.maximum_decode_bytes,
        )
        .unwrap();
        let matte = AlphaMatte::from_u8(decoded.width, decoded.height, alpha.unwrap());
        let backing = chroma::select_alpha_backing(&decoded, &matte);
        let source = chroma::prepare_compact_source_alpha(&decoded, &matte);
        let (processing, alpha) = resize_processing(&source, Some(&matte), true, 1254);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let result = pool
            .install(|| vectorize_processing(processing, alpha.as_ref(), true, backing, &config))
            .unwrap();
        assert!(result.paint_order.line_regions > 0);
        assert!(result.paint_order.changed_ranks > 0);
        assert!(result.paint_order.accepted, "{:?}", result.paint_order);
        #[derive(serde::Deserialize)]
        struct ReportedInk {
            pixels: Vec<usize>,
            neighbor_pixel: usize,
        }
        let reported: Vec<ReportedInk> = serde_json::from_str(include_str!(
            "test-data/remojii-reported-material-regions.json"
        ))
        .unwrap();
        for (case, ink) in reported.iter().enumerate() {
            let ink_owner = result.labels[ink.neighbor_pixel];
            assert!(
                ink.pixels.iter().all(|&i| result.labels[i] == ink_owner),
                "reported material case {case} must join its adjacent face"
            );
        }
        let mut areas = std::collections::HashMap::<u32, usize>::new();
        for &label in &result.labels {
            *areas.entry(label).or_default() += 1;
        }
        // These two reported faces formerly emitted the same outline four
        // and six times for weak residual colour corrections.
        fn count_face_paths(group: &resvg::usvg::Group, x: f32, y: f32) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => count_face_paths(group, x, y),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        usize::from(
                            (b.x() - x).abs() < 2.0
                                && (b.y() - y).abs() < 2.0
                                && b.width() < 45.0
                                && b.height() < 45.0,
                        )
                    }
                    _ => 0,
                })
                .sum()
        }
        let tree = parse_svg_document(&result.document).unwrap();
        fn shell_contours(group: &resvg::usvg::Group) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => shell_contours(group),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        if (b.x() - 155.977).abs() < 2.0 && (b.y() - 63.639).abs() < 2.0 {
                            assert_eq!(
                                path.data()
                                    .segments()
                                    .filter(|s| matches!(
                                        s,
                                        resvg::tiny_skia::PathSegment::MoveTo(_)
                                    ))
                                    .count(),
                                1,
                                "covered shell pattern must not remain as holes in the lower paint"
                            );
                            1
                        } else {
                            0
                        }
                    }
                    _ => 0,
                })
                .sum()
        }
        assert_eq!(shell_contours(tree.root()), 1);
        assert!(result.geometry.covered_holes_removed > 0);

        fn shadow_copies(
            group: &resvg::usvg::Group,
            x: f32,
            y: f32,
            width: f32,
            height: f32,
        ) -> usize {
            group
                .children()
                .iter()
                .map(|node| match node {
                    resvg::usvg::Node::Group(group) => shadow_copies(group, x, y, width, height),
                    resvg::usvg::Node::Path(path) if path.fill().is_some() => {
                        let b = path.abs_bounding_box();
                        usize::from(
                            (b.x() - x).abs() < 2.0
                                && (b.y() - y).abs() < 2.0
                                && b.width() < width
                                && b.height() < height,
                        )
                    }
                    _ => 0,
                })
                .sum()
        }
        assert_eq!(
            shadow_copies(tree.root(), 368.57, 994.89, 65.0, 80.0),
            1,
            "the book shadow must have one paint, not repeated residual geometry"
        );

        assert_eq!(
            shadow_copies(tree.root(), 733.65, 639.28, 120.0, 60.0),
            1,
            "the image-right eyelid shadow must have one paint"
        );

        fn assert_paint_has_no_auxiliary_strokes(group: &resvg::usvg::Group) {
            for node in group.children() {
                match node {
                    resvg::usvg::Node::Group(group) => assert_paint_has_no_auxiliary_strokes(group),
                    resvg::usvg::Node::Path(path) => assert!(
                        path.stroke().is_none(),
                        "paint face emitted an auxiliary seam stroke"
                    ),
                    _ => {}
                }
            }
        }
        let paint_group = tree
            .root()
            .children()
            .iter()
            .find_map(|n| match n {
                resvg::usvg::Node::Group(g) if g.id() == "paint-layer" => Some(g),
                _ => None,
            })
            .unwrap();
        assert_paint_has_no_auxiliary_strokes(paint_group);
        for (x, y) in [(863.93, 407.78), (925.97, 825.06)] {
            assert_eq!(
                count_face_paths(tree.root(), x, y),
                1,
                "reported face at {x},{y} retained duplicate correction layers"
            );
        }
        let fragments: Vec<Vec<usize>> =
            serde_json::from_str(include_str!("test-data/remojii-third-ninth-regions.json"))
                .unwrap();
        for pixels in fragments {
            let owner = result.labels[pixels[0]];
            assert!(
                pixels.iter().all(|&i| result.labels[i] == owner) && areas[&owner] > pixels.len(),
                "reported third/ninth fragment retained an independent owner"
            );
        }
        let sixth: Vec<usize> =
            serde_json::from_str(include_str!("test-data/remojii-sixth-black-region.json"))
                .unwrap();
        assert!(
            sixth
                .iter()
                .all(|&i| areas[&result.labels[i]] > sixth.len()),
            "sixth black face still has an independent owner"
        );
        for object in objects {
            assert!(
                object
                    .pixels
                    .iter()
                    .all(|&i| areas[&result.labels[i]] > object.pixels.len()),
                "reported object {} still has its own small owner",
                object.rank
            );
        }
        assert!(!result.document.contains("<mask"));
        assert!(!result.document.contains("fill-opacity=\"0\""));
    }

    #[test]
    fn remojii_alpha_boundary_has_few_nodes_along_the_upper_left_rim() {
        let input = image::load_from_memory(include_bytes!("test-data/remojii-rim-alpha.png"))
            .unwrap()
            .to_luma8();
        let matte = AlphaMatte::from_u8(
            input.width() as usize,
            input.height() as usize,
            input.into_raw(),
        );
        let path = matte
            .isocontours(0.5)
            .iter()
            .map(|contour| fitted_alpha_contour_path_data(contour))
            .collect::<String>();
        let mut tokens = path.split_whitespace();
        let mut rim_nodes = 0;
        while let Some(command) = tokens.next() {
            let count = match command {
                "M" | "L" => 2,
                "C" => 6,
                "Z" => 0,
                _ => panic!("unexpected {command}"),
            };
            let values: Vec<f32> = tokens
                .by_ref()
                .take(count)
                .map(|s| s.parse().unwrap())
                .collect();
            if count > 0 {
                let (x, y) = (values[count - 2], values[count - 1]);
                rim_nodes += usize::from((80.0..420.0).contains(&x) && (50.0..350.0).contains(&y));
            }
        }
        assert!(
            rim_nodes <= 24,
            "smooth upper-left rim retained {rim_nodes} nodes"
        );
    }

    #[test]
    fn alpha_weighted_resize_preserves_visible_colour_in_base_and_crops() {
        let source = SourceRaster::from_rgb8_fn(128, 64, |i| {
            if i % 2 == 0 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            }
        });
        let matte = AlphaMatte::from_u8(
            128,
            64,
            (0..128 * 64)
                .map(|i| if i % 2 == 0 { 255 } else { 1 })
                .collect(),
        );
        let crop = source.crop(0, 0, 128, 64);
        let (base, alpha) = resize_processing(&source, Some(&matte), true, 64);
        let (child, _) = resize_processing(&crop, Some(&matte), true, 64);
        for image in [&base, &child] {
            let rgb = image.get(32, 16);
            assert!((rgb[0] - 255.0 / 256.0).abs() < 0.001, "{rgb:?}");
            assert!((rgb[2] - 1.0 / 256.0).abs() < 0.001, "{rgb:?}");
        }
        assert!((alpha.as_ref().unwrap().get(16 * 64 + 32) - 128.0 / 255.0).abs() < 0.005);
        let core =
            vectorize_processing(base, alpha.as_ref(), true, [1.0; 3], &Config::default()).unwrap();
        let tree = parse_svg_document(&core.document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let pixel = pixmap.pixels()[16 * 64 + 32];
        assert!(pixel.alpha() > 0);
        assert!(f32::from(pixel.red()) / f32::from(pixel.alpha()) > 0.98);
        assert!(f32::from(pixel.blue()) / f32::from(pixel.alpha()) < 0.02);
    }

    #[test]
    fn small_inputs_select_native_dimensions_without_a_probe() {
        let source = Raster::blank(1024, 1024, [0.5; 3]);
        let config = Config::default();
        let selection = select_dimension(&source, &config);
        assert_eq!(
            selection.selected_dimension,
            estimate_dimension(&source, &config).selected_dimension
        );
        assert_eq!(selection.probe_width, 0);
        let limited = Config {
            maximum_dimension: 64,
            ..config
        };
        assert_eq!(select_dimension(&source, &limited).selected_dimension, 64);
    }

    #[test]
    fn native_vector_join_does_not_inherit_coarse_preview_blur() {
        let document = r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M0 8H32V8.5H0Z"/><circle cx="16" cy="16" r="5"/></svg>"#;
        let tree = parse_svg_document(document).unwrap();
        let coarse = render_svg_document_on(document, 32, 32, [1.0; 3]).unwrap();
        let whole = SourceRect {
            x: 0,
            y: 0,
            width: 128,
            height: 128,
        };
        let expanded = SourceRect {
            x: 32,
            y: 28,
            width: 64,
            height: 64,
        };
        let core = SourceRect {
            x: 40,
            y: 36,
            width: 48,
            height: 52,
        };
        let native = render_svg_tree_on(
            &tree,
            64,
            64,
            resvg::tiny_skia::Transform::from_row(4.0, 0.0, 0.0, 4.0, -32.0, -28.0),
            [1.0; 3],
        )
        .unwrap();
        assert!(!crate::adaptive::refinement_boundary_matches(
            &coarse, &native, whole, whole, core, expanded
        ));
        assert!(crate::adaptive::refinement_boundary_matches(
            &native, &native, expanded, whole, core, expanded
        ));
        let mut damaged = native.clone();
        damaged.pixels[(core.y - expanded.y) * 64 + 25] = [0.0; 3];
        assert!(!crate::adaptive::refinement_boundary_matches(
            &native, &damaged, expanded, whole, core, expanded
        ));
    }

    #[test]
    fn factored_separator_stays_behind_foreground_and_leaves_background_clear() {
        let mut document = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect x="14" y="6" width="4" height="20" fill="#202020"/></svg>"##.to_owned();
        crate::separators::prepend(
            &mut document,
            &[crate::separators::Separator {
                rect: [0.0, 32.0, 128.0, 6.0],
                color: [1.0; 3],
                opacity: 1.0,
            }],
            [0.25, 0.25],
        );
        let raster = render_svg_document_on(&document, 32, 32, [0.0, 1.0, 0.0]).unwrap();
        assert_eq!(raster.get(0, 8), [1.0; 3]);
        assert_eq!(raster.get(16, 8), [32.0 / 255.0; 3]);
        assert_eq!(raster.get(0, 0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn exact_final_paint_merge_compacts_adjacent_equal_owners() {
        let source = Raster::blank(2, 1, [0.25, 0.5, 0.75]);
        let mut segmentation = Segmentation {
            width: 2,
            height: 1,
            labels: vec![0, 1],
            paint_keys: vec![0, 1],
            paint_samples: vec![true; 2],
            canonical: source.clone(),
            regions: vec![
                crate::segment::RegionStats {
                    id: 0,
                    area: 1,
                    min_x: 0,
                    min_y: 0,
                    max_x: 1,
                    max_y: 1,
                    mean_rgb: [0.25, 0.5, 0.75],
                    mean_lab: rgb_to_oklab([0.25, 0.5, 0.75]),
                },
                crate::segment::RegionStats {
                    id: 1,
                    area: 1,
                    min_x: 1,
                    min_y: 0,
                    max_x: 2,
                    max_y: 1,
                    mean_rgb: [0.25, 0.5, 0.75],
                    mean_lab: rgb_to_oklab([0.25, 0.5, 0.75]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let mut paints = vec![
            Paint::Solid {
                color: [0.25, 0.5, 0.75],
            },
            // Different fitting values, identical serialized #4080bf.
            Paint::Solid {
                color: [0.2501, 0.5001, 0.7501],
            },
        ];
        assert_eq!(
            merge_exact_final_paints(&source, &mut segmentation, &mut paints),
            1
        );
        assert_eq!(segmentation.labels, vec![0, 0]);
        assert_eq!(segmentation.regions.len(), 1);
        assert_eq!(paints.len(), 1);
    }

    #[test]
    fn complexity_probe_distinguishes_sparse_and_dense_edges() {
        let width = 128;
        let height = 128;
        let flat = Raster::blank(width, height, [0.5; 3]);
        let mut split = flat.clone();
        for y in 0..height {
            for x in width / 2..width {
                split.pixels[y * width + x] = [0.9; 3];
            }
        }
        let mut tiled = flat.clone();
        for y in 0..height {
            for x in 0..width {
                if (x / 4 + y / 4) % 2 == 0 {
                    tiled.pixels[y * width + x] = [0.9; 3];
                }
            }
        }
        let config = Config {
            maximum_dimension: 128,
            auto_minimum_dimension: 64,
            auto_maximum_dimension: 128,
            ..Config::default()
        };
        let flat_probe = estimate_dimension(&flat, &config);
        let split_probe = estimate_dimension(&split, &config);
        let tiled_probe = estimate_dimension(&tiled, &config);
        assert_eq!(flat_probe.edge_density, 0.0);
        assert!(split_probe.edge_density > flat_probe.edge_density);
        assert!(tiled_probe.edge_density > split_probe.edge_density);
        assert!(tiled_probe.complexity > split_probe.complexity);
        assert!(split_probe.complexity > flat_probe.complexity);
    }

    #[test]
    fn complexity_probe_honours_the_general_maximum_dimension() {
        let image = Raster::blank(640, 480, [0.5; 3]);
        let config = Config {
            maximum_dimension: 192,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            ..Config::default()
        };
        assert_eq!(estimate_dimension(&image, &config).selected_dimension, 192);
    }

    #[test]
    fn refinement_queue_preserves_order_and_limits_nested_jobs() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for limit in [1, 2, 4, 8, 10] {
                    let active = AtomicUsize::new(0);
                    let peak = AtomicUsize::new(0);
                    let tasks: Vec<_> = (0usize..37).collect();
                    let outcomes = bounded_map(&tasks, limit, |&i| {
                        let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        let sum: usize = (0..1024).into_par_iter().map(|j| i + j).sum();
                        active.fetch_sub(1, Ordering::SeqCst);
                        sum
                    });
                    assert_eq!(
                        outcomes,
                        tasks
                            .iter()
                            .map(|i| i * 1024 + 1023 * 1024 / 2)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(active.load(Ordering::SeqCst), 0);
                    assert!(peak.load(Ordering::SeqCst) <= limit);
                }
            });
        }
    }

    #[test]
    fn default_worker_count_uses_half_the_cpus_capped_at_ten() {
        assert_eq!(default_execution_thread_count(1), 1);
        assert_eq!(default_execution_thread_count(2), 1);
        assert_eq!(default_execution_thread_count(3), 1);
        assert_eq!(default_execution_thread_count(4), 2);
        assert_eq!(default_execution_thread_count(6), 3);
        assert_eq!(default_execution_thread_count(8), 4);
        assert_eq!(default_execution_thread_count(20), 10);
        assert_eq!(default_execution_thread_count(64), 10);
    }

    #[test]
    fn conversion_writes_only_the_named_svg() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("picvec-contract-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("chosen-name.svg");
        let mut raster = Raster::blank(32, 24, [0.92, 0.92, 0.96]);
        for y in 5..19 {
            for x in 6..26 {
                let amount = (x - 6) as f32 / 19.0;
                raster.pixels[y * 32 + x] = [0.15 + 0.65 * amount, 0.12, 0.72 - 0.4 * amount];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.quality.is_none());
        assert!(!summary.source_alpha.detected);
        assert!(!summary.chroma_key.enabled);
        assert!(summary.adaptive_refinement.enabled);
        assert_eq!(summary.adaptive_refinement.accepted_regions, 0);
        assert_eq!(summary.adaptive_refinement.source_scale, 1.0);
        let files: HashSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files,
            HashSet::from(["input.png".to_string(), "chosen-name.svg".to_string()])
        );
        let document = fs::read_to_string(&output).unwrap();
        assert!(document.starts_with("<?xml"));
        assert!(document.contains("<svg"));
        assert!(!document.contains("silhouette\""));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn chroma_key_removes_outer_and_enclosed_background_but_keeps_white_subject() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut raster = Raster::blank(64, 64, [0.0, 1.0, 0.0]);
        // An opaque white foreground detail verifies that white is not
        // confused with the keyed background.
        for y in 3..13 {
            for x in 3..13 {
                raster.pixels[y * 64 + x] = [1.0; 3];
            }
        }
        // Red ring around a disconnected island of keyed background.
        for y in 14..50 {
            for x in 14..50 {
                raster.pixels[y * 64 + x] = [1.0, 0.0, 0.0];
            }
        }
        // One-pixel 50% coverage shoulder, as produced by raster
        // antialiasing of red over the green backing.
        for position in 14..50 {
            raster.pixels[14 * 64 + position] = [0.5, 0.5, 0.0];
            raster.pixels[49 * 64 + position] = [0.5, 0.5, 0.0];
            raster.pixels[position * 64 + 14] = [0.5, 0.5, 0.0];
            raster.pixels[position * 64 + 49] = [0.5, 0.5, 0.0];
        }
        for y in 26..38 {
            for x in 26..38 {
                raster.pixels[y * 64 + x] = [0.0, 1.0, 0.0];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                remove_chroma_key_background: true,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.chroma_key.enabled);
        assert!(summary.chroma_key.detected);
        assert_eq!(summary.chroma_key.key_color, Some([0, 255, 0]));
        assert!(summary.chroma_key.removed_regions >= 2);

        let document = fs::read_to_string(&output).unwrap();
        for (_, suffix) in document.match_indices('#') {
            let Some(hex) = suffix.get(1..7) else {
                continue;
            };
            let Ok(color) = u32::from_str_radix(hex, 16) else {
                continue;
            };
            let red = (color >> 16) & 0xff;
            let green = (color >> 8) & 0xff;
            let blue = color & 0xff;
            assert!(
                green < 200 || red >= 80 || blue >= 80,
                "key-coloured antialias paint leaked into SVG: #{hex}"
            );
        }
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let alpha = |x: usize, y: usize| pixmap.pixels()[y * 64 + x].alpha();
        assert_eq!(alpha(2, 2), 0, "outer key should be transparent");
        assert!(
            alpha(8, 8) > 240,
            "white foreground should remain opaque: {}",
            alpha(8, 8)
        );
        assert!(alpha(18, 18) > 240, "red foreground should remain opaque");
        assert_eq!(alpha(32, 32), 0, "enclosed key should be transparent");

        let opaque_output = directory.path().join("opaque.svg");
        let opaque_summary = vectorize(
            &input,
            &opaque_output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(!opaque_summary.source_alpha.detected);
        assert!(!opaque_summary.chroma_key.enabled);
        let opaque_document = fs::read_to_string(&opaque_output).unwrap();
        let opaque_tree = parse_svg_document(&opaque_document).unwrap();
        let mut opaque_pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &opaque_tree,
            resvg::tiny_skia::Transform::identity(),
            &mut opaque_pixmap.as_mut(),
        );
        assert!(
            opaque_pixmap.pixels()[2 * 64 + 2].alpha() > 240,
            "opaque chroma input must remain opaque without the option"
        );
    }

    #[test]
    fn source_alpha_is_removed_without_the_chroma_option() {
        use image::{ImageBuffer, Rgba};

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut image = ImageBuffer::from_pixel(64, 64, Rgba([17_u8, 31, 47, 0]));
        // Opaque black and white details must both survive regardless of the
        // temporary saturated backing selected for RGB vectorization.
        for y in 4..14 {
            for x in 4..14 {
                image.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        for y in 16..50 {
            for x in 16..50 {
                image.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        for y in 27..39 {
            for x in 27..39 {
                image.put_pixel(x, y, Rgba([17, 31, 47, 0]));
            }
        }
        image.save(&input).unwrap();

        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.source_alpha.detected);
        assert!(summary.source_alpha.temporary_backing_color.is_some());
        assert_eq!(summary.source_alpha.quantization_bits, 8);
        assert_eq!(summary.source_alpha.mask_paths, 0);
        assert!(summary.source_alpha.removed_regions >= 2);
        assert!(!summary.chroma_key.enabled);

        let document = fs::read_to_string(&output).unwrap();
        assert!(!document.contains("source-alpha-clip"));
        assert!(!document.contains("source-alpha-mask"));
        assert!(!document.contains("<mask"));
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let alpha = |x: usize, y: usize| pixmap.pixels()[y * 64 + x].alpha();
        assert_eq!(alpha(2, 2), 0);
        assert!(alpha(8, 8) > 240, "opaque white should remain");
        assert!(alpha(20, 20) > 240, "opaque black should remain");
        assert_eq!(
            alpha(32, 32),
            0,
            "enclosed source alpha should remain clear"
        );
    }

    #[test]
    fn shaded_shoulder_has_no_spurs_at_paint_junctions() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("shoulder.png");
        let output = directory.path().join("shoulder.svg");
        fs::write(&input, include_bytes!("test-data/shoulder-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 802,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2508,
            3208,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        let mut positions = Vec::new();
        for x in 195..239 {
            let values: Vec<_> = (345 * 4..385 * 4)
                .map(|y| rendered.get(x * 4, y)[1])
                .collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len()).find(|&i| values[i] > 0.4).unwrap();
            let t = (0.4 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push((crossing as f32 - 1.0 + t) * 0.25);
        }
        let curvature: Vec<_> = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .collect();
        assert!(
            curvature.iter().all(|&v| v < 0.6),
            "rim spikes: {curvature:?}"
        );
        assert!(curvature.iter().sum::<f32>() / (curvature.len() as f32) < 0.12);
    }

    #[test]
    fn round_window_control_does_not_reintroduce_polygonal_outer_ink() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("window.png");
        let output = directory.path().join("window.svg");
        fs::write(&input, include_bytes!("test-data/round-window-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 705,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2820,
            2752,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // These exterior samples were dark protrusions of the polygonal rim.
        // Check serialized Paint plus residual strokes, not just ellipse fits.
        for (x, y) in [(135, 100), (170, 116), (174, 112)] {
            let p = rendered.get(x * 4, y * 4);
            assert!(p[2] > 0.5 && p[0] < 0.2, "outer rim at {x},{y}: {p:?}");
        }
        let fill = rendered.get(157 * 4, 98 * 4);
        assert!(fill[0] > 0.9 && fill[1] > 0.5 && fill[2] < 0.2);
    }

    #[test]
    fn third_round_button_has_a_smooth_inner_rim_in_the_final_svg() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("buttons.png");
        let output = directory.path().join("buttons.svg");
        fs::write(&input, include_bytes!("test-data/round-buttons-source.png")).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 718,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.svg.outline_bands >= 3);
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            2872,
            2764,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // Replacing a band's interior must retain the original exterior
        // coverage. This blue backdrop sample previously became a white gap.
        let backdrop = rendered.get(794, 348);
        assert!(
            backdrop[0] < 0.5 && backdrop[2] < 0.85,
            "missing band underpaint: {backdrop:?}"
        );
        let mut positions = Vec::new();
        for x in 208..233 {
            let values: Vec<_> = (68 * 4..98 * 4)
                .map(|y| rendered.get(x * 4, y)[1])
                .collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len())
                .find(|&i| values[i] > 0.55)
                .unwrap();
            let t = (0.55 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push(68.0 + (crossing as f32 - 1.0 + t) * 0.25);
        }
        let curvature: Vec<_> = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .collect();
        assert!(curvature.iter().sum::<f32>() / (curvature.len() as f32) < 0.12);
        assert!(curvature.iter().all(|&v| v < 0.4));
    }

    #[test]
    fn pie_highlight_does_not_acquire_an_orange_notch() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("pie.png");
        let output = directory.path().join("pie.svg");
        fs::write(&input, include_bytes!("test-data/pie-highlight-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 710,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let rendered = render_svg_document_on(&document, 564, 710, [1.0; 3]).unwrap();
        // These bright native samples were assigned to the orange interior
        // during antialias cleanup. Check the final emitted image, not just
        // the intermediate partition or the number of simplified contours.
        for y in [462, 463] {
            let pixel = rendered.get(114, y);
            assert!(
                pixel[1] > 0.8 && pixel[2] > 0.6,
                "highlight overwritten at (114,{y}): {pixel:?}"
            );
        }
    }

    #[test]
    fn wifi_inner_rim_is_visibly_smooth_after_final_serialization() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("wifi.png");
        let output = directory.path().join("wifi.svg");
        fs::write(&input, include_bytes!("test-data/wifi-source.png")).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 310,
                auto_dimension: false,
                adaptive_refinement: false,
                remove_chroma_key_background: true,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        // Seam underpaint can make an extra outline band unnecessary. Check
        // the rendered rim's colour, contrast and roughness below instead of
        // requiring a particular number of reconstruction primitives.
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let rendered = render_svg_tree_on(
            &tree,
            1240,
            1060,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        // Locate the rim in the source after removing its green key. A fixed
        // y=203 dark-colour assertion instead darkens the source's cyan fill
        // when a different perceptual partition moves the fitted boundary.
        let source_image = Raster::from_dynamic(
            &image::load_from_memory(include_bytes!("test-data/wifi-source.png")).unwrap(),
        );
        let key = chroma::detect(&source_image).unwrap();
        let matte = chroma::pull_matte(&source_image, key);
        let foreground = chroma::separate_foreground(&source_image, &matte, key.sampled);
        let reference = chroma::composite_over(&foreground, &matte, [1.0; 3]);
        let luminance = |p: [f32; 3]| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722;
        let mut rim_colour_error = 0.0;
        for x in 148..=154 {
            let y = (200..=204)
                .min_by(|&a, &b| {
                    luminance(reference.get(x, a)).total_cmp(&luminance(reference.get(x, b)))
                })
                .unwrap();
            let expected = reference.get(x, y);
            // Allow at most one source pixel of subpixel boundary movement.
            let colour_error =
                |pixel| crate::color::delta_e_ok(rgb_to_oklab(expected), rgb_to_oklab(pixel));
            let actual = ((y - 1) * 4..=(y + 1) * 4)
                .map(|row| rendered.get(x * 4, row))
                .min_by(|a, b| colour_error(*a).total_cmp(&colour_error(*b)))
                .unwrap();
            let error = colour_error(actual);
            assert!(
                error < 12.0,
                "source rim colour at {x},{y}: {error}, {actual:?}"
            );
            rim_colour_error += error;
            let source_contrast = luminance(reference.get(x, 205)) - luminance(expected);
            // Colour matching may select a partially covered rim pixel. Use
            // the actual darkest core, not that antialias sample, for contrast.
            let core_luminance = ((y - 1) * 4..=(y + 1) * 4)
                .map(|row| luminance(rendered.get(x * 4, row)))
                .fold(f32::INFINITY, f32::min);
            let rendered_contrast = luminance(rendered.get(x * 4, 205 * 4)) - core_luminance;
            assert!(
                rendered_contrast >= source_contrast * 0.75,
                "dot rim contrast at {x}: {rendered_contrast} vs source {source_contrast}"
            );
        }
        assert!(
            rim_colour_error / 7.0 < 6.0,
            "mean rim colour error: {}",
            rim_colour_error / 7.0
        );
        // Track the visible inner edge of the upper side of the middle arc.
        // The previous final SVG measured 0.093 px of second-difference
        // roughness here at 4x; checking master counts did not detect it.
        let mut positions = Vec::new();
        for x in 95..206 {
            let expected = 205.0 - (130.0_f32.powi(2) - (x as f32 - 150.0).powi(2)).sqrt();
            let a = ((expected - 12.0) * 4.0) as usize;
            let b = ((expected + 12.0) * 4.0) as usize;
            let values: Vec<_> = (a..b).map(|y| luminance(rendered.get(x * 4, y))).collect();
            let core = (0..values.len())
                .min_by(|&a, &b| values[a].total_cmp(&values[b]))
                .unwrap();
            let crossing = (core + 1..values.len())
                .find(|&i| values[i] > 0.42)
                .expect("unbroken visible rim");
            let t = (0.42 - values[crossing - 1]) / (values[crossing] - values[crossing - 1]);
            positions.push((a as f32 + crossing as f32 - 1.0 + t) * 0.25);
        }
        let roughness = positions
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .sum::<f32>()
            / (positions.len() - 2) as f32;
        assert!(roughness < 0.08, "visible inner rim roughness: {roughness}");
        let native = render_svg_document_on(&document, 310, 265, [1.0; 3]).unwrap();
        let source = image::load_from_memory(include_bytes!("test-data/wifi-source.png"))
            .unwrap()
            .to_rgb8();
        let mut error = 0.0;
        let mut count = 0;
        for (i, pixel) in source.pixels().enumerate() {
            if pixel[2] as i16 > pixel[1] as i16 + 5 && pixel[2] as i16 > pixel[0] as i16 + 5 {
                error += crate::color::delta_e_ok(
                    rgb_to_oklab(pixel.0.map(|v| v as f32 / 255.0)),
                    rgb_to_oklab(native.pixels[i]),
                );
                count += 1;
            }
        }
        assert!(
            error / (count as f32) < 4.0,
            "foreground colour error: {}",
            error / count as f32
        );
    }

    #[test]
    fn outline_contours_keep_line_arc_and_branch_connections() {
        let directory = tempfile::tempdir().unwrap();
        for gap in [false, true] {
            // An eight-pixel outline separates two colours, turns through
            // circular corners and joins a tapered branch. It must keep its
            // filled contour even where a uniform centre-line would fit.
            let source_svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="192" height="144">
                <path fill="#3399cc" d="M0 0H192V144H0Z"/>
                <path fill="#000000" d="M48 24H144A32 32 0 0 1 176 56V88A32 32 0 0 1 144 120H48A32 32 0 0 1 16 88V56A32 32 0 0 1 48 24Z"/>
                <path fill="#eec488" d="M48 32H144A24 24 0 0 1 168 56V88A24 24 0 0 1 144 112H48A24 24 0 0 1 24 88V56A24 24 0 0 1 48 32Z"/>
                <path fill="#000000" d="M76 116H86L110 140H106Z"/>
                {}</svg>"##,
                if gap {
                    r##"<path fill="#3399cc" d="M88 22H94V34H88Z"/>"##
                } else {
                    ""
                }
            );
            let source_tree = parse_svg_document(&source_svg).unwrap();
            let mut source = resvg::tiny_skia::Pixmap::new(192, 144).unwrap();
            resvg::render(
                &source_tree,
                resvg::tiny_skia::Transform::identity(),
                &mut source.as_mut(),
            );
            let input = directory.path().join("outline.png");
            let output = directory.path().join("outline.svg");
            source.save_png(&input).unwrap();
            let summary = vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 192,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            assert_eq!(summary.structural.recovered_boundary_strokes, 0);
            let document = fs::read_to_string(&output).unwrap();
            let paint = document
                .split("id=\"paint-layer\"")
                .nth(1)
                .unwrap()
                .split("</g>")
                .next()
                .unwrap();
            assert!(
                paint.contains('L'),
                "straight spans must remain in Paint paths"
            );
            if !gap {
                assert!(
                    paint.contains('A'),
                    "supported circular spans must remain in Paint paths"
                );
            }
            let tree = parse_svg_document(&document).unwrap();
            let mut rendered = resvg::tiny_skia::Pixmap::new(192, 144).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut rendered.as_mut(),
            );
            // Inspect the whole dark interior, including both straight/arc
            // joins and the branch junction. Ignore only the AA fringe.
            for y in 2..142 {
                for x in 2..190 {
                    let core = (y - 1..=y + 1).all(|py| {
                        (x - 1..=x + 1).all(|px| source.pixels()[py * 192 + px].red() < 20)
                    });
                    if core {
                        let p = rendered.pixels()[y * 192 + x];
                        assert!(
                            p.red().max(p.green()).max(p.blue()) < 100,
                            "broken outline at {x},{y}, gap={gap}: {p:?}"
                        );
                    }
                }
            }
            if gap {
                assert!(
                    rendered.pixels()[28 * 192 + 91].blue() > 150,
                    "the authored gap must stay open"
                );
            }
        }
    }

    #[test]
    fn native_alpha_white_rim_is_continuous_after_rendering() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let mut image = ImageBuffer::from_pixel(96, 96, Rgba([255_u8, 255, 255, 0]));
        for y in 0..96 {
            for x in 0..96 {
                let r = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                if r < 38.0 {
                    let grey =
                        (255.0 * (0.3 + 0.7 * ((r - 36.5) / 1.0).clamp(0.0, 1.0))).round() as u8;
                    image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
                }
            }
        }
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        image.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 96,
                auto_dimension: false,
                adaptive_refinement: false,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(summary.structural.recovered_alpha_boundary_strokes > 0);
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for degrees in 0..360 {
            let angle = degrees as f32 * std::f32::consts::PI / 180.0;
            let mut peak = 0.0_f32;
            for j in 0..13 {
                let r = 36.0 + j as f32 * 0.25;
                let x = 47.5 + r * angle.cos();
                let y = 47.5 + r * angle.sin();
                let ix = x as usize;
                let iy = y as usize;
                let tx = x - ix as f32;
                let ty = y - iy as f32;
                let value: f32 = [
                    (ix, iy, (1.0 - tx) * (1.0 - ty)),
                    (ix + 1, iy, tx * (1.0 - ty)),
                    (ix, iy + 1, (1.0 - tx) * ty),
                    (ix + 1, iy + 1, tx * ty),
                ]
                .into_iter()
                .map(|(x, y, w)| {
                    let p = pixmap.pixels()[y * 96 + x];
                    w * (f32::from(p.red()) - 0.3 * f32::from(p.alpha()))
                })
                .sum();
                peak = peak.max(value);
            }
            assert!(peak > 30.0, "broken white rim at {degrees} degrees: {peak}");
        }
    }

    #[test]
    fn curved_highlight_remains_visible_beside_a_dark_seam() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        for transparent in [false, true] {
            let mut image = ImageBuffer::from_pixel(
                192,
                192,
                Rgba([255_u8, 255, 255, if transparent { 0 } else { 255 }]),
            );
            for y in 12..180 {
                for x in 12..180 {
                    let grey = 205 + (x / 8) as u8;
                    image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
                }
                let centre = (96.0 + 25.0 * (y as f32 / 28.0).sin()).round() as u32;
                for x in centre - 4..centre {
                    image.put_pixel(x, y, Rgba([100, 100, 100, 255]));
                }
                for x in centre..centre + 3 {
                    image.put_pixel(x, y, Rgba([252, 252, 252, 255]));
                }
            }
            let input = directory.path().join("input.png");
            let output = directory.path().join("output.svg");
            image.save(&input).unwrap();
            vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 192,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(192, 192).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in 20..172 {
                let centre = (96.0 + 25.0 * (y as f32 / 28.0).sin()).round() as usize;
                let peak = pixmap.pixels()[y * 192 + centre..y * 192 + centre + 4]
                    .iter()
                    .map(|p| p.red())
                    .max()
                    .unwrap();
                // Require at least half the local source contrast after curve antialiasing.
                let background = 205 + ((centre + 1) / 8) as u8;
                assert!(
                    f32::from(peak) >= 0.5 * f32::from(252_u16 + u16::from(background)),
                    "lost highlight at y={y}: {peak}, transparent={transparent}"
                );
            }
        }
    }

    #[test]
    fn transparent_greyscale_edges_do_not_invent_colours() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let config = Config {
            maximum_dimension: 96,
            auto_dimension: false,
            adaptive_refinement: false,
            rayon_threads: 1,
            ..Config::default()
        };
        let mut renders = Vec::new();
        for hidden in [[255, 0, 255, 0], [0, 255, 0, 0]] {
            let mut image = ImageBuffer::from_pixel(96, 96, Rgba(hidden));
            for y in 0..96 {
                for x in 0..96 {
                    let d = ((x as f32 - 48.0).powi(2) + (y as f32 - 48.0).powi(2)).sqrt();
                    let alpha = ((34.5 - d).clamp(0.0, 1.0) * 255.0) as u8;
                    if alpha > 0 {
                        let grey = if (45..49).contains(&x) {
                            250
                        } else {
                            110 + x as u8
                        };
                        image.put_pixel(x, y, Rgba([grey, grey, grey, alpha]));
                    }
                }
            }
            let input = directory.path().join("input.png");
            let output = directory.path().join("output.svg");
            image.save(&input).unwrap();
            vectorize(&input, &output, &config).unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for p in pixmap.pixels() {
                let channels = [p.red(), p.green(), p.blue()];
                assert!(
                    channels.iter().max().unwrap() - channels.iter().min().unwrap() <= 1,
                    "invented colour: {p:?}"
                );
            }
            renders.push(pixmap.take());
        }
        assert_eq!(
            renders[0], renders[1],
            "hidden RGB changed the visible result"
        );
    }

    #[test]
    fn translucent_black_grid_stays_neutral_and_connected() {
        use image::{ImageBuffer, Rgba};
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("grid.png");
        let output = directory.path().join("grid.svg");
        let mut image = ImageBuffer::from_pixel(96, 96, Rgba([255_u8, 0, 255, 0]));
        // Other Paint fits must not leak the normalization backing into the
        // masked grid's underpaint, even when the image contains gradients.
        for y in 12..84 {
            for x in 6..34 {
                let grey = (90 + 4 * x) as u8;
                image.put_pixel(x, y, Rgba([grey, grey, grey, 255]));
            }
        }
        for y in 12..40 {
            for x in 65..88 {
                image.put_pixel(x, y, Rgba([0, 255, 0, 255]));
            }
        }
        for y in 3..93 {
            image.put_pixel(47, y, Rgba([0, 0, 0, 210]));
            image.put_pixel(48, y, Rgba([0, 0, 0, 204]));
        }
        for x in 3..93 {
            image.put_pixel(x, 47, Rgba([0, 0, 0, 210]));
            image.put_pixel(x, 48, Rgba([0, 0, 0, 204]));
        }
        image.put_pixel(47, 47, Rgba([0, 0, 0, 255]));
        // A real break must remain a break.
        for y in 70..76 {
            for x in 47..49 {
                image.put_pixel(x, y, Rgba([255, 0, 255, 0]));
            }
        }
        image.save(&input).unwrap();
        vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 96,
                auto_dimension: false,
                adaptive_refinement: false,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        let document = fs::read_to_string(output).unwrap();
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for y in 6..90 {
            if (69..77).contains(&y) {
                continue;
            }
            let pixels = &pixmap.pixels()[y * 96 + 46..y * 96 + 50];
            assert!(pixels.iter().any(|p| p.alpha() > 64), "grid gap at y={y}");
            for p in pixels.iter().filter(|p| p.alpha() > 32) {
                assert!(
                    p.red().max(p.green()).max(p.blue()) <= 3,
                    "tinted black grid at y={y}: {p:?}"
                );
            }
        }
        assert_eq!(pixmap.pixels()[73 * 96 + 47].alpha(), 0);
    }

    #[test]
    #[ignore = "renders the full-size car sample"]
    fn car_headlight_boundary_has_no_dark_spurs() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("car.svg");
        vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1254, 1254).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        // The AA samples at both tips used to acquire the near-black Paint
        // of the adjacent rim. Inspect the final composite, not just labels.
        for (x, y) in [(170, 637), (281, 593)] {
            let expected = source.get_pixel(x, y).0;
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()];
            assert!(
                expected
                    .into_iter()
                    .zip(actual)
                    .all(|(a, b)| a.abs_diff(b) <= 64),
                "headlight spur at ({x}, {y}): source {expected:?}, SVG {actual:?}"
            );
        }
        // The bumper highlight brightens towards the lower trim. The former
        // radial focus reversed this slope in the final rendered SVG.
        let upper = pixmap.pixels()[739 * 1254 + 216];
        let lower = pixmap.pixels()[745 * 1254 + 216];
        assert!(
            lower.green() >= upper.green() + 3,
            "bumper gradient reversed: {} -> {}",
            upper.green(),
            lower.green()
        );
        assert!(lower.green().abs_diff(source.get_pixel(216, 745)[1]) <= 24);
        // Residual corrections along the window rim must not spread a dark
        // patch into otherwise smooth blue glass.
        for (x, y) in [
            (664, 403),
            (664, 405),
            (670, 403),
            (690, 510),
            (735, 504),
            (819, 491),
        ] {
            let expected = source.get_pixel(x, y).0.map(|v| v as f32 / 255.0);
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()].map(|v| v as f32 / 255.0);
            let error = crate::color::delta_e_ok(rgb_to_oklab(expected), rgb_to_oklab(actual));
            assert!(
                error <= 3.0,
                "unsupported window shadow at {x},{y}: {error}"
            );
        }
        // Residual overlays must not reverse the smooth source ramps under
        // the mirror or above the front wheel into circular dark/light spots.
        for ((x0, y0), (x1, y1)) in [((628, 550), (628, 556)), ((380, 637), (386, 637))] {
            let lightness = |x: u32, y: u32| {
                let p = pixmap.pixels()[(y * 1254 + x) as usize];
                rgb_to_oklab([p.red(), p.green(), p.blue()].map(|c| c as f32 / 255.0)).l
            };
            let source_lightness =
                |x, y| rgb_to_oklab(source.get_pixel(x, y).0.map(|c| c as f32 / 255.0)).l;
            assert!(source_lightness(x1, y1) > source_lightness(x0, y0));
            assert!(
                lightness(x1, y1) >= lightness(x0, y0) - 0.5,
                "circular residual reversed the highlight at {x0},{y0} -> {x1},{y1}"
            );
        }
        // Smooth body highlights must not acquire nested quantizer bands.
        // Check both the rear quarter and the door, including the left bend.
        // resvg renders 672 / 1053 such steps in the original sample. Keep
        // substantial headroom for antialiasing while requiring a large drop.
        for (name, x0, y0, x1, y1, maximum_jumps, maximum_mean_error) in [
            ("rear", 950, 475, 1120, 524, 250, 2.8),
            ("door", 558, 543, 840, 610, 300, 4.1),
        ] {
            let mut jumps = 0;
            let mut colour_error = 0_u64;
            let mut colour_samples = 0_u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let reference = source.get_pixel(x, y).0;
                    if reference[0] < 220 || !(36..175).contains(&reference[1]) {
                        continue;
                    }
                    let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
                    let actual = [pixel.red(), pixel.green(), pixel.blue()];
                    colour_error += (0..3)
                        .map(|c| u64::from(actual[c].abs_diff(reference[c])))
                        .sum::<u64>();
                    colour_samples += 3;
                    for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                        let neighbour = source.get_pixel(nx, ny).0;
                        let rendered = pixmap.pixels()[(ny * 1254 + nx) as usize];
                        let rendered = [rendered.red(), rendered.green(), rendered.blue()];
                        let source_step = (0..3)
                            .map(|c| reference[c].abs_diff(neighbour[c]))
                            .max()
                            .unwrap();
                        let output_step = (0..3)
                            .map(|c| actual[c].abs_diff(rendered[c]))
                            .max()
                            .unwrap();
                        if source_step <= 3 && output_step > source_step + 8 {
                            jumps += 1;
                        }
                    }
                }
            }
            assert!(
                jumps <= maximum_jumps,
                "nested {name} highlight bands: {jumps}"
            );
            let mean = colour_error as f64 / colour_samples.max(1) as f64;
            assert!(
                mean <= maximum_mean_error,
                "{name} highlight colour error: {mean}"
            );
        }
        // Coverage repair must retain the previously restored rim reflection.
        for (x, y) in [(365, 762), (386, 738)] {
            let pixel = pixmap.pixels()[y * 1254 + x];
            assert!(
                pixel.red() > 120 && pixel.green() > 120 && pixel.blue() > 120,
                "wheel highlight disappeared at ({x}, {y})"
            );
        }
    }

    #[test]
    #[ignore = "renders the full-size car sample"]
    fn car_front_fender_highlight_keeps_local_shading() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/car.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("car.svg");
        let summary = vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            (summary.processing_width, summary.processing_height),
            (1254, 1254)
        );
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1254, 1254).unwrap();
        // Match the opaque source/browser background instead of measuring
        // premultiplied RGB against transparent black at antialiased edges.
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let mut error = 0_u64;
        let mut channels = 0_u64;
        // The curved highlight above the front wheel, excluding the lamp.
        // Keep the existing colour-error gate, including antialiased boundaries.
        // The reproducible pre-OKLab HEAD measures 3.737 with this renderer.
        for y in 565..645 {
            for x in 320..530 {
                let reference = source.get_pixel(x, y).0;
                if reference[0] <= 190 || reference[1] >= 160 {
                    continue;
                }
                let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
                for (expected, actual) in
                    reference
                        .into_iter()
                        .zip([pixel.red(), pixel.green(), pixel.blue()])
                {
                    error += u64::from(expected.abs_diff(actual));
                    channels += 1;
                }
            }
        }
        // Previously reported fender/bumper artifacts and the lower window
        // ridge: evaluate the final
        // composite as well as the field fit, so antialias seams are included.
        for (x, y) in [(335, 655), (528, 592), (286, 752), (780, 506), (780, 509)] {
            let expected = source.get_pixel(x, y).0;
            let pixel = pixmap.pixels()[(y * 1254 + x) as usize];
            let actual = [pixel.red(), pixel.green(), pixel.blue()];
            let worst = expected
                .into_iter()
                .zip(actual)
                .map(|(a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(
                worst <= 10,
                "false colour at ({x},{y}): {actual:?} != {expected:?}"
            );
        }
        // A good interior colour fit does not guarantee a smooth boundary.
        // Track the native upper wheel-arch step at subpixel precision and
        // measure its turning separately from the shading above it.
        let mut edge_positions = [Vec::new(), Vec::new()];
        for x in 360..491 {
            let y = (620..694)
                .min_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[1]) - i16::from(source.get_pixel(x, y)[1])
                })
                .unwrap();
            let mut weights = [0.0_f64; 2];
            let mut moments = [0.0_f64; 2];
            for offset in -4_i32..5 {
                let row = (y as i32 + offset) as u32;
                let a = pixmap.pixels()[(row * 1254 + x) as usize].green();
                let b = pixmap.pixels()[((row + 1) * 1254 + x) as usize].green();
                let steps = [
                    source.get_pixel(x, row)[1].saturating_sub(source.get_pixel(x, row + 1)[1]),
                    a.saturating_sub(b),
                ];
                for (i, step) in steps.into_iter().enumerate() {
                    weights[i] += f64::from(step);
                    moments[i] += f64::from(step) * (f64::from(row) + 0.5);
                }
            }
            for i in 0..2 {
                assert!(weights[i] > 0.0);
                edge_positions[i].push(moments[i] / weights[i]);
            }
        }
        let roughness = edge_positions[1]
            .windows(3)
            .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
            .sum::<f64>()
            / (edge_positions[1].len() - 2) as f64;
        let displacement = edge_positions[0]
            .iter()
            .zip(&edge_positions[1])
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / edge_positions[0].len() as f64;
        assert!(
            roughness < 0.16,
            "wheel-arch contour roughness: {roughness} (pre-OKLab: 0.149; regression: 0.180)"
        );
        assert!(
            displacement < 0.4,
            "wheel-arch contour moved {displacement}px from source"
        );
        assert!(channels > 40_000);
        let mean_error = error as f64 / channels as f64;
        assert!(
            mean_error < 3.65,
            "front fender highlight error: {mean_error}, regions: {}, output: {}",
            summary.geometry.regions,
            directory.keep().display()
        );
        // Inspect both sides of the narrow window rim in the final SVG.
        // The old fill-only contour fit hid a broken/stepped outer ink edge,
        // and pixel colour repairs introduced a 0.81px kink on the inner one.
        let enlarged = render_svg_tree_on(
            &tree,
            5016,
            5016,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            [1.0; 3],
        )
        .unwrap();
        let mut rim_edges = [Vec::new(), Vec::new()];
        for x in 775..885 {
            let glass_edge = (332..364)
                .max_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[2]) - i16::from(source.get_pixel(x, y)[2])
                })
                .unwrap();
            let core = (glass_edge - 3..=glass_edge)
                .min_by_key(|&y| source.get_pixel(x, y).0.into_iter().max().unwrap())
                .unwrap();
            let start = (core - 2) * 4;
            let levels: Vec<_> = (start..start + 20)
                .map(|y| {
                    enlarged
                        .get(x as usize * 4 + 2, y as usize)
                        .into_iter()
                        .fold(0.0_f32, f32::max)
                })
                .collect();
            let threshold = 30.0 / 255.0;
            let first = levels
                .iter()
                .position(|&v| v < threshold)
                .expect("window ink gap");
            let last = levels.iter().rposition(|&v| v < threshold).unwrap();
            assert!(
                first > 0 && last + 1 < levels.len(),
                "rim left its source corridor at {x}"
            );
            for (edge, (a, b)) in [(first - 1, first), (last, last + 1)]
                .into_iter()
                .enumerate()
            {
                let t = (threshold - levels[a]) / (levels[b] - levels[a]);
                rim_edges[edge].push((start as f32 + a as f32 + 0.5 + t) * 0.25);
            }
        }
        for (edge, positions) in rim_edges.iter().enumerate() {
            let turns: Vec<_> = positions
                .windows(3)
                .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
                .collect();
            assert!(
                turns.iter().all(|&v| v < 0.4),
                "window rim {edge} kink: {turns:?}"
            );
            if edge == 1 {
                let roughness = turns.iter().sum::<f32>() / turns.len() as f32;
                assert!(roughness < 0.08, "window inner rim roughness: {roughness}");
            }
        }
        // The neighbouring grey trim has a separate boundary. Checking the
        // glass-facing edge alone missed its former >1px staircase.
        let mut grey_edges = [Vec::new(), Vec::new(), Vec::new()];
        for x in 775..885 {
            let y = (332..364)
                .max_by_key(|&y| {
                    i16::from(source.get_pixel(x, y + 1)[2]) - i16::from(source.get_pixel(x, y)[2])
                })
                .unwrap();
            let start = (y - 11) * 4;
            let levels: Vec<_> = (start..(y - 3) * 4)
                .map(|row| {
                    let rgb = enlarged.get(x as usize * 4 + 2, row as usize);
                    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                })
                .collect();
            for (edge, threshold) in [40.0_f32, 50.0, 60.0].into_iter().enumerate() {
                let threshold = threshold / 255.0;
                let i = levels
                    .windows(2)
                    .position(|p| p[0] < threshold && p[1] >= threshold)
                    .unwrap_or_else(|| panic!("grey trim edge missing at {x}"));
                let t = (threshold - levels[i]) / (levels[i + 1] - levels[i]);
                grey_edges[edge].push((start as f32 + i as f32 + 0.5 + t) * 0.25);
            }
        }
        for positions in grey_edges {
            let turns: Vec<_> = positions
                .windows(3)
                .map(|p| (p[2] - 2.0 * p[1] + p[0]).abs())
                .collect();
            assert!(turns.iter().all(|&v| v < 0.55), "grey trim kink: {turns:?}");
            let mean = turns.iter().sum::<f32>() / turns.len() as f32;
            assert!(mean < 0.09, "grey trim roughness: {mean}");
        }
    }

    #[test]
    #[ignore = "renders the full-size cliparts sample"]
    fn cliparts_highlights_preserve_colour_and_authored_transparency() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/cliparts.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("cliparts.svg");
        vectorize(
            &input,
            &output,
            &Config {
                rayon_threads: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let source = image::open(input).unwrap().to_rgba8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1600, 1200).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        for (name, (x0, y0, x1, y1), limit) in [
            ("penguin highlight", (945, 457, 975, 470), 7.0),
            ("flask liquid", (1380, 160, 1430, 215), 2.0),
            // Before source-supported ink refinement: 35.54 and 6.52.
            ("dotted oval upper edge", (500, 309, 530, 320), 8.0),
            ("dotted oval", (468, 310, 546, 368), 4.0),
        ] {
            let mut error = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let reference = source.get_pixel(x, y).0;
                    let alpha = reference[3] as f64 / 255.0;
                    let pixel = pixmap.pixels()[(y * 1600 + x) as usize];
                    for (c, actual) in [pixel.red(), pixel.green(), pixel.blue()]
                        .into_iter()
                        .enumerate()
                    {
                        error += (reference[c] as f64 * alpha + 255.0 * (1.0 - alpha)
                            - actual as f64)
                            .abs();
                    }
                }
            }
            let mean = error / (3 * (x1 - x0) * (y1 - y0)) as f64;
            assert!(mean < limit, "{name}: mean RGB error {mean}");
        }
    }

    #[test]
    #[ignore = "renders the full-size cliparts sample"]
    fn cliparts_penguin_inner_foot_outlines_remain_continuous() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("sample/input/cliparts.png");
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("cliparts.svg");
        let summary = vectorize(&input, &output, &Config::default()).unwrap();
        assert_eq!(
            (summary.processing_width, summary.processing_height),
            (1600, 1200)
        );
        let source = image::open(input).unwrap().to_rgb8();
        let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(1600, 1200).unwrap();
        pixmap.fill(resvg::tiny_skia::Color::WHITE);
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        // Follow the source's dark centre on each side of the white gap.
        // Permit one pixel of fitting displacement, but never a missing row.
        for (left, right) in [(989, 998), (1000, 1018)] {
            for y in 685..723 {
                let x = (left..right)
                    .min_by_key(|&x| *source.get_pixel(x, y).0.iter().max().unwrap())
                    .unwrap();
                assert!(
                    (x - 1..=x + 1).any(|sample_x| {
                        let p = pixmap.pixels()[(y * 1600 + sample_x) as usize];
                        p.red().max(p.green()).max(p.blue()) <= 128
                    }),
                    "penguin foot outline lost at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn tonal_details_survive_rendering_with_authored_gaps() {
        use image::{ImageBuffer, Rgb};

        for (background, detail) in [
            ([238_u8, 238, 242], [228_u8, 228, 232]),
            ([24, 24, 28], [32, 32, 36]),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let input = directory.path().join("faint-seam.png");
            let output = directory.path().join("faint-seam.svg");
            let mut image = ImageBuffer::from_pixel(96, 96, Rgb(background));
            for y in 8..88 {
                if (43..53).contains(&y) {
                    continue;
                }
                for x in 46..50 {
                    image.put_pixel(x, y, Rgb(detail));
                }
            }
            image.save(&input).unwrap();
            vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 96,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 1,
                    ..Config::default()
                },
            )
            .unwrap();
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(96, 96).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in (12..39).chain(57..84) {
                let wall = pixmap.pixels()[y * 96 + 40].red();
                let seam = pixmap.pixels()[y * 96 + 48].red();
                assert!(
                    wall.abs_diff(seam) >= 4,
                    "faint seam lost at {y}: wall={wall}, seam={seam}"
                );
            }
            let wall = pixmap.pixels()[48 * 96 + 40].red();
            let gap = pixmap.pixels()[48 * 96 + 48].red();
            assert!(wall.abs_diff(gap) <= 2, "authored seam gap was filled");
        }
    }

    #[test]
    fn face_alpha_preserves_foreground_equal_to_temporary_backing() {
        use image::{ImageBuffer, Rgba};

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.svg");
        let mut image = ImageBuffer::from_pixel(64, 64, Rgba([17_u8, 31, 47, 0]));

        // Cyan dominates covered pixels, making red the farthest temporary
        // normalization colour. The disconnected opaque red square must not
        // merge with that normalized transparent RGB and disappear.
        for y in 16..56 {
            for x in 16..56 {
                image.put_pixel(x, y, Rgba([0, 255, 255, 255]));
            }
        }
        for y in 4..12 {
            for x in 4..12 {
                image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        // Broad authored translucent areas exercise the two intermediate
        // alpha levels independently of boundary antialiasing.
        for y in 22..30 {
            for x in 22..30 {
                image.put_pixel(x, y, Rgba([0, 255, 0, 80]));
            }
        }
        for y in 38..46 {
            for x in 38..46 {
                image.put_pixel(x, y, Rgba([0, 0, 255, 180]));
            }
        }
        image.save(&input).unwrap();

        let summary = vectorize(
            &input,
            &output,
            &Config {
                maximum_dimension: 64,
                auto_dimension: false,
                adaptive_refinement: false,
                smoothing_radius: 1,
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                rayon_threads: 1,
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(
            summary.source_alpha.temporary_backing_color,
            Some([255, 0, 0])
        );
        assert_eq!(summary.source_alpha.quantization_bits, 8);
        assert_eq!(summary.source_alpha.mask_paths, 0);

        let document = fs::read_to_string(&output).unwrap();
        assert!(document.contains("fill-opacity=\"0.314\""));
        assert!(document.contains("fill-opacity=\"0.706\""));
        let tree = parse_svg_document(&document).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(64, 64).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let pixel = |x: usize, y: usize| pixmap.pixels()[y * 64 + x];
        assert_eq!(pixel(2, 2).alpha(), 0);
        assert!(pixel(8, 8).alpha() > 240);
        assert!(pixel(8, 8).red() > 240);
        assert!(pixel(8, 8).green() < 10);
        assert!((75..=95).contains(&pixel(26, 26).alpha()));
        assert!((160..=180).contains(&pixel(42, 42).alpha()));
    }

    #[test]
    fn authored_alpha_ramp_is_not_reduced_to_four_bands() {
        use image::{ImageBuffer, Rgba};
        for (low, span) in [(16.0, 224.0), (220.0, 32.0)] {
            let directory = tempfile::tempdir().unwrap();
            let input = directory.path().join("ramp.png");
            let output = directory.path().join("ramp.svg");
            let mut image = ImageBuffer::from_pixel(128, 128, Rgba([0_u8, 0, 255, 0]));
            for y in 16..112 {
                for x in 16..112 {
                    image.put_pixel(
                        x,
                        y,
                        Rgba([
                            0,
                            0,
                            255,
                            (low + span * (y - 16) as f32 / 95.0).round() as u8,
                        ]),
                    );
                }
            }
            image.save(&input).unwrap();
            let summary = vectorize(
                &input,
                &output,
                &Config {
                    maximum_dimension: 128,
                    auto_dimension: false,
                    adaptive_refinement: false,
                    rayon_threads: 4,
                    ..Config::default()
                },
            )
            .unwrap();
            assert_eq!(summary.source_alpha.quantization_bits, 8);
            // Each coverage layer now has a separate seam stroke underneath
            // the fills; allow eight layers plus their eight underpass paths.
            assert!(summary.source_alpha.mask_paths <= 16);
            let tree = parse_svg_document(&fs::read_to_string(output).unwrap()).unwrap();
            let mut pixmap = resvg::tiny_skia::Pixmap::new(128, 128).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            for y in 24..104 {
                let expected = image.get_pixel(64, y).0[3];
                let actual = pixmap.pixels()[y as usize * 128 + 64].alpha();
                assert!(
                    expected.abs_diff(actual) <= 5,
                    "y={y}: alpha {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn source_antialias_coverage_builds_one_opaque_vector_silhouette() {
        let matte = AlphaMatte::new(
            5,
            3,
            [0.0, 0.25, 0.75, 1.0, 1.0]
                .into_iter()
                .cycle()
                .take(15)
                .collect(),
        );
        let path = matte
            .isocontours(0.5)
            .iter()
            .map(|contour| fitted_alpha_contour_path_data(contour))
            .collect::<String>();
        assert!(!path.is_empty());
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn quality_metrics_require_explicit_opt_in() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "picvec-quality-contract-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("output.svg");
        let mut raster = Raster::blank(32, 24, [0.9, 0.9, 0.9]);
        for y in 6..18 {
            for x in 8..24 {
                raster.pixels[y * 32 + x] = [0.2, 0.4, 0.75];
            }
        }
        raster.save(&input).unwrap();
        let summary = vectorize(
            &input,
            &output,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                compute_quality_metrics: true,
                ..Config::default()
            },
        )
        .unwrap();
        let quality = summary.quality.unwrap();
        assert!(quality.delta_e_ok_mean.is_finite());
        assert!(quality.delta_e_ok_p90.is_finite());
        assert!(quality.delta_e_ok_p99.is_finite());
        assert!(quality.global_ssim.is_finite());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn concurrent_conversions_to_one_output_remain_atomic() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "picvec-concurrent-contract-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let input = directory.join("input.png");
        let output = directory.join("shared.svg");
        let mut raster = Raster::blank(32, 24, [0.88, 0.91, 0.95]);
        for y in 4..20 {
            for x in 5..27 {
                raster.pixels[y * 32 + x] = if x < 16 {
                    [0.16, 0.24, 0.72]
                } else {
                    [0.78, 0.18, 0.25]
                };
            }
        }
        raster.save(&input).unwrap();

        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let input = input.clone();
                let output = output.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    vectorize(
                        &input,
                        &output,
                        &Config {
                            segmentation_min_size: 2,
                            minimum_gradient_area: 8,
                            rayon_threads: 1,
                            ..Config::default()
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        let document = fs::read_to_string(&output).unwrap();
        assert!(document.starts_with("<?xml"));
        assert!(document.contains("<svg"));
        let files: HashSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files,
            HashSet::from(["input.png".to_string(), "shared.svg".to_string()])
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
