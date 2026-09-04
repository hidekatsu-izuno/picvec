use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use tempfile::{Builder as TemporaryFileBuilder, NamedTempFile};

use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::collections::HashSet;

use crate::adaptive::{
    compose_refinements, perceptual_score, plan_candidates, AdaptiveRefinementSummary,
    EmbeddedRefinement, SourceRect,
};
use crate::chroma::{self, AlphaMatte, AlphaTransparencySummary, ChromaKeySummary};
use crate::color::{rgb_to_lab, Lab};
use crate::config::Config;
use crate::edge::{classify, dilate, dilate_square, perceptual_smooth, EdgeSummary};
use crate::geometry::{build_with_topology as build_geometry, GeometrySummary};
use crate::gradient::{
    fit_all_without_topology, merge_partition, merge_source_supported_paints, refresh_summary,
    GradientSummary, Paint,
};
use crate::hierarchy::{HierarchicalTopology, HierarchicalTopologySummary};
use crate::metrics::QualityMetrics;
use crate::optimize::{summarize as optimization_summary, OptimizationSummary};
use crate::ownership::{resolve as resolve_boundary_ownership, BoundaryOwnershipSummary};
use crate::raster::Raster;
use crate::segment::{
    refine_thin_paint_ownership, regularize_boundaries, replace_final_exact_paint_labels, segment,
    Segmentation, SegmentationSummary,
};
use crate::structural::{analyse as analyse_structural, StructuralInk, StructuralSummary};
use crate::svg::{serialize_filtered as serialize_svg, SvgSummary};
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
                if current == following || paints[current] != paints[following] {
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

fn estimate_dimension(image: &Raster, config: &Config) -> ComplexityProbe {
    let probe_max = image.width.max(image.height).min(1024) as u32;
    let probe = image.resize_max(probe_max);
    let lab: Vec<Lab> = probe.pixels.iter().copied().map(rgb_to_lab).collect();
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
    let source_pixels = (image.width * image.height).max(1) as f32;
    let estimated = image.width.max(image.height) as f32 * (target_pixels / source_pixels).sqrt();
    let (automatic_minimum, automatic_maximum) = config.automatic_dimension_bounds();
    let selected = estimated
        .round()
        .clamp(automatic_minimum as f32, automatic_maximum as f32)
        .min(image.width.max(image.height) as f32) as u32;
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
    background: [f32; 3],
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
    let width = u32::try_from(width)
        .map_err(|_| -> Error { "SVG preview width exceeds the renderer limit".into() })?;
    let height = u32::try_from(height)
        .map_err(|_| -> Error { "SVG preview height exceeds the renderer limit".into() })?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| -> Error { "could not allocate the SVG preview raster".into() })?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
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

fn physical_core_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let mut cores = HashSet::<(String, String)>::new();
        for entry in fs::read_dir("/sys/devices/system/cpu").ok()? {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let name = name.to_str()?;
            if !name.strip_prefix("cpu").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            }) {
                continue;
            }
            let topology = entry.path().join("topology");
            let package = fs::read_to_string(topology.join("physical_package_id")).ok()?;
            let core = fs::read_to_string(topology.join("core_id")).ok()?;
            cores.insert((package.trim().to_owned(), core.trim().to_owned()));
        }
        (!cores.is_empty()).then_some(cores.len())
    }
    #[cfg(not(target_os = "linux"))]
    None
}

fn execution_thread_count(config: &Config) -> usize {
    let logical = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if config.rayon_threads > 0 {
        return config.rayon_threads.min(logical).max(1);
    }
    physical_core_count().unwrap_or(logical).min(logical).max(1)
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
    const MAXIMUM_JOBS: usize = 8;
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

struct EvaluatedRefinement {
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
fn adaptively_refine(
    vector_source: Option<&Raster>,
    source_matte: Option<&AlphaMatte>,
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
    let source = vector_source;
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
    let baseline_whole = perceptual_score(source, whole, &base_render, whole);
    summary.baseline_mean_delta_e = baseline_whole.mean_delta_e;
    summary.refined_mean_delta_e = baseline_whole.mean_delta_e;

    let mut candidates = plan_candidates(
        source,
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
    // The coarse error/model-cost ratio is an optimistic rate bound. Regions
    // below it cannot beat the full candidate's stricter measured SVG-byte
    // charge often enough to justify running another complete pipeline. The
    // same bound retains compact icon features and rejects costly stochastic
    // photo texture without identifying either content type.
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
    child_config.retain_diagnostics = false;
    let base_scale = 1.0 / source_scale;
    let mut evaluated = Vec::<EvaluatedRefinement>::new();
    // Keep enough independent jobs in flight to cover serial geometry and
    // segmentation stages, while estimating their dense working sets before
    // committing memory. The cap prevents a transiently high MemAvailable
    // value from turning a large source into an unbounded allocation burst.
    let parallel_jobs =
        adaptive_parallel_jobs(&candidates, (input_width, input_height), execution_threads);
    summary.parallel_jobs = parallel_jobs;
    for batch in candidates.chunks(parallel_jobs) {
        let outcomes = batch
            .par_iter()
            .map(|candidate| -> Result<RefinementOutcome> {
                let margin = (candidate.core.width.min(candidate.core.height) / 64).clamp(8, 24);
                let expanded = candidate.core.expanded(margin, input_width, input_height);
                let crop =
                    vector_source.crop(expanded.x, expanded.y, expanded.width, expanded.height);
                let crop_matte = source_matte.map(|matte| {
                    matte.crop(expanded.x, expanded.y, expanded.width, expanded.height)
                });
                let probe = if child_config.auto_dimension {
                    let (_, automatic_maximum) = child_config.automatic_dimension_bounds();
                    if crop.width.max(crop.height) <= automatic_maximum as usize
                        && crop.pixels.len() as f32 <= MINIMUM_AUTOMATIC_TARGET_PIXELS
                    {
                        ComplexityProbe {
                            selected_dimension: crop.width.max(crop.height) as u32,
                            target_pixels: MINIMUM_AUTOMATIC_TARGET_PIXELS,
                            ..ComplexityProbe::default()
                        }
                    } else {
                        estimate_dimension(&crop, &child_config)
                    }
                } else {
                    ComplexityProbe {
                        selected_dimension: child_config
                            .maximum_dimension
                            .min(crop.width.max(crop.height) as u32),
                        ..ComplexityProbe::default()
                    }
                };
                let processing = crop.resize_max(probe.selected_dimension.max(64));
                let processing_matte = crop_matte
                    .as_ref()
                    .map(|matte| matte.resized(processing.width, processing.height));
                let local_scale = (processing.width as f32 / expanded.width.max(1) as f32)
                    .min(processing.height as f32 / expanded.height.max(1) as f32);
                if local_scale <= 1.1 * base_scale {
                    return Ok(RefinementOutcome::NotFiner);
                }
                let child = vectorize_processing(
                    processing,
                    processing_matte.as_ref(),
                    core.preview_background,
                    &child_config,
                )?;
                let child_render = render_svg_document_on(
                    &child.document,
                    child.processing_reference.width,
                    child.processing_reference.height,
                    core.preview_background,
                )?;
                let refined = perceptual_score(source, candidate.core, &child_render, expanded);
                let combined_gain = candidate.baseline.combined - refined.combined;
                if combined_gain < config.adaptive_min_perceptual_gain
                    || refined.p90_delta_e > candidate.baseline.p90_delta_e + 0.25
                    || refined.missing_edge_fraction
                        > candidate.baseline.missing_edge_fraction + 0.025
                {
                    return Ok(RefinementOutcome::QualityRejected);
                }
                let bytes_per_source_pixel =
                    child.svg.bytes as f32 / candidate.core.area().max(1) as f32;
                let complexity_charge = config.adaptive_complexity_penalty
                    * candidate.model_cost.sqrt()
                    * bytes_per_source_pixel;
                if combined_gain < complexity_charge {
                    return Ok(RefinementOutcome::ComplexityRejected);
                }
                let child_svg_bytes = child.svg.bytes.max(1);
                Ok(RefinementOutcome::Accepted(Box::new(EvaluatedRefinement {
                    embedded: EmbeddedRefinement {
                        core: candidate.core,
                        expanded,
                        document: child.document,
                        processing_width: child.processing_reference.width,
                        processing_height: child.processing_reference.height,
                    },
                    svg: child.svg,
                    baseline_mean: candidate.baseline.mean_delta_e,
                    refined_mean: refined.mean_delta_e,
                    rate: combined_gain * candidate.core.area() as f32 / child_svg_bytes as f32,
                })))
            })
            .collect::<Result<Vec<_>>>()?;
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
    }

    // A global byte budget converts the local ordering into a deterministic
    // best-first refinement pass.  Candidate generation order cannot change
    // which equal-cost regions win.
    evaluated.sort_by(|left, right| {
        right
            .rate
            .total_cmp(&left.rate)
            .then_with(|| left.embedded.core.y.cmp(&right.embedded.core.y))
            .then_with(|| left.embedded.core.x.cmp(&right.embedded.core.x))
    });
    let mut accepted = Vec::<EmbeddedRefinement>::new();
    for refinement in evaluated {
        if summary.added_svg_bytes.saturating_add(refinement.svg.bytes)
            > config.adaptive_svg_budget_bytes
        {
            summary.rejected_for_complexity += 1;
            continue;
        }
        let area_weight = refinement.embedded.core.area() as f32 / whole.area().max(1) as f32;
        summary.estimated_global_delta_e_reduction +=
            (refinement.baseline_mean - refinement.refined_mean).max(0.0) * area_weight;
        summary.added_svg_bytes += refinement.svg.bytes;
        core.svg.add_elements_from(&refinement.svg);
        accepted.push(refinement.embedded);
    }
    summary.accepted_regions = accepted.len();
    summary.refined_mean_delta_e =
        (summary.baseline_mean_delta_e - summary.estimated_global_delta_e_reduction).max(0.0);
    if accepted.is_empty() {
        return Ok(summary);
    }
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
    let (decoded, decoded_alpha) = Raster::load_with_alpha(
        input,
        config.maximum_input_dimension,
        config.maximum_input_pixels,
        config.maximum_decode_bytes,
    )?;
    let input_width = decoded.width;
    let input_height = decoded.height;
    let source_has_alpha = decoded_alpha.is_some();
    let (source, input_matte, preview_background, detected_key, alpha_backing) =
        if let Some(alpha) = decoded_alpha {
            let matte = AlphaMatte::new(input_width, input_height, alpha);
            let backing = chroma::select_alpha_backing(&decoded, &matte);
            let composed = chroma::composite_over(&decoded, &matte, backing);
            (
                chroma::separate_foreground(&composed, &matte, backing),
                Some(matte),
                backing,
                None,
                Some(backing),
            )
        } else if let Some(key) = config
            .remove_chroma_key_background
            .then(|| chroma::detect(&decoded))
            .flatten()
        {
            let matte = chroma::pull_matte(&decoded, key);
            let separated = chroma::separate_foreground(&decoded, &matte, key.sampled);
            (separated, Some(matte), key.sampled, Some(key), None)
        } else {
            (decoded, None, [1.0; 3], None, None)
        };
    let complexity = if config.auto_dimension {
        estimate_dimension(&source, config)
    } else {
        ComplexityProbe {
            selected_dimension: config
                .maximum_dimension
                .min(source.width.max(source.height) as u32),
            ..ComplexityProbe::default()
        }
    };
    let processing = source.resize_max(complexity.selected_dimension.max(64));
    let processing_matte = input_matte
        .as_ref()
        .map(|matte| matte.resized(processing.width, processing.height));
    let processing_width = processing.width;
    let processing_height = processing.height;
    let source_scale = (input_width as f32 / processing_width.max(1) as f32)
        .max(input_height as f32 / processing_height.max(1) as f32);
    let retain_adaptive_source =
        config.adaptive_refinement && source_scale >= config.adaptive_min_source_scale;
    let adaptive_source = retain_adaptive_source.then_some(source);
    let adaptive_matte = if retain_adaptive_source {
        input_matte
    } else {
        None
    };
    let mut core = vectorize_processing(
        processing,
        processing_matte.as_ref(),
        preview_background,
        config,
    )?;
    let adaptive_refinement = adaptively_refine(
        adaptive_source.as_ref(),
        adaptive_matte.as_ref(),
        (input_width, input_height),
        &mut core,
        config,
        execution_threads,
    )?;
    let to_u8 =
        |color: [f32; 3]| color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8);
    let source_alpha = AlphaTransparencySummary {
        detected: source_has_alpha,
        temporary_backing_color: alpha_backing.map(to_u8),
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
    preview_background: [f32; 3],
    hierarchical_topology: HierarchicalTopologySummary,
    edge_roles: EdgeSummary,
    segmentation: SegmentationSummary,
    structural: StructuralSummary,
    ownership: BoundaryOwnershipSummary,
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
    preview_background: [f32; 3],
    config: &Config,
) -> Result<CoreVectorization> {
    let started = Instant::now();
    let mut checkpoint = started;
    save_pipeline_diagnostic("source", &processing);
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
    let (paint_reference, structural_candidates) = analyse_structural(&processing, &mut roles);
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
                    let lab = rgb_to_lab(pixel);
                    [lab.l, lab.a, lab.b]
                })
                .collect(),
        );
        save_pipeline_diagnostic("smoothed-lab", &smoothed_lab);
    }
    report_progress(config, "perceptual-smoothing", started, &mut checkpoint);
    let mut segmentation = segment(&smoothed, &roles, config);
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
        if barrier {
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
    let refined_structural_ownership = refine_thin_paint_ownership(
        &paint_reference,
        &mut segmentation,
        &structural_candidates.paint_ownership_mask,
        &structural_candidates.source_line_mask,
    );
    save_label_diagnostic(
        "thin-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(config, "thin-paint-ownership", started, &mut checkpoint);
    let ridge_analysis = crate::ridge::analyze(&segmentation.canonical);
    segmentation.paint_samples = crate::ridge::adjust_paint_samples_from_analysis(
        &segmentation.canonical,
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
    // Paint residuals are represented as transparent layers on their owning
    // face.  Splitting those residuals into ordinary labels would turn a
    // smooth tone correction back into a hard shared-boundary staircase.
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
    let (mut paints, mut gradient_report) = fit_all_without_topology(
        &paint_reference,
        &processing,
        &segmentation,
        &strong_branches,
        config,
    );
    report_progress(config, "paint-fitting", started, &mut checkpoint);
    let supported_paint_merges = merge_source_supported_paints(
        &paint_reference,
        &processing,
        &mut segmentation,
        &mut paints,
        config,
    );
    report_progress(
        config,
        "source-supported-paint-merge",
        started,
        &mut checkpoint,
    );
    let exact_paint_merges =
        merge_exact_final_paints(&paint_reference, &mut segmentation, &mut paints);
    if supported_paint_merges.merges > 0 || exact_paint_merges > 0 {
        refresh_summary(&mut gradient_report, &paints);
    }
    save_label_diagnostic(
        "paint-merged-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    gradient_report.source_supported_paint_merges = supported_paint_merges.merges;
    gradient_report.source_supported_boundary_edges_removed =
        supported_paint_merges.boundary_edges_removed;
    let excluded_regions = chroma_matte
        .map(|matte| chroma::background_regions(&segmentation.labels, paints.len(), matte))
        .unwrap_or_default();
    let removed_background_regions = excluded_regions.iter().filter(|&&removed| removed).count();
    let topology = HierarchicalTopology::build(&segmentation);
    report_progress(config, "exact-paint-merge", started, &mut checkpoint);
    let (geometry, geometry_report) = build_geometry(&segmentation, &topology);
    report_progress(config, "shared-geometry", started, &mut checkpoint);
    // Resolve source ownership against the exact shared Paint partition.
    // Overlap is deliberately absent here: it is a seam underpaint, not an
    // authored Paint or structural owner.
    let paint_render = render_svg_preview(
        (processing.width, processing.height),
        (&geometry, &paints),
        &StructuralInk::empty(),
        0.0,
        false,
        &excluded_regions,
        preview_background,
    )?;
    report_progress(config, "paint-preview", started, &mut checkpoint);
    let optimization = optimization_summary(&geometry, &paints, &geometry_report);
    let mut ownership = resolve_boundary_ownership(
        &processing,
        &paint_render,
        &structural_candidates,
        &geometry_report.paint_junctions,
        config.shared_boundary_overlap,
    );
    if let Some(matte) = chroma_matte {
        ownership
            .structural
            .retain_strokes(|stroke| matte.retains_stroke(&stroke.points));
    }
    ownership.summary.structural_strokes = ownership.structural.strokes.len();
    report_progress(config, "structural-selection", started, &mut checkpoint);
    // The complete preview is report-only. Structural ownership is already
    // authoritative, so the normal conversion path does not render it again.
    #[cfg(feature = "diagnostics")]
    let quality = if config.compute_quality_metrics {
        let residual_render = render_svg_preview(
            (processing.width, processing.height),
            (&geometry, &paints),
            &ownership.structural,
            ownership.paint_overlap,
            excluded_regions.iter().all(|&excluded| !excluded),
            &excluded_regions,
            preview_background,
        )?;
        report_progress(config, "quality-preview", started, &mut checkpoint);
        let quality = crate::metrics::compare(&processing, &residual_render);
        report_progress(config, "quality-metrics", started, &mut checkpoint);
        Some(quality)
    } else {
        None
    };
    #[cfg(not(feature = "diagnostics"))]
    let quality = None;
    let ownership_summary = ownership.summary.clone();
    let paint_overlap = ownership.paint_overlap;
    let structural = ownership.structural;
    let (document, svg_report) = serialize_svg(
        processing.width,
        processing.height,
        &geometry,
        &paints,
        &structural,
        paint_overlap,
        excluded_regions.iter().all(|&excluded| !excluded),
        &excluded_regions,
    );
    report_progress(config, "final-svg", started, &mut checkpoint);
    Ok(CoreVectorization {
        document,
        processing_reference: processing,
        labels: segmentation.labels,
        removed_background_regions,
        preview_background,
        hierarchical_topology: topology.summary,
        edge_roles: roles.summary,
        segmentation: segmentation.summary,
        structural: structural.summary,
        ownership: ownership_summary,
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
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::time::{SystemTime, UNIX_EPOCH};

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
                    mean_lab: rgb_to_lab([0.25, 0.5, 0.75]),
                },
                crate::segment::RegionStats {
                    id: 1,
                    area: 1,
                    min_x: 1,
                    min_y: 0,
                    max_x: 2,
                    max_y: 1,
                    mean_rgb: [0.25, 0.5, 0.75],
                    mean_lab: rgb_to_lab([0.25, 0.5, 0.75]),
                },
            ],
            summary: SegmentationSummary::default(),
        };
        let mut paints = vec![
            Paint::Solid {
                color: [0.25, 0.5, 0.75],
            },
            Paint::Solid {
                color: [0.25, 0.5, 0.75],
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
        assert!(summary.source_alpha.removed_regions >= 2);
        assert!(!summary.chroma_key.enabled);

        let document = fs::read_to_string(&output).unwrap();
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
        assert!(quality.delta_e00_mean.is_finite());
        assert!(quality.delta_e00_p90.is_finite());
        assert!(quality.delta_e00_p99.is_finite());
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
