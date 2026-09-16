use crate::svg_document::Document;
use crate::time::Instant;
#[cfg(any(not(target_arch = "wasm32"), feature = "diagnostics"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;

use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use tempfile::{Builder as TemporaryFileBuilder, NamedTempFile};

use std::collections::HashMap;

use crate::adaptive::{
    compose_refinements, matching_refinement_core, perceptual_score, plan_candidates,
    refinements_cover_canvas, AdaptiveRefinementSummary, EmbeddedRefinement, PerceptualScore,
    SourceRect,
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

#[cfg(not(target_arch = "wasm32"))]
fn output_parent(output: &Path) -> &Path {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(not(target_arch = "wasm32"))]
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

/// Render a document representing `document_source` directly onto a native
/// source-space crop. Both axes use the actual document size; a crop may have
/// a different origin and independently rounded processing dimensions.
fn render_source_region(
    tree: &resvg::usvg::Tree,
    document_source: SourceRect,
    region: SourceRect,
    background: [f32; 3],
) -> Result<Raster> {
    render_svg_tree_on(
        tree,
        region.width,
        region.height,
        resvg::tiny_skia::Transform::from_row(
            document_source.width as f32 / tree.size().width(),
            0.0,
            0.0,
            document_source.height as f32 / tree.size().height(),
            document_source.x as f32 - region.x as f32,
            document_source.y as f32 - region.y as f32,
        ),
        background,
    )
}

struct RefinementComparison {
    core: SourceRect,
    baseline: PerceptualScore,
    refined: PerceptualScore,
    boundary_matches: bool,
}

impl RefinementComparison {
    fn gain(&self) -> f32 {
        self.baseline.combined - self.refined.combined
    }

    fn improves(&self, config: &Config) -> bool {
        self.boundary_matches
            && self.gain() >= config.adaptive_min_perceptual_gain
            && self.refined.p90_delta_e <= self.baseline.p90_delta_e + 0.25
            && self.refined.missing_edge_fraction <= self.baseline.missing_edge_fraction + 0.025
    }
}

/// The coarse planning preview must not decide the measured gain. Otherwise
/// its interpolation blur looks like a vector defect, and a higher-resolution
/// copy of the very same geometry can appear to improve the result.
#[allow(clippy::too_many_arguments)]
fn compare_refinement<S: RasterSource + ?Sized>(
    source: &S,
    matte: Option<&AlphaMatte>,
    base_tree: &resvg::usvg::Tree,
    child_tree: &resvg::usvg::Tree,
    proposed_core: SourceRect,
    expanded: SourceRect,
    background: [f32; 3],
) -> Result<RefinementComparison> {
    let whole = SourceRect {
        x: 0,
        y: 0,
        width: source.width(),
        height: source.height(),
    };
    // Reuse these two crops for the join check and both quality scores. Never
    // allocate a complete source-size canvas merely to validate a local patch.
    let baseline = render_source_region(base_tree, whole, expanded, background)?;
    let refined = render_source_region(child_tree, expanded, expanded, background)?;
    let matched_core = matching_refinement_core(
        &baseline,
        &refined,
        expanded,
        whole,
        proposed_core,
        expanded,
        matte,
    );
    let core = matched_core.unwrap_or(proposed_core);
    Ok(RefinementComparison {
        core,
        baseline: perceptual_score(source, core, &baseline, expanded),
        refined: perceptual_score(source, core, &refined, expanded),
        boundary_matches: matched_core.is_some(),
    })
}

/// Convert one raster into exactly the SVG path requested by the caller.
/// No source copy, rendered PNG, or JSON sidecar is produced.
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn default_execution_thread_count(cpu_count: usize) -> usize {
    (cpu_count / 2).clamp(1, 10)
}

#[cfg(not(target_arch = "wasm32"))]
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
    let base_tree = parse_svg_document(&core.document)?;
    let base_render = render_svg_tree_on(
        &base_tree,
        core.processing_reference.width,
        core.processing_reference.height,
        resvg::tiny_skia::Transform::identity(),
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
    // Reuse the parsed base for source-resolution join and quality validation.
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
                let child_tree = parse_svg_document(&child.document)?;
                let comparison = compare_refinement(
                    reference_source,
                    source_matte,
                    &base_tree,
                    &child_tree,
                    candidate.core,
                    expanded,
                    core.preview_background,
                )?;
                let core = comparison.core;
                let baseline = comparison.baseline;
                let refined = comparison.refined;
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
                        comparison.boundary_matches,
                        child.svg.bytes,
                    );
                }
                if !comparison.improves(config) {
                    return Ok(RefinementOutcome::QualityRejected);
                }
                let rate = candidate.measured_rate(comparison.gain(), core.area(), child.svg.bytes);
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
            core.quality = Some(crate::metrics::compare_on(
                &core.processing_reference,
                &final_render,
                core.preview_background,
            ));
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = final_render;
    } else {
        parse_svg_document(&core.document)?;
    }
    Ok(summary)
}

#[cfg(not(target_arch = "wasm32"))]
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
    let (document, mut summary) =
        vectorize_decoded(decoded, decoded_alpha, config, execution_threads, started)?;
    let temporary = temporary_svg(output, "output")?;
    fs::write(temporary.path(), document.as_bytes())?;
    temporary
        .persist(output)
        .map_err(|error| -> Error { error.error.into() })?;
    summary.output = output.to_path_buf();
    summary.elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(summary)
}

/// Convert encoded PNG/JPEG bytes entirely in memory. `Summary::output` is empty.
pub fn vectorize_bytes(input: &[u8], config: &Config) -> Result<(String, Summary)> {
    config.validate()?;
    let run = || {
        let started = Instant::now();
        let decoded = Raster::decode_reader(
            image::ImageReader::new(std::io::Cursor::new(input)).with_guessed_format()?,
            config.maximum_input_dimension,
            config.maximum_input_pixels,
            config.maximum_decode_bytes,
        )?;
        let (source, alpha) = SourceRaster::from_decoded(decoded);
        vectorize_decoded(source, alpha, config, rayon::current_num_threads(), started)
    };
    #[cfg(target_arch = "wasm32")]
    {
        run()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(execution_thread_count(config))
            .build()?;
        pool.install(run)
    }
}

fn vectorize_decoded(
    decoded: SourceRaster,
    decoded_alpha: Option<Vec<u8>>,
    config: &Config,
    execution_threads: usize,
    started: Instant,
) -> Result<(String, Summary)> {
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
        core.quality = Some(crate::metrics::compare_on(
            &reference,
            &rendered,
            preview_background,
        ));
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
    (core.svg.objects, core.svg.path_subpaths) = core.document.counts();
    Ok((
        core.document.to_string(),
        Summary {
            input_width,
            input_height,
            processing_width,
            processing_height,
            output: PathBuf::new(),
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
        },
    ))
}

struct CoreVectorization {
    document: Document,
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
    let mut validation_fragments = None;
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
        if !crate::paint_order::validate_cached(
            &reference,
            &document,
            &processing,
            chroma_matte,
            &segmentation.labels,
            &mut order_proposal.summary,
            &mut validation_fragments,
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
        let (filled, report, removed) = crate::occlusion::simplify_cached(
            &mut geometry,
            (document, svg_report),
            &segmentation.labels,
            processing.width,
            material_alpha.as_ref().or(chroma_matte),
            validation_fragments.take(),
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
    drop(validation_fragments);
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
        Some(crate::metrics::compare_on(
            &processing_reference,
            &final_render,
            preview_background,
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
include!("../tests/unit/pipeline.rs");
