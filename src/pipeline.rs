use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use std::collections::HashMap;

use crate::color::{rgb_to_lab, Lab};
use crate::config::Config;
use crate::edge::{classify, dilate_square, perceptual_smooth, EdgeSummary};
use crate::geometry::{build as build_geometry, GeometrySummary};
use crate::gradient::{fit_all, merge_partition, GradientSummary};
use crate::metrics::QualityMetrics;
use crate::optimize::{summarize as optimization_summary, OptimizationSummary};
use crate::raster::Raster;
use crate::segment::{
    refine_thin_paint_ownership, regularize_boundaries, segment, split_adaptive_paint_patches,
    SegmentationSummary,
};
use crate::structural::{
    analyse as analyse_structural, select_missing_with_junctions as select_missing_structural,
    StructuralInk, StructuralSummary,
};
use crate::svg::{write as write_svg, SvgSummary};
use crate::{Error, Result};

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
    pub complexity: ComplexityProbe,
    pub edge_roles: EdgeSummary,
    pub segmentation: SegmentationSummary,
    pub structural: StructuralSummary,
    pub gradients: GradientSummary,
    pub geometry: GeometrySummary,
    pub optimization: OptimizationSummary,
    pub svg: SvgSummary,
    pub quality: QualityMetrics,
}

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

fn save_mask_diagnostic(name: &str, mask: &[bool], width: usize, height: usize) {
    let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") else {
        return;
    };
    let values: Vec<u8> = mask.iter().map(|&value| u8::from(value)).collect();
    let path = PathBuf::from(format!("{prefix}-{name}-{width}x{height}.u8"));
    let _ = fs::write(path, values);
}

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
    let edge_cut = crate::raster::percentile(magnitude.clone(), 0.85);
    let edge_density = magnitude
        .iter()
        .filter(|&&value| value >= edge_cut.max(1e-4))
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
    let target_pixels = 1_200_000.0 + 800_000.0 * complexity;
    let source_pixels = (image.width * image.height).max(1) as f32;
    let estimated = image.width.max(image.height) as f32 * (target_pixels / source_pixels).sqrt();
    let selected = estimated
        .round()
        .clamp(
            config.auto_minimum_dimension.max(64) as f32,
            config
                .auto_maximum_dimension
                .max(config.auto_minimum_dimension) as f32,
        )
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

fn temporary_path(output: &Path) -> Result<PathBuf> {
    let file_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| -> Error { "output path must have a UTF-8 file name".into() })?;
    Ok(output.with_file_name(format!(".{file_name}.picvec-tmp")))
}

fn render_svg_preview(
    output: &Path,
    dimensions: (usize, usize),
    paint_layer: (
        &[crate::geometry::RegionGeometry],
        &[crate::gradient::Paint],
    ),
    structural: &StructuralInk,
    config: &Config,
    suffix: &str,
    overlap: bool,
) -> Result<Raster> {
    let (width, height) = dimensions;
    let (geometry, paints) = paint_layer;
    let file_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| -> Error { "output path must have a UTF-8 file name".into() })?;
    let base_path = output.with_file_name(format!(".{file_name}.picvec-{suffix}.svg"));
    let mut base_config = config.clone();
    if !overlap {
        base_config.shared_boundary_overlap = 0.0;
    }
    if let Err(error) = write_svg(
        &base_path,
        width,
        height,
        geometry,
        paints,
        structural,
        &base_config,
    ) {
        let _ = fs::remove_file(&base_path);
        return Err(error);
    }
    let rendered = Command::new("rsvg-convert")
        .arg("--width")
        .arg(width.to_string())
        .arg("--height")
        .arg(height.to_string())
        .arg(&base_path)
        .output();
    let _ = fs::remove_file(&base_path);
    let rendered = rendered?;
    if !rendered.status.success() {
        return Err(format!(
            "rsvg-convert failed while validating the SVG preview: {}",
            String::from_utf8_lossy(&rendered.stderr)
        )
        .into());
    }
    let decoded = image::load_from_memory(&rendered.stdout)?;
    Ok(Raster::from_dynamic(&decoded))
}

/// Convert one raster into exactly the SVG path requested by the caller.
/// No source copy, rendered PNG, or JSON sidecar is produced.
pub fn vectorize(input: &Path, output: &Path, config: &Config) -> Result<Summary> {
    if output
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| !value.eq_ignore_ascii_case("svg"))
        .unwrap_or(true)
    {
        return Err("output path must end in .svg".into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let started = Instant::now();
    let mut checkpoint = started;
    let input_image = Raster::load(input)?;
    let complexity = if config.auto_dimension {
        estimate_dimension(&input_image, config)
    } else {
        ComplexityProbe {
            selected_dimension: config
                .maximum_dimension
                .min(input_image.width.max(input_image.height) as u32),
            ..ComplexityProbe::default()
        }
    };
    let processing = input_image.resize_max(complexity.selected_dimension.max(64));
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
    refine_thin_paint_ownership(
        &paint_reference,
        &mut segmentation,
        &structural_candidates.paint_ownership_mask,
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
    let strong_branches =
        crate::ridge::strong_branches_from_analysis(&segmentation.canonical, &ridge_analysis);
    drop(ridge_analysis);
    save_mask_diagnostic(
        "fitted-paint-samples",
        &segmentation.paint_samples,
        processing.width,
        processing.height,
    );
    split_adaptive_paint_patches(&paint_reference, &processing, &mut segmentation);
    save_label_diagnostic(
        "final-labels",
        &segmentation.labels,
        processing.width,
        processing.height,
    );
    report_progress(config, "adaptive-paint-patches", started, &mut checkpoint);
    let (geometry, geometry_report) = build_geometry(&segmentation);
    report_progress(config, "shared-geometry", started, &mut checkpoint);
    let (paints, gradient_report) = fit_all(
        &paint_reference,
        &processing,
        &segmentation,
        &strong_branches,
        config,
    );
    report_progress(config, "paint-fitting", started, &mut checkpoint);
    let paint_render = render_svg_preview(
        output,
        (processing.width, processing.height),
        (&geometry, &paints),
        &StructuralInk::empty(),
        config,
        "base",
        false,
    )?;
    report_progress(config, "paint-preview", started, &mut checkpoint);
    let optimization = optimization_summary(&geometry, &paints, &geometry_report);
    let residual_structural = select_missing_structural(
        &processing,
        &paint_render,
        &structural_candidates,
        &geometry_report.paint_junctions,
    );
    report_progress(config, "structural-selection", started, &mut checkpoint);
    let residual_render = render_svg_preview(
        output,
        (processing.width, processing.height),
        (&geometry, &paints),
        &residual_structural,
        config,
        "residual-ink",
        true,
    )?;
    report_progress(config, "residual-preview", started, &mut checkpoint);
    let quality = crate::metrics::compare(&processing, &residual_render);
    report_progress(config, "quality-selection", started, &mut checkpoint);
    // Python's ownership model always emits the residual structural layer.
    // A global mean-error contest against the complete candidate overlay can
    // prefer hundreds of duplicate boundary strokes because they improve a
    // few dark pixels while visibly thickening otherwise correct interfaces.
    let structural = residual_structural;
    let temporary = temporary_path(output)?;
    let svg_report = match write_svg(
        &temporary,
        processing.width,
        processing.height,
        &geometry,
        &paints,
        &structural,
        config,
    ) {
        Ok(report) => report,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    report_progress(config, "final-svg", started, &mut checkpoint);
    if output.exists() {
        fs::remove_file(output)?;
    }
    fs::rename(&temporary, output)?;
    Ok(Summary {
        input_width: input_image.width,
        input_height: input_image.height,
        processing_width: processing.width,
        processing_height: processing.height,
        output: output.to_path_buf(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        complexity,
        edge_roles: roles.summary,
        segmentation: segmentation.summary,
        structural: structural.summary,
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
    use std::time::{SystemTime, UNIX_EPOCH};

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
        vectorize(
            &input,
            &output,
            &Config {
                segmentation_min_size: 2,
                minimum_gradient_area: 8,
                ..Config::default()
            },
        )
        .unwrap();
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
}
