use std::path::PathBuf;

use clap::Parser;
use picvec::{vectorize, Config};

#[derive(Debug, Parser)]
#[command(
    name = "picvec",
    version,
    about = "Perceptual raster to editable SVG converter"
)]
struct Arguments {
    /// Input raster image.
    input: PathBuf,
    /// Exact SVG file to create.
    output_svg: PathBuf,
    /// Upper bound for automatic input sizing.
    #[arg(long, default_value_t = 1600)]
    max_dimension: u32,
    /// Reject source rasters exceeding this width or height before decoding.
    #[arg(long, default_value_t = 32_768)]
    max_input_dimension: u32,
    /// Reject source rasters exceeding this total size before decoding.
    #[arg(long, default_value_t = 32)]
    max_input_megapixels: u64,
    /// Best-effort image-decoder allocation limit in MiB.
    #[arg(long, default_value_t = 512)]
    max_decode_mib: u64,
    #[arg(long, default_value_t = 4)]
    smoothing_radius: u32,
    #[arg(long, default_value_t = 24)]
    segmentation_min_size: u32,
    #[arg(long, default_value_t = 2.5)]
    quantization_dark_delta_e: f32,
    #[arg(long, default_value_t = 5.0)]
    quantization_light_delta_e: f32,
    #[arg(long, default_value_t = 2.3)]
    gradient_merge_error: f32,
    /// Samples used by the inexpensive Paint coherence gate.
    #[arg(long, default_value_t = 64)]
    paint_primary_samples: usize,
    /// Final-region density required to enable the Paint coherence gate.
    #[arg(long, default_value_t = 0.015)]
    paint_primary_min_region_density: f32,
    /// Spatial coherence required before a normal face runs full Paint fit.
    #[arg(long, default_value_t = 0.06)]
    paint_primary_threshold: f32,
    /// Spatial coherence required for a face below minimum gradient area.
    #[arg(long, default_value_t = 0.16)]
    paint_primary_small_threshold: f32,
    /// Rayon workers; zero selects the detected physical core count.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Path or command name for librsvg's renderer.
    #[arg(long, default_value = "rsvg-convert")]
    rsvg_convert: PathBuf,
    /// Print an in-memory diagnostic report to stderr; no sidecar is written.
    #[arg(long)]
    verbose: bool,
}

fn run() -> picvec::Result<()> {
    let arguments = Arguments::parse();
    let maximum = arguments.max_dimension;
    let maximum_decode_bytes = arguments
        .max_decode_mib
        .checked_mul(1024 * 1024)
        .ok_or("max_decode_mib is too large")?;
    let maximum_input_pixels = arguments
        .max_input_megapixels
        .checked_mul(1_000_000)
        .ok_or("max_input_megapixels is too large")?;
    let defaults = Config::default();
    let config = Config {
        maximum_input_dimension: arguments.max_input_dimension,
        maximum_input_pixels,
        maximum_decode_bytes,
        maximum_dimension: maximum,
        auto_dimension: true,
        auto_minimum_dimension: defaults.auto_minimum_dimension.min(maximum),
        auto_maximum_dimension: maximum,
        smoothing_radius: arguments.smoothing_radius,
        segmentation_min_size: arguments.segmentation_min_size,
        quantization_dark_delta_e: arguments.quantization_dark_delta_e,
        quantization_light_delta_e: arguments.quantization_light_delta_e,
        gradient_merge_error: arguments.gradient_merge_error,
        paint_primary_sample_budget: arguments.paint_primary_samples,
        paint_primary_min_region_density: arguments.paint_primary_min_region_density,
        paint_primary_min_explained_variance: arguments.paint_primary_threshold,
        paint_primary_small_min_explained_variance: arguments.paint_primary_small_threshold,
        rayon_threads: arguments.threads,
        rsvg_convert: arguments.rsvg_convert,
        retain_diagnostics: arguments.verbose,
        ..defaults
    };
    let summary = vectorize(&arguments.input, &arguments.output_svg, &config)?;
    if arguments.verbose {
        eprintln!("{}", serde_json::to_string_pretty(&summary)?);
    }
    eprintln!(
        "wrote {} ({}x{}, {} regions, {:.3}s)",
        summary.output.display(),
        summary.processing_width,
        summary.processing_height,
        summary.geometry.regions,
        summary.elapsed_seconds,
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("picvec: {error}");
        std::process::exit(1);
    }
}
