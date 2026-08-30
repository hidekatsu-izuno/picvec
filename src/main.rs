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
    /// Print an in-memory diagnostic report to stderr; no sidecar is written.
    #[arg(long)]
    verbose: bool,
}

fn run() -> picvec::Result<()> {
    let arguments = Arguments::parse();
    let maximum = arguments.max_dimension.max(64);
    let defaults = Config::default();
    let config = Config {
        maximum_dimension: maximum,
        auto_dimension: true,
        auto_minimum_dimension: defaults.auto_minimum_dimension.min(maximum),
        auto_maximum_dimension: maximum,
        smoothing_radius: arguments.smoothing_radius,
        segmentation_min_size: arguments.segmentation_min_size.max(1),
        quantization_dark_delta_e: arguments.quantization_dark_delta_e.max(0.1),
        quantization_light_delta_e: arguments.quantization_light_delta_e.max(0.1),
        gradient_merge_error: arguments.gradient_merge_error.max(0.0),
        paint_primary_sample_budget: arguments.paint_primary_samples.max(8),
        paint_primary_min_region_density: arguments.paint_primary_min_region_density.max(0.0),
        paint_primary_min_explained_variance: arguments.paint_primary_threshold.clamp(0.0, 1.0),
        paint_primary_small_min_explained_variance: arguments
            .paint_primary_small_threshold
            .clamp(0.0, 1.0),
        rayon_threads: arguments.threads,
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
