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
    /// Remove an automatically detected red/green/blue/cyan/magenta/yellow background.
    #[arg(long)]
    remove_chroma_key_background: bool,
    /// Disable source-resolution rate-distortion refinement.
    #[arg(long)]
    no_adaptive_refinement: bool,
    /// Maximum additional SVG size for adaptive regions in MiB (0: unlimited).
    #[arg(long, default_value_t = Config::default().adaptive_svg_budget_bytes / (1024 * 1024))]
    adaptive_svg_budget_mib: usize,
    /// Additional passes can remove more internal contours at an extra time cost (1..=8).
    #[arg(long, default_value_t = 1)]
    paint_merge_passes: usize,
    /// Palette tolerance multiplier in units of 100 * OKLab distance.
    #[arg(long, default_value_t = Config::default().oklab_palette_threshold_scale)]
    oklab_palette_threshold_scale: f32,
    /// Rayon workers; zero selects min(4, half the detected CPU count).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Render the completed SVG and report 100-scaled OKLab distance/SSIM diagnostics.
    #[cfg(feature = "diagnostics")]
    #[arg(long)]
    quality_metrics: bool,
    /// Print an in-memory diagnostic report to stderr; no sidecar is written.
    #[cfg(feature = "diagnostics")]
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
    let adaptive_svg_budget_bytes = arguments
        .adaptive_svg_budget_mib
        .checked_mul(1024 * 1024)
        .ok_or("adaptive_svg_budget_mib is too large")?;
    let defaults = Config::default();
    #[cfg(feature = "diagnostics")]
    let (compute_quality_metrics, retain_diagnostics) =
        (arguments.quality_metrics, arguments.verbose);
    #[cfg(not(feature = "diagnostics"))]
    let (compute_quality_metrics, retain_diagnostics) = (false, false);
    let config = Config {
        maximum_input_dimension: arguments.max_input_dimension,
        maximum_input_pixels,
        maximum_decode_bytes,
        maximum_dimension: maximum,
        auto_dimension: true,
        auto_minimum_dimension: defaults.auto_minimum_dimension.min(maximum),
        auto_maximum_dimension: maximum,
        remove_chroma_key_background: arguments.remove_chroma_key_background,
        adaptive_refinement: !arguments.no_adaptive_refinement,
        adaptive_svg_budget_bytes,
        paint_merge_passes: arguments.paint_merge_passes,
        oklab_palette_threshold_scale: arguments.oklab_palette_threshold_scale,
        rayon_threads: arguments.threads,
        compute_quality_metrics,
        retain_diagnostics,
        ..defaults
    };
    let summary = vectorize(&arguments.input, &arguments.output_svg, &config)?;
    #[cfg(feature = "diagnostics")]
    {
        if arguments.verbose {
            eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        } else if let Some(quality) = &summary.quality {
            eprintln!("{}", serde_json::to_string_pretty(quality)?);
        }
    }
    eprintln!(
        "wrote {} ({}x{}, {} regions, {:.3}s)",
        summary.output.display(),
        summary.processing_width,
        summary.processing_height,
        summary.geometry.regions,
        summary.elapsed_seconds,
    );
    if summary.adaptive_refinement.rejected_for_budget > 0 {
        eprintln!("kept {} regions at base resolution: the {} MiB adaptive SVG budget was exhausted; increase --adaptive-svg-budget-mib or set it to 0 (unlimited) to retain more detail",
            summary.adaptive_refinement.rejected_for_budget, arguments.adaptive_svg_budget_mib);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("picvec: {error}");
        std::process::exit(1);
    }
}
