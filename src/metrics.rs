//! Native raster comparison helpers used by tests and development benchmarks.
//! They are deliberately not part of normal conversion output.

use serde::Serialize;

use crate::color::{delta_e2000, relative_luminance, rgb_to_lab};
use crate::raster::{percentile, Raster};

#[derive(Clone, Debug, Default, Serialize)]
pub struct QualityMetrics {
    pub delta_e00_mean: f32,
    pub delta_e00_p90: f32,
    pub delta_e00_p99: f32,
    /// Single-window SSIM over the complete luminance image.
    pub global_ssim: f32,
}

pub fn compare(reference: &Raster, candidate: &Raster) -> QualityMetrics {
    assert_eq!(
        (reference.width, reference.height),
        (candidate.width, candidate.height)
    );
    let delta: Vec<f32> = reference
        .pixels
        .iter()
        .zip(&candidate.pixels)
        .map(|(&a, &b)| delta_e2000(rgb_to_lab(a), rgb_to_lab(b)))
        .collect();
    let mean = delta.iter().sum::<f32>() / delta.len().max(1) as f32;
    let reference_luma: Vec<f32> = reference
        .pixels
        .iter()
        .copied()
        .map(relative_luminance)
        .collect();
    let candidate_luma: Vec<f32> = candidate
        .pixels
        .iter()
        .copied()
        .map(relative_luminance)
        .collect();
    let mean_x = reference_luma.iter().sum::<f32>() / reference_luma.len().max(1) as f32;
    let mean_y = candidate_luma.iter().sum::<f32>() / candidate_luma.len().max(1) as f32;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    let mut covariance = 0.0;
    for (&x, &y) in reference_luma.iter().zip(&candidate_luma) {
        variance_x += (x - mean_x).powi(2);
        variance_y += (y - mean_y).powi(2);
        covariance += (x - mean_x) * (y - mean_y);
    }
    let divisor = reference_luma.len().saturating_sub(1).max(1) as f32;
    variance_x /= divisor;
    variance_y /= divisor;
    covariance /= divisor;
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);
    let global_ssim = ((2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2))
        / ((mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)).max(1e-12);
    QualityMetrics {
        delta_e00_mean: mean,
        delta_e00_p90: percentile(delta.clone(), 0.90),
        delta_e00_p99: percentile(delta, 0.99),
        global_ssim,
    }
}
