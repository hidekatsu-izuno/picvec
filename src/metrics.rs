//! Native raster comparison helpers used by tests and development benchmarks.
//! They are deliberately not part of normal conversion output.

use serde::Serialize;

use crate::color::{delta_e_ok, relative_luminance, rgb_to_oklab};
use crate::raster::{percentile, Raster};

#[derive(Clone, Debug, Default, Serialize)]
pub struct QualityMetrics {
    /// Dimensions of the compared rasters, in evaluated pixels.
    pub width: usize,
    pub height: usize,
    /// SVG composite backing; absent for callers comparing arbitrary RGB rasters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison_background: Option<[f32; 3]>,
    pub delta_e_ok_mean: f32,
    pub delta_e_ok_p90: f32,
    pub delta_e_ok_p99: f32,
    /// Single-window SSIM over the complete luminance image.
    pub global_ssim: f32,
    /// Mean uniform-window SSIM on linear luminance, over valid window centres.
    pub local_ssim: f32,
    /// Normally 7; the largest odd window that fits is used for tiny rasters.
    pub local_ssim_window: usize,
    /// Up to eight 64-pixel tiles with the largest mean OKLab error.
    pub worst_tiles: Vec<QualityTile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QualityTile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub delta_e_ok_mean: f32,
    pub delta_e_ok_max: f32,
    /// Window centres in this tile; absent if the tile has no valid centre.
    pub local_ssim: Option<f32>,
}

const TILE_SIZE: usize = 64;

pub(crate) fn compare_on(
    reference: &Raster,
    candidate: &Raster,
    background: [f32; 3],
) -> QualityMetrics {
    let mut quality = compare(reference, candidate);
    quality.comparison_background = Some(background);
    quality
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
        .map(|(&a, &b)| delta_e_ok(rgb_to_oklab(a), rgb_to_oklab(b)))
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
    let columns = reference.width.div_ceil(TILE_SIZE);
    let tile_count = columns * reference.height.div_ceil(TILE_SIZE);
    let (local_ssim, local_ssim_window, tile_ssim) = local_similarity(
        &reference_luma,
        &candidate_luma,
        reference.width,
        reference.height,
    );
    let mut tile_errors = vec![(0.0_f64, 0.0_f32); tile_count];
    for (i, &error) in delta.iter().enumerate() {
        let tile = (i / reference.width / TILE_SIZE) * columns + i % reference.width / TILE_SIZE;
        tile_errors[tile].0 += error as f64;
        tile_errors[tile].1 = tile_errors[tile].1.max(error);
    }
    let mut worst_tiles: Vec<_> = tile_errors
        .into_iter()
        .enumerate()
        .map(|(i, (sum, maximum))| {
            let x = i % columns * TILE_SIZE;
            let y = i / columns * TILE_SIZE;
            let width = (reference.width - x).min(TILE_SIZE);
            let height = (reference.height - y).min(TILE_SIZE);
            let (ssim, count) = tile_ssim[i];
            QualityTile {
                x,
                y,
                width,
                height,
                delta_e_ok_mean: (sum / (width * height) as f64) as f32,
                delta_e_ok_max: maximum,
                local_ssim: (count > 0).then(|| (ssim / count as f64) as f32),
            }
        })
        .collect();
    worst_tiles.sort_by(|a, b| {
        b.delta_e_ok_mean
            .total_cmp(&a.delta_e_ok_mean)
            .then_with(|| a.y.cmp(&b.y))
            .then_with(|| a.x.cmp(&b.x))
    });
    worst_tiles.truncate(8);
    QualityMetrics {
        width: reference.width,
        height: reference.height,
        comparison_background: None,
        delta_e_ok_mean: mean,
        delta_e_ok_p90: percentile(delta.clone(), 0.90),
        delta_e_ok_p99: percentile(delta, 0.99),
        global_ssim,
        local_ssim,
        local_ssim_window,
        worst_tiles,
    }
}

/// Sliding column sums avoid five full-image summed-area tables. Accumulate in
/// f64 so low-contrast windows do not lose variance through cancellation.
fn local_similarity(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
) -> (f32, usize, Vec<(f64, usize)>) {
    let columns = width.div_ceil(TILE_SIZE);
    let mut tiles = vec![(0.0, 0); columns * height.div_ceil(TILE_SIZE)];
    let limit = width.min(height).min(7);
    if limit == 0 {
        return (1.0, 0, tiles);
    }
    let window = if limit % 2 == 0 { limit - 1 } else { limit };
    let radius = window / 2;
    let count = (window * window) as f64;
    let correction = if count > 1.0 {
        count / (count - 1.0)
    } else {
        0.0
    };
    let moments = |i: usize| {
        let x = a[i] as f64;
        let y = b[i] as f64;
        [x, y, x * x, y * y, x * y]
    };
    let mut vertical = vec![[0.0; 5]; width];
    for y in 0..window {
        for (x, column) in vertical.iter_mut().enumerate() {
            let m = moments(y * width + x);
            for k in 0..5 {
                column[k] += m[k];
            }
        }
    }
    let mut total = 0.0;
    let mut centres = 0;
    for y in radius..height - radius {
        if y > radius {
            for (x, column) in vertical.iter_mut().enumerate() {
                let old = moments((y - radius - 1) * width + x);
                let new = moments((y + radius) * width + x);
                for k in 0..5 {
                    column[k] += new[k] - old[k];
                }
            }
        }
        let mut sum = [0.0; 5];
        for column in &vertical[..window] {
            for k in 0..5 {
                sum[k] += column[k];
            }
        }
        for x in radius..width - radius {
            if x > radius {
                for k in 0..5 {
                    sum[k] += vertical[x + radius][k] - vertical[x - radius - 1][k];
                }
            }
            let mx = sum[0] / count;
            let my = sum[1] / count;
            let vx = (sum[2] / count - mx * mx).max(0.0) * correction;
            let vy = (sum[3] / count - my * my).max(0.0) * correction;
            let covariance = (sum[4] / count - mx * my) * correction;
            let value = ((2.0 * mx * my + 0.0001) * (2.0 * covariance + 0.0009))
                / ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009));
            total += value;
            centres += 1;
            let tile = y / TILE_SIZE * columns + x / TILE_SIZE;
            tiles[tile].0 += value;
            tiles[tile].1 += 1;
        }
    }
    ((total / centres as f64) as f32, window, tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_ssim_matches_independent_window_statistics() {
        for (width, height) in [(1, 1), (2, 9), (3, 2), (9, 11), (21, 17), (131, 69)] {
            let a: Vec<_> = (0..width * height)
                .map(|i| ((i * 19 % 103) as f32 / 103.0).powi(2))
                .collect();
            let b: Vec<_> = a
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 17 == 0 { v * 0.7 } else { v })
                .collect();
            let (actual, window, _) = local_similarity(&a, &b, width, height);
            let mut expected = 0.0;
            let mut n = 0;
            for y in 0..=height - window {
                for x in 0..=width - window {
                    let mut values = Vec::new();
                    for dy in 0..window {
                        for dx in 0..window {
                            let i = (y + dy) * width + x + dx;
                            values.push((a[i] as f64, b[i] as f64));
                        }
                    }
                    let count = values.len() as f64;
                    let mx = values.iter().map(|p| p.0).sum::<f64>() / count;
                    let my = values.iter().map(|p| p.1).sum::<f64>() / count;
                    let divisor = (count - 1.0).max(1.0);
                    let vx = values.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>() / divisor;
                    let vy = values.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>() / divisor;
                    let cov = values.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / divisor;
                    expected += ((2.0 * mx * my + 0.0001) * (2.0 * cov + 0.0009))
                        / ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009));
                    n += 1;
                }
            }
            assert!(
                (actual as f64 - expected / n as f64).abs() < 1e-6,
                "{width}x{height}: {actual} vs {}",
                expected / n as f64
            );
            let identity = local_similarity(&a, &a, width, height).0;
            assert!((identity - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn diagnostic_tiles_locate_damage_and_clip_to_image_bounds() {
        let reference = Raster::blank(130, 70, [0.5; 3]);
        let mut candidate = reference.clone();
        for y in 64..70 {
            for x in 128..130 {
                candidate.pixels[y * 130 + x] = [0.0; 3];
            }
        }
        let quality = compare_on(&reference, &candidate, [1.0; 3]);
        assert_eq!((quality.width, quality.height), (130, 70));
        assert_eq!(quality.comparison_background, Some([1.0; 3]));
        assert_eq!(quality.local_ssim_window, 7);
        assert!(quality.local_ssim < 1.0);
        let worst = &quality.worst_tiles[0];
        assert_eq!(
            (worst.x, worst.y, worst.width, worst.height),
            (128, 64, 2, 6)
        );
        assert!(worst.delta_e_ok_mean > quality.delta_e_ok_mean * 100.0);
        // This narrow edge tile has no valid 7x7 window centre.
        assert!(worst.local_ssim.is_none());
        let identical = compare(&reference, &reference);
        assert_eq!(identical.delta_e_ok_mean, 0.0);
        assert_eq!(identical.local_ssim, 1.0);
        assert!(identical
            .worst_tiles
            .iter()
            .all(|t| t.delta_e_ok_max == 0.0));
    }
}
