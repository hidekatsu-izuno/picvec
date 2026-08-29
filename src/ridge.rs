//! Final-paint ridge sampling translated from `line_ridges.py` and
//! `_ridge_paint_sample_mask`.
//!
//! The segmentation pass intentionally excludes antialiased boundary pixels
//! from Paint fitting.  A thin line can otherwise lose every centre sample,
//! so the Python reference detects dark/bright ridge centres once more on the
//! final canonical geometry, removes ridge shoulders, and restores only local
//! L* extrema.  This module preserves that ordering.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::color::{delta_e2000, Lab};
use crate::edge::{lab_pixels, preprocess_lab_pixels};
use crate::raster::Raster;

#[derive(Clone, Debug)]
pub struct RidgeEvidence {
    pub dark_response: Vec<f32>,
    pub bright_response: Vec<f32>,
    pub dark_mask: Vec<bool>,
    pub bright_mask: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct StrongRidgeBranches {
    pub dark: Vec<bool>,
    pub bright: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct RidgeAnalysis {
    evidence: RidgeEvidence,
    labs: Vec<Lab>,
}

pub fn analyze(image: &Raster) -> RidgeAnalysis {
    RidgeAnalysis {
        evidence: detect(image),
        labs: lab_pixels(image),
    }
}

#[inline]
fn reflect_index(mut value: isize, length: usize) -> usize {
    let length = length as isize;
    while value < 0 || value >= length {
        if value < 0 {
            value = -value - 1;
        } else {
            value = 2 * length - value - 1;
        }
    }
    value as usize
}

fn numpy_sum_f64(values: &[f64]) -> f64 {
    if values.len() < 8 {
        return values.iter().fold(-0.0_f64, |sum, &value| sum + value);
    }
    if values.len() <= 128 {
        let mut partial = [0.0_f64; 8];
        partial.copy_from_slice(&values[..8]);
        let remainder_start = values.len() - values.len() % 8;
        let mut index = 8;
        while index < remainder_start {
            for lane in 0..8 {
                partial[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut result = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for &value in &values[remainder_start..] {
            result += value;
        }
        return result;
    }
    let mut middle = values.len() / 2;
    middle -= middle % 8;
    numpy_sum_f64(&values[..middle]) + numpy_sum_f64(&values[middle..])
}

fn gaussian_kernel(sigma: f64, order: usize, truncate: f64) -> Vec<f64> {
    let radius = (truncate * sigma + 0.5) as usize;
    let sigma2 = sigma * sigma;
    let mut phi: Vec<f64> = (-(radius as isize)..=radius as isize)
        .map(|x| -0.5 / sigma2 * (x * x) as f64)
        .collect();
    crate::svml::exp_f64_in_place(&mut phi);
    let total = numpy_sum_f64(&phi);
    phi.iter_mut().for_each(|value| *value /= total);
    if order == 1 {
        for (offset, value) in (-(radius as isize)..=radius as isize).zip(&mut phi) {
            *value *= -(offset as f64) / sigma2;
        }
    }
    // scipy.ndimage.gaussian_filter1d calls correlate1d with the reversed
    // Gaussian kernel.
    phi.reverse();
    phi
}

fn correlate_axis(
    input: &[f32],
    width: usize,
    height: usize,
    axis: usize,
    weights: &[f64],
) -> Vec<f32> {
    let radius = (weights.len() / 2) as isize;
    let mut output = vec![0.0_f32; input.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, output) in row.iter_mut().enumerate() {
                let mut sum = 0.0_f64;
                for (position, &weight) in weights.iter().enumerate() {
                    let offset = position as isize - radius;
                    let (sample_x, sample_y) = if axis == 0 {
                        (x, reflect_index(y as isize + offset, height))
                    } else {
                        (reflect_index(x as isize + offset, width), y)
                    };
                    sum += input[sample_y * width + sample_x] as f64 * weight;
                }
                *output = sum as f32;
            }
        });
    output
}

fn gaussian_filter(
    input: &[f32],
    width: usize,
    height: usize,
    sigma: f64,
    orders: [usize; 2],
) -> Vec<f32> {
    let scaled = sigma / std::f64::consts::SQRT_2;
    let first = gaussian_kernel(scaled, orders[0], 8.0);
    let second = gaussian_kernel(scaled, orders[1], 8.0);
    let intermediate = correlate_axis(input, width, height, 0, &first);
    correlate_axis(&intermediate, width, height, 1, &second)
}

fn meijering(input: &[f32], width: usize, height: usize, sigmas: &[f64]) -> Vec<f32> {
    let mut filtered = vec![0.0_f32; input.len()];
    for &sigma in sigmas {
        let gradient_row = gaussian_filter(input, width, height, sigma, [1, 0]);
        let gradient_column = gaussian_filter(input, width, height, sigma, [0, 1]);
        let hrr = gaussian_filter(&gradient_row, width, height, sigma, [1, 0]);
        let hrc = gaussian_filter(&gradient_row, width, height, sigma, [0, 1]);
        let hcc = gaussian_filter(&gradient_column, width, height, sigma, [0, 1]);
        let values: Vec<f32> = (0..input.len())
            .into_par_iter()
            .map(|index| {
                let centre = (hrr[index] + hcc[index]) / 2.0;
                let difference = (hrr[index] - hcc[index]) / 2.0;
                let half_root = (hrc[index] * hrc[index] + difference * difference).sqrt();
                let first = centre + half_root;
                let second = centre - half_root;
                let first_normalized = first + (1.0_f32 / 3.0) * second;
                let second_normalized = (1.0_f32 / 3.0) * first + second;
                if first_normalized.abs() >= second_normalized.abs() {
                    first_normalized
                } else {
                    second_normalized
                }
                .max(0.0)
            })
            .collect();
        let maximum = values.par_iter().copied().reduce(|| 0.0, f32::max);
        if maximum > 0.0 {
            filtered
                .par_iter_mut()
                .zip(values.into_par_iter())
                .for_each(|(destination, value)| {
                    *destination = destination.max(value / maximum);
                });
        }
    }
    filtered
}

#[doc(hidden)]
pub fn debug_bright_parts(image: &Raster) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let width = image.width;
    let height = image.height;
    let labs = preprocess_lab_pixels(image);
    let luminance: Vec<f32> = labs.iter().map(|value| value.l / 100.0).collect();
    let scale = width.max(height) as f64 / 1024.0;
    let sigmas: Vec<f64> = [1.0, 1.5, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|value| (value * scale).max(0.5))
        .collect();
    let mut radii: Vec<usize> = [2.0, 3.0, 4.0, 6.0]
        .into_iter()
        .map(|value| (value * scale).round().max(1.0) as usize)
        .collect();
    radii.sort_unstable();
    radii.dedup();
    let inverted: Vec<f32> = luminance.iter().map(|&value| -value).collect();
    let bright_hessian = meijering(&inverted, width, height, &sigmas);
    let (_, bright_tophat) = top_hats(&luminance, width, height, &radii);
    (luminance, bright_hessian, bright_tophat)
}

fn disk_offsets(radius: usize) -> Vec<(isize, isize)> {
    let radius_squared = radius * radius;
    let radius = radius as isize;
    let mut offsets = Vec::new();
    for y in -radius..=radius {
        for x in -radius..=radius {
            if (x * x + y * y) as usize <= radius_squared {
                offsets.push((x, y));
            }
        }
    }
    offsets
}

fn morphology(
    input: &[f32],
    width: usize,
    height: usize,
    offsets: &[(isize, isize)],
    dilate: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; input.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, output) in row.iter_mut().enumerate() {
                let mut selected = if dilate {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                for &(offset_x, offset_y) in offsets {
                    let sample_x = reflect_index(x as isize + offset_x, width);
                    let sample_y = reflect_index(y as isize + offset_y, height);
                    let value = input[sample_y * width + sample_x];
                    selected = if dilate {
                        selected.max(value)
                    } else {
                        selected.min(value)
                    };
                }
                *output = selected;
            }
        });
    output
}

fn top_hats(input: &[f32], width: usize, height: usize, radii: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let mut dark = vec![0.0_f32; input.len()];
    let mut bright = vec![0.0_f32; input.len()];
    for &radius in radii {
        let offsets = disk_offsets(radius);
        let eroded = morphology(input, width, height, &offsets, false);
        let opened = morphology(&eroded, width, height, &offsets, true);
        let dilated = morphology(input, width, height, &offsets, true);
        let closed = morphology(&dilated, width, height, &offsets, false);
        dark.par_iter_mut()
            .zip(bright.par_iter_mut())
            .enumerate()
            .for_each(|(index, (dark, bright))| {
                *bright = bright.max(input[index] - opened[index]);
                *dark = dark.max(closed[index] - input[index]);
            });
    }
    (dark, bright)
}

fn percentile_995(values: &[f32]) -> f64 {
    let mut sorted: Vec<f32> = values
        .iter()
        .copied()
        .filter(|&value| value > 1e-8)
        .collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f32::total_cmp);
    let position = 0.995_f64 * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower] as f64
    } else {
        let fraction = position - lower as f64;
        sorted[lower] as f64 * (1.0 - fraction) + sorted[upper] as f64 * fraction
    }
}

fn robust_unit(values: &[f32]) -> Vec<f32> {
    let scale = percentile_995(values).max(1e-8);
    if scale <= 1e-8 {
        return vec![0.0; values.len()];
    }
    values
        .par_iter()
        .map(|&value| ((value.max(0.0) as f64 / scale) as f32).clamp(0.0, 1.0))
        .collect()
}

fn hysteresis(values: &[f32], width: usize, height: usize, low: f32, high: f32) -> Vec<bool> {
    let low_mask: Vec<bool> = values.iter().map(|&value| value > low).collect();
    let high_mask: Vec<bool> = values.iter().map(|&value| value > high).collect();
    let mut result = vec![false; values.len()];
    let mut seen = vec![false; values.len()];
    let neighbours = [(0_isize, -1_isize), (-1, 0), (1, 0), (0, 1)];
    for start in 0..values.len() {
        if seen[start] || !low_mask[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        let mut connected_to_high = false;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            connected_to_high |= high_mask[index];
            let x = index % width;
            let y = index / width;
            for &(dx, dy) in &neighbours {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let neighbour = ny as usize * width + nx as usize;
                if !seen[neighbour] && low_mask[neighbour] {
                    seen[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if connected_to_high {
            for index in component {
                result[index] = true;
            }
        }
    }
    result
}

pub fn detect(image: &Raster) -> RidgeEvidence {
    let width = image.width;
    let height = image.height;
    // ClassicalLineRidgeDetector uses preprocess.srgb_to_lab rather than
    // skimage.rgb2lab.  Their matrices differ enough to move normalized ridge
    // responses across the strong-branch threshold.
    let labs = preprocess_lab_pixels(image);
    let luminance: Vec<f32> = labs.par_iter().map(|value| value.l / 100.0).collect();
    let scale = width.max(height) as f64 / 1024.0;
    let sigmas: Vec<f64> = [1.0, 1.5, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|value| (value * scale).max(0.5))
        .collect();
    let mut radii: Vec<usize> = [2.0, 3.0, 4.0, 6.0]
        .into_iter()
        .map(|value| (value * scale).round().max(1.0) as usize)
        .collect();
    radii.sort_unstable();
    radii.dedup();
    let dark_hessian = meijering(&luminance, width, height, &sigmas);
    let inverted: Vec<f32> = luminance.par_iter().map(|&value| -value).collect();
    let bright_hessian = meijering(&inverted, width, height, &sigmas);
    let (dark_tophat, bright_tophat) = top_hats(&luminance, width, height, &radii);
    let dark_hessian = robust_unit(&dark_hessian);
    let bright_hessian = robust_unit(&bright_hessian);
    let dark_tophat_unit = robust_unit(&dark_tophat);
    let bright_tophat_unit = robust_unit(&bright_tophat);
    let mut dark_response = vec![0.0_f32; luminance.len()];
    let mut bright_response = vec![0.0_f32; luminance.len()];
    dark_response
        .par_iter_mut()
        .zip(bright_response.par_iter_mut())
        .enumerate()
        .for_each(|(index, (dark_response, bright_response))| {
            *dark_response = (dark_hessian[index] * dark_tophat_unit[index]).sqrt();
            *bright_response = (bright_hessian[index] * bright_tophat_unit[index]).sqrt();
            if dark_tophat[index] < 0.015 {
                *dark_response = 0.0;
            }
            if bright_tophat[index] < 0.015 {
                *bright_response = 0.0;
            }
        });
    let dark_mask = hysteresis(&dark_response, width, height, 0.08, 0.20);
    let bright_mask = hysteresis(&bright_response, width, height, 0.08, 0.20);
    RidgeEvidence {
        dark_response,
        bright_response,
        dark_mask,
        bright_mask,
    }
}

fn propagate_black_ridges(
    candidates: &[bool],
    labs: &[Lab],
    width: usize,
    height: usize,
) -> Vec<bool> {
    let mut seeds = vec![false; candidates.len()];
    let mut support = vec![false; candidates.len()];
    for index in 0..candidates.len() {
        if !candidates[index] {
            continue;
        }
        let value = labs[index];
        seeds[index] = delta_e2000(
            value,
            Lab {
                l: 0.0,
                a: 0.0,
                b: 0.0,
            },
        ) <= 2.3;
        support[index] = value.l <= 25.0
            && delta_e2000(
                value,
                Lab {
                    l: value.l,
                    a: 0.0,
                    b: 0.0,
                },
            ) <= 4.6;
    }
    let mut result = seeds.clone();
    let mut queue: VecDeque<usize> = seeds
        .iter()
        .enumerate()
        .filter_map(|(index, &seed)| seed.then_some(index))
        .collect();
    while let Some(index) = queue.pop_front() {
        let x = index % width;
        let y = index / width;
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let neighbour = ny as usize * width + nx as usize;
                if support[neighbour] && !result[neighbour] {
                    result[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
    }
    result
}

fn strong_branch_support(
    ridge: &[bool],
    response: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
    minimum_span: f32,
) -> Vec<bool> {
    let core: Vec<bool> = ridge
        .iter()
        .zip(response)
        .map(|(&is_ridge, &value)| is_ridge && value >= threshold.max(0.0))
        .collect();
    let mut retained = vec![false; ridge.len()];
    let mut seen = vec![false; ridge.len()];
    for start in 0..core.len() {
        if !core[start] || seen[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0_usize;
        let mut max_y = 0_usize;
        seen[start] = true;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let neighbour = ny as usize * width + nx as usize;
                    if core[neighbour] && !seen[neighbour] {
                        seen[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        let span = (max_x - min_x + 1).max(max_y - min_y + 1) as f32;
        if span >= minimum_span.max(0.0) {
            for index in component {
                retained[index] = true;
            }
        }
    }
    if !retained.iter().any(|&value| value) {
        return retained;
    }

    // scipy's EDT condition in the reference is distance <= sqrt(2): this is
    // exactly the immediate 8-neighbourhood of the retained core on a pixel
    // grid.  Read from the immutable core mask so shoulders do not propagate.
    let core_retained = retained.clone();
    for index in 0..ridge.len() {
        if !ridge[index] || core_retained[index] {
            continue;
        }
        let x = index % width;
        let y = index / width;
        'neighbours: for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0
                    && ny >= 0
                    && nx < width as isize
                    && ny < height as isize
                    && core_retained[ny as usize * width + nx as usize]
                {
                    retained[index] = true;
                    break 'neighbours;
                }
            }
        }
    }
    retained
}

/// Return the source-supported long chromatic-dark and bright ridge branches
/// used by the Python Paint fitter to exclude antialiased shoulder samples.
pub fn strong_branches_from_analysis(
    image: &Raster,
    analysis: &RidgeAnalysis,
) -> StrongRidgeBranches {
    let evidence = &analysis.evidence;
    let labs = &analysis.labs;
    let dark_candidates: Vec<bool> = evidence
        .dark_mask
        .iter()
        .zip(evidence.dark_response.iter().zip(&evidence.bright_response))
        .map(|(&mask, (&dark, &bright))| mask && dark > bright)
        .collect();
    let bright_ridges: Vec<bool> = evidence
        .bright_mask
        .iter()
        .zip(&dark_candidates)
        .map(|(&mask, &dark)| mask && !dark)
        .collect();
    let black_ridges = propagate_black_ridges(&dark_candidates, labs, image.width, image.height);
    let chromatic_dark_ridges: Vec<bool> = dark_candidates
        .iter()
        .zip(&black_ridges)
        .zip(labs)
        .map(|((&candidate, &black), lab)| candidate && !black && lab.a.hypot(lab.b) >= 18.0)
        .collect();
    let minimum_span = 14.0 * image.width.max(image.height) as f32 / 1024.0;
    StrongRidgeBranches {
        dark: strong_branch_support(
            &chromatic_dark_ridges,
            &evidence.dark_response,
            image.width,
            image.height,
            0.40,
            minimum_span.max(0.5),
        ),
        bright: strong_branch_support(
            &bright_ridges,
            &evidence.bright_response,
            image.width,
            image.height,
            0.40,
            minimum_span.max(0.5),
        ),
    }
}

pub fn strong_branches(image: &Raster) -> StrongRidgeBranches {
    strong_branches_from_analysis(image, &analyze(image))
}

fn local_extreme(lightness: &[f32], width: usize, height: usize, minimum: bool) -> Vec<f32> {
    let mut result = vec![0.0_f32; lightness.len()];
    result
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, output) in row.iter_mut().enumerate() {
                let mut selected = if minimum {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                        let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                        let value = lightness[sy * width + sx];
                        selected = if minimum {
                            selected.min(value)
                        } else {
                            selected.max(value)
                        };
                    }
                }
                *output = selected;
            }
        });
    result
}

pub fn adjust_paint_samples_from_analysis(
    image: &Raster,
    original: &[bool],
    analysis: &RidgeAnalysis,
) -> Vec<bool> {
    assert_eq!(image.pixels.len(), original.len());
    let evidence = &analysis.evidence;
    let labs = &analysis.labs;
    let dark_candidates: Vec<bool> = evidence
        .dark_mask
        .iter()
        .zip(evidence.dark_response.iter().zip(&evidence.bright_response))
        .map(|(&mask, (&dark, &bright))| mask && dark > bright)
        .collect();
    let bright: Vec<bool> = evidence
        .bright_mask
        .iter()
        .zip(&dark_candidates)
        .map(|(&mask, &dark)| mask && !dark)
        .collect();
    let dark_black = propagate_black_ridges(&dark_candidates, labs, image.width, image.height);
    let dark: Vec<bool> = dark_candidates
        .iter()
        .zip(&dark_black)
        .zip(labs)
        .map(|((&candidate, &black), lab)| black || candidate && lab.a.hypot(lab.b) >= 18.0)
        .collect();
    let lightness: Vec<f32> = labs.iter().map(|lab| lab.l).collect();
    let minima = local_extreme(&lightness, image.width, image.height, true);
    let maxima = local_extreme(&lightness, image.width, image.height, false);
    let mut samples = original.to_vec();
    for index in 0..samples.len() {
        if dark[index] || bright[index] {
            samples[index] = false;
        }
        if (dark[index] && lightness[index] <= minima[index] + 1e-6)
            || (bright[index] && lightness[index] >= maxima[index] - 1e-6)
        {
            samples[index] = true;
        }
    }
    samples
}

pub fn adjust_paint_samples(image: &Raster, original: &[bool]) -> Vec<bool> {
    adjust_paint_samples_from_analysis(image, original, &analyze(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_analysis_matches_independent_ridge_consumers() {
        let mut image = Raster::blank(32, 24, [0.82, 0.82, 0.82]);
        for x in 4..28 {
            image.pixels[7 * image.width + x] = [0.05, 0.08, 0.12];
            image.pixels[16 * image.width + x] = [0.96, 0.75, 0.12];
        }
        let original = vec![true; image.pixels.len()];
        let analysis = analyze(&image);

        assert_eq!(
            adjust_paint_samples_from_analysis(&image, &original, &analysis),
            adjust_paint_samples(&image, &original)
        );
        let shared = strong_branches_from_analysis(&image, &analysis);
        let independent = strong_branches(&image);
        assert_eq!(shared.dark, independent.dark);
        assert_eq!(shared.bright, independent.bright);
    }
}
