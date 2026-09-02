use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;
use serde::Serialize;

use crate::color::{
    delta_e2000, delta_e2000_pairs, delta_e76, rgb_to_lab, skimage_lab_values_to_rgb, Lab,
};
use crate::config::Config;
use crate::edge::{
    dilate_square, lab_pixels, preprocess_lab_pixels, preprocess_lab_values, EdgeRoles,
};
use crate::geometry::Point;
use crate::hierarchy::HierarchicalTopology;
use crate::raster::{percentile, Raster};
use crate::segment::{replace_merged_labels, replace_source_supported_paint_labels, Segmentation};
use crate::union_find::UnionFind;

#[derive(Clone, Debug, PartialEq)]
pub struct ColorStop {
    pub offset: f64,
    pub color: [f64; 3],
}

/// Alpha profile for a Paint overlay.  Keeping opacity separate from
/// `ColorStop` preserves the existing Office-compatible base gradients while
/// allowing a smooth local correction to fade out without creating a new
/// topology boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct OpacityStop {
    pub offset: f64,
    pub opacity: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaintOverlay {
    pub paint: Box<Paint>,
    pub opacity_stops: Vec<OpacityStop>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinearPreset {
    LeftToRight,
    TopToBottom,
    TopLeftToBottomRight,
    TopRightToBottomLeft,
    Fitted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RadialOrigin {
    Center,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Fitted,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    Solid {
        color: [f32; 3],
    },
    Linear {
        preset: LinearPreset,
        start: Point,
        end: Point,
        stops: Vec<ColorStop>,
    },
    Radial {
        origin: RadialOrigin,
        center: Point,
        radius: Point,
        stops: Vec<ColorStop>,
    },
    /// A single topology face with smooth residual Paint corrections.
    ///
    /// The overlay geometry is clipped by emitting the same face path for
    /// every layer.  Unlike adaptive face splitting, no internal SVG edge is
    /// introduced, so a quantized shading contour cannot become a staircase.
    Layered {
        base: Box<Paint>,
        overlays: Vec<PaintOverlay>,
    },
}

type CoupledPaintAssignments = Vec<(usize, Paint, f32)>;

#[derive(Clone, Debug, Default, Serialize)]
pub struct GradientSummary {
    pub solid_regions: usize,
    pub linear_regions: usize,
    pub radial_regions: usize,
    pub layered_regions: usize,
    pub overlay_layers: usize,
    pub fitted_direction_linear_regions: usize,
    pub fitted_focus_radial_regions: usize,
    pub coupled_linear_regions: usize,
    pub source_supported_paint_merges: usize,
    pub source_supported_boundary_edges_removed: usize,
    pub maximum_stops: usize,
    pub primary_gate_active: bool,
    pub primary_solid_regions: usize,
    pub full_fit_regions: usize,
}

pub(crate) fn refresh_summary(summary: &mut GradientSummary, paints: &[Paint]) {
    summary.solid_regions = 0;
    summary.linear_regions = 0;
    summary.radial_regions = 0;
    summary.layered_regions = 0;
    summary.overlay_layers = 0;
    summary.fitted_direction_linear_regions = 0;
    summary.fitted_focus_radial_regions = 0;
    summary.maximum_stops = 0;
    for paint in paints {
        match paint {
            Paint::Solid { .. } => summary.solid_regions += 1,
            Paint::Linear { preset, stops, .. } => {
                summary.linear_regions += 1;
                summary.fitted_direction_linear_regions +=
                    usize::from(*preset == LinearPreset::Fitted);
                summary.maximum_stops = summary.maximum_stops.max(stops.len());
            }
            Paint::Radial { origin, stops, .. } => {
                summary.radial_regions += 1;
                summary.fitted_focus_radial_regions += usize::from(*origin == RadialOrigin::Fitted);
                summary.maximum_stops = summary.maximum_stops.max(stops.len());
            }
            Paint::Layered { base, overlays } => {
                summary.layered_regions += 1;
                summary.overlay_layers += overlays.len();
                for component in std::iter::once(base.as_ref())
                    .chain(overlays.iter().map(|overlay| overlay.paint.as_ref()))
                {
                    match component {
                        Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => {
                            summary.maximum_stops = summary.maximum_stops.max(stops.len());
                        }
                        Paint::Solid { .. } | Paint::Layered { .. } => {}
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

fn bounds(indices: &[usize], width: usize) -> Bounds {
    let mut result = Bounds {
        min_x: f32::INFINITY,
        min_y: f32::INFINITY,
        max_x: f32::NEG_INFINITY,
        max_y: f32::NEG_INFINITY,
    };
    for &index in indices {
        let x = (index % width) as f32;
        let y = (index / width) as f32;
        result.min_x = result.min_x.min(x);
        result.max_x = result.max_x.max(x);
        result.min_y = result.min_y.min(y);
        result.max_y = result.max_y.max(y);
    }
    result.max_x = result.max_x.max(result.min_x + 1.0);
    result.max_y = result.max_y.max(result.min_y + 1.0);
    result
}

fn sampled_indices(indices: &[usize], maximum: usize) -> Vec<usize> {
    if indices.len() <= maximum {
        return indices.to_vec();
    }
    if maximum <= 1 {
        return indices.first().copied().into_iter().collect();
    }
    // numpy.linspace(0, n - 1, maximum, dtype=int64): retain exactly the
    // requested count, including both ends, instead of a ceil-stride subset
    // that can undersample by almost one third.
    (0..maximum)
        .map(|position| {
            let source = position * (indices.len() - 1) / (maximum - 1);
            indices[source]
        })
        .collect()
}

/// Cheap upper-bound estimate of how much RGB variation can be explained by
/// the spatial models available to Paint. An affine x/y model covers every
/// linear direction (and deliberately over-approximates a one-dimensional
/// gradient); the five radial presets cover symmetric shading that has little
/// linear correlation. Only faces below the configured coherence threshold
/// are classified Solid here.
fn primary_gradient_coherence(
    source: &Raster,
    samples: &[usize],
    region_bounds: Bounds,
    directional_only: bool,
    sample_budget: usize,
) -> f32 {
    let samples = sampled_indices(samples, sample_budget.max(8));
    if samples.len() < 4 {
        return 1.0;
    }
    let divisor = samples.len() as f32;
    let mean_rgb = [0, 1, 2].map(|channel| {
        samples
            .iter()
            .map(|&index| source.pixels[index][channel])
            .sum::<f32>()
            / divisor
    });
    let mean_x = samples
        .iter()
        .map(|&index| (index % source.width) as f32)
        .sum::<f32>()
        / divisor;
    let mean_y = samples
        .iter()
        .map(|&index| (index / source.width) as f32)
        .sum::<f32>()
        / divisor;
    let mut total = 0.0_f32;
    let mut xx = 0.0_f32;
    let mut xy = 0.0_f32;
    let mut yy = 0.0_f32;
    let mut x_color = [0.0_f32; 3];
    let mut y_color = [0.0_f32; 3];
    for &index in &samples {
        let x = (index % source.width) as f32 - mean_x;
        let y = (index / source.width) as f32 - mean_y;
        xx += x * x;
        xy += x * y;
        yy += y * y;
        for channel in 0..3 {
            let value = source.pixels[index][channel] - mean_rgb[channel];
            total += value * value;
            x_color[channel] += x * value;
            y_color[channel] += y * value;
        }
    }
    if total <= 1e-10 {
        return 0.0;
    }
    let determinant = xx * yy - xy * xy;
    let mut best_explained = if determinant > 1e-8 {
        (0..3)
            .map(|channel| {
                (yy * x_color[channel] * x_color[channel]
                    - 2.0 * xy * x_color[channel] * y_color[channel]
                    + xx * y_color[channel] * y_color[channel])
                    / determinant
            })
            .sum::<f32>()
    } else {
        0.0
    };
    if !directional_only {
        for origin in [
            RadialOrigin::Center,
            RadialOrigin::TopLeft,
            RadialOrigin::TopRight,
            RadialOrigin::BottomLeft,
            RadialOrigin::BottomRight,
        ] {
            let (center, radius) = radial_geometry(origin, region_bounds);
            let parameters = samples
                .iter()
                .map(|&index| radial_parameter(index, source.width, center, radius))
                .collect::<Vec<_>>();
            let mean_parameter = parameters.iter().sum::<f32>() / divisor;
            let mut variance = 0.0_f32;
            let mut covariance = [0.0_f32; 3];
            for (&index, &parameter) in samples.iter().zip(&parameters) {
                let parameter = parameter - mean_parameter;
                variance += parameter * parameter;
                for channel in 0..3 {
                    covariance[channel] +=
                        parameter * (source.pixels[index][channel] - mean_rgb[channel]);
                }
            }
            if variance > 1e-8 {
                let explained = covariance
                    .iter()
                    .map(|value| value * value / variance)
                    .sum::<f32>();
                best_explained = best_explained.max(explained);
            }
        }
    }
    (best_explained / total).clamp(0.0, 1.0)
}

fn numpy_sum_f32(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return values.iter().fold(-0.0_f32, |sum, &value| sum + value);
    }
    if values.len() <= 128 {
        let mut partial = [0.0_f32; 8];
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
    numpy_sum_f32(&values[..middle]) + numpy_sum_f32(&values[middle..])
}

fn mean_color(source: &Raster, indices: &[usize]) -> [f32; 3] {
    let divisor = indices.len().max(1) as f32;
    // np.mean(values, axis=0) reduces a C-contiguous N x 3 array along its
    // strided axis.  NumPy's strided reduction is sequential, unlike the
    // pairwise reduction used for a contiguous one-dimensional array.
    let mut sums = [-0.0_f32; 3];
    for &index in indices {
        for (channel, sum) in sums.iter_mut().enumerate() {
            *sum += source.pixels[index][channel];
        }
    }
    sums.map(|sum| sum / divisor)
}

fn mean_color_f64(source: &Raster, indices: &[usize]) -> [f32; 3] {
    let mut result = [0.0_f64; 3];
    for &index in indices {
        for (channel, sum) in result.iter_mut().enumerate() {
            *sum += source.pixels[index][channel] as f64;
        }
    }
    let divisor = indices.len().max(1) as f64;
    [
        (result[0] / divisor) as f32,
        (result[1] / divisor) as f32,
        (result[2] / divisor) as f32,
    ]
}

fn interpolate(stops: &[ColorStop], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    if stops.len() == 2 && stops[0].offset == 0.0 && stops[1].offset == 1.0 {
        let start = stops[0].color.map(|value| value as f32);
        let end = stops[1].color.map(|value| value as f32);
        return [
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
            start[2] + t * (end[2] - start[2]),
        ];
    }
    let position = t as f64;
    if let Some(stop) = stops.iter().find(|stop| stop.offset == position) {
        return stop.color.map(|value| value as f32);
    }
    let upper = stops
        .iter()
        .position(|stop| stop.offset >= position)
        .unwrap_or(stops.len() - 1);
    if upper == 0 {
        return stops[0].color.map(|value| value as f32);
    }
    let first = &stops[upper - 1];
    let second = &stops[upper];
    let amount =
        ((position - first.offset) / (second.offset - first.offset).max(1e-6)).clamp(0.0, 1.0);
    [
        (first.color[0] + amount * (second.color[0] - first.color[0])) as f32,
        (first.color[1] + amount * (second.color[1] - first.color[1])) as f32,
        (first.color[2] + amount * (second.color[2] - first.color[2])) as f32,
    ]
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

#[allow(clippy::needless_range_loop)]
fn solve_system(mut matrix: Vec<Vec<f64>>, mut target: Vec<f64>) -> Vec<f64> {
    let count = target.len();
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|&left, &right| {
                matrix[left][column]
                    .abs()
                    .total_cmp(&matrix[right][column].abs())
            })
            .unwrap_or(column);
        matrix.swap(column, pivot);
        target.swap(column, pivot);
        let divisor = matrix[column][column];
        if divisor.abs() <= 1e-12 {
            continue;
        }
        for entry in &mut matrix[column][column..] {
            *entry /= divisor;
        }
        target[column] /= divisor;
        for row in 0..count {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.abs() <= 1e-15 {
                continue;
            }
            for entry in column..count {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            target[row] -= factor * target[column];
        }
    }
    target
}

fn interpolation_weights(parameter: f32, offsets: &[f64]) -> (usize, usize, f32) {
    let parameter = parameter.clamp(0.0, 1.0);
    let offset_values: Vec<f32> = offsets.iter().map(|&offset| offset as f32).collect();
    let right = offset_values
        .iter()
        .position(|&offset| offset >= parameter)
        .unwrap_or(offsets.len() - 1)
        .max(1);
    let left = right - 1;
    let alpha = ((parameter - offset_values[left])
        / (offset_values[right] - offset_values[left]).max(1e-6))
    .clamp(0.0, 1.0);
    (left, right, alpha)
}

#[allow(clippy::needless_range_loop)]
fn fitted_stops(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    offsets: &[f64],
) -> Vec<ColorStop> {
    let bin_count = 64_usize.min(samples.len().max(8)).max(8);
    let mut bins = vec![Vec::<[f32; 3]>::new(); bin_count];
    for (&index, &parameter) in samples.iter().zip(parameters) {
        let bin = ((parameter.clamp(0.0, 1.0) * bin_count as f32) as usize).min(bin_count - 1);
        bins[bin].push(source.pixels[index]);
    }
    let positions: Vec<f32> = (0..bin_count)
        .map(|index| (index as f32 + 0.5) / bin_count as f32)
        .collect();
    let mut profile = vec![[f32::NAN; 3]; bin_count];
    let mut weights: Vec<f32> = bins.iter().map(|values| values.len() as f32).collect();
    for (index, values) in bins.iter().enumerate() {
        if values.is_empty() {
            continue;
        }
        for channel in 0..3 {
            let mut channel_values: Vec<f32> = values.iter().map(|value| value[channel]).collect();
            profile[index][channel] = median(&mut channel_values);
        }
    }
    let valid: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| (weight > 0.0).then_some(index))
        .collect();
    if valid.len() == 1 {
        let colour = profile[valid[0]];
        profile.fill(colour);
    } else if !valid.is_empty() {
        for index in 0..bin_count {
            if profile[index][0].is_finite() {
                continue;
            }
            let right_position = valid.partition_point(|&value| value < index);
            let left = valid[right_position.saturating_sub(1)];
            let right = valid[right_position.min(valid.len() - 1)];
            let amount = if left == right {
                0.0
            } else {
                (positions[index] - positions[left]) / (positions[right] - positions[left])
            };
            for channel in 0..3 {
                profile[index][channel] =
                    profile[left][channel] * (1.0 - amount) + profile[right][channel] * amount;
            }
        }
    }
    // scipy gaussian_filter1d(sigma=1, mode=nearest), truncated to four
    // sigma, expressed directly because the profile contains only 64 bins.
    let kernel: Vec<f32> = (-4_i32..=4)
        .map(|offset| (-0.5 * (offset * offset) as f32).exp())
        .collect();
    let kernel_sum = kernel.iter().sum::<f32>();
    let original = profile.clone();
    for index in 0..bin_count {
        for channel in 0..3 {
            profile[index][channel] = kernel
                .iter()
                .enumerate()
                .map(|(kernel_index, &weight)| {
                    let offset = kernel_index as isize - 4;
                    let sample =
                        (index as isize + offset).clamp(0, bin_count as isize - 1) as usize;
                    original[sample][channel] * weight
                })
                .sum::<f32>()
                / kernel_sum;
        }
    }
    let positive: Vec<f32> = weights
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    let cap = percentile(positive, 0.90).max(1.0);
    for weight in &mut weights {
        *weight = weight.clamp(1.0, cap);
    }
    let mean_weight = weights.iter().sum::<f32>() / weights.len().max(1) as f32;
    for weight in &mut weights {
        *weight /= mean_weight.max(1e-6);
    }

    let count = offsets.len();
    if count == 2 && offsets[0] == 0.0 && offsets[1] == 1.0 {
        let right = positions.clone();
        let left: Vec<f32> = positions.iter().map(|&value| 1.0 - value).collect();
        let aa = numpy_sum_f32(
            &weights
                .iter()
                .zip(&left)
                .map(|(&weight, &value)| weight * value * value)
                .collect::<Vec<_>>(),
        );
        let ab = numpy_sum_f32(
            &weights
                .iter()
                .zip(&left)
                .zip(&right)
                .map(|((&weight, &first), &second)| weight * first * second)
                .collect::<Vec<_>>(),
        );
        let bb = numpy_sum_f32(
            &weights
                .iter()
                .zip(&right)
                .map(|(&weight, &value)| weight * value * value)
                .collect::<Vec<_>>(),
        );
        let mut left_values = [-0.0_f32; 3];
        let mut right_values = [-0.0_f32; 3];
        for index in 0..bin_count {
            let left_weight = weights[index] * left[index];
            let right_weight = weights[index] * right[index];
            for channel in 0..3 {
                left_values[channel] += left_weight * profile[index][channel];
                right_values[channel] += right_weight * profile[index][channel];
            }
        }
        let determinant_f64 = aa as f64 * bb as f64 - ab as f64 * ab as f64;
        let determinant = determinant_f64 as f32;
        let colors = if determinant_f64 > 1e-8 {
            [
                [0, 1, 2].map(|channel| {
                    ((left_values[channel] * bb - right_values[channel] * ab) / determinant)
                        .clamp(0.0, 1.0)
                }),
                [0, 1, 2].map(|channel| {
                    ((right_values[channel] * aa - left_values[channel] * ab) / determinant)
                        .clamp(0.0, 1.0)
                }),
            ]
        } else {
            let total = weights.iter().sum::<f32>().max(1e-6);
            let average = [0, 1, 2].map(|channel| {
                weights
                    .iter()
                    .zip(&profile)
                    .map(|(&weight, value)| weight * value[channel])
                    .sum::<f32>()
                    / total
            });
            [average, average]
        };
        return [0.0_f64, 1.0]
            .into_iter()
            .enumerate()
            .map(|(index, offset)| ColorStop {
                offset,
                color: colors[index].map(f64::from),
            })
            .collect();
    }
    let mut normal = vec![vec![0.0_f64; count]; count];
    let mut targets = vec![vec![0.0_f64; count]; 3];
    for index in 0..bin_count {
        let (left, right, alpha) = interpolation_weights(positions[index], offsets);
        let basis = [(left, 1.0 - alpha), (right, alpha)];
        let weight = weights[index] as f64;
        for &(first, first_value) in &basis {
            for &(second, second_value) in &basis {
                normal[first][second] += weight * first_value as f64 * second_value as f64;
            }
            for channel in 0..3 {
                targets[channel][first] +=
                    weight * first_value as f64 * profile[index][channel] as f64;
            }
        }
    }
    if count >= 3 {
        let lambda = 0.35_f64;
        for row in 0..count - 2 {
            let difference = [(row, 1.0_f64), (row + 1, -2.0), (row + 2, 1.0)];
            for &(first, first_value) in &difference {
                for &(second, second_value) in &difference {
                    normal[first][second] += lambda * first_value * second_value;
                }
            }
        }
    }
    let colors: Vec<Vec<f64>> = targets
        .into_iter()
        .map(|target| solve_system(normal.clone(), target))
        .collect();
    offsets
        .iter()
        .enumerate()
        .map(|(index, &offset)| ColorStop {
            offset,
            color: [
                colors[0][index].clamp(0.0, 1.0),
                colors[1][index].clamp(0.0, 1.0),
                colors[2][index].clamp(0.0, 1.0),
            ],
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct ErrorStats {
    mean: f32,
    percentile: f32,
}

fn paint_error(
    source: &Raster,
    samples: &[usize],
    predicted: impl Fn(usize) -> [f32; 3],
) -> ErrorStats {
    if samples.is_empty() {
        return ErrorStats {
            mean: 0.0,
            percentile: 0.0,
        };
    }
    let references: Vec<[f32; 3]> = samples.iter().map(|&index| source.pixels[index]).collect();
    let rendered: Vec<[f32; 3]> = samples.iter().map(|&index| predicted(index)).collect();
    let reference_lab = preprocess_color_values(references);
    let rendered_lab = preprocess_color_values(rendered);
    let errors = delta_e2000_pairs(&reference_lab, &rendered_lab);
    ErrorStats {
        mean: numpy_sum_f32(&errors) / errors.len() as f32,
        percentile: percentile(errors, 0.90),
    }
}

/// Measure an emitted sRGB Paint against the skimage-compatible Lab values
/// already computed for the native source.  Paint selection in the Python
/// implementation deliberately uses `skimage.color.rgb2lab`; the
/// preprocess-Lab transform above belongs to boundary evidence and is not an
/// interchangeable approximation at the acceptance thresholds.
fn paint_error_against_labs(
    source_labs: &[Lab],
    samples: &[usize],
    predicted: impl Fn(usize) -> [f32; 3],
) -> ErrorStats {
    if samples.is_empty() {
        return ErrorStats {
            mean: 0.0,
            percentile: 0.0,
        };
    }
    let references: Vec<Lab> = samples.iter().map(|&index| source_labs[index]).collect();
    let rendered = Raster::new(
        samples.len(),
        1,
        samples.iter().map(|&index| predicted(index)).collect(),
    );
    let rendered_labs = lab_pixels(&rendered);
    let errors = delta_e2000_pairs(&references, &rendered_labs);
    ErrorStats {
        mean: numpy_sum_f32(&errors) / errors.len() as f32,
        percentile: percentile(errors, 0.90),
    }
}

fn gradient_error_against_labs(
    source_labs: &[Lab],
    samples: &[usize],
    parameters: &[f32],
    stops: &[ColorStop],
) -> ErrorStats {
    let lookup: HashMap<usize, f32> = samples
        .iter()
        .copied()
        .zip(parameters.iter().copied())
        .collect();
    paint_error_against_labs(source_labs, samples, |index| {
        interpolate(stops, lookup[&index])
    })
}

fn paint_stats_against_labs(
    source_labs: &[Lab],
    samples: &[usize],
    width: usize,
    paint: &Paint,
) -> ErrorStats {
    paint_error_against_labs(source_labs, samples, |index| paint_at(paint, index, width))
}

fn preprocess_color_values(colors: Vec<[f32; 3]>) -> Vec<Lab> {
    if colors.is_empty() {
        return Vec::new();
    }
    preprocess_lab_values(&colors)
}

fn objective(stats: ErrorStats) -> f32 {
    stats.mean.max(0.60 * stats.percentile)
}

fn linear_geometry(preset: LinearPreset, bounds: Bounds) -> (Point, Point) {
    let centre_x = (bounds.min_x + bounds.max_x) * 0.5;
    let centre_y = (bounds.min_y + bounds.max_y) * 0.5;
    match preset {
        LinearPreset::LeftToRight | LinearPreset::Fitted => (
            Point {
                x: bounds.min_x,
                y: centre_y,
            },
            Point {
                x: bounds.max_x,
                y: centre_y,
            },
        ),
        LinearPreset::TopToBottom => (
            Point {
                x: centre_x,
                y: bounds.min_y,
            },
            Point {
                x: centre_x,
                y: bounds.max_y,
            },
        ),
        LinearPreset::TopLeftToBottomRight => (
            Point {
                x: bounds.min_x,
                y: bounds.min_y,
            },
            Point {
                x: bounds.max_x,
                y: bounds.max_y,
            },
        ),
        LinearPreset::TopRightToBottomLeft => (
            Point {
                x: bounds.max_x,
                y: bounds.min_y,
            },
            Point {
                x: bounds.min_x,
                y: bounds.max_y,
            },
        ),
    }
}

fn canonical_direction(mut direction: (f32, f32)) -> Option<(f32, f32)> {
    let length = (direction.0 * direction.0 + direction.1 * direction.1).sqrt();
    if length <= 1e-6 {
        return None;
    }
    direction.0 /= length;
    direction.1 /= length;
    if direction.0 < -1e-6 || (direction.0.abs() <= 1e-6 && direction.1 < 0.0) {
        direction.0 = -direction.0;
        direction.1 = -direction.1;
    }
    Some(direction)
}

fn fitted_linear_directions_from_lightness(
    source: &Raster,
    samples: &[usize],
    lightness: &[f32],
) -> Vec<(f32, f32)> {
    if samples.len() < 3 {
        return Vec::new();
    }
    let divisor = samples.len() as f32;
    let centre_x = samples
        .iter()
        .map(|&index| (index % source.width) as f32)
        .sum::<f32>()
        / divisor;
    let centre_y = samples
        .iter()
        .map(|&index| (index / source.width) as f32)
        .sum::<f32>()
        / divisor;
    let mean_lightness = lightness.iter().sum::<f32>() / divisor;
    let (mut xx, mut xy, mut yy) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut xl, mut yl) = (0.0_f64, 0.0_f64);
    for (&index, &l) in samples.iter().zip(lightness) {
        let dx = (index % source.width) as f32 - centre_x;
        let dy = (index / source.width) as f32 - centre_y;
        let dl = l - mean_lightness;
        xx += (dx * dx) as f64;
        xy += (dx * dy) as f64;
        yy += (dy * dy) as f64;
        xl += (dx * dl) as f64;
        yl += (dy * dl) as f64;
    }
    let spatial_angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut result = Vec::<(f32, f32)>::new();
    let mut principal = (spatial_angle.cos() as f32, spatial_angle.sin() as f32);
    // LAPACK's symmetric 2x2 eigenvector orientation, as exposed by
    // `numpy.linalg.eigh`: the dominant x component is negative, while a
    // dominant y component is positive.  Reversing the vector is visually
    // equivalent only if every stop is also reversed, but it is not textual
    // SVG parity and changes the robust stop fitting order.
    let reverse = if xx >= yy {
        principal.0 > 0.0
    } else {
        principal.1 < 0.0
    };
    if reverse {
        principal.0 = -principal.0;
        principal.1 = -principal.1;
    }
    result.push(principal);
    let determinant = xx * yy - xy * xy;
    if determinant.abs() > 1e-6 {
        let plane = (
            (yy * xl - xy * yl) / determinant,
            (xx * yl - xy * xl) / determinant,
        );
        let length = (plane.0 * plane.0 + plane.1 * plane.1).sqrt();
        if length > 1e-6 {
            let direction = ((plane.0 / length) as f32, (plane.1 / length) as f32);
            if result
                .iter()
                .all(|current| (current.0 * direction.0 + current.1 * direction.1).abs() < 0.999)
            {
                result.push(direction);
            }
        }
    }
    result
}

fn fitted_linear_directions(source: &Raster, samples: &[usize]) -> Vec<(f32, f32)> {
    let lightness: Vec<f32> =
        preprocess_color_values(samples.iter().map(|&index| source.pixels[index]).collect())
            .into_iter()
            .map(|value| value.l)
            .collect();
    fitted_linear_directions_from_lightness(source, samples, &lightness)
}

fn fitted_linear_geometry(
    samples: &[usize],
    width: usize,
    direction: (f32, f32),
) -> (Point, Point) {
    let divisor = samples.len().max(1) as f32;
    let centre = Point {
        x: samples
            .iter()
            .map(|&index| (index % width) as f32)
            .sum::<f32>()
            / divisor,
        y: samples
            .iter()
            .map(|&index| (index / width) as f32)
            .sum::<f32>()
            / divisor,
    };
    let (low, high) =
        samples
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &index| {
                let x = (index % width) as f32 - centre.x;
                let y = (index / width) as f32 - centre.y;
                let projection = x * direction.0 + y * direction.1;
                (low.min(projection), high.max(projection))
            });
    let span = (high - low).max(1.0);
    let low = if low.is_finite() { low } else { -0.5 * span };
    let high = if high.is_finite() { high } else { 0.5 * span };
    (
        Point {
            x: centre.x + direction.0 * low,
            y: centre.y + direction.1 * low,
        },
        Point {
            x: centre.x + direction.0 * high,
            y: centre.y + direction.1 * high,
        },
    )
}

fn legacy_linear_geometry_parameters(
    samples: &[usize],
    width: usize,
    direction: (f32, f32),
) -> (Point, Point, Vec<f32>) {
    let divisor = samples.len().max(1) as f32;
    let center = Point {
        x: samples
            .iter()
            .map(|&index| (index % width) as f32)
            .sum::<f32>()
            / divisor,
        y: samples
            .iter()
            .map(|&index| (index / width) as f32)
            .sum::<f32>()
            / divisor,
    };
    let projections: Vec<f32> = samples
        .iter()
        .map(|&index| {
            let dx = (index % width) as f32 - center.x;
            let dy = (index / width) as f32 - center.y;
            dx * direction.0 + dy * direction.1
        })
        .collect();
    let low = projections.iter().copied().fold(f32::INFINITY, f32::min);
    let high = projections
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let span = (high - low).max(1e-5);
    let start = Point {
        x: center.x + direction.0 * low,
        y: center.y + direction.1 * low,
    };
    let end = Point {
        x: center.x + direction.0 * high,
        y: center.y + direction.1 * high,
    };
    let parameters = projections
        .into_iter()
        .map(|projection| (projection - low) / span)
        .collect();
    (start, end, parameters)
}

fn linear_parameter(index: usize, width: usize, start: Point, end: Point) -> f32 {
    let point = Point {
        x: (index % width) as f32,
        y: (index / width) as f32,
    };
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy).max(1e-6)
}

fn radial_geometry(origin: RadialOrigin, bounds: Bounds) -> (Point, Point) {
    let width = (bounds.max_x - bounds.min_x).max(1.0);
    let height = (bounds.max_y - bounds.min_y).max(1.0);
    let center = match origin {
        RadialOrigin::Center => Point {
            x: (bounds.min_x + bounds.max_x) * 0.5,
            y: (bounds.min_y + bounds.max_y) * 0.5,
        },
        RadialOrigin::TopLeft => Point {
            x: bounds.min_x,
            y: bounds.min_y,
        },
        RadialOrigin::TopRight => Point {
            x: bounds.max_x,
            y: bounds.min_y,
        },
        RadialOrigin::BottomLeft => Point {
            x: bounds.min_x,
            y: bounds.max_y,
        },
        RadialOrigin::BottomRight => Point {
            x: bounds.max_x,
            y: bounds.max_y,
        },
        RadialOrigin::Fitted => Point {
            x: (bounds.min_x + bounds.max_x) * 0.5,
            y: (bounds.min_y + bounds.max_y) * 0.5,
        },
    };
    let radius = if matches!(origin, RadialOrigin::Center | RadialOrigin::Fitted) {
        Point {
            x: width * 0.5,
            y: height * 0.5,
        }
    } else {
        Point {
            x: width,
            y: height,
        }
    };
    (center, radius)
}

fn radial_parameter(index: usize, width: usize, center: Point, radius: Point) -> f32 {
    let x = index % width;
    let y = index / width;
    ((((x as f32 - center.x) / radius.x.max(1e-3)).powi(2)
        + ((y as f32 - center.y) / radius.y.max(1e-3)).powi(2))
    .sqrt())
    .clamp(0.0, 1.0)
}

fn paint_with_stops(template: &Paint, stops: Vec<ColorStop>) -> Paint {
    match template {
        Paint::Linear {
            preset, start, end, ..
        } => Paint::Linear {
            preset: *preset,
            start: *start,
            end: *end,
            stops,
        },
        Paint::Radial {
            origin,
            center,
            radius,
            ..
        } => Paint::Radial {
            origin: *origin,
            center: *center,
            radius: *radius,
            stops,
        },
        Paint::Solid { color } => Paint::Solid { color: *color },
        Paint::Layered { .. } => template.clone(),
    }
}

fn gradient_error(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    stops: &[ColorStop],
) -> ErrorStats {
    let lookup: HashMap<usize, f32> = samples
        .iter()
        .copied()
        .zip(parameters.iter().copied())
        .collect();
    paint_error(source, samples, |index| interpolate(stops, lookup[&index]))
}

fn weighted_median_lab(values: &[(f32, f32)]) -> f32 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total = ordered.iter().map(|value| value.1).sum::<f32>();
    let target = 0.5 * total;
    let mut cumulative = 0.0_f32;
    for (value, weight) in ordered {
        cumulative += weight;
        if cumulative >= target {
            return value;
        }
    }
    0.0
}

/// Python `_fit_stops`: three robust stops in Lab, converted back to the
/// exact sRGB field that is serialized to SVG.
fn legacy_stops(source_labs: &[Lab], samples: &[usize], parameters: &[f32]) -> Vec<ColorStop> {
    let mut stop_labs = Vec::<Lab>::with_capacity(3);
    for stop in 0..3 {
        let mut selected = Vec::<(usize, f32)>::new();
        for (position, &parameter) in parameters.iter().enumerate() {
            let parameter = parameter.clamp(0.0, 1.0);
            let weight = match stop {
                0 if parameter <= 0.5 => 1.0 - 2.0 * parameter,
                1 if parameter <= 0.5 => 2.0 * parameter,
                1 => 2.0 - 2.0 * parameter,
                2 => 2.0 * parameter - 1.0,
                _ => 0.0,
            };
            if weight > 0.20 {
                selected.push((position, weight.max(1e-6)));
            }
        }
        if selected.is_empty() {
            for (position, &parameter) in parameters.iter().enumerate() {
                let parameter = parameter.clamp(0.0, 1.0);
                let weight = match stop {
                    0 if parameter <= 0.5 => 1.0 - 2.0 * parameter,
                    1 if parameter <= 0.5 => 2.0 * parameter,
                    1 => 2.0 - 2.0 * parameter,
                    2 => 2.0 * parameter - 1.0,
                    _ => 0.0,
                };
                if weight > 0.0 {
                    selected.push((position, weight.max(1e-6)));
                }
            }
        }
        if selected.is_empty() {
            selected.extend((0..samples.len()).map(|position| (position, 1.0)));
        }
        let channel = |value: Lab, channel: usize| match channel {
            0 => value.l,
            1 => value.a,
            _ => value.b,
        };
        let fitted = [0, 1, 2].map(|channel_index| {
            weighted_median_lab(
                &selected
                    .iter()
                    .map(|&(position, weight)| {
                        (
                            channel(source_labs[samples[position]], channel_index),
                            weight,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        });
        stop_labs.push(Lab {
            l: fitted[0],
            a: fitted[1],
            b: fitted[2],
        });
    }
    let lows = [0, 1, 2].map(|channel| {
        percentile(
            samples
                .iter()
                .map(|&index| match channel {
                    0 => source_labs[index].l,
                    1 => source_labs[index].a,
                    _ => source_labs[index].b,
                })
                .collect(),
            0.02,
        )
    });
    let highs = [0, 1, 2].map(|channel| {
        percentile(
            samples
                .iter()
                .map(|&index| match channel {
                    0 => source_labs[index].l,
                    1 => source_labs[index].a,
                    _ => source_labs[index].b,
                })
                .collect(),
            0.98,
        )
    });
    for value in &mut stop_labs {
        value.l = value.l.clamp(lows[0], highs[0]);
        value.a = value.a.clamp(lows[1], highs[1]);
        value.b = value.b.clamp(lows[2], highs[2]);
    }
    skimage_lab_values_to_rgb(&stop_labs)
        .into_iter()
        .zip([0.0_f64, 0.5, 1.0])
        .map(|(color, offset)| ColorStop {
            offset,
            color: color.map(f64::from),
        })
        .collect()
}

fn legacy_gradient_error_against_labs(
    source_labs: &[Lab],
    samples: &[usize],
    parameters: &[f32],
    stops: &[ColorStop],
) -> ErrorStats {
    let lookup: HashMap<usize, f32> = samples
        .iter()
        .copied()
        .zip(parameters.iter().copied())
        .collect();
    let colors: Vec<[f32; 3]> = stops
        .iter()
        .map(|stop| stop.color.map(|value| value as f32))
        .collect();
    paint_error_against_labs(source_labs, samples, |index| {
        let parameter = lookup[&index].clamp(0.0, 1.0);
        let (first, second, first_weight, second_weight) = if parameter <= 0.5 {
            (0, 1, 1.0 - 2.0 * parameter, 2.0 * parameter)
        } else {
            (1, 2, 2.0 - 2.0 * parameter, 2.0 * parameter - 1.0)
        };
        [0, 1, 2].map(|channel| {
            first_weight * colors[first][channel] + second_weight * colors[second][channel]
        })
    })
}

fn add_office_stops(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    parameters: &[f32],
    template: &Paint,
    initial_stats: ErrorStats,
    maximum: usize,
) -> (Paint, ErrorStats) {
    let initial_stops = match template {
        Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => stops.clone(),
        Paint::Solid { .. } | Paint::Layered { .. } => unreachable!(),
    };
    let mut offsets = vec![0.0_f64, 1.0];
    let mut current_stops = initial_stops;
    let mut current_stats = initial_stats;
    let mut accepted_stops = current_stops.clone();
    let mut accepted_stats = current_stats;
    let initial_objective = objective(initial_stats);
    let mut accepted_objective = initial_objective;
    let mut smooth_profile = None::<bool>;
    while offsets.len() < maximum.clamp(2, 5) {
        let mut best: Option<(f32, f64, Vec<ColorStop>, ErrorStats)> = None;
        for step in 1..=9 {
            let offset = 0.1_f64 + (step - 1) as f64 * 0.1_f64;
            if offsets.iter().any(|&value| (value - offset).abs() < 0.075) {
                continue;
            }
            let mut proposed = offsets.clone();
            proposed.push(offset);
            proposed.sort_by(f64::total_cmp);
            let stops = fitted_stops(source, samples, parameters, &proposed);
            let stats = gradient_error_against_labs(source_labs, samples, parameters, &stops);
            let candidate = objective(stats);
            if best
                .as_ref()
                .map(|value| candidate < value.0)
                .unwrap_or(true)
            {
                best = Some((candidate, offset, stops, stats));
            }
        }
        let Some((candidate, chosen, stops, stats)) = best else {
            break;
        };
        if candidate >= objective(current_stats) {
            break;
        }
        let immediate_gain = objective(current_stats) - candidate;
        let cumulative_gain = accepted_objective - candidate;
        if immediate_gain < 0.15 && cumulative_gain < 0.15 {
            let smooth = *smooth_profile
                .get_or_insert_with(|| merge_profile_is_smooth(source, samples, parameters));
            if !smooth {
                break;
            }
        }
        offsets.push(chosen);
        offsets.sort_by(f64::total_cmp);
        current_stops = stops;
        current_stats = stats;
        let accepted_gain = if smooth_profile == Some(true) {
            initial_objective - candidate
        } else {
            cumulative_gain
        };
        if accepted_gain >= 0.15 {
            accepted_stops = current_stops.clone();
            accepted_stats = current_stats;
            accepted_objective = candidate;
        }
    }
    (paint_with_stops(template, accepted_stops), accepted_stats)
}

fn legacy_gradient_candidate(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    region_bounds: Bounds,
    directional_only: bool,
) -> (Paint, ErrorStats) {
    let mut candidates = Vec::<(Paint, ErrorStats)>::new();
    let lightness: Vec<f32> = samples.iter().map(|&index| source_labs[index].l).collect();
    let fitted_directions = fitted_linear_directions_from_lightness(source, samples, &lightness);
    let directions = if directional_only && !fitted_directions.is_empty() {
        vec![fitted_directions[0]]
    } else {
        let mut defaults = vec![
            (1.0_f32, 0.0_f32),
            (0.0, 1.0),
            (2.0_f32.powf(-0.5), 2.0_f32.powf(-0.5)),
            (2.0_f32.powf(-0.5), -2.0_f32.powf(-0.5)),
        ];
        defaults.extend(fitted_directions);
        defaults
    };
    for direction in directions {
        let (start, end, parameters) =
            legacy_linear_geometry_parameters(samples, source.width, direction);
        let stops = legacy_stops(source_labs, samples, &parameters);
        let stats = legacy_gradient_error_against_labs(source_labs, samples, &parameters, &stops);
        candidates.push((
            Paint::Linear {
                preset: LinearPreset::Fitted,
                start,
                end,
                stops,
            },
            stats,
        ));
    }

    let divisor = samples.len().max(1) as f32;
    let center = Point {
        x: samples
            .iter()
            .map(|&index| (index % source.width) as f32)
            .sum::<f32>()
            / divisor,
        y: samples
            .iter()
            .map(|&index| (index / source.width) as f32)
            .sum::<f32>()
            / divisor,
    };
    let lower = percentile(lightness.clone(), 0.20);
    let mut weighted = Point::default();
    let mut total = 0.0_f32;
    for (&index, &value) in samples.iter().zip(&lightness) {
        let weight = (value - lower).max(0.0) + 1e-3;
        weighted.x += (index % source.width) as f32 * weight;
        weighted.y += (index / source.width) as f32 * weight;
        total += weight;
    }
    let focus = if total > 1e-6 {
        Point {
            x: weighted.x / total,
            y: weighted.y / total,
        }
    } else {
        center
    };
    let radius = Point {
        x: ((region_bounds.max_x - region_bounds.min_x) * 0.5).max(0.5),
        y: ((region_bounds.max_y - region_bounds.min_y) * 0.5).max(0.5),
    };
    for radial_center in if directional_only {
        Vec::new()
    } else {
        vec![center, focus]
    } {
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| radial_parameter(index, source.width, radial_center, radius))
            .collect();
        let stops = legacy_stops(source_labs, samples, &parameters);
        let stats = legacy_gradient_error_against_labs(source_labs, samples, &parameters, &stops);
        candidates.push((
            Paint::Radial {
                origin: RadialOrigin::Fitted,
                center: radial_center,
                radius,
                stops,
            },
            stats,
        ));
    }
    candidates
        .into_iter()
        .min_by(|left, right| left.1.mean.total_cmp(&right.1.mean))
        .expect("at least the four default linear gradients")
}

fn office_gradient_candidate(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    region_bounds: Bounds,
    maximum_stops: usize,
) -> Option<(Paint, ErrorStats)> {
    let mut candidates = Vec::<(f32, Paint, Vec<f32>, Option<ErrorStats>)>::new();
    {
        let mut push_linear = |preset: LinearPreset, start: Point, end: Point| {
            let parameters: Vec<f32> = samples
                .iter()
                .map(|&index| linear_parameter(index, source.width, start, end).clamp(0.0, 1.0))
                .collect();
            let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
            let paint = Paint::Linear {
                preset,
                start,
                end,
                stops,
            };
            candidates.push((
                paint_rgb_mse(source, samples, &paint),
                paint,
                parameters,
                None,
            ));
        };
        for preset in [
            LinearPreset::LeftToRight,
            LinearPreset::TopToBottom,
            LinearPreset::TopLeftToBottomRight,
            LinearPreset::TopRightToBottomLeft,
        ] {
            let (start, end) = linear_geometry(preset, region_bounds);
            push_linear(preset, start, end);
        }
        // Estimate continuous angles from the source lightness plane instead
        // of constraining rotation to horizontal, vertical, or 45-degree
        // presets.
        for direction in fitted_linear_directions(source, samples) {
            let (start, end) = fitted_linear_geometry(samples, source.width, direction);
            push_linear(LinearPreset::Fitted, start, end);
        }
    }
    for origin in [
        RadialOrigin::Center,
        RadialOrigin::TopLeft,
        RadialOrigin::TopRight,
        RadialOrigin::BottomLeft,
        RadialOrigin::BottomRight,
    ] {
        let (center, radius) = radial_geometry(origin, region_bounds);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| radial_parameter(index, source.width, center, radius))
            .collect();
        let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
        let paint = Paint::Radial {
            origin,
            center,
            radius,
            stops,
        };
        candidates.push((
            paint_rgb_mse(source, samples, &paint),
            paint,
            parameters,
            None,
        ));
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    for candidate in candidates.iter_mut().take(3) {
        candidate.3 = Some(paint_stats_against_labs(
            source_labs,
            samples,
            source.width,
            &candidate.1,
        ));
    }
    let selected = (0..3)
        .min_by(|&left, &right| {
            objective(candidates[left].3.expect("finalist stats"))
                .total_cmp(&objective(candidates[right].3.expect("finalist stats")))
        })
        .expect("three Office finalists");
    let selected_two_stop = candidates[selected].1.clone();
    let (mut gradient, mut gradient_stats) = add_office_stops(
        source,
        source_labs,
        samples,
        &candidates[selected].2,
        &candidates[selected].1,
        candidates[selected].3.expect("selected stats"),
        maximum_stops,
    );

    let mean = mean_color(source, samples);
    let solid = Paint::Solid { color: mean };
    let solid_stats = paint_stats_against_labs(source_labs, samples, source.width, &solid);
    let percentile_guard = solid_stats.percentile + (0.10 * solid_stats.percentile).max(1.0);
    let mut accepted =
        solid_stats.mean >= gradient_stats.mean && gradient_stats.percentile <= percentile_guard;
    if !accepted && maximum_stops > 2 {
        for candidate in &mut candidates {
            if candidate.3.is_none() {
                candidate.3 = Some(paint_stats_against_labs(
                    source_labs,
                    samples,
                    source.width,
                    &candidate.1,
                ));
            }
        }
        let mut rescue_indices = vec![0_usize, 1, 2];
        let mut by_mean: Vec<usize> = (0..candidates.len()).collect();
        by_mean.sort_by(|&left, &right| {
            candidates[left]
                .3
                .expect("candidate stats")
                .mean
                .total_cmp(&candidates[right].3.expect("candidate stats").mean)
        });
        for index in by_mean.into_iter().take(3) {
            if !rescue_indices.contains(&index) {
                rescue_indices.push(index);
            }
        }
        let mut best = (gradient.clone(), gradient_stats);
        for index in rescue_indices {
            if candidates[index].1 == selected_two_stop {
                continue;
            }
            let expanded = add_office_stops(
                source,
                source_labs,
                samples,
                &candidates[index].2,
                &candidates[index].1,
                candidates[index].3.expect("candidate stats"),
                maximum_stops,
            );
            if objective(expanded.1) < objective(best.1) {
                best = expanded;
            }
        }
        gradient = best.0;
        gradient_stats = best.1;
        accepted = solid_stats.mean >= gradient_stats.mean
            && gradient_stats.percentile <= percentile_guard;
    }
    accepted.then_some((gradient, gradient_stats))
}

#[allow(clippy::too_many_arguments)]
fn fit_region(
    label: usize,
    source: &Raster,
    source_labs: &[Lab],
    indices: &[usize],
    paint_indices: &[usize],
    canonical_solid: [f32; 3],
    directional_only: bool,
    sample_budget: usize,
    use_primary_gate: bool,
    config: &Config,
) -> (Paint, f32, bool) {
    let sample_source = if paint_indices.is_empty() {
        indices
    } else {
        paint_indices
    };
    // A bounded, row-major stratified sample acts as the finest Paint leaf
    // budget. Larger flat interiors therefore do not dominate the fit, while
    // edge-rich and small regions still retain every available observation.
    let samples = sampled_indices(sample_source, sample_budget);
    let region_bounds = bounds(&samples, source.width);
    let small_region = indices.len() < config.minimum_gradient_area as usize;
    let threshold = if small_region {
        config.paint_primary_small_min_explained_variance
    } else {
        config.paint_primary_min_explained_variance
    }
    .clamp(0.0, 1.0);
    let traced = cfg!(feature = "diagnostics")
        && std::env::var("PICVEC_TRACE_PAINT_LABEL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            == Some(label);
    let run_full_fit = !use_primary_gate
        || traced
        || primary_gradient_coherence(
            source,
            &samples,
            region_bounds,
            directional_only,
            config.paint_primary_sample_budget,
        ) >= threshold;
    let (paint, error) = fit_region_samples(
        label,
        source,
        source_labs,
        &samples,
        indices.len(),
        region_bounds,
        canonical_solid,
        directional_only,
        run_full_fit,
        config,
    );
    (paint, error, run_full_fit)
}

fn gradient_gain_is_sufficient(
    solid: ErrorStats,
    gradient: ErrorStats,
    small_region: bool,
    minimum_improvement: f32,
    region_area: usize,
    minimum_gradient_area: usize,
) -> bool {
    let required_gain = if small_region {
        // Charge every new gradient the same minimum total perceptual gain
        // as a face at the configured area threshold.  Without this
        // normalisation, many tiny faces can each buy an SVG definition with
        // a small per-pixel improvement and make the vector representation
        // substantially more complex for little image-wide benefit.
        // One full DeltaE00 just-noticeable-difference over the minimum face
        // area is the fixed perceptual budget for adding an SVG definition.
        small_region_office_required_gain(minimum_improvement, region_area, minimum_gradient_area)
    } else {
        0.05 * 2.3
    };
    let relative_gain = !small_region || gradient.mean <= solid.mean * 0.88;
    relative_gain
        && solid.mean - gradient.mean >= required_gain
        && gradient.percentile <= solid.percentile + 1e-4
}

fn small_region_office_required_gain(
    minimum_improvement: f32,
    region_area: usize,
    minimum_gradient_area: usize,
) -> f32 {
    (4.0 * minimum_improvement) * minimum_gradient_area.max(1) as f32 / region_area.max(1) as f32
}

#[allow(clippy::too_many_arguments)]
fn fit_region_samples(
    label: usize,
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    area: usize,
    region_bounds: Bounds,
    canonical_solid: [f32; 3],
    directional_only: bool,
    run_full_fit: bool,
    config: &Config,
) -> (Paint, f32) {
    // The reference fits Solid from the quantized canonical image while all
    // model errors are measured on native Paint samples.  Using the source
    // mean here makes every nominally solid face a different colour before
    // gradient selection even starts.
    let solid_color = canonical_solid;
    let solid_error = paint_error_against_labs(source_labs, samples, |_| solid_color);
    if !run_full_fit {
        return (Paint::Solid { color: solid_color }, solid_error.mean);
    }
    let trace = cfg!(feature = "diagnostics")
        && std::env::var("PICVEC_TRACE_PAINT_LABEL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            == Some(label);
    let small_region = area < config.minimum_gradient_area as usize;
    let minimum_improvement = 0.25 * 2.3;
    // Small faces start from Solid and can only be promoted by the Office
    // candidate. Its error is non-negative, so when the fixed complexity
    // charge exceeds the complete Solid error no gradient can possibly win.
    // Return before the median/range calculation as well as candidate fitting.
    if small_region
        && solid_error.mean
            < small_region_office_required_gain(
                minimum_improvement,
                area,
                config.minimum_gradient_area as usize,
            )
    {
        return (Paint::Solid { color: solid_color }, solid_error.mean);
    }
    let sample_labs: Vec<Lab> = samples.iter().map(|&index| source_labs[index]).collect();
    let median_lab = Lab {
        l: median(&mut sample_labs.iter().map(|value| value.l).collect::<Vec<_>>()),
        a: median(&mut sample_labs.iter().map(|value| value.a).collect::<Vec<_>>()),
        b: median(&mut sample_labs.iter().map(|value| value.b).collect::<Vec<_>>()),
    };
    let perceptual_range = percentile(
        delta_e2000_pairs(&sample_labs, &vec![median_lab; sample_labs.len()]),
        0.90,
    );
    // A complete just-noticeable-difference is too coarse for smooth,
    // low-chroma shading: on light monochrome artwork it can make a visibly
    // modelled ramp look flat after vectorization. The improvement and
    // complexity gates below still prevent gratuitous SVG gradients.
    if perceptual_range <= config.solid_color_max_delta_e {
        return (Paint::Solid { color: solid_color }, solid_error.mean);
    }
    // Rank every geometry family with one perceptual objective. Previously
    // the source-fitted direction and the Office presets had different
    // promotion thresholds, so a less accurate radial preset could replace
    // Solid after a better continuously rotated linear fit was rejected.
    let mut candidates = vec![(Paint::Solid { color: solid_color }, solid_error)];
    if !small_region {
        candidates.push(legacy_gradient_candidate(
            source,
            source_labs,
            samples,
            region_bounds,
            directional_only,
        ));
    }

    let office_candidate = office_gradient_candidate(
        source,
        source_labs,
        samples,
        region_bounds,
        config.maximum_gradient_stops,
    )
    .filter(|(paint, _)| !directional_only || matches!(paint, Paint::Linear { .. }));
    if let Some(candidate) = office_candidate.clone() {
        candidates.push(candidate);
    }
    let (selected, selected_stats) = candidates
        .into_iter()
        .min_by(|left, right| objective(left.1).total_cmp(&objective(right.1)))
        .expect("Solid always supplies one Paint candidate");
    if trace {
        eprintln!(
            "paint trace label={label} area={area} samples={} solid={solid_error:?} selected={selected:?} selected_stats={selected_stats:?} office={office_candidate:?}",
            samples.len(),
        );
    }
    if !matches!(selected, Paint::Solid { .. })
        && gradient_gain_is_sufficient(
            solid_error,
            selected_stats,
            small_region,
            minimum_improvement,
            area,
            config.minimum_gradient_area as usize,
        )
    {
        return (selected, selected_stats.mean);
    }
    (Paint::Solid { color: solid_color }, solid_error.mean)
}

fn fitted_stops_direct(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    offsets: &[f64],
) -> Vec<ColorStop> {
    let count = offsets.len();
    if count == 2 && offsets[0] == 0.0 && offsets[1] == 1.0 {
        let right = parameters.to_vec();
        let left: Vec<f32> = parameters
            .iter()
            .map(|&parameter| 1.0 - parameter)
            .collect();
        let aa = numpy_sum_f32(&left.iter().map(|value| value * value).collect::<Vec<_>>());
        let ab = numpy_sum_f32(
            &left
                .iter()
                .zip(&right)
                .map(|(&first, &second)| first * second)
                .collect::<Vec<_>>(),
        );
        let bb = numpy_sum_f32(&right.iter().map(|value| value * value).collect::<Vec<_>>());
        let mut left_values = [-0.0_f32; 3];
        let mut right_values = [-0.0_f32; 3];
        // These are np.sum(N x 3, axis=0), hence the same sequential strided
        // reduction as mean_color rather than NumPy's contiguous pairwise sum.
        for ((&index, &left_weight), &right_weight) in samples.iter().zip(&left).zip(&right) {
            for channel in 0..3 {
                left_values[channel] += left_weight * source.pixels[index][channel];
                right_values[channel] += right_weight * source.pixels[index][channel];
            }
        }
        // `float(np.sum(...))` promotes these three coefficients to Python
        // float before the determinant is evaluated.  NumPy then casts that
        // scalar back to float32 for the array division below.
        let determinant_f64 = aa as f64 * bb as f64 - ab as f64 * ab as f64;
        let determinant = determinant_f64 as f32;
        if cfg!(feature = "diagnostics") && std::env::var("PICVEC_TRACE_FIT_SAMPLES").is_ok() {
            let requested = std::env::var("PICVEC_TRACE_FIT_SAMPLES").unwrap_or_default();
            let actual = samples
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            if requested == actual {
                eprintln!(
                    "trace fit direct parameters={parameters:?} aa={aa:?} ab={ab:?} bb={bb:?} det64={determinant_f64:?} det={determinant:?} left_values={left_values:?} right_values={right_values:?}"
                );
            }
        }
        let colors = if determinant_f64 > 1e-8 {
            [
                [
                    (left_values[0] * bb - right_values[0] * ab) / determinant,
                    (left_values[1] * bb - right_values[1] * ab) / determinant,
                    (left_values[2] * bb - right_values[2] * ab) / determinant,
                ],
                [
                    (right_values[0] * aa - left_values[0] * ab) / determinant,
                    (right_values[1] * aa - left_values[1] * ab) / determinant,
                    (right_values[2] * aa - left_values[2] * ab) / determinant,
                ],
            ]
        } else {
            let color = mean_color(source, samples);
            [color, color]
        };
        return offsets
            .iter()
            .enumerate()
            .map(|(index, &offset)| ColorStop {
                offset,
                color: [
                    colors[index][0].clamp(0.0, 1.0) as f64,
                    colors[index][1].clamp(0.0, 1.0) as f64,
                    colors[index][2].clamp(0.0, 1.0) as f64,
                ],
            })
            .collect();
    }
    let mut normal = vec![vec![0.0_f64; count]; count];
    let mut targets = vec![vec![0.0_f64; count]; 3];
    for (&index, &parameter) in samples.iter().zip(parameters) {
        let (left, right, alpha) = interpolation_weights(parameter, offsets);
        let basis = [(left, 1.0 - alpha), (right, alpha)];
        for &(first, first_value) in &basis {
            for &(second, second_value) in &basis {
                normal[first][second] += first_value as f64 * second_value as f64;
            }
            for (channel, target) in targets.iter_mut().enumerate() {
                target[first] += first_value as f64 * source.pixels[index][channel] as f64;
            }
        }
    }
    if count >= 3 {
        let lambda = 0.35_f64;
        for row in 0..count - 2 {
            let difference = [(row, 1.0_f64), (row + 1, -2.0), (row + 2, 1.0)];
            for &(first, first_value) in &difference {
                for &(second, second_value) in &difference {
                    normal[first][second] += lambda * first_value * second_value;
                }
            }
        }
    }
    let colors: Vec<Vec<f64>> = targets
        .into_iter()
        .map(|target| solve_system(normal.clone(), target))
        .collect();
    offsets
        .iter()
        .enumerate()
        .map(|(index, &offset)| ColorStop {
            offset,
            color: [
                colors[0][index].clamp(0.0, 1.0),
                colors[1][index].clamp(0.0, 1.0),
                colors[2][index].clamp(0.0, 1.0),
            ],
        })
        .collect()
}

fn paint_at(paint: &Paint, index: usize, width: usize) -> [f32; 3] {
    match paint {
        Paint::Solid { color } => *color,
        Paint::Linear {
            start, end, stops, ..
        } => interpolate(stops, linear_parameter(index, width, *start, *end)),
        Paint::Radial {
            center,
            radius,
            stops,
            ..
        } => interpolate(stops, radial_parameter(index, width, *center, *radius)),
        Paint::Layered { base, overlays } => {
            overlays
                .iter()
                .fold(paint_at(base, index, width), |under, overlay| {
                    let over = paint_at(&overlay.paint, index, width);
                    let parameter = match overlay.paint.as_ref() {
                        Paint::Linear { start, end, .. } => {
                            linear_parameter(index, width, *start, *end)
                        }
                        Paint::Radial { center, radius, .. } => {
                            radial_parameter(index, width, *center, *radius)
                        }
                        Paint::Solid { .. } | Paint::Layered { .. } => 0.0,
                    };
                    let alpha = interpolate_opacity(&overlay.opacity_stops, parameter);
                    [0, 1, 2].map(|channel| under[channel] * (1.0 - alpha) + over[channel] * alpha)
                })
        }
    }
}

fn interpolate_opacity(stops: &[OpacityStop], parameter: f32) -> f32 {
    if stops.is_empty() {
        return 1.0;
    }
    let parameter = parameter.clamp(0.0, 1.0) as f64;
    if parameter <= stops[0].offset {
        return stops[0].opacity.clamp(0.0, 1.0) as f32;
    }
    for pair in stops.windows(2) {
        if parameter <= pair[1].offset {
            let span = (pair[1].offset - pair[0].offset).max(1e-12);
            let amount = ((parameter - pair[0].offset) / span).clamp(0.0, 1.0);
            return (pair[0].opacity * (1.0 - amount) + pair[1].opacity * amount).clamp(0.0, 1.0)
                as f32;
        }
    }
    stops
        .last()
        .map(|stop| stop.opacity.clamp(0.0, 1.0) as f32)
        .unwrap_or(1.0)
}

fn paint_stats(source: &Raster, samples: &[usize], paint: &Paint) -> ErrorStats {
    paint_error(source, samples, |index| {
        paint_at(paint, index, source.width)
    })
}

fn expand_merge_stops(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    template: &Paint,
    initial_stops: Vec<ColorStop>,
    initial_stats: ErrorStats,
    maximum: usize,
) -> (Paint, ErrorStats) {
    let trace = cfg!(feature = "diagnostics")
        && std::env::var("PICVEC_TRACE_FIT_SAMPLES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| part.parse::<usize>().ok())
                    .eq(samples.iter().copied())
            })
            .unwrap_or(false);
    let mut offsets = vec![0.0_f64, 1.0];
    let mut current_stops = initial_stops;
    let mut current_stats = initial_stats;
    let mut accepted_stops = current_stops.clone();
    let mut accepted_stats = current_stats;
    let initial_objective = objective(initial_stats);
    let mut accepted_objective = initial_objective;
    let mut smooth_profile = None::<bool>;
    while offsets.len() < maximum.clamp(2, 5) {
        let mut best: Option<(f32, f64, Vec<ColorStop>, ErrorStats)> = None;
        for step in 1..=9 {
            let offset = 0.1_f64 + (step - 1) as f64 * 0.1_f64;
            if offsets.iter().any(|&value| (value - offset).abs() < 0.075) {
                continue;
            }
            let mut proposed = offsets.clone();
            proposed.push(offset);
            proposed.sort_by(f64::total_cmp);
            let stops = fitted_stops_direct(source, samples, parameters, &proposed);
            let stats = gradient_error(source, samples, parameters, &stops);
            let candidate = objective(stats);
            if trace {
                eprintln!(
                    "trace expand offsets={:?} candidate={:?} stats=({:?},{:?}) stops={:?}",
                    proposed, candidate, stats.mean, stats.percentile, stops,
                );
            }
            if best
                .as_ref()
                .map(|value| candidate < value.0)
                .unwrap_or(true)
            {
                best = Some((candidate, offset, stops, stats));
            }
        }
        let Some((candidate, chosen, stops, stats)) = best else {
            break;
        };
        if trace {
            eprintln!(
                "trace expand best chosen={:?} candidate={:?} current={:?} accepted={:?}",
                chosen,
                candidate,
                objective(current_stats),
                accepted_objective,
            );
        }
        if candidate >= objective(current_stats) {
            break;
        }
        let immediate_gain = objective(current_stats) - candidate;
        let cumulative_gain = accepted_objective - candidate;
        if immediate_gain < 0.15 && cumulative_gain < 0.15 {
            let smooth = *smooth_profile
                .get_or_insert_with(|| merge_profile_is_smooth(source, samples, parameters));
            if trace {
                eprintln!("trace expand smooth={smooth}");
            }
            if !smooth {
                break;
            }
        }
        offsets.push(chosen);
        offsets.sort_by(f64::total_cmp);
        current_stops = stops;
        current_stats = stats;
        let accepted_gain = if smooth_profile == Some(true) {
            initial_objective - candidate
        } else {
            cumulative_gain
        };
        if accepted_gain >= 0.15 {
            accepted_stops = current_stops.clone();
            accepted_stats = current_stats;
            accepted_objective = candidate;
        }
    }
    (paint_with_stops(template, accepted_stops), accepted_stats)
}

fn merge_profile_is_smooth(source: &Raster, samples: &[usize], parameters: &[f32]) -> bool {
    let bin_count = 64_usize.min(samples.len().max(8)).max(8);
    let mut bins = vec![Vec::<[f32; 3]>::new(); bin_count];
    for (&index, &parameter) in samples.iter().zip(parameters) {
        let bin = ((parameter.clamp(0.0, 1.0) * bin_count as f32) as usize).min(bin_count - 1);
        bins[bin].push(source.pixels[index]);
    }
    let positions: Vec<f32> = (0..bin_count)
        .map(|index| (index as f32 + 0.5) / bin_count as f32)
        .collect();
    let mut profile = vec![[f32::NAN; 3]; bin_count];
    let valid: Vec<usize> = bins
        .iter()
        .enumerate()
        .filter_map(|(index, values)| (!values.is_empty()).then_some(index))
        .collect();
    for &index in &valid {
        for channel in 0..3 {
            let mut values: Vec<f32> = bins[index].iter().map(|value| value[channel]).collect();
            profile[index][channel] = median(&mut values);
        }
    }
    if valid.len() == 1 {
        let color = profile[valid[0]];
        profile.fill(color);
    } else if !valid.is_empty() {
        for index in 0..bin_count {
            if profile[index][0].is_finite() {
                continue;
            }
            let right_position = valid.partition_point(|&value| value < index);
            let left = valid[right_position.saturating_sub(1)];
            let right = valid[right_position.min(valid.len() - 1)];
            let amount = if left == right {
                0.0
            } else {
                (positions[index] - positions[left]) / (positions[right] - positions[left])
            };
            let left_color = profile[left];
            let right_color = profile[right];
            for (channel, value) in profile[index].iter_mut().enumerate() {
                *value = left_color[channel] * (1.0 - amount) + right_color[channel] * amount;
            }
        }
    }
    let kernel: Vec<f64> = (-4_i32..=4)
        .map(|offset| (-0.5 * (offset * offset) as f64).exp())
        .collect();
    let kernel_sum = kernel.iter().sum::<f64>();
    let original = profile.clone();
    for (index, color) in profile.iter_mut().enumerate() {
        for (channel, value) in color.iter_mut().enumerate() {
            *value = (kernel
                .iter()
                .enumerate()
                .map(|(kernel_index, &weight)| {
                    let offset = kernel_index as isize - 4;
                    let sample =
                        (index as isize + offset).clamp(0, bin_count as isize - 1) as usize;
                    original[sample][channel] as f64 * weight
                })
                .sum::<f64>()
                / kernel_sum) as f32;
        }
    }
    let labs = preprocess_lab_values(&profile);
    labs.windows(2)
        .map(|pair| delta_e2000(pair[0], pair[1]))
        .fold(0.0_f32, f32::max)
        <= 5.0
}

/// Fit the exact Solid/linear/radial family used by the exporter, without
/// profile smoothing.  Region merging must explain the actual samples on both
/// sides of a seam; using the final per-region profile here can hide a local
/// failure on the smaller child.
fn fit_merge_paint(
    source: &Raster,
    samples: &[usize],
    region_bounds: Bounds,
    maximum_stops: usize,
) -> (Paint, ErrorStats) {
    let trace_fit = cfg!(feature = "diagnostics")
        && std::env::var("PICVEC_TRACE_FIT_SAMPLES")
            .ok()
            .map(|value| {
                let requested: Vec<usize> = value
                    .split(',')
                    .filter_map(|part| part.parse::<usize>().ok())
                    .collect();
                requested == samples
            })
            .unwrap_or(false);
    let solid_color = mean_color(source, samples);
    let solid = Paint::Solid { color: solid_color };
    let solid_stats = paint_stats(source, samples, &solid);
    let mut candidates = Vec::<(f32, Paint, Vec<f32>, ErrorStats)>::new();
    for preset in [
        LinearPreset::LeftToRight,
        LinearPreset::TopToBottom,
        LinearPreset::TopLeftToBottomRight,
        LinearPreset::TopRightToBottomLeft,
    ] {
        let (start, end) = linear_geometry(preset, region_bounds);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| linear_parameter(index, source.width, start, end).clamp(0.0, 1.0))
            .collect();
        let stops = fitted_stops_direct(source, samples, &parameters, &[0.0, 1.0]);
        let paint = Paint::Linear {
            preset,
            start,
            end,
            stops,
        };
        let stats = paint_stats(source, samples, &paint);
        candidates.push((
            paint_rgb_mse(source, samples, &paint),
            paint,
            parameters,
            stats,
        ));
    }
    for origin in [
        RadialOrigin::Center,
        RadialOrigin::TopLeft,
        RadialOrigin::TopRight,
        RadialOrigin::BottomLeft,
        RadialOrigin::BottomRight,
    ] {
        let (center, radius) = radial_geometry(origin, region_bounds);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| radial_parameter(index, source.width, center, radius))
            .collect();
        let stops = fitted_stops_direct(source, samples, &parameters, &[0.0, 1.0]);
        let paint = Paint::Radial {
            origin,
            center,
            radius,
            stops,
        };
        let stats = paint_stats(source, samples, &paint);
        candidates.push((
            paint_rgb_mse(source, samples, &paint),
            paint,
            parameters,
            stats,
        ));
    }
    if trace_fit {
        for (mse, paint, _, stats) in &candidates {
            let references: Vec<[f32; 3]> =
                samples.iter().map(|&index| source.pixels[index]).collect();
            let rendered: Vec<[f32; 3]> = samples
                .iter()
                .map(|&index| paint_at(paint, index, source.width))
                .collect();
            let reference_lab = preprocess_color_values(references);
            let rendered_lab = preprocess_color_values(rendered);
            eprintln!(
                "trace fit candidate mse={:?} stats=({:?},{:?}) errors={:?} reference_lab={:?} rendered_lab={:?} paint={:?}",
                mse,
                stats.mean,
                stats.percentile,
                delta_e2000_pairs(&reference_lab, &rendered_lab),
                reference_lab,
                rendered_lab,
                paint,
            );
        }
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    // RGB MSE first selects three cheap two-stop finalists.  The reference
    // expands only the best perceptual finalist; expanding all three here can
    // find a lower-error model that Python never offered and changes the RAG
    // merge order even though every threshold is otherwise identical.
    let finalist_count = candidates.len().min(3);
    let selected = (0..finalist_count)
        .min_by(|&left, &right| {
            objective(candidates[left].3).total_cmp(&objective(candidates[right].3))
        })
        .unwrap_or(0);
    let expand = |candidate_index: usize| {
        let (_, template, parameters, two_stop_stats) = &candidates[candidate_index];
        let initial_stops = match template {
            Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => stops.clone(),
            Paint::Solid { .. } | Paint::Layered { .. } => unreachable!(),
        };
        expand_merge_stops(
            source,
            samples,
            parameters,
            template,
            initial_stops,
            *two_stop_stats,
            maximum_stops,
        )
    };
    let mut best = expand(selected);
    if trace_fit {
        eprintln!("trace fit selected={} expanded={:?}", selected, best);
    }
    let required = 0.35_f32.max(0.08 * solid_stats.mean);
    let percentile_guard = solid_stats.percentile + 1.0_f32.max(0.10 * solid_stats.percentile);
    let mut accepted =
        solid_stats.mean - best.1.mean >= required && best.1.percentile <= percentile_guard;

    // A two-stop shortlist can rank differently after non-linear stops are
    // added.  Python retries only when the initially selected gradient would
    // fall back to Solid, using the other RGB finalists plus up to three best
    // two-stop mean-DeltaE geometries (at most six unique candidates).
    if !accepted && maximum_stops > 2 {
        let mut rescue_indices: Vec<usize> = (0..finalist_count).collect();
        let mut by_mean: Vec<usize> = (0..candidates.len()).collect();
        by_mean
            .sort_by(|&left, &right| candidates[left].3.mean.total_cmp(&candidates[right].3.mean));
        for candidate in by_mean.into_iter().take(3) {
            if !rescue_indices.contains(&candidate) {
                rescue_indices.push(candidate);
            }
        }
        for candidate in rescue_indices {
            if candidate == selected {
                continue;
            }
            let expanded = expand(candidate);
            if trace_fit {
                eprintln!("trace fit rescue={} expanded={:?}", candidate, expanded);
            }
            if objective(expanded.1) < objective(best.1) {
                best = expanded;
            }
        }
        accepted =
            solid_stats.mean - best.1.mean >= required && best.1.percentile <= percentile_guard;
    }
    if accepted {
        best
    } else {
        (solid, solid_stats)
    }
}

fn paint_rgb_mse(source: &Raster, samples: &[usize], paint: &Paint) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut squared = Vec::with_capacity(samples.len() * 3);
    for &index in samples {
        let predicted = paint_at(paint, index, source.width);
        for (channel, &value) in predicted.iter().enumerate() {
            let difference = value - source.pixels[index][channel];
            squared.push(difference * difference);
        }
    }
    numpy_sum_f32(&squared) / squared.len() as f32
}

/// Explain smooth two-dimensional residuals without cutting the owning face.
///
/// A small set of elliptical, transparent radial layers is deliberately used
/// instead of a raster patch.  The candidate search is RGB-only and therefore
/// cheap; the caller still applies the ordinary CIEDE2000 acceptance gate to
/// the winning Paint.
fn fit_layered_residual_paint(
    source: &Raster,
    samples: &[usize],
    region_bounds: Bounds,
    base: Paint,
    maximum_layers: usize,
) -> (Paint, ErrorStats) {
    if samples.len() < 48 || maximum_layers == 0 {
        let stats = paint_stats(source, samples, &base);
        return (base, stats);
    }
    let span_x = (region_bounds.max_x - region_bounds.min_x).max(1.0);
    let span_y = (region_bounds.max_y - region_bounds.min_y).max(1.0);
    if span_x.max(span_y) < 12.0 {
        let stats = paint_stats(source, samples, &base);
        return (base, stats);
    }

    let base_stats = paint_stats(source, samples, &base);
    let original_base = base.clone();
    let mut current = base;
    let mut current_mse = paint_rgb_mse(source, samples, &current);
    let mut overlays = Vec::<PaintOverlay>::new();
    for _ in 0..maximum_layers.min(3) {
        let mut ranked = samples
            .iter()
            .copied()
            .map(|index| {
                let predicted = paint_at(&current, index, source.width);
                let error = (0..3)
                    .map(|channel| {
                        let difference = source.pixels[index][channel] - predicted[channel];
                        difference * difference
                    })
                    .sum::<f32>();
                (index, error)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut centres = Vec::<usize>::new();
        for (index, _) in ranked {
            let x = (index % source.width) as f32;
            let y = (index / source.width) as f32;
            if centres.iter().all(|&existing| {
                let ex = (existing % source.width) as f32;
                let ey = (existing / source.width) as f32;
                ((x - ex) / span_x).hypot((y - ey) / span_y) >= 0.16
            }) {
                centres.push(index);
                if centres.len() == 1 {
                    break;
                }
            }
        }

        let mut best = None::<(f32, PaintOverlay)>;
        for centre_index in centres {
            let center = Point {
                x: (centre_index % source.width) as f32,
                y: (centre_index / source.width) as f32,
            };
            for radius_scale in [0.24_f32, 0.46, 0.72] {
                let radius = Point {
                    x: (span_x * radius_scale).max(6.0),
                    y: (span_y * radius_scale).max(6.0),
                };
                for peak_opacity in [0.55_f32, 0.80] {
                    let opacity_stops = vec![
                        OpacityStop {
                            offset: 0.0,
                            opacity: peak_opacity as f64,
                        },
                        OpacityStop {
                            offset: 0.55,
                            opacity: (peak_opacity * 0.35) as f64,
                        },
                        OpacityStop {
                            offset: 1.0,
                            opacity: 0.0,
                        },
                    ];
                    let mut target = [0.0_f64; 3];
                    let mut denominator = 0.0_f64;
                    for &index in samples {
                        let parameter = radial_parameter(index, source.width, center, radius);
                        let alpha = interpolate_opacity(&opacity_stops, parameter) as f64;
                        if alpha <= 1e-5 {
                            continue;
                        }
                        let under = paint_at(&current, index, source.width);
                        denominator += alpha * alpha;
                        for channel in 0..3 {
                            target[channel] += alpha
                                * (source.pixels[index][channel] as f64
                                    - under[channel] as f64 * (1.0 - alpha));
                        }
                    }
                    if denominator <= 1e-8 {
                        continue;
                    }
                    let color = target.map(|value| (value / denominator).clamp(0.0, 1.0));
                    let overlay = PaintOverlay {
                        paint: Box::new(Paint::Radial {
                            origin: RadialOrigin::Fitted,
                            center,
                            radius,
                            stops: vec![
                                ColorStop { offset: 0.0, color },
                                ColorStop { offset: 1.0, color },
                            ],
                        }),
                        opacity_stops,
                    };
                    let candidate = Paint::Layered {
                        base: Box::new(current.clone()),
                        overlays: vec![overlay.clone()],
                    };
                    let mse = paint_rgb_mse(source, samples, &candidate);
                    if best.as_ref().is_none_or(|(best_mse, _)| mse < *best_mse) {
                        best = Some((mse, overlay));
                    }
                }
            }
        }
        let Some((candidate_mse, overlay)) = best else {
            break;
        };
        // Require a material residual reduction so layers do not accumulate
        // merely to chase rounding noise in already coherent Paint.
        if candidate_mse > current_mse * 0.92 || current_mse - candidate_mse < 1e-7 {
            break;
        }
        current = Paint::Layered {
            base: Box::new(current),
            overlays: vec![overlay.clone()],
        };
        overlays.push(overlay);
        current_mse = candidate_mse;
    }

    if overlays.is_empty() {
        return (current, base_stats);
    }
    // Flatten the temporary one-layer nesting so SVG emission can reuse one
    // face geometry for every overlay.
    let layered = Paint::Layered {
        base: Box::new(original_base.clone()),
        overlays,
    };
    let stats = paint_stats(source, samples, &layered);
    if objective(stats) + 0.02 < objective(base_stats) {
        (layered, stats)
    } else {
        (original_base, base_stats)
    }
}

#[derive(Clone)]
struct MergeRegion {
    labels: HashSet<usize>,
    pixels: Vec<usize>,
    samples: Vec<usize>,
    bounds: Bounds,
    paint: Paint,
}

#[derive(Clone)]
struct StructuralColorBoundary {
    labels: (usize, usize),
    sample_pairs: Vec<(usize, usize)>,
    median_delta_e: f32,
}

#[derive(Clone)]
struct MergeProposal {
    samples: Vec<usize>,
    paint: Paint,
    score: f32,
}

#[derive(Clone)]
struct MergeQueueEntry {
    score: f32,
    sequence: usize,
    left: usize,
    right: usize,
    left_version: usize,
    right_version: usize,
    proposal: MergeProposal,
}

impl PartialEq for MergeQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl Eq for MergeQueueEntry {}

impl PartialOrd for MergeQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

fn union_bounds(first: Bounds, second: Bounds) -> Bounds {
    Bounds {
        min_x: first.min_x.min(second.min_x),
        min_y: first.min_y.min(second.min_y),
        max_x: first.max_x.max(second.max_x),
        max_y: first.max_y.max(second.max_y),
    }
}

fn balanced_samples(
    first: &[usize],
    second: &[usize],
    first_weight: usize,
    second_weight: usize,
    maximum: usize,
) -> Vec<usize> {
    let total = (first_weight + second_weight).max(1);
    let first_count = (((maximum as f64 * first_weight as f64 / total as f64).round_ties_even())
        as usize)
        .clamp(1, maximum.saturating_sub(1));
    let second_count = maximum.saturating_sub(first_count).max(1);
    let mut result = sampled_indices(first, first_count);
    result.extend(sampled_indices(second, second_count));
    result
}

fn balanced_sample_parts(
    first: &[usize],
    second: &[usize],
    first_weight: usize,
    second_weight: usize,
    maximum: usize,
) -> (Vec<usize>, Vec<usize>) {
    let total = (first_weight + second_weight).max(1);
    let first_count = (((maximum as f64 * first_weight as f64 / total as f64).round_ties_even())
        as usize)
        .clamp(1, maximum.saturating_sub(1));
    let second_count = maximum.saturating_sub(first_count).max(1);
    (
        sampled_indices(first, first_count),
        sampled_indices(second, second_count),
    )
}

fn inset_region_samples(
    pixels: &[usize],
    labels: &[u32],
    width: usize,
    height: usize,
    inset: f64,
) -> Vec<usize> {
    if pixels.is_empty() || inset <= 0.0 {
        return pixels.to_vec();
    }
    let owner = labels[pixels[0]];
    // scipy.distance_transform_edt(mask) > inset is equivalent to requiring
    // every integer pixel-centre site at Euclidean distance <= inset to have
    // the same owner.  The inset is resolution-scaled (1.5 px at 768), not a
    // fixed radius-two stencil.
    let radius = inset.ceil() as isize;
    let inset_squared = inset * inset;
    let mut interior = Vec::new();
    'pixel: for &index in pixels {
        let x = (index % width) as isize;
        let y = (index / width) as isize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if (dx * dx + dy * dy) as f64 > inset_squared {
                    continue;
                }
                let px = x + dx;
                let py = y + dy;
                if px < 0
                    || py < 0
                    || px >= width as isize
                    || py >= height as isize
                    || labels[py as usize * width + px as usize] != owner
                {
                    continue 'pixel;
                }
            }
        }
        interior.push(index);
    }
    let minimum = pixels
        .len()
        .min(8_usize.max(64_usize.min(pixels.len() / 4)));
    if interior.len() < minimum {
        pixels.to_vec()
    } else {
        interior
    }
}

fn same_paint_family(first: &Paint, second: &Paint) -> bool {
    match (first, second) {
        (Paint::Solid { .. }, Paint::Solid { .. }) => true,
        (Paint::Linear { preset: left, .. }, Paint::Linear { preset: right, .. }) => left == right,
        (Paint::Radial { origin: left, .. }, Paint::Radial { origin: right, .. }) => left == right,
        _ => false,
    }
}

fn fit_like_merge_paint(
    source: &Raster,
    samples: &[usize],
    region_bounds: Bounds,
    template: &Paint,
    solid: &Paint,
    solid_stats: ErrorStats,
) -> (Paint, ErrorStats) {
    let geometry = match template {
        Paint::Solid { .. } | Paint::Layered { .. } => {
            return (solid.clone(), solid_stats);
        }
        Paint::Linear { preset, .. } => {
            let (start, end) = linear_geometry(*preset, region_bounds);
            let parameters: Vec<f32> = samples
                .iter()
                .map(|&index| linear_parameter(index, source.width, start, end).clamp(0.0, 1.0))
                .collect();
            let stops = fitted_stops_direct(source, samples, &parameters, &[0.0, 1.0]);
            Paint::Linear {
                preset: *preset,
                start,
                end,
                stops,
            }
        }
        Paint::Radial { origin, .. } => {
            let (center, radius) = radial_geometry(*origin, region_bounds);
            let parameters: Vec<f32> = samples
                .iter()
                .map(|&index| radial_parameter(index, source.width, center, radius))
                .collect();
            let stops = fitted_stops_direct(source, samples, &parameters, &[0.0, 1.0]);
            Paint::Radial {
                origin: *origin,
                center,
                radius,
                stops,
            }
        }
    };
    let stats = paint_stats(source, samples, &geometry);
    let required = 0.35_f32.max(0.08 * solid_stats.mean);
    let percentile_guard = solid_stats.percentile + 1.0_f32.max(0.10 * solid_stats.percentile);
    if solid_stats.mean - stats.mean >= required && stats.percentile <= percentile_guard {
        (geometry, stats)
    } else {
        (solid.clone(), solid_stats)
    }
}

fn merge_proposal(
    source: &Raster,
    first: &MergeRegion,
    second: &MergeRegion,
    config: &Config,
) -> MergeProposal {
    let trace_pair = cfg!(feature = "diagnostics")
        && std::env::var("PICVEC_TRACE_MERGE_PAIR")
            .ok()
            .and_then(|value| {
                let mut values = value
                    .split(',')
                    .filter_map(|part| part.parse::<usize>().ok());
                Some((values.next()?, values.next()?))
            })
            .map(|(left, right)| {
                (first.labels.len() == 1
                    && second.labels.len() == 1
                    && first.labels.contains(&left)
                    && second.labels.contains(&right))
                    || (first.labels.len() == 1
                        && second.labels.len() == 1
                        && first.labels.contains(&right)
                        && second.labels.contains(&left))
            })
            .unwrap_or(false);
    let (first_samples, second_samples) = balanced_sample_parts(
        &first.samples,
        &second.samples,
        first.pixels.len(),
        second.pixels.len(),
        384,
    );
    let mut quick_samples = first_samples.clone();
    quick_samples.extend_from_slice(&second_samples);
    let first_mean = mean_color_f64(source, &first_samples);
    let second_mean = mean_color_f64(source, &second_samples);
    let mean_labs = preprocess_color_values(vec![first_mean, second_mean]);
    let mean_delta = delta_e2000_pairs(&mean_labs[..1], &mean_labs[1..])[0];
    let hard_edge = mean_delta > 22.0;
    let limit = if hard_edge {
        config
            .gradient_merge_error
            .min((0.25 * config.gradient_merge_error).max(0.75))
    } else {
        config.gradient_merge_error
    };
    let solid = Paint::Solid {
        color: mean_color(source, &quick_samples),
    };
    let solid_stats = paint_stats(source, &quick_samples, &solid);
    let mut paint = solid.clone();
    let mut score = objective(solid_stats)
        .max(objective(paint_stats(source, &first_samples, &solid)))
        .max(objective(paint_stats(source, &second_samples, &solid)));
    if score > limit {
        let combined_bounds = union_bounds(first.bounds, second.bounds);
        let mut preferred = Vec::<&Paint>::new();
        for candidate in [&first.paint, &second.paint] {
            if matches!(candidate, Paint::Solid { .. })
                || preferred
                    .iter()
                    .any(|existing| same_paint_family(existing, candidate))
            {
                continue;
            }
            preferred.push(candidate);
        }
        for template in preferred {
            let (candidate, candidate_stats) = fit_like_merge_paint(
                source,
                &quick_samples,
                combined_bounds,
                template,
                &solid,
                solid_stats,
            );
            let candidate_score = objective(candidate_stats)
                .max(objective(paint_stats(source, &first_samples, &candidate)))
                .max(objective(paint_stats(source, &second_samples, &candidate)));
            if candidate_score <= limit {
                paint = candidate;
                score = candidate_score;
                break;
            }
        }
        if score > limit {
            let (candidate, candidate_stats) = fit_merge_paint(
                source,
                &quick_samples,
                combined_bounds,
                config.maximum_gradient_stops,
            );
            paint = candidate;
            score = objective(candidate_stats)
                .max(objective(paint_stats(source, &first_samples, &paint)))
                .max(objective(paint_stats(source, &second_samples, &paint)));
        }
    }
    if hard_edge && score > limit {
        score = f32::INFINITY;
    }
    if trace_pair {
        eprintln!(
            "trace merge pair labels={:?}/{:?} pixels={}/{} sample_counts={}/{}/{} first_mean={:?} second_mean={:?} mean_delta={:?} solid={:?} solid_stats=({:?},{:?}) paint={:?} score={:?}",
            first.labels,
            second.labels,
            first.pixels.len(),
            second.pixels.len(),
            first.samples.len(),
            second.samples.len(),
            quick_samples.len(),
            first_mean,
            second_mean,
            mean_delta,
            solid,
            solid_stats.mean,
            solid_stats.percentile,
            paint,
            score,
        );
        let union_stats = paint_stats(source, &quick_samples, &paint);
        let first_stats = paint_stats(source, &first_samples, &paint);
        let second_stats = paint_stats(source, &second_samples, &paint);
        eprintln!(
            "trace stats union=({:?},{:?}) first=({:?},{:?}) second=({:?},{:?})",
            union_stats.mean,
            union_stats.percentile,
            first_stats.mean,
            first_stats.percentile,
            second_stats.mean,
            second_stats.percentile,
        );
    }
    MergeProposal {
        samples: balanced_samples(
            &first.samples,
            &second.samples,
            first.pixels.len(),
            second.pixels.len(),
            256,
        ),
        paint,
        score,
    }
}

fn pair(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn python_int_set_order(values: &[usize]) -> Vec<usize> {
    const LINEAR_PROBES: usize = 9;
    const PERTURB_SHIFT: usize = 5;

    fn slot(table: &[Option<usize>], value: usize) -> usize {
        let mask = table.len() - 1;
        let mut index = value & mask;
        let mut perturb = value;
        loop {
            let mut probe_index = index;
            let mut probes = if probe_index + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            loop {
                if table[probe_index].is_none() || table[probe_index] == Some(value) {
                    return probe_index;
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                probe_index += 1;
            }
            perturb >>= PERTURB_SHIFT;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    // CPython uses a distinct absent-key insertion probe while rebuilding a
    // resized set. Unlike set_add_entry's do/while, set_insert_clean checks
    // exactly LINEAR_PROBES following slots. Reusing `slot` here changes the
    // iteration order of larger adjacency sets and therefore equal-score
    // merge candidates.
    fn clean_slot(table: &[Option<usize>], value: usize) -> usize {
        let mask = table.len() - 1;
        let mut index = value & mask;
        let mut perturb = value;
        loop {
            if table[index].is_none() {
                return index;
            }
            if index + LINEAR_PROBES <= mask {
                let mut probe_index = index;
                for _ in 0..LINEAR_PROBES {
                    probe_index += 1;
                    if table[probe_index].is_none() {
                        return probe_index;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    let mut table = vec![None; 8];
    let mut used = 0_usize;
    for &value in values {
        let index = slot(&table, value);
        if table[index] == Some(value) {
            continue;
        }
        table[index] = Some(value);
        used += 1;
        if used * 5 >= (table.len() - 1) * 3 {
            let minimum = used * if used <= 50_000 { 4 } else { 2 };
            let mut size = 8_usize;
            while size <= minimum {
                size *= 2;
            }
            let previous = std::mem::replace(&mut table, vec![None; size]);
            for item in previous.into_iter().flatten() {
                let target = clean_slot(&table, item);
                table[target] = Some(item);
            }
        }
    }
    table.into_iter().flatten().collect()
}

#[allow(clippy::too_many_arguments)]
fn merge_candidate_proposal(
    source: &Raster,
    regions: &[Option<MergeRegion>],
    protected_by_label: &[HashSet<usize>],
    structural_boundaries: &[StructuralColorBoundary],
    structural_by_label: &[Vec<usize>],
    left: usize,
    right: usize,
    config: &Config,
) -> Option<MergeProposal> {
    let (left, right) = pair(left, right);
    let (Some(first), Some(second)) = (&regions[left], &regions[right]) else {
        return None;
    };
    if first.labels.iter().any(|&label| {
        protected_by_label[label]
            .iter()
            .any(|other| second.labels.contains(other))
    }) {
        return None;
    }
    let mut proposal = merge_proposal(source, first, second, config);
    let union_labels: HashSet<usize> = first.labels.union(&second.labels).copied().collect();
    let mut checked_boundaries = HashSet::<usize>::new();
    for &label in &union_labels {
        for &boundary_index in &structural_by_label[label] {
            if !checked_boundaries.insert(boundary_index) {
                continue;
            }
            let boundary = &structural_boundaries[boundary_index];
            if !union_labels.contains(&boundary.labels.0)
                || !union_labels.contains(&boundary.labels.1)
            {
                continue;
            }
            let first_colors: Vec<[f32; 3]> = boundary
                .sample_pairs
                .iter()
                .map(|&(first_index, _)| paint_at(&proposal.paint, first_index, source.width))
                .collect();
            let second_colors: Vec<[f32; 3]> = boundary
                .sample_pairs
                .iter()
                .map(|&(_, second_index)| paint_at(&proposal.paint, second_index, source.width))
                .collect();
            let first_lab = preprocess_color_values(first_colors);
            let second_lab = preprocess_color_values(second_colors);
            let mut predicted = delta_e2000_pairs(&first_lab, &second_lab);
            if median(&mut predicted) < 0.45 * boundary.median_delta_e {
                proposal.score = f32::INFINITY;
                break;
            }
        }
        if !proposal.score.is_finite() {
            break;
        }
    }
    Some(proposal)
}

#[allow(clippy::too_many_arguments)]
fn push_merge_candidate(
    source: &Raster,
    regions: &[Option<MergeRegion>],
    versions: &[usize],
    protected_by_label: &[HashSet<usize>],
    structural_boundaries: &[StructuralColorBoundary],
    structural_by_label: &[Vec<usize>],
    left: usize,
    right: usize,
    config: &Config,
    sequence: &mut usize,
    queue: &mut BinaryHeap<MergeQueueEntry>,
) {
    let (left, right) = pair(left, right);
    let Some(proposal) = merge_candidate_proposal(
        source,
        regions,
        protected_by_label,
        structural_boundaries,
        structural_by_label,
        left,
        right,
        config,
    ) else {
        return;
    };
    queue.push(MergeQueueEntry {
        score: proposal.score,
        sequence: *sequence,
        left,
        right,
        left_version: versions[left],
        right_version: versions[right],
        proposal,
    });
    *sequence += 1;
}

/// Merge quantizer bands only when a single Office-compatible Paint explains
/// their union and each child independently.  Strong measured interfaces and
/// explicit face barriers are propagated through the RAG, so a later merge
/// cannot cross a boundary merely because an intermediate band disappeared.
pub fn merge_partition(
    source: &Raster,
    edge_reference: &Raster,
    segmentation: &mut Segmentation,
    roles: &EdgeRoles,
    config: &Config,
) -> usize {
    if config.gradient_merge_error <= 0.0 || segmentation.regions.len() <= 1 {
        return 0;
    }
    let count = segmentation.regions.len();
    let mut pixels = vec![Vec::<usize>::new(); count];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        pixels[label as usize].push(index);
    }
    let paint_region_inset = (2.0 * source.width.max(source.height) as f64 / 1024.0).max(0.5);
    let mut regions: Vec<Option<MergeRegion>> = pixels
        .into_iter()
        .map(|pixels| {
            let selected = inset_region_samples(
                &pixels,
                &segmentation.labels,
                source.width,
                source.height,
                paint_region_inset,
            );
            // RegionCandidate stores at most 1024 samples.  Every later
            // resampling step operates on this fixed set, not on all inset
            // pixels.
            let selected = sampled_indices(&selected, 1024);
            let paint = Paint::Solid {
                color: mean_color(source, &selected),
            };
            Some(MergeRegion {
                labels: HashSet::new(),
                bounds: bounds(&pixels, source.width),
                pixels,
                samples: selected,
                paint,
            })
        })
        .collect();
    for (label, region) in regions.iter_mut().enumerate() {
        region.as_mut().unwrap().labels.insert(label);
    }
    let barrier = dilate_square(&roles.face_barrier, source.width, source.height, 1);
    let edge_lab = preprocess_lab_pixels(edge_reference);
    let mut adjacency = vec![HashSet::<usize>::new(); count];
    let mut evidence = HashMap::<(usize, usize), (Vec<(usize, usize)>, Vec<f32>, bool)>::new();
    let mut inspect = |first: usize, second: usize| {
        let first_label = segmentation.labels[first] as usize;
        let second_label = segmentation.labels[second] as usize;
        if first_label == second_label {
            return;
        }
        adjacency[first_label].insert(second_label);
        adjacency[second_label].insert(first_label);
        let entry = evidence
            .entry(pair(first_label, second_label))
            .or_insert_with(|| (Vec::new(), Vec::new(), false));
        entry.0.push(if first_label < second_label {
            (first, second)
        } else {
            (second, first)
        });
        entry.1.push(delta_e2000(edge_lab[first], edge_lab[second]));
        entry.2 |= barrier[first] || barrier[second];
    };
    for y in 0..source.height {
        for x in 0..source.width {
            let index = y * source.width + x;
            if x + 1 < source.width {
                inspect(index, index + 1);
            }
        }
    }
    for y in 0..source.height.saturating_sub(1) {
        for x in 0..source.width {
            let index = y * source.width + x;
            inspect(index, index + source.width);
        }
    }
    let mut protected_pairs = HashSet::<(usize, usize)>::new();
    let mut structural_boundaries = Vec::<StructuralColorBoundary>::new();
    let merge_edge_minimum_length = ((8.0 * source.width.max(source.height) as f64 / 1024.0)
        .max(0.5))
    .round_ties_even()
    .max(1.0) as usize;
    for (key, (sample_pairs, mut deltas, explicit_barrier)) in evidence {
        let length = deltas.len();
        let strong_fraction =
            deltas.iter().filter(|&&value| value >= 3.0).count() as f32 / length.max(1) as f32;
        let boundary_median = median(&mut deltas);
        if explicit_barrier {
            protected_pairs.insert(key);
        }
        if length >= merge_edge_minimum_length && boundary_median >= 3.0 && strong_fraction >= 0.45
        {
            let selected = if sample_pairs.len() <= 256 {
                sample_pairs
            } else {
                sampled_indices(&(0..sample_pairs.len()).collect::<Vec<_>>(), 256)
                    .into_iter()
                    .map(|index| sample_pairs[index])
                    .collect()
            };
            structural_boundaries.push(StructuralColorBoundary {
                labels: key,
                sample_pairs: selected,
                median_delta_e: boundary_median,
            });
        }
    }
    let mut structural_by_label = vec![Vec::<usize>::new(); count];
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let mut protected: Vec<(usize, usize)> = protected_pairs.iter().copied().collect();
        protected.sort_unstable();
        let mut structural: Vec<serde_json::Value> = structural_boundaries
            .iter()
            .map(|boundary| {
                serde_json::json!({
                    "labels": boundary.labels,
                    "median_delta_e": boundary.median_delta_e,
                    "sample_count": boundary.sample_pairs.len(),
                })
            })
            .collect();
        structural.sort_by_key(|value| {
            value["labels"][0]
                .as_u64()
                .unwrap_or(0)
                .saturating_mul(count as u64)
                + value["labels"][1].as_u64().unwrap_or(0)
        });
        if let Ok(value) = serde_json::to_string_pretty(&serde_json::json!({
            "protected": protected,
            "structural": structural,
        })) {
            let _ = std::fs::write(format!("{prefix}-merge-boundaries.json"), value);
        }
    }
    for (boundary_index, boundary) in structural_boundaries.iter().enumerate() {
        structural_by_label[boundary.labels.0].push(boundary_index);
        structural_by_label[boundary.labels.1].push(boundary_index);
    }
    let mut protected_by_label = vec![HashSet::<usize>::new(); count];
    for &(first, second) in &protected_pairs {
        protected_by_label[first].insert(second);
        protected_by_label[second].insert(first);
    }
    let mut versions = vec![0_usize; count];
    let mut queue = BinaryHeap::<MergeQueueEntry>::new();
    let mut sequence = 0_usize;
    // skimage.graph.RAG inserts nodes and edges while scipy.generic_filter
    // scans the label raster.  Python then copies every adjacency view into a
    // set before queueing proposals.  Equal-score fits are resolved by that
    // insertion/iteration sequence, so label-sorted queueing is not
    // equivalent even though it contains the same pairs.
    let mut rag_node_order = Vec::<usize>::with_capacity(count);
    let mut rag_node_seen = vec![false; count];
    let mut rag_adjacency = vec![Vec::<usize>::new(); count];
    let mut rag_edges = HashSet::<(usize, usize)>::new();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = y * source.width + x;
            let center = segmentation.labels[index] as usize;
            for (dx, dy) in [(0_isize, -1_isize), (-1, 0), (1, 0), (0, 1)] {
                let px =
                    (x as isize + dx).clamp(0, source.width.saturating_sub(1) as isize) as usize;
                let py =
                    (y as isize + dy).clamp(0, source.height.saturating_sub(1) as isize) as usize;
                let other = segmentation.labels[py * source.width + px] as usize;
                if center == other || !rag_edges.insert(pair(center, other)) {
                    continue;
                }
                for node in [center, other] {
                    if !rag_node_seen[node] {
                        rag_node_seen[node] = true;
                        rag_node_order.push(node);
                    }
                }
                rag_adjacency[center].push(other);
                rag_adjacency[other].push(center);
            }
        }
    }
    let mut initial_pairs = Vec::<(usize, usize)>::new();
    for left in rag_node_order {
        for right in python_int_set_order(&rag_adjacency[left]) {
            if left < right {
                initial_pairs.push((left, right));
            }
        }
    }
    let initial_proposals: Vec<Option<MergeProposal>> = initial_pairs
        .par_iter()
        .map(|&(left, right)| {
            merge_candidate_proposal(
                source,
                &regions,
                &protected_by_label,
                &structural_boundaries,
                &structural_by_label,
                left,
                right,
                config,
            )
        })
        .collect();
    for ((left, right), proposal) in initial_pairs.into_iter().zip(initial_proposals) {
        if let Some(proposal) = proposal {
            queue.push(MergeQueueEntry {
                score: proposal.score,
                sequence,
                left,
                right,
                left_version: versions[left],
                right_version: versions[right],
                proposal,
            });
            sequence += 1;
        }
    }
    let mut accepted = 0_usize;
    #[cfg(feature = "diagnostics")]
    let retain_merge_diagnostics = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some();
    #[cfg(feature = "diagnostics")]
    let mut accepted_diagnostics = Vec::<serde_json::Value>::new();
    while let Some(entry) = queue.pop() {
        if entry.score > config.gradient_merge_error {
            break;
        }
        if regions[entry.left].is_none()
            || regions[entry.right].is_none()
            || versions[entry.left] != entry.left_version
            || versions[entry.right] != entry.right_version
        {
            continue;
        }
        let mut first = regions[entry.left].take().unwrap();
        let second = regions[entry.right].take().unwrap();
        #[cfg(feature = "diagnostics")]
        if retain_merge_diagnostics {
            let mut left_labels: Vec<usize> = first.labels.iter().copied().collect();
            let mut right_labels: Vec<usize> = second.labels.iter().copied().collect();
            left_labels.sort_unstable();
            right_labels.sort_unstable();
            let paint_kind = match &entry.proposal.paint {
                Paint::Solid { .. } => "solid",
                Paint::Linear { .. } => "linear",
                Paint::Radial { .. } => "radial",
                Paint::Layered { .. } => "layered",
            };
            accepted_diagnostics.push(serde_json::json!({
                "left": left_labels,
                "right": right_labels,
                "score": entry.score,
                "paint": paint_kind,
            }));
        }
        first.labels.extend(second.labels);
        first.pixels.extend(second.pixels);
        first.samples = entry.proposal.samples.clone();
        first.paint = entry.proposal.paint.clone();
        first.bounds = union_bounds(first.bounds, second.bounds);
        let neighbours: HashSet<usize> = adjacency[entry.left]
            .union(&adjacency[entry.right])
            .copied()
            .filter(|&value| value != entry.left && value != entry.right)
            .collect();
        regions[entry.left] = Some(first);
        versions[entry.left] += 1;
        versions[entry.right] += 1;
        adjacency[entry.left] = neighbours.clone();
        adjacency[entry.right].clear();
        let mut ordered_neighbours: Vec<usize> = neighbours.iter().copied().collect();
        ordered_neighbours.sort_unstable();
        for neighbour in &ordered_neighbours {
            adjacency[*neighbour].remove(&entry.left);
            adjacency[*neighbour].remove(&entry.right);
            adjacency[*neighbour].insert(entry.left);
        }
        for neighbour in ordered_neighbours {
            push_merge_candidate(
                source,
                &regions,
                &versions,
                &protected_by_label,
                &structural_boundaries,
                &structural_by_label,
                entry.left,
                neighbour,
                config,
                &mut sequence,
                &mut queue,
            );
        }
        accepted += 1;
    }
    if accepted == 0 {
        return 0;
    }
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        if let Ok(value) = serde_json::to_string_pretty(&accepted_diagnostics) {
            let _ = std::fs::write(format!("{prefix}-merge-accepted.json"), value);
        }
    }
    let mut labels = vec![0_u32; segmentation.labels.len()];
    for (owner, region) in regions.into_iter().flatten().enumerate() {
        for index in region.pixels {
            labels[index] = owner as u32;
        }
    }
    replace_merged_labels(source, segmentation, labels, accepted);
    accepted
}

#[derive(Clone, Debug)]
struct SmoothPaintBoundary {
    left: usize,
    right: usize,
    points: Vec<Point>,
    length: usize,
    median_delta_e: f32,
    percentile_delta_e: f32,
    gradient_sample_fraction: f32,
    median_gradient_discontinuity: f32,
    percentile_gradient_discontinuity: f32,
}

/// Relative change of the colour slope across four consecutive samples.
///
/// A quantizer boundary in smooth shading can have a sizeable one-pixel
/// DeltaE while the Lab slope on both sides still predicts that change.  A
/// material edge has the opposite signature: most of the colour change is
/// concentrated in the middle step, so the neighbouring slopes disagree.
fn lab_gradient_discontinuity(previous: Lab, first: Lab, second: Lab, following: Lab) -> f32 {
    let vector = |left: Lab, right: Lab| [right.l - left.l, right.a - left.a, right.b - left.b];
    let norm =
        |value: [f32; 3]| (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    let difference = |left: [f32; 3], right: [f32; 3]| {
        norm([left[0] - right[0], left[1] - right[1], left[2] - right[2]])
    };
    let left = vector(previous, first);
    let centre = vector(first, second);
    let right = vector(second, following);
    let scale = norm(centre).max(0.5 * (norm(left) + norm(right))).max(1e-4);
    0.5 * (difference(left, centre) + difference(centre, right)) / scale
}

fn boundary_gradient_discontinuity(
    labs: &[Lab],
    segmentation: &Segmentation,
    first: usize,
    second: usize,
) -> Option<f32> {
    let width = segmentation.width;
    let first_label = segmentation.labels[first];
    let second_label = segmentation.labels[second];
    let (previous, following) = if second == first + 1 && first / width == second / width {
        let x = first % width;
        if x == 0 || x + 2 >= width {
            return None;
        }
        (first - 1, second + 1)
    } else if second == first + width {
        let y = first / width;
        if y == 0 || y + 2 >= segmentation.height {
            return None;
        }
        (first - width, second + width)
    } else {
        return None;
    };
    if segmentation.labels[previous] != first_label
        || segmentation.labels[following] != second_label
    {
        return None;
    }
    Some(lab_gradient_discontinuity(
        labs[previous],
        labs[first],
        labs[second],
        labs[following],
    ))
}

fn boundary_has_quantized_shading(boundary: &SmoothPaintBoundary) -> bool {
    boundary.gradient_sample_fraction >= 1.0 / 3.0
        && boundary.length >= 32
        && boundary.median_delta_e <= 0.75
        && boundary.percentile_delta_e <= 8.0
        && boundary.median_gradient_discontinuity <= 1.0
        && boundary.percentile_gradient_discontinuity <= 1.5
}

fn boundary_has_continuous_gradient(boundary: &SmoothPaintBoundary) -> bool {
    let consistent_slope = boundary.gradient_sample_fraction >= 1.0 / 3.0
        && boundary.median_delta_e <= 12.0
        && boundary.percentile_delta_e <= 18.0
        && boundary.median_gradient_discontinuity <= 0.80
        && boundary.percentile_gradient_discontinuity <= 0.95;
    // A long quantizer contour can cross a small ridge or junction.  Its p90
    // slope statistic then looks discontinuous even though the typical source
    // step is below one JND.  Retain the median evidence so broad highlights
    // do not remain as nested flat-colour bands.
    consistent_slope || boundary_has_quantized_shading(boundary)
}

fn boundary_has_low_source_delta(boundary: &SmoothPaintBoundary) -> bool {
    boundary.median_delta_e <= 1.5 && boundary.percentile_delta_e <= 3.0
}

fn boundary_is_smooth(boundary: &SmoothPaintBoundary) -> bool {
    boundary_has_low_source_delta(boundary) || boundary_has_continuous_gradient(boundary)
}

fn paint_at_point(paint: &Paint, point: Point) -> [f32; 3] {
    match paint {
        Paint::Solid { color } => *color,
        Paint::Linear {
            start, end, stops, ..
        } => {
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let parameter = ((point.x - start.x) * dx + (point.y - start.y) * dy)
                / (dx * dx + dy * dy).max(1e-8);
            interpolate(stops, parameter.clamp(0.0, 1.0))
        }
        Paint::Radial {
            center,
            radius,
            stops,
            ..
        } => {
            let parameter = (((point.x - center.x) / radius.x.max(1e-6)).powi(2)
                + ((point.y - center.y) / radius.y.max(1e-6)).powi(2))
            .sqrt()
            .clamp(0.0, 1.0);
            interpolate(stops, parameter)
        }
        Paint::Layered { base, overlays } => {
            overlays
                .iter()
                .fold(paint_at_point(base, point), |under, overlay| {
                    let over = paint_at_point(&overlay.paint, point);
                    let parameter = match overlay.paint.as_ref() {
                        Paint::Linear { start, end, .. } => {
                            let dx = end.x - start.x;
                            let dy = end.y - start.y;
                            ((point.x - start.x) * dx + (point.y - start.y) * dy)
                                / (dx * dx + dy * dy).max(1e-8)
                        }
                        Paint::Radial { center, radius, .. } => {
                            (((point.x - center.x) / radius.x.max(1e-6)).powi(2)
                                + ((point.y - center.y) / radius.y.max(1e-6)).powi(2))
                            .sqrt()
                        }
                        Paint::Solid { .. } | Paint::Layered { .. } => 0.0,
                    };
                    let alpha = interpolate_opacity(&overlay.opacity_stops, parameter);
                    [0, 1, 2].map(|channel| under[channel] * (1.0 - alpha) + over[channel] * alpha)
                })
        }
    }
}

fn errors_for_indices(
    source_labs: &[Lab],
    samples: &[usize],
    width: usize,
    paint: &Paint,
) -> Vec<f32> {
    let references: Vec<Lab> = samples.iter().map(|&index| source_labs[index]).collect();
    let rendered = Raster::new(
        samples.len(),
        1,
        samples
            .iter()
            .map(|&index| paint_at(paint, index, width))
            .collect(),
    );
    delta_e2000_pairs(&references, &lab_pixels(&rendered))
}

fn seam_errors_at_points(first: &Paint, second: &Paint, points: &[Point]) -> Vec<f32> {
    let first_rgb: Vec<[f32; 3]> = points
        .iter()
        .map(|&point| paint_at_point(first, point))
        .collect();
    let second_rgb: Vec<[f32; 3]> = points
        .iter()
        .map(|&point| paint_at_point(second, point))
        .collect();
    let first_lab = lab_pixels(&Raster::new(points.len(), 1, first_rgb));
    let second_lab = lab_pixels(&Raster::new(points.len(), 1, second_rgb));
    delta_e2000_pairs(&first_lab, &second_lab)
}

fn smooth_paint_boundaries(
    boundary_source: &Raster,
    segmentation: &Segmentation,
    minimum_length: usize,
    include_non_smooth: bool,
) -> Vec<SmoothPaintBoundary> {
    let labs = lab_pixels(boundary_source);
    let mut pairs =
        std::collections::BTreeMap::<(usize, usize), Vec<(usize, Point, f32, Option<f32>)>>::new();
    for y in 0..segmentation.height {
        for x in 0..segmentation.width.saturating_sub(1) {
            let first_index = y * segmentation.width + x;
            let second_index = first_index + 1;
            let first = segmentation.labels[first_index] as usize;
            let second = segmentation.labels[second_index] as usize;
            if first == second {
                continue;
            }
            pairs.entry(pair(first, second)).or_default().push((
                first_index,
                Point {
                    x: x as f32 + 0.5,
                    y: y as f32,
                },
                delta_e2000(labs[first_index], labs[second_index]),
                boundary_gradient_discontinuity(&labs, segmentation, first_index, second_index),
            ));
        }
    }
    for y in 0..segmentation.height.saturating_sub(1) {
        for x in 0..segmentation.width {
            let first_index = y * segmentation.width + x;
            let second_index = first_index + segmentation.width;
            let first = segmentation.labels[first_index] as usize;
            let second = segmentation.labels[second_index] as usize;
            if first == second {
                continue;
            }
            pairs.entry(pair(first, second)).or_default().push((
                first_index,
                Point {
                    x: x as f32,
                    y: y as f32 + 0.5,
                },
                delta_e2000(labs[first_index], labs[second_index]),
                boundary_gradient_discontinuity(&labs, segmentation, first_index, second_index),
            ));
        }
    }

    let mut result = Vec::<SmoothPaintBoundary>::new();
    for ((left, right), edges) in pairs {
        let mut by_cell = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (edge, &(cell, _, _, _)) in edges.iter().enumerate() {
            by_cell.entry(cell).or_default().push(edge);
        }
        let cells: HashSet<usize> = by_cell.keys().copied().collect();
        let mut seen = HashSet::<usize>::new();
        for &start in by_cell.keys() {
            if !seen.insert(start) {
                continue;
            }
            let mut queue = std::collections::VecDeque::from([start]);
            let mut component = HashSet::<usize>::from([start]);
            while let Some(cell) = queue.pop_front() {
                let x = cell % segmentation.width;
                let y = cell / segmentation.width;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let px = x as isize + dx;
                        let py = y as isize + dy;
                        if px < 0
                            || py < 0
                            || px >= segmentation.width as isize
                            || py >= segmentation.height as isize
                        {
                            continue;
                        }
                        let neighbour = py as usize * segmentation.width + px as usize;
                        if cells.contains(&neighbour) && seen.insert(neighbour) {
                            component.insert(neighbour);
                            queue.push_back(neighbour);
                        }
                    }
                }
            }
            let selected: Vec<usize> = edges
                .iter()
                .enumerate()
                .filter_map(|(index, &(cell, _, _, _))| component.contains(&cell).then_some(index))
                .collect();
            if selected.len() < minimum_length.max(1) {
                continue;
            }
            let errors: Vec<f32> = selected.iter().map(|&index| edges[index].2).collect();
            let boundary_median = median(&mut errors.clone());
            let boundary_p90 = percentile(errors, 0.90);
            let gradient_discontinuities: Vec<f32> = selected
                .iter()
                .filter_map(|&index| edges[index].3)
                .collect();
            let gradient_sample_fraction =
                gradient_discontinuities.len() as f32 / selected.len().max(1) as f32;
            let gradient_median = if gradient_discontinuities.len() >= 4 {
                median(&mut gradient_discontinuities.clone())
            } else {
                f32::INFINITY
            };
            let gradient_p90 = if gradient_discontinuities.len() >= 4 {
                percentile(gradient_discontinuities, 0.90)
            } else {
                f32::INFINITY
            };
            let point_indices = sampled_indices(&selected, 96);
            let boundary = SmoothPaintBoundary {
                left,
                right,
                points: point_indices
                    .into_iter()
                    .map(|index| edges[index].1)
                    .collect(),
                length: selected.len(),
                median_delta_e: boundary_median,
                percentile_delta_e: boundary_p90,
                gradient_sample_fraction,
                median_gradient_discontinuity: gradient_median,
                percentile_gradient_discontinuity: gradient_p90,
            };
            if include_non_smooth || boundary_is_smooth(&boundary) {
                result.push(boundary);
            }
        }
    }
    result
}

fn background_label(segmentation: &Segmentation) -> usize {
    let mut counts = vec![0_usize; segmentation.regions.len()];
    for x in 0..segmentation.width {
        counts[segmentation.labels[x] as usize] += 1;
        counts[segmentation.labels[(segmentation.height - 1) * segmentation.width + x] as usize] +=
            1;
    }
    for y in 1..segmentation.height.saturating_sub(1) {
        counts[segmentation.labels[y * segmentation.width] as usize] += 1;
        counts[segmentation.labels[y * segmentation.width + segmentation.width - 1] as usize] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(label, &count)| (count, std::cmp::Reverse(label)))
        .map(|value| value.0)
        .unwrap_or(0)
}

#[derive(Clone)]
struct HarmonizeProposal {
    score: f32,
    candidate_mean: f32,
    left_owner: usize,
    right_owner: usize,
    paint: Paint,
    boundary_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn harmonize_adjacent_paints(
    source: &Raster,
    source_labs: &[Lab],
    segmentation: &Segmentation,
    region_paint_indices: &[Vec<usize>],
    paint_boundaries: &[SmoothPaintBoundary],
    paints: &mut [Paint],
    errors: &mut [f32],
    config: &Config,
) -> usize {
    let background = background_label(segmentation);
    let boundaries: Vec<&SmoothPaintBoundary> = paint_boundaries
        .iter()
        .filter(|boundary| {
            boundary.length >= 8
                && (boundary_has_low_source_delta(boundary)
                    || boundary_has_quantized_shading(boundary))
                && boundary.left != background
                && boundary.right != background
                && !region_paint_indices[boundary.left].is_empty()
                && !region_paint_indices[boundary.right].is_empty()
        })
        .collect();
    let count = paints.len();
    let label_samples: Vec<Vec<usize>> = region_paint_indices
        .iter()
        .map(|indices| sampled_indices(indices, 1024))
        .collect();
    let mut members: HashMap<usize, HashSet<usize>> = (0..count)
        .map(|label| (label, HashSet::from([label])))
        .collect();
    let mut owner: Vec<usize> = (0..count).collect();
    let mut accepted = 0_usize;
    for _ in 0..3 {
        let mut grouped =
            std::collections::BTreeMap::<(usize, usize), Vec<&SmoothPaintBoundary>>::new();
        for boundary in &boundaries {
            let first = owner[boundary.left];
            let second = owner[boundary.right];
            if first != second {
                grouped
                    .entry(pair(first, second))
                    .or_default()
                    .push(boundary);
            }
        }
        let mut proposals: Vec<HarmonizeProposal> = grouped
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter_map(|((left_owner, right_owner), shared)| {
                let mut union_labels: Vec<usize> = members[&left_owner]
                    .union(&members[&right_owner])
                    .copied()
                    .collect();
                union_labels.sort_unstable();
                if union_labels.len() > 8 {
                    return None;
                }
                let mut union_samples = Vec::<usize>::new();
                for &label in &union_labels {
                    union_samples.extend_from_slice(&label_samples[label]);
                }
                union_samples = sampled_indices(&union_samples, 8192);
                let (candidate, _) = office_gradient_candidate(
                    source,
                    source_labs,
                    &union_samples,
                    bounds(&union_samples, source.width),
                    config.maximum_gradient_stops,
                )?;
                let mut baseline_errors = Vec::<f32>::new();
                for &label in &union_labels {
                    baseline_errors.extend(errors_for_indices(
                        source_labs,
                        &label_samples[label],
                        source.width,
                        &paints[label],
                    ));
                }
                let candidate_errors =
                    errors_for_indices(source_labs, &union_samples, source.width, &candidate);
                let baseline_mean = numpy_sum_f32(&baseline_errors) / baseline_errors.len() as f32;
                let candidate_mean =
                    numpy_sum_f32(&candidate_errors) / candidate_errors.len() as f32;
                let baseline_p90 = percentile(baseline_errors, 0.90);
                let candidate_p90 = percentile(candidate_errors, 0.90);
                let mut seam_errors = Vec::<f32>::new();
                for boundary in &shared {
                    seam_errors.extend(seam_errors_at_points(
                        &paints[boundary.left],
                        &paints[boundary.right],
                        &boundary.points,
                    ));
                }
                let seam_mean = numpy_sum_f32(&seam_errors) / seam_errors.len() as f32;
                let seam_p90 = percentile(seam_errors, 0.90);
                if seam_p90 < 0.75 {
                    return None;
                }
                let mean_regression = candidate_mean - baseline_mean;
                let p90_regression = candidate_p90 - baseline_p90;
                let quantized_shading = shared
                    .iter()
                    .any(|boundary| boundary_has_quantized_shading(boundary));
                let maximum_mean_regression = if quantized_shading {
                    0.42 * config.gradient_merge_error
                } else {
                    (0.05 * 2.3_f32).min(0.10 * seam_mean)
                };
                let maximum_p90_regression = if quantized_shading {
                    0.95 * config.gradient_merge_error
                } else {
                    0.10 * 2.3
                };
                let score = mean_regression - 0.10 * seam_mean;
                if mean_regression > maximum_mean_regression
                    || p90_regression > maximum_p90_regression
                {
                    return None;
                }
                if score > 0.0 {
                    return None;
                }
                Some(HarmonizeProposal {
                    score,
                    candidate_mean,
                    left_owner,
                    right_owner,
                    paint: candidate,
                    boundary_count: shared.len(),
                })
            })
            .collect();
        proposals.sort_by(|left, right| {
            left.score
                .total_cmp(&right.score)
                .then(left.candidate_mean.total_cmp(&right.candidate_mean))
                .then(left.left_owner.cmp(&right.left_owner))
                .then(left.right_owner.cmp(&right.right_owner))
        });
        let mut consumed = HashSet::<usize>::new();
        let mut pass_merges = 0_usize;
        for proposal in proposals {
            if consumed.contains(&proposal.left_owner) || consumed.contains(&proposal.right_owner) {
                continue;
            }
            let keep = proposal.left_owner.min(proposal.right_owner);
            let remove = proposal.left_owner.max(proposal.right_owner);
            let removed = members.remove(&remove).unwrap_or_default();
            let combined = members.entry(keep).or_default();
            combined.extend(removed);
            for &label in combined.iter() {
                owner[label] = keep;
                paints[label] = proposal.paint.clone();
                errors[label] = proposal.candidate_mean;
            }
            consumed.insert(proposal.left_owner);
            consumed.insert(proposal.right_owner);
            pass_merges += 1;
            accepted += 1;
            let _ = proposal.boundary_count;
        }
        if pass_merges == 0 {
            break;
        }
    }
    accepted
}

#[allow(dead_code)]
fn couple_linear(
    source: &Raster,
    segmentation: &Segmentation,
    region_indices: &[Vec<usize>],
    region_paint_indices: &[Vec<usize>],
    paints: &mut [Paint],
    errors: &mut [f32],
    config: &Config,
) -> usize {
    let count = paints.len();
    let mut pairs = HashSet::<(u32, u32)>::new();
    for y in 0..segmentation.height {
        for x in 0..segmentation.width {
            let index = y * segmentation.width + x;
            for neighbour in [
                (x + 1 < segmentation.width).then_some(index + 1),
                (y + 1 < segmentation.height).then_some(index + segmentation.width),
            ]
            .into_iter()
            .flatten()
            {
                let a = segmentation.labels[index];
                let b = segmentation.labels[neighbour];
                if a != b {
                    pairs.insert(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
    }
    let mut union = UnionFind::new(count);
    for (a, b) in pairs {
        let (
            Paint::Linear {
                preset: pa,
                start: start_a,
                end: end_a,
                stops: sa,
                ..
            },
            Paint::Linear {
                preset: pb,
                start: start_b,
                end: end_b,
                stops: sb,
                ..
            },
        ) = (&paints[a as usize], &paints[b as usize])
        else {
            continue;
        };
        if pa != pb {
            continue;
        }
        if *pa == LinearPreset::Fitted {
            let first = canonical_direction((end_a.x - start_a.x, end_a.y - start_a.y));
            let second = canonical_direction((end_b.x - start_b.x, end_b.y - start_b.y));
            let aligned = first
                .zip(second)
                .map(|(left, right)| (left.0 * right.0 + left.1 * right.1).abs() >= 0.965)
                .unwrap_or(false);
            if !aligned {
                continue;
            }
        }
        let endpoint_distance = delta_e76(
            rgb_to_lab(sa.last().unwrap().color.map(|value| value as f32)),
            rgb_to_lab(sb.first().unwrap().color.map(|value| value as f32)),
        )
        .min(delta_e76(
            rgb_to_lab(sa.first().unwrap().color.map(|value| value as f32)),
            rgb_to_lab(sb.last().unwrap().color.map(|value| value as f32)),
        ));
        if endpoint_distance <= config.gradient_merge_error * 3.0 {
            union.union(a as usize, b as usize);
        }
    }
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for region in 0..count {
        groups.entry(union.find(region)).or_default().push(region);
    }
    let mut coupled = 0;
    for members in groups.values().filter(|members| members.len() > 1) {
        let preset = match paints[members[0]] {
            Paint::Linear { preset, .. } => preset,
            _ => continue,
        };
        let mut indices = Vec::new();
        let mut paint_indices = Vec::new();
        for &member in members {
            indices.extend_from_slice(&region_indices[member]);
            paint_indices.extend_from_slice(&region_paint_indices[member]);
        }
        let samples = sampled_indices(
            if paint_indices.is_empty() {
                &indices
            } else {
                &paint_indices
            },
            8192,
        );
        let region_bounds = bounds(&indices, source.width);
        let (start, end) = linear_geometry(preset, region_bounds);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| linear_parameter(index, source.width, start, end).clamp(0.0, 1.0))
            .collect();
        let stop_count = config.maximum_gradient_stops.clamp(2, 5);
        let offsets: Vec<f64> = (0..stop_count)
            .map(|index| index as f64 / (stop_count - 1) as f64)
            .collect();
        let stops = fitted_stops(source, &samples, &parameters, &offsets);
        let combined_error = gradient_error(source, &samples, &parameters, &stops).mean;
        let old_error = members
            .iter()
            .map(|&member| errors[member] * region_indices[member].len() as f32)
            .sum::<f32>()
            / indices.len().max(1) as f32;
        if preset == LinearPreset::Fitted {
            let mut directions = fitted_linear_directions(source, &samples);
            for &member in members {
                if let Paint::Linear { start, end, .. } = paints[member] {
                    if let Some(direction) = canonical_direction((end.x - start.x, end.y - start.y))
                    {
                        directions.push(direction);
                    }
                }
            }
            directions.extend([
                (1.0, 0.0),
                (0.0, 1.0),
                (
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ),
                (
                    std::f32::consts::FRAC_1_SQRT_2,
                    -std::f32::consts::FRAC_1_SQRT_2,
                ),
            ]);
            let mut unique = Vec::<(f32, f32)>::new();
            for direction in directions {
                if unique.iter().all(|current| {
                    (current.0 * direction.0 + current.1 * direction.1).abs() < 0.999
                }) {
                    unique.push(direction);
                }
            }
            let mut best: Option<(f32, CoupledPaintAssignments)> = None;
            for direction in unique {
                let mut assignments = Vec::<(usize, Paint, f32)>::new();
                let mut weighted_error = 0.0_f32;
                let mut weight = 0_usize;
                for &member in members {
                    let member_source = if region_paint_indices[member].is_empty() {
                        &region_indices[member]
                    } else {
                        &region_paint_indices[member]
                    };
                    let member_samples = sampled_indices(member_source, 4096);
                    let (start, end) =
                        fitted_linear_geometry(&member_samples, source.width, direction);
                    let parameters: Vec<f32> = member_samples
                        .iter()
                        .map(|&index| {
                            linear_parameter(index, source.width, start, end).clamp(0.0, 1.0)
                        })
                        .collect();
                    let stops = fitted_stops(source, &member_samples, &parameters, &offsets);
                    let error = gradient_error(source, &member_samples, &parameters, &stops).mean;
                    assignments.push((
                        member,
                        Paint::Linear {
                            preset: LinearPreset::Fitted,
                            start,
                            end,
                            stops,
                        },
                        error,
                    ));
                    weighted_error += error * region_indices[member].len() as f32;
                    weight += region_indices[member].len();
                }
                let score = weighted_error / weight.max(1) as f32;
                if best
                    .as_ref()
                    .map(|current| score < current.0)
                    .unwrap_or(true)
                {
                    best = Some((score, assignments));
                }
            }
            if let Some((score, assignments)) = best {
                if score <= old_error + 1e-4 {
                    for (member, paint, error) in assignments {
                        paints[member] = paint;
                        errors[member] = error;
                        coupled += 1;
                    }
                }
            }
            continue;
        }
        if combined_error <= old_error + 1e-4 {
            let paint = Paint::Linear {
                preset,
                start,
                end,
                stops,
            };
            for &member in members {
                paints[member] = paint.clone();
                errors[member] = combined_error;
                coupled += 1;
            }
        }
    }
    coupled
}

#[derive(Clone, Debug)]
struct CouplingBoundary {
    boundary: SmoothPaintBoundary,
    seam_p90: f32,
    same_paint_key: bool,
}

/// Rank the interfaces that compete for a pairwise continuity solve.
///
/// A long, source-supported ramp interface contributes many more visible
/// false-edge pixels than a short interface with the same rendered seam.
/// The fourth-root weight is deliberately weak: it breaks that scheduling
/// tie without letting a long, low-error boundary eclipse a short severe one.
/// Flat/quantized interfaces retain the former seam-only order because their
/// source evidence does not identify a continuous material ramp.
fn coupling_boundary_priority(boundary: &CouplingBoundary) -> f32 {
    let length_weight = if boundary_has_continuous_gradient(&boundary.boundary)
        && !boundary_has_low_source_delta(&boundary.boundary)
    {
        (boundary.boundary.length as f32).sqrt().sqrt()
    } else {
        1.0
    };
    boundary.seam_p90 * length_weight
}

fn coupling_regression_limits(
    has_continuous_gradient_boundary: bool,
    has_patch_boundary: bool,
    config: &Config,
) -> (f32, f32) {
    let mut maximum_mean = if has_continuous_gradient_boundary {
        0.35
    } else if has_patch_boundary {
        0.25
    } else {
        0.15
    } * config.gradient_merge_error;
    let mut maximum_p90 = if has_continuous_gradient_boundary {
        0.95
    } else if has_patch_boundary {
        0.75
    } else {
        0.50
    } * config.gradient_merge_error;
    // Native faces are independently meaningful Paint owners.  Seam removal
    // must not buy continuity by visibly degrading either interior.  Patch
    // faces are different slices of one owner and retain their wider gate.
    if !has_patch_boundary {
        maximum_mean = maximum_mean.min(0.05);
        maximum_p90 = maximum_p90.min(0.20);
    }
    (maximum_mean, maximum_p90)
}

const FIXED_STOP_OFFSETS: [f32; 5] = [0.0, 0.25, 0.50, 0.75, 1.0];

fn five_stop_basis(parameter: f32, offsets: &[f32; 5]) -> [f64; 5] {
    let parameter = parameter.clamp(0.0, 1.0);
    let left = offsets
        .partition_point(|&offset| offset <= parameter)
        .saturating_sub(1)
        .min(3);
    let span = (offsets[left + 1] - offsets[left]).max(1e-6);
    let alpha = ((parameter - offsets[left]) / span).clamp(0.0, 1.0) as f64;
    let mut basis = [0.0_f64; 5];
    basis[left] = 1.0 - alpha;
    basis[left + 1] = alpha;
    basis
}

fn coupled_parameter(paint: &Paint, index: usize, width: usize) -> f32 {
    match paint {
        Paint::Linear { start, end, .. } => linear_parameter(index, width, *start, *end),
        Paint::Radial { center, radius, .. } => radial_parameter(index, width, *center, *radius),
        Paint::Solid { .. } | Paint::Layered { .. } => 0.5,
    }
    .clamp(0.0, 1.0)
}

fn coupled_parameter_point(paint: &Paint, point: Point) -> f32 {
    match paint {
        Paint::Linear { start, end, .. } => {
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy).max(1e-8)
        }
        Paint::Radial { center, radius, .. } => (((point.x - center.x) / radius.x.max(1e-6))
            .powi(2)
            + ((point.y - center.y) / radius.y.max(1e-6)).powi(2))
        .sqrt(),
        Paint::Solid { .. } | Paint::Layered { .. } => 0.5,
    }
    .clamp(0.0, 1.0)
}

fn bilinear_sample(image: &Raster, point: Point) -> [f32; 3] {
    let x = point.x.clamp(0.0, image.width.saturating_sub(1) as f32);
    let y = point.y.clamp(0.0, image.height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let ax = x - x0 as f32;
    let ay = y - y0 as f32;
    [0, 1, 2].map(|channel| {
        let upper = image.pixels[y0 * image.width + x0][channel] * (1.0 - ax)
            + image.pixels[y0 * image.width + x1][channel] * ax;
        let lower = image.pixels[y1 * image.width + x0][channel] * (1.0 - ax)
            + image.pixels[y1 * image.width + x1][channel] * ax;
        upper * (1.0 - ay) + lower * ay
    })
}

fn patch_boundary_halo(boundary: &SmoothPaintBoundary, segmentation: &Segmentation) -> Vec<Point> {
    let mut result = Vec::<Point>::new();
    for &point in &boundary.points {
        for (dx, dy) in [
            (0.0_f32, 0.0_f32),
            (-2.0, 0.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, -2.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (0.0, 2.0),
        ] {
            let candidate = Point {
                x: (point.x + dx).clamp(0.0, segmentation.width.saturating_sub(1) as f32),
                y: (point.y + dy).clamp(0.0, segmentation.height.saturating_sub(1) as f32),
            };
            let x = candidate.x.round_ties_even() as usize;
            let y = candidate.y.round_ties_even() as usize;
            let label = segmentation.labels[y * segmentation.width + x] as usize;
            if label == boundary.left || label == boundary.right {
                result.push(candidate);
            }
        }
    }
    result
}

fn extended_linear_geometry(
    samples: &[usize],
    width: usize,
    direction: (f32, f32),
) -> (Point, Point) {
    let divisor = samples.len().max(1) as f32;
    let center = Point {
        x: samples
            .iter()
            .map(|&index| (index % width) as f32)
            .sum::<f32>()
            / divisor,
        y: samples
            .iter()
            .map(|&index| (index / width) as f32)
            .sum::<f32>()
            / divisor,
    };
    let (low, high) =
        samples
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &index| {
                let projection = ((index % width) as f32 - center.x) * direction.0
                    + ((index / width) as f32 - center.y) * direction.1;
                (low.min(projection), high.max(projection))
            });
    let span = (high - low).max(1.0);
    let margin = 2.0_f32.max(0.10 * span);
    (
        Point {
            x: center.x + direction.0 * (low - margin),
            y: center.y + direction.1 * (low - margin),
        },
        Point {
            x: center.x + direction.0 * (high + margin),
            y: center.y + direction.1 * (high + margin),
        },
    )
}

fn coupled_candidate_directions(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
) -> Vec<(f32, f32)> {
    let divisor = samples.len().max(1) as f32;
    let center_x = samples
        .iter()
        .map(|&index| (index % source.width) as f32)
        .sum::<f32>()
        / divisor;
    let center_y = samples
        .iter()
        .map(|&index| (index / source.width) as f32)
        .sum::<f32>()
        / divisor;
    let mean_l = samples
        .iter()
        .map(|&index| source_labs[index].l)
        .sum::<f32>()
        / divisor;
    let (mut xx, mut xy, mut yy, mut xl, mut yl) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    for &index in samples {
        let dx = (index % source.width) as f32 - center_x;
        let dy = (index / source.width) as f32 - center_y;
        let dl = source_labs[index].l - mean_l;
        xx += (dx * dx) as f64;
        xy += (dx * dy) as f64;
        yy += (dy * dy) as f64;
        xl += (dx * dl) as f64;
        yl += (dy * dl) as f64;
    }
    let mut directions = Vec::<(f32, f32)>::new();
    let determinant = xx * yy - xy * xy;
    if determinant.abs() > 1e-6 {
        let plane = (
            (yy * xl - xy * yl) / determinant,
            (xx * yl - xy * xl) / determinant,
        );
        let length = (plane.0 * plane.0 + plane.1 * plane.1).sqrt();
        if length > 1e-6 {
            directions.push(((plane.0 / length) as f32, (plane.1 / length) as f32));
        }
    }
    if samples.len() >= 3 {
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        let large = (angle.cos() as f32, angle.sin() as f32);
        directions.push((-large.1, large.0));
        directions.push(large);
    }
    directions.extend([
        (1.0, 0.0),
        (0.0, 1.0),
        (
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        (
            std::f32::consts::FRAC_1_SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
        ),
    ]);
    let mut unique = Vec::<(f32, f32)>::new();
    let mut keys = HashSet::<(i32, i32)>::new();
    for direction in directions {
        let Some(direction) = canonical_direction(direction) else {
            continue;
        };
        let key = (
            (direction.0 * 1000.0).round_ties_even() as i32,
            (direction.1 * 1000.0).round_ties_even() as i32,
        );
        if keys.insert(key) {
            unique.push(direction);
        }
    }
    unique
}

fn fit_coupled_geometry(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    geometry: &Paint,
) -> (Paint, ErrorStats) {
    let mut normal = vec![vec![0.0_f64; 5]; 5];
    let mut rhs = vec![vec![0.0_f64; 5]; 3];
    for &sample in samples {
        let basis = five_stop_basis(
            coupled_parameter(geometry, sample, source.width),
            &FIXED_STOP_OFFSETS,
        );
        for first in 0..5 {
            for second in 0..5 {
                normal[first][second] += basis[first] * basis[second];
            }
            for (channel, target) in rhs.iter_mut().enumerate() {
                target[first] += basis[first] * source.pixels[sample][channel] as f64;
            }
        }
    }
    for row in 0..3 {
        let difference = [1.0_f64, -2.0, 1.0];
        for first in 0..3 {
            for second in 0..3 {
                normal[row + first][row + second] += 0.05 * difference[first] * difference[second];
            }
        }
    }
    let solved: Vec<Vec<f64>> = rhs
        .into_iter()
        .map(|target| solve_system(normal.clone(), target))
        .collect();
    let colors: Vec<[f32; 3]> = (0..5)
        .map(|stop| {
            [
                solved[0][stop].clamp(0.0, 1.0) as f32,
                solved[1][stop].clamp(0.0, 1.0) as f32,
                solved[2][stop].clamp(0.0, 1.0) as f32,
            ]
        })
        .collect();
    let paint = paint_with_coupled_stops(geometry, &colors, &FIXED_STOP_OFFSETS);
    let stats = paint_stats_against_labs(source_labs, samples, source.width, &paint);
    (paint, stats)
}

fn choose_coupled_geometry(
    source: &Raster,
    source_labs: &[Lab],
    samples: &[usize],
    current: &Paint,
) -> Paint {
    let mut geometries = Vec::<Paint>::new();
    if !matches!(current, Paint::Solid { .. }) {
        geometries.push(current.clone());
    }
    for direction in coupled_candidate_directions(source, source_labs, samples) {
        let (start, end) = extended_linear_geometry(samples, source.width, direction);
        geometries.push(Paint::Linear {
            preset: LinearPreset::Fitted,
            start,
            end,
            stops: Vec::new(),
        });
    }
    geometries
        .into_iter()
        .map(|geometry| fit_coupled_geometry(source, source_labs, samples, &geometry))
        .min_by(|left, right| objective(left.1).total_cmp(&objective(right.1)))
        .map(|value| value.0)
        .unwrap_or_else(|| current.clone())
}

fn paint_with_coupled_stops(template: &Paint, colours: &[[f32; 3]], offsets: &[f32; 5]) -> Paint {
    let stops: Vec<ColorStop> = colours
        .iter()
        .zip(offsets)
        .map(|(&color, &offset)| ColorStop {
            offset: offset as f64,
            color: color.map(f64::from),
        })
        .collect();
    match template {
        Paint::Linear {
            preset, start, end, ..
        } => Paint::Linear {
            preset: *preset,
            start: *start,
            end: *end,
            stops,
        },
        Paint::Radial {
            center,
            radius,
            origin,
            ..
        } => Paint::Radial {
            center: *center,
            radius: *radius,
            origin: *origin,
            stops,
        },
        Paint::Solid { .. } | Paint::Layered { .. } => {
            unreachable!("non-gradient Paint must be promoted before coupling")
        }
    }
}

/// Keep the five-stop Office limit while placing knots where an incident
/// smooth interface needs a steep, continuous transition.  Uniform 25%
/// knots unnecessarily spread a narrow highlight across a whole face and
/// force the continuity solver to trade a visible seam for a broad blur.
fn coupling_stop_offsets(
    member: usize,
    geometry: &Paint,
    current: &Paint,
    boundaries: &[&CouplingBoundary],
) -> [f32; 5] {
    let mut boundary_offsets = Vec::<(f32, f32)>::new();
    for boundary in boundaries
        .iter()
        .filter(|boundary| boundary.boundary.left == member || boundary.boundary.right == member)
    {
        let mut parameters: Vec<f32> = boundary
            .boundary
            .points
            .iter()
            .map(|&point| coupled_parameter_point(geometry, point))
            .collect();
        if !parameters.is_empty() {
            boundary_offsets.push((boundary.seam_p90, median(&mut parameters)));
        }
    }
    boundary_offsets.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut selected = vec![0.0_f32, 1.0];
    {
        let mut insert = |offset: f32| {
            let offset = offset.clamp(0.0, 1.0);
            if selected.len() < 5
                && offset > 0.015
                && offset < 0.985
                && selected
                    .iter()
                    .all(|&existing| (existing - offset).abs() >= 0.015)
            {
                selected.push(offset);
            }
        };
        for (_, offset) in boundary_offsets {
            insert(offset);
        }
        match current {
            Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => {
                for stop in stops {
                    insert(stop.offset as f32);
                }
            }
            Paint::Solid { .. } | Paint::Layered { .. } => {}
        }
        for offset in FIXED_STOP_OFFSETS {
            insert(offset);
        }
    }
    while selected.len() < 5 {
        selected.sort_by(f32::total_cmp);
        let (index, _) = selected
            .windows(2)
            .enumerate()
            .max_by(|left, right| (left.1[1] - left.1[0]).total_cmp(&(right.1[1] - right.1[0])))
            .unwrap();
        selected.push(0.5 * (selected[index] + selected[index + 1]));
    }
    selected.sort_by(f32::total_cmp);
    selected.try_into().unwrap()
}

fn weighted_percentile(mut values: Vec<(f32, f32)>, quantile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|first, second| first.0.total_cmp(&second.0));
    let target = quantile.clamp(0.0, 1.0) * values.iter().map(|value| value.1).sum::<f32>();
    let mut cumulative = 0.0_f32;
    for (value, weight) in values {
        cumulative += weight;
        if cumulative >= target {
            return value;
        }
    }
    0.0
}

#[allow(clippy::too_many_arguments)]
fn couple_adjacent_paints(
    source: &Raster,
    source_labs: &[Lab],
    segmentation: &Segmentation,
    region_indices: &[Vec<usize>],
    region_paint_indices: &[Vec<usize>],
    paint_boundaries: &[SmoothPaintBoundary],
    paints: &mut [Paint],
    errors: &mut [f32],
    config: &Config,
) -> usize {
    let background = background_label(segmentation);
    let data_samples: Vec<Vec<usize>> = region_paint_indices
        .iter()
        .map(|indices| sampled_indices(indices, 768))
        .collect();
    let mut candidates = Vec::<CouplingBoundary>::new();
    for boundary in paint_boundaries.iter().cloned() {
        let left = boundary.left;
        let right = boundary.right;
        if left == background
            || right == background
            || region_paint_indices[left].is_empty()
            || region_paint_indices[right].is_empty()
        {
            continue;
        }
        let same_paint_key = segmentation
            .paint_keys
            .get(left)
            .zip(segmentation.paint_keys.get(right))
            .map(|(first, second)| first == second)
            .unwrap_or(false);
        if boundary.length < 8 && !same_paint_key {
            continue;
        }
        if !same_paint_key && !boundary_is_smooth(&boundary) {
            continue;
        }
        let seam_deltas = seam_errors_at_points(&paints[left], &paints[right], &boundary.points);
        let seam_p90 = percentile(seam_deltas, 0.90);
        if seam_p90 >= 0.75 || same_paint_key {
            candidates.push(CouplingBoundary {
                boundary,
                seam_p90,
                same_paint_key,
            });
        }
    }
    candidates.sort_by(|first, second| {
        second
            .same_paint_key
            .cmp(&first.same_paint_key)
            .then(coupling_boundary_priority(second).total_cmp(&coupling_boundary_priority(first)))
            .then(second.seam_p90.total_cmp(&first.seam_p90))
    });
    let mut union = UnionFind::new(paints.len());
    let mut sizes = vec![1_usize; paints.len()];
    for boundary in &candidates {
        let left = union.find(boundary.boundary.left);
        let right = union.find(boundary.boundary.right);
        // Artificial patches may share one large solve, but native shading
        // interfaces stay pairwise so a local correction cannot spread
        // through a whole connected material.
        let maximum_group_size = if boundary.same_paint_key { 64 } else { 2 };
        if left == right || sizes[left] + sizes[right] > maximum_group_size {
            continue;
        }
        let combined = sizes[left] + sizes[right];
        union.union(left, right);
        let root = union.find(left);
        sizes[root] = combined;
    }
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for boundary in &candidates {
        for label in [boundary.boundary.left, boundary.boundary.right] {
            groups.entry(union.find(label)).or_default().push(label);
        }
    }
    for members in groups.values_mut() {
        members.sort_unstable();
        members.dedup();
    }
    let mut ordered_groups: Vec<Vec<usize>> = groups.into_values().collect();
    ordered_groups.sort_by_key(|members| members[0]);
    let paint_snapshot = paints.to_vec();
    let group_updates = std::sync::Mutex::new(Vec::<Vec<(usize, Paint, f32)>>::new());
    ordered_groups.par_iter().for_each(|members| {
        if members.len() < 2 {
            return;
        }
        let boundaries: Vec<&CouplingBoundary> = candidates
            .iter()
            .filter(|boundary| {
                members.binary_search(&boundary.boundary.left).is_ok()
                    && members.binary_search(&boundary.boundary.right).is_ok()
            })
            .collect();
        if boundaries.is_empty() {
            return;
        }
        let mut shared_union = UnionFind::new(paint_snapshot.len());
        for boundary in boundaries.iter().filter(|boundary| boundary.same_paint_key) {
            shared_union.union(boundary.boundary.left, boundary.boundary.right);
        }
        let mut patch_groups = HashMap::<usize, Vec<usize>>::new();
        for &member in members {
            let root = shared_union.find(member);
            patch_groups.entry(root).or_default().push(member);
        }
        let mut shared_geometries = HashMap::<usize, Paint>::new();
        for patch_group in patch_groups.values().filter(|group| group.len() > 1) {
            let mut all_samples = Vec::<usize>::new();
            for &member in patch_group {
                all_samples.extend_from_slice(&data_samples[member]);
            }
            let directions = coupled_candidate_directions(source, source_labs, &all_samples);
            let mut best: Option<(f32, Vec<(usize, Paint)>)> = None;
            for direction in directions {
                let mut fitted = Vec::<(usize, Paint)>::new();
                let mut weighted_score = 0.0_f32;
                let mut sample_count = 0_usize;
                for &member in patch_group {
                    let (start, end) =
                        extended_linear_geometry(&data_samples[member], source.width, direction);
                    let geometry = Paint::Linear {
                        preset: LinearPreset::Fitted,
                        start,
                        end,
                        stops: Vec::new(),
                    };
                    let (paint, stats) =
                        fit_coupled_geometry(source, source_labs, &data_samples[member], &geometry);
                    weighted_score += data_samples[member].len() as f32 * objective(stats);
                    sample_count += data_samples[member].len();
                    fitted.push((member, paint));
                }
                let score = weighted_score / sample_count.max(1) as f32;
                if best.as_ref().map(|value| score < value.0).unwrap_or(true) {
                    best = Some((score, fitted));
                }
            }
            if let Some((_, fitted)) = best {
                shared_geometries.extend(fitted);
            }
        }
        let geometries: Vec<Paint> = members
            .par_iter()
            .map(|&member| {
                shared_geometries.get(&member).cloned().unwrap_or_else(|| {
                    choose_coupled_geometry(
                        source,
                        source_labs,
                        &data_samples[member],
                        &paint_snapshot[member],
                    )
                })
            })
            .collect();
        let stop_offsets: Vec<[f32; 5]> = members
            .iter()
            .zip(&geometries)
            .map(|(&member, geometry)| {
                coupling_stop_offsets(member, geometry, &paint_snapshot[member], &boundaries)
            })
            .collect();
        let variable_count = members.len() * 5;
        let mut normal = vec![vec![0.0_f64; variable_count]; variable_count];
        let mut rhs = vec![vec![0.0_f64; variable_count]; 3];
        for (position, &member) in members.iter().enumerate() {
            let samples = &data_samples[member];
            let data_weight =
                (region_indices[member].len() as f64 / samples.len().max(1) as f64).max(1.0);
            for &sample in samples {
                let basis = five_stop_basis(
                    coupled_parameter(&geometries[position], sample, source.width),
                    &stop_offsets[position],
                );
                let block = position * 5;
                for first in 0..5 {
                    for second in 0..5 {
                        normal[block + first][block + second] +=
                            data_weight * basis[first] * basis[second];
                    }
                    for (channel, channel_rhs) in rhs.iter_mut().enumerate().take(3) {
                        channel_rhs[block + first] +=
                            data_weight * basis[first] * source.pixels[sample][channel] as f64;
                    }
                }
            }
        }
        for boundary in &boundaries {
            let left_position = members.binary_search(&boundary.boundary.left).unwrap();
            let right_position = members.binary_search(&boundary.boundary.right).unwrap();
            let constraint_points = if boundary.same_paint_key {
                patch_boundary_halo(&boundary.boundary, segmentation)
            } else {
                boundary.boundary.points.clone()
            };
            if constraint_points.is_empty() {
                continue;
            }
            let boundary_weight_scale = (16.0 / constraint_points.len().max(1) as f64).min(1.0);
            let continuity_scale = if boundary.same_paint_key {
                16.0
            } else if boundary_has_continuous_gradient(&boundary.boundary) {
                2.0
            } else {
                1.0
            };
            let target_scale = if boundary.same_paint_key { 16.0 } else { 1.0 };
            let continuity_weight = 64.0 * continuity_scale * boundary_weight_scale;
            let target_weight = 3.0 * target_scale * boundary_weight_scale;
            for point in constraint_points {
                let left_basis = five_stop_basis(
                    coupled_parameter_point(&geometries[left_position], point),
                    &stop_offsets[left_position],
                );
                let right_basis = five_stop_basis(
                    coupled_parameter_point(&geometries[right_position], point),
                    &stop_offsets[right_position],
                );
                let left_block = left_position * 5;
                let right_block = right_position * 5;
                for first in 0..5 {
                    for second in 0..5 {
                        normal[left_block + first][left_block + second] +=
                            continuity_weight * left_basis[first] * left_basis[second]
                                + target_weight * left_basis[first] * left_basis[second];
                        normal[right_block + first][right_block + second] +=
                            continuity_weight * right_basis[first] * right_basis[second]
                                + target_weight * right_basis[first] * right_basis[second];
                        let cross = continuity_weight * left_basis[first] * right_basis[second];
                        normal[left_block + first][right_block + second] -= cross;
                        normal[right_block + second][left_block + first] -= cross;
                    }
                    let target = bilinear_sample(source, point);
                    for (channel, channel_rhs) in rhs.iter_mut().enumerate().take(3) {
                        channel_rhs[left_block + first] +=
                            target_weight * left_basis[first] * target[channel] as f64;
                        channel_rhs[right_block + first] +=
                            target_weight * right_basis[first] * target[channel] as f64;
                    }
                }
            }
        }
        for position in 0..members.len() {
            let block = position * 5;
            for row in 0..3 {
                let difference = [1.0_f64, -2.0, 1.0];
                for first in 0..3 {
                    for second in 0..3 {
                        normal[block + row + first][block + row + second] +=
                            0.05 * difference[first] * difference[second];
                    }
                }
            }
        }
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += 1e-6;
        }
        let solved: Vec<Vec<f64>> = rhs
            .into_par_iter()
            .map(|channel| solve_system(normal.clone(), channel))
            .collect();
        let proposed: Vec<Paint> = geometries
            .iter()
            .enumerate()
            .map(|(position, geometry)| {
                let colours: Vec<[f32; 3]> = (0..5)
                    .map(|stop| {
                        [
                            solved[0][position * 5 + stop].clamp(0.0, 1.0) as f32,
                            solved[1][position * 5 + stop].clamp(0.0, 1.0) as f32,
                            solved[2][position * 5 + stop].clamp(0.0, 1.0) as f32,
                        ]
                    })
                    .collect();
                paint_with_coupled_stops(geometry, &colours, &stop_offsets[position])
            })
            .collect();
        let mut baseline_errors = Vec::<(f32, f32)>::new();
        let mut proposed_errors = Vec::<(f32, f32)>::new();
        for (position, &member) in members.iter().enumerate() {
            let samples = &data_samples[member];
            let weight = region_indices[member].len() as f32 / samples.len().max(1) as f32;
            let baseline =
                errors_for_indices(source_labs, samples, source.width, &paint_snapshot[member]);
            let candidate =
                errors_for_indices(source_labs, samples, source.width, &proposed[position]);
            for value in baseline {
                baseline_errors.push((value, weight));
            }
            for value in candidate {
                proposed_errors.push((value, weight));
            }
        }
        let baseline_mean = baseline_errors
            .iter()
            .map(|value| value.0 * value.1)
            .sum::<f32>()
            / baseline_errors
                .iter()
                .map(|value| value.1)
                .sum::<f32>()
                .max(1e-8);
        let proposed_mean = proposed_errors
            .iter()
            .map(|value| value.0 * value.1)
            .sum::<f32>()
            / proposed_errors
                .iter()
                .map(|value| value.1)
                .sum::<f32>()
                .max(1e-8);
        let mean_regression = proposed_mean - baseline_mean;
        let p90_regression = weighted_percentile(proposed_errors.clone(), 0.90)
            - weighted_percentile(baseline_errors.clone(), 0.90);
        let mut before_seams = Vec::new();
        let mut after_seams = Vec::new();
        for boundary in boundaries
            .iter()
            .filter(|boundary| boundary.seam_p90 >= 0.75)
        {
            let left_position = members.binary_search(&boundary.boundary.left).unwrap();
            let right_position = members.binary_search(&boundary.boundary.right).unwrap();
            before_seams.extend(seam_errors_at_points(
                &paint_snapshot[boundary.boundary.left],
                &paint_snapshot[boundary.boundary.right],
                &boundary.boundary.points,
            ));
            after_seams.extend(seam_errors_at_points(
                &proposed[left_position],
                &proposed[right_position],
                &boundary.boundary.points,
            ));
        }
        if before_seams.is_empty() {
            return;
        }
        let before_mean = before_seams.iter().sum::<f32>() / before_seams.len().max(1) as f32;
        let after_mean = after_seams.iter().sum::<f32>() / after_seams.len().max(1) as f32;
        let before_p90 = percentile(before_seams, 0.90);
        let after_p90 = percentile(after_seams, 0.90);
        let score = mean_regression + 0.35 * (after_mean - before_mean);
        let has_patch_boundary = boundaries.iter().any(|boundary| boundary.same_paint_key);
        let has_continuous_gradient_boundary = boundaries
            .iter()
            .any(|boundary| boundary_has_continuous_gradient(&boundary.boundary));
        // Keep artificial patch slices on their wider reconstruction gate,
        // while independently meaningful native faces use the sub-JND limits
        // above. The pairwise solve bounds the spatial reach in either case.
        let (maximum_mean_regression, maximum_p90_regression) = coupling_regression_limits(
            has_continuous_gradient_boundary,
            has_patch_boundary,
            config,
        );
        let accepted = mean_regression <= maximum_mean_regression
            && p90_regression <= maximum_p90_regression
            && before_p90 - after_p90 >= 0.50
            && after_p90 <= 2.3_f32.max(0.80 * before_p90)
            && score <= 0.0;
        if accepted {
            let updates = members
                .iter()
                .enumerate()
                .map(|(position, &member)| {
                    let paint = proposed[position].clone();
                    let error = paint_stats_against_labs(
                        source_labs,
                        &data_samples[member],
                        source.width,
                        &paint,
                    )
                    .mean;
                    (member, paint, error)
                })
                .collect();
            group_updates.lock().unwrap().push(updates);
        }
    });
    let mut group_updates = group_updates.into_inner().unwrap();
    group_updates.sort_by_key(|updates| updates.first().map(|value| value.0).unwrap_or(usize::MAX));
    let accepted = group_updates.iter().map(Vec::len).sum();
    for updates in group_updates {
        for (member, paint, error) in updates {
            paints[member] = paint;
            errors[member] = error;
        }
    }
    accepted
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SupportedPaintMergeReport {
    pub merges: usize,
    pub boundary_edges_removed: usize,
}

/// Remove a final Paint interface only when both its geometry and colour are
/// unsupported by the native source.
///
/// This is intentionally downstream of Paint fitting.  Quantizer labels are
/// topology owners, so mere colour similarity is insufficient: every
/// connected interface must be below a native-source JND, the two emitted
/// Paints must already agree at that interface, and a single Office-compatible
/// candidate must preserve the measured error on both incident faces.  The
/// accepted labels are compacted before shared geometry is rebuilt, which
/// removes the obsolete master curve instead of hiding it with an overlay.
pub(crate) fn merge_source_supported_paints(
    source: &Raster,
    boundary_source: &Raster,
    segmentation: &mut Segmentation,
    paints: &mut Vec<Paint>,
    config: &Config,
) -> SupportedPaintMergeReport {
    let count = segmentation.regions.len();
    if count < 2
        || paints.len() != count
        || source.width != segmentation.width
        || source.height != segmentation.height
        || boundary_source.width != segmentation.width
        || boundary_source.height != segmentation.height
    {
        return SupportedPaintMergeReport::default();
    }

    #[derive(Default)]
    struct BoundaryEvidence {
        length: usize,
        maximum_median: f32,
        maximum_p90: f32,
        smooth_contacts: usize,
        rough_contacts: usize,
        points: Vec<Point>,
    }

    let mut evidence = HashMap::<(usize, usize), BoundaryEvidence>::new();
    for boundary in smooth_paint_boundaries(boundary_source, segmentation, 2, true) {
        let entry = evidence
            .entry(pair(boundary.left, boundary.right))
            .or_default();
        entry.length += boundary.length;
        entry.maximum_median = entry.maximum_median.max(boundary.median_delta_e);
        entry.maximum_p90 = entry.maximum_p90.max(boundary.percentile_delta_e);
        if boundary_is_smooth(&boundary) {
            entry.smooth_contacts += 1;
        } else {
            entry.rough_contacts += 1;
        }
        entry.points.extend(boundary.points);
    }
    let mut candidates = evidence.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.1
            .maximum_p90
            .total_cmp(&right.1.maximum_p90)
            .then_with(|| left.1.maximum_median.total_cmp(&right.1.maximum_median))
            .then_with(|| right.1.length.cmp(&left.1.length))
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut region_pixels = vec![Vec::<usize>::new(); count];
    let mut region_samples = vec![Vec::<usize>::new(); count];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        let label = label as usize;
        region_pixels[label].push(index);
        if segmentation.paint_samples[index] {
            region_samples[label].push(index);
        }
    }
    for label in 0..count {
        if region_samples[label].is_empty() {
            region_samples[label] = region_pixels[label].clone();
        }
        region_samples[label] = sampled_indices(&region_samples[label], 768);
    }
    let source_labs = lab_pixels(source);
    let mut used = vec![false; count];
    let mut accepted = Vec::<(usize, usize, Paint, usize)>::new();
    let mut considered = 0_usize;
    let mut rejected_evidence = 0_usize;
    let mut rejected_seam = 0_usize;
    let mut layered_selected = 0_usize;
    let mut rejected_nonfinite = 0_usize;
    let mut rejected_face = 0_usize;
    let mut rejected_combined = 0_usize;
    for ((left, right), boundary) in candidates {
        considered += 1;
        // All disconnected contacts of the label pair must lack a material
        // transition.  One supported contact makes the labels independent.
        if used[left]
            || used[right]
            || paints[left] == paints[right]
            || boundary.smooth_contacts == 0
            || boundary.rough_contacts > 0
            || boundary.maximum_median > 8.0
            || boundary.maximum_p90 > 12.0
            || boundary.points.is_empty()
        {
            rejected_evidence += 1;
            continue;
        }
        let seam_errors = seam_errors_at_points(&paints[left], &paints[right], &boundary.points);
        if percentile(seam_errors, 0.90) > 8.0 {
            rejected_seam += 1;
            continue;
        }
        let left_region = MergeRegion {
            labels: HashSet::from([left]),
            pixels: region_pixels[left].clone(),
            samples: region_samples[left].clone(),
            bounds: bounds(&region_pixels[left], source.width),
            paint: paints[left].clone(),
        };
        let right_region = MergeRegion {
            labels: HashSet::from([right]),
            pixels: region_pixels[right].clone(),
            samples: region_samples[right].clone(),
            bounds: bounds(&region_pixels[right], source.width),
            paint: paints[right].clone(),
        };
        let mut proposal = merge_proposal(source, &left_region, &right_region, config);
        let combined_bounds = union_bounds(left_region.bounds, right_region.bounds);
        let layered_samples = balanced_samples(
            &region_samples[left],
            &region_samples[right],
            region_pixels[left].len(),
            region_pixels[right].len(),
            256,
        );
        let (layered, layered_stats) = fit_layered_residual_paint(
            source,
            &layered_samples,
            combined_bounds,
            proposal.paint.clone(),
            1,
        );
        let layered_score = objective(layered_stats)
            .max(objective(paint_stats(
                source,
                &region_samples[left],
                &layered,
            )))
            .max(objective(paint_stats(
                source,
                &region_samples[right],
                &layered,
            )));
        if layered_score < proposal.score {
            proposal.paint = layered;
            proposal.score = layered_score;
            layered_selected += usize::from(matches!(proposal.paint, Paint::Layered { .. }));
        }
        if !proposal.score.is_finite() {
            rejected_nonfinite += 1;
            continue;
        }
        let left_baseline = paint_stats_against_labs(
            &source_labs,
            &region_samples[left],
            source.width,
            &paints[left],
        );
        let right_baseline = paint_stats_against_labs(
            &source_labs,
            &region_samples[right],
            source.width,
            &paints[right],
        );
        let left_candidate = paint_stats_against_labs(
            &source_labs,
            &region_samples[left],
            source.width,
            &proposal.paint,
        );
        let right_candidate = paint_stats_against_labs(
            &source_labs,
            &region_samples[right],
            source.width,
            &proposal.paint,
        );
        if left_candidate.mean > left_baseline.mean + 0.30
            || right_candidate.mean > right_baseline.mean + 0.30
            || left_candidate.percentile > left_baseline.percentile + 0.75
            || right_candidate.percentile > right_baseline.percentile + 0.75
        {
            rejected_face += 1;
            continue;
        }
        let mut baseline_errors = errors_for_indices(
            &source_labs,
            &region_samples[left],
            source.width,
            &paints[left],
        );
        baseline_errors.extend(errors_for_indices(
            &source_labs,
            &region_samples[right],
            source.width,
            &paints[right],
        ));
        let mut candidate_errors = errors_for_indices(
            &source_labs,
            &region_samples[left],
            source.width,
            &proposal.paint,
        );
        candidate_errors.extend(errors_for_indices(
            &source_labs,
            &region_samples[right],
            source.width,
            &proposal.paint,
        ));
        let baseline_mean = numpy_sum_f32(&baseline_errors) / baseline_errors.len().max(1) as f32;
        let candidate_mean =
            numpy_sum_f32(&candidate_errors) / candidate_errors.len().max(1) as f32;
        if candidate_mean > baseline_mean + 0.01
            || percentile(candidate_errors, 0.90) > percentile(baseline_errors, 0.90) + 0.04
        {
            rejected_combined += 1;
            continue;
        }
        used[left] = true;
        used[right] = true;
        accepted.push((left, right, proposal.paint, boundary.length));
    }
    if accepted.is_empty() {
        if cfg!(feature = "diagnostics") && config.retain_diagnostics {
            eprintln!(
                "picvec layered merge candidates: considered {considered}, evidence {rejected_evidence}, seam {rejected_seam}, layered {layered_selected}, nonfinite {rejected_nonfinite}, face {rejected_face}, combined {rejected_combined}, accepted 0"
            );
        }
        return SupportedPaintMergeReport::default();
    }

    if cfg!(feature = "diagnostics") && config.retain_diagnostics {
        eprintln!(
            "picvec layered merge candidates: considered {considered}, evidence {rejected_evidence}, seam {rejected_seam}, layered {layered_selected}, nonfinite {rejected_nonfinite}, face {rejected_face}, combined {rejected_combined}, accepted {}",
            accepted.len(),
        );
    }

    let mut owners = UnionFind::new(count);
    for &(left, right, _, _) in &accepted {
        owners.union(left, right);
    }
    let roots = (0..count)
        .map(|label| owners.find(label))
        .collect::<Vec<_>>();
    let mut replacement = HashMap::<usize, Paint>::new();
    for (left, _, paint, _) in &accepted {
        replacement.insert(roots[*left], paint.clone());
    }
    let mut unique = roots.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut representative = vec![usize::MAX; count];
    for (label, &root) in roots.iter().enumerate() {
        representative[root] = representative[root].min(label);
    }
    let merged_paints = unique
        .iter()
        .map(|&root| {
            replacement
                .get(&root)
                .cloned()
                .unwrap_or_else(|| paints[representative[root]].clone())
        })
        .collect::<Vec<_>>();
    let labels = segmentation
        .labels
        .iter()
        .map(|&label| roots[label as usize] as u32)
        .collect::<Vec<_>>();
    replace_source_supported_paint_labels(source, segmentation, labels, accepted.len());
    *paints = merged_paints;
    SupportedPaintMergeReport {
        merges: accepted.len(),
        boundary_edges_removed: accepted.iter().map(|value| value.3).sum(),
    }
}

pub fn fit_all(
    source: &Raster,
    boundary_source: &Raster,
    segmentation: &Segmentation,
    topology: &HierarchicalTopology,
    strong_branches: &crate::ridge::StrongRidgeBranches,
    config: &Config,
) -> (Vec<Paint>, GradientSummary) {
    let fit_started = std::time::Instant::now();
    let source_labs = lab_pixels(source);
    let mut region_indices = vec![Vec::<usize>::new(); segmentation.regions.len()];
    let mut region_paint_indices = vec![Vec::<usize>::new(); segmentation.regions.len()];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        region_indices[label as usize].push(index);
        if segmentation.paint_samples[index] {
            region_paint_indices[label as usize].push(index);
        }
    }
    if cfg!(feature = "diagnostics") && config.retain_diagnostics {
        eprintln!(
            "picvec paint substage setup: {:.3}s",
            fit_started.elapsed().as_secs_f64()
        );
    }
    let initial_started = std::time::Instant::now();
    let region_density = segmentation.regions.len() as f32
        / (segmentation.width * segmentation.height).max(1) as f32;
    let use_primary_gate = region_density >= config.paint_primary_min_region_density;
    let fitted: Vec<(Paint, f32, bool)> = region_indices
        .par_iter()
        .zip(region_paint_indices.par_iter())
        .enumerate()
        .map(|(label, (indices, paint_indices))| {
            let strong_dark = indices
                .iter()
                .filter(|&&index| strong_branches.dark[index])
                .count()
                * 2
                >= indices.len();
            let strong_bright = !strong_dark
                && indices
                    .iter()
                    .filter(|&&index| strong_branches.bright[index])
                    .count()
                    * 2
                    >= indices.len();
            let branch_paint_indices: Vec<usize> = if strong_dark {
                paint_indices
                    .iter()
                    .copied()
                    .filter(|&index| strong_branches.dark[index])
                    .collect()
            } else if strong_bright {
                paint_indices
                    .iter()
                    .copied()
                    .filter(|&index| strong_branches.bright[index])
                    .collect()
            } else {
                Vec::new()
            };
            let selected_paint_indices = if strong_dark || strong_bright {
                &branch_paint_indices
            } else {
                paint_indices
            };
            let canonical_solid = indices
                .first()
                .map(|&index| segmentation.canonical.pixels[index])
                .unwrap_or([0.0; 3]);
            fit_region(
                label,
                source,
                &source_labs,
                indices,
                selected_paint_indices,
                canonical_solid,
                strong_dark,
                topology.paint_sample_budget(label, 8192),
                use_primary_gate,
                config,
            )
        })
        .collect();
    let full_fit_regions = fitted.iter().filter(|value| value.2).count();
    if cfg!(feature = "diagnostics") && config.retain_diagnostics {
        eprintln!(
            "picvec paint substage initial: {:.3}s (primary solid {}, full fit {})",
            initial_started.elapsed().as_secs_f64(),
            fitted.len() - full_fit_regions,
            full_fit_regions,
        );
    }
    let mut paints: Vec<Paint> = fitted.iter().map(|value| value.0.clone()).collect();
    let mut errors: Vec<f32> = fitted.iter().map(|value| value.1).collect();
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PAINT_DIAGNOSTICS") {
        save_paint_kinds(&format!("{prefix}-initial.json"), &paints);
        save_paint_details(&format!("{prefix}-initial-details.json"), &paints);
    }
    let paint_boundaries = smooth_paint_boundaries(boundary_source, segmentation, 2, true);
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PAINT_DIAGNOSTICS") {
        let values: Vec<serde_json::Value> = paint_boundaries
            .iter()
            .map(|boundary| {
                serde_json::json!({
                    "labels": [boundary.left, boundary.right],
                    "length": boundary.length,
                    "median_delta_e": boundary.median_delta_e,
                    "p90_delta_e": boundary.percentile_delta_e,
                    "gradient_sample_fraction": boundary.gradient_sample_fraction,
                    "median_gradient_discontinuity": boundary.median_gradient_discontinuity,
                    "p90_gradient_discontinuity": boundary.percentile_gradient_discontinuity,
                    "smooth": boundary_is_smooth(boundary),
                })
            })
            .collect();
        if let Ok(document) = serde_json::to_vec(&values) {
            let _ = std::fs::write(format!("{prefix}-boundaries.json"), document);
        }
    }
    let harmonize_started = std::time::Instant::now();
    let coupled = harmonize_adjacent_paints(
        source,
        &source_labs,
        segmentation,
        &region_paint_indices,
        &paint_boundaries,
        &mut paints,
        &mut errors,
        config,
    );
    if cfg!(feature = "diagnostics") && config.retain_diagnostics {
        eprintln!(
            "picvec paint substage harmonize: {:.3}s",
            harmonize_started.elapsed().as_secs_f64()
        );
    }
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PAINT_DIAGNOSTICS") {
        save_paint_kinds(&format!("{prefix}-harmonized.json"), &paints);
        save_paint_details(&format!("{prefix}-harmonized-details.json"), &paints);
    }
    let coupling_started = std::time::Instant::now();
    let mut locally_coupled = 0_usize;
    // A second disjoint-pair pass lets continuity propagate to the next seam
    // without ever turning a whole shading component into one global fit.
    for _ in 0..2 {
        let updated = couple_adjacent_paints(
            source,
            &source_labs,
            segmentation,
            &region_indices,
            &region_paint_indices,
            &paint_boundaries,
            &mut paints,
            &mut errors,
            config,
        );
        locally_coupled += updated;
        if updated == 0 {
            break;
        }
    }
    if cfg!(feature = "diagnostics") && config.retain_diagnostics {
        eprintln!(
            "picvec paint substage couple: {:.3}s",
            coupling_started.elapsed().as_secs_f64()
        );
    }
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PAINT_DIAGNOSTICS") {
        save_paint_kinds(&format!("{prefix}-coupled.json"), &paints);
        save_paint_details(&format!("{prefix}-coupled-details.json"), &paints);
    }
    let mut summary = GradientSummary {
        coupled_linear_regions: coupled + locally_coupled,
        primary_gate_active: use_primary_gate,
        primary_solid_regions: fitted.len() - full_fit_regions,
        full_fit_regions,
        ..GradientSummary::default()
    };
    refresh_summary(&mut summary, &paints);
    (paints, summary)
}

#[cfg(feature = "diagnostics")]
fn save_paint_kinds(path: &str, paints: &[Paint]) {
    let values: Vec<&str> = paints
        .iter()
        .map(|paint| match paint {
            Paint::Solid { .. } => "solid",
            Paint::Linear { .. } => "linear",
            Paint::Radial { .. } => "radial",
            Paint::Layered { .. } => "layered",
        })
        .collect();
    if let Ok(document) = serde_json::to_vec(&values) {
        let _ = std::fs::write(path, document);
    }
}

#[cfg(feature = "diagnostics")]
fn save_paint_details(path: &str, paints: &[Paint]) {
    let values: Vec<serde_json::Value> = paints
        .iter()
        .map(|paint| match paint {
            Paint::Solid { color } => serde_json::json!({
                "kind": "solid",
                "color": color,
            }),
            Paint::Linear {
                preset,
                start,
                end,
                stops,
            } => serde_json::json!({
                "kind": "linear",
                "preset": format!("{preset:?}"),
                "geometry": [start.x, start.y, end.x, end.y],
                "stops": stops.iter().map(|stop| serde_json::json!([
                    stop.offset, stop.color
                ])).collect::<Vec<_>>(),
            }),
            Paint::Radial {
                origin,
                center,
                radius,
                stops,
            } => serde_json::json!({
                "kind": "radial",
                "origin": format!("{origin:?}"),
                "geometry": [center.x, center.y, radius.x, radius.y],
                "stops": stops.iter().map(|stop| serde_json::json!([
                    stop.offset, stop.color
                ])).collect::<Vec<_>>(),
            }),
            Paint::Layered { base, overlays } => serde_json::json!({
                "kind": "layered",
                "base": format!("{base:?}"),
                "overlays": overlays.iter().map(|overlay| serde_json::json!({
                    "paint": format!("{:?}", overlay.paint),
                    "opacity_stops": overlay.opacity_stops.iter().map(|stop| [
                        stop.offset, stop.opacity
                    ]).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            }),
        })
        .collect();
    if let Ok(document) = serde_json::to_vec(&values) {
        let _ = std::fs::write(path, document);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_face_segmentation(source: &Raster) -> Segmentation {
        let labels = (0..source.height)
            .flat_map(|_| (0..source.width).map(|x| u32::from(x >= source.width / 2)))
            .collect::<Vec<_>>();
        let half = source.width / 2;
        let area = half * source.height;
        Segmentation {
            width: source.width,
            height: source.height,
            labels,
            paint_keys: vec![0, 1],
            paint_samples: vec![true; source.width * source.height],
            canonical: source.clone(),
            regions: vec![
                crate::segment::RegionStats {
                    id: 0,
                    area,
                    min_x: 0,
                    min_y: 0,
                    max_x: half,
                    max_y: source.height,
                    mean_rgb: source.pixels[0],
                    mean_lab: rgb_to_lab(source.pixels[0]),
                },
                crate::segment::RegionStats {
                    id: 1,
                    area,
                    min_x: half,
                    min_y: 0,
                    max_x: source.width,
                    max_y: source.height,
                    mean_rgb: source.pixels[half],
                    mean_lab: rgb_to_lab(source.pixels[half]),
                },
            ],
            summary: crate::segment::SegmentationSummary::default(),
        }
    }

    #[test]
    fn interpolation_hits_endpoints() {
        let stops = vec![
            ColorStop {
                offset: 0.0,
                color: [0.0; 3],
            },
            ColorStop {
                offset: 1.0,
                color: [1.0; 3],
            },
        ];
        assert_eq!(interpolate(&stops, 0.0), [0.0; 3]);
        assert_eq!(interpolate(&stops, 1.0), [1.0; 3]);
    }

    #[test]
    fn primary_gate_separates_coherent_gradient_from_unstructured_texture() {
        let width = 16;
        let height = 16;
        let coherent = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|_| {
                    (0..width).map(|x| {
                        let value = x as f32 / (width - 1) as f32;
                        [value, 0.25 + 0.5 * value, 1.0 - value]
                    })
                })
                .collect(),
        );
        let texture = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let value = if (x + y).is_multiple_of(2) { 0.2 } else { 0.8 };
                        [value, 1.0 - value, value]
                    })
                })
                .collect(),
        );
        let indices = (0..width * height).collect::<Vec<_>>();
        let region_bounds = bounds(&indices, width);
        let coherent_score =
            primary_gradient_coherence(&coherent, &indices, region_bounds, false, 64);
        let texture_score =
            primary_gradient_coherence(&texture, &indices, region_bounds, false, 64);
        assert!(coherent_score > 0.95, "{coherent_score}");
        assert!(texture_score < 0.06, "{texture_score}");
    }

    #[test]
    fn office_fit_uses_the_source_angle_for_a_rotated_linear_ramp() {
        let width = 31;
        let height = 29;
        let source_direction = (0.26_f32, 0.97_f32);
        let maximum_projection =
            (width - 1) as f32 * source_direction.0 + (height - 1) as f32 * source_direction.1;
        let source = Raster::new(
            width,
            height,
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let parameter = (x as f32 * source_direction.0
                            + y as f32 * source_direction.1)
                            / maximum_projection;
                        [
                            0.18 + 0.52 * parameter,
                            0.32 + 0.36 * parameter,
                            0.78 - 0.31 * parameter,
                        ]
                    })
                })
                .collect(),
        );
        let samples = (0..width * height).collect::<Vec<_>>();
        let source_labs = lab_pixels(&source);
        let (paint, _) =
            office_gradient_candidate(&source, &source_labs, &samples, bounds(&samples, width), 5)
                .expect("a coherent ramp must produce a gradient candidate");
        let Paint::Linear {
            preset: LinearPreset::Fitted,
            start,
            end,
            ..
        } = paint
        else {
            panic!("a non-cardinal ramp must retain its continuously fitted angle");
        };
        let fitted = (end.x - start.x, end.y - start.y);
        let fitted_length = fitted.0.hypot(fitted.1);
        let source_length = source_direction.0.hypot(source_direction.1);
        let alignment = (fitted.0 * source_direction.0 + fitted.1 * source_direction.1).abs()
            / (fitted_length * source_length);
        assert!(alignment > 0.995, "angle alignment was {alignment}");
    }

    #[test]
    fn gradient_discontinuity_separates_a_ramp_from_a_step() {
        let lab = |lightness| Lab {
            l: lightness,
            a: 12.0,
            b: -7.0,
        };
        assert!(lab_gradient_discontinuity(lab(20.0), lab(25.0), lab(30.0), lab(35.0)) < 1e-6);
        assert!(lab_gradient_discontinuity(lab(20.0), lab(20.0), lab(40.0), lab(40.0)) > 0.99);
    }

    #[test]
    fn smooth_boundary_detection_accepts_a_steep_continuous_ramp() {
        let pixels = (0..8)
            .flat_map(|_| {
                (0..12).map(|x| {
                    let value = 0.15 + 0.70 * x as f32 / 11.0;
                    [value; 3]
                })
            })
            .collect();
        let source = Raster::new(12, 8, pixels);
        let segmentation = two_face_segmentation(&source);
        let boundaries = smooth_paint_boundaries(&source, &segmentation, 8, false);
        assert_eq!(boundaries.len(), 1);
        assert!(boundaries[0].median_delta_e > 3.0);
        assert!(boundary_has_continuous_gradient(&boundaries[0]));
    }

    #[test]
    fn smooth_boundary_detection_rejects_a_material_step() {
        let mut pixels = vec![[0.2; 3]; 12 * 8];
        for y in 0..8 {
            for x in 6..12 {
                pixels[y * 12 + x] = [0.8; 3];
            }
        }
        let source = Raster::new(12, 8, pixels);
        let segmentation = two_face_segmentation(&source);
        assert!(smooth_paint_boundaries(&source, &segmentation, 8, false).is_empty());
    }

    fn coupling_boundary(length: usize, median_delta_e: f32) -> CouplingBoundary {
        CouplingBoundary {
            boundary: SmoothPaintBoundary {
                left: 0,
                right: 1,
                points: Vec::new(),
                length,
                median_delta_e,
                percentile_delta_e: median_delta_e + 1.0,
                gradient_sample_fraction: 1.0,
                median_gradient_discontinuity: 0.2,
                percentile_gradient_discontinuity: 0.3,
            },
            seam_p90: 4.0,
            same_paint_key: false,
        }
    }

    #[test]
    fn continuous_ramp_priority_accounts_for_visible_boundary_length() {
        let short = coupling_boundary(16, 6.0);
        let long = coupling_boundary(256, 6.0);
        assert!(coupling_boundary_priority(&long) > coupling_boundary_priority(&short));

        let flat_short = coupling_boundary(16, 0.5);
        let flat_long = coupling_boundary(256, 0.5);
        assert_eq!(
            coupling_boundary_priority(&flat_short),
            coupling_boundary_priority(&flat_long)
        );
    }

    #[test]
    fn native_continuity_gate_keeps_interior_colour_regression_sub_jnd() {
        let config = Config::default();
        let native = coupling_regression_limits(true, false, &config);
        assert_eq!(native, (0.05, 0.20));

        let patch = coupling_regression_limits(true, true, &config);
        assert!(patch.0 > native.0);
        assert!(patch.1 > native.1);
    }

    #[test]
    fn source_supported_merge_removes_only_an_unsupported_paint_interface() {
        let source = Raster::blank(8, 8, [0.5; 3]);
        let mut segmentation = two_face_segmentation(&source);
        let mut paints = vec![
            Paint::Solid { color: [0.498; 3] },
            Paint::Solid { color: [0.502; 3] },
        ];
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config::default(),
        );
        assert_eq!(report.merges, 1);
        assert_eq!(report.boundary_edges_removed, 8);
        assert_eq!(segmentation.regions.len(), 1);
        assert_eq!(paints, vec![Paint::Solid { color: [0.5; 3] }]);
    }

    #[test]
    fn source_supported_merge_preserves_a_native_material_transition() {
        let mut pixels = vec![[0.2; 3]; 8 * 8];
        for y in 0..8 {
            for x in 4..8 {
                pixels[y * 8 + x] = [0.8; 3];
            }
        }
        let source = Raster::new(8, 8, pixels);
        let mut segmentation = two_face_segmentation(&source);
        let mut paints = vec![
            Paint::Solid { color: [0.2; 3] },
            Paint::Solid { color: [0.8; 3] },
        ];
        let report = merge_source_supported_paints(
            &source,
            &source,
            &mut segmentation,
            &mut paints,
            &Config::default(),
        );
        assert_eq!(report, SupportedPaintMergeReport::default());
        assert_eq!(segmentation.regions.len(), 2);
        assert_eq!(paints.len(), 2);
    }

    #[test]
    fn layered_paint_fades_without_an_internal_boundary() {
        let paint = Paint::Layered {
            base: Box::new(Paint::Solid { color: [0.0; 3] }),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Radial {
                    origin: RadialOrigin::Fitted,
                    center: Point { x: 1.0, y: 0.0 },
                    radius: Point { x: 1.0, y: 1.0 },
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: [1.0, 0.0, 0.0],
                        },
                        ColorStop {
                            offset: 1.0,
                            color: [1.0, 0.0, 0.0],
                        },
                    ],
                }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 1.0,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        };
        assert_eq!(paint_at(&paint, 1, 3), [1.0, 0.0, 0.0]);
        assert_eq!(paint_at(&paint, 0, 3), [0.0; 3]);
    }

    #[test]
    fn small_region_gradient_requires_strict_measured_gain() {
        let selected = ErrorStats {
            mean: 10.0,
            percentile: 6.0,
        };
        let minimum_improvement = 0.25 * 2.3;
        assert!(gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 7.5,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 8.0,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 7.5,
                percentile: 6.1,
            },
            true,
            minimum_improvement,
            60,
            64,
        ));
        assert!(!gradient_gain_is_sufficient(
            selected,
            ErrorStats {
                mean: 3.0,
                percentile: 6.0,
            },
            true,
            minimum_improvement,
            16,
            64,
        ));
    }

    #[test]
    fn normal_region_uses_one_gain_gate_for_every_gradient_model() {
        let solid = ErrorStats {
            mean: 2.4,
            percentile: 3.8,
        };
        let gradient = ErrorStats {
            mean: 1.9,
            percentile: 3.3,
        };
        assert!(gradient_gain_is_sufficient(
            solid,
            gradient,
            false,
            0.25 * 2.3,
            14_000,
            64,
        ));
    }
}
