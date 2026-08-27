use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;
use serde::Serialize;

use crate::color::{delta_e2000, delta_e76, rgb_to_lab};
use crate::config::Config;
use crate::edge::{dilate_square, EdgeRoles};
use crate::geometry::Point;
use crate::raster::{percentile, Raster};
use crate::segment::{replace_merged_labels, Segmentation};
use crate::union_find::UnionFind;

#[derive(Clone, Debug, PartialEq)]
pub struct ColorStop {
    pub offset: f32,
    pub color: [f32; 3],
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
}

type CoupledPaintAssignments = Vec<(usize, Paint, f32)>;

#[derive(Clone, Debug, Default, Serialize)]
pub struct GradientSummary {
    pub solid_regions: usize,
    pub linear_regions: usize,
    pub radial_regions: usize,
    pub fitted_direction_linear_regions: usize,
    pub fitted_focus_radial_regions: usize,
    pub coupled_linear_regions: usize,
    pub maximum_stops: usize,
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

fn mean_color(source: &Raster, indices: &[usize]) -> [f32; 3] {
    let mut result = [0.0_f64; 3];
    for &index in indices {
        for (channel, value) in result.iter_mut().enumerate() {
            *value += source.pixels[index][channel] as f64;
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
    let upper = stops
        .iter()
        .position(|stop| stop.offset >= t)
        .unwrap_or(stops.len() - 1);
    if upper == 0 {
        return stops[0].color;
    }
    let first = &stops[upper - 1];
    let second = &stops[upper];
    let amount = ((t - first.offset) / (second.offset - first.offset).max(1e-6)).clamp(0.0, 1.0);
    [
        first.color[0] * (1.0 - amount) + second.color[0] * amount,
        first.color[1] * (1.0 - amount) + second.color[1] * amount,
        first.color[2] * (1.0 - amount) + second.color[2] * amount,
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

fn interpolation_weights(parameter: f32, offsets: &[f32]) -> (usize, usize, f32) {
    let parameter = parameter.clamp(0.0, 1.0);
    let right = offsets
        .iter()
        .position(|&offset| offset >= parameter)
        .unwrap_or(offsets.len() - 1)
        .max(1);
    let left = right - 1;
    let alpha =
        ((parameter - offsets[left]) / (offsets[right] - offsets[left]).max(1e-6)).clamp(0.0, 1.0);
    (left, right, alpha)
}

#[allow(clippy::needless_range_loop)]
fn fitted_stops(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    offsets: &[f32],
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
                colors[0][index].clamp(0.0, 1.0) as f32,
                colors[1][index].clamp(0.0, 1.0) as f32,
                colors[2][index].clamp(0.0, 1.0) as f32,
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
    let errors: Vec<f32> = samples
        .iter()
        .map(|&index| {
            delta_e2000(
                rgb_to_lab(source.pixels[index]),
                rgb_to_lab(predicted(index)),
            )
        })
        .collect();
    ErrorStats {
        mean: errors.iter().sum::<f32>() / errors.len() as f32,
        percentile: percentile(errors, 0.90),
    }
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

fn fitted_linear_directions(source: &Raster, samples: &[usize]) -> Vec<(f32, f32)> {
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
    let lightness: Vec<f32> = samples
        .iter()
        .map(|&index| rgb_to_lab(source.pixels[index]).l)
        .collect();
    let mean_lightness = lightness.iter().sum::<f32>() / divisor;
    let (mut xx, mut xy, mut yy, mut xl, mut yl) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&index, &l) in samples.iter().zip(&lightness) {
        let dx = (index % source.width) as f32 - centre_x;
        let dy = (index / source.width) as f32 - centre_y;
        let dl = l - mean_lightness;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
        xl += dx * dl;
        yl += dy * dl;
    }
    let spatial_angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut result = Vec::<(f32, f32)>::new();
    if let Some(direction) = canonical_direction((spatial_angle.cos(), spatial_angle.sin())) {
        result.push(direction);
    }
    let determinant = xx * yy - xy * xy;
    if determinant.abs() > 1e-6 {
        let plane = (
            (yy * xl - xy * yl) / determinant,
            (xx * yl - xy * xl) / determinant,
        );
        if let Some(direction) = canonical_direction(plane) {
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

fn fitted_radial_geometries(
    source: &Raster,
    samples: &[usize],
    region_bounds: Bounds,
) -> Vec<(Point, Point)> {
    if samples.len() < 3 {
        return Vec::new();
    }
    let lightness: Vec<f32> = samples
        .iter()
        .map(|&index| rgb_to_lab(source.pixels[index]).l)
        .collect();
    let mut ordered = lightness.clone();
    let lower = percentile(ordered.clone(), 0.20);
    let upper = percentile(std::mem::take(&mut ordered), 0.80);
    let radius = Point {
        x: ((region_bounds.max_x - region_bounds.min_x) * 0.5).max(0.5),
        y: ((region_bounds.max_y - region_bounds.min_y) * 0.5).max(0.5),
    };
    [true, false]
        .into_iter()
        .filter_map(|bright| {
            let mut weighted = Point::default();
            let mut total = 0.0_f32;
            for (&index, &value) in samples.iter().zip(&lightness) {
                let weight = if bright {
                    (value - lower).max(0.0) + 1e-3
                } else {
                    (upper - value).max(0.0) + 1e-3
                };
                weighted.x += (index % source.width) as f32 * weight;
                weighted.y += (index / source.width) as f32 * weight;
                total += weight;
            }
            (total > 1e-6).then_some((
                Point {
                    x: weighted.x / total,
                    y: weighted.y / total,
                },
                radius,
            ))
        })
        .collect()
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

fn add_gradient_stops(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    template: &Paint,
    initial_stops: Vec<ColorStop>,
    initial_stats: ErrorStats,
    maximum: usize,
) -> (Paint, ErrorStats) {
    let mut offsets = vec![0.0_f32, 1.0];
    let mut current_stops = initial_stops;
    let mut current_stats = initial_stats;
    let mut accepted_stops = current_stops.clone();
    let mut accepted_stats = current_stats;
    let initial_objective = objective(initial_stats);
    while offsets.len() < maximum.clamp(2, 5) {
        let mut best: Option<(f32, f32, Vec<ColorStop>, ErrorStats)> = None;
        for step in 1..=9 {
            let offset = step as f32 / 10.0;
            if offsets.iter().any(|&value| (value - offset).abs() < 0.075) {
                continue;
            }
            let mut proposed = offsets.clone();
            proposed.push(offset);
            proposed.sort_by(f32::total_cmp);
            let stops = fitted_stops(source, samples, parameters, &proposed);
            let stats = gradient_error(source, samples, parameters, &stops);
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
        offsets.push(chosen);
        offsets.sort_by(f32::total_cmp);
        current_stops = stops;
        current_stats = stats;
        if initial_objective - candidate >= 0.15 {
            accepted_stops = current_stops.clone();
            accepted_stats = current_stats;
        }
    }
    (paint_with_stops(template, accepted_stops), accepted_stats)
}

fn fit_region(
    source: &Raster,
    indices: &[usize],
    paint_indices: &[usize],
    config: &Config,
) -> (Paint, f32) {
    let sample_source = if paint_indices.is_empty() {
        indices
    } else {
        paint_indices
    };
    let samples = sampled_indices(sample_source, 4096);
    fit_region_samples(
        source,
        &samples,
        indices.len(),
        bounds(indices, source.width),
        config,
    )
}

fn fit_region_samples(
    source: &Raster,
    samples: &[usize],
    area: usize,
    region_bounds: Bounds,
    config: &Config,
) -> (Paint, f32) {
    let solid_color = mean_color(source, samples);
    let solid_error = paint_error(source, samples, |_| solid_color);
    if area < config.minimum_gradient_area as usize {
        return (Paint::Solid { color: solid_color }, solid_error.mean);
    }
    let mut candidates = Vec::<(Paint, ErrorStats, Vec<f32>)>::new();
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
        let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
        let error = gradient_error(source, samples, &parameters, &stops);
        candidates.push((
            Paint::Linear {
                preset,
                start,
                end,
                stops,
            },
            error,
            parameters,
        ));
    }
    for direction in fitted_linear_directions(source, samples) {
        let (start, end) = fitted_linear_geometry(samples, source.width, direction);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| linear_parameter(index, source.width, start, end).clamp(0.0, 1.0))
            .collect();
        let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
        let error = gradient_error(source, samples, &parameters, &stops);
        candidates.push((
            Paint::Linear {
                preset: LinearPreset::Fitted,
                start,
                end,
                stops,
            },
            error,
            parameters,
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
        let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
        let error = gradient_error(source, samples, &parameters, &stops);
        candidates.push((
            Paint::Radial {
                origin,
                center,
                radius,
                stops,
            },
            error,
            parameters,
        ));
    }
    for (center, radius) in fitted_radial_geometries(source, samples, region_bounds) {
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| radial_parameter(index, source.width, center, radius))
            .collect();
        let stops = fitted_stops(source, samples, &parameters, &[0.0, 1.0]);
        let error = gradient_error(source, samples, &parameters, &stops);
        candidates.push((
            Paint::Radial {
                origin: RadialOrigin::Fitted,
                center,
                radius,
                stops,
            },
            error,
            parameters,
        ));
    }
    candidates.sort_by(|left, right| objective(left.1).total_cmp(&objective(right.1)));
    let (template, two_stop_stats, parameters) = &candidates[0];
    let initial_stops = match template {
        Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => stops.clone(),
        Paint::Solid { .. } => unreachable!(),
    };
    let (candidate, candidate_stats) = add_gradient_stops(
        source,
        samples,
        parameters,
        template,
        initial_stops,
        *two_stop_stats,
        config.maximum_gradient_stops,
    );
    // The final exporter gives an Office-compatible five-stop candidate one
    // last chance with a 0.05-JND mean gain, provided p90 does not regress.
    // Using the coarser region-merge threshold here incorrectly flattened
    // many car-body highlights back to solid bands.
    let required = 0.05 * 2.3;
    if solid_error.mean - candidate_stats.mean >= required
        && candidate_stats.percentile <= solid_error.percentile + 1e-4
    {
        (candidate, candidate_stats.mean)
    } else {
        (Paint::Solid { color: solid_color }, solid_error.mean)
    }
}

fn fitted_stops_direct(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    offsets: &[f32],
) -> Vec<ColorStop> {
    let count = offsets.len();
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
                colors[0][index].clamp(0.0, 1.0) as f32,
                colors[1][index].clamp(0.0, 1.0) as f32,
                colors[2][index].clamp(0.0, 1.0) as f32,
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
    }
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
    let mut offsets = vec![0.0_f32, 1.0];
    let mut current_stops = initial_stops;
    let mut current_stats = initial_stats;
    let mut accepted_stops = current_stops.clone();
    let mut accepted_stats = current_stats;
    let initial_objective = objective(initial_stats);
    let mut accepted_objective = initial_objective;
    while offsets.len() < maximum.clamp(2, 5) {
        let mut best: Option<(f32, f32, Vec<ColorStop>, ErrorStats)> = None;
        for step in 1..=9 {
            let offset = step as f32 / 10.0;
            if offsets.iter().any(|&value| (value - offset).abs() < 0.075) {
                continue;
            }
            let mut proposed = offsets.clone();
            proposed.push(offset);
            proposed.sort_by(f32::total_cmp);
            let stops = fitted_stops_direct(source, samples, parameters, &proposed);
            let stats = gradient_error(source, samples, parameters, &stops);
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
        offsets.push(chosen);
        offsets.sort_by(f32::total_cmp);
        current_stops = stops;
        current_stats = stats;
        let cumulative_gain = accepted_objective - candidate;
        if initial_objective - candidate >= 0.15 || cumulative_gain >= 0.15 {
            accepted_stops = current_stops.clone();
            accepted_stats = current_stats;
            accepted_objective = candidate;
        }
    }
    (paint_with_stops(template, accepted_stops), accepted_stats)
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
            Paint::Solid { .. } => unreachable!(),
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
    samples
        .iter()
        .map(|&index| {
            let predicted = paint_at(paint, index, source.width);
            (0..3)
                .map(|channel| (source.pixels[index][channel] - predicted[channel]).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / samples.len() as f32
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
        Paint::Solid { .. } => return (solid.clone(), solid_stats),
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
    let quick_samples = balanced_samples(
        &first.samples,
        &second.samples,
        first.pixels.len(),
        second.pixels.len(),
        384,
    );
    let first_samples = sampled_indices(&first.samples, 384);
    let second_samples = sampled_indices(&second.samples, 384);
    let first_mean = mean_color(source, &first_samples);
    let second_mean = mean_color(source, &second_samples);
    let mean_delta = delta_e2000(rgb_to_lab(first_mean), rgb_to_lab(second_mean));
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
    MergeProposal {
        samples: balanced_samples(
            &first.samples,
            &second.samples,
            first.pixels.len(),
            second.pixels.len(),
            1024,
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
            let mut predicted: Vec<f32> = boundary
                .sample_pairs
                .iter()
                .map(|&(first_index, second_index)| {
                    delta_e2000(
                        rgb_to_lab(paint_at(&proposal.paint, first_index, source.width)),
                        rgb_to_lab(paint_at(&proposal.paint, second_index, source.width)),
                    )
                })
                .collect();
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
    let edge_lab: Vec<_> = edge_reference
        .pixels
        .iter()
        .copied()
        .map(rgb_to_lab)
        .collect();
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
    let mut initial_pairs = Vec::<(usize, usize)>::new();
    for (left, neighbours) in adjacency.iter().enumerate() {
        let mut ordered: Vec<usize> = neighbours.iter().copied().collect();
        ordered.sort_unstable();
        for right in ordered {
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
    let retain_merge_diagnostics = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some();
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
        if retain_merge_diagnostics {
            let mut left_labels: Vec<usize> = first.labels.iter().copied().collect();
            let mut right_labels: Vec<usize> = second.labels.iter().copied().collect();
            left_labels.sort_unstable();
            right_labels.sort_unstable();
            let paint_kind = match &entry.proposal.paint {
                Paint::Solid { .. } => "solid",
                Paint::Linear { .. } => "linear",
                Paint::Radial { .. } => "radial",
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
            rgb_to_lab(sa.last().unwrap().color),
            rgb_to_lab(sb.first().unwrap().color),
        )
        .min(delta_e76(
            rgb_to_lab(sa.first().unwrap().color),
            rgb_to_lab(sb.last().unwrap().color),
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
        let offsets: Vec<f32> = (0..stop_count)
            .map(|index| index as f32 / (stop_count - 1) as f32)
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
    left: usize,
    right: usize,
    samples: Vec<(usize, usize)>,
    seam_p90: f32,
}

fn five_stop_basis(parameter: f32) -> [f64; 5] {
    let value = parameter.clamp(0.0, 1.0) * 4.0;
    let left = (value.floor() as usize).min(3);
    let alpha = (value - left as f32) as f64;
    let mut basis = [0.0_f64; 5];
    basis[left] = 1.0 - alpha;
    basis[left + 1] = alpha;
    basis
}

fn coupled_parameter(paint: &Paint, index: usize, width: usize) -> f32 {
    match paint {
        Paint::Linear { start, end, .. } => linear_parameter(index, width, *start, *end),
        Paint::Radial { center, radius, .. } => radial_parameter(index, width, *center, *radius),
        Paint::Solid { .. } => 0.5,
    }
    .clamp(0.0, 1.0)
}

fn promote_solid_geometry(source: &Raster, samples: &[usize]) -> Paint {
    let region_bounds = bounds(samples, source.width);
    let offsets = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    let mut directions = fitted_linear_directions(source, samples);
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
    let mut best: Option<(f32, Paint)> = None;
    for direction in directions {
        let (start, end) = fitted_linear_geometry(samples, source.width, direction);
        let parameters: Vec<f32> = samples
            .iter()
            .map(|&index| linear_parameter(index, source.width, start, end))
            .collect();
        let stops = fitted_stops(source, samples, &parameters, &offsets);
        let paint = Paint::Linear {
            preset: LinearPreset::Fitted,
            start,
            end,
            stops,
        };
        let score = objective(paint_stats(source, samples, &paint));
        if best.as_ref().map(|value| score < value.0).unwrap_or(true) {
            best = Some((score, paint));
        }
    }
    best.map(|value| value.1).unwrap_or_else(|| {
        let (start, end) = linear_geometry(LinearPreset::LeftToRight, region_bounds);
        Paint::Linear {
            preset: LinearPreset::LeftToRight,
            start,
            end,
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: mean_color(source, samples),
                },
                ColorStop {
                    offset: 1.0,
                    color: mean_color(source, samples),
                },
            ],
        }
    })
}

fn paint_with_five_stops(template: &Paint, colours: &[[f32; 3]]) -> Paint {
    let stops: Vec<ColorStop> = colours
        .iter()
        .enumerate()
        .map(|(index, &color)| ColorStop {
            offset: index as f32 / 4.0,
            color,
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
        Paint::Solid { .. } => unreachable!("solid Paint must be promoted before coupling"),
    }
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
    segmentation: &Segmentation,
    region_indices: &[Vec<usize>],
    region_paint_indices: &[Vec<usize>],
    paints: &mut [Paint],
    errors: &mut [f32],
    config: &Config,
) -> usize {
    let mut pair_samples = HashMap::<(usize, usize), Vec<(usize, usize)>>::new();
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
                let first = segmentation.labels[index] as usize;
                let second = segmentation.labels[neighbour] as usize;
                if first == second {
                    continue;
                }
                let key = pair(first, second);
                let oriented = if first <= second {
                    (index, neighbour)
                } else {
                    (neighbour, index)
                };
                pair_samples.entry(key).or_default().push(oriented);
            }
        }
    }
    let mut ordered_pairs: Vec<_> = pair_samples.into_iter().collect();
    ordered_pairs.sort_by_key(|value| value.0);
    let mut candidates = Vec::<CouplingBoundary>::new();
    for ((left, right), samples) in ordered_pairs {
        if samples.len() < 8 {
            continue;
        }
        let sampled = sampled_indices(&(0..samples.len()).collect::<Vec<_>>(), 256);
        let mut source_deltas = Vec::with_capacity(sampled.len());
        let mut seam_deltas = Vec::with_capacity(sampled.len());
        for sample in sampled {
            let (first, second) = samples[sample];
            source_deltas.push(delta_e2000(
                rgb_to_lab(source.pixels[first]),
                rgb_to_lab(source.pixels[second]),
            ));
            let first_colour = paint_at(&paints[left], first, source.width);
            let second_colour = paint_at(&paints[right], first, source.width);
            seam_deltas.push(delta_e2000(
                rgb_to_lab(first_colour),
                rgb_to_lab(second_colour),
            ));
        }
        let mut source_for_median = source_deltas.clone();
        let source_median = median(&mut source_for_median);
        let source_p90 = percentile(source_deltas, 0.90);
        let seam_p90 = percentile(seam_deltas, 0.90);
        if source_median <= 1.5 && source_p90 <= 3.0 && seam_p90 >= 0.75 {
            candidates.push(CouplingBoundary {
                left,
                right,
                samples,
                seam_p90,
            });
        }
    }
    candidates.sort_by(|first, second| {
        second
            .seam_p90
            .total_cmp(&first.seam_p90)
            .then(first.left.cmp(&second.left))
            .then(first.right.cmp(&second.right))
    });
    let mut union = UnionFind::new(paints.len());
    let mut sizes = vec![1_usize; paints.len()];
    for boundary in &candidates {
        let left = union.find(boundary.left);
        let right = union.find(boundary.right);
        if left == right || sizes[left] + sizes[right] > 64 {
            continue;
        }
        let combined = sizes[left] + sizes[right];
        union.union(left, right);
        let root = union.find(left);
        sizes[root] = combined;
    }
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for boundary in &candidates {
        for label in [boundary.left, boundary.right] {
            groups.entry(union.find(label)).or_default().push(label);
        }
    }
    for members in groups.values_mut() {
        members.sort_unstable();
        members.dedup();
    }
    let mut accepted = 0_usize;
    let mut ordered_groups: Vec<Vec<usize>> = groups.into_values().collect();
    ordered_groups.sort_by_key(|members| members[0]);
    for members in ordered_groups {
        if members.len() < 2 {
            continue;
        }
        let boundaries: Vec<&CouplingBoundary> = candidates
            .iter()
            .filter(|boundary| {
                members.binary_search(&boundary.left).is_ok()
                    && members.binary_search(&boundary.right).is_ok()
            })
            .collect();
        if boundaries.is_empty() {
            continue;
        }
        let geometries: Vec<Paint> = members
            .iter()
            .map(|&member| {
                if matches!(paints[member], Paint::Solid { .. }) {
                    let source_indices = if region_paint_indices[member].is_empty() {
                        &region_indices[member]
                    } else {
                        &region_paint_indices[member]
                    };
                    promote_solid_geometry(source, &sampled_indices(source_indices, 768))
                } else {
                    paints[member].clone()
                }
            })
            .collect();
        let variable_count = members.len() * 5;
        let mut normal = vec![vec![0.0_f64; variable_count]; variable_count];
        let mut rhs = vec![vec![0.0_f64; variable_count]; 3];
        for (position, &member) in members.iter().enumerate() {
            let source_indices = if region_paint_indices[member].is_empty() {
                &region_indices[member]
            } else {
                &region_paint_indices[member]
            };
            let samples = sampled_indices(source_indices, 768);
            let data_weight =
                (region_indices[member].len() as f64 / samples.len().max(1) as f64).max(1.0);
            for &sample in &samples {
                let basis = five_stop_basis(coupled_parameter(
                    &geometries[position],
                    sample,
                    source.width,
                ));
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
            let left_position = members.binary_search(&boundary.left).unwrap();
            let right_position = members.binary_search(&boundary.right).unwrap();
            let boundary_weight_scale = (16.0 / boundary.samples.len().max(1) as f64).min(1.0);
            let continuity_weight = 64.0 * boundary_weight_scale;
            let target_weight = 3.0 * boundary_weight_scale;
            for &(left_sample, right_sample) in &boundary.samples {
                let left_basis = five_stop_basis(coupled_parameter(
                    &geometries[left_position],
                    left_sample,
                    source.width,
                ));
                let right_basis = five_stop_basis(coupled_parameter(
                    &geometries[right_position],
                    left_sample,
                    source.width,
                ));
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
                    let target = [
                        0.5 * (source.pixels[left_sample][0] + source.pixels[right_sample][0]),
                        0.5 * (source.pixels[left_sample][1] + source.pixels[right_sample][1]),
                        0.5 * (source.pixels[left_sample][2] + source.pixels[right_sample][2]),
                    ];
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
            .into_iter()
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
                paint_with_five_stops(geometry, &colours)
            })
            .collect();
        let mut baseline_errors = Vec::<(f32, f32)>::new();
        let mut proposed_errors = Vec::<(f32, f32)>::new();
        for (position, &member) in members.iter().enumerate() {
            let source_indices = if region_paint_indices[member].is_empty() {
                &region_indices[member]
            } else {
                &region_paint_indices[member]
            };
            let samples = sampled_indices(source_indices, 768);
            let weight = region_indices[member].len() as f32 / samples.len().max(1) as f32;
            for &sample in &samples {
                let actual = rgb_to_lab(source.pixels[sample]);
                baseline_errors.push((
                    delta_e2000(
                        actual,
                        rgb_to_lab(paint_at(&paints[member], sample, source.width)),
                    ),
                    weight,
                ));
                proposed_errors.push((
                    delta_e2000(
                        actual,
                        rgb_to_lab(paint_at(&proposed[position], sample, source.width)),
                    ),
                    weight,
                ));
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
        for boundary in &boundaries {
            let left_position = members.binary_search(&boundary.left).unwrap();
            let right_position = members.binary_search(&boundary.right).unwrap();
            for &(sample, _) in &boundary.samples {
                before_seams.push(delta_e2000(
                    rgb_to_lab(paint_at(&paints[boundary.left], sample, source.width)),
                    rgb_to_lab(paint_at(&paints[boundary.right], sample, source.width)),
                ));
                after_seams.push(delta_e2000(
                    rgb_to_lab(paint_at(&proposed[left_position], sample, source.width)),
                    rgb_to_lab(paint_at(&proposed[right_position], sample, source.width)),
                ));
            }
        }
        let before_mean = before_seams.iter().sum::<f32>() / before_seams.len().max(1) as f32;
        let after_mean = after_seams.iter().sum::<f32>() / after_seams.len().max(1) as f32;
        let before_p90 = percentile(before_seams, 0.90);
        let after_p90 = percentile(after_seams, 0.90);
        let score = mean_regression + 0.35 * (after_mean - before_mean);
        if mean_regression <= 0.15 * config.gradient_merge_error
            && p90_regression <= 0.50 * config.gradient_merge_error
            && before_p90 - after_p90 >= 0.50
            && after_p90 <= 2.3_f32.max(0.80 * before_p90)
            && score <= 0.0
        {
            for (position, &member) in members.iter().enumerate() {
                paints[member] = proposed[position].clone();
                errors[member] = paint_stats(
                    source,
                    &sampled_indices(&region_indices[member], 768),
                    &paints[member],
                )
                .mean;
                accepted += 1;
            }
        }
    }
    accepted
}

pub fn fit_all(
    source: &Raster,
    segmentation: &Segmentation,
    config: &Config,
) -> (Vec<Paint>, GradientSummary) {
    let mut region_indices = vec![Vec::<usize>::new(); segmentation.regions.len()];
    let mut region_paint_indices = vec![Vec::<usize>::new(); segmentation.regions.len()];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        region_indices[label as usize].push(index);
        if segmentation.paint_samples[index] {
            region_paint_indices[label as usize].push(index);
        }
    }
    let fitted: Vec<(Paint, f32)> = region_indices
        .iter()
        .zip(&region_paint_indices)
        .map(|(indices, paint_indices)| fit_region(source, indices, paint_indices, config))
        .collect();
    let mut paints: Vec<Paint> = fitted.iter().map(|value| value.0.clone()).collect();
    let mut errors: Vec<f32> = fitted.iter().map(|value| value.1).collect();
    let coupled = couple_linear(
        source,
        segmentation,
        &region_indices,
        &region_paint_indices,
        &mut paints,
        &mut errors,
        config,
    );
    let locally_coupled = couple_adjacent_paints(
        source,
        segmentation,
        &region_indices,
        &region_paint_indices,
        &mut paints,
        &mut errors,
        config,
    );
    let mut summary = GradientSummary {
        coupled_linear_regions: coupled + locally_coupled,
        maximum_stops: paints
            .iter()
            .map(|paint| match paint {
                Paint::Solid { .. } => 0,
                Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => stops.len(),
            })
            .max()
            .unwrap_or(0),
        ..GradientSummary::default()
    };
    for paint in &paints {
        match paint {
            Paint::Solid { .. } => summary.solid_regions += 1,
            Paint::Linear { preset, .. } => {
                summary.linear_regions += 1;
                summary.fitted_direction_linear_regions +=
                    usize::from(*preset == LinearPreset::Fitted);
            }
            Paint::Radial { origin, .. } => {
                summary.radial_regions += 1;
                summary.fitted_focus_radial_regions += usize::from(*origin == RadialOrigin::Fitted);
            }
        }
    }
    (paints, summary)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
