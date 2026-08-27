use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};

use crate::color::{delta_e2000, delta_e76, lab_to_rgb, rgb_to_lab, Lab};
use crate::config::Config;
use crate::raster::{percentile, Raster};

#[derive(Clone, Debug, Default, Serialize)]
pub struct EdgeSummary {
    pub skeleton_pixels: usize,
    pub edge_chain_count: usize,
    pub edge_chain_samples: usize,
    pub boundary_pixels: usize,
    pub ridge_role_pixels: usize,
    pub ridge_on_boundary_pixels: usize,
    pub unknown_pixels: usize,
    pub visible_ridge_pixels: usize,
    pub visible_ridge_coverage_pixels: usize,
    pub dark_boundary_pixels: usize,
    pub shading_pixels: usize,
    pub face_barrier_pixels: usize,
    pub visible_ridge_graph_edges: usize,
    pub visible_ridge_graph_edges_before_join: usize,
    pub dark_boundary_graph_edges: usize,
    pub wide_dark_outline_graph_edges_before_join: usize,
    pub dark_boundary_candidates: usize,
    pub dark_ridge_candidates: usize,
    pub dark_ridge_supported: usize,
    pub profile_ridge_candidates: usize,
    pub profile_ridge_extensions: usize,
    pub visible_ridge_graph_length: f32,
    pub visible_ridge_median_width: f32,
    pub visible_ridge_width_weighted_length: f32,
    pub dark_boundary_graph_length: f32,
    pub dark_boundary_median_width: f32,
    pub gradient_threshold: f32,
    pub ridge_threshold: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceEdge {
    /// Source-centre geometry stays in float64, as in NumPy's stroke graph.
    /// The sampled edge field is float32, but truncating the accumulated
    /// sub-pixel coordinates changes strict disk ownership at a few pixels.
    pub points: Vec<[f64; 2]>,
    pub width: f64,
    pub role: &'static str,
    width_samples: Vec<(f64, usize)>,
}

#[derive(Clone, Debug)]
pub struct EdgeRoles {
    pub width: usize,
    pub height: usize,
    pub boundary: Vec<bool>,
    pub visible_ridge_centres: Vec<bool>,
    pub visible_ridge_coverage: Vec<bool>,
    pub dark_boundary: Vec<bool>,
    pub shading: Vec<bool>,
    pub face_barrier: Vec<bool>,
    pub gradient: Vec<f32>,
    pub visible_ridge_graph: Vec<SourceEdge>,
    pub dark_boundary_graph: Vec<SourceEdge>,
    pub summary: EdgeSummary,
}

#[derive(Clone, Copy, Debug, Default)]
struct Measurement {
    gradient: f32,
    ridge_response: f32,
    darkness: f32,
    side_agreement: bool,
    valley_response: f32,
    gx: f32,
    gy: f32,
}

pub fn lab_pixels(image: &Raster) -> Vec<Lab> {
    image.pixels.par_iter().copied().map(rgb_to_lab).collect()
}

#[inline]
fn at(values: &[Lab], width: usize, height: usize, x: isize, y: isize) -> Lab {
    let px = x.clamp(0, width.saturating_sub(1) as isize) as usize;
    let py = y.clamp(0, height.saturating_sub(1) as isize) as usize;
    values[py * width + px]
}

pub fn dilate(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    (0..mask.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r * r {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px >= 0
                        && py >= 0
                        && px < width as isize
                        && py < height as isize
                        && mask[py as usize * width + px as usize]
                    {
                        return true;
                    }
                }
            }
            false
        })
        .collect()
}

/// Match ``scipy.ndimage.binary_dilation`` with an all-ones square
/// structuring element. Several ownership stages in the reference use a
/// square deliberately; the Euclidean-disc helper above is reserved for
/// measured stroke coverage.
pub fn dilate_square(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    (0..mask.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let radius = radius as isize;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px >= 0
                        && py >= 0
                        && px < width as isize
                        && py < height as isize
                        && mask[py as usize * width + px as usize]
                    {
                        return true;
                    }
                }
            }
            false
        })
        .collect()
}

pub fn erode(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    (0..mask.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r * r {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px < 0
                        || py < 0
                        || px >= width as isize
                        || py >= height as isize
                        || !mask[py as usize * width + px as usize]
                    {
                        return false;
                    }
                }
            }
            true
        })
        .collect()
}

fn filter_ridge_components(
    candidates: &[bool],
    measurements: &[Measurement],
    width: usize,
    height: usize,
    minimum_span: usize,
    minimum_elongation: f32,
) -> Vec<bool> {
    let mut result = vec![false; candidates.len()];
    let mut seen = vec![false; candidates.len()];
    for start in 0..candidates.len() {
        if !candidates[start] || seen[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut darkness = 0.0;
        seen[start] = true;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x + 1);
            max_y = max_y.max(y + 1);
            darkness += measurements[index].darkness;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                        continue;
                    }
                    let neighbour = py as usize * width + px as usize;
                    if candidates[neighbour] && !seen[neighbour] {
                        seen[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        let box_width = max_x.saturating_sub(min_x).max(1);
        let box_height = max_y.saturating_sub(min_y).max(1);
        let span = box_width.max(box_height);
        let thickness = box_width.min(box_height);
        let elongation = span as f32 / thickness as f32;
        let fill = component.len() as f32 / (box_width * box_height).max(1) as f32;
        let mean_darkness = darkness / component.len().max(1) as f32;
        let line_like = component.len() >= 4
            && span >= minimum_span
            && mean_darkness >= 3.5
            && ((elongation >= minimum_elongation && fill <= 0.72)
                || (component.len() >= 18
                    && elongation >= (minimum_elongation * 0.8).max(1.75)
                    && fill <= 0.30));
        if line_like {
            for index in component {
                result[index] = true;
            }
        }
    }
    result
}

fn sample_scalar(values: &[Measurement], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let top = values[y0 * width + x0].gradient * (1.0 - tx) + values[y0 * width + x1].gradient * tx;
    let bottom =
        values[y1 * width + x0].gradient * (1.0 - tx) + values[y1 * width + x1].gradient * tx;
    top * (1.0 - ty) + bottom * ty
}

fn classify_legacy(image: &Raster) -> EdgeRoles {
    let width = image.width;
    let height = image.height;
    let lab = lab_pixels(image);
    let measurements: Vec<Measurement> = (0..image.pixels.len())
        .into_par_iter()
        .map(|index| {
            let x = (index % width) as isize;
            let y = (index / width) as isize;
            let gx = -at(&lab, width, height, x - 1, y - 1).l
                + at(&lab, width, height, x + 1, y - 1).l
                - 2.0 * at(&lab, width, height, x - 1, y).l
                + 2.0 * at(&lab, width, height, x + 1, y).l
                - at(&lab, width, height, x - 1, y + 1).l
                + at(&lab, width, height, x + 1, y + 1).l;
            let gy = -at(&lab, width, height, x - 1, y - 1).l
                - 2.0 * at(&lab, width, height, x, y - 1).l
                - at(&lab, width, height, x + 1, y - 1).l
                + at(&lab, width, height, x - 1, y + 1).l
                + 2.0 * at(&lab, width, height, x, y + 1).l
                + at(&lab, width, height, x + 1, y + 1).l;
            let gradient = (gx * gx + gy * gy).sqrt() / 8.0;
            let centre = lab[index];
            let mut ridge_response = 0.0_f32;
            let mut valley_response = 0.0_f32;
            let mut side_lightness = centre.l;
            let mut side_agreement = false;
            for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                let first = at(&lab, width, height, x + dx, y + dy);
                let second = at(&lab, width, height, x - dx, y - dy);
                let d1 = delta_e76(centre, first);
                let d2 = delta_e76(centre, second);
                let sides = delta_e76(first, second);
                let response = (d1.min(d2) - 0.45 * sides).max(0.0);
                if first.l >= centre.l + 3.0 && second.l >= centre.l + 3.0 {
                    valley_response = valley_response.max(d1.min(d2));
                }
                if response > ridge_response {
                    ridge_response = response;
                    side_lightness = (first.l + second.l) * 0.5;
                    side_agreement = sides <= 0.75 * d1.min(d2) + 2.5;
                }
            }
            Measurement {
                gradient,
                ridge_response,
                darkness: side_lightness - centre.l,
                side_agreement,
                valley_response,
                gx,
                gy,
            }
        })
        .collect();
    let nonzero_gradient: Vec<f32> = measurements
        .iter()
        .map(|value| value.gradient)
        .filter(|v| *v > 0.2)
        .collect();
    let gradient_threshold = percentile(nonzero_gradient, 0.90).clamp(4.5, 12.0);
    let nonzero_ridge: Vec<f32> = measurements
        .iter()
        .map(|value| value.ridge_response.max(value.valley_response))
        .filter(|v| *v > 0.2)
        .collect();
    let ridge_threshold = percentile(nonzero_ridge, 0.985).clamp(6.0, 14.0);
    let mut boundary = vec![false; image.pixels.len()];
    let ordinary_candidates: Vec<bool> = measurements
        .iter()
        .map(|value| {
            value.darkness >= 2.0 && value.ridge_response >= ridge_threshold && value.side_agreement
        })
        .collect();
    let valley_candidates: Vec<bool> = measurements
        .iter()
        .map(|value| value.darkness >= 2.0 && value.valley_response >= ridge_threshold.max(7.0))
        .collect();
    let closed_ordinary = erode(
        &dilate(&ordinary_candidates, width, height, 1),
        width,
        height,
        1,
    );
    let ordinary: Vec<bool> = ordinary_candidates
        .iter()
        .zip(&closed_ordinary)
        .map(|(&raw, &closed)| raw || closed)
        .collect();
    let closed_valley = erode(
        &dilate(&valley_candidates, width, height, 2),
        width,
        height,
        2,
    );
    let valley: Vec<bool> = valley_candidates
        .iter()
        .zip(&closed_valley)
        .map(|(&raw, &closed)| raw || closed)
        .collect();
    let ordinary = filter_ridge_components(&ordinary, &measurements, width, height, 6, 2.35);
    let valley = filter_ridge_components(&valley, &measurements, width, height, 16, 2.5);
    let ridge: Vec<bool> = ordinary
        .iter()
        .zip(&valley)
        .map(|(&first, &second)| first || second)
        .collect();
    let mut shading = vec![false; image.pixels.len()];
    let mut dark_boundary = vec![false; image.pixels.len()];
    let mut gradient = vec![0.0; image.pixels.len()];
    for (index, measurement) in measurements.iter().enumerate() {
        let edge = measurement.gradient;
        let darkness = measurement.darkness;
        gradient[index] = edge;
        let authored_ridge = ridge[index];
        let x = index % width;
        let y = index / width;
        let length = (measurement.gx * measurement.gx + measurement.gy * measurement.gy).sqrt();
        let (nx, ny) = if length > 1e-6 {
            (measurement.gx / length, measurement.gy / length)
        } else {
            (0.0, 0.0)
        };
        let before = sample_scalar(&measurements, width, height, x as f32 - nx, y as f32 - ny);
        let after = sample_scalar(&measurements, width, height, x as f32 + nx, y as f32 + ny);
        let local_maximum = edge >= before && edge >= after;
        let material_boundary = edge >= gradient_threshold && local_maximum && !authored_ridge;
        boundary[index] = material_boundary;
        dark_boundary[index] = material_boundary && lab[index].l < 42.0 && darkness > -1.5;
        shading[index] = !authored_ridge
            && !material_boundary
            && edge >= 0.35 * gradient_threshold
            && edge < gradient_threshold;
    }
    let ridge_coverage = dilate(&ridge, width, height, 1);
    let dark_boundary = dilate(&dark_boundary, width, height, 1);
    let face_barrier = boundary.clone();
    let summary = EdgeSummary {
        boundary_pixels: boundary.iter().filter(|&&v| v).count(),
        visible_ridge_pixels: ridge.iter().filter(|&&v| v).count(),
        visible_ridge_coverage_pixels: ridge_coverage.iter().filter(|&&v| v).count(),
        dark_boundary_pixels: dark_boundary.iter().filter(|&&v| v).count(),
        shading_pixels: shading.iter().filter(|&&v| v).count(),
        face_barrier_pixels: face_barrier.iter().filter(|&&v| v).count(),
        visible_ridge_graph_edges: 0,
        dark_boundary_graph_edges: 0,
        gradient_threshold,
        ridge_threshold,
        ..EdgeSummary::default()
    };
    EdgeRoles {
        width,
        height,
        boundary,
        visible_ridge_centres: ridge,
        visible_ridge_coverage: ridge_coverage,
        dark_boundary,
        shading,
        face_barrier,
        gradient,
        visible_ridge_graph: Vec::new(),
        dark_boundary_graph: Vec::new(),
        summary,
    }
}

#[derive(Clone, Debug)]
struct EdgeField {
    tangent_u: Vec<f32>,
    tangent_v: Vec<f32>,
    strength: Vec<f32>,
    edge: Vec<f32>,
}

fn reflected_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let mut value = index;
    let bound = length as isize;
    while value < 0 || value >= bound {
        value = if value < 0 {
            -value - 1
        } else {
            2 * bound - value - 1
        };
    }
    value as usize
}

fn gaussian_kernel(sigma: f64, derivative: bool) -> Vec<f64> {
    let sigma = sigma.max(0.15);
    let radius = (4.0 * sigma + 0.5).floor().max(1.0) as isize;
    let gaussian: Vec<f64> = (-radius..=radius)
        .map(|offset| {
            let position = offset as f64;
            (-0.5 * position * position / (sigma * sigma)).exp()
        })
        .collect();
    let sum: f64 = gaussian.iter().sum();
    let kernel: Vec<f64> = gaussian
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let normalized = value / sum.max(1e-8);
            if derivative {
                let position = index as isize - radius;
                -(position as f64) * normalized / (sigma * sigma)
            } else {
                normalized
            }
        })
        .collect();
    if derivative {
        // scipy.ndimage uses the analytically differentiated, normalized
        // discrete Gaussian.  Do not renormalize its first moment: at the
        // native sigma=0.6 scale it is 0.976609, and forcing it to one changes
        // cross-scale winners and promotes weak shading to hard boundaries.
    } else {
        debug_assert!((kernel.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
    kernel
}

/// SciPy's order-zero ``gaussian_filter1d`` kernel for coordinate tracks.
/// Image fields deliberately remain float32, but NumPy unwraps and smooths
/// graph angles in float64 before constructing normal offsets.
fn gaussian_kernel_f64(sigma: f64) -> Vec<f64> {
    let sigma = sigma.max(0.15);
    let radius = (4.0 * sigma + 0.5).floor().max(1.0) as isize;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| {
            let position = offset as f64;
            (-0.5 * position * position / (sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= total.max(1e-16);
    }
    kernel
}

fn interpolate_nonfinite(values: &mut [f64]) -> bool {
    let known: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect();
    if known.is_empty() {
        return false;
    }
    let first = known[0];
    let first_value = values[first];
    values[..first].fill(first_value);
    let last = *known.last().unwrap_or(&first);
    let last_value = values[last];
    values[last + 1..].fill(last_value);
    for pair in known.windows(2) {
        let left = pair[0];
        let right = pair[1];
        let start = values[left];
        let end = values[right];
        for (offset, target) in values[left + 1..right].iter_mut().enumerate() {
            let amount = (offset + 1) as f64 / (right - left) as f64;
            *target = start * (1.0 - amount) + end * amount;
        }
    }
    true
}

fn convolve_horizontal(values: &[f32], width: usize, _height: usize, kernel: &[f64]) -> Vec<f32> {
    let radius = kernel.len() as isize / 2;
    (0..values.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            kernel
                .iter()
                .enumerate()
                .map(|(offset, &weight)| {
                    let px = reflected_index(x as isize + offset as isize - radius, width);
                    values[y * width + px] as f64 * weight
                })
                .sum::<f64>() as f32
        })
        .collect()
}

fn convolve_vertical(values: &[f32], width: usize, height: usize, kernel: &[f64]) -> Vec<f32> {
    let radius = kernel.len() as isize / 2;
    (0..values.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            kernel
                .iter()
                .enumerate()
                .map(|(offset, &weight)| {
                    let py = reflected_index(y as isize + offset as isize - radius, height);
                    values[py * width + x] as f64 * weight
                })
                .sum::<f64>() as f32
        })
        .collect()
}

fn separable_filter(
    values: &[f32],
    width: usize,
    height: usize,
    horizontal: &[f64],
    vertical: &[f64],
) -> Vec<f32> {
    let intermediate = convolve_horizontal(values, width, height, horizontal);
    convolve_vertical(&intermediate, width, height, vertical)
}

fn profile_scales(width: usize, height: usize) -> Vec<f64> {
    let base = [0.6_f64, 1.2, 2.4];
    let factor = (width.max(height) as f64 / 1024.0).max(1.0);
    let mut result = base.to_vec();
    if factor > 1.0 + 1e-5 {
        result.extend(base.map(|value| value * factor));
    }
    result.sort_by(f64::total_cmp);
    result.dedup_by(|first, second| (*first - *second).abs() < 1e-4);
    result
}

fn estimate_profile_edge_field(image: &Raster, lab: &[Lab]) -> EdgeField {
    let width = image.width;
    let height = image.height;
    let count = image.pixels.len();
    let channels = [
        lab.iter().map(|value| value.l / 100.0).collect::<Vec<_>>(),
        lab.iter().map(|value| value.a / 128.0).collect::<Vec<_>>(),
        lab.iter().map(|value| value.b / 128.0).collect::<Vec<_>>(),
    ];
    let mut selected_score = vec![f32::NEG_INFINITY; count];
    let mut selected_edge = vec![0.0_f32; count];
    let mut selected_u = vec![0.0_f32; count];
    let mut selected_v = vec![0.0_f32; count];
    for sigma in profile_scales(width, height) {
        let gaussian = gaussian_kernel(sigma, false);
        let derivative = gaussian_kernel(sigma, true);
        let mut jxx = vec![0.0_f32; count];
        let mut jxy = vec![0.0_f32; count];
        let mut jyy = vec![0.0_f32; count];
        for channel in &channels {
            let gx = separable_filter(channel, width, height, &derivative, &gaussian);
            let gy = separable_filter(channel, width, height, &gaussian, &derivative);
            jxx.par_iter_mut()
                .zip(&gx)
                .for_each(|(target, &value)| *target += value * value);
            jxy.par_iter_mut()
                .zip(&gx)
                .zip(&gy)
                .for_each(|((target, &x), &y)| *target += x * y);
            jyy.par_iter_mut()
                .zip(&gy)
                .for_each(|(target, &value)| *target += value * value);
        }
        let integration = gaussian_kernel((1.1 * sigma).max(0.7), false);
        jxx = separable_filter(&jxx, width, height, &integration, &integration);
        jxy = separable_filter(&jxy, width, height, &integration, &integration);
        jyy = separable_filter(&jyy, width, height, &integration, &integration);
        selected_score
            .par_iter_mut()
            .zip(&mut selected_edge)
            .zip(&mut selected_u)
            .zip(&mut selected_v)
            .enumerate()
            .for_each(|(index, (((score_out, edge_out), u_out), v_out))| {
                let trace = jxx[index] + jyy[index];
                let discriminant = ((jxx[index] - jyy[index]).powi(2)
                    + 4.0 * jxy[index] * jxy[index])
                    .max(0.0)
                    .sqrt();
                let major = 0.5 * (trace + discriminant);
                let minor = 0.5 * (trace - discriminant);
                let coherence = ((major - minor) / (major + minor + 1e-8)).clamp(0.0, 1.0);
                let edge = major.max(0.0).sqrt();
                let score = edge * coherence * sigma.sqrt() as f32;
                if score > *score_out {
                    // The principal tensor eigenvector is the edge normal;
                    // doubled tangent angle equals doubled normal angle plus
                    // pi, hence the negation below.
                    let double_normal = (2.0 * jxy[index]).atan2(jxx[index] - jyy[index]);
                    *score_out = score;
                    *edge_out = edge;
                    *u_out = -double_normal.cos();
                    *v_out = -double_normal.sin();
                }
            });
    }
    let score_normalizer = percentile(
        selected_score
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect(),
        0.985,
    )
    .max(1e-8);
    let edge_normalizer = percentile(
        selected_edge
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect(),
        0.985,
    )
    .max(1e-8);
    let strength: Vec<f32> = selected_score
        .iter()
        .map(|value| (*value / score_normalizer).clamp(0.0, 1.0))
        .collect();
    for index in 0..count {
        selected_u[index] *= strength[index];
        selected_v[index] *= strength[index];
        selected_edge[index] = (selected_edge[index] / edge_normalizer).clamp(0.0, 1.0);
    }
    EdgeField {
        tangent_u: selected_u,
        tangent_v: selected_v,
        strength,
        edge: selected_edge,
    }
}

fn bilinear(values: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    // ndimage.map_coordinates receives float32 coordinates, but evaluates
    // its interpolation weights and accumulator in C double precision before
    // storing the float32 output.
    let tx = x as f64 - x0 as f64;
    let ty = y as f64 - y0 as f64;
    let top = values[y0 * width + x0] as f64 * (1.0 - tx) + values[y0 * width + x1] as f64 * tx;
    let bottom = values[y1 * width + x0] as f64 * (1.0 - tx) + values[y1 * width + x1] as f64 * tx;
    (top * (1.0 - ty) + bottom * ty) as f32
}

fn bilinear_f64(values: &[f32], width: usize, height: usize, x: f64, y: f64) -> f32 {
    let x = x.clamp(0.0, width.saturating_sub(1) as f64);
    let y = y.clamp(0.0, height.saturating_sub(1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let top = values[y0 * width + x0] as f64 * (1.0 - tx) + values[y0 * width + x1] as f64 * tx;
    let bottom = values[y1 * width + x0] as f64 * (1.0 - tx) + values[y1 * width + x1] as f64 * tx;
    (top * (1.0 - ty) + bottom * ty) as f32
}

fn oriented_nonmaximum(field: &EdgeField, width: usize, height: usize) -> Vec<f32> {
    (0..field.edge.len())
        .into_par_iter()
        .map(|index| {
            if field.strength[index] <= 0.0 {
                return 0.0;
            }
            let x = (index % width) as f32;
            let y = (index / width) as f32;
            let tangent = 0.5 * field.tangent_v[index].atan2(field.tangent_u[index]);
            let nx = -tangent.sin();
            let ny = tangent.cos();
            let before = bilinear(&field.edge, width, height, x - nx, y - ny);
            let after = bilinear(&field.edge, width, height, x + nx, y + ny);
            if field.edge[index] >= before && field.edge[index] >= after {
                field.edge[index]
            } else {
                0.0
            }
        })
        .collect()
}

fn edge_hysteresis(nms: &[f32], field: &EdgeField, width: usize, height: usize) -> Vec<bool> {
    let low: Vec<bool> = nms.iter().map(|value| *value >= 0.045).collect();
    let mut selected = vec![false; nms.len()];
    let mut queue: VecDeque<usize> = nms
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= 0.16).then_some(index))
        .collect();
    for &index in &queue {
        selected[index] = true;
    }
    while let Some(index) = queue.pop_front() {
        for neighbour in {
            let x = index % width;
            let y = index / width;
            let mut result = Vec::with_capacity(8);
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px >= 0 && py >= 0 && px < width as isize && py < height as isize {
                        result.push(py as usize * width + px as usize);
                    }
                }
            }
            result
        } {
            if low[neighbour] && !selected[neighbour] {
                selected[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }
    let scale = (width.max(height) as f64 / 1024.0).max(1.0);
    let preliminary = thin_edge_mask(&selected, width, height);
    let repaired = bridge_short_gaps(&preliminary, nms, field, width, height, 8.0 * scale);
    let thinned = thin_edge_mask(&repaired, width, height);
    let final_repaired = bridge_short_gaps(&thinned, nms, field, width, height, 8.0 * scale);
    let final_mask = thin_edge_mask(&final_repaired, width, height);
    if let Some(prefix) = std::env::var_os("PICVEC_EDGE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        for (name, mask) in [
            ("hysteresis", &selected),
            ("preliminary", &preliminary),
            ("repaired-first", &repaired),
            ("thinned-second", &thinned),
            ("repaired-second", &final_repaired),
            ("skeleton-internal", &final_mask),
        ] {
            let raster = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
                image::Luma([if mask[y as usize * width + x as usize] {
                    255
                } else {
                    0
                }])
            });
            let _ = raster.save(format!("{prefix}-{name}.png"));
        }
    }
    final_mask
}

fn thin_edge_mask(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut result = mask.to_vec();
    const FIRST: &[u8] = &[
        14, 20, 22, 28, 30, 52, 54, 56, 60, 62, 80, 84, 86, 88, 92, 94, 112, 116, 118, 120, 124,
        126, 193, 208, 209, 212, 216, 217, 220, 224, 225, 240, 241, 244, 248, 249, 252,
    ];
    const SECOND: &[u8] = &[
        5, 7, 13, 14, 15, 28, 29, 30, 31, 65, 67, 69, 71, 77, 79, 97, 99, 101, 103, 131, 133, 135,
        141, 143, 157, 159, 193, 195, 197, 199, 205, 207, 224, 225, 227, 229, 231,
    ];
    loop {
        let old_count = result.iter().filter(|&&value| value).count();
        for lut in [FIRST, SECOND] {
            let snapshot = result.clone();
            let remove: Vec<usize> = (0..height)
                .into_par_iter()
                .flat_map_iter(|y| {
                    let snapshot = &snapshot;
                    (0..width).filter_map(move |x| {
                        let index = y * width + x;
                        if !snapshot[index] {
                            return None;
                        }
                        let selected = |dx: isize, dy: isize| {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            px >= 0
                                && py >= 0
                                && px < width as isize
                                && py < height as isize
                                && snapshot[py as usize * width + px as usize]
                        };
                        let mut pattern = 0_u8;
                        for (bit, dx, dy) in [
                            (8, -1, -1),
                            (4, 0, -1),
                            (2, 1, -1),
                            (16, -1, 0),
                            (1, 1, 0),
                            (32, -1, 1),
                            (64, 0, 1),
                            (128, 1, 1),
                        ] {
                            if selected(dx, dy) {
                                pattern |= bit;
                            }
                        }
                        lut.binary_search(&pattern).is_ok().then_some(index)
                    })
                })
                .collect();
            for index in remove {
                result[index] = false;
            }
        }
        if result.iter().filter(|&&value| value).count() == old_count {
            break;
        }
    }
    result
}

fn full_degree(mask: &[bool], width: usize, height: usize, index: usize) -> usize {
    let x = index % width;
    let y = index / width;
    let mut count = 0_usize;
    for dy in -1_isize..=1 {
        for dx in -1_isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let px = x as isize + dx;
            let py = y as isize + dy;
            if px >= 0
                && py >= 0
                && px < width as isize
                && py < height as isize
                && mask[py as usize * width + px as usize]
            {
                count += 1;
            }
        }
    }
    count
}

fn raster_line(mut x0: isize, mut y0: isize, x1: isize, y1: isize) -> Vec<(usize, usize)> {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    let mut points = Vec::new();
    loop {
        points.push((x0 as usize, y0 as usize));
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
    points
}

fn bridge_short_gaps(
    skeleton: &[bool],
    nms: &[f32],
    field: &EdgeField,
    width: usize,
    height: usize,
    maximum_gap: f64,
) -> Vec<bool> {
    let endpoints: Vec<usize> = (0..skeleton.len())
        .filter(|&index| skeleton[index] && full_degree(skeleton, width, height, index) == 1)
        .collect();
    if endpoints.len() < 2 {
        return skeleton.to_vec();
    }
    let cosine = 16.0_f64.to_radians().cos();
    let mut candidates = Vec::<(f64, usize, usize)>::new();
    for first in 0..endpoints.len() {
        let first_index = endpoints[first];
        let first_x = (first_index % width) as f64;
        let first_y = (first_index / width) as f64;
        let first_theta =
            0.5 * (field.tangent_v[first_index] as f64).atan2(field.tangent_u[first_index] as f64);
        for (second, &second_index) in endpoints.iter().enumerate().skip(first + 1) {
            let second_x = (second_index % width) as f64;
            let second_y = (second_index / width) as f64;
            let vx = second_x - first_x;
            let vy = second_y - first_y;
            let distance = vx.hypot(vy);
            if !(1.5..=maximum_gap).contains(&distance) {
                continue;
            }
            let direction_x = vx / distance;
            let direction_y = vy / distance;
            let second_theta = 0.5
                * (field.tangent_v[second_index] as f64)
                    .atan2(field.tangent_u[second_index] as f64);
            let first_alignment =
                (first_theta.cos() * direction_x + first_theta.sin() * direction_y).abs();
            let second_alignment =
                (second_theta.cos() * direction_x + second_theta.sin() * direction_y).abs();
            if first_alignment.min(second_alignment) < cosine {
                continue;
            }
            let sample_count = (distance * 2.0).ceil().max(2.0) as usize + 1;
            let target_angle = 2.0 * direction_y.atan2(direction_x);
            let target_u = target_angle.cos();
            let target_v = target_angle.sin();
            let mut edge_sum = 0.0_f64;
            let mut alignments = Vec::new();
            for sample in 0..sample_count {
                let amount = sample as f64 / (sample_count - 1) as f64;
                let x = first_x + amount * vx;
                let y = first_y + amount * vy;
                edge_sum += bilinear_f64(nms, width, height, x, y) as f64;
                let u = bilinear_f64(&field.tangent_u, width, height, x, y) as f64;
                let v = bilinear_f64(&field.tangent_v, width, height, x, y) as f64;
                let magnitude = u.hypot(v);
                if magnitude > 0.015 {
                    alignments.push(((u * target_u + v * target_v) / magnitude).abs());
                }
            }
            let edge_support = edge_sum / sample_count as f64;
            alignments.sort_by(f64::total_cmp);
            let alignment = if alignments.is_empty() {
                0.0
            } else if alignments.len() % 2 == 0 {
                let middle = alignments.len() / 2;
                0.5 * (alignments[middle - 1] + alignments[middle])
            } else {
                alignments[alignments.len() / 2]
            };
            if edge_support < 0.018 || alignment < cosine {
                continue;
            }
            let score = distance + 2.0 * (2.0 - first_alignment - second_alignment) - edge_support;
            candidates.push((score, first, second));
        }
    }
    candidates.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then(first.1.cmp(&second.1))
            .then(first.2.cmp(&second.2))
    });
    let mut output = skeleton.to_vec();
    let mut used = vec![false; endpoints.len()];
    for (_, first, second) in candidates {
        if used[first] || used[second] {
            continue;
        }
        let first_index = endpoints[first];
        let second_index = endpoints[second];
        for (x, y) in raster_line(
            (first_index % width) as isize,
            (first_index / width) as isize,
            (second_index % width) as isize,
            (second_index / width) as isize,
        ) {
            output[y * width + x] = true;
        }
        used[first] = true;
        used[second] = true;
    }
    output
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfileRole {
    Boundary,
    Ridge,
    RidgeOnBoundary,
    Shading,
    Unknown,
}

#[derive(Clone, Copy, Debug)]
struct ProfileClassification {
    role: ProfileRole,
    confidence: f64,
    centre: f64,
    width: f64,
    dark_contrast: f64,
}

#[derive(Clone, Debug)]
struct ClassifiedChain {
    pixels: Vec<usize>,
    points: Vec<[f64; 2]>,
    normals: Vec<[f64; 2]>,
    profiles: Vec<ProfileClassification>,
}

fn median_four_f64(values: &[[f32; 3]]) -> [f64; 3] {
    let mut result = [0.0_f64; 3];
    for channel in 0..3 {
        let mut samples = [
            values[0][channel],
            values[1][channel],
            values[2][channel],
            values[3][channel],
        ];
        samples.sort_by(f32::total_cmp);
        result[channel] = 0.5 * (samples[1] as f64 + samples[2] as f64);
    }
    result
}

fn lab_vector_distance(first: [f32; 3], second: [f32; 3]) -> f32 {
    ((first[0] - second[0]).powi(2)
        + (first[1] - second[1]).powi(2)
        + (first[2] - second[2]).powi(2))
    .sqrt()
}

fn lab_vector_distance_f64(first: [f64; 3], second: [f64; 3]) -> f64 {
    ((first[0] - second[0]).powi(2)
        + (first[1] - second[1]).powi(2)
        + (first[2] - second[2]).powi(2))
    .sqrt()
}

fn threshold_crossing(
    values: &[f64],
    offsets: &[f32],
    first: usize,
    second: usize,
    target: f64,
) -> f64 {
    let delta = values[second] - values[first];
    if delta.abs() <= 1e-8 {
        return 0.5 * (offsets[first] as f64 + offsets[second] as f64);
    }
    let amount = ((target - values[first]) / delta).clamp(0.0, 1.0);
    offsets[first] as f64 + amount * (offsets[second] - offsets[first]) as f64
}

fn classify_profile(samples: &[[f32; 3]], offsets: &[f32], radius: f32) -> ProfileClassification {
    const MINIMUM_CONTRAST: f64 = 0.06;
    let radius = radius as f64;
    let maximum_width = 8.0 * (radius / 7.0).max(1.0);
    let left = median_four_f64(&samples[..4]);
    let right = median_four_f64(&samples[samples.len() - 4..]);
    let side_delta = lab_vector_distance_f64(left, right);
    let lightness: Vec<f64> = samples.iter().map(|value| value[0] as f64).collect();
    let mut candidate: Option<(f64, f64, f64, f64)> = None;
    for dark in [true, false] {
        let extremum_index = (0..lightness.len())
            .min_by(|&first, &second| {
                let ordering = lightness[first].total_cmp(&lightness[second]);
                if dark {
                    ordering
                } else {
                    ordering.reverse()
                }
            })
            .unwrap_or(0);
        let extremum = lightness[extremum_index];
        let contrast = if dark {
            left[0].min(right[0]) - extremum
        } else {
            extremum - left[0].max(right[0])
        };
        if contrast < MINIMUM_CONTRAST {
            continue;
        }
        let threshold = if dark {
            extremum + 0.5 * contrast
        } else {
            extremum - 0.5 * contrast
        };
        let mut lower = extremum_index;
        let mut upper = extremum_index;
        while lower > 0
            && if dark {
                lightness[lower - 1] <= threshold
            } else {
                lightness[lower - 1] >= threshold
            }
        {
            lower -= 1;
        }
        while upper + 1 < lightness.len()
            && if dark {
                lightness[upper + 1] <= threshold
            } else {
                lightness[upper + 1] >= threshold
            }
        {
            upper += 1;
        }
        if lower == 0 || upper + 1 == lightness.len() {
            continue;
        }
        let low = threshold_crossing(&lightness, offsets, lower - 1, lower, threshold);
        let high = threshold_crossing(&lightness, offsets, upper, upper + 1, threshold);
        let width = high - low;
        let centre = 0.5 * (low + high);
        if !(0.8..=maximum_width).contains(&width) || centre.abs() > 0.5 * width + 1.5 {
            continue;
        }
        let score = (contrast / MINIMUM_CONTRAST).clamp(0.0, 4.0)
            * (1.0 - centre.abs() / radius).clamp(0.0, 1.0);
        if candidate.map(|value| score > value.0).unwrap_or(true) {
            candidate = Some((
                score,
                centre,
                width,
                if dark { contrast } else { -contrast },
            ));
        }
    }

    // Full-colour bounded residual.  This is the part that preserves a
    // chromatic one-pixel line whose L* alone is not an extremum.
    let residual: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let amount = (offsets[index] as f64 + radius) / (2.0 * radius);
            let baseline = [
                left[0] * (1.0 - amount) + right[0] * amount,
                left[1] * (1.0 - amount) + right[1] * amount,
                left[2] * (1.0 - amount) + right[2] * amount,
            ];
            lab_vector_distance_f64(
                [sample[0] as f64, sample[1] as f64, sample[2] as f64],
                baseline,
            )
        })
        .collect();
    let residual_index = (0..residual.len())
        .max_by(|&first, &second| residual[first].total_cmp(&residual[second]))
        .unwrap_or(0);
    let residual_peak = residual[residual_index];
    let gradients: Vec<f64> = samples
        .windows(2)
        .map(|pair| {
            lab_vector_distance_f64(
                [pair[0][0] as f64, pair[0][1] as f64, pair[0][2] as f64],
                [pair[1][0] as f64, pair[1][1] as f64, pair[1][2] as f64],
            )
        })
        .collect();
    let left_gradient = gradients[..residual_index]
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let right_gradient = gradients[residual_index.min(gradients.len())..]
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if residual_peak >= MINIMUM_CONTRAST
        && left_gradient.min(right_gradient) >= 0.5 * MINIMUM_CONTRAST
    {
        let threshold = 0.5 * residual_peak;
        let mut lower = residual_index;
        let mut upper = residual_index;
        while lower > 0 && residual[lower - 1] >= threshold {
            lower -= 1;
        }
        while upper + 1 < residual.len() && residual[upper + 1] >= threshold {
            upper += 1;
        }
        if lower > 0 && upper + 1 < residual.len() {
            let low = threshold_crossing(&residual, offsets, lower - 1, lower, threshold);
            let high = threshold_crossing(&residual, offsets, upper, upper + 1, threshold);
            let width = high - low;
            let centre = 0.5 * (low + high);
            if (0.8..=maximum_width).contains(&width) && centre.abs() <= 0.5 * width + 1.5 {
                let score = (residual_peak / MINIMUM_CONTRAST).clamp(0.0, 4.0)
                    * (1.0 - centre.abs() / radius).clamp(0.0, 1.0);
                if candidate.map(|value| score > value.0).unwrap_or(true) {
                    candidate = Some((
                        score,
                        centre,
                        width,
                        left[0].min(right[0]) - lightness[residual_index],
                    ));
                }
            }
        }
    }
    if let Some((score, centre, width, dark_contrast)) = candidate {
        return ProfileClassification {
            role: if side_delta >= 0.08 {
                ProfileRole::RidgeOnBoundary
            } else {
                ProfileRole::Ridge
            },
            confidence: (score / 4.0).clamp(0.0, 1.0),
            centre,
            width,
            dark_contrast,
        };
    }
    let tail_lightness = (left[0] - right[0]).abs();
    // The two-exit chromatic residual above uses full Lab gradients, but the
    // fallback step/shading decision in the Python source is intentionally
    // based only on dL*/dn.  Reusing the colour gradient here promoted weak
    // hue changes to hard Paint barriers.
    let derivative = lightness
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f64, f64::max);
    let tail_contrast = tail_lightness.max(side_delta);
    if tail_contrast >= MINIMUM_CONTRAST && derivative >= 0.5 * MINIMUM_CONTRAST {
        ProfileClassification {
            role: ProfileRole::Boundary,
            confidence: (tail_contrast.max(derivative) / (2.0 * MINIMUM_CONTRAST)).clamp(0.0, 1.0),
            centre: f64::NAN,
            width: f64::NAN,
            dark_contrast: 0.0,
        }
    } else if derivative >= 0.25 * MINIMUM_CONTRAST {
        ProfileClassification {
            role: ProfileRole::Shading,
            confidence: (derivative / MINIMUM_CONTRAST).clamp(0.0, 1.0),
            centre: f64::NAN,
            width: f64::NAN,
            dark_contrast: 0.0,
        }
    } else {
        ProfileClassification {
            role: ProfileRole::Unknown,
            confidence: 0.0,
            centre: f64::NAN,
            width: f64::NAN,
            dark_contrast: 0.0,
        }
    }
}

fn normalized_lab_sample(lab: &[Lab], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    let sample_channel = |channel: usize| {
        let x = x.clamp(0.0, width.saturating_sub(1) as f32);
        let y = y.clamp(0.0, height.saturating_sub(1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width.saturating_sub(1));
        let y1 = (y0 + 1).min(height.saturating_sub(1));
        let tx = x as f64 - x0 as f64;
        let ty = y as f64 - y0 as f64;
        let value = |index: usize| match channel {
            0 => (lab[index].l / 100.0) as f64,
            1 => (lab[index].a / 128.0) as f64,
            _ => (lab[index].b / 128.0) as f64,
        };
        let top = value(y0 * width + x0) * (1.0 - tx) + value(y0 * width + x1) * tx;
        let bottom = value(y1 * width + x0) * (1.0 - tx) + value(y1 * width + x1) * tx;
        (top * (1.0 - ty) + bottom * ty) as f32
    };
    [sample_channel(0), sample_channel(1), sample_channel(2)]
}

fn classify_skeleton_chains(
    skeleton: &[bool],
    field: &EdgeField,
    lab: &[Lab],
    width: usize,
    height: usize,
    radius: f32,
) -> Vec<ClassifiedChain> {
    let step = 0.25_f32;
    let sample_count = (2.0 * radius / step).round() as usize + 1;
    let offsets: Vec<f32> = (0..sample_count)
        .map(|index| -radius + index as f32 * step)
        .collect();
    let smooth_kernel = gaussian_kernel_f64(0.8);
    let smooth_radius = smooth_kernel.len() / 2;
    trace_edge_chains(skeleton, width, height)
        .into_iter()
        .filter_map(|pixels| {
            // classify_edge_role_raster skips chains shorter than
            // minimum_chain_points (the reference default is three).
            if pixels.len() < 3 {
                return None;
            }
            let mut points = Vec::<[f64; 2]>::with_capacity(pixels.len());
            for &pixel in &pixels {
                let x = (pixel % width) as f32;
                let y = (pixel / width) as f32;
                let tangent = 0.5 * field.tangent_v[pixel].atan2(field.tangent_u[pixel]);
                let nx = -tangent.sin();
                let ny = tangent.cos();
                let displacement_values: Vec<f32> =
                    (0..=8).map(|offset| -1.0 + 0.25 * offset as f32).collect();
                let edge_values: Vec<f32> = displacement_values
                    .iter()
                    .map(|offset| {
                        bilinear(&field.edge, width, height, x + offset * nx, y + offset * ny)
                    })
                    .collect();
                let minimum = edge_values.iter().copied().fold(f32::INFINITY, f32::min);
                let mut weight = 0.0_f32;
                let mut weighted = 0.0_f32;
                for (&offset, &value) in displacement_values.iter().zip(&edge_values) {
                    let amount = (value - minimum).max(0.0).powi(2);
                    weighted += offset * amount;
                    weight += amount;
                }
                let displacement = if weight > 1e-8 {
                    weighted / weight
                } else {
                    0.0
                };
                points.push([
                    x as f64 + 0.5 + (displacement * nx) as f64,
                    y as f64 + 0.5 + (displacement * ny) as f64,
                ]);
            }
            let mut angles = Vec::<f64>::with_capacity(points.len());
            for index in 0..points.len() {
                let sample_x = (points[index][0] - 0.5) as f32;
                let sample_y = (points[index][1] - 0.5) as f32;
                let u = bilinear(&field.tangent_u, width, height, sample_x, sample_y);
                let v = bilinear(&field.tangent_v, width, height, sample_x, sample_y);
                let theta = 0.5 * v.atan2(u);
                let mut angle = (theta.sin() as f64).atan2(theta.cos() as f64);
                let before = points[index.saturating_sub(1)];
                let after = points[(index + 1).min(points.len() - 1)];
                if angle.cos() * (after[0] - before[0]) + angle.sin() * (after[1] - before[1]) < 0.0
                {
                    angle += std::f64::consts::PI;
                }
                if let Some(&previous) = angles.last() {
                    while angle - previous > std::f64::consts::PI {
                        angle -= 2.0 * std::f64::consts::PI;
                    }
                    while angle - previous < -std::f64::consts::PI {
                        angle += 2.0 * std::f64::consts::PI;
                    }
                }
                angles.push(angle);
            }
            if angles.len() >= 3 {
                let original = angles.clone();
                for (index, target) in angles.iter_mut().enumerate() {
                    *target = smooth_kernel
                        .iter()
                        .enumerate()
                        .map(|(offset, &weight)| {
                            let source = (index as isize + offset as isize - smooth_radius as isize)
                                .clamp(0, original.len().saturating_sub(1) as isize)
                                as usize;
                            original[source] * weight
                        })
                        .sum();
                }
            }
            let normals: Vec<[f64; 2]> = angles
                .iter()
                .map(|&angle| [-angle.sin(), angle.cos()])
                .collect();
            let profiles: Vec<ProfileClassification> = points
                .iter()
                .zip(&normals)
                .map(|(point, normal)| {
                    let samples: Vec<[f32; 3]> = offsets
                        .iter()
                        .map(|offset| {
                            normalized_lab_sample(
                                lab,
                                width,
                                height,
                                (point[0] - 0.5 + *offset as f64 * normal[0]) as f32,
                                (point[1] - 0.5 + *offset as f64 * normal[1]) as f32,
                            )
                        })
                        .collect();
                    classify_profile(&samples, &offsets, radius)
                })
                .collect();
            Some(ClassifiedChain {
                pixels,
                points,
                normals,
                profiles,
            })
        })
        .collect()
}

fn rasterize_profile_band(
    mask: &mut [bool],
    width: usize,
    height: usize,
    centre_x: f64,
    centre_y: f64,
    band_width: f64,
) {
    // stroke_graph.rasterize_edges uses a half-pixel antialias ownership
    // margin.  This is intentionally not a post-hoc dilation: the mask is
    // derived from the same centre-line and measured width as the SVG path.
    let radius = (0.5 * band_width + 0.5).max(0.75);
    let minimum_x = (centre_x - radius - 0.5).floor().max(0.0) as usize;
    let maximum_x = (centre_x + radius - 0.5)
        .ceil()
        .min(width.saturating_sub(1) as f64) as usize;
    let minimum_y = (centre_y - radius - 0.5).floor().max(0.0) as usize;
    let maximum_y = (centre_y + radius - 0.5)
        .ceil()
        .min(height.saturating_sub(1) as f64) as usize;
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let dx = x as f64 + 0.5 - centre_x;
            let dy = y as f64 + 0.5 - centre_y;
            // skimage.draw.disk uses a strict radius test in array-index
            // coordinates.  An extra epsilon here materially broadens a
            // one-pixel diagonal line, so retain the source implementation's
            // exact ownership footprint.
            if dx * dx + dy * dy < radius * radius {
                mask[y * width + x] = true;
            }
        }
    }
}

fn rasterize_source_graph(
    edges: &[SourceEdge],
    width: usize,
    height: usize,
    use_width: bool,
) -> Vec<bool> {
    let mut result = vec![false; width * height];
    for edge in edges {
        if edge.points.is_empty() {
            continue;
        }
        if edge.points.len() == 1 {
            let band_width = if use_width { edge.width } else { 0.5 };
            rasterize_profile_band(
                &mut result,
                width,
                height,
                edge.points[0][0],
                edge.points[0][1],
                band_width,
            );
            continue;
        }
        for pair in edge.points.windows(2) {
            let distance = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            let count = (2.0 * distance).ceil().max(1.0) as usize;
            for sample in 0..=count {
                let amount = sample as f64 / count as f64;
                let x = pair[0][0] * (1.0 - amount) + pair[1][0] * amount;
                let y = pair[0][1] * (1.0 - amount) + pair[1][1] * amount;
                let band_width = if use_width { edge.width } else { 0.5 };
                rasterize_profile_band(&mut result, width, height, x, y, band_width);
            }
        }
    }
    result
}

fn edge_neighbours(mask: &[bool], width: usize, height: usize, index: usize) -> Vec<usize> {
    let x = index % width;
    let y = index / width;
    let mut result = Vec::with_capacity(8);
    for dy in -1_isize..=1 {
        for dx in -1_isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let px = x as isize + dx;
            let py = y as isize + dy;
            if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                continue;
            }
            let neighbour = py as usize * width + px as usize;
            if !mask[neighbour] {
                continue;
            }
            // Match geometry_raster_edge_vectorizer._neighbours: an
            // orthogonally connected corner owns the turn, so the diagonal
            // is not also emitted as a triangular shortcut.
            if dx != 0
                && dy != 0
                && (mask[y * width + px as usize] || mask[py as usize * width + x])
            {
                continue;
            }
            result.push(neighbour);
        }
    }
    result
}

fn trace_edge_chains(mask: &[bool], width: usize, height: usize) -> Vec<Vec<usize>> {
    let degree: Vec<usize> = (0..mask.len())
        .map(|index| {
            if mask[index] {
                edge_neighbours(mask, width, height, index).len()
            } else {
                0
            }
        })
        .collect();
    let mut starts: Vec<usize> = (0..mask.len())
        .filter(|&index| mask[index] && degree[index] != 2)
        .collect();
    starts.extend((0..mask.len()).filter(|&index| mask[index] && degree[index] == 2));
    let mut visited = HashSet::<(usize, usize)>::new();
    let edge_key = |first: usize, second: usize| {
        if first < second {
            (first, second)
        } else {
            (second, first)
        }
    };
    let mut chains = Vec::new();
    for start in starts {
        for following in edge_neighbours(mask, width, height, start) {
            if visited.contains(&edge_key(start, following)) {
                continue;
            }
            let mut chain = vec![start, following];
            let mut previous = start;
            let mut current = following;
            visited.insert(edge_key(previous, current));
            while degree[current] == 2 {
                let Some(candidate) = edge_neighbours(mask, width, height, current)
                    .into_iter()
                    .find(|&value| {
                        value != previous && !visited.contains(&edge_key(current, value))
                    })
                else {
                    break;
                };
                visited.insert(edge_key(current, candidate));
                chain.push(candidate);
                previous = current;
                current = candidate;
            }
            chains.push(chain);
        }
    }
    chains
}

fn remove_tiny_components(mask: &mut [bool], width: usize, height: usize, maximum: usize) {
    let mut seen = vec![false; mask.len()];
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        seen[start] = true;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                        continue;
                    }
                    let neighbour = py as usize * width + px as usize;
                    if mask[neighbour] && !seen[neighbour] {
                        seen[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        if component.len() <= maximum {
            for index in component {
                mask[index] = false;
            }
        }
    }
}

fn median_sorted(values: &[f32]) -> f32 {
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

// Felzenszwalb-Huttenlocher exact squared Euclidean distance transform.
// This is scipy.ndimage.distance_transform_edt for a binary 2-D mask, split
// into its two separable one-dimensional passes.
fn distance_transform_1d(values: &[f32]) -> Vec<f32> {
    let count = values.len();
    if count == 0 {
        return Vec::new();
    }
    let mut sites = Vec::<usize>::new();
    for (index, &value) in values.iter().enumerate() {
        if value.is_finite() {
            sites.push(index);
        }
    }
    if sites.is_empty() {
        return vec![f32::INFINITY; count];
    }
    let mut envelope = vec![0_usize; sites.len()];
    let mut boundaries = vec![0.0_f32; sites.len() + 1];
    envelope[0] = sites[0];
    boundaries[0] = f32::NEG_INFINITY;
    boundaries[1] = f32::INFINITY;
    let mut last = 0_usize;
    for &site in sites.iter().skip(1) {
        let mut crossing;
        loop {
            let previous = envelope[last];
            crossing = ((values[site] + (site * site) as f32)
                - (values[previous] + (previous * previous) as f32))
                / (2.0 * (site as f32 - previous as f32));
            if crossing > boundaries[last] || last == 0 {
                break;
            }
            last -= 1;
        }
        if crossing <= boundaries[last] && last == 0 {
            envelope[0] = site;
            boundaries[1] = f32::INFINITY;
        } else {
            last += 1;
            envelope[last] = site;
            boundaries[last] = crossing;
            boundaries[last + 1] = f32::INFINITY;
        }
    }
    let mut output = vec![0.0_f32; count];
    let mut selected = 0_usize;
    for (index, target) in output.iter_mut().enumerate() {
        while selected < last && boundaries[selected + 1] < index as f32 {
            selected += 1;
        }
        let site = envelope[selected];
        let delta = index as f32 - site as f32;
        *target = delta * delta + values[site];
    }
    output
}

fn distance_to_background(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0_f32; mask.len()];
    for y in 0..height {
        let values: Vec<f32> = (0..width)
            .map(|x| {
                if mask[y * width + x] {
                    f32::INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let distance = distance_transform_1d(&values);
        horizontal[y * width..(y + 1) * width].copy_from_slice(&distance);
    }
    let mut squared = vec![0.0_f32; mask.len()];
    for x in 0..width {
        let values: Vec<f32> = (0..height).map(|y| horizontal[y * width + x]).collect();
        let distance = distance_transform_1d(&values);
        for y in 0..height {
            squared[y * width + x] = distance[y];
        }
    }
    squared.into_iter().map(f32::sqrt).collect()
}

fn pattern_index(mask: &[bool], width: usize, height: usize, index: usize) -> usize {
    let x = index % width;
    let y = index / width;
    let mut pattern = 0_usize;
    for row in 0..3 {
        for column in 0..3 {
            let px = x as isize + column as isize - 1;
            let py = y as isize + row as isize - 1;
            if px >= 0
                && py >= 0
                && px < width as isize
                && py < height as isize
                && mask[py as usize * width + px as usize]
            {
                pattern |= 1 << (row * 3 + column);
            }
        }
    }
    pattern
}

fn pattern_component_count(pattern: usize) -> usize {
    let mut selected = [false; 9];
    for (index, value) in selected.iter_mut().enumerate() {
        *value = pattern & (1 << index) != 0;
    }
    let mut seen = [false; 9];
    let mut count = 0_usize;
    for start in 0..9 {
        if !selected[start] || seen[start] {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(index) = queue.pop_front() {
            let x = index % 3;
            let y = index / 3;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px < 0 || py < 0 || px >= 3 || py >= 3 {
                        continue;
                    }
                    let other = py as usize * 3 + px as usize;
                    if selected[other] && !seen[other] {
                        seen[other] = true;
                        queue.push_back(other);
                    }
                }
            }
        }
    }
    count
}

/// NumPy ``default_rng(0)`` PCG64 stream used by scikit-image's
/// ``medial_axis(..., rng=0)``.  The fixed state is SeedSequence(0)'s public
/// PCG64 initial state; output and cached-u32 ordering match NumPy.
struct NumpyPcg64 {
    state: u128,
    cached_u32: Option<u32>,
}

impl NumpyPcg64 {
    const MULTIPLIER: u128 = 47_026_247_687_942_121_848_144_207_491_837_523_525;
    const INCREMENT: u128 = 87_136_372_517_582_989_555_478_159_403_783_844_777;

    fn seed_zero() -> Self {
        Self {
            state: 35_399_562_948_360_463_058_890_781_895_381_311_971,
            cached_u32: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        let xorshifted = ((self.state >> 64) as u64) ^ self.state as u64;
        xorshifted.rotate_right((self.state >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.cached_u32.take() {
            return value;
        }
        let value = self.next_u64();
        self.cached_u32 = Some((value >> 32) as u32);
        value as u32
    }

    fn inclusive_interval(&mut self, maximum: u32) -> u32 {
        let mut mask = maximum;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= maximum {
                return value;
            }
        }
    }

    fn permutation(count: usize) -> Vec<usize> {
        let mut values: Vec<usize> = (0..count).collect();
        let mut generator = Self::seed_zero();
        for index in (1..count).rev() {
            let other = generator.inclusive_interval(index as u32) as usize;
            values.swap(index, other);
        }
        values
    }
}

fn medial_axis(mask: &[bool], width: usize, height: usize) -> (Vec<bool>, Vec<f32>) {
    let distance = distance_to_background(mask, width, height);
    let keep_table: Vec<bool> = (0_usize..512)
        .map(|pattern| {
            if pattern & (1 << 4) == 0 {
                return false;
            }
            let without_centre = pattern & !(1 << 4);
            pattern.count_ones() < 3
                || pattern_component_count(pattern) != pattern_component_count(without_centre)
        })
        .collect();
    let foreground: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|&(_, &active)| active)
        .map(|(index, _)| index)
        .collect();
    // scikit-image assigns the permutation values to row-major foreground
    // samples, then lexsorts by distance, cornerness, and that tie breaker.
    let tiebreaker = NumpyPcg64::permutation(foreground.len());
    let mut order: Vec<(usize, f32, usize, usize)> = foreground
        .into_iter()
        .zip(tiebreaker)
        .map(|(index, random)| {
            let pattern = pattern_index(mask, width, height, index);
            let cornerness = 9 - pattern.count_ones() as usize;
            (index, distance[index], cornerness, random)
        })
        .collect();
    order.sort_by(|first, second| {
        first
            .1
            .total_cmp(&second.1)
            .then(first.2.cmp(&second.2))
            .then(first.3.cmp(&second.3))
    });
    let mut result = mask.to_vec();
    for (index, _, _, _) in order {
        let pattern = pattern_index(&result, width, height, index);
        if !keep_table[pattern] {
            result[index] = false;
        }
    }
    (result, distance)
}

fn local_maximum(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    (0..values.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let mut maximum = f32::NEG_INFINITY;
            for dy in -(radius as isize)..=radius as isize {
                let py = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                for dx in -(radius as isize)..=radius as isize {
                    let px = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                    maximum = maximum.max(values[py * width + px]);
                }
            }
            maximum
        })
        .collect()
}

fn dark_ridge_support(
    lab: &[Lab],
    width: usize,
    height: usize,
    scale: f32,
) -> (Vec<bool>, Vec<bool>) {
    let lightness: Vec<f32> = lab.iter().map(|value| value.l / 100.0).collect();
    let local_radius = (7.0 * scale).round().max(2.0) as usize;
    let local_light = local_maximum(&lightness, width, height, local_radius);
    let absolute_dark = lightness.iter().map(|&value| value <= 0.35).collect();
    let locally_dark = lightness
        .iter()
        .zip(&local_light)
        .map(|(&value, &maximum)| value <= 0.78 && maximum - value >= 0.10)
        .collect();
    (absolute_dark, locally_dark)
}

fn graph_endpoint(edge: &SourceEdge, at_start: bool) -> ([f64; 2], [f64; 2]) {
    let endpoint = if at_start {
        edge.points[0]
    } else {
        edge.points[edge.points.len() - 1]
    };
    let reach = 5.min(edge.points.len() - 1);
    let interior = if at_start {
        edge.points[reach]
    } else {
        edge.points[edge.points.len() - reach - 1]
    };
    let dx = endpoint[0] - interior[0];
    let dy = endpoint[1] - interior[1];
    let length = dx.hypot(dy).max(1e-8);
    (endpoint, [dx / length, dy / length])
}

fn graph_bridge_supported(
    first: [f64; 2],
    second: [f64; 2],
    support: &[bool],
    width: usize,
    height: usize,
) -> bool {
    let first_x = (first[0] - 0.5).round() as isize;
    let first_y = (first[1] - 0.5).round() as isize;
    let second_x = (second[0] - 0.5).round() as isize;
    let second_y = (second[1] - 0.5).round() as isize;
    let line = raster_line(first_x, first_y, second_x, second_y);
    let mut inside = 0_usize;
    let mut selected = 0_usize;
    for (x, y) in line {
        if x >= width || y >= height {
            continue;
        }
        inside += 1;
        selected += usize::from(support[y * width + x]);
    }
    inside > 0 && selected as f32 / inside as f32 >= 0.80
}

fn merge_source_edges(
    first: &SourceEdge,
    first_at_start: bool,
    second: &SourceEdge,
    second_at_start: bool,
) -> SourceEdge {
    let mut first_points = if first_at_start {
        first.points.iter().rev().copied().collect::<Vec<_>>()
    } else {
        first.points.clone()
    };
    let mut second_points = if second_at_start {
        second.points.clone()
    } else {
        second.points.iter().rev().copied().collect::<Vec<_>>()
    };
    let first_last = first_points.len() - 1;
    let shared = [
        0.5 * (first_points[first_last][0] + second_points[0][0]),
        0.5 * (first_points[first_last][1] + second_points[0][1]),
    ];
    first_points[first_last] = shared;
    second_points[0] = shared;
    first_points.extend_from_slice(&second_points[1..]);
    let mut width_samples = if first.width_samples.is_empty() {
        vec![(first.width, first.points.len().max(1))]
    } else {
        first.width_samples.clone()
    };
    width_samples.extend(if second.width_samples.is_empty() {
        vec![(second.width, second.points.len().max(1))]
    } else {
        second.width_samples.clone()
    });
    width_samples.sort_by(|first, second| first.0.total_cmp(&second.0));
    let middle = 0.5 * width_samples.iter().map(|value| value.1).sum::<usize>() as f32;
    let mut cumulative = 0_usize;
    let mut merged_width = width_samples.last().map(|value| value.0).unwrap_or(1.2);
    for &(width, weight) in &width_samples {
        cumulative += weight;
        if cumulative as f32 >= middle {
            merged_width = width;
            break;
        }
    }
    let role = if first.role == "ridge-on-boundary" || second.role == "ridge-on-boundary" {
        "ridge-on-boundary"
    } else if first.role == "dark-boundary" || second.role == "dark-boundary" {
        "dark-boundary"
    } else {
        "ridge"
    };
    SourceEdge {
        points: first_points,
        width: merged_width,
        role,
        width_samples,
    }
}

fn connect_source_edges(
    edges: &mut Vec<SourceEdge>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_gap: f32,
) {
    let cosine = 15.0_f64.to_radians().cos();
    loop {
        if edges.len() < 2 {
            break;
        }
        let mut candidates = Vec::<(f64, usize, bool, usize, bool)>::new();
        for first in 0..edges.len() {
            for second in first + 1..edges.len() {
                for first_at_start in [true, false] {
                    let (first_point, first_tangent) =
                        graph_endpoint(&edges[first], first_at_start);
                    for second_at_start in [true, false] {
                        let (second_point, second_tangent) =
                            graph_endpoint(&edges[second], second_at_start);
                        let dx = second_point[0] - first_point[0];
                        let dy = second_point[1] - first_point[1];
                        let distance = dx.hypot(dy);
                        if distance > maximum_gap as f64 {
                            continue;
                        }
                        let opposing = -(first_tangent[0] * second_tangent[0]
                            + first_tangent[1] * second_tangent[1]);
                        let coincident_radius =
                            (0.5 * (edges[first].width + edges[second].width) + 0.5).max(2.25);
                        let (first_alignment, second_alignment) = if distance <= coincident_radius {
                            if opposing < cosine {
                                continue;
                            }
                            (1.0, 1.0)
                        } else {
                            let direction_x = dx / distance;
                            let direction_y = dy / distance;
                            let first_alignment =
                                first_tangent[0] * direction_x + first_tangent[1] * direction_y;
                            let second_alignment = -(second_tangent[0] * direction_x
                                + second_tangent[1] * direction_y);
                            if first_alignment.min(second_alignment).min(opposing) < cosine {
                                continue;
                            }
                            (first_alignment, second_alignment)
                        };
                        if !graph_bridge_supported(
                            first_point,
                            second_point,
                            support,
                            width,
                            height,
                        ) {
                            continue;
                        }
                        let score =
                            distance + 2.0 * (3.0 - first_alignment - second_alignment - opposing);
                        candidates.push((score, first, first_at_start, second, second_at_start));
                    }
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|first, second| first.0.total_cmp(&second.0));
        let mut best = std::collections::HashMap::<(usize, bool), (f64, (usize, bool))>::new();
        for &(score, first, first_start, second, second_start) in &candidates {
            for (owner, other) in [
                ((first, first_start), (second, second_start)),
                ((second, second_start), (first, first_start)),
            ] {
                if best
                    .get(&owner)
                    .map(|value| score < value.0)
                    .unwrap_or(true)
                {
                    best.insert(owner, (score, other));
                }
            }
        }
        let mut selected = Vec::new();
        let mut used = vec![false; edges.len()];
        for candidate in candidates {
            let (_, first, first_start, second, second_start) = candidate;
            if !used[first]
                && !used[second]
                && best.get(&(first, first_start)).map(|value| value.1)
                    == Some((second, second_start))
                && best.get(&(second, second_start)).map(|value| value.1)
                    == Some((first, first_start))
            {
                used[first] = true;
                used[second] = true;
                selected.push((first, first_start, second, second_start));
            }
        }
        if selected.is_empty() {
            break;
        }
        let mut retained: Vec<SourceEdge> = edges
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(_, edge)| edge.clone())
            .collect();
        for (first, first_start, second, second_start) in selected {
            retained.push(merge_source_edges(
                &edges[first],
                first_start,
                &edges[second],
                second_start,
            ));
        }
        *edges = retained;
    }
}

fn profiles_for_points(
    points: &[[f64; 2]],
    field: &EdgeField,
    lab: &[Lab],
    width: usize,
    height: usize,
    radius: f32,
) -> Vec<ProfileClassification> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut angles = Vec::<f64>::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let u = bilinear(
            &field.tangent_u,
            width,
            height,
            (point[0] - 0.5) as f32,
            (point[1] - 0.5) as f32,
        );
        let v = bilinear(
            &field.tangent_v,
            width,
            height,
            (point[0] - 0.5) as f32,
            (point[1] - 0.5) as f32,
        );
        let theta = 0.5 * v.atan2(u);
        let mut angle = (theta.sin() as f64).atan2(theta.cos() as f64);
        let previous = points[index.saturating_sub(1)];
        let following = points[(index + 1).min(points.len() - 1)];
        if angle.cos() * (following[0] - previous[0]) + angle.sin() * (following[1] - previous[1])
            < 0.0
        {
            angle += std::f64::consts::PI;
        }
        if let Some(&previous_angle) = angles.last() {
            while angle - previous_angle > std::f64::consts::PI {
                angle -= 2.0 * std::f64::consts::PI;
            }
            while angle - previous_angle < -std::f64::consts::PI {
                angle += 2.0 * std::f64::consts::PI;
            }
        }
        angles.push(angle);
    }
    if angles.len() >= 3 {
        let kernel = gaussian_kernel_f64(0.8);
        let kernel_radius = kernel.len() / 2;
        let original = angles.clone();
        for (index, target) in angles.iter_mut().enumerate() {
            *target = kernel
                .iter()
                .enumerate()
                .map(|(offset, &weight)| {
                    let source = (index as isize + offset as isize - kernel_radius as isize)
                        .clamp(0, original.len().saturating_sub(1) as isize)
                        as usize;
                    original[source] * weight
                })
                .sum();
        }
    }
    let step = 0.25_f32;
    let sample_count = (2.0 * radius / step).round() as usize + 1;
    let offsets: Vec<f32> = (0..sample_count)
        .map(|index| -radius + index as f32 * step)
        .collect();
    points
        .iter()
        .zip(angles)
        .map(|(point, angle)| {
            let nx = -angle.sin();
            let ny = angle.cos();
            let samples: Vec<[f32; 3]> = offsets
                .iter()
                .map(|offset| {
                    normalized_lab_sample(
                        lab,
                        width,
                        height,
                        (point[0] - 0.5 + *offset as f64 * nx) as f32,
                        (point[1] - 0.5 + *offset as f64 * ny) as f32,
                    )
                })
                .collect();
            classify_profile(&samples, &offsets, radius)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn add_supported_medial_ridges(
    image: &Raster,
    lab: &[Lab],
    field: &EdgeField,
    absolute_dark: &[bool],
    locally_dark: &[bool],
    scale: f32,
    face_barrier: &mut [bool],
    raw_graph: &mut Vec<SourceEdge>,
) -> (usize, usize) {
    let width = image.width;
    let height = image.height;
    let absolute_near = dilate(absolute_dark, width, height, 1);
    let relative_extension: Vec<bool> = locally_dark
        .iter()
        .zip(&absolute_near)
        .map(|(&relative, &near)| relative && !near)
        .collect();
    let radius = 7.0 * scale;
    let mut candidate_count = 0_usize;
    let mut supported_count = 0_usize;
    for (candidate_index, (mut candidate, maximum_width)) in [
        (absolute_dark.to_vec(), 6.0 * scale),
        (relative_extension, 4.0 * scale),
    ]
    .into_iter()
    .enumerate()
    {
        remove_tiny_components(&mut candidate, width, height, 2);
        let (mut medial, distance) = medial_axis(&candidate, width, height);
        if let Some(prefix) = std::env::var_os("PICVEC_EDGE_DIAGNOSTICS") {
            let prefix = prefix.to_string_lossy();
            for (name, mask) in [("candidate", &candidate), ("medial", &medial)] {
                let raster = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
                    image::Luma([if mask[y as usize * width + x as usize] {
                        255
                    } else {
                        0
                    }])
                });
                let _ = raster.save(format!("{prefix}-{name}-{candidate_index}.png"));
            }
        }
        for index in 0..medial.len() {
            medial[index] &= distance[index] <= 0.5 * maximum_width;
        }
        for chain in trace_edge_chains(&medial, width, height) {
            if chain.len() < 3 {
                continue;
            }
            let points: Vec<(f64, f64)> = chain
                .iter()
                .map(|&index| ((index % width) as f64 + 0.5, (index / width) as f64 + 0.5))
                .collect();
            let chain_length: f64 = points
                .windows(2)
                .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                .sum();
            let mut measured_widths: Vec<f32> =
                chain.iter().map(|&index| 2.0 * distance[index]).collect();
            measured_widths.sort_by(f32::total_cmp);
            let fallback_width = median_sorted(&measured_widths).max(1.2) as f64;
            if !(0.8..=maximum_width as f64).contains(&fallback_width)
                || chain_length < (3.0 * fallback_width).max((2.0 * scale) as f64)
            {
                continue;
            }
            candidate_count += 1;
            let mut angles = Vec::<f64>::with_capacity(points.len());
            for (index, &(x, y)) in points.iter().enumerate() {
                let u = bilinear(
                    &field.tangent_u,
                    width,
                    height,
                    (x - 0.5) as f32,
                    (y - 0.5) as f32,
                );
                let v = bilinear(
                    &field.tangent_v,
                    width,
                    height,
                    (x - 0.5) as f32,
                    (y - 0.5) as f32,
                );
                let theta = 0.5 * v.atan2(u);
                let mut angle = (theta.sin() as f64).atan2(theta.cos() as f64);
                let previous = points[index.saturating_sub(1)];
                let following = points[(index + 1).min(points.len() - 1)];
                if angle.cos() * (following.0 - previous.0)
                    + angle.sin() * (following.1 - previous.1)
                    < 0.0
                {
                    angle += std::f64::consts::PI;
                }
                if let Some(&previous_angle) = angles.last() {
                    while angle - previous_angle > std::f64::consts::PI {
                        angle -= 2.0 * std::f64::consts::PI;
                    }
                    while angle - previous_angle < -std::f64::consts::PI {
                        angle += 2.0 * std::f64::consts::PI;
                    }
                }
                angles.push(angle);
            }
            if angles.len() >= 3 {
                let kernel = gaussian_kernel_f64(0.8);
                let kernel_radius = kernel.len() / 2;
                let original = angles.clone();
                for (index, target) in angles.iter_mut().enumerate() {
                    *target = kernel
                        .iter()
                        .enumerate()
                        .map(|(offset, &weight)| {
                            let source = (index as isize + offset as isize - kernel_radius as isize)
                                .clamp(0, original.len().saturating_sub(1) as isize)
                                as usize;
                            original[source] * weight
                        })
                        .sum();
                }
            }
            let step = 0.25_f32;
            let sample_count = (2.0 * radius / step).round() as usize + 1;
            let offsets: Vec<f32> = (0..sample_count)
                .map(|index| -radius + index as f32 * step)
                .collect();
            let profiles: Vec<(ProfileClassification, f64, f64)> = points
                .iter()
                .zip(&angles)
                .map(|(&(x, y), &angle)| {
                    let nx = -angle.sin();
                    let ny = angle.cos();
                    let samples: Vec<[f32; 3]> = offsets
                        .iter()
                        .map(|offset| {
                            normalized_lab_sample(
                                lab,
                                width,
                                height,
                                (x - 0.5 + *offset as f64 * nx) as f32,
                                (y - 0.5 + *offset as f64 * ny) as f32,
                            )
                        })
                        .collect();
                    (classify_profile(&samples, &offsets, radius), nx, ny)
                })
                .collect();
            let mut valid: Vec<bool> = profiles
                .iter()
                .map(|(profile, _, _)| {
                    matches!(
                        profile.role,
                        ProfileRole::Ridge | ProfileRole::RidgeOnBoundary
                    ) && profile.confidence >= 0.8
                        && profile.width.is_finite()
                })
                .collect();
            let cumulative: Vec<f64> = std::iter::once(0.0)
                .chain(points.windows(2).scan(0.0, |total, pair| {
                    *total += (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1);
                    Some(*total)
                }))
                .collect();
            let mut index = 0_usize;
            while index < valid.len() {
                if valid[index] {
                    index += 1;
                    continue;
                }
                let first = index;
                while index < valid.len() && !valid[index] {
                    index += 1;
                }
                if first > 0
                    && index < valid.len()
                    && cumulative[index] - cumulative[first - 1] <= (2.0 * scale) as f64
                {
                    valid[first..index].fill(true);
                }
            }
            let mut index = 0_usize;
            while index < valid.len() {
                if !valid[index] {
                    index += 1;
                    continue;
                }
                let first = index;
                while index + 1 < valid.len() && valid[index + 1] {
                    index += 1;
                }
                let last = index;
                index += 1;
                if last + 1 - first < 3 {
                    continue;
                }
                let finite_widths: Vec<f64> = profiles[first..=last]
                    .iter()
                    .map(|value| value.0.width)
                    .filter(|value| value.is_finite())
                    .collect();
                if finite_widths.is_empty() {
                    continue;
                }
                let mut finite_widths = finite_widths;
                finite_widths.sort_by(f64::total_cmp);
                let ownership_width = {
                    let middle = finite_widths.len() / 2;
                    let median = if finite_widths.len().is_multiple_of(2) {
                        0.5 * (finite_widths[middle - 1] + finite_widths[middle])
                    } else {
                        finite_widths[middle]
                    };
                    median.min(fallback_width).clamp(0.8, (6.0 * scale) as f64)
                };
                let mut run_points = Vec::<[f64; 2]>::with_capacity(last + 1 - first);
                let mut centre_offsets: Vec<f64> = profiles[first..=last]
                    .iter()
                    .map(|value| value.0.centre)
                    .collect();
                if !interpolate_nonfinite(&mut centre_offsets) {
                    continue;
                }
                if centre_offsets.len() >= 3 {
                    let kernel = gaussian_kernel_f64(0.8);
                    let kernel_radius = kernel.len() / 2;
                    let original = centre_offsets.clone();
                    for (offset_index, target) in centre_offsets.iter_mut().enumerate() {
                        *target = kernel
                            .iter()
                            .enumerate()
                            .map(|(kernel_index, &weight)| {
                                let source = (offset_index as isize + kernel_index as isize
                                    - kernel_radius as isize)
                                    .clamp(0, original.len().saturating_sub(1) as isize)
                                    as usize;
                                original[source] * weight
                            })
                            .sum();
                    }
                }
                for (offset_index, local) in (first..=last).enumerate() {
                    let (x, y) = points[local];
                    let centre_x = x + centre_offsets[offset_index] * profiles[local].1;
                    let centre_y = y + centre_offsets[offset_index] * profiles[local].2;
                    run_points.push([centre_x, centre_y]);
                }
                let shifted_length: f64 = run_points
                    .windows(2)
                    .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                    .sum();
                if shifted_length < (3.0 * ownership_width).max((2.0 * scale) as f64) {
                    continue;
                }
                // The centre correction changes the two normal exits at a
                // contact. Python therefore classifies the corrected graph a
                // second time before assigning visible-vs-boundary ownership.
                let shifted_profiles =
                    profiles_for_points(&run_points, field, lab, width, height, radius);
                let ridge_on_boundary_count = shifted_profiles
                    .iter()
                    .filter(|profile| profile.role == ProfileRole::RidgeOnBoundary)
                    .count();
                let graph_is_boundary = ridge_on_boundary_count > shifted_profiles.len() / 2;
                for (local, profile) in (first..=last).zip(&shifted_profiles) {
                    if profile.role == ProfileRole::RidgeOnBoundary {
                        face_barrier[chain[local]] = true;
                    }
                }
                raw_graph.push(SourceEdge {
                    points: run_points,
                    width: ownership_width,
                    role: if graph_is_boundary {
                        "ridge-on-boundary"
                    } else {
                        "ridge"
                    },
                    width_samples: Vec::new(),
                });
                supported_count += 1;
            }
        }
    }
    (candidate_count, supported_count)
}

fn point_tangents(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let reach = points.len().saturating_sub(1).min(3);
    (0..points.len())
        .map(|index| {
            let before = points[index.saturating_sub(reach)];
            let after = points[(index + reach).min(points.len() - 1)];
            let dx = after[0] - before[0];
            let dy = after[1] - before[1];
            let length = dx.hypot(dy).max(1e-8);
            [dx / length, dy / length]
        })
        .collect()
}

fn remove_profile_overlap(candidates: Vec<SourceEdge>, owners: &[SourceEdge]) -> Vec<SourceEdge> {
    if owners.is_empty() {
        return candidates;
    }
    let owner_samples: Vec<([f64; 2], [f64; 2])> = owners
        .iter()
        .flat_map(|edge| {
            let tangents = point_tangents(&edge.points);
            edge.points
                .iter()
                .copied()
                .zip(tangents)
                .collect::<Vec<_>>()
        })
        .collect();
    let cosine = 20.0_f64.to_radians().cos();
    let mut result = Vec::new();
    for candidate in candidates {
        if candidate.points.len() < 2 {
            continue;
        }
        let tangents = point_tangents(&candidate.points);
        let keep: Vec<bool> = candidate
            .points
            .iter()
            .zip(&tangents)
            .map(|(point, tangent)| {
                !owner_samples.iter().any(|(owner_point, owner_tangent)| {
                    (point[0] - owner_point[0]).hypot(point[1] - owner_point[1]) <= 1.5
                        && (tangent[0] * owner_tangent[0] + tangent[1] * owner_tangent[1]).abs()
                            >= cosine
                })
            })
            .collect();
        let mut index = 0_usize;
        while index < keep.len() {
            if !keep[index] {
                index += 1;
                continue;
            }
            let mut first = index;
            while index + 1 < keep.len() && keep[index + 1] {
                index += 1;
            }
            let mut last = index;
            index += 1;
            first = first.saturating_sub(1);
            if last + 1 < candidate.points.len() {
                last += 1;
            }
            if last > first {
                result.push(SourceEdge {
                    points: candidate.points[first..=last].to_vec(),
                    width: candidate.width,
                    role: candidate.role,
                    width_samples: candidate.width_samples.clone(),
                });
            }
        }
    }
    result
}

fn nonoverlapping_extensions(
    mut candidates: Vec<SourceEdge>,
    owners: &[SourceEdge],
) -> Vec<SourceEdge> {
    candidates.sort_by(|first, second| {
        let length = |edge: &SourceEdge| {
            edge.points
                .windows(2)
                .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                .sum::<f64>()
        };
        length(second).total_cmp(&length(first))
    });
    let mut accepted = owners.to_vec();
    let mut extensions = Vec::new();
    for candidate in candidates {
        let remaining = remove_profile_overlap(vec![candidate], &accepted);
        accepted.extend(remaining.iter().cloned());
        extensions.extend(remaining);
    }
    extensions
}

#[allow(clippy::too_many_arguments)]
fn add_profile_supported_ridges(
    classified_chains: &[ClassifiedChain],
    lab: &[Lab],
    absolute_dark: &[bool],
    locally_dark: &[bool],
    width: usize,
    height: usize,
    scale: f32,
    raw_ridge_graph: &mut Vec<SourceEdge>,
    dark_boundary_candidates: &mut Vec<SourceEdge>,
) -> (usize, usize) {
    let mut profile_ridge_candidates = Vec::<SourceEdge>::new();
    for classified_chain in classified_chains {
        if classified_chain.pixels.len() < 3 {
            continue;
        }
        let chain = &classified_chain.pixels;
        let points = &classified_chain.points;
        let normals = &classified_chain.normals;
        let profiles = &classified_chain.profiles;
        let mut valid = vec![false; chain.len()];
        for index in 0..chain.len() {
            let profile = profiles[index];
            if !matches!(
                profile.role,
                ProfileRole::Ridge | ProfileRole::RidgeOnBoundary
            ) || !profile.width.is_finite()
            {
                continue;
            }
            let provisional_x = points[index][0] + profile.centre * normals[index][0];
            let provisional_y = points[index][1] + profile.centre * normals[index][1];
            let x = (provisional_x - 0.5)
                .round_ties_even()
                .clamp(0.0, width.saturating_sub(1) as f64) as usize;
            let y = (provisional_y - 0.5)
                .round_ties_even()
                .clamp(0.0, height.saturating_sub(1) as f64) as usize;
            let support_index = y * width + x;
            let palette_supported = profile.confidence >= 0.8
                && ((absolute_dark[support_index] && profile.width <= (6.0 * scale) as f64)
                    || (locally_dark[support_index]
                        && !absolute_dark[support_index]
                        && profile.width <= (4.0 * scale) as f64));
            let bounded_dark = profile.confidence >= 0.65
                && profile.dark_contrast >= 0.06
                && profile.width <= (4.0 * scale) as f64;
            valid[index] = palette_supported || bounded_dark;
        }
        let cumulative: Vec<f64> = std::iter::once(0.0)
            .chain(points.windows(2).scan(0.0, |total, pair| {
                *total += (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
                Some(*total)
            }))
            .collect();
        let mut index = 0_usize;
        while index < valid.len() {
            if valid[index] {
                index += 1;
                continue;
            }
            let first = index;
            while index < valid.len() && !valid[index] {
                index += 1;
            }
            if first > 0
                && index < valid.len()
                && cumulative[index] - cumulative[first - 1] <= (2.0 * scale) as f64
            {
                valid[first..index].fill(true);
            }
        }
        let mut index = 0_usize;
        while index < valid.len() {
            if !valid[index] {
                index += 1;
                continue;
            }
            let first = index;
            while index + 1 < valid.len() && valid[index + 1] {
                index += 1;
            }
            let last = index;
            index += 1;
            if last + 1 - first < 3 {
                continue;
            }
            let mut offsets: Vec<f64> = profiles[first..=last]
                .iter()
                .map(|profile| profile.centre)
                .collect();
            let mut local_widths: Vec<f64> = profiles[first..=last]
                .iter()
                .map(|profile| profile.width)
                .collect();
            if !interpolate_nonfinite(&mut offsets) || !interpolate_nonfinite(&mut local_widths) {
                continue;
            }
            let mut sorted_widths = local_widths.clone();
            sorted_widths.sort_by(f64::total_cmp);
            let middle = sorted_widths.len() / 2;
            let measured_width = if sorted_widths.len().is_multiple_of(2) {
                0.5 * (sorted_widths[middle - 1] + sorted_widths[middle])
            } else {
                sorted_widths[middle]
            };
            if offsets.len() >= 3 {
                let kernel = gaussian_kernel_f64(0.8);
                let radius = kernel.len() / 2;
                let original = offsets.clone();
                for (index, target) in offsets.iter_mut().enumerate() {
                    *target = kernel
                        .iter()
                        .enumerate()
                        .map(|(offset, &weight)| {
                            let source = (index as isize + offset as isize - radius as isize)
                                .clamp(0, original.len().saturating_sub(1) as isize)
                                as usize;
                            original[source] * weight
                        })
                        .sum();
                }
            }
            let run_points: Vec<[f64; 2]> = (first..=last)
                .enumerate()
                .map(|(local, source)| {
                    [
                        points[source][0] + offsets[local] * normals[source][0],
                        points[source][1] + offsets[local] * normals[source][1],
                    ]
                })
                .collect();
            let run_length: f64 = run_points
                .windows(2)
                .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                .sum();
            if run_length < (3.0 * measured_width).max((2.0 * scale) as f64) {
                continue;
            }
            let boundary_count = profiles[first..=last]
                .iter()
                .filter(|profile| profile.role == ProfileRole::RidgeOnBoundary)
                .count();
            let is_boundary = boundary_count > (last + 1 - first) / 2;
            profile_ridge_candidates.push(SourceEdge {
                points: run_points,
                width: measured_width,
                role: if is_boundary {
                    "ridge-on-boundary"
                } else {
                    "ridge"
                },
                width_samples: Vec::new(),
            });
        }

        // Faithful port of dark_boundary_strokes(): a hard step only owns
        // an overlay when a sufficiently dark, source-supported contour can
        // be inset into its dark side.  A generic dark boundary pixel is not
        // itself structural ink.
        let mut valid = vec![false; chain.len()];
        let mut inset_points = points.clone();
        for index in 0..chain.len() {
            let profile = profiles[index];
            if profile.role != ProfileRole::Boundary || profile.confidence < 0.8 {
                continue;
            }
            let normal = normals[index];
            let sample_x = points[index][0] - 0.5;
            let sample_y = points[index][1] - 0.5;
            let minus = normalized_lab_sample(
                lab,
                width,
                height,
                (sample_x - 1.25 * normal[0]) as f32,
                (sample_y - 1.25 * normal[1]) as f32,
            );
            let plus = normalized_lab_sample(
                lab,
                width,
                height,
                (sample_x + 1.25 * normal[0]) as f32,
                (sample_y + 1.25 * normal[1]) as f32,
            );
            let (dark, light, sign) = if minus[0] <= plus[0] {
                (minus, plus, -1.0_f64)
            } else {
                (plus, minus, 1.0_f64)
            };
            if dark[0] > 0.35 || lab_vector_distance(light, dark) < 0.06 {
                continue;
            }
            valid[index] = true;
            let inset = 0.5 + 0.5 * 1.2;
            inset_points[index] = [
                points[index][0] + sign * inset * normal[0],
                points[index][1] + sign * inset * normal[1],
            ];
        }
        let mut index = 0_usize;
        while index < valid.len() {
            if !valid[index] {
                index += 1;
                continue;
            }
            let first = index;
            while index + 1 < valid.len() && valid[index + 1] {
                index += 1;
            }
            let last = index;
            index += 1;
            if last + 1 - first < 3 {
                continue;
            }
            let length: f64 = inset_points[first..=last]
                .windows(2)
                .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                .sum();
            if length < 3.6 {
                continue;
            }
            dark_boundary_candidates.push(SourceEdge {
                points: inset_points[first..=last].to_vec(),
                width: 1.2,
                role: "dark-boundary",
                width_samples: Vec::new(),
            });
        }
    }
    let candidate_count = profile_ridge_candidates.len();
    let extensions = nonoverlapping_extensions(profile_ridge_candidates, raw_ridge_graph);
    let extension_count = extensions.len();
    raw_ridge_graph.extend(extensions);
    (candidate_count, extension_count)
}

fn classify_normal_profile_edges(image: &Raster) -> EdgeRoles {
    let width = image.width;
    let height = image.height;
    let lab = lab_pixels(image);
    let field = estimate_profile_edge_field(image, &lab);
    if let Some(prefix) = std::env::var_os("PICVEC_EDGE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        let write_f32 = |name: &str, values: &[f32]| {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
            for &value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let _ = std::fs::write(format!("{prefix}-{name}.f32le"), bytes);
        };
        write_f32("field-u", &field.tangent_u);
        write_f32("field-v", &field.tangent_v);
        write_f32("field-strength", &field.strength);
        write_f32("field-edge", &field.edge);
        let mut normalized_lab = Vec::with_capacity(lab.len() * 3);
        for value in &lab {
            normalized_lab.extend([value.l / 100.0, value.a / 128.0, value.b / 128.0]);
        }
        write_f32("lab", &normalized_lab);
    }
    let nms = oriented_nonmaximum(&field, width, height);
    let skeleton = edge_hysteresis(&nms, &field, width, height);
    if let Some(prefix) = std::env::var_os("PICVEC_EDGE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        let mut nms_bytes = Vec::with_capacity(nms.len() * std::mem::size_of::<f32>());
        for &value in &nms {
            nms_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let _ = std::fs::write(format!("{prefix}-nms.f32le"), nms_bytes);
        let raster = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
            image::Luma([if skeleton[y as usize * width + x as usize] {
                255
            } else {
                0
            }])
        });
        let _ = raster.save(format!("{prefix}-skeleton.png"));
    }
    let scale = (width.max(height) as f32 / 1024.0).max(1.0);
    let radius = 7.0 * scale;
    let classified_chains =
        classify_skeleton_chains(&skeleton, &field, &lab, width, height, radius);
    let edge_chain_count = classified_chains.len();
    let edge_chain_samples = classified_chains
        .iter()
        .filter(|chain| chain.pixels.len() >= 3)
        .map(|chain| chain.pixels.len())
        .sum();
    let mut boundary = vec![false; image.pixels.len()];
    let mut shading = vec![false; image.pixels.len()];
    let mut face_barrier = vec![false; image.pixels.len()];
    let mut ridge_role_pixels = 0_usize;
    let mut ridge_on_boundary_pixels = 0_usize;
    let mut unknown_pixels = 0_usize;
    for chain in &classified_chains {
        for (&index, &profile) in chain.pixels.iter().zip(&chain.profiles) {
            match profile.role {
                ProfileRole::Boundary => {
                    boundary[index] = true;
                    face_barrier[index] = true;
                }
                ProfileRole::RidgeOnBoundary => {
                    ridge_on_boundary_pixels += 1;
                    face_barrier[index] = true;
                }
                ProfileRole::Ridge => ridge_role_pixels += 1,
                ProfileRole::Shading => shading[index] = true,
                ProfileRole::Unknown => unknown_pixels += 1,
            }
        }
    }

    // Construct graph ownership in the same order as the Python source:
    // supported medial ribbons, non-overlapping profile extensions, then a
    // narrow visible graph and a separate underpaint-preserving dark contour
    // graph.  Raster masks are derived only after graph joining.
    let (absolute_dark, locally_dark) = dark_ridge_support(&lab, width, height, scale);
    let mut raw_ridge_graph = Vec::<SourceEdge>::new();
    let (dark_ridge_candidates, dark_ridge_supported) = add_supported_medial_ridges(
        image,
        &lab,
        &field,
        &absolute_dark,
        &locally_dark,
        scale,
        &mut face_barrier,
        &mut raw_ridge_graph,
    );
    let mut profile_dark_boundary_candidates = Vec::<SourceEdge>::new();
    let (profile_ridge_candidates, profile_ridge_extensions) = add_profile_supported_ridges(
        &classified_chains,
        &lab,
        &absolute_dark,
        &locally_dark,
        width,
        height,
        scale,
        &mut raw_ridge_graph,
        &mut profile_dark_boundary_candidates,
    );
    let ownership_width = 4.0 * scale;
    let mut visible_ridge_graph = Vec::<SourceEdge>::new();
    let mut wide_dark_outline_graph = Vec::<SourceEdge>::new();
    for graph in raw_ridge_graph {
        if graph.role == "ridge" && graph.width <= ownership_width as f64 {
            visible_ridge_graph.push(graph);
        } else {
            wide_dark_outline_graph.push(graph);
        }
    }
    let visible_ridge_graph_edges_before_join = visible_ridge_graph.len();
    let wide_dark_outline_graph_edges_before_join = wide_dark_outline_graph.len();
    let dark_boundary_candidates = profile_dark_boundary_candidates.len();
    let support: Vec<bool> = absolute_dark
        .iter()
        .zip(&locally_dark)
        .map(|(&absolute, &relative)| absolute || relative)
        .collect();
    connect_source_edges(
        &mut visible_ridge_graph,
        &support,
        width,
        height,
        6.0 * scale,
    );
    wide_dark_outline_graph.extend(profile_dark_boundary_candidates);
    let mut dark_boundary_graph =
        nonoverlapping_extensions(wide_dark_outline_graph, &visible_ridge_graph);
    connect_source_edges(
        &mut dark_boundary_graph,
        &support,
        width,
        height,
        3.0_f32.min(6.0 * scale),
    );
    let ridge_centres = rasterize_source_graph(&visible_ridge_graph, width, height, false);
    let ridge_coverage = rasterize_source_graph(&visible_ridge_graph, width, height, true);
    let dark_boundary = rasterize_source_graph(&dark_boundary_graph, width, height, true);
    let graph_length = |graph: &[SourceEdge]| {
        graph
            .iter()
            .flat_map(|edge| edge.points.windows(2))
            .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
            .sum::<f64>() as f32
    };
    let median_width = |graph: &[SourceEdge]| {
        let mut widths: Vec<f64> = graph.iter().map(|edge| edge.width).collect();
        widths.sort_by(f64::total_cmp);
        widths.get(widths.len() / 2).copied().unwrap_or(0.0) as f32
    };
    let gradient_threshold = percentile(
        nms.iter().copied().filter(|value| *value > 0.0).collect(),
        0.90,
    );
    let summary = EdgeSummary {
        skeleton_pixels: skeleton.iter().filter(|&&value| value).count(),
        edge_chain_count,
        edge_chain_samples,
        boundary_pixels: boundary.iter().filter(|&&value| value).count(),
        ridge_role_pixels,
        ridge_on_boundary_pixels,
        unknown_pixels,
        visible_ridge_pixels: ridge_centres.iter().filter(|&&value| value).count(),
        visible_ridge_coverage_pixels: ridge_coverage.iter().filter(|&&value| value).count(),
        dark_boundary_pixels: dark_boundary.iter().filter(|&&value| value).count(),
        shading_pixels: shading.iter().filter(|&&value| value).count(),
        face_barrier_pixels: face_barrier.iter().filter(|&&value| value).count(),
        visible_ridge_graph_edges: visible_ridge_graph.len(),
        visible_ridge_graph_edges_before_join,
        dark_boundary_graph_edges: dark_boundary_graph.len(),
        wide_dark_outline_graph_edges_before_join,
        dark_boundary_candidates,
        dark_ridge_candidates,
        dark_ridge_supported,
        profile_ridge_candidates,
        profile_ridge_extensions,
        visible_ridge_graph_length: graph_length(&visible_ridge_graph),
        visible_ridge_median_width: median_width(&visible_ridge_graph),
        visible_ridge_width_weighted_length: visible_ridge_graph
            .iter()
            .map(|edge| {
                edge.width
                    * edge
                        .points
                        .windows(2)
                        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                        .sum::<f64>()
            })
            .sum::<f64>() as f32,
        dark_boundary_graph_length: graph_length(&dark_boundary_graph),
        dark_boundary_median_width: median_width(&dark_boundary_graph),
        gradient_threshold,
        ridge_threshold: 0.65,
    };
    EdgeRoles {
        width,
        height,
        boundary,
        visible_ridge_centres: ridge_centres,
        visible_ridge_coverage: ridge_coverage,
        dark_boundary,
        shading,
        face_barrier,
        gradient: field.edge,
        visible_ridge_graph,
        dark_boundary_graph,
        summary,
    }
}

pub fn classify(image: &Raster) -> EdgeRoles {
    let profile = classify_normal_profile_edges(image);
    if let Some(prefix) = std::env::var_os("PICVEC_EDGE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        for (name, mask) in [
            ("boundary", &profile.boundary),
            ("ridge-centres", &profile.visible_ridge_centres),
            ("ridge-coverage", &profile.visible_ridge_coverage),
            ("dark-boundary", &profile.dark_boundary),
            ("face-barrier", &profile.face_barrier),
            ("shading", &profile.shading),
        ] {
            let raster =
                image::GrayImage::from_fn(profile.width as u32, profile.height as u32, |x, y| {
                    image::Luma([if mask[y as usize * profile.width + x as usize] {
                        255
                    } else {
                        0
                    }])
                });
            let _ = raster.save(format!("{prefix}-{name}.png"));
        }
        if let Ok(graphs) = serde_json::to_string_pretty(&serde_json::json!({
            "visible": &profile.visible_ridge_graph,
            "dark": &profile.dark_boundary_graph,
        })) {
            let _ = std::fs::write(format!("{prefix}-graphs.json"), graphs);
        }
    }
    // Degenerate or nearly flat images can leave the normalized edge field
    // without a useful high-threshold seed.  Preserve the previous local
    // classifier only as a defensive fallback, never as a mixed owner map.
    if profile.summary.face_barrier_pixels == 0
        && profile.summary.visible_ridge_coverage_pixels == 0
    {
        classify_legacy(image)
    } else {
        profile
    }
}

fn adaptive_tolerance(lightness: f32, config: &Config) -> f32 {
    let amount = ((lightness - config.dark_knee_lstar) / (100.0 - config.dark_knee_lstar).max(1.0))
        .clamp(0.0, 1.0);
    let smooth = amount * amount * (3.0 - 2.0 * amount);
    config.smoothing_dark_delta_e
        + (config.smoothing_light_delta_e - config.smoothing_dark_delta_e) * smooth
}

/// Small-radius bilateral smoothing that never averages through a strong
/// perceptual edge.  The implementation is parallel by output row/pixel.
pub fn perceptual_smooth(image: &Raster, config: &Config) -> Raster {
    let radius = config.smoothing_radius as isize;
    if radius == 0 {
        return image.clone();
    }
    let lab = lab_pixels(image);
    let sigma = config.smoothing_spatial_sigma.max(0.1);
    let pixels: Vec<[f32; 3]> = (0..image.pixels.len())
        .into_par_iter()
        .map(|index| {
            let x = (index % image.width) as isize;
            let y = (index / image.width) as isize;
            let centre = lab[index];
            let mut sum = [0.0_f32; 3];
            let mut weight_sum = 0.0_f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = (x + dx).clamp(0, image.width as isize - 1) as usize;
                    let py = (y + dy).clamp(0, image.height as isize - 1) as usize;
                    let sample_index = py * image.width + px;
                    let sample = lab[sample_index];
                    let distance = delta_e2000(centre, sample);
                    let threshold =
                        adaptive_tolerance(0.5 * (centre.l + sample.l), config).max(1e-3);
                    let spatial = (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp();
                    let range = (-(distance * distance) / (2.0 * threshold * threshold)).exp();
                    let weight = spatial * range;
                    sum[0] += sample.l * weight;
                    sum[1] += sample.a * weight;
                    sum[2] += sample.b * weight;
                    weight_sum += weight;
                }
            }
            if weight_sum <= 1e-8 {
                image.pixels[index]
            } else {
                lab_to_rgb(Lab {
                    l: sum[0] / weight_sum,
                    a: sum[1] / weight_sum,
                    b: sum[2] / weight_sum,
                })
            }
        })
        .collect();
    Raster::new(image.width, image.height, pixels)
}

#[cfg(test)]
mod tests {
    use super::NumpyPcg64;

    #[test]
    fn numpy_seed_zero_permutation_matches_reference() {
        assert_eq!(
            NumpyPcg64::permutation(10),
            vec![4, 6, 2, 7, 3, 5, 9, 0, 8, 1]
        );
    }
}
