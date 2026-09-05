use std::collections::{HashMap, HashSet, VecDeque};

use rayon::prelude::*;
use serde::Serialize;

use crate::color::{delta_e2000, delta_e76, rgb_to_lab, Lab};
use crate::edge::{dilate, dilate_square, erode, lab_pixels, EdgeRoles};
use crate::geometry::{fitted_structural_open_path_data_with_tangents, Point};
use crate::raster::{percentile, Raster};

#[path = "stroke_model.rs"]
mod stroke_model;

#[derive(Clone, Debug)]
pub struct StructuralStroke {
    pub points: Vec<Point>,
    pub path_data: Option<String>,
    precise_points: Option<Vec<[f64; 2]>>,
    pub color: [f32; 3],
    pub width: f32,
    pub role: &'static str,
    width_samples: Vec<(f32, usize)>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StructuralSummary {
    pub source_coverage_pixels: usize,
    pub source_line_pixels: usize,
    pub skeleton_pixels: usize,
    pub stroke_count: usize,
    pub underpainted_pixels: usize,
    pub antialias_unmixed_pixels: usize,
    pub silhouette_fill_count: usize,
    pub residual_legacy_strokes: usize,
    pub visible_ridge_strokes: usize,
    pub boundary_profile_strokes: usize,
    pub recovered_boundary_strokes: usize,
    pub recovered_alpha_boundary_strokes: usize,
}

#[derive(Clone, Debug)]
pub struct StructuralInk {
    pub strokes: Vec<StructuralStroke>,
    /// Medial-ridge core plus source-modelled AA shoulders transferred out of Paint.
    pub paint_ownership_mask: Vec<bool>,
    /// Role-filtered raster lines frozen during Paint regularization. This is
    /// narrower than the complete set of source graph candidates.
    pub source_line_mask: Vec<bool>,
    /// Legacy analyser line ownership before edge-role suppression.  The
    /// residual selector intersects its result with this mask before tracing
    /// the single authoritative residual source-line graph.
    legacy_line_mask: Vec<bool>,
    /// Edge-role-owned line raster (silhouette faces excluded).
    role_line_mask: Vec<bool>,
    visible_ridge_coverage: Vec<bool>,
    dark_boundary_coverage: Vec<bool>,
    face_barrier: Vec<bool>,
    pub(crate) boundary_stroke_mask: Vec<bool>,
    pub summary: StructuralSummary,
}

impl StructuralInk {
    /// Alpha owns the silhouette independently of straight-colour analysis.
    /// Reject a centreline without covered source support: its finite width
    /// could otherwise overlap the fitted mask despite being entirely outside.
    pub(crate) fn retain_source_alpha_supported_strokes(
        &mut self,
        source: &Raster,
        matte: &crate::chroma::AlphaMatte,
    ) {
        self.retain_strokes(|stroke| {
            stroke.points.windows(2).any(|pair| {
                let steps = pair[0].distance(pair[1]).ceil().max(1.0) as usize;
                (0..=steps).any(|step| {
                    let t = step as f32 / steps as f32;
                    let x = (pair[0].x + t * (pair[1].x - pair[0].x) - 0.5)
                        .clamp(0.0, (source.width - 1) as f32);
                    let y = (pair[0].y + t * (pair[1].y - pair[0].y) - 0.5)
                        .clamp(0.0, (source.height - 1) as f32);
                    let (ix, iy) = (x as usize, y as usize);
                    let (tx, ty) = (x - ix as f32, y - iy as f32);
                    let coverage: f32 = [
                        (ix, iy, (1.0 - tx) * (1.0 - ty)),
                        ((ix + 1).min(source.width - 1), iy, tx * (1.0 - ty)),
                        (ix, (iy + 1).min(source.height - 1), (1.0 - tx) * ty),
                        (
                            (ix + 1).min(source.width - 1),
                            (iy + 1).min(source.height - 1),
                            tx * ty,
                        ),
                    ]
                    .into_iter()
                    .map(|(px, py, weight)| weight * matte.get(py * source.width + px))
                    .sum();
                    coverage > 1e-4
                })
            })
        });
    }

    pub(crate) fn recover_alpha_boundary(
        &mut self,
        source: &Raster,
        matte: &crate::chroma::AlphaMatte,
    ) {
        let strokes = stroke_model::recover_alpha_boundary(source, matte);
        self.summary.recovered_alpha_boundary_strokes = strokes
            .iter()
            .filter(|s| s.role == "alpha-boundary-stroke")
            .count();
        self.strokes.extend(strokes);
        self.summary.stroke_count = self.strokes.len();
    }

    pub fn empty() -> Self {
        Self {
            strokes: Vec::new(),
            paint_ownership_mask: Vec::new(),
            source_line_mask: Vec::new(),
            legacy_line_mask: Vec::new(),
            role_line_mask: Vec::new(),
            visible_ridge_coverage: Vec::new(),
            dark_boundary_coverage: Vec::new(),
            face_barrier: Vec::new(),
            boundary_stroke_mask: Vec::new(),
            summary: StructuralSummary::default(),
        }
    }

    pub(crate) fn retain_strokes(&mut self, mut retain: impl FnMut(&StructuralStroke) -> bool) {
        self.strokes.retain(|stroke| retain(stroke));
        self.summary.stroke_count = self.strokes.len();
        self.summary.recovered_boundary_strokes = self
            .strokes
            .iter()
            .filter(|s| s.role == "boundary-stroke")
            .count();
        self.summary.recovered_alpha_boundary_strokes = self
            .strokes
            .iter()
            .filter(|s| s.role == "alpha-boundary-stroke")
            .count();
        self.summary.residual_legacy_strokes = self
            .strokes
            .iter()
            .filter(|stroke| stroke.role == "legacy-structural")
            .count();
        self.summary.visible_ridge_strokes = self
            .strokes
            .iter()
            .filter(|stroke| stroke.role == "ridge")
            .count();
        self.summary.boundary_profile_strokes = self
            .strokes
            .iter()
            .filter(|stroke| {
                matches!(
                    stroke.role,
                    "ridge-on-boundary"
                        | "bright-ridge-on-boundary"
                        | "coloured-ridge-on-boundary"
                        | "dark-boundary"
                )
            })
            .count();
    }
}

fn neighbour_indices(index: usize, width: usize, height: usize) -> Vec<usize> {
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
}

/// Exact port of scikit-image 0.26 `_fast_skeletonize`.  Its 256-entry
/// Zhang-Suen classification table is not equivalent at dense diagonal
/// contacts to the commonly reproduced boolean conditions; using the latter
/// leaves two-pixel junction blocks and turns one residual line into many
/// short graph branches.
pub fn skeletonize(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    const LUT: [u8; 256] = [
        0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0,
        3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
        3, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
    ];
    let mut result = mask.to_vec();
    loop {
        let mut changed = false;
        for first_pass in [true, false] {
            let snapshot = result.clone();
            for y in 0..height {
                for x in 0..width {
                    let index = y * width + x;
                    if !snapshot[index] {
                        continue;
                    }
                    let sample = |dx: isize, dy: isize| -> usize {
                        let px = x as isize + dx;
                        let py = y as isize + dy;
                        usize::from(
                            px >= 0
                                && py >= 0
                                && px < width as isize
                                && py < height as isize
                                && snapshot[py as usize * width + px as usize],
                        )
                    };
                    let neighbourhood = sample(-1, -1)
                        + 2 * sample(0, -1)
                        + 4 * sample(1, -1)
                        + 8 * sample(1, 0)
                        + 16 * sample(1, 1)
                        + 32 * sample(0, 1)
                        + 64 * sample(-1, 1)
                        + 128 * sample(-1, 0);
                    let class = LUT[neighbourhood];
                    if class == 3 || (class == 1 && first_pass) || (class == 2 && !first_pass) {
                        result[index] = false;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    result
}

fn remove_small_components(mask: &mut [bool], width: usize, height: usize, minimum: usize) {
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
            for neighbour in neighbour_indices(index, width, height) {
                if mask[neighbour] && !seen[neighbour] {
                    seen[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if component.len() < minimum {
            for index in component {
                mask[index] = false;
            }
        }
    }
}

fn local_lab_mean(lab: &[Lab], width: usize, height: usize, radius: usize) -> Vec<Lab> {
    let padded_width = width + 2 * radius;
    let padded_height = height + 2 * radius;
    let stride = padded_width + 1;
    let mut integral = vec![[0.0_f64; 3]; stride * (padded_height + 1)];
    for y in 0..padded_height {
        let mut row = [0.0_f64; 3];
        let source_y = y.saturating_sub(radius).min(height.saturating_sub(1));
        for x in 0..padded_width {
            let source_x = x.saturating_sub(radius).min(width.saturating_sub(1));
            let sample = lab[source_y * width + source_x];
            row[0] += sample.l as f64;
            row[1] += sample.a as f64;
            row[2] += sample.b as f64;
            let above = integral[y * stride + x + 1];
            integral[(y + 1) * stride + x + 1] =
                [above[0] + row[0], above[1] + row[1], above[2] + row[2]];
        }
    }
    (0..lab.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let x0 = x;
            let y0 = y;
            let x1 = x + 2 * radius + 1;
            let y1 = y + 2 * radius + 1;
            let top_left = integral[y0 * stride + x0];
            let top_right = integral[y0 * stride + x1];
            let bottom_left = integral[y1 * stride + x0];
            let bottom_right = integral[y1 * stride + x1];
            let area = ((x1 - x0) * (y1 - y0)).max(1) as f64;
            Lab {
                l: ((bottom_right[0] - top_right[0] - bottom_left[0] + top_left[0]) / area) as f32,
                a: ((bottom_right[1] - top_right[1] - bottom_left[1] + top_left[1]) / area) as f32,
                b: ((bottom_right[2] - top_right[2] - bottom_left[2] + top_left[2]) / area) as f32,
            }
        })
        .collect()
}

fn connected_components(mask: &[bool], width: usize, height: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut result = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        seen[start] = true;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            for neighbour in neighbour_indices(index, width, height) {
                if mask[neighbour] && !seen[neighbour] {
                    seen[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        result.push(component);
    }
    result
}

fn median_lab(lab: &[Lab], indices: &[usize]) -> Lab {
    let mut lightness: Vec<f32> = indices.iter().map(|&index| lab[index].l).collect();
    let mut a: Vec<f32> = indices.iter().map(|&index| lab[index].a).collect();
    let mut b: Vec<f32> = indices.iter().map(|&index| lab[index].b).collect();
    Lab {
        l: percentile(std::mem::take(&mut lightness), 0.5),
        a: percentile(std::mem::take(&mut a), 0.5),
        b: percentile(std::mem::take(&mut b), 0.5),
    }
}

fn binary_propagation(seeds: &[bool], support: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut result = seeds.to_vec();
    let mut queue: VecDeque<usize> = seeds
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| selected.then_some(index))
        .collect();
    while let Some(index) = queue.pop_front() {
        let x = index % width;
        let y = index / width;
        let neighbours = [
            (x > 0).then(|| index - 1),
            (x + 1 < width).then(|| index + 1),
            (y > 0).then(|| index - width),
            (y + 1 < height).then(|| index + width),
        ];
        for neighbour in neighbours.into_iter().flatten() {
            if support[neighbour] && !result[neighbour] {
                result[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }
    result
}

/// Match structural_ink._complete_same_colour_silhouette_holes. Enclosed
/// raster gaps are filled only through pixels whose Lab colour belongs to
/// the adjacent inner rim; genuinely different counters stay open.
fn complete_same_colour_silhouette_holes(
    silhouettes: &[bool],
    lab: &[Lab],
    width: usize,
    height: usize,
    maximum_delta_e: f32,
) -> Vec<bool> {
    let mut completed = silhouettes.to_vec();
    for component in connected_components(silhouettes, width, height) {
        let minimum_x = component
            .iter()
            .map(|index| index % width)
            .min()
            .unwrap_or(0);
        let maximum_x = component
            .iter()
            .map(|index| index % width)
            .max()
            .unwrap_or(0);
        let minimum_y = component
            .iter()
            .map(|index| index / width)
            .min()
            .unwrap_or(0);
        let maximum_y = component
            .iter()
            .map(|index| index / width)
            .max()
            .unwrap_or(0);
        let local_width = maximum_x - minimum_x + 1;
        let local_height = maximum_y - minimum_y + 1;
        let mut component_mask = vec![false; local_width * local_height];
        for &index in &component {
            let x = index % width - minimum_x;
            let y = index / width - minimum_y;
            component_mask[y * local_width + x] = true;
        }

        // scipy.ndimage.binary_fill_holes uses the cross-shaped default
        // connectivity for its exterior propagation.
        let mut exterior = vec![false; component_mask.len()];
        let mut queue = VecDeque::new();
        for y in 0..local_height {
            for x in 0..local_width {
                if x != 0 && x + 1 != local_width && y != 0 && y + 1 != local_height {
                    continue;
                }
                let index = y * local_width + x;
                if !component_mask[index] && !exterior[index] {
                    exterior[index] = true;
                    queue.push_back(index);
                }
            }
        }
        while let Some(index) = queue.pop_front() {
            let x = index % local_width;
            let y = index / local_width;
            for neighbour in [
                (x > 0).then(|| index - 1),
                (x + 1 < local_width).then(|| index + 1),
                (y > 0).then(|| index - local_width),
                (y + 1 < local_height).then(|| index + local_width),
            ]
            .into_iter()
            .flatten()
            {
                if !component_mask[neighbour] && !exterior[neighbour] {
                    exterior[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        let holes: Vec<bool> = component_mask
            .iter()
            .zip(&exterior)
            .map(|(&inside, &outside)| !inside && !outside)
            .collect();
        for hole in connected_components(&holes, local_width, local_height) {
            let hole_set: HashSet<usize> = hole.iter().copied().collect();
            let mut rim = HashSet::new();
            let mut seeds = Vec::new();
            for &index in &hole {
                let x = index % local_width;
                let y = index / local_width;
                let mut adjacent = false;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let px = x as isize + dx;
                        let py = y as isize + dy;
                        if px < 0
                            || py < 0
                            || px >= local_width as isize
                            || py >= local_height as isize
                        {
                            continue;
                        }
                        let neighbour = py as usize * local_width + px as usize;
                        if component_mask[neighbour] {
                            adjacent = true;
                            rim.insert(neighbour);
                        }
                    }
                }
                if adjacent {
                    seeds.push(index);
                }
            }
            if seeds.is_empty() || rim.is_empty() {
                continue;
            }
            let rim_global: Vec<usize> = rim
                .into_iter()
                .map(|index| {
                    (minimum_y + index / local_width) * width + minimum_x + index % local_width
                })
                .collect();
            let rim_colour = median_lab(lab, &rim_global);
            let mut support = vec![false; component_mask.len()];
            for &index in &hole {
                let global =
                    (minimum_y + index / local_width) * width + minimum_x + index % local_width;
                support[index] = delta_e76(lab[global], rim_colour) <= maximum_delta_e;
            }
            let mut selected = vec![false; component_mask.len()];
            let mut queue = VecDeque::new();
            for seed in seeds {
                if support[seed] && !selected[seed] {
                    selected[seed] = true;
                    queue.push_back(seed);
                }
            }
            while let Some(index) = queue.pop_front() {
                for neighbour in neighbour_indices(index, local_width, local_height) {
                    if hole_set.contains(&neighbour) && support[neighbour] && !selected[neighbour] {
                        selected[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
            for (index, &selected) in selected.iter().enumerate() {
                if selected {
                    let global =
                        (minimum_y + index / local_width) * width + minimum_x + index % local_width;
                    completed[global] = true;
                }
            }
        }
    }
    completed
}

/// Native source structural-line classifier used by the Python pipeline.
/// A line must differ from both sides while those two sides agree; this is
/// what separates a medial ridge from an ordinary two-material boundary.
fn source_structural_lines(source: &Raster) -> (Vec<bool>, Vec<bool>) {
    let width = source.width;
    let height = source.height;
    let lab: Vec<Lab> = source.pixels.par_iter().copied().map(rgb_to_lab).collect();
    let local = local_lab_mean(&lab, width, height, 4);
    let lightness: Vec<f32> = lab.iter().map(|sample| sample.l).collect();
    let local_darkness: Vec<f32> = lab
        .iter()
        .zip(&local)
        .map(|(sample, mean)| (mean.l - sample.l).max(0.0))
        .collect();
    let salience: Vec<f32> = lab
        .iter()
        .zip(&local)
        .map(|(&sample, &mean)| delta_e76(sample, mean))
        .collect();
    let bilateral: Vec<f32> = (0..lab.len())
        .into_par_iter()
        .map(|index| {
            let x = (index % width) as isize;
            let y = (index / width) as isize;
            let centre = lab[index];
            let sample = |dx: isize, dy: isize| {
                let px = (x + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                let py = (y + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                lab[py * width + px]
            };
            [(1_isize, 0_isize), (0, 1), (1, 1), (1, -1)]
                .into_iter()
                .map(|(dx, dy)| {
                    let first = sample(dx * 2, dy * 2);
                    let second = sample(-dx * 2, -dy * 2);
                    let first_distance = delta_e76(centre, first);
                    let second_distance = delta_e76(centre, second);
                    let side_distance = delta_e76(first, second);
                    let minimum = first_distance.min(second_distance);
                    if side_distance <= 0.75 * minimum + 2.5 {
                        (minimum - 0.45 * side_distance).max(0.0)
                    } else {
                        0.0
                    }
                })
                .fold(0.0_f32, f32::max)
        })
        .collect();
    let positive: Vec<f32> = bilateral
        .iter()
        .copied()
        .filter(|value| *value > 1e-4)
        .collect();
    let strong = percentile(positive, 0.82).clamp(7.0, 18.0);
    let weak = (0.45 * strong).max(3.5);
    let dark_seed = percentile(lightness.clone(), 0.05).min(30.0);
    let dark_region = percentile(lightness, 0.12).max(dark_seed + 8.0).min(42.0);
    let seeds: Vec<bool> = (0..lab.len())
        .map(|index| {
            bilateral[index] >= strong
                && (local_darkness[index] >= 2.0
                    || salience[index] >= 5.0
                    || lab[index].l <= dark_seed)
        })
        .collect();
    let support: Vec<bool> = (0..lab.len())
        .map(|index| {
            bilateral[index] >= weak
                && (local_darkness[index] >= 1.0
                    || salience[index] >= 2.5
                    || lab[index].l <= dark_region)
        })
        .collect();
    let propagated = binary_propagation(&seeds, &support, width, height);
    let candidates = erode(&dilate(&propagated, width, height, 1), width, height, 1);
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let raster = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
            image::Luma([if candidates[y as usize * width + x as usize] {
                255
            } else {
                0
            }])
        });
        let _ = raster.save(format!(
            "{}-legacy-candidates.png",
            prefix.to_string_lossy()
        ));
    }
    let candidate_near = dilate(&candidates, width, height, 2);

    let dark_core: Vec<bool> = (0..lab.len())
        .map(|index| {
            lab[index].l <= dark_region
                && (candidate_near[index]
                    || local_darkness[index] >= 3.0
                    || lab[index].l <= dark_seed)
        })
        .collect();
    let opened = dilate(&erode(&dark_core, width, height, 3), width, height, 3);
    let mut broad = opened;
    remove_small_components(&mut broad, width, height, 32);
    let broad_extent = dilate(&broad, width, height, 3);
    let silhouette_proposal: Vec<bool> = broad_extent
        .iter()
        .zip(&dark_core)
        .map(|(&wide, &dark)| wide && dark)
        .collect();
    let mut silhouettes = vec![false; lab.len()];
    let mut material_exclusions = vec![false; lab.len()];
    let mut accepted_components = Vec::<(Vec<usize>, Lab)>::new();
    for component in connected_components(&silhouette_proposal, width, height) {
        if component.len() < 32 {
            continue;
        }
        let component_set: HashSet<usize> = component.iter().copied().collect();
        let mut ring_set = HashSet::<usize>::new();
        for &index in &component {
            let x = index % width;
            let y = index / width;
            for dy in -2_isize..=2 {
                for dx in -2_isize..=2 {
                    if dx * dx + dy * dy > 4 {
                        continue;
                    }
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                        continue;
                    }
                    let neighbour = py as usize * width + px as usize;
                    if !component_set.contains(&neighbour) {
                        ring_set.insert(neighbour);
                    }
                }
            }
        }
        let ring: Vec<usize> = ring_set.into_iter().collect();
        let component_colour = median_lab(&lab, &component);
        let ring_colour = if ring.is_empty() {
            component_colour
        } else {
            median_lab(&lab, &ring)
        };
        let internal_delta: Vec<f32> = component
            .iter()
            .map(|&index| delta_e76(lab[index], component_colour))
            .collect();
        let internal_p90 = percentile(internal_delta.clone(), 0.90);
        let mut outliers = HashSet::<usize>::new();
        for (&index, &distance) in component.iter().zip(&internal_delta) {
            if distance > 15.0 {
                outliers.insert(index);
            }
        }
        let mut retained_outliers = HashSet::<usize>::new();
        while let Some(&start) = outliers.iter().next() {
            let mut queue = VecDeque::from([start]);
            let mut outlier_component = Vec::new();
            outliers.remove(&start);
            while let Some(index) = queue.pop_front() {
                outlier_component.push(index);
                for neighbour in neighbour_indices(index, width, height) {
                    if outliers.remove(&neighbour) {
                        queue.push_back(neighbour);
                    }
                }
            }
            if outlier_component.len() >= 6 {
                retained_outliers.extend(outlier_component);
            }
        }
        let owned_count = component
            .iter()
            .filter(|&&index| !retained_outliers.contains(&index))
            .count();
        let ring_contrast = delta_e76(component_colour, ring_colour);
        if component_colour.l <= dark_region
            && ring_contrast >= 6.0
            && internal_p90 <= 10.0
            && owned_count >= 32
        {
            for &index in &component {
                if retained_outliers.contains(&index) {
                    material_exclusions[index] = true;
                } else {
                    silhouettes[index] = true;
                }
            }
            accepted_components.push((
                component
                    .iter()
                    .copied()
                    .filter(|index| !retained_outliers.contains(index))
                    .collect(),
                component_colour,
            ));
        }
    }
    // Recover same-colour thin branches attached to an accepted broad seed.
    for (component, seed_colour) in accepted_components {
        let mut queue: VecDeque<usize> = component.into_iter().collect();
        while let Some(index) = queue.pop_front() {
            for neighbour in [
                (index % width > 0).then(|| index - 1),
                (index % width + 1 < width).then(|| index + 1),
                (index / width > 0).then(|| index - width),
                (index / width + 1 < height).then(|| index + width),
            ]
            .into_iter()
            .flatten()
            {
                if !silhouettes[neighbour]
                    && dark_core[neighbour]
                    && delta_e76(lab[neighbour], seed_colour) <= 15.0
                {
                    silhouettes[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
    }
    silhouettes = complete_same_colour_silhouette_holes(&silhouettes, &lab, width, height, 15.0);
    let deep_candidate = erode(&candidates, width, height, 3);
    let silhouette_interior = erode(&silhouette_proposal, width, height, 2);
    let mut lines: Vec<bool> = (0..lab.len())
        .map(|index| {
            let chroma = (lab[index].a * lab[index].a + lab[index].b * lab[index].b).sqrt();
            let chromatic_exclusion = material_exclusions[index] && chroma >= 12.0;
            candidates[index]
                && !deep_candidate[index]
                && !silhouettes[index]
                && !silhouette_interior[index]
                && !chromatic_exclusion
                && (lab[index].l <= dark_region + 5.0
                    || (chroma >= 12.0
                        && lab[index].l <= dark_region + 25.0
                        && local_darkness[index] >= 2.0))
        })
        .collect();
    remove_small_components(&mut lines, width, height, 6);
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let raster = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
            image::Luma([if lines[y as usize * width + x as usize] {
                255
            } else {
                0
            }])
        });
        let _ = raster.save(format!("{}-legacy-lines.png", prefix.to_string_lossy()));
    }
    (lines, silhouettes)
}

fn edge_key(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn trace_skeleton(mask: &[bool], width: usize, height: usize) -> Vec<Vec<usize>> {
    // Faithful port of line_details._trace_skeleton_continuously.  At a
    // junction, pair the most opposite incident directions and continue
    // through it; only genuinely unpaired branches start a new path.
    let adjacency: Vec<Vec<usize>> = (0..mask.len())
        .map(|index| {
            if !mask[index] {
                return Vec::new();
            }
            let mut neighbours: Vec<usize> = neighbour_indices(index, width, height)
                .into_iter()
                .filter(|&other| mask[other])
                .collect();
            neighbours.sort_unstable();
            neighbours
        })
        .collect();
    let mut continuation = HashMap::<(usize, usize), usize>::new();
    let mut unpaired = Vec::<(usize, usize)>::new();
    for centre in 0..mask.len() {
        if !mask[centre] {
            continue;
        }
        let neighbours = &adjacency[centre];
        let mut available: HashSet<usize> = neighbours.iter().copied().collect();
        let centre_x = (centre % width) as f32;
        let centre_y = (centre / width) as f32;
        let mut pair_scores = Vec::<(f32, usize, usize)>::new();
        for (position, &left) in neighbours.iter().enumerate() {
            let left_x = (left % width) as f32 - centre_x;
            let left_y = (left / width) as f32 - centre_y;
            let left_length = left_x.hypot(left_y).max(1e-6);
            for &right in neighbours.iter().skip(position + 1) {
                let right_x = (right % width) as f32 - centre_x;
                let right_y = (right / width) as f32 - centre_y;
                let right_length = right_x.hypot(right_y).max(1e-6);
                let score = (left_x * right_x + left_y * right_y) / (left_length * right_length);
                pair_scores.push((score, left, right));
            }
        }
        pair_scores.sort_by(|first, second| {
            first
                .0
                .total_cmp(&second.0)
                .then(first.1.cmp(&second.1))
                .then(first.2.cmp(&second.2))
        });
        for (_, left, right) in pair_scores {
            if !available.contains(&left) || !available.contains(&right) {
                continue;
            }
            continuation.insert((left, centre), right);
            continuation.insert((right, centre), left);
            available.remove(&left);
            available.remove(&right);
        }
        let mut remaining: Vec<usize> = available.into_iter().collect();
        remaining.sort_unstable();
        unpaired.extend(remaining.into_iter().map(|neighbour| (centre, neighbour)));
    }
    let mut remaining = HashSet::<(usize, usize)>::new();
    for (pixel, neighbours) in adjacency.iter().enumerate() {
        for &neighbour in neighbours {
            remaining.insert(edge_key(pixel, neighbour));
        }
    }
    let trace = |mut previous: usize,
                 mut current: usize,
                 remaining: &mut HashSet<(usize, usize)>|
     -> Vec<usize> {
        let mut chain = vec![previous, current];
        remaining.remove(&edge_key(previous, current));
        while let Some(&following) = continuation.get(&(previous, current)) {
            if !remaining.contains(&edge_key(current, following)) {
                break;
            }
            chain.push(following);
            remaining.remove(&edge_key(current, following));
            previous = current;
            current = following;
            if current == chain[0] {
                break;
            }
        }
        chain
    };
    unpaired.sort_by_key(|&(centre, neighbour)| (adjacency[centre].len(), centre, neighbour));
    let mut paths = Vec::new();
    for (previous, current) in unpaired {
        if remaining.contains(&edge_key(previous, current)) {
            paths.push(trace(previous, current, &mut remaining));
        }
    }
    while let Some(&(previous, current)) = remaining.iter().min() {
        paths.push(trace(previous, current, &mut remaining));
    }
    paths
}

fn median_color(source: &Raster, indices: &[usize]) -> [f32; 3] {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for &index in indices {
        for (channel, values) in channels.iter_mut().enumerate() {
            values.push(source.pixels[index][channel]);
        }
    }
    let mut result = [0.0; 3];
    for channel in 0..3 {
        channels[channel].sort_by(|a, b| a.total_cmp(b));
        result[channel] = channels[channel][channels[channel].len() / 2];
    }
    result
}

fn distance_transform_1d(values: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let finite: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect();
    if finite.is_empty() {
        return (
            vec![f32::INFINITY; values.len()],
            vec![usize::MAX; values.len()],
        );
    }
    let mut sites = vec![0_usize; finite.len()];
    let mut intersections = vec![0.0_f32; finite.len() + 1];
    let mut last = 0_usize;
    sites[0] = finite[0];
    intersections[0] = f32::NEG_INFINITY;
    intersections[1] = f32::INFINITY;
    for &site in finite.iter().skip(1) {
        let mut previous = sites[last];
        let mut intersection = ((values[site] + (site * site) as f32)
            - (values[previous] + (previous * previous) as f32))
            / (2.0 * (site as f32 - previous as f32));
        while last > 0 && intersection <= intersections[last] {
            last -= 1;
            previous = sites[last];
            intersection = ((values[site] + (site * site) as f32)
                - (values[previous] + (previous * previous) as f32))
                / (2.0 * (site as f32 - previous as f32));
        }
        last += 1;
        sites[last] = site;
        intersections[last] = intersection;
        intersections[last + 1] = f32::INFINITY;
    }
    let mut distance = vec![0.0_f32; values.len()];
    let mut nearest = vec![0_usize; values.len()];
    let mut envelope = 0_usize;
    for index in 0..values.len() {
        while intersections[envelope + 1] < index as f32 {
            envelope += 1;
        }
        let site = sites[envelope];
        distance[index] = (index as f32 - site as f32).powi(2) + values[site];
        nearest[index] = site;
    }
    (distance, nearest)
}

/// Exact squared-Euclidean nearest-site transform.  The returned index is a
/// source pixel, so a thin removed ridge inherits one incident Paint rather
/// than an invented average of both sides.
fn nearest_site_indices(sites: &[bool], width: usize, height: usize) -> Vec<usize> {
    if sites.iter().all(|value| !*value) {
        return (0..sites.len()).collect();
    }
    // SciPy's feature transform processes axis 0 before axis 1. This order is
    // observable for equidistant sites: column ownership wins the outer tie,
    // then row ownership. Reversing the separable passes gives a diagonal
    // one-pixel ridge the opposite Paint side at hundreds of samples.
    let mut vertical_distance = vec![f32::INFINITY; sites.len()];
    let mut vertical_site = vec![usize::MAX; sites.len()];
    for x in 0..width {
        let values: Vec<f32> = (0..height)
            .map(|y| {
                if sites[y * width + x] {
                    0.0
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        let (distance, nearest) = distance_transform_1d(&values);
        for y in 0..height {
            vertical_distance[y * width + x] = distance[y];
            if nearest[y] != usize::MAX {
                vertical_site[y * width + x] = nearest[y] * width + x;
            }
        }
    }
    let mut result = vec![0_usize; sites.len()];
    for y in 0..height {
        let values: Vec<f32> = (0..width)
            .map(|x| vertical_distance[y * width + x])
            .collect();
        let (_, nearest_x) = distance_transform_1d(&values);
        for x in 0..width {
            let source_x = nearest_x[x];
            result[y * width + x] = if source_x == usize::MAX {
                y * width + x
            } else {
                vertical_site[y * width + source_x]
            };
        }
    }
    result
}

fn gaussian_blur_rgb(image: &Raster, sigma: f32) -> Raster {
    let sigma = sigma.max(1e-3);
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let value = offset as f32;
            (-0.5 * value * value / (sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= total.max(1e-12);
    }
    let horizontal: Vec<[f32; 3]> = (0..image.pixels.len())
        .into_par_iter()
        .map(|index| {
            let x = index % image.width;
            let y = index / image.width;
            let mut sum = [0.0_f32; 3];
            for (offset, &weight) in kernel.iter().enumerate() {
                let px = (x as isize + offset as isize - radius)
                    .clamp(0, image.width.saturating_sub(1) as isize)
                    as usize;
                let sample = image.pixels[y * image.width + px];
                for channel in 0..3 {
                    sum[channel] += weight * sample[channel];
                }
            }
            sum
        })
        .collect();
    let pixels: Vec<[f32; 3]> = (0..image.pixels.len())
        .into_par_iter()
        .map(|index| {
            let x = index % image.width;
            let y = index / image.width;
            let mut sum = [0.0_f32; 3];
            for (offset, &weight) in kernel.iter().enumerate() {
                let py = (y as isize + offset as isize - radius)
                    .clamp(0, image.height.saturating_sub(1) as isize)
                    as usize;
                let sample = horizontal[py * image.width + x];
                for channel in 0..3 {
                    sum[channel] += weight * sample[channel];
                }
            }
            sum
        })
        .collect();
    Raster::new(image.width, image.height, pixels)
}

fn nearest_underpaint(source: &Raster, mask: &[bool]) -> Raster {
    if !mask.iter().any(|value| *value) || mask.iter().all(|value| *value) {
        return source.clone();
    }
    let paint_sites: Vec<bool> = mask.iter().map(|value| !*value).collect();
    let nearest = nearest_site_indices(&paint_sites, source.width, source.height);
    let mut result = source.clone();
    let mut distances = vec![0.0_f32; mask.len()];
    let mut maximum_radius = 0.0_f32;
    for index in 0..mask.len() {
        if mask[index] {
            result.pixels[index] = source.pixels[nearest[index]];
            let x = index % source.width;
            let y = index / source.width;
            let source_x = nearest[index] % source.width;
            let source_y = nearest[index] / source.width;
            let distance = (x as f32 - source_x as f32).hypot(y as f32 - source_y as f32);
            distances[index] = distance;
            maximum_radius = maximum_radius.max(distance);
        }
    }
    if maximum_radius > 1.0 {
        let diffuse: Vec<bool> = mask
            .iter()
            .zip(&distances)
            .map(|(&owned, &distance)| owned && distance > 1.0)
            .collect();
        let mut sigma = (maximum_radius / 2.5).clamp(1.5, 12.0);
        while sigma > 1.0 {
            let smoothed = gaussian_blur_rgb(&result, sigma);
            for (index, &selected) in diffuse.iter().enumerate() {
                if selected {
                    result.pixels[index] = smoothed.pixels[index];
                }
            }
            sigma *= 0.5;
        }
    }
    result
}

fn unmix_structural_antialias(
    source: &Raster,
    underpaint: &mut Raster,
    structural: &[bool],
) -> Vec<bool> {
    if !structural.iter().any(|value| *value) || structural.iter().all(|value| *value) {
        return vec![false; structural.len()];
    }
    let protected = dilate_square(structural, source.width, source.height, 1);
    let shoulder: Vec<bool> = protected
        .iter()
        .zip(structural)
        .map(|(&outer, &core)| outer && !core)
        .collect();
    let nearest_ink = nearest_site_indices(structural, source.width, source.height);
    let mut selected = vec![false; structural.len()];
    for index in 0..shoulder.len() {
        if !shoulder[index] {
            continue;
        }
        let x = index % source.width;
        let y = index / source.width;
        let ink_index = nearest_ink[index];
        let ink_x = ink_index % source.width;
        let ink_y = ink_index / source.width;
        let outward_x = x as isize - ink_x as isize;
        let outward_y = y as isize - ink_y as isize;
        let observed = source.pixels[index];
        let ink = source.pixels[ink_index];
        let mut best_score = f32::INFINITY;
        let mut best_paint = underpaint.pixels[index];
        for dy in -3_isize..=3 {
            for dx in -3_isize..=3 {
                if dx == 0 && dy == 0 || dx * outward_x + dy * outward_y <= 0 {
                    continue;
                }
                let px = x as isize + dx;
                let py = y as isize + dy;
                if px < 0 || py < 0 || px >= source.width as isize || py >= source.height as isize {
                    continue;
                }
                let candidate = py as usize * source.width + px as usize;
                if protected[candidate] {
                    continue;
                }
                let paint = underpaint.pixels[candidate];
                let direction = [ink[0] - paint[0], ink[1] - paint[1], ink[2] - paint[2]];
                let denominator = direction.iter().map(|value| value * value).sum::<f32>();
                if denominator <= 1e-6 {
                    continue;
                }
                let alpha = ((observed[0] - paint[0]) * direction[0]
                    + (observed[1] - paint[1]) * direction[1]
                    + (observed[2] - paint[2]) * direction[2])
                    / denominator;
                if !(0.03..=0.97).contains(&alpha) {
                    continue;
                }
                let reconstructed = [
                    paint[0] * (1.0 - alpha) + ink[0] * alpha,
                    paint[1] * (1.0 - alpha) + ink[1] * alpha,
                    paint[2] * (1.0 - alpha) + ink[2] * alpha,
                ];
                let residual = ((observed[0] - reconstructed[0]).powi(2)
                    + (observed[1] - reconstructed[1]).powi(2)
                    + (observed[2] - reconstructed[2]).powi(2))
                .sqrt();
                if residual > 0.06 {
                    continue;
                }
                let score = residual + 0.001 * ((dx * dx + dy * dy) as f32).sqrt();
                if score < best_score {
                    best_score = score;
                    best_paint = paint;
                }
            }
        }
        if best_score.is_finite() {
            underpaint.pixels[index] = best_paint;
            selected[index] = true;
        }
    }
    selected
}

/// Recover the opaque side of the rasterized silhouette boundary at the
/// same alpha=0.5 ownership level used by the Python reference.
fn extend_structural_silhouette_antialias(source: &Raster, structural: &[bool]) -> Vec<bool> {
    if !structural.iter().any(|value| *value) || structural.iter().all(|value| *value) {
        return structural.to_vec();
    }
    let underpaint = nearest_underpaint(source, structural);
    let protected = dilate_square(structural, source.width, source.height, 1);
    let shoulder: Vec<bool> = protected
        .iter()
        .zip(structural)
        .map(|(&outer, &core)| outer && !core)
        .collect();
    let nearest_ink = nearest_site_indices(structural, source.width, source.height);
    let mut extended = structural.to_vec();
    for index in 0..shoulder.len() {
        if !shoulder[index] {
            continue;
        }
        let x = index % source.width;
        let y = index / source.width;
        let ink_index = nearest_ink[index];
        let ink_x = ink_index % source.width;
        let ink_y = ink_index / source.width;
        let outward_x = x as isize - ink_x as isize;
        let outward_y = y as isize - ink_y as isize;
        let observed = source.pixels[index];
        let ink = source.pixels[ink_index];
        let mut best_score = f32::INFINITY;
        let mut best_alpha = f32::NAN;
        for dy in -3_isize..=3 {
            for dx in -3_isize..=3 {
                if dx == 0 && dy == 0 || dx * outward_x + dy * outward_y <= 0 {
                    continue;
                }
                let px = x as isize + dx;
                let py = y as isize + dy;
                if px < 0 || py < 0 || px >= source.width as isize || py >= source.height as isize {
                    continue;
                }
                let candidate = py as usize * source.width + px as usize;
                if protected[candidate] {
                    continue;
                }
                let paint = underpaint.pixels[candidate];
                let direction = [ink[0] - paint[0], ink[1] - paint[1], ink[2] - paint[2]];
                let denominator = direction.iter().map(|value| value * value).sum::<f32>();
                if denominator <= 1e-6 {
                    continue;
                }
                let alpha = ((observed[0] - paint[0]) * direction[0]
                    + (observed[1] - paint[1]) * direction[1]
                    + (observed[2] - paint[2]) * direction[2])
                    / denominator;
                if !(0.03..=0.97).contains(&alpha) {
                    continue;
                }
                let reconstructed = [
                    paint[0] * (1.0 - alpha) + ink[0] * alpha,
                    paint[1] * (1.0 - alpha) + ink[1] * alpha,
                    paint[2] * (1.0 - alpha) + ink[2] * alpha,
                ];
                let residual = ((observed[0] - reconstructed[0]).powi(2)
                    + (observed[1] - reconstructed[1]).powi(2)
                    + (observed[2] - reconstructed[2]).powi(2))
                .sqrt();
                if residual > 0.06 {
                    continue;
                }
                let score = residual + 0.001 * ((dx * dx + dy * dy) as f32).sqrt();
                if score < best_score {
                    best_score = score;
                    best_alpha = alpha;
                }
            }
        }
        if best_score.is_finite() && best_alpha >= 0.5 {
            extended[index] = true;
        }
    }
    extended
}

/// Transfer only source-supported medial ridges out of Paint.  Dark filled
/// faces remain Paint-owned; there is intentionally no median-colour
/// silhouette overlay that could flatten tyre or shadow gradients.
pub fn analyse(source: &Raster, roles: &mut EdgeRoles) -> (Raster, StructuralInk) {
    let recovered = stroke_model::recover(source, &roles.dark_boundary_graph);
    let (mut classified_lines, classified_silhouettes) = source_structural_lines(source);
    let classified_silhouettes =
        extend_structural_silhouette_antialias(source, &classified_silhouettes);
    for (line, &silhouette) in classified_lines.iter_mut().zip(&classified_silhouettes) {
        *line &= !silhouette;
    }
    let legacy_line_mask = classified_lines.clone();
    let shading_corridor = dilate_square(&roles.shading, source.width, source.height, 1);
    // Only a profile-confirmed medial ridge has the same Paint owner on both
    // sides and may be removed before quantization.  Other structural
    // candidates remain in the complete Paint base until residual selection.
    // Python builds the legacy/role-owned line candidates before the shared
    // prequantization mask is extended with antialias ownership.  Preserve
    // that original graph coverage for candidate construction below.
    let original_visible_ridge_coverage = roles.visible_ridge_coverage.clone();
    let mut paint_reference = nearest_underpaint(source, &original_visible_ridge_coverage);
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        let _ = paint_reference.save(std::path::Path::new(&format!(
            "{prefix}-nearest-underpaint.png"
        )));
    }
    let antialias_ownership = unmix_structural_antialias(
        source,
        &mut paint_reference,
        &original_visible_ridge_coverage,
    );
    let antialias_unmixed_pixels = antialias_ownership.iter().filter(|&&value| value).count();
    // Python's structural_prequantization_mask returns the ridge coverage
    // array by reference.  Extending that mask with source-modelled AA
    // ownership therefore also extends the downstream ridge support.  Keep
    // the same shared-mask semantics here: the graph core and its recovered
    // AA shoulders remain one structural owner through residual selection.
    for (ridge, &shoulder) in roles
        .visible_ridge_coverage
        .iter_mut()
        .zip(&antialias_ownership)
    {
        *ridge |= shoulder;
    }
    let mut underpaint_ownership = roles.visible_ridge_coverage.clone();
    for &(index, paint) in &recovered.updates {
        paint_reference.pixels[index] = paint;
        underpaint_ownership[index] = true;
        roles.face_barrier[index] = false;
        roles.dark_boundary[index] = false;
    }
    let mut source_graph_coverage = original_visible_ridge_coverage.clone();
    for edge in &roles.dark_boundary_graph {
        for pair in edge.points.windows(2) {
            let length = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            let steps = (2.0 * length).ceil().max(1.0) as usize;
            let radius = (0.5 * edge.width + 0.5).max(0.75);
            for step in 0..=steps {
                let amount = step as f64 / steps as f64;
                let point_x = pair[0][0] + amount * (pair[1][0] - pair[0][0]);
                let point_y = pair[0][1] + amount * (pair[1][1] - pair[0][1]);
                let minimum_x = (point_x - radius - 0.5).floor().max(0.0) as usize;
                let maximum_x = (point_x + radius - 0.5)
                    .ceil()
                    .min(source.width.saturating_sub(1) as f64)
                    as usize;
                let minimum_y = (point_y - radius - 0.5).floor().max(0.0) as usize;
                let maximum_y = (point_y + radius - 0.5)
                    .ceil()
                    .min(source.height.saturating_sub(1) as f64)
                    as usize;
                for y in minimum_y..=maximum_y {
                    for x in minimum_x..=maximum_x {
                        let dx = x as f64 + 0.5 - point_x;
                        let dy = y as f64 + 0.5 - point_y;
                        if dx * dx + dy * dy < radius * radius {
                            source_graph_coverage[y * source.width + x] = true;
                        }
                    }
                }
            }
        }
    }
    let source_line_mask: Vec<bool> = classified_lines
        .iter()
        .zip(&classified_silhouettes)
        .zip(&shading_corridor)
        .zip(&original_visible_ridge_coverage)
        .map(|(((&classified, &silhouette), &shading), &ridge)| {
            silhouette || (classified && !shading) || ridge
        })
        .collect();
    let mut role_line_mask: Vec<bool> = classified_lines
        .iter()
        .zip(&shading_corridor)
        .zip(&original_visible_ridge_coverage)
        .map(|((&classified, &shading), &ridge)| (classified && !shading) || ridge)
        .collect();
    for (index, &owned) in recovered.mask.iter().enumerate() {
        if owned {
            role_line_mask[index] = false;
        }
    }
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let raster =
            image::GrayImage::from_fn(source.width as u32, source.height as u32, |x, y| {
                image::Luma([
                    if source_line_mask[y as usize * source.width + x as usize] {
                        255
                    } else {
                        0
                    },
                ])
            });
        let _ = raster.save(format!("{}-source-line-mask.png", prefix.to_string_lossy()));
    }
    let unclassified_source_coverage: Vec<bool> = classified_lines
        .iter()
        .zip(&shading_corridor)
        .zip(&source_graph_coverage)
        .map(|((&classified, &shading), &graph)| classified && !shading && !graph)
        .collect();
    // Do not speculate a provisional legacy graph here.  Whether Paint owns a
    // legacy source line is known only after the exact Paint partition has
    // rendered; constructing and then discarding an early skeleton duplicated
    // work and could never be the authoritative topology.
    let mut strokes = roles
        .visible_ridge_graph
        .iter()
        .chain(&roles.dark_boundary_graph)
        .filter_map(|edge| {
            if edge.points.len() < 2 {
                return None;
            }
            let points: Vec<Point> = edge
                .points
                .iter()
                .map(|value| Point {
                    x: value[0] as f32,
                    y: value[1] as f32,
                })
                .collect();
            if points.len() < 2 {
                return None;
            }
            let point_count = points.len();
            let indices: Vec<usize> = edge
                .points
                .iter()
                .map(|value| {
                    let x = (value[0] - 0.5)
                        .round()
                        .clamp(0.0, source.width.saturating_sub(1) as f64)
                        as usize;
                    let y = (value[1] - 0.5)
                        .round()
                        .clamp(0.0, source.height.saturating_sub(1) as f64)
                        as usize;
                    y * source.width + x
                })
                .collect();
            Some(StructuralStroke {
                // Keep detector samples through residual profile measurement.
                // The selected complete edge or arclength interval is
                // simplified only after ownership has been decided.
                points,
                path_data: None,
                precise_points: Some(edge.points.clone()),
                color: median_color(source, &indices),
                width: edge.width.max(1.2) as f32,
                role: edge.role,
                width_samples: vec![(edge.width.max(1.2) as f32, point_count)],
            })
        })
        .collect::<Vec<_>>();
    let recovered_boundary_strokes = recovered.strokes.len();
    strokes.extend(recovered.strokes);
    let summary = StructuralSummary {
        recovered_boundary_strokes,
        recovered_alpha_boundary_strokes: 0,
        source_coverage_pixels: unclassified_source_coverage
            .iter()
            .zip(&source_graph_coverage)
            .filter(|&(unclassified, graph)| *unclassified || *graph)
            .count(),
        source_line_pixels: source_line_mask.iter().filter(|&&value| value).count(),
        skeleton_pixels: 0,
        stroke_count: strokes.len(),
        underpainted_pixels: underpaint_ownership.iter().filter(|&&value| value).count(),
        antialias_unmixed_pixels,
        silhouette_fill_count: 0,
        residual_legacy_strokes: 0,
        visible_ridge_strokes: strokes
            .iter()
            .filter(|stroke| stroke.role == "ridge")
            .count(),
        boundary_profile_strokes: strokes
            .iter()
            .filter(|stroke| {
                matches!(
                    stroke.role,
                    "ridge-on-boundary"
                        | "bright-ridge-on-boundary"
                        | "coloured-ridge-on-boundary"
                        | "dark-boundary"
                )
            })
            .count(),
    };
    (
        paint_reference,
        StructuralInk {
            strokes,
            paint_ownership_mask: underpaint_ownership,
            source_line_mask,
            legacy_line_mask,
            role_line_mask,
            visible_ridge_coverage: roles.visible_ridge_coverage.clone(),
            dark_boundary_coverage: roles.dark_boundary.clone(),
            face_barrier: roles.face_barrier.clone(),
            boundary_stroke_mask: recovered.mask,
            summary,
        },
    )
}

fn bilinear_lab_precise(values: &[Lab], width: usize, height: usize, point: [f64; 2]) -> Lab {
    let x = (point[0] - 0.5).clamp(0.0, width.saturating_sub(1) as f64);
    let y = (point[1] - 0.5).clamp(0.0, height.saturating_sub(1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let interpolate = |channel: fn(Lab) -> f32| {
        let top = channel(values[y0 * width + x0]) as f64 * (1.0 - tx)
            + channel(values[y0 * width + x1]) as f64 * tx;
        let bottom = channel(values[y1 * width + x0]) as f64 * (1.0 - tx)
            + channel(values[y1 * width + x1]) as f64 * tx;
        (top * (1.0 - ty) + bottom * ty) as f32
    };
    Lab {
        l: interpolate(|value| value.l),
        a: interpolate(|value| value.a),
        b: interpolate(|value| value.b),
    }
}

fn precise_normal_at(points: &[[f64; 2]], index: usize) -> [f64; 2] {
    let before = points[index.saturating_sub(1)];
    let after = points[(index + 1).min(points.len() - 1)];
    let dx = after[0] - before[0];
    let dy = after[1] - before[1];
    let length = dx.hypot(dy).max(1e-6);
    [-dy / length, dx / length]
}

fn residual_line_masks(
    source_lab: &[Lab],
    rendered_lab: &[Lab],
    source_lines: &[bool],
    width: usize,
    height: usize,
) -> (Vec<bool>, Vec<bool>) {
    let represented: Vec<bool> = (0..source_lab.len())
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let reference = source_lab[index];
            let chroma = reference.a.hypot(reference.b);
            let mut minimum_delta = f32::INFINITY;
            let mut minimum_lightness = f32::INFINITY;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let px = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                    let py = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                    let candidate = rendered_lab[py * width + px];
                    minimum_delta = minimum_delta.min(delta_e76(reference, candidate));
                    minimum_lightness = minimum_lightness.min(candidate.l);
                }
            }
            minimum_delta <= 4.0
                || (chroma < 12.0 && reference.l <= 50.0 && minimum_lightness <= reference.l + 4.0)
        })
        .collect();
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        let bytes = represented
            .iter()
            .map(|&value| u8::from(value))
            .collect::<Vec<_>>();
        let _ = std::fs::write(
            format!("{}-represented.u8", prefix.to_string_lossy()),
            bytes,
        );
    }
    let mut selected = vec![false; source_lines.len()];
    let mut measured = vec![false; source_lines.len()];
    for component in connected_components(source_lines, width, height) {
        let mut missing = vec![false; source_lines.len()];
        let mut missing_area = 0_usize;
        for &index in &component {
            if !represented[index] {
                missing[index] = true;
                missing_area += 1;
            }
        }
        if missing_area < 3
            || (missing_area < 8 && missing_area as f32 / (component.len().max(1) as f32) < 0.12)
        {
            continue;
        }
        let expanded = dilate(&missing, width, height, 1);
        let component_residual: Vec<usize> = component
            .iter()
            .copied()
            .filter(|&index| expanded[index])
            .collect();
        for &index in &component_residual {
            measured[index] = true;
        }
        if component_residual.len() as f32 / component.len().max(1) as f32 >= 0.75 {
            for &index in &component {
                selected[index] = true;
            }
        } else {
            for index in component_residual {
                selected[index] = true;
            }
        }
    }
    remove_small_components(&mut selected, width, height, 3);
    remove_small_components(&mut measured, width, height, 3);
    (selected, measured)
}

fn mask_fraction_along(
    stroke: &StructuralStroke,
    mask: &[bool],
    width: usize,
    height: usize,
) -> f32 {
    if stroke.points.is_empty() {
        return 0.0;
    }
    let selected = stroke
        .points
        .iter()
        .filter(|point| {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, height.saturating_sub(1) as f32) as usize;
            mask[y * width + x]
        })
        .count();
    selected as f32 / stroke.points.len() as f32
}

fn classify_boundary_role(
    stroke: &StructuralStroke,
    source_lab: &[Lab],
    width: usize,
    height: usize,
) -> &'static str {
    if stroke.role != "ridge-on-boundary" {
        return stroke.role;
    }
    let mut chroma = stroke
        .points
        .iter()
        .map(|point| {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, height.saturating_sub(1) as f32) as usize;
            let value = source_lab[y * width + x];
            value.a.hypot(value.b)
        })
        .collect::<Vec<_>>();
    chroma.sort_by(f32::total_cmp);
    let median_chroma = if chroma.is_empty() {
        0.0
    } else if chroma.len() % 2 == 0 {
        let middle = chroma.len() / 2;
        0.5 * (chroma[middle - 1] + chroma[middle])
    } else {
        chroma[chroma.len() / 2]
    };
    if median_chroma >= 12.0 {
        "coloured-ridge-on-boundary"
    } else {
        stroke.role
    }
}

/// A short dark extremum at an already-dark material boundary is commonly a
/// resize/sharpening undershoot, not an authored constant-width ink stroke.
/// Keep it in Paint so shared geometry can follow the source boundary without
/// turning the run into a round-capped structural capsule.
///
/// This is intentionally a whole-profile ownership decision rather than a
/// coordinate or raster mask. Long contours remain structural, as do short
/// contours with a light incident face; only the ambiguous short run between
/// two dark faces is returned to Paint.
fn paint_owned_dark_boundary_undershoot(
    stroke: &StructuralStroke,
    source_lab: &[Lab],
    width: usize,
    height: usize,
) -> bool {
    if stroke.role != "ridge-on-boundary" || stroke.points.len() < 3 {
        return false;
    }
    let precise_points = stroke.precise_points.clone().unwrap_or_else(|| {
        stroke
            .points
            .iter()
            .map(|point| [point.x as f64, point.y as f64])
            .collect()
    });
    let length = precise_points
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
        .sum::<f64>();
    if length > 5.0 * stroke.width.max(0.4) as f64 {
        return false;
    }

    let mut dark_sides = 0_usize;
    let mut undershoot = 0_usize;
    for (index, &point) in precise_points.iter().enumerate() {
        let [nx, ny] = precise_normal_at(&precise_points, index);
        let sample = |offset: f64| {
            bilinear_lab_precise(
                source_lab,
                width,
                height,
                [point[0] + offset * nx, point[1] + offset * ny],
            )
        };
        let sides = [sample(-1.5), sample(1.5)];
        let centre = [-0.75_f64, 0.0, 0.75]
            .into_iter()
            .map(sample)
            .min_by(|first, second| first.l.total_cmp(&second.l))
            .unwrap_or_else(|| sample(0.0));
        if sides[0].l.max(sides[1].l) < 30.0 {
            dark_sides += 1;
        }
        if centre.l + 6.0 < sides[0].l.min(sides[1].l) {
            undershoot += 1;
        }
    }
    dark_sides * 10 >= precise_points.len() * 9 && undershoot * 4 >= precise_points.len() * 3
}

fn boundary_profile_flags(
    stroke: &StructuralStroke,
    source_lab: &[Lab],
    rendered_lab: &[Lab],
    width: usize,
    height: usize,
    rendered_core_lightness_excess: f32,
) -> (Vec<bool>, Vec<bool>) {
    let offsets = [-0.75_f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
    let coloured = stroke.role == "coloured-ridge-on-boundary";
    let bright = stroke.role == "bright-ridge-on-boundary";
    let mut missing = Vec::with_capacity(stroke.points.len());
    let mut valley = Vec::with_capacity(stroke.points.len());
    let precise_points = stroke.precise_points.clone().unwrap_or_else(|| {
        stroke
            .points
            .iter()
            .map(|point| [point.x as f64, point.y as f64])
            .collect()
    });
    #[cfg(feature = "diagnostics")]
    let debug_target = precise_points
        .iter()
        .any(|point| (point[0] - 224.1524).hypot(point[1] - 769.4954) < 0.01);
    #[cfg(feature = "diagnostics")]
    let mut debug_values = Vec::<serde_json::Value>::new();
    for (index, &point) in precise_points.iter().enumerate() {
        let [nx, ny] = precise_normal_at(&precise_points, index);
        let sample = |values: &[Lab], offset: f32| {
            bilinear_lab_precise(
                values,
                width,
                height,
                [point[0] + offset as f64 * nx, point[1] + offset as f64 * ny],
            )
        };
        let source_profile = offsets
            .iter()
            .map(|&offset| sample(source_lab, offset))
            .collect::<Vec<_>>();
        let rendered_profile = offsets
            .iter()
            .map(|&offset| sample(rendered_lab, offset))
            .collect::<Vec<_>>();
        let source_sides = [sample(source_lab, -1.5), sample(source_lab, 1.5)];
        let rendered_sides = [sample(rendered_lab, -1.5), sample(rendered_lab, 1.5)];
        let target = if coloured {
            source_profile[3]
        } else if bright {
            source_profile
                .iter()
                .copied()
                .max_by(|first, second| first.l.total_cmp(&second.l))
                .unwrap_or(source_profile[3])
        } else {
            source_profile
                .iter()
                .copied()
                .min_by(|first, second| first.l.total_cmp(&second.l))
                .unwrap_or(source_profile[3])
        };
        let source_dark_contrast = source_sides[0].l.min(source_sides[1].l) - target.l;
        let source_bright_contrast = target.l - source_sides[0].l.max(source_sides[1].l);
        let source_side_colour_contrast =
            delta_e2000(source_sides[0], target).min(delta_e2000(source_sides[1], target));
        let source_line_contrast = if bright {
            source_bright_contrast.max(source_side_colour_contrast)
        } else if coloured {
            source_dark_contrast.max(source_side_colour_contrast)
        } else {
            source_dark_contrast
        };
        let rendered_minimum_lightness = rendered_profile
            .iter()
            .map(|value| value.l)
            .fold(f32::INFINITY, f32::min);
        let rendered_dark_contrast =
            rendered_sides[0].l.min(rendered_sides[1].l) - rendered_minimum_lightness;
        let rendered_maximum_lightness = rendered_profile
            .iter()
            .map(|value| value.l)
            .fold(f32::NEG_INFINITY, f32::max);
        let rendered_bright_contrast =
            rendered_maximum_lightness - rendered_sides[0].l.max(rendered_sides[1].l);
        let minimum_error = rendered_profile
            .iter()
            .map(|&value| delta_e2000(value, target))
            .fold(f32::INFINITY, f32::min);
        let has_valley = source_line_contrast >= 4.0;
        let contrast_missing = if bright {
            rendered_bright_contrast < 0.65 * source_bright_contrast
                || target.l - rendered_maximum_lightness >= rendered_core_lightness_excess
        } else {
            rendered_dark_contrast < 0.65 * source_dark_contrast
                || rendered_minimum_lightness - target.l >= rendered_core_lightness_excess
        };
        let is_missing = minimum_error > 4.0 && has_valley && contrast_missing;
        valley.push(has_valley);
        missing.push(is_missing);
        #[cfg(feature = "diagnostics")]
        if debug_target {
            debug_values.push(serde_json::json!({
                "point": point,
                "source_dark": source_dark_contrast,
                "source_bright": source_bright_contrast,
                "side_colour": source_side_colour_contrast,
                "line": source_line_contrast,
                "error": minimum_error,
                "rendered_dark": rendered_dark_contrast,
                "rendered_bright": rendered_bright_contrast,
                "light_excess": rendered_minimum_lightness - target.l,
                "valley": has_valley,
                "missing": is_missing,
            }));
        }
    }
    #[cfg(feature = "diagnostics")]
    if debug_target {
        if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
            let suffix = if rendered_core_lightness_excess.is_finite() {
                "missing"
            } else {
                "wide"
            };
            if let Ok(bytes) = serde_json::to_vec_pretty(&debug_values) {
                let _ = std::fs::write(
                    format!("{}-profile-target-{suffix}.json", prefix.to_string_lossy()),
                    bytes,
                );
            }
        }
    }
    (missing, valley)
}

fn flags_to_mask(
    stroke: &StructuralStroke,
    flags: &[bool],
    mask: &mut [bool],
    width: usize,
    height: usize,
) {
    for (point, &selected) in stroke.points.iter().zip(flags) {
        if !selected {
            continue;
        }
        let x = (point.x - 0.5)
            .round()
            .clamp(0.0, width.saturating_sub(1) as f32) as usize;
        let y = (point.y - 0.5)
            .round()
            .clamp(0.0, height.saturating_sub(1) as f32) as usize;
        mask[y * width + x] = true;
    }
}

fn split_supported_runs(
    stroke: &StructuralStroke,
    source_valley: &[bool],
) -> Vec<StructuralStroke> {
    if stroke.points.len() < 2 {
        return Vec::new();
    }
    let mut selected = source_valley.to_vec();
    let precise_points = stroke.precise_points.clone().unwrap_or_else(|| {
        stroke
            .points
            .iter()
            .map(|point| [point.x as f64, point.y as f64])
            .collect()
    });
    let cumulative = std::iter::once(0.0_f64)
        .chain(precise_points.windows(2).scan(0.0_f64, |total, pair| {
            *total += (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            Some(*total)
        }))
        .collect::<Vec<_>>();
    let mut index = 0_usize;
    while index < selected.len() {
        if selected[index] {
            index += 1;
            continue;
        }
        let first = index;
        while index < selected.len() && !selected[index] {
            index += 1;
        }
        if first > 0 && index < selected.len() && cumulative[index] - cumulative[first - 1] <= 3.0 {
            selected[first..index].fill(true);
        }
    }
    let mut result = Vec::new();
    let mut index = 0_usize;
    while index < selected.len() {
        if !selected[index] {
            index += 1;
            continue;
        }
        let run_first = index;
        while index + 1 < selected.len() && selected[index + 1] {
            index += 1;
        }
        let run_last = index;
        index += 1;
        let mut first = run_first;
        let mut last = run_last;
        while first > 0 && cumulative[run_first] - cumulative[first - 1] <= 1.0 {
            first -= 1;
        }
        while last + 1 < stroke.points.len() && cumulative[last + 1] - cumulative[run_last] <= 1.0 {
            last += 1;
        }
        if cumulative[last] - cumulative[first] < 2.0 {
            continue;
        }
        result.push(StructuralStroke {
            points: stroke.points[first..=last].to_vec(),
            path_data: None,
            precise_points: stroke
                .precise_points
                .as_ref()
                .map(|points| points[first..=last].to_vec()),
            color: stroke.color,
            width: stroke.width,
            role: stroke.role,
            width_samples: stroke.width_samples.clone(),
        });
    }
    result
}

fn graph_endpoint(stroke: &StructuralStroke, at_start: bool) -> (Point, (f32, f32)) {
    let endpoint = if at_start {
        stroke.points[0]
    } else {
        stroke.points[stroke.points.len() - 1]
    };
    let reach = 5.min(stroke.points.len() - 1);
    let interior = if at_start {
        stroke.points[reach]
    } else {
        stroke.points[stroke.points.len() - reach - 1]
    };
    let dx = endpoint.x - interior.x;
    let dy = endpoint.y - interior.y;
    let length = dx.hypot(dy).max(1e-8);
    (endpoint, (dx / length, dy / length))
}

/// Return the same lexicographically ordered pairs as a dense `first < second`
/// scan, restricted to points within `maximum_distance`. A cell has exactly
/// the query radius as its side, so every qualifying pair is in one of the
/// surrounding 3x3 cells.
fn nearby_point_pairs(points: &[Point], maximum_distance: f32) -> Vec<(usize, usize)> {
    if points.len() < 2 || maximum_distance <= 0.0 {
        return Vec::new();
    }
    let cell_key = |point: Point| {
        (
            (point.x / maximum_distance).floor() as i32,
            (point.y / maximum_distance).floor() as i32,
        )
    };
    let mut cells = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, &point) in points.iter().enumerate() {
        cells.entry(cell_key(point)).or_default().push(index);
    }
    let mut pairs = Vec::<(usize, usize)>::new();
    for (first, &point) in points.iter().enumerate() {
        let (cell_x, cell_y) = cell_key(point);
        for y in cell_y - 1..=cell_y + 1 {
            for x in cell_x - 1..=cell_x + 1 {
                let Some(indices) = cells.get(&(x, y)) else {
                    continue;
                };
                for &second in indices {
                    if second > first && point.distance(points[second]) <= maximum_distance {
                        pairs.push((first, second));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

fn raster_line_points(first: Point, second: Point) -> Vec<(usize, usize)> {
    let mut x0 = (first.x - 0.5).round() as isize;
    let mut y0 = (first.y - 0.5).round() as isize;
    let x1 = (second.x - 0.5).round() as isize;
    let y1 = (second.y - 0.5).round() as isize;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    let mut result = Vec::new();
    loop {
        if x0 >= 0 && y0 >= 0 {
            result.push((x0 as usize, y0 as usize));
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
    result
}

fn graph_bridge_supported(
    first: Point,
    second: Point,
    support: &[bool],
    width: usize,
    height: usize,
) -> bool {
    let mut inside = 0_usize;
    let mut selected = 0_usize;
    for (x, y) in raster_line_points(first, second) {
        if x >= width || y >= height {
            continue;
        }
        inside += 1;
        selected += usize::from(support[y * width + x]);
    }
    inside > 0 && selected as f32 / inside as f32 >= 0.80
}

fn weighted_graph_width(strokes: &[&StructuralStroke]) -> (f32, Vec<(f32, usize)>) {
    let mut samples = Vec::new();
    for stroke in strokes {
        if stroke.width_samples.is_empty() {
            samples.push((stroke.width, stroke.points.len().max(1)));
        } else {
            samples.extend_from_slice(&stroke.width_samples);
        }
    }
    samples.sort_by(|first, second| first.0.total_cmp(&second.0));
    let middle = 0.5 * samples.iter().map(|value| value.1).sum::<usize>() as f32;
    let mut cumulative = 0_usize;
    let mut width = samples.last().map(|value| value.0).unwrap_or(1.0);
    for &(candidate, weight) in &samples {
        cumulative += weight;
        if cumulative as f32 >= middle {
            width = candidate;
            break;
        }
    }
    (width, samples)
}

fn merge_graph_edges(
    first: &StructuralStroke,
    first_at_start: bool,
    second: &StructuralStroke,
    second_at_start: bool,
) -> StructuralStroke {
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
    let last = first_points.len() - 1;
    let shared = Point {
        x: 0.5 * (first_points[last].x + second_points[0].x),
        y: 0.5 * (first_points[last].y + second_points[0].y),
    };
    first_points[last] = shared;
    second_points[0] = shared;
    first_points.extend_from_slice(&second_points[1..]);
    let (width, width_samples) = weighted_graph_width(&[first, second]);
    let role =
        if first.role == "bright-ridge-on-boundary" || second.role == "bright-ridge-on-boundary" {
            "bright-ridge-on-boundary"
        } else if first.role == "coloured-ridge-on-boundary"
            || second.role == "coloured-ridge-on-boundary"
        {
            "coloured-ridge-on-boundary"
        } else if first.role == "ridge-on-boundary" || second.role == "ridge-on-boundary" {
            "ridge-on-boundary"
        } else if first.role == "dark-boundary" || second.role == "dark-boundary" {
            "dark-boundary"
        } else {
            "ridge"
        };
    StructuralStroke {
        points: first_points,
        path_data: None,
        precise_points: None,
        color: first.color,
        width,
        role,
        width_samples,
    }
}

fn connect_graph_edges(
    mut current: Vec<StructuralStroke>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_gap: f32,
    maximum_angle_degrees: f32,
) -> Vec<StructuralStroke> {
    current.retain(|edge| edge.points.len() >= 2);
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    loop {
        if current.len() < 2 {
            break;
        }
        let records = current
            .iter()
            .enumerate()
            .flat_map(|(edge, stroke)| {
                [true, false].into_iter().map(move |at_start| {
                    let (point, tangent) = graph_endpoint(stroke, at_start);
                    (edge, at_start, point, tangent)
                })
            })
            .collect::<Vec<_>>();
        let points = records.iter().map(|record| record.2).collect::<Vec<_>>();
        let mut candidates = Vec::<(f32, usize, bool, usize, bool)>::new();
        for (first_record, second_record) in nearby_point_pairs(&points, maximum_gap) {
            let (first, first_at_start, first_point, first_tangent) = records[first_record];
            let (second, second_at_start, second_point, second_tangent) = records[second_record];
            if first == second {
                continue;
            }
            let dx = second_point.x - first_point.x;
            let dy = second_point.y - first_point.y;
            let distance = dx.hypot(dy);
            let opposing =
                -(first_tangent.0 * second_tangent.0 + first_tangent.1 * second_tangent.1);
            let coincident_radius =
                (0.5 * (current[first].width + current[second].width) + 0.5).max(2.25);
            let (first_alignment, second_alignment) = if distance <= coincident_radius {
                if opposing < cosine {
                    continue;
                }
                (1.0, 1.0)
            } else {
                let direction = (dx / distance, dy / distance);
                let first_alignment = first_tangent.0 * direction.0 + first_tangent.1 * direction.1;
                let second_alignment =
                    -(second_tangent.0 * direction.0 + second_tangent.1 * direction.1);
                if first_alignment.min(second_alignment).min(opposing) < cosine {
                    continue;
                }
                (first_alignment, second_alignment)
            };
            if !graph_bridge_supported(first_point, second_point, support, width, height) {
                continue;
            }
            let score = distance + 2.0 * (3.0 - first_alignment - second_alignment - opposing);
            candidates.push((score, first, first_at_start, second, second_at_start));
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|first, second| {
            first
                .0
                .total_cmp(&second.0)
                .then(first.1.cmp(&second.1))
                .then(first.2.cmp(&second.2))
                .then(first.3.cmp(&second.3))
                .then(first.4.cmp(&second.4))
        });
        let mut best = HashMap::<(usize, bool), (f32, (usize, bool))>::new();
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
        let mut used = vec![false; current.len()];
        for &(_, first, first_start, second, second_start) in &candidates {
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
        let mut retained = current
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(_, edge)| edge.clone())
            .collect::<Vec<_>>();
        for (first, first_start, second, second_start) in selected {
            retained.push(merge_graph_edges(
                &current[first],
                first_start,
                &current[second],
                second_start,
            ));
        }
        current = retained;
    }
    current
}

fn graph_point_tangents(points: &[Point]) -> Vec<(f32, f32)> {
    if points.len() <= 1 {
        return vec![(0.0, 0.0); points.len()];
    }
    let reach = 3.min(points.len() - 1);
    (0..points.len())
        .map(|index| {
            let first = points[index.saturating_sub(reach)];
            let last = points[(index + reach).min(points.len() - 1)];
            let dx = last.x - first.x;
            let dy = last.y - first.y;
            let length = dx.hypot(dy);
            if length > 1e-10 {
                (dx / length, dy / length)
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
}

fn point_key(point: Point) -> (i64, i64) {
    (
        (point.x * 1_000_000.0).round() as i64,
        (point.y * 1_000_000.0).round() as i64,
    )
}

/// Select geometric straight-through continuations at graph junctions and
/// return a directed path tangent for each participating endpoint.  This is
/// the deterministic counterpart of the topology-aware stroke initialization
/// used by recent line-art vectorizers: it does not guess semantics, and only
/// pairs branches whose local directions already agree within 50 degrees.
fn graph_continuation_tangents(strokes: &[StructuralStroke]) -> HashMap<(usize, bool), Point> {
    let mut groups = HashMap::<(i64, i64), Vec<(usize, bool, Point)>>::new();
    for (stroke_index, stroke) in strokes.iter().enumerate() {
        if stroke.points.len() < 2 {
            continue;
        }
        for at_start in [true, false] {
            let (endpoint, tangent) = graph_endpoint(stroke, at_start);
            groups.entry(point_key(endpoint)).or_default().push((
                stroke_index,
                at_start,
                Point {
                    x: tangent.0,
                    y: tangent.1,
                },
            ));
        }
    }

    let minimum_opposition = 50.0_f32.to_radians().cos();
    let mut result = HashMap::new();
    for members in groups.values().filter(|members| members.len() >= 2) {
        let mut candidates = Vec::<(f32, usize, usize)>::new();
        for first in 0..members.len() {
            for second in first + 1..members.len() {
                if members[first].0 == members[second].0 {
                    continue;
                }
                let opposition = -(members[first].2.x * members[second].2.x
                    + members[first].2.y * members[second].2.y);
                if opposition >= minimum_opposition {
                    candidates.push((opposition, first, second));
                }
            }
        }
        candidates.sort_by(|first, second| {
            second
                .0
                .total_cmp(&first.0)
                .then(first.1.cmp(&second.1))
                .then(first.2.cmp(&second.2))
        });
        let mut used = HashSet::new();
        for (_, first, second) in candidates {
            if used.contains(&first) || used.contains(&second) {
                continue;
            }
            let mut axis = Point {
                x: members[first].2.x - members[second].2.x,
                y: members[first].2.y - members[second].2.y,
            };
            let length = axis.x.hypot(axis.y);
            if length <= 1e-8 {
                continue;
            }
            axis.x /= length;
            axis.y /= length;
            if axis.x * members[first].2.x + axis.y * members[first].2.y < 0.0 {
                axis.x = -axis.x;
                axis.y = -axis.y;
            }
            for (member, outward) in [
                (first, axis),
                (
                    second,
                    Point {
                        x: -axis.x,
                        y: -axis.y,
                    },
                ),
            ] {
                let (stroke, at_start, _) = members[member];
                let directed = if at_start {
                    Point {
                        x: -outward.x,
                        y: -outward.y,
                    }
                } else {
                    outward
                };
                result.insert((stroke, at_start), directed);
            }
            used.extend([first, second]);
        }
    }
    result
}

/// Move centre-line samples by at most half a pixel toward the source colour
/// response along the local normal.  The profile search is followed by a
/// three-sample median, which removes alternating pixel-centre jitter without
/// moving graph endpoints; high-turn samples themselves are never scored.
fn refine_stroke_centerline(
    stroke: &StructuralStroke,
    source_lab: &[Lab],
    width: usize,
    height: usize,
) -> Vec<Point> {
    if stroke.points.len() < 5
        || !matches!(
            stroke.role,
            "ridge" | "bright-ridge-on-boundary" | "legacy-structural"
        )
    {
        return stroke.points.clone();
    }
    let target = rgb_to_lab(stroke.color);
    let tangents = graph_point_tangents(&stroke.points);
    let offsets = [-0.5_f32, -0.25, 0.0, 0.25, 0.5];
    let mut selected = vec![0.0_f32; stroke.points.len()];
    for index in 1..stroke.points.len() - 1 {
        let tangent = tangents[index];
        if tangent.0 == 0.0 && tangent.1 == 0.0 {
            continue;
        }
        let point = stroke.points[index];
        let incoming = stroke.points[index - 1];
        let outgoing = stroke.points[index + 1];
        let left_length = incoming.distance(point);
        let right_length = outgoing.distance(point);
        if left_length > 1e-8 && right_length > 1e-8 {
            let left = (
                (point.x - incoming.x) / left_length,
                (point.y - incoming.y) / left_length,
            );
            let right = (
                (outgoing.x - point.x) / right_length,
                (outgoing.y - point.y) / right_length,
            );
            if left.0 * right.0 + left.1 * right.1 < 45.0_f32.to_radians().cos() {
                continue;
            }
        }
        let normal = (-tangent.1, tangent.0);
        let score = |offset: f32| {
            let sample = bilinear_lab_precise(
                source_lab,
                width,
                height,
                [
                    point.x as f64 + offset as f64 * normal.0 as f64,
                    point.y as f64 + offset as f64 * normal.1 as f64,
                ],
            );
            delta_e2000(sample, target) + 0.75 * offset * offset
        };
        let centre_score = score(0.0);
        let (best_offset, best_score) = offsets
            .iter()
            .copied()
            .map(|offset| (offset, score(offset)))
            .min_by(|first, second| first.1.total_cmp(&second.1))
            .unwrap_or((0.0, centre_score));
        if best_score + 0.15 < centre_score {
            selected[index] = best_offset;
        }
    }
    let mut regularized = selected.clone();
    for index in 1..selected.len() - 1 {
        let mut neighbourhood = [selected[index - 1], selected[index], selected[index + 1]];
        neighbourhood.sort_by(f32::total_cmp);
        regularized[index] = neighbourhood[1];
    }
    stroke
        .points
        .iter()
        .zip(tangents)
        .zip(regularized)
        .map(|((&point, tangent), offset)| Point {
            x: point.x - offset * tangent.1,
            y: point.y + offset * tangent.0,
        })
        .collect()
}

fn straight_graph_line(
    points: &[Point],
    tolerance: f32,
    minimum_length: f32,
) -> Option<(Point, Point)> {
    if points.len() < 2 {
        return None;
    }
    let divisor = points.len() as f64;
    let centre_x = points.iter().map(|point| point.x as f64).sum::<f64>() / divisor;
    let centre_y = points.iter().map(|point| point.y as f64).sum::<f64>() / divisor;
    let (mut xx, mut xy, mut yy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for point in points {
        let dx = point.x as f64 - centre_x;
        let dy = point.y as f64 - centre_y;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let direction = if xx == yy && xy == 0.0 {
        (0.0_f64, 1.0_f64)
    } else {
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        (angle.cos(), angle.sin())
    };
    let projections = points
        .iter()
        .map(|point| {
            (point.x as f64 - centre_x) * direction.0 + (point.y as f64 - centre_y) * direction.1
        })
        .collect::<Vec<_>>();
    let minimum = projections.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = projections
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut start = Point {
        x: (centre_x + minimum * direction.0) as f32,
        y: (centre_y + minimum * direction.1) as f32,
    };
    let mut end = Point {
        x: (centre_x + maximum * direction.0) as f32,
        y: (centre_y + maximum * direction.1) as f32,
    };
    let chord_x = (maximum - minimum) * direction.0;
    let chord_y = (maximum - minimum) * direction.1;
    let length = chord_x.hypot(chord_y);
    if length < minimum_length.max(0.0) as f64 {
        return None;
    }
    let mut distances = points
        .iter()
        .map(|point| {
            (chord_x * (point.y as f64 - start.y as f64)
                - chord_y * (point.x as f64 - start.x as f64))
                .abs()
                / length.max(1e-10)
        })
        .collect::<Vec<_>>();
    distances.sort_by(f64::total_cmp);
    let position = 0.95 * distances.len().saturating_sub(1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let distance_95 = if low == high {
        distances[low]
    } else {
        let amount = position - low as f64;
        distances[low] * (1.0 - amount) + distances[high] * amount
    };
    if distance_95 > tolerance.max(0.0) as f64 {
        return None;
    }
    if (end.x - start.x) * (points[points.len() - 1].x - points[0].x)
        + (end.y - start.y) * (points[points.len() - 1].y - points[0].y)
        < 0.0
    {
        std::mem::swap(&mut start, &mut end);
    }
    Some((start, end))
}

fn median_rgb(mut samples: Vec<[f32; 3]>) -> [f32; 3] {
    if samples.is_empty() {
        return [0.0; 3];
    }
    let mut result = [0.0_f32; 3];
    for channel in 0..3 {
        samples.sort_by(|first, second| first[channel].total_cmp(&second[channel]));
        let middle = samples.len() / 2;
        result[channel] = if samples.len().is_multiple_of(2) {
            0.5 * (samples[middle - 1][channel] + samples[middle][channel])
        } else {
            samples[middle][channel]
        };
    }
    result
}

fn sample_graph_color(source: &Raster, stroke: &StructuralStroke) -> [f32; 3] {
    let indices = stroke
        .points
        .iter()
        .map(|point| {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, source.width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, source.height.saturating_sub(1) as f32) as usize;
            y * source.width + x
        })
        .collect::<Vec<_>>();
    let mut samples = indices
        .iter()
        .map(|&index| source.pixels[index])
        .collect::<Vec<_>>();
    if stroke.role == "coloured-ridge-on-boundary" && samples.len() > 2 {
        let luminance = samples
            .iter()
            .map(|pixel| 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])
            .collect::<Vec<_>>();
        let core_count = samples
            .len()
            .min(((0.40 * samples.len() as f32).ceil() as usize).max(2));
        let mut order = (0..samples.len()).collect::<Vec<_>>();
        order.sort_by(|&first, &second| {
            luminance[first]
                .total_cmp(&luminance[second])
                .then(first.cmp(&second))
        });
        let core = order[..core_count]
            .iter()
            .map(|&index| samples[index])
            .collect::<Vec<_>>();
        let mut all_luminance = luminance.clone();
        let mut core_luminance = order[..core_count]
            .iter()
            .map(|&index| luminance[index])
            .collect::<Vec<_>>();
        let median = |values: &mut Vec<f32>| {
            values.sort_by(f32::total_cmp);
            let middle = values.len() / 2;
            if values.len().is_multiple_of(2) {
                0.5 * (values[middle - 1] + values[middle])
            } else {
                values[middle]
            }
        };
        if median(&mut all_luminance) - median(&mut core_luminance) >= 0.15 {
            samples = core;
        }
    }
    if stroke.role == "bright-ridge-on-boundary" {
        samples = indices
            .iter()
            .map(|&index| {
                let x = index % source.width;
                let y = index / source.width;
                let mut brightest = source.pixels[index];
                let mut maximum =
                    0.2126 * brightest[0] + 0.7152 * brightest[1] + 0.0722 * brightest[2];
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let px = (x as isize + dx).clamp(0, source.width.saturating_sub(1) as isize)
                            as usize;
                        let py = (y as isize + dy)
                            .clamp(0, source.height.saturating_sub(1) as isize)
                            as usize;
                        let candidate = source.pixels[py * source.width + px];
                        let luminance =
                            0.2126 * candidate[0] + 0.7152 * candidate[1] + 0.0722 * candidate[2];
                        if luminance > maximum {
                            maximum = luminance;
                            brightest = candidate;
                        }
                    }
                }
                brightest
            })
            .collect();
    } else if !matches!(
        stroke.role,
        "legacy-structural" | "coloured-ridge-on-boundary"
    ) {
        samples = indices
            .iter()
            .map(|&index| {
                let x = index % source.width;
                let y = index / source.width;
                let mut darkest = source.pixels[index];
                let mut minimum = 0.2126 * darkest[0] + 0.7152 * darkest[1] + 0.0722 * darkest[2];
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let px = (x as isize + dx).clamp(0, source.width.saturating_sub(1) as isize)
                            as usize;
                        let py = (y as isize + dy)
                            .clamp(0, source.height.saturating_sub(1) as isize)
                            as usize;
                        let candidate = source.pixels[py * source.width + px];
                        let luminance =
                            0.2126 * candidate[0] + 0.7152 * candidate[1] + 0.0722 * candidate[2];
                        if luminance < minimum {
                            minimum = luminance;
                            darkest = candidate;
                        }
                    }
                }
                darkest
            })
            .collect();
    }
    median_rgb(samples)
}

fn remove_graph_aligned_overlap(
    edges: Vec<StructuralStroke>,
    owners: &[StructuralStroke],
    maximum_distance: f32,
    maximum_angle_degrees: f32,
    include_stroke_envelope: bool,
) -> Vec<StructuralStroke> {
    let mut owner_values = Vec::<(Point, (f32, f32), f32)>::new();
    for owner in owners.iter().filter(|owner| !owner.points.is_empty()) {
        for ((&point, tangent), _) in owner
            .points
            .iter()
            .zip(graph_point_tangents(&owner.points))
            .zip(0..)
        {
            owner_values.push((point, tangent, owner.width.max(0.0)));
        }
    }
    if owner_values.is_empty() {
        return edges;
    }
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    let maximum_owner_width = owner_values
        .iter()
        .map(|value| value.2)
        .fold(0.0_f32, f32::max);
    let neighbour_count = owner_values.len().min(16);
    // Querying every owner point for every candidate point dominated dense
    // photographs. A fixed grid only changes candidate discovery: the final
    // distance/index sort below retains the exact dense-scan ordering.
    let cell_size = maximum_distance.max(1.0);
    let cell_key = |point: Point| {
        (
            (point.x / cell_size).floor() as i32,
            (point.y / cell_size).floor() as i32,
        )
    };
    let mut owner_cells = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, &(point, _, _)) in owner_values.iter().enumerate() {
        owner_cells.entry(cell_key(point)).or_default().push(index);
    }
    let mut result = Vec::new();
    for edge in edges {
        if edge.points.len() < 2 {
            continue;
        }
        let tangents = graph_point_tangents(&edge.points);
        let query_distance = maximum_distance.max(0.0)
            + if include_stroke_envelope {
                0.5 * (edge.width.max(0.0) + maximum_owner_width)
            } else {
                0.0
            };
        let mut keep = Vec::with_capacity(edge.points.len());
        let mut neighbours = Vec::<(f32, usize)>::new();
        for (&point, tangent) in edge.points.iter().zip(tangents) {
            neighbours.clear();
            let minimum_cell_x = ((point.x - query_distance) / cell_size).floor() as i32;
            let maximum_cell_x = ((point.x + query_distance) / cell_size).floor() as i32;
            let minimum_cell_y = ((point.y - query_distance) / cell_size).floor() as i32;
            let maximum_cell_y = ((point.y + query_distance) / cell_size).floor() as i32;
            for cell_y in minimum_cell_y..=maximum_cell_y {
                for cell_x in minimum_cell_x..=maximum_cell_x {
                    let Some(indices) = owner_cells.get(&(cell_x, cell_y)) else {
                        continue;
                    };
                    for &index in indices {
                        let distance = point.distance(owner_values[index].0);
                        if distance <= query_distance {
                            neighbours.push((distance, index));
                        }
                    }
                }
            }
            neighbours
                .sort_by(|first, second| first.0.total_cmp(&second.0).then(first.1.cmp(&second.1)));
            let aligned = neighbours
                .iter()
                .take(neighbour_count)
                .any(|&(distance, index)| {
                    let (_, owner_tangent, owner_width) = owner_values[index];
                    let local_limit = maximum_distance.max(0.0)
                        + if include_stroke_envelope {
                            0.5 * (edge.width.max(0.0) + owner_width)
                        } else {
                            0.0
                        };
                    distance <= local_limit
                        && (tangent.0 * owner_tangent.0 + tangent.1 * owner_tangent.1).abs()
                            >= cosine
                });
            keep.push(!aligned);
        }
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
            if last + 1 < edge.points.len() {
                last += 1;
            }
            if last + 1 - first < 2 {
                continue;
            }
            result.push(StructuralStroke {
                points: edge.points[first..=last].to_vec(),
                path_data: None,
                precise_points: None,
                color: edge.color,
                width: edge.width,
                role: edge.role,
                width_samples: vec![(edge.width, last + 1 - first)],
            });
        }
    }
    result
}

fn snap_graph_intersections(
    mut current: Vec<StructuralStroke>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_extension: f32,
    minimum_angle_degrees: f32,
) -> Vec<StructuralStroke> {
    if current.len() < 2 || maximum_extension <= 0.0 {
        return current;
    }
    let junction_support = dilate(support, width, height, 1);
    let minimum_sine = minimum_angle_degrees.clamp(0.0, 90.0).to_radians().sin();
    let records = current
        .iter()
        .enumerate()
        .flat_map(|(edge, stroke)| {
            [true, false].into_iter().map(move |at_start| {
                let (point, tangent) = graph_endpoint(stroke, at_start);
                (edge, at_start, point, tangent)
            })
        })
        .collect::<Vec<_>>();
    let points = records.iter().map(|record| record.2).collect::<Vec<_>>();
    let mut candidates = Vec::<(f32, usize, usize, Point)>::new();
    for (first, second) in nearby_point_pairs(&points, 2.0 * maximum_extension) {
        let (first_edge, _, first_point, first_tangent) = records[first];
        let (second_edge, _, second_point, second_tangent) = records[second];
        if first_edge == second_edge {
            continue;
        }
        let cross = first_tangent.0 * second_tangent.1 - first_tangent.1 * second_tangent.0;
        if cross.abs() < minimum_sine {
            continue;
        }
        let delta = (
            second_point.x - first_point.x,
            second_point.y - first_point.y,
        );
        let first_distance = (delta.0 * second_tangent.1 - delta.1 * second_tangent.0) / cross;
        let second_distance = (first_tangent.1 * delta.0 - first_tangent.0 * delta.1) / cross;
        if first_distance < -1e-6
            || second_distance < -1e-6
            || first_distance > maximum_extension
            || second_distance > maximum_extension
        {
            continue;
        }
        let first_node = Point {
            x: first_point.x + first_distance * first_tangent.0,
            y: first_point.y + first_distance * first_tangent.1,
        };
        let second_node = Point {
            x: second_point.x + second_distance * second_tangent.0,
            y: second_point.y + second_distance * second_tangent.1,
        };
        let node = Point {
            x: 0.5 * (first_node.x + second_node.x),
            y: 0.5 * (first_node.y + second_node.y),
        };
        if !graph_bridge_supported(first_point, node, &junction_support, width, height)
            || !graph_bridge_supported(second_point, node, &junction_support, width, height)
        {
            continue;
        }
        candidates.push((first_distance + second_distance, first, second, node));
    }
    let mut best = HashMap::<usize, (f32, usize)>::new();
    for (index, &(score, first, second, _)) in candidates.iter().enumerate() {
        if best
            .get(&first)
            .map(|value| score < value.0)
            .unwrap_or(true)
        {
            best.insert(first, (score, index));
        }
        if best
            .get(&second)
            .map(|value| score < value.0)
            .unwrap_or(true)
        {
            best.insert(second, (score, index));
        }
    }
    let mut replacements = HashMap::<(usize, bool), Point>::new();
    let mut used = HashSet::new();
    for (index, &(_, first, second, node)) in candidates.iter().enumerate() {
        if used.contains(&first)
            || used.contains(&second)
            || best.get(&first).map(|value| value.1) != Some(index)
            || best.get(&second).map(|value| value.1) != Some(index)
        {
            continue;
        }
        replacements.insert((records[first].0, records[first].1), node);
        replacements.insert((records[second].0, records[second].1), node);
        used.insert(first);
        used.insert(second);
    }
    for (edge_index, edge) in current.iter_mut().enumerate() {
        if let Some(&point) = replacements.get(&(edge_index, true)) {
            edge.points[0] = point;
        }
        if let Some(&point) = replacements.get(&(edge_index, false)) {
            let last = edge.points.len() - 1;
            edge.points[last] = point;
        }
    }
    current
}

fn snap_graph_mutual_continuations(
    mut current: Vec<StructuralStroke>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_gap: f32,
    maximum_angle_degrees: f32,
    anchor_role: &'static str,
) -> Vec<StructuralStroke> {
    if current.len() < 2 || maximum_gap <= 0.0 {
        return current;
    }
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    let records = current
        .iter()
        .enumerate()
        .flat_map(|(edge, stroke)| {
            [true, false].into_iter().map(move |at_start| {
                let (point, tangent) = graph_endpoint(stroke, at_start);
                (edge, at_start, point, tangent)
            })
        })
        .collect::<Vec<_>>();
    let points = records.iter().map(|record| record.2).collect::<Vec<_>>();
    let mut candidates = Vec::<(f32, usize, usize)>::new();
    for (first, second) in nearby_point_pairs(&points, maximum_gap) {
        let (first_edge, _, first_point, first_tangent) = records[first];
        let (second_edge, _, second_point, second_tangent) = records[second];
        if first_edge == second_edge {
            continue;
        }
        let dx = second_point.x - first_point.x;
        let dy = second_point.y - first_point.y;
        let distance = dx.hypot(dy);
        let opposing = -(first_tangent.0 * second_tangent.0 + first_tangent.1 * second_tangent.1);
        let (first_alignment, second_alignment) = if distance <= 1e-8 {
            if opposing < cosine {
                continue;
            }
            (1.0, 1.0)
        } else {
            let direction = (dx / distance, dy / distance);
            let first_alignment = first_tangent.0 * direction.0 + first_tangent.1 * direction.1;
            let second_alignment =
                -(second_tangent.0 * direction.0 + second_tangent.1 * direction.1);
            if first_alignment.min(second_alignment).min(opposing) < cosine {
                continue;
            }
            (first_alignment, second_alignment)
        };
        if !graph_bridge_supported(first_point, second_point, support, width, height) {
            continue;
        }
        candidates.push((
            distance + 2.0 * (3.0 - first_alignment - second_alignment - opposing),
            first,
            second,
        ));
    }
    let mut parents = (0..current.len()).collect::<Vec<_>>();
    fn root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }
    for &(_, first, second) in &candidates {
        let first_root = root(&mut parents, records[first].0);
        let second_root = root(&mut parents, records[second].0);
        if first_root != second_root {
            parents[second_root] = first_root;
        }
    }
    let anchored = current
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| (edge.role == anchor_role).then_some(index))
        .map(|index| root(&mut parents, index))
        .collect::<HashSet<_>>();
    candidates.retain(|candidate| anchored.contains(&root(&mut parents, records[candidate.1].0)));
    let mut best = HashMap::<usize, (f32, usize)>::new();
    for &(score, first, second) in &candidates {
        if best
            .get(&first)
            .map(|value| score < value.0)
            .unwrap_or(true)
        {
            best.insert(first, (score, second));
        }
        if best
            .get(&second)
            .map(|value| score < value.0)
            .unwrap_or(true)
        {
            best.insert(second, (score, first));
        }
    }
    candidates.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then(first.1.cmp(&second.1))
            .then(first.2.cmp(&second.2))
    });
    let mut replacements = HashMap::<(usize, bool), Point>::new();
    for &(_, first, second) in &candidates {
        if best.get(&first).map(|value| value.1) != Some(second)
            || best.get(&second).map(|value| value.1) != Some(first)
        {
            continue;
        }
        let midpoint = Point {
            x: 0.5 * (records[first].2.x + records[second].2.x),
            y: 0.5 * (records[first].2.y + records[second].2.y),
        };
        replacements.insert((records[first].0, records[first].1), midpoint);
        replacements.insert((records[second].0, records[second].1), midpoint);
    }
    for (edge_index, edge) in current.iter_mut().enumerate() {
        if let Some(&point) = replacements.get(&(edge_index, true)) {
            edge.points[0] = point;
        }
        if let Some(&point) = replacements.get(&(edge_index, false)) {
            let last = edge.points.len() - 1;
            edge.points[last] = point;
        }
    }
    current
}

fn shortest_skeleton_route(
    skeleton: &[bool],
    width: usize,
    height: usize,
    first: usize,
    second: usize,
    maximum_length: f32,
) -> Option<(f32, Vec<usize>)> {
    if first == second {
        return Some((0.0, vec![first]));
    }
    let maximum = maximum_length.max(0.0);
    let mut distances = HashMap::<usize, f32>::from([(first, 0.0)]);
    let mut previous = HashMap::<usize, usize>::new();
    let mut pending = vec![(0.0_f32, first)];
    let neighbours = [
        (-1_isize, -1_isize, std::f32::consts::SQRT_2),
        (-1, 0, 1.0),
        (-1, 1, std::f32::consts::SQRT_2),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, std::f32::consts::SQRT_2),
        (1, 0, 1.0),
        (1, 1, std::f32::consts::SQRT_2),
    ];
    while !pending.is_empty() {
        let next = pending
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first.0.total_cmp(&second.0).then(first.1.cmp(&second.1))
            })
            .map(|value| value.0)
            .unwrap_or(0);
        let (distance, pixel) = pending.swap_remove(next);
        if distances.get(&pixel).copied() != Some(distance) {
            continue;
        }
        if pixel == second {
            break;
        }
        let row = pixel / width;
        let column = pixel % width;
        for (row_delta, column_delta, step) in neighbours {
            let next_row = row as isize + row_delta;
            let next_column = column as isize + column_delta;
            if next_row < 0
                || next_row >= height as isize
                || next_column < 0
                || next_column >= width as isize
            {
                continue;
            }
            let neighbour = next_row as usize * width + next_column as usize;
            let next_distance = distance + step;
            if next_distance > maximum
                || !skeleton[neighbour]
                || next_distance >= distances.get(&neighbour).copied().unwrap_or(f32::INFINITY)
            {
                continue;
            }
            distances.insert(neighbour, next_distance);
            previous.insert(neighbour, pixel);
            pending.push((next_distance, neighbour));
        }
    }
    let distance = distances.get(&second).copied()?;
    let mut route = vec![second];
    while route.last().copied() != Some(first) {
        route.push(*previous.get(route.last().unwrap())?);
    }
    route.reverse();
    Some((distance, route))
}

fn point_along_graph_polyline(points: &[Point], distance: f32) -> Point {
    if points.len() < 2 {
        return points[0];
    }
    let target = distance.max(0.0);
    let mut cumulative = 0.0_f32;
    for pair in points.windows(2) {
        let length = pair[0].distance(pair[1]);
        if cumulative + length >= target && length > 1e-10 {
            let amount = (target - cumulative) / length;
            return Point {
                x: pair[0].x + amount * (pair[1].x - pair[0].x),
                y: pair[0].y + amount * (pair[1].y - pair[0].y),
            };
        }
        cumulative += length;
    }
    points[points.len() - 1]
}

fn split_graph_polyline_midpoint(points: &[Point]) -> (Vec<Point>, Vec<Point>) {
    let target = 0.5
        * points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum::<f32>();
    let mut cumulative = 0.0_f32;
    for (index, pair) in points.windows(2).enumerate() {
        let length = pair[0].distance(pair[1]);
        if cumulative + length + 1e-10 < target {
            cumulative += length;
            continue;
        }
        let amount = if length <= 1e-10 {
            0.0
        } else {
            (target - cumulative) / length
        };
        let midpoint = Point {
            x: pair[0].x + amount * (pair[1].x - pair[0].x),
            y: pair[0].y + amount * (pair[1].y - pair[0].y),
        };
        let mut first = points[..=index].to_vec();
        first.push(midpoint);
        let mut second = points[index + 1..]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        second.push(midpoint);
        return (first, second);
    }
    (points.to_vec(), vec![points[points.len() - 1]])
}

#[allow(clippy::too_many_arguments)]
fn extend_graph_mutual_supported_continuations(
    mut current: Vec<StructuralStroke>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_gap: f32,
    maximum_angle_degrees: f32,
    maximum_path_stretch: f32,
    maximum_projection_distance: f32,
    anchor_role: &'static str,
) -> Vec<StructuralStroke> {
    if current.len() < 2 || maximum_gap <= 0.0 {
        return current;
    }
    let source_skeleton = skeletonize(support, width, height);
    let skeleton_indices = source_skeleton
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| selected.then_some(index))
        .collect::<Vec<_>>();
    if skeleton_indices.is_empty() {
        return current;
    }
    let records = current
        .iter()
        .enumerate()
        .flat_map(|(edge_index, edge)| {
            [true, false].into_iter().map(move |at_start| {
                let (point, tangent) = graph_endpoint(edge, at_start);
                (edge_index, at_start, point, tangent)
            })
        })
        .collect::<Vec<_>>();
    let point_key = |point: Point| {
        (
            (point.x * 1_000_000.0).round() as i64,
            (point.y * 1_000_000.0).round() as i64,
        )
    };
    let connection_key = |first: Point, second: Point| {
        let first = point_key(first);
        let second = point_key(second);
        if first <= second {
            (first, second)
        } else {
            (second, first)
        }
    };
    let existing_connections = current
        .iter()
        .map(|edge| connection_key(edge.points[0], edge.points[edge.points.len() - 1]))
        .collect::<HashSet<_>>();
    let nearest_skeleton = |point: Point| {
        skeleton_indices
            .iter()
            .copied()
            .map(|index| {
                let candidate = Point {
                    x: (index % width) as f32 + 0.5,
                    y: (index / width) as f32 + 0.5,
                };
                (point.distance(candidate), index)
            })
            .filter(|value| value.0 <= maximum_projection_distance.max(0.0))
            .min_by(|first, second| first.0.total_cmp(&second.0).then(first.1.cmp(&second.1)))
    };
    let skeleton_degree = |index: usize| {
        let row = index / width;
        let column = index % width;
        let mut degree = 0_usize;
        for row_delta in -1_isize..=1 {
            for column_delta in -1_isize..=1 {
                if row_delta == 0 && column_delta == 0 {
                    continue;
                }
                let next_row = row as isize + row_delta;
                let next_column = column as isize + column_delta;
                if next_row >= 0
                    && next_row < height as isize
                    && next_column >= 0
                    && next_column < width as isize
                    && source_skeleton[next_row as usize * width + next_column as usize]
                {
                    degree += 1;
                }
            }
        }
        degree
    };
    let stretch_limit = maximum_path_stretch.max(1.0);
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    let mut candidates = Vec::<(f32, usize, usize, Vec<Point>)>::new();
    let record_points = records.iter().map(|record| record.2).collect::<Vec<_>>();
    for (first, second) in nearby_point_pairs(&record_points, maximum_gap) {
        if records[first].0 == records[second].0 {
            continue;
        }
        let distance = records[first].2.distance(records[second].2);
        if distance <= 1e-8 {
            continue;
        }
        if existing_connections.contains(&connection_key(records[first].2, records[second].2)) {
            continue;
        }
        let Some((first_projection, first_pixel)) = nearest_skeleton(records[first].2) else {
            continue;
        };
        let Some((second_projection, second_pixel)) = nearest_skeleton(records[second].2) else {
            continue;
        };
        let maximum_route =
            (stretch_limit * distance - first_projection - second_projection).max(0.0);
        let Some((route_length, route_pixels)) = shortest_skeleton_route(
            &source_skeleton,
            width,
            height,
            first_pixel,
            second_pixel,
            maximum_route,
        ) else {
            continue;
        };
        let geodesic_length = first_projection + route_length + second_projection;
        if geodesic_length > stretch_limit * distance + 1e-8
            || route_pixels
                .iter()
                .skip(1)
                .take(route_pixels.len().saturating_sub(2))
                .any(|&index| skeleton_degree(index) >= 4)
        {
            continue;
        }
        let mut route_points = vec![records[first].2];
        route_points.extend(route_pixels.iter().map(|&index| Point {
            x: (index % width) as f32 + 0.5,
            y: (index / width) as f32 + 0.5,
        }));
        route_points.push(records[second].2);
        route_points.dedup_by(|first, second| first.distance(*second) <= 1e-8);
        if route_points.len() < 2 {
            continue;
        }
        let reach = 3.0_f32.min(0.35 * geodesic_length);
        let first_direction_point = point_along_graph_polyline(&route_points, reach);
        let reversed = route_points.iter().rev().copied().collect::<Vec<_>>();
        let second_direction_point = point_along_graph_polyline(&reversed, reach);
        let unit = |dx: f32, dy: f32| {
            let length = dx.hypot(dy);
            if length > 1e-10 {
                (dx / length, dy / length)
            } else {
                (0.0, 0.0)
            }
        };
        let first_direction = unit(
            first_direction_point.x - records[first].2.x,
            first_direction_point.y - records[first].2.y,
        );
        let second_direction = unit(
            second_direction_point.x - records[second].2.x,
            second_direction_point.y - records[second].2.y,
        );
        let first_alignment =
            records[first].3 .0 * first_direction.0 + records[first].3 .1 * first_direction.1;
        let second_alignment =
            records[second].3 .0 * second_direction.0 + records[second].3 .1 * second_direction.1;
        if first_alignment.min(second_alignment) < cosine {
            continue;
        }
        let score = geodesic_length + 2.0 * (2.0 - first_alignment - second_alignment);
        candidates.push((score, first, second, route_points));
    }
    if candidates.is_empty() {
        return current;
    }
    let mut parents = (0..current.len()).collect::<Vec<_>>();
    fn edge_root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }
    for candidate in &candidates {
        let first = edge_root(&mut parents, records[candidate.1].0);
        let second = edge_root(&mut parents, records[candidate.2].0);
        if first != second {
            parents[second] = first;
        }
    }
    let anchored = current
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| (edge.role == anchor_role).then_some(index))
        .map(|index| edge_root(&mut parents, index))
        .collect::<HashSet<_>>();
    candidates
        .retain(|candidate| anchored.contains(&edge_root(&mut parents, records[candidate.1].0)));
    let mut best = HashMap::<usize, (f32, usize)>::new();
    for candidate in &candidates {
        for (first, second) in [(candidate.1, candidate.2), (candidate.2, candidate.1)] {
            if best
                .get(&first)
                .map(|value| candidate.0 < value.0)
                .unwrap_or(true)
            {
                best.insert(first, (candidate.0, second));
            }
        }
    }
    candidates.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then(first.1.cmp(&second.1))
            .then(first.2.cmp(&second.2))
    });
    let mut extensions = HashMap::<(usize, bool), Vec<Point>>::new();
    for (_, first, second, route_points) in candidates {
        if best.get(&first).map(|value| value.1) != Some(second)
            || best.get(&second).map(|value| value.1) != Some(first)
        {
            continue;
        }
        let (first_half, second_half) = split_graph_polyline_midpoint(&route_points);
        extensions.insert((records[first].0, records[first].1), first_half);
        extensions.insert((records[second].0, records[second].1), second_half);
    }
    for (edge_index, edge) in current.iter_mut().enumerate() {
        if let Some(extension) = extensions.get(&(edge_index, true)) {
            let mut points = extension.iter().rev().copied().collect::<Vec<_>>();
            points.extend_from_slice(&edge.points[1..]);
            edge.points = points;
        }
        if let Some(extension) = extensions.get(&(edge_index, false)) {
            edge.points.pop();
            edge.points.extend_from_slice(extension);
        }
    }
    current
}

fn snap_graph_junction_endpoints(
    mut current: Vec<StructuralStroke>,
    support: &[bool],
    width: usize,
    height: usize,
    maximum_gap: f32,
    maximum_angle_degrees: f32,
) -> Vec<StructuralStroke> {
    if current.len() < 2 || maximum_gap <= 0.0 {
        return current;
    }
    let junction_support = dilate(support, width, height, 1);
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    let records = current
        .iter()
        .enumerate()
        .flat_map(|(edge, stroke)| {
            [true, false].into_iter().map(move |at_start| {
                let (point, tangent) = graph_endpoint(stroke, at_start);
                (edge, at_start, point, tangent)
            })
        })
        .collect::<Vec<_>>();
    let mut parents = (0..records.len()).collect::<Vec<_>>();
    let mut members = (0..records.len())
        .map(|index| HashSet::from([index]))
        .collect::<Vec<_>>();
    fn endpoint_root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }
    let record_points = records.iter().map(|record| record.2).collect::<Vec<_>>();
    let mut pairs = Vec::<(f32, usize, usize)>::new();
    for (first, second) in nearby_point_pairs(&record_points, maximum_gap) {
        pairs.push((records[first].2.distance(records[second].2), first, second));
    }
    pairs.sort_by(|first, second| {
        first
            .0
            .total_cmp(&second.0)
            .then(first.1.cmp(&second.1))
            .then(first.2.cmp(&second.2))
    });
    for (distance, first, second) in pairs {
        if records[first].0 == records[second].0 {
            continue;
        }
        let dx = records[second].2.x - records[first].2.x;
        let dy = records[second].2.y - records[first].2.y;
        if distance > 1e-8 {
            let direction = (dx / distance, dy / distance);
            let first_alignment =
                records[first].3 .0 * direction.0 + records[first].3 .1 * direction.1;
            let second_alignment =
                -(records[second].3 .0 * direction.0 + records[second].3 .1 * direction.1);
            if first_alignment.max(second_alignment) < cosine
                || !graph_bridge_supported(
                    records[first].2,
                    records[second].2,
                    &junction_support,
                    width,
                    height,
                )
            {
                continue;
            }
        }
        let first_root = endpoint_root(&mut parents, first);
        let second_root = endpoint_root(&mut parents, second);
        if first_root == second_root {
            continue;
        }
        let compatible = members[first_root].iter().all(|&left| {
            members[second_root]
                .iter()
                .all(|&right| records[left].2.distance(records[right].2) <= maximum_gap)
        });
        if !compatible {
            continue;
        }
        parents[second_root] = first_root;
        let moved = std::mem::take(&mut members[second_root]);
        members[first_root].extend(moved);
    }
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for index in 0..records.len() {
        let root = endpoint_root(&mut parents, index);
        groups.entry(root).or_default().push(index);
    }
    let mut replacements = HashMap::<(usize, bool), Point>::new();
    for group in groups.values().filter(|group| group.len() >= 2) {
        let centre = Point {
            x: group.iter().map(|&index| records[index].2.x).sum::<f32>() / group.len() as f32,
            y: group.iter().map(|&index| records[index].2.y).sum::<f32>() / group.len() as f32,
        };
        let compatible = group.iter().all(|&index| {
            let point = records[index].2;
            let distance = point.distance(centre);
            if distance <= 1e-8 {
                return true;
            }
            let direction = (
                (centre.x - point.x) / distance,
                (centre.y - point.y) / distance,
            );
            records[index].3 .0 * direction.0 + records[index].3 .1 * direction.1 >= cosine
                && graph_bridge_supported(point, centre, &junction_support, width, height)
        });
        if compatible {
            for &index in group {
                replacements.insert((records[index].0, records[index].1), centre);
            }
        }
    }
    for (edge_index, edge) in current.iter_mut().enumerate() {
        if let Some(&point) = replacements.get(&(edge_index, true)) {
            edge.points[0] = point;
        }
        if let Some(&point) = replacements.get(&(edge_index, false)) {
            let last = edge.points.len() - 1;
            edge.points[last] = point;
        }
    }
    current
}

#[allow(clippy::too_many_arguments)]
fn snap_graph_to_paint_junctions(
    mut current: Vec<StructuralStroke>,
    nodes: &[Point],
    node_costs: &[f32],
    support: &[bool],
    width: usize,
    height: usize,
    maximum_distance: f32,
    maximum_angle_degrees: f32,
    maximum_node_cost: f32,
) -> Vec<StructuralStroke> {
    if current.is_empty()
        || nodes.is_empty()
        || nodes.len() != node_costs.len()
        || maximum_distance <= 0.0
    {
        return current;
    }
    let junction_support = dilate(support, width, height, 1);
    let cosine = maximum_angle_degrees.max(0.0).to_radians().cos();
    // Only nodes within `maximum_distance` can be selected. Indexing them in
    // cells of that size turns the former endpoints x all-nodes scan into an
    // exact 3x3-neighbourhood lookup. Candidate indices are sorted back into
    // source order so all existing tie-breaking remains unchanged.
    let cell_size = maximum_distance;
    let cell_key = |point: Point| {
        (
            (point.x / cell_size).floor() as i32,
            (point.y / cell_size).floor() as i32,
        )
    };
    let mut node_cells = HashMap::<(i32, i32), Vec<usize>>::new();
    for (node_index, &node) in nodes.iter().enumerate() {
        node_cells
            .entry(cell_key(node))
            .or_default()
            .push(node_index);
    }
    let mut endpoint_groups = HashMap::<(i64, i64), Vec<(usize, bool, Point, (f32, f32))>>::new();
    for (edge_index, edge) in current.iter().enumerate() {
        for at_start in [true, false] {
            let (point, tangent) = graph_endpoint(edge, at_start);
            endpoint_groups
                .entry((
                    (point.x * 1_000_000.0).round() as i64,
                    (point.y * 1_000_000.0).round() as i64,
                ))
                .or_default()
                .push((edge_index, at_start, point, tangent));
        }
    }
    let mut replacements = HashMap::<(usize, bool), Point>::new();
    for members in endpoint_groups
        .values()
        .filter(|members| members.len() >= 2)
    {
        let endpoint = members[0].2;
        let (cell_x, cell_y) = cell_key(endpoint);
        let mut candidate_nodes = Vec::<usize>::new();
        for y in cell_y - 1..=cell_y + 1 {
            for x in cell_x - 1..=cell_x + 1 {
                if let Some(indices) = node_cells.get(&(x, y)) {
                    candidate_nodes.extend(indices);
                }
            }
        }
        candidate_nodes.sort_unstable();
        let mut ranked = Vec::<(f32, f32, f32, usize)>::new();
        for node_index in candidate_nodes {
            let node = nodes[node_index];
            let cost = node_costs[node_index];
            let distance = endpoint.distance(node);
            if distance > maximum_distance || !cost.is_finite() || cost > maximum_node_cost {
                continue;
            }
            let mut worst_alignment_error = 0.0_f32;
            let mut compatible = true;
            for &(_, _, member_endpoint, tangent) in members {
                let member_distance = member_endpoint.distance(node);
                if member_distance <= 1e-8 {
                    continue;
                }
                let direction = (
                    (node.x - member_endpoint.x) / member_distance,
                    (node.y - member_endpoint.y) / member_distance,
                );
                let alignment = (tangent.0 * direction.0 + tangent.1 * direction.1).abs();
                if alignment < cosine
                    || !graph_bridge_supported(
                        member_endpoint,
                        node,
                        &junction_support,
                        width,
                        height,
                    )
                {
                    compatible = false;
                    break;
                }
                worst_alignment_error = worst_alignment_error.max(1.0 - alignment);
            }
            if compatible {
                ranked.push((cost, worst_alignment_error, distance, node_index));
            }
        }
        ranked.sort_by(|first, second| {
            first
                .0
                .total_cmp(&second.0)
                .then(first.1.total_cmp(&second.1))
                .then(first.2.total_cmp(&second.2))
                .then(first.3.cmp(&second.3))
        });
        if let Some(&(_, _, _, node_index)) = ranked.first() {
            for &(edge_index, at_start, _, _) in members {
                replacements.insert((edge_index, at_start), nodes[node_index]);
            }
        }
    }
    for (edge_index, edge) in current.iter_mut().enumerate() {
        if let Some(&point) = replacements.get(&(edge_index, true)) {
            edge.points[0] = point;
        }
        if let Some(&point) = replacements.get(&(edge_index, false)) {
            let last = edge.points.len() - 1;
            edge.points[last] = point;
        }
    }
    current
}

/// Resolve the single authoritative legacy source-line topology after Paint
/// ownership is known.  No speculative legacy graph is built and discarded
/// during structural analysis.
pub fn select_missing(
    source: &Raster,
    rendered: &Raster,
    structural: &StructuralInk,
) -> StructuralInk {
    select_missing_with_junctions(source, rendered, structural, &[])
}

pub fn select_missing_with_junctions(
    source: &Raster,
    rendered: &Raster,
    structural: &StructuralInk,
    paint_junctions: &[Point],
) -> StructuralInk {
    if source.width != rendered.width || source.height != rendered.height {
        return structural.clone();
    }
    let width = source.width;
    let height = source.height;
    let source_lab = lab_pixels(source);
    let rendered_lab = lab_pixels(rendered);
    let (mut residual_lines, mut measured_lines) = residual_line_masks(
        &source_lab,
        &rendered_lab,
        &structural.role_line_mask,
        width,
        height,
    );
    let residual_legacy_line_mask: Vec<bool> = residual_lines
        .iter()
        .zip(&structural.legacy_line_mask)
        .map(|(&residual, &legacy)| residual && legacy)
        .collect();
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        let selected_bytes = residual_lines
            .iter()
            .map(|&value| u8::from(value))
            .collect::<Vec<_>>();
        let _ = std::fs::write(
            format!(
                "{}-residual-lines-before-ridge.u8",
                prefix.to_string_lossy()
            ),
            selected_bytes,
        );
        for (name, mask) in [
            ("legacy-lines", &structural.legacy_line_mask),
            ("role-lines", &structural.role_line_mask),
            ("residual-legacy-lines", &residual_legacy_line_mask),
            ("visible-ridge-coverage", &structural.visible_ridge_coverage),
        ] {
            let bytes = mask
                .iter()
                .map(|&value| u8::from(value))
                .collect::<Vec<_>>();
            let _ = std::fs::write(format!("{}-{}.u8", prefix.to_string_lossy(), name), bytes);
        }
    }
    for (line, &ridge) in residual_lines
        .iter_mut()
        .zip(&structural.visible_ridge_coverage)
    {
        *line |= ridge;
    }
    for (measured, &ridge) in measured_lines
        .iter_mut()
        .zip(&structural.visible_ridge_coverage)
    {
        *measured |= ridge;
    }
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        for (name, mask) in [
            ("face-barrier", &structural.face_barrier),
            ("residual-lines-with-ridge", &residual_lines),
        ] {
            let bytes = mask
                .iter()
                .map(|&value| u8::from(value))
                .collect::<Vec<_>>();
            let _ = std::fs::write(format!("{}-{}.u8", prefix.to_string_lossy(), name), bytes);
        }
    }

    let graph_candidates = structural.strokes.to_vec();
    let mut visible_graph = Vec::new();
    let mut boundary_graph = Vec::new();
    for mut stroke in graph_candidates {
        if stroke.role == "boundary-stroke"
            || (!structural.boundary_stroke_mask.is_empty()
                && mask_fraction_along(&stroke, &structural.boundary_stroke_mask, width, height)
                    > 0.5)
        {
            continue;
        }
        if matches!(stroke.role, "ridge" | "bright-ridge-on-boundary") {
            visible_graph.push(stroke);
        } else {
            stroke.role = classify_boundary_role(&stroke, &source_lab, width, height);
            if !paint_owned_dark_boundary_undershoot(&stroke, &source_lab, width, height) {
                boundary_graph.push(stroke);
            }
        }
    }
    let mut missing_support = vec![false; width * height];
    let mut source_valley_support = vec![false; width * height];
    let mut edge_wide_support = vec![false; width * height];
    let mut boundary_flags = Vec::with_capacity(boundary_graph.len());
    for stroke in &boundary_graph {
        let (missing, valley) =
            boundary_profile_flags(stroke, &source_lab, &rendered_lab, width, height, 5.0);
        let (wide, _) = boundary_profile_flags(
            stroke,
            &source_lab,
            &rendered_lab,
            width,
            height,
            f32::INFINITY,
        );
        flags_to_mask(stroke, &missing, &mut missing_support, width, height);
        flags_to_mask(stroke, &valley, &mut source_valley_support, width, height);
        flags_to_mask(stroke, &wide, &mut edge_wide_support, width, height);
        boundary_flags.push((missing, valley));
    }
    let primary_owner_corridor = dilate_square(&structural.paint_ownership_mask, width, height, 1);
    for index in 0..missing_support.len() {
        if primary_owner_corridor[index] {
            missing_support[index] = false;
            edge_wide_support[index] = false;
        }
    }
    let scheduled_residual_corridor = dilate_square(&residual_legacy_line_mask, width, height, 1);
    for index in 0..edge_wide_support.len() {
        if scheduled_residual_corridor[index] {
            edge_wide_support[index] = false;
        }
    }

    let mut complete_profile_edges = Vec::new();
    let mut interval_edges = Vec::new();
    for (stroke, (_, valley)) in boundary_graph.into_iter().zip(boundary_flags) {
        let complete_threshold = match stroke.role {
            "coloured-ridge-on-boundary" => 0.30,
            "dark-boundary" => 0.90,
            _ => 0.60,
        };
        let wide_fraction = mask_fraction_along(&stroke, &edge_wide_support, width, height);
        if wide_fraction >= complete_threshold {
            complete_profile_edges.push(stroke);
            continue;
        }
        for interval in split_supported_runs(&stroke, &valley) {
            let threshold = if interval.role == "dark-boundary" {
                0.90
            } else {
                0.15
            };
            if mask_fraction_along(&interval, &missing_support, width, height) >= threshold {
                interval_edges.push(interval);
            }
        }
    }
    let mut selected_graph = visible_graph;
    selected_graph.extend(complete_profile_edges);
    selected_graph.extend(interval_edges);
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        let document = serde_json::json!({
            "legacy_lines": structural.legacy_line_mask.iter().filter(|&&value| value).count(),
            "role_lines": structural.role_line_mask.iter().filter(|&&value| value).count(),
            "selected_residual_lines": residual_lines.iter().filter(|&&value| value).count(),
            "measured_lines": measured_lines.iter().filter(|&&value| value).count(),
            "residual_legacy_lines": residual_legacy_line_mask.iter().filter(|&&value| value).count(),
            "missing_profile": missing_support.iter().filter(|&&value| value).count(),
            "valley_profile": source_valley_support.iter().filter(|&&value| value).count(),
            "wide_profile": edge_wide_support.iter().filter(|&&value| value).count(),
            "selected_graph": selected_graph.len(),
        });
        if let Ok(bytes) = serde_json::to_vec_pretty(&document) {
            let _ = std::fs::write(
                format!("{}-selection.json", prefix.to_string_lossy()),
                bytes,
            );
        }
    }

    let distance_sites: Vec<bool> = residual_legacy_line_mask
        .iter()
        .map(|value| !value)
        .collect();
    let nearest_background = nearest_site_indices(&distance_sites, width, height);
    let mut skeleton = skeletonize(&residual_legacy_line_mask, width, height);
    remove_small_components(&mut skeleton, width, height, 1);
    let residual_skeleton_pixels = skeleton.iter().filter(|&&value| value).count();
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        let _ = std::fs::write(
            format!("{}-residual-source-skeleton.json", prefix.to_string_lossy()),
            format!(
                "{{\"pixels\":{},\"paths\":{}}}",
                residual_skeleton_pixels,
                trace_skeleton(&skeleton, width, height).len(),
            ),
        );
    }
    let mut residual_legacy_graph = Vec::new();
    for indices in trace_skeleton(&skeleton, width, height) {
        if indices.len() < 2 {
            continue;
        }
        let mut half_widths = indices
            .iter()
            .map(|&index| {
                let nearest = nearest_background[index];
                let dx = (index % width) as f32 - (nearest % width) as f32;
                let dy = (index / width) as f32 - (nearest / width) as f32;
                dx.hypot(dy)
            })
            .collect::<Vec<_>>();
        half_widths.sort_by(f32::total_cmp);
        let half_width = if half_widths.len() % 2 == 0 {
            let middle = half_widths.len() / 2;
            0.5 * (half_widths[middle - 1] + half_widths[middle])
        } else {
            half_widths[half_widths.len() / 2]
        };
        residual_legacy_graph.push(StructuralStroke {
            points: indices
                .iter()
                .map(|&index| Point {
                    x: (index % width) as f32 + 0.5,
                    y: (index / width) as f32 + 0.5,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: median_color(source, &indices),
            width: (2.0 * half_width - 1.0).clamp(1.0, 4.0),
            role: "legacy-structural",
            width_samples: vec![((2.0 * half_width - 1.0).clamp(1.0, 4.0), indices.len())],
        });
    }

    let primary_graph = selected_graph
        .iter()
        .filter(|edge| matches!(edge.role, "ridge" | "bright-ridge-on-boundary"))
        .cloned()
        .collect::<Vec<_>>();
    let secondary_graph = selected_graph
        .into_iter()
        .filter(|edge| !matches!(edge.role, "ridge" | "bright-ridge-on-boundary"))
        .collect::<Vec<_>>();
    let secondary_graph = remove_graph_aligned_overlap(
        secondary_graph,
        &primary_graph,
        std::f32::consts::FRAC_1_SQRT_2,
        20.0,
        true,
    );
    let mut selected_graph = primary_graph;
    selected_graph.extend(secondary_graph);
    let intersection_support: Vec<bool> = residual_lines
        .iter()
        .zip(&source.pixels)
        .map(|(&selected, pixel)| {
            selected || 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2] <= 0.35
        })
        .collect();
    selected_graph = snap_graph_intersections(
        selected_graph,
        &intersection_support,
        width,
        height,
        6.0,
        20.0,
    );
    let boundary_continuation_support: Vec<bool> = residual_lines
        .iter()
        .zip(&structural.visible_ridge_coverage)
        .zip(&structural.dark_boundary_coverage)
        .zip(&structural.face_barrier)
        .zip(&residual_legacy_line_mask)
        .map(
            |((((&selected, &ridge), &dark), &barrier), &residual_legacy)| {
                selected || ridge || dark || barrier || residual_legacy
            },
        )
        .collect();
    let boundary_indices = selected_graph
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            matches!(
                edge.role,
                "ridge-on-boundary" | "coloured-ridge-on-boundary" | "dark-boundary"
            )
            .then_some(index)
        })
        .collect::<Vec<_>>();
    let boundary_values = boundary_indices
        .iter()
        .map(|&index| selected_graph[index].clone())
        .collect::<Vec<_>>();
    let boundary_values = snap_graph_mutual_continuations(
        boundary_values,
        &boundary_continuation_support,
        width,
        height,
        6.5 * (width.max(height) as f32 / 1024.0).max(1.0),
        50.0,
        "coloured-ridge-on-boundary",
    );
    let boundary_values = extend_graph_mutual_supported_continuations(
        boundary_values,
        &boundary_continuation_support,
        width,
        height,
        6.5 * (width.max(height) as f32 / 1024.0).max(1.0),
        70.0,
        1.35,
        2.0,
        "coloured-ridge-on-boundary",
    );
    for (&index, edge) in boundary_indices.iter().zip(boundary_values) {
        selected_graph[index] = edge;
    }
    let source_ink_support: Vec<bool> = residual_lines
        .iter()
        .zip(&source.pixels)
        .map(|(&selected, pixel)| {
            selected || 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2] <= 0.65
        })
        .collect();
    let connected_residual_legacy_graph = connect_graph_edges(
        residual_legacy_graph,
        &source_ink_support,
        width,
        height,
        6.0,
        15.0,
    );
    let residual_legacy_graph = remove_graph_aligned_overlap(
        connected_residual_legacy_graph,
        &selected_graph,
        1.5,
        20.0,
        false,
    );
    selected_graph.extend(residual_legacy_graph);
    selected_graph = connect_graph_edges(
        selected_graph,
        &source_ink_support,
        width,
        height,
        6.0,
        15.0,
    );
    selected_graph = snap_graph_junction_endpoints(
        selected_graph,
        &source_ink_support,
        width,
        height,
        6.0,
        70.0,
    );
    let paint_junction_costs = paint_junctions
        .iter()
        .map(|point| {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, height.saturating_sub(1) as f32) as usize;
            let mut minimum = f32::INFINITY;
            for row in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for column in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let pixel = source.pixels[row * width + column];
                    minimum =
                        minimum.min(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]);
                }
            }
            minimum
        })
        .collect::<Vec<_>>();
    // The Python writer receives the 3x3-dilated Paint face barrier, then
    // unions it with `selected_lines` before shared-node snapping.
    let paint_face_support = dilate_square(&structural.face_barrier, width, height, 1);
    let paint_junction_support = paint_face_support
        .iter()
        .zip(&residual_lines)
        .map(|(&barrier, &line)| barrier || line)
        .collect::<Vec<_>>();
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_STRUCTURAL_DIAGNOSTICS") {
        let bytes = paint_junction_support
            .iter()
            .map(|&value| u8::from(value))
            .collect::<Vec<_>>();
        let _ = std::fs::write(
            format!("{}-paint-junction-support.u8", prefix.to_string_lossy()),
            bytes,
        );
    }
    selected_graph = snap_graph_to_paint_junctions(
        selected_graph,
        paint_junctions,
        &paint_junction_costs,
        &paint_junction_support,
        width,
        height,
        4.0,
        65.0,
        0.45,
    );
    selected_graph.retain(|edge| {
        if !matches!(
            edge.role,
            "ridge-on-boundary"
                | "bright-ridge-on-boundary"
                | "coloured-ridge-on-boundary"
                | "dark-boundary"
        ) {
            return true;
        }
        let travelled = edge
            .points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum::<f32>();
        travelled >= (2.5 * edge.width.max(0.0)).max(2.0)
    });
    selected_graph.retain(|stroke| {
        structural.boundary_stroke_mask.is_empty()
            || mask_fraction_along(stroke, &structural.boundary_stroke_mask, width, height) <= 0.5
    });
    selected_graph.extend(
        structural
            .strokes
            .iter()
            .filter(|stroke| stroke.role == "boundary-stroke")
            .cloned(),
    );
    let mut endpoint_counts = HashMap::<(i64, i64), usize>::new();
    for edge in &selected_graph {
        for point in [edge.points[0], edge.points[edge.points.len() - 1]] {
            let key = point_key(point);
            *endpoint_counts.entry(key).or_default() += 1;
        }
    }
    let endpoint_tangents = graph_continuation_tangents(&selected_graph);
    // Indexed collection preserves SVG order while sharing the caller's worker limit.
    let strokes: Vec<_> = selected_graph
        .into_par_iter()
        .enumerate()
        .filter_map(|(stroke_index, stroke)| {
            let length = stroke
                .points
                .windows(2)
                .map(|pair| pair[0].distance(pair[1]))
                .sum::<f32>();
            let minimum = if stroke.role == "legacy-structural" {
                4.0
            } else {
                (1.5 * stroke.width).min(4.0).max(2.0_f32.min(4.0))
            };
            if length < minimum {
                return None;
            }
            let width_scale = if stroke.role == "coloured-ridge-on-boundary" {
                0.8
            } else {
                1.0
            };
            let start_key = point_key(stroke.points[0]);
            let end_key = point_key(stroke.points[stroke.points.len() - 1]);
            let shared_start = endpoint_counts.get(&start_key).copied().unwrap_or(0) > 1;
            let shared_end = endpoint_counts.get(&end_key).copied().unwrap_or(0) > 1;
            let straight =
                straight_graph_line(&stroke.points, std::f32::consts::FRAC_1_SQRT_2, minimum);
            let (points, path_data) = if let Some((mut start, mut end)) = straight {
                if shared_start {
                    start = stroke.points[0];
                }
                if shared_end {
                    end = stroke.points[stroke.points.len() - 1];
                }
                (vec![start, end], None)
            } else {
                let fitting_points = refine_stroke_centerline(&stroke, &source_lab, width, height);
                let path_data = fitted_structural_open_path_data_with_tangents(
                    &fitting_points,
                    0.75,
                    0.45,
                    endpoint_tangents.get(&(stroke_index, true)).copied(),
                    endpoint_tangents.get(&(stroke_index, false)).copied(),
                );
                (stroke.points.clone(), Some(path_data))
            };
            if points.len() < 2 {
                return None;
            }
            let color = sample_graph_color(source, &stroke);
            Some(StructuralStroke {
                points,
                path_data,
                precise_points: None,
                color,
                width: (stroke.width * width_scale).max(0.4),
                role: stroke.role,
                width_samples: stroke.width_samples.clone(),
            })
        })
        .collect();
    let mut summary = structural.summary.clone();
    summary.skeleton_pixels = residual_skeleton_pixels;
    summary.stroke_count = strokes.len();
    summary.residual_legacy_strokes = strokes
        .iter()
        .filter(|stroke| stroke.role == "legacy-structural")
        .count();
    summary.visible_ridge_strokes = strokes
        .iter()
        .filter(|stroke| stroke.role == "ridge")
        .count();
    summary.boundary_profile_strokes = strokes
        .iter()
        .filter(|stroke| {
            matches!(
                stroke.role,
                "ridge-on-boundary"
                    | "bright-ridge-on-boundary"
                    | "coloured-ridge-on-boundary"
                    | "dark-boundary"
            )
        })
        .count();
    StructuralInk {
        strokes,
        paint_ownership_mask: structural.paint_ownership_mask.clone(),
        source_line_mask: structural.source_line_mask.clone(),
        legacy_line_mask: structural.legacy_line_mask.clone(),
        role_line_mask: structural.role_line_mask.clone(),
        visible_ridge_coverage: structural.visible_ridge_coverage.clone(),
        dark_boundary_coverage: structural.dark_boundary_coverage.clone(),
        face_barrier: structural.face_barrier.clone(),
        boundary_stroke_mask: structural.boundary_stroke_mask.clone(),
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn horizontal_boundary_stroke(last_x: usize) -> StructuralStroke {
        StructuralStroke {
            points: (2..=last_x)
                .map(|x| Point {
                    x: x as f32 + 0.5,
                    y: 4.5,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: [0.02; 3],
            width: 2.0,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        }
    }

    fn dark_boundary_profile(light_side: f32) -> Vec<Lab> {
        let width = 18;
        let height = 9;
        let mut values = vec![Lab::default(); width * height];
        for y in 0..height {
            let lightness = if y < 4 {
                light_side
            } else if y == 4 {
                5.0
            } else {
                22.0
            };
            for x in 0..width {
                values[y * width + x] = Lab {
                    l: lightness,
                    a: 0.0,
                    b: 0.0,
                };
            }
        }
        values
    }

    fn graph_stroke(points: &[(f32, f32)]) -> StructuralStroke {
        StructuralStroke {
            points: points.iter().map(|&(x, y)| Point { x, y }).collect(),
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 1.0,
            role: "legacy-structural",
            width_samples: Vec::new(),
        }
    }

    #[test]
    fn junction_grouping_selects_the_straight_through_pair() {
        let strokes = vec![
            graph_stroke(&[(-4.0, 0.0), (-2.0, 0.0), (0.0, 0.0)]),
            graph_stroke(&[(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]),
            graph_stroke(&[(0.0, 0.0), (0.0, 2.0), (0.0, 4.0)]),
        ];
        let tangents = graph_continuation_tangents(&strokes);
        assert_eq!(tangents.get(&(0, false)), Some(&Point { x: 1.0, y: 0.0 }));
        assert_eq!(tangents.get(&(1, true)), Some(&Point { x: 1.0, y: 0.0 }));
        assert!(!tangents.contains_key(&(2, true)));
    }

    #[test]
    fn source_profile_refines_a_pixel_centred_skeleton_without_moving_endpoints() {
        let mut source = Raster::blank(9, 9, [1.0; 3]);
        for x in 1..8 {
            source.pixels[3 * 9 + x] = [0.0; 3];
            source.pixels[4 * 9 + x] = [0.45; 3];
        }
        let stroke = graph_stroke(&(1..8).map(|x| (x as f32 + 0.5, 4.5)).collect::<Vec<_>>());
        let refined = refine_stroke_centerline(&stroke, &lab_pixels(&source), 9, 9);
        assert_eq!(refined[0], stroke.points[0]);
        assert_eq!(
            refined[refined.len() - 1],
            stroke.points[stroke.points.len() - 1]
        );
        assert!(refined[3].y < 4.25, "refined={refined:?}");
        assert!(refined[3].y >= 4.0);
    }

    #[test]
    fn spatial_point_pairs_match_dense_lexicographic_scan() {
        let points = vec![
            Point { x: -4.0, y: 0.0 },
            Point { x: 0.0, y: 0.0 },
            Point { x: 3.0, y: 4.0 },
            Point { x: 4.99, y: 0.0 },
            Point { x: 5.01, y: 0.0 },
            Point { x: 12.0, y: 8.0 },
        ];
        let radius = 5.0;
        let dense = (0..points.len())
            .flat_map(|first| {
                let points = &points;
                (first + 1..points.len()).filter_map(move |second| {
                    (points[first].distance(points[second]) <= radius).then_some((first, second))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(nearby_point_pairs(&points, radius), dense);
    }

    #[test]
    fn short_dark_boundary_undershoot_stays_paint_owned() {
        let source_lab = dark_boundary_profile(18.0);
        assert!(paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(10),
            &source_lab,
            18,
            9,
        ));
    }

    #[test]
    fn ordinary_or_long_boundary_ridge_remains_structural() {
        let light_incident_face = dark_boundary_profile(70.0);
        assert!(!paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(10),
            &light_incident_face,
            18,
            9,
        ));

        let dark_incident_faces = dark_boundary_profile(18.0);
        assert!(!paint_owned_dark_boundary_undershoot(
            &horizontal_boundary_stroke(14),
            &dark_incident_faces,
            18,
            9,
        ));
    }

    #[test]
    fn bright_boundary_ridge_uses_positive_profile_polarity() {
        let width = 18;
        let height = 9;
        let mut source = Raster::blank(width, height, [0.18; 3]);
        for x in 4..14 {
            source.pixels[4 * width + x] = [0.92; 3];
        }
        let rendered = Raster::blank(width, height, [0.18; 3]);
        let stroke = StructuralStroke {
            points: (4..14)
                .map(|x| Point {
                    x: x as f32 + 0.5,
                    y: 4.5,
                })
                .collect(),
            path_data: None,
            precise_points: None,
            color: [0.92; 3],
            width: 1.2,
            role: "bright-ridge-on-boundary",
            width_samples: Vec::new(),
        };
        let (missing, supported) = boundary_profile_flags(
            &stroke,
            &lab_pixels(&source),
            &lab_pixels(&rendered),
            width,
            height,
            5.0,
        );

        assert!(missing.iter().filter(|&&value| value).count() >= 8);
        assert!(supported.iter().all(|&value| value));
        assert!(sample_graph_color(&source, &stroke)[0] > 0.8);
    }

    #[test]
    fn thinning_keeps_a_line_connected() {
        let mut mask = vec![false; 32 * 16];
        for y in 6..10 {
            for x in 3..29 {
                mask[y * 32 + x] = true;
            }
        }
        let thin = skeletonize(&mask, 32, 16);
        assert!(thin.iter().filter(|&&v| v).count() >= 20);
        assert!(thin.iter().filter(|&&v| v).count() < 35);
    }

    #[test]
    fn underpaint_uses_one_incident_face_without_mixing_sides() {
        let mut source = Raster::blank(5, 3, [1.0, 0.0, 0.0]);
        for y in 0..3 {
            for x in 3..5 {
                source.pixels[y * 5 + x] = [0.0, 0.0, 1.0];
            }
        }
        let mut mask = vec![false; 15];
        for y in 0..3 {
            mask[y * 5 + 2] = true;
        }
        let result = nearest_underpaint(&source, &mask);
        for y in 0..3 {
            let colour = result.pixels[y * 5 + 2];
            assert!(colour == [1.0, 0.0, 0.0] || colour == [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn structural_antialias_shoulder_is_returned_to_paint() {
        let mut source = Raster::blank(7, 5, [1.0, 1.0, 1.0]);
        let mut structural = vec![false; 35];
        for y in 1..4 {
            structural[y * 7 + 3] = true;
            source.pixels[y * 7 + 3] = [0.0, 0.0, 0.0];
            source.pixels[y * 7 + 4] = [0.5, 0.5, 0.5];
        }
        let mut underpaint = nearest_underpaint(&source, &structural);
        let selected = unmix_structural_antialias(&source, &mut underpaint, &structural);
        assert!(selected.iter().filter(|&&value| value).count() >= 3);
        for y in 1..4 {
            assert!(underpaint.pixels[y * 7 + 4][0] > 0.9);
        }
    }
}
