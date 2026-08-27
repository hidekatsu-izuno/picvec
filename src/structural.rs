use std::collections::{HashMap, HashSet, VecDeque};

use rayon::prelude::*;
use serde::Serialize;

use crate::color::{delta_e2000, delta_e76, rgb_to_lab, Lab};
use crate::edge::{dilate, dilate_square, erode, EdgeRoles};
use crate::geometry::{bounded_fairing_open, simplify_open, Point};
use crate::raster::{percentile, Raster};

#[derive(Clone, Debug)]
pub struct StructuralStroke {
    pub points: Vec<Point>,
    pub color: [f32; 3],
    pub width: f32,
    pub role: &'static str,
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
    pub fallback_strokes: usize,
    pub visible_ridge_strokes: usize,
    pub boundary_profile_strokes: usize,
}

#[derive(Clone, Debug)]
pub struct StructuralInk {
    pub strokes: Vec<StructuralStroke>,
    /// Medial-ridge core plus source-modelled AA shoulders transferred out of Paint.
    pub paint_ownership_mask: Vec<bool>,
    /// Role-filtered raster lines frozen during Paint regularization. This is
    /// narrower than the complete set of source graph candidates.
    pub source_line_mask: Vec<bool>,
    pub summary: StructuralSummary,
}

impl StructuralInk {
    pub fn empty() -> Self {
        Self {
            strokes: Vec::new(),
            paint_ownership_mask: Vec::new(),
            source_line_mask: Vec::new(),
            summary: StructuralSummary::default(),
        }
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

/// Zhang-Suen thinning.  It preserves connectivity while reducing a measured
/// ridge band to a centre-line graph independent of scan direction.
pub fn skeletonize(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut result = mask.to_vec();
    if width < 3 || height < 3 {
        return result;
    }
    loop {
        let mut changed = false;
        for phase in 0..2 {
            let snapshot = result.clone();
            let mut remove = Vec::new();
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let index = y * width + x;
                    if !snapshot[index] {
                        continue;
                    }
                    let p = [
                        snapshot[(y - 1) * width + x],
                        snapshot[(y - 1) * width + x + 1],
                        snapshot[y * width + x + 1],
                        snapshot[(y + 1) * width + x + 1],
                        snapshot[(y + 1) * width + x],
                        snapshot[(y + 1) * width + x - 1],
                        snapshot[y * width + x - 1],
                        snapshot[(y - 1) * width + x - 1],
                    ];
                    let neighbours = p.iter().filter(|&&value| value).count();
                    if !(2..=6).contains(&neighbours) {
                        continue;
                    }
                    let transitions = (0..8).filter(|&i| !p[i] && p[(i + 1) % 8]).count();
                    if transitions != 1 {
                        continue;
                    }
                    let keep_a = if phase == 0 {
                        !p[2] || !p[4] || (!p[0] && !p[6])
                    } else {
                        !p[0] || !p[6] || (!p[2] && !p[4])
                    };
                    if keep_a {
                        remove.push(index);
                    }
                }
            }
            if !remove.is_empty() {
                changed = true;
                for index in remove {
                    result[index] = false;
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
            (x > 0).then_some(index - 1),
            (x + 1 < width).then_some(index + 1),
            (y > 0).then_some(index - width),
            (y + 1 < height).then_some(index + width),
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

/// Native source structural-line classifier used by the Python pipeline.
/// A line must differ from both sides while those two sides agree; this is
/// what separates a medial ridge from an ordinary two-material boundary.
fn source_structural_lines(source: &Raster) -> Vec<bool> {
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
                (index % width > 0).then_some(index - 1),
                (index % width + 1 < width).then_some(index + 1),
                (index / width > 0).then_some(index - width),
                (index / width + 1 < height).then_some(index + width),
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
    lines
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

fn join_continuous_strokes(mut strokes: Vec<StructuralStroke>) -> Vec<StructuralStroke> {
    const MAXIMUM_GAP: f32 = 2.6;
    loop {
        let mut best: Option<(usize, usize, bool, bool, f32)> = None;
        let mut endpoint_buckets = HashMap::<(i32, i32), Vec<usize>>::new();
        for (index, stroke) in strokes.iter().enumerate() {
            if stroke.points.len() < 2 {
                continue;
            }
            for point in [stroke.points[0], stroke.points[stroke.points.len() - 1]] {
                endpoint_buckets
                    .entry((
                        (point.x / MAXIMUM_GAP).floor() as i32,
                        (point.y / MAXIMUM_GAP).floor() as i32,
                    ))
                    .or_default()
                    .push(index);
            }
        }
        let mut candidate_pairs = HashSet::<(usize, usize)>::new();
        for (index, stroke) in strokes.iter().enumerate() {
            if stroke.points.len() < 2 {
                continue;
            }
            for point in [stroke.points[0], stroke.points[stroke.points.len() - 1]] {
                let cell_x = (point.x / MAXIMUM_GAP).floor() as i32;
                let cell_y = (point.y / MAXIMUM_GAP).floor() as i32;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(nearby) = endpoint_buckets.get(&(cell_x + dx, cell_y + dy)) {
                            for &other in nearby {
                                if index != other {
                                    candidate_pairs.insert(edge_key(index, other));
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut candidate_pairs: Vec<(usize, usize)> = candidate_pairs.into_iter().collect();
        candidate_pairs.sort_unstable();
        for (first, second) in candidate_pairs {
            if strokes[first].points.len() < 2 {
                continue;
            }
            if strokes[second].points.len() < 2
                || strokes[first].role != strokes[second].role
                || delta_e76(
                    rgb_to_lab(strokes[first].color),
                    rgb_to_lab(strokes[second].color),
                ) > 6.0
                || strokes[first].width.max(strokes[second].width)
                    / strokes[first].width.min(strokes[second].width).max(0.25)
                    > 1.8
            {
                continue;
            }
            for reverse_first in [false, true] {
                for reverse_second in [false, true] {
                    let a = &strokes[first].points;
                    let b = &strokes[second].points;
                    let a_end = if reverse_first { a[0] } else { a[a.len() - 1] };
                    let a_before = if reverse_first { a[1] } else { a[a.len() - 2] };
                    let b_start = if reverse_second { b[b.len() - 1] } else { b[0] };
                    let b_after = if reverse_second { b[b.len() - 2] } else { b[1] };
                    let gap = a_end.distance(b_start);
                    if gap > MAXIMUM_GAP {
                        continue;
                    }
                    let first_tangent = (a_end.x - a_before.x, a_end.y - a_before.y);
                    let second_tangent = (b_after.x - b_start.x, b_after.y - b_start.y);
                    let first_length = (first_tangent.0.powi(2) + first_tangent.1.powi(2)).sqrt();
                    let second_length =
                        (second_tangent.0.powi(2) + second_tangent.1.powi(2)).sqrt();
                    let alignment = (first_tangent.0 * second_tangent.0
                        + first_tangent.1 * second_tangent.1)
                        / (first_length * second_length).max(1e-6);
                    if alignment < 0.72 {
                        continue;
                    }
                    let score = alignment - 0.08 * gap;
                    if best.map(|value| score > value.4).unwrap_or(true) {
                        best = Some((first, second, reverse_first, reverse_second, score));
                    }
                }
            }
        }
        let Some((first, second, reverse_first, reverse_second, _)) = best else {
            break;
        };
        let mut first_points = strokes[first].points.clone();
        let mut second_points = strokes[second].points.clone();
        if reverse_first {
            first_points.reverse();
        }
        if reverse_second {
            second_points.reverse();
        }
        if first_points.last() == second_points.first() {
            second_points.remove(0);
        }
        first_points.extend(second_points);
        let first_weight = strokes[first].points.len() as f32;
        let second_weight = strokes[second].points.len() as f32;
        let total = first_weight + second_weight;
        let joined = StructuralStroke {
            points: simplify_open(&first_points, 0.75),
            color: [
                (strokes[first].color[0] * first_weight + strokes[second].color[0] * second_weight)
                    / total,
                (strokes[first].color[1] * first_weight + strokes[second].color[1] * second_weight)
                    / total,
                (strokes[first].color[2] * first_weight + strokes[second].color[2] * second_weight)
                    / total,
            ],
            width: (strokes[first].width * first_weight + strokes[second].width * second_weight)
                / total,
            role: strokes[first].role,
        };
        strokes[first] = joined;
        strokes.swap_remove(second);
    }
    strokes
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

fn local_width(mask: &[bool], width: usize, height: usize, indices: &[usize]) -> f32 {
    let mut widths = Vec::new();
    for &index in indices.iter().step_by((indices.len() / 64).max(1)) {
        let x = index % width;
        let y = index / width;
        let mut radius = 1_usize;
        'search: for candidate in 1..=5 {
            for dy in -(candidate as isize)..=candidate as isize {
                for dx in -(candidate as isize)..=candidate as isize {
                    if dx.abs().max(dy.abs()) != candidate as isize {
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
                        radius = candidate;
                        break 'search;
                    }
                }
            }
        }
        widths.push((radius as f32 * 2.0 - 0.5).max(1.0));
    }
    widths.sort_by(|a, b| a.total_cmp(b));
    widths
        .get(widths.len() / 2)
        .copied()
        .unwrap_or(1.25)
        .clamp(0.75, 8.0)
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

/// Transfer only source-supported medial ridges out of Paint.  Dark filled
/// faces remain Paint-owned; there is intentionally no median-colour
/// silhouette overlay that could flatten tyre or shadow gradients.
pub fn analyse(source: &Raster, roles: &mut EdgeRoles) -> (Raster, StructuralInk) {
    let classified_lines = source_structural_lines(source);
    let shading_corridor = dilate_square(&roles.shading, source.width, source.height, 1);
    // Only a profile-confirmed medial ridge has the same Paint owner on both
    // sides and may be removed before quantization.  Other structural
    // candidates remain in the complete Paint base until residual selection.
    let mut paint_reference = nearest_underpaint(source, &roles.visible_ridge_coverage);
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        let _ = paint_reference.save(std::path::Path::new(&format!(
            "{prefix}-nearest-underpaint.png"
        )));
    }
    let antialias_ownership =
        unmix_structural_antialias(source, &mut paint_reference, &roles.visible_ridge_coverage);
    let antialias_unmixed_pixels = antialias_ownership.iter().filter(|&&value| value).count();
    // The AA shoulder is Paint-transfer ownership, not new edge-role
    // evidence.  The reference keeps EdgeRoleRaster unchanged and extends
    // only its local prequantization mask.
    let underpaint_ownership: Vec<bool> = roles
        .visible_ridge_coverage
        .iter()
        .zip(&antialias_ownership)
        .map(|(&ridge, &shoulder)| ridge || shoulder)
        .collect();
    let mut source_graph_coverage = roles.visible_ridge_coverage.clone();
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
        .zip(&shading_corridor)
        .zip(&roles.visible_ridge_coverage)
        .map(|((&classified, &shading), &ridge)| (classified && !shading) || ridge)
        .collect();
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
    let fallback_coverage: Vec<bool> = classified_lines
        .iter()
        .zip(&shading_corridor)
        .zip(&source_graph_coverage)
        .map(|((&classified, &shading), &graph)| classified && !shading && !graph)
        .collect();
    let mut skeleton = skeletonize(&fallback_coverage, source.width, source.height);
    remove_small_components(&mut skeleton, source.width, source.height, 4);
    let paths = trace_skeleton(&skeleton, source.width, source.height);
    let strokes = paths
        .into_iter()
        .filter_map(|indices| {
            let points: Vec<Point> = indices
                .iter()
                .map(|&index| Point {
                    x: (index % source.width) as f32 + 0.5,
                    y: (index / source.width) as f32 + 0.5,
                })
                .collect();
            let simplified = bounded_fairing_open(&points, 0.75);
            if simplified.len() < 2 {
                return None;
            }
            Some(StructuralStroke {
                points: simplified,
                color: median_color(source, &indices),
                width: local_width(&fallback_coverage, source.width, source.height, &indices),
                role: "fallback",
            })
        })
        .collect::<Vec<_>>();
    let mut strokes = join_continuous_strokes(strokes);
    let graph_strokes = roles
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
                color: median_color(source, &indices),
                width: edge.width.max(1.2) as f32,
                role: edge.role,
            })
        })
        .collect::<Vec<_>>();
    strokes.extend(graph_strokes);
    let summary = StructuralSummary {
        source_coverage_pixels: fallback_coverage
            .iter()
            .zip(&source_graph_coverage)
            .filter(|&(fallback, graph)| *fallback || *graph)
            .count(),
        source_line_pixels: source_line_mask.iter().filter(|&&value| value).count(),
        skeleton_pixels: skeleton.iter().filter(|&&value| value).count(),
        stroke_count: strokes.len(),
        underpainted_pixels: underpaint_ownership.iter().filter(|&&value| value).count(),
        antialias_unmixed_pixels,
        silhouette_fill_count: 0,
        fallback_strokes: strokes
            .iter()
            .filter(|stroke| stroke.role == "fallback")
            .count(),
        visible_ridge_strokes: strokes
            .iter()
            .filter(|stroke| stroke.role == "ridge")
            .count(),
        boundary_profile_strokes: strokes
            .iter()
            .filter(|stroke| {
                matches!(
                    stroke.role,
                    "ridge-on-boundary" | "coloured-ridge-on-boundary" | "dark-boundary"
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
            summary,
        },
    )
}

fn sampled_stroke_points(points: &[Point]) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut result = vec![points[0]];
    for pair in points.windows(2) {
        let length = pair[0].distance(pair[1]);
        let steps = (length * 2.0).ceil().max(1.0) as usize;
        for step in 1..=steps {
            let amount = step as f32 / steps as f32;
            result.push(Point {
                x: pair[0].x + amount * (pair[1].x - pair[0].x),
                y: pair[0].y + amount * (pair[1].y - pair[0].y),
            });
        }
    }
    result
}

fn bilinear_lab(values: &[Lab], width: usize, height: usize, point: Point) -> Lab {
    let x = (point.x - 0.5).clamp(0.0, width.saturating_sub(1) as f32);
    let y = (point.y - 0.5).clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let interpolate = |channel: fn(Lab) -> f32| {
        let top =
            channel(values[y0 * width + x0]) * (1.0 - tx) + channel(values[y0 * width + x1]) * tx;
        let bottom =
            channel(values[y1 * width + x0]) * (1.0 - tx) + channel(values[y1 * width + x1]) * tx;
        top * (1.0 - ty) + bottom * ty
    };
    Lab {
        l: interpolate(|value| value.l),
        a: interpolate(|value| value.a),
        b: interpolate(|value| value.b),
    }
}

fn normal_at(points: &[Point], index: usize) -> (f32, f32) {
    let before = points[index.saturating_sub(1)];
    let after = points[(index + 1).min(points.len() - 1)];
    let dx = after.x - before.x;
    let dy = after.y - before.y;
    let length = dx.hypot(dy).max(1e-6);
    (-dy / length, dx / length)
}

/// Port of perceptual_pipeline._boundary_ridge_profile_measurements.  The
/// decision is made at a fixed arclength position and searches only across
/// the normal, so a tangential gap cannot borrow a matching neighbouring
/// source pixel.
fn boundary_profile_measurements(
    stroke: &StructuralStroke,
    source_lab: &[Lab],
    rendered_lab: &[Lab],
    width: usize,
    height: usize,
    primary_owner_corridor: &[bool],
) -> (Vec<bool>, Vec<bool>, bool) {
    let chroma_samples: Vec<f32> = stroke
        .points
        .iter()
        .map(|&point| {
            let value = bilinear_lab(source_lab, width, height, point);
            value.a.hypot(value.b)
        })
        .collect();
    let mut sorted_chroma = chroma_samples;
    sorted_chroma.sort_by(f32::total_cmp);
    let coloured = stroke.role == "ridge-on-boundary"
        && sorted_chroma
            .get(sorted_chroma.len() / 2)
            .copied()
            .unwrap_or(0.0)
            >= 12.0;
    let offsets = [-0.75_f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
    let mut missing = Vec::with_capacity(stroke.points.len());
    let mut source_valley = Vec::with_capacity(stroke.points.len());
    for (index, &point) in stroke.points.iter().enumerate() {
        let (nx, ny) = normal_at(&stroke.points, index);
        let sample = |values: &[Lab], offset: f32| {
            bilinear_lab(
                values,
                width,
                height,
                Point {
                    x: point.x + offset * nx,
                    y: point.y + offset * ny,
                },
            )
        };
        let source_profile: Vec<Lab> = offsets
            .iter()
            .map(|&offset| sample(source_lab, offset))
            .collect();
        let rendered_profile: Vec<Lab> = offsets
            .iter()
            .map(|&offset| sample(rendered_lab, offset))
            .collect();
        let source_sides = [sample(source_lab, -1.5), sample(source_lab, 1.5)];
        let rendered_sides = [sample(rendered_lab, -1.5), sample(rendered_lab, 1.5)];
        let target = if coloured {
            source_profile[3]
        } else {
            source_profile
                .iter()
                .copied()
                .min_by(|first, second| first.l.total_cmp(&second.l))
                .unwrap_or(source_profile[3])
        };
        let source_dark_contrast = source_sides[0].l.min(source_sides[1].l) - target.l;
        let source_side_colour_contrast =
            delta_e2000(source_sides[0], target).min(delta_e2000(source_sides[1], target));
        let source_line_contrast = if coloured {
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
        let minimum_error = rendered_profile
            .iter()
            .map(|&value| delta_e2000(value, target))
            .fold(f32::INFINITY, f32::min);
        let x = (point.x - 0.5)
            .round()
            .clamp(0.0, width.saturating_sub(1) as f32) as usize;
        let y = (point.y - 0.5)
            .round()
            .clamp(0.0, height.saturating_sub(1) as f32) as usize;
        let has_valley = source_line_contrast >= 4.0;
        source_valley.push(has_valley);
        missing.push(
            !primary_owner_corridor[y * width + x]
                && minimum_error > 4.0
                && has_valley
                && (rendered_dark_contrast < 0.65 * source_dark_contrast
                    || rendered_minimum_lightness - target.l >= 5.0),
        );
    }
    (missing, source_valley, coloured)
}

fn push_simplified_complete(
    output: &mut Vec<StructuralStroke>,
    stroke: &StructuralStroke,
    role: &'static str,
    width_scale: f32,
) {
    let points = bounded_fairing_open(&stroke.points, 0.55);
    if points.len() >= 2 {
        output.push(StructuralStroke {
            points,
            color: stroke.color,
            width: width_scale * stroke.width,
            role,
        });
    }
}

/// Keep only the structural ink that the native-resolution Paint render does
/// not already represent.  This gives every authored line one visual owner
/// and prevents a correctly traced Paint boundary from receiving a second,
/// independently widened stroke.
pub fn select_missing(
    source: &Raster,
    rendered: &Raster,
    structural: &StructuralInk,
) -> StructuralInk {
    if source.width != rendered.width || source.height != rendered.height {
        return structural.clone();
    }
    let source_lab: Vec<Lab> = source.pixels.par_iter().copied().map(rgb_to_lab).collect();
    let rendered_lab: Vec<Lab> = rendered
        .pixels
        .par_iter()
        .copied()
        .map(rgb_to_lab)
        .collect();
    let mut primary_owner_corridor = vec![false; source.width * source.height];
    for stroke in structural
        .strokes
        .iter()
        .filter(|stroke| stroke.role == "ridge")
    {
        for point in sampled_stroke_points(&stroke.points) {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, source.width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, source.height.saturating_sub(1) as f32) as usize;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px >= 0
                        && py >= 0
                        && px < source.width as isize
                        && py < source.height as isize
                    {
                        primary_owner_corridor[py as usize * source.width + px as usize] = true;
                    }
                }
            }
        }
    }
    let mut strokes = Vec::<StructuralStroke>::new();
    for stroke in &structural.strokes {
        if stroke.role == "ridge" {
            // Paint ownership was removed for these graph owners before
            // quantization, so the complete trajectory is unconditional.
            push_simplified_complete(&mut strokes, stroke, "ridge", 1.0);
            continue;
        }
        if matches!(stroke.role, "ridge-on-boundary" | "dark-boundary") {
            let (missing, mut valley, coloured) = boundary_profile_measurements(
                stroke,
                &source_lab,
                &rendered_lab,
                source.width,
                source.height,
                &primary_owner_corridor,
            );
            let resolved_role = if coloured {
                "coloured-ridge-on-boundary"
            } else {
                stroke.role
            };
            let width_scale = if coloured { 0.8 } else { 1.0 };
            let complete_threshold = if coloured {
                0.30
            } else if stroke.role == "dark-boundary" {
                0.90
            } else {
                0.60
            };
            let missing_fraction =
                missing.iter().filter(|&&value| value).count() as f32 / missing.len().max(1) as f32;
            if missing_fraction >= complete_threshold {
                push_simplified_complete(&mut strokes, stroke, resolved_role, width_scale);
                continue;
            }

            let cumulative: Vec<f32> = std::iter::once(0.0)
                .chain(stroke.points.windows(2).scan(0.0, |total, pair| {
                    *total += pair[0].distance(pair[1]);
                    Some(*total)
                }))
                .collect();
            // split_edges_by_support_runs(maximum_internal_gap=3): keep a
            // source valley continuous across only a short raster miss.
            let mut index = 0_usize;
            while index < valley.len() {
                if valley[index] {
                    index += 1;
                    continue;
                }
                let first = index;
                while index < valley.len() && !valley[index] {
                    index += 1;
                }
                if first > 0
                    && index < valley.len()
                    && cumulative[index] - cumulative[first - 1] <= 3.0
                {
                    valley[first..index].fill(true);
                }
            }
            let interval_threshold = if stroke.role == "dark-boundary" {
                0.90
            } else {
                0.15
            };
            let mut index = 0_usize;
            while index < valley.len() {
                if !valley[index] {
                    index += 1;
                    continue;
                }
                let mut first = index;
                while index + 1 < valley.len() && valley[index + 1] {
                    index += 1;
                }
                let mut last = index;
                index += 1;
                if cumulative[last] - cumulative[first] < 2.0 {
                    continue;
                }
                while first > 0 && cumulative[first] - cumulative[first - 1] <= 1.0 {
                    first -= 1;
                }
                while last + 1 < stroke.points.len()
                    && cumulative[last + 1] - cumulative[last] <= 1.0
                {
                    last += 1;
                }
                let fraction = missing[first..=last].iter().filter(|&&value| value).count() as f32
                    / (last + 1 - first) as f32;
                if fraction < interval_threshold {
                    continue;
                }
                let interval = StructuralStroke {
                    points: stroke.points[first..=last].to_vec(),
                    color: stroke.color,
                    width: stroke.width,
                    role: resolved_role,
                };
                push_simplified_complete(&mut strokes, &interval, resolved_role, width_scale);
            }
            continue;
        }
        let samples = sampled_stroke_points(&stroke.points);
        if samples.len() < 2 {
            continue;
        }
        let mut missing = Vec::<bool>::with_capacity(samples.len());
        for point in &samples {
            let x = (point.x - 0.5)
                .round()
                .clamp(0.0, source.width.saturating_sub(1) as f32) as usize;
            let y = (point.y - 0.5)
                .round()
                .clamp(0.0, source.height.saturating_sub(1) as f32) as usize;
            let reference = source_lab[y * source.width + x];
            let chroma = (reference.a * reference.a + reference.b * reference.b).sqrt();
            let mut minimum_delta = f32::INFINITY;
            let mut minimum_lightness = f32::INFINITY;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let px = (x as isize + dx).clamp(0, source.width as isize - 1) as usize;
                    let py = (y as isize + dy).clamp(0, source.height as isize - 1) as usize;
                    let candidate = rendered_lab[py * source.width + px];
                    minimum_delta = minimum_delta.min(delta_e76(reference, candidate));
                    minimum_lightness = minimum_lightness.min(candidate.l);
                }
            }
            let represented = minimum_delta <= 4.0
                || (chroma < 12.0 && reference.l <= 50.0 && minimum_lightness <= reference.l + 4.0);
            missing.push(!represented);
        }
        let missing_count = missing.iter().filter(|&&value| value).count();
        if missing_count < 3 {
            continue;
        }
        if missing_count as f32 / missing.len() as f32 >= 0.75 {
            strokes.push(stroke.clone());
            continue;
        }
        let measured = missing.clone();
        for index in 0..missing.len() {
            if measured[index] {
                if index > 0 {
                    missing[index - 1] = true;
                }
                if index + 1 < missing.len() {
                    missing[index + 1] = true;
                }
            }
        }
        let mut index = 0;
        while index < missing.len() {
            if !missing[index] {
                index += 1;
                continue;
            }
            let first = index;
            while index + 1 < missing.len() && missing[index + 1] {
                index += 1;
            }
            let last = index;
            index += 1;
            if last.saturating_sub(first) < 2 {
                continue;
            }
            let points = bounded_fairing_open(&samples[first..=last], 0.75);
            if points.len() >= 2 {
                strokes.push(StructuralStroke {
                    points,
                    color: stroke.color,
                    width: stroke.width,
                    role: stroke.role,
                });
            }
        }
    }
    // write_structural_ink_svg(minimum_line_length=4.0) removes compact
    // residual fragments after ownership has been resolved.  Keeping them in
    // Rust produced round-cap specks and inflated the path count.
    strokes.retain(|stroke| {
        stroke
            .points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum::<f32>()
            >= 4.0
    });
    let mut summary = structural.summary.clone();
    summary.stroke_count = strokes.len();
    summary.fallback_strokes = strokes
        .iter()
        .filter(|stroke| stroke.role == "fallback")
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
                "ridge-on-boundary" | "coloured-ridge-on-boundary" | "dark-boundary"
            )
        })
        .count();
    StructuralInk {
        strokes,
        paint_ownership_mask: structural.paint_ownership_mask.clone(),
        source_line_mask: structural.source_line_mask.clone(),
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
