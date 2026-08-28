use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use serde::Serialize;

use crate::color::{delta_e2000, lab_pixels_to_rgb, lab_to_rgb, rgb_to_lab, Lab};
use crate::config::Config;
use crate::edge::{lab_pixels, EdgeRoles};
use crate::raster::{percentile, Raster};
use crate::union_find::UnionFind;

#[derive(Clone, Debug, Default, Serialize)]
pub struct SegmentationSummary {
    pub graph_edges: usize,
    pub histogram_cells: usize,
    pub palette_colours: usize,
    pub initial_regions: usize,
    pub raw_regions: usize,
    pub corrected_regions: usize,
    pub regularized_regions: usize,
    pub merged_regions: usize,
    pub forced_small_region_merges: usize,
    pub compatible_region_merges: usize,
    pub paint_aware_region_merge_proposals: usize,
    pub paint_aware_region_merges: usize,
    pub paint_aware_merge_render_rejected: bool,
    /// Adjacent final regions whose serialized Paint is exactly identical
    /// and can therefore share one topology owner without a colour change.
    pub exact_paint_region_merges: usize,
    pub preserved_independent_materials: usize,
    pub antialias_pixels: usize,
    pub core_sampled_regions: usize,
    pub antialias_split_regions: usize,
    pub boundary_regularization_passes: usize,
    pub boundary_regularization_moved_pixels: usize,
    pub boundary_regularization_source_edges: usize,
    pub thin_paint_examined: usize,
    pub thin_paint_protected: usize,
    pub thin_paint_refined: usize,
    pub thin_paint_rollbacks: usize,
    pub thin_paint_reassigned_pixels: usize,
    pub thin_paint_removed_regions: usize,
    pub adaptive_patch_candidate_faces: usize,
    pub adaptive_patch_split_faces: usize,
    pub adaptive_patch_added_regions: usize,
    pub effective_minimum_area: usize,
    pub local_minimum_area: usize,
    pub local_median_area: usize,
    pub local_maximum_area: usize,
}

#[derive(Clone, Debug)]
pub struct RegionStats {
    pub id: u32,
    pub area: usize,
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
    pub mean_rgb: [f32; 3],
    pub mean_lab: Lab,
}

#[derive(Clone, Debug)]
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<u32>,
    /// Source Paint owner retained when adaptive patches split one face into
    /// several geometry labels.  Python carries the corresponding
    /// `final_keys`; local Paint coupling uses equality only to recognize an
    /// artificial patch boundary.
    pub paint_keys: Vec<u32>,
    pub paint_samples: Vec<bool>,
    /// Quantized/regularized Paint prototypes used only for discrete
    /// ownership decisions. Gradient fitting continues to sample the native
    /// underpaint reference.
    pub canonical: Raster,
    pub regions: Vec<RegionStats>,
    pub summary: SegmentationSummary,
}

#[inline]
fn srgb_to_linear_channel(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_channel(value: f32) -> f32 {
    let value = value.max(0.0);
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
    .clamp(0.0, 1.0)
}

fn projection_alpha(value: [f32; 3], first: [f32; 3], second: [f32; 3]) -> f32 {
    let direction = [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ];
    let squared = direction
        .iter()
        .map(|channel| channel * channel)
        .sum::<f32>()
        .max(1e-8);
    ((value[0] - second[0]) * direction[0]
        + (value[1] - second[1]) * direction[1]
        + (value[2] - second[2]) * direction[2])
        / squared
}

fn mixture_prediction(
    value: [f32; 3],
    first: [f32; 3],
    second: [f32; 3],
    linear: bool,
) -> (f32, [f32; 3]) {
    let transform = |rgb: [f32; 3]| {
        if linear {
            [
                srgb_to_linear_channel(rgb[0]),
                srgb_to_linear_channel(rgb[1]),
                srgb_to_linear_channel(rgb[2]),
            ]
        } else {
            rgb
        }
    };
    let first_value = transform(first);
    let second_value = transform(second);
    let alpha = projection_alpha(transform(value), first_value, second_value);
    let amount = alpha.clamp(0.0, 1.0);
    let mixed = [
        second_value[0] + amount * (first_value[0] - second_value[0]),
        second_value[1] + amount * (first_value[1] - second_value[1]),
        second_value[2] + amount * (first_value[2] - second_value[2]),
    ];
    let predicted = if linear {
        [
            linear_to_srgb_channel(mixed[0]),
            linear_to_srgb_channel(mixed[1]),
            linear_to_srgb_channel(mixed[2]),
        ]
    } else {
        mixed
    };
    (alpha, predicted)
}

fn pair_mixture(
    first_value: [f32; 3],
    second_value: [f32; 3],
    first_parent: [f32; 3],
    second_parent: [f32; 3],
) -> (f32, f32, f32) {
    let mut best = (0.0, 0.0, f32::INFINITY);
    for linear in [false, true] {
        let (first_alpha, first_prediction) =
            mixture_prediction(first_value, first_parent, second_parent, linear);
        let (second_alpha, second_prediction) =
            mixture_prediction(second_value, first_parent, second_parent, linear);
        let error = delta_e2000(rgb_to_lab(first_value), rgb_to_lab(first_prediction)).max(
            delta_e2000(rgb_to_lab(second_value), rgb_to_lab(second_prediction)),
        );
        if error < best.2 {
            best = (first_alpha, second_alpha, error);
        }
    }
    best
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LabKey(i16, i16, i16);

#[derive(Clone, Debug)]
struct PaletteEntry {
    lab: Lab,
    weight: usize,
}

fn adaptive_tolerance(lightness: f32, config: &Config) -> f32 {
    let amount = ((lightness - config.dark_knee_lstar) / (100.0 - config.dark_knee_lstar).max(1.0))
        .clamp(0.0, 1.0);
    let amount = amount * amount * (3.0 - 2.0 * amount);
    config.quantization_dark_delta_e
        + (config.quantization_light_delta_e - config.quantization_dark_delta_e) * amount
}

fn effective_minimum_area(config: &Config, width: usize, height: usize) -> usize {
    let scale = width.max(height) as f32 / config.segmentation_reference_dimension.max(1) as f32;
    ((config.segmentation_min_size.max(1) as f32 * scale * scale).round() as usize).max(1)
}

fn local_area_map(roles: &EdgeRoles, config: &Config, maximum: usize) -> Vec<usize> {
    let minimum = config.segmentation_min_size.max(1) as usize;
    if !config.local_detail_adaptation || minimum >= maximum {
        return vec![maximum; roles.width * roles.height];
    }
    let scale = (roles.width.max(roles.height) as f32
        / config.segmentation_reference_dimension.max(1) as f32)
        .max(1.0);
    let mut window = (config.local_detail_window.max(1.0) * scale).round() as usize;
    window = window.max(9);
    if window.is_multiple_of(2) {
        window += 1;
    }
    let radius = window / 2;
    let padded_width = roles.width + 2 * radius;
    let padded_height = roles.height + 2 * radius;
    let stride = padded_width + 1;
    let mut integral = vec![0_i64; stride * (padded_height + 1)];
    for y in 0..padded_height {
        let mut row_sum = 0_i64;
        let source_y = (y as isize - radius as isize)
            .clamp(0, roles.height.saturating_sub(1) as isize) as usize;
        for x in 0..padded_width {
            let source_x = (x as isize - radius as isize)
                .clamp(0, roles.width.saturating_sub(1) as isize)
                as usize;
            let index = source_y * roles.width + source_x;
            row_sum += (roles.face_barrier[index] || roles.visible_ridge_centres[index]) as i64;
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row_sum;
        }
    }
    (0..roles.width * roles.height)
        .map(|index| {
            let x = index % roles.width;
            let y = index / roles.width;
            let x0 = x;
            let y0 = y;
            let x1 = x + window;
            let y1 = y + window;
            let count = integral[y1 * stride + x1] + integral[y0 * stride + x0]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0];
            let density = count.max(0) as f32 / (window * window) as f32 * scale;
            let pressure = (density / config.local_detail_density_pivot.max(1e-3))
                .max(0.0)
                .powi(4);
            let blend = 1.0 / (1.0 + pressure);
            (minimum as f32 + (maximum - minimum) as f32 * blend).round_ties_even() as usize
        })
        .collect()
}

fn weighted_merge(first: Lab, first_weight: usize, second: Lab, second_weight: usize) -> Lab {
    let total = (first_weight + second_weight).max(1) as f32;
    Lab {
        l: (first.l * first_weight as f32 + second.l * second_weight as f32) / total,
        a: (first.a * first_weight as f32 + second.a * second_weight as f32) / total,
        b: (first.b * first_weight as f32 + second.b * second_weight as f32) / total,
    }
}

fn build_palette(lab: &[Lab], config: &Config) -> (Vec<u32>, Vec<Lab>, usize) {
    let mut histogram = HashMap::<LabKey, usize>::new();
    for value in lab {
        *histogram
            .entry(LabKey(
                value.l.round_ties_even() as i16,
                value.a.round_ties_even() as i16,
                value.b.round_ties_even() as i16,
            ))
            .or_default() += 1;
    }
    let histogram_cells = histogram.len();
    let mut bins: Vec<(LabKey, usize)> = histogram.into_iter().collect();
    bins.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut palette = Vec::<PaletteEntry>::new();
    let mut assignments = HashMap::<LabKey, u32>::new();
    for (key, count) in bins {
        let colour = Lab {
            l: key.0 as f32,
            a: key.1 as f32,
            b: key.2 as f32,
        };
        // Match the reference implementation exactly: compare a histogram
        // cell with every existing palette representative.  The former
        // bucket shortcut could omit the true CIEDE2000 nearest colour after
        // a representative moved, changing both ownership and topology.
        let best = (0..palette.len())
            .map(|index| (index, delta_e2000(palette[index].lab, colour)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let selected = if let Some((index, distance)) = best {
            let threshold = adaptive_tolerance((palette[index].lab.l + colour.l) * 0.5, config);
            if distance <= threshold {
                let previous = palette[index].clone();
                palette[index].lab = weighted_merge(previous.lab, previous.weight, colour, count);
                palette[index].weight += count;
                index
            } else {
                let index = palette.len();
                palette.push(PaletteEntry {
                    lab: colour,
                    weight: count,
                });
                index
            }
        } else {
            let index = palette.len();
            palette.push(PaletteEntry {
                lab: colour,
                weight: count,
            });
            index
        };
        assignments.insert(key, selected as u32);
    }
    let map = lab
        .iter()
        .map(|value| {
            assignments[&LabKey(
                value.l.round_ties_even() as i16,
                value.a.round_ties_even() as i16,
                value.b.round_ties_even() as i16,
            )]
        })
        .collect();
    (
        map,
        palette.into_iter().map(|entry| entry.lab).collect(),
        histogram_cells,
    )
}

fn compact_connected(values: &[u32], width: usize, height: usize) -> (Vec<u32>, usize) {
    let mut union = UnionFind::new(values.len());
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if x + 1 < width && values[index] == values[index + 1] {
                union.union(index, index + 1);
            }
            if y + 1 < height && values[index] == values[index + width] {
                union.union(index, index + width);
            }
        }
    }
    let mut pixel_roots = Vec::with_capacity(values.len());
    let mut components = HashMap::<usize, (u32, usize)>::new();
    for (index, &palette) in values.iter().enumerate() {
        let root = union.find(index);
        pixel_roots.push(root);
        components
            .entry(root)
            .and_modify(|component| component.1 = component.1.min(index))
            .or_insert((palette, index));
    }
    // scipy.ndimage.label is invoked once per palette in ascending palette
    // order.  Within one palette it numbers components by their first
    // row-major sample.  Preserve that exact ordering because later passes
    // use stable label order to resolve otherwise equal candidates.
    let mut ordered: Vec<(usize, u32, usize)> = components
        .into_iter()
        .map(|(root, (palette, first))| (root, palette, first))
        .collect();
    ordered.sort_by_key(|&(_, palette, first)| (palette, first));
    let root_labels: HashMap<usize, u32> = ordered
        .iter()
        .enumerate()
        .map(|(label, &(root, _, _))| (root, label as u32))
        .collect();
    let labels = pixel_roots
        .into_iter()
        .map(|root| root_labels[&root])
        .collect();
    (labels, ordered.len())
}

fn compact_values(values: &[u32]) -> (Vec<u32>, usize) {
    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let compact = values
        .iter()
        .map(|value| unique.binary_search(value).unwrap_or(0) as u32)
        .collect();
    (compact, unique.len())
}

fn component_pixels(labels: &[u32], count: usize) -> Vec<Vec<usize>> {
    let mut pixels = vec![Vec::new(); count];
    for (index, &label) in labels.iter().enumerate() {
        pixels[label as usize].push(index);
    }
    pixels
}

fn has_core(labels: &[u32], width: usize, height: usize, label: u32, indices: &[usize]) -> bool {
    indices.iter().any(|&index| {
        let x = index % width;
        let y = index / width;
        x > 0
            && y > 0
            && x + 1 < width
            && y + 1 < height
            && labels[index - 1] == label
            && labels[index + 1] == label
            && labels[index - width] == label
            && labels[index + width] == label
    })
}

fn median_channel(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn median_lab(values: &[Lab], indices: &[usize]) -> Lab {
    let mut lightness: Vec<f32> = indices.iter().map(|&index| values[index].l).collect();
    let mut a: Vec<f32> = indices.iter().map(|&index| values[index].a).collect();
    let mut b: Vec<f32> = indices.iter().map(|&index| values[index].b).collect();
    Lab {
        l: median_channel(&mut lightness),
        a: median_channel(&mut a),
        b: median_channel(&mut b),
    }
}

fn project_to_lab_segment(value: Lab, first: Lab, second: Lab) -> Lab {
    let dl = second.l - first.l;
    let da = second.a - first.a;
    let db = second.b - first.b;
    let length_squared = dl * dl + da * da + db * db;
    if length_squared <= 1e-8 {
        return first;
    }
    let parameter =
        (((value.l - first.l) * dl + (value.a - first.a) * da + (value.b - first.b) * db)
            / length_squared)
            .clamp(0.0, 1.0);
    Lab {
        l: first.l + parameter * dl,
        a: first.a + parameter * da,
        b: first.b + parameter * db,
    }
}

fn mixture_error(source: Lab, neighbours: &[Lab]) -> f32 {
    if neighbours.is_empty() {
        return f32::INFINITY;
    }
    if neighbours.len() == 1 {
        return delta_e2000(source, neighbours[0]);
    }
    let mut best = f32::INFINITY;
    for (first_index, &first) in neighbours[..neighbours.len() - 1].iter().enumerate() {
        for &second in neighbours.iter().skip(first_index + 1) {
            let mixture = project_to_lab_segment(source, first, second);
            best = best.min(delta_e2000(source, mixture));
        }
    }
    if best <= 1.5 {
        return best;
    }
    // Raster antialiasing is commonly closer to an sRGB interpolation path
    // than a straight Lab segment.  Evaluate that nonlinear path only for
    // the ambiguous remainder, as the Python reference does.
    let source_rgb = lab_to_rgb(source);
    for (first_index, &first) in neighbours[..neighbours.len() - 1].iter().enumerate() {
        let start = lab_to_rgb(first);
        for &second in neighbours.iter().skip(first_index + 1) {
            let end = lab_to_rgb(second);
            let direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
            let length_squared = direction.iter().map(|value| value * value).sum::<f32>();
            if length_squared <= 1e-8 {
                continue;
            }
            let parameter = (((source_rgb[0] - start[0]) * direction[0]
                + (source_rgb[1] - start[1]) * direction[1]
                + (source_rgb[2] - start[2]) * direction[2])
                / length_squared)
                .clamp(0.0, 1.0);
            let mixture = rgb_to_lab([
                start[0] + parameter * direction[0],
                start[1] + parameter * direction[1],
                start[2] + parameter * direction[2],
            ]);
            best = best.min(delta_e2000(source, mixture));
        }
    }
    best
}

#[derive(Clone, Copy, Debug)]
struct PixelBounds {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

impl PixelBounds {
    fn include(&mut self, x: usize, y: usize) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x + 1);
        self.max_y = self.max_y.max(y + 1);
    }
}

#[allow(clippy::too_many_arguments)]
fn merge_small_components(
    palette_map: &mut [u32],
    palette_lab: &[Lab],
    source_lab: &[Lab],
    width: usize,
    height: usize,
    local_area: &[usize],
    maximum_area: usize,
    config: &Config,
) -> (usize, usize, usize) {
    let mut reassigned = 0;
    let mut merged = 0;
    let mut preserved = 0;
    let mut bounds = vec![None::<PixelBounds>; palette_lab.len()];
    for (index, &palette) in palette_map.iter().enumerate() {
        let x = index % width;
        let y = index / width;
        match &mut bounds[palette as usize] {
            Some(value) => value.include(x, y),
            slot @ None => {
                *slot = Some(PixelBounds {
                    min_x: x,
                    min_y: y,
                    max_x: x + 1,
                    max_y: y + 1,
                });
            }
        }
    }

    let mut visited = vec![0_u32; palette_map.len()];
    let mut epoch = 0_u32;
    for palette in 0..palette_lab.len() {
        let Some(extent) = bounds[palette] else {
            continue;
        };
        epoch = epoch.wrapping_add(1).max(1);
        if epoch == 1 {
            visited.fill(0);
        }
        for y in extent.min_y..extent.max_y {
            for x in extent.min_x..extent.max_x {
                let start = y * width + x;
                if palette_map[start] != palette as u32 || visited[start] == epoch {
                    continue;
                }
                let mut queue = VecDeque::from([start]);
                let mut component = Vec::new();
                visited[start] = epoch;
                while let Some(index) = queue.pop_front() {
                    component.push(index);
                    let px = index % width;
                    let py = index / width;
                    let neighbours = [
                        (px > 0).then(|| index - 1),
                        (px + 1 < width).then(|| index + 1),
                        (py > 0).then(|| index - width),
                        (py + 1 < height).then(|| index + width),
                    ];
                    for neighbour in neighbours.into_iter().flatten() {
                        if palette_map[neighbour] == palette as u32 && visited[neighbour] != epoch {
                            visited[neighbour] = epoch;
                            queue.push_back(neighbour);
                        }
                    }
                }
                if component.len() >= maximum_area {
                    continue;
                }
                let mut local_limits: Vec<usize> =
                    component.iter().map(|&index| local_area[index]).collect();
                local_limits.sort_unstable();
                let local_minimum = if local_limits.len().is_multiple_of(2) {
                    (local_limits[local_limits.len() / 2 - 1]
                        + local_limits[local_limits.len() / 2])
                        / 2
                } else {
                    local_limits[local_limits.len() / 2]
                };
                let locally_visible = component.len() >= local_minimum;
                let component_lookup: HashSet<usize> = component.iter().copied().collect();
                let mut neighbour_ids = HashSet::<u32>::new();
                for &index in &component {
                    let px = index % width;
                    let py = index / width;
                    for neighbour in [
                        (px > 0).then(|| index - 1),
                        (px + 1 < width).then(|| index + 1),
                        (py > 0).then(|| index - width),
                        (py + 1 < height).then(|| index + width),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let owner = palette_map[neighbour];
                        if owner != palette as u32 {
                            neighbour_ids.insert(owner);
                        }
                    }
                }
                if neighbour_ids.is_empty() {
                    continue;
                }
                let mut neighbours: Vec<u32> = neighbour_ids.into_iter().collect();
                neighbours.sort_unstable();
                let neighbour_colours: Vec<Lab> = neighbours
                    .iter()
                    .map(|&owner| palette_lab[owner as usize])
                    .collect();
                let source = median_lab(source_lab, &component);
                let best = neighbour_colours
                    .iter()
                    .enumerate()
                    .map(|(offset, &colour)| (offset, delta_e2000(source, colour)))
                    .min_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| left.0.cmp(&right.0))
                    })
                    .unwrap();
                let best_colour = neighbour_colours[best.0];
                let threshold = adaptive_tolerance((source.l + best_colour.l) * 0.5, config);
                let minimum_l = neighbour_colours
                    .iter()
                    .map(|value| value.l)
                    .fold(f32::INFINITY, f32::min);
                let maximum_l = neighbour_colours
                    .iter()
                    .map(|value| value.l)
                    .fold(f32::NEG_INFINITY, f32::max);
                let extremum = source.l <= minimum_l - 6.0 || source.l >= maximum_l + 6.0;
                let source_chroma = (source.a * source.a + source.b * source.b).sqrt();
                let maximum_chroma = neighbour_colours
                    .iter()
                    .map(|value| (value.a * value.a + value.b * value.b).sqrt())
                    .fold(0.0_f32, f32::max);
                let core = component.iter().any(|&index| {
                    let px = index % width;
                    let py = index / width;
                    px > 0
                        && py > 0
                        && px + 1 < width
                        && py + 1 < height
                        && component_lookup.contains(&(index - 1))
                        && component_lookup.contains(&(index + 1))
                        && component_lookup.contains(&(index - width))
                        && component_lookup.contains(&(index + width))
                });
                let residual = mixture_error(source, &neighbour_colours);
                let independent = component.len() >= 3
                    && best.1 > threshold
                    && residual > (0.75 * threshold).max(2.5)
                    && (core || source_chroma >= maximum_chroma + 4.0);
                let local_material = locally_visible
                    && component.len() >= 3
                    && core
                    && best.1 > 0.5 * threshold
                    && residual > (0.5 * threshold).max(1.5);
                if (component.len() >= 3 && extremum && best.1 > threshold)
                    || independent
                    || local_material
                {
                    preserved += 1;
                    continue;
                }

                let mut changed_component = false;
                for &index in &component {
                    let value = source_lab[index];
                    let mut selected: Option<(usize, f32)> = None;
                    for (offset, &colour) in neighbour_colours.iter().enumerate() {
                        let error = delta_e2000(value, colour);
                        let pixel_threshold =
                            adaptive_tolerance((value.l + colour.l) * 0.5, config);
                        if error > (2.0 * pixel_threshold).max(35.0) {
                            continue;
                        }
                        if selected
                            .map(|current| {
                                error < current.1 || (error == current.1 && offset < current.0)
                            })
                            .unwrap_or(true)
                        {
                            selected = Some((offset, error));
                        }
                    }
                    let Some((offset, _)) = selected else {
                        continue;
                    };
                    let owner = neighbours[offset];
                    palette_map[index] = owner;
                    let px = index % width;
                    let py = index / width;
                    match &mut bounds[owner as usize] {
                        Some(value) => value.include(px, py),
                        slot @ None => {
                            *slot = Some(PixelBounds {
                                min_x: px,
                                min_y: py,
                                max_x: px + 1,
                                max_y: py + 1,
                            });
                        }
                    }
                    reassigned += 1;
                    changed_component = true;
                }
                merged += usize::from(changed_component);
            }
        }
    }
    (reassigned, merged, preserved)
}

fn region_stats(image: &Raster, labels: &[u32], count: usize) -> Vec<RegionStats> {
    let mut areas = vec![0_usize; count];
    let mut min_x = vec![usize::MAX; count];
    let mut min_y = vec![usize::MAX; count];
    let mut max_x = vec![0_usize; count];
    let mut max_y = vec![0_usize; count];
    let mut sums = vec![[0.0_f64; 3]; count];
    for (index, (&label, &rgb)) in labels.iter().zip(&image.pixels).enumerate() {
        let region = label as usize;
        let x = index % image.width;
        let y = index / image.width;
        areas[region] += 1;
        min_x[region] = min_x[region].min(x);
        min_y[region] = min_y[region].min(y);
        max_x[region] = max_x[region].max(x + 1);
        max_y[region] = max_y[region].max(y + 1);
        for channel in 0..3 {
            sums[region][channel] += rgb[channel] as f64;
        }
    }
    (0..count)
        .map(|id| {
            let divisor = areas[id].max(1) as f64;
            let mean_rgb = [
                (sums[id][0] / divisor) as f32,
                (sums[id][1] / divisor) as f32,
                (sums[id][2] / divisor) as f32,
            ];
            RegionStats {
                id: id as u32,
                area: areas[id],
                min_x: min_x[id],
                min_y: min_y[id],
                max_x: max_x[id],
                max_y: max_y[id],
                mean_rgb,
                mean_lab: rgb_to_lab(mean_rgb),
            }
        })
        .collect()
}

fn region_mean_raster_for(image: &Raster, labels: &[u32], count: usize) -> Raster {
    // Python's _region_mean_image averages in Lab, using float64 bincount
    // accumulators, casts the region means to float32, expands them back to
    // image shape, and only then converts the complete array to sRGB.
    let labs = lab_pixels(image);
    let mut areas = vec![0_usize; count];
    let mut sums = vec![[0.0_f64; 3]; count];
    for (&label, lab) in labels.iter().zip(labs) {
        let region = label as usize;
        areas[region] += 1;
        sums[region][0] += lab.l as f64;
        sums[region][1] += lab.a as f64;
        sums[region][2] += lab.b as f64;
    }
    let means: Vec<Lab> = (0..count)
        .map(|region| {
            let divisor = areas[region].max(1) as f64;
            Lab {
                l: (sums[region][0] / divisor) as f32,
                a: (sums[region][1] / divisor) as f32,
                b: (sums[region][2] / divisor) as f32,
            }
        })
        .collect();
    let mean_pixels: Vec<Lab> = labels.iter().map(|&label| means[label as usize]).collect();
    let pixels = lab_pixels_to_rgb(&mean_pixels);
    Raster::new(image.width, image.height, pixels)
}

#[derive(Clone, Debug)]
struct AntialiasCorrection {
    labels: Vec<u32>,
    paint_samples: Vec<bool>,
    antialias_pixels: usize,
    core_sampled_regions: usize,
    split_regions: usize,
}

#[derive(Clone, Debug)]
struct FlowEdge {
    to: usize,
    reverse: usize,
    capacity: f64,
}

#[derive(Clone, Debug)]
struct PreflowEdge {
    to: usize,
    reverse: usize,
    capacity: f64,
    flow: f64,
}

#[derive(Clone, Debug, Default)]
struct PreflowLevel {
    active: BTreeSet<usize>,
    inactive: BTreeSet<usize>,
}

fn add_authored_edge(
    graph: &mut [Vec<(usize, f64)>],
    node_order: &mut Vec<usize>,
    seen: &mut [bool],
    from: usize,
    to: usize,
    capacity: f64,
) {
    for node in [from, to] {
        if !seen[node] {
            seen[node] = true;
            node_order.push(node);
        }
    }
    if let Some(edge) = graph[from].iter_mut().find(|edge| edge.0 == to) {
        edge.1 = capacity;
    } else {
        graph[from].push((to, capacity));
    }
}

fn add_preflow_pair(graph: &mut [Vec<PreflowEdge>], from: usize, to: usize, capacity: f64) {
    if let Some(index) = graph[from].iter().position(|edge| edge.to == to) {
        graph[from][index].capacity = capacity;
        return;
    }
    let forward_reverse = graph[to].len();
    let reverse_reverse = graph[from].len();
    graph[from].push(PreflowEdge {
        to,
        reverse: forward_reverse,
        capacity,
        flow: 0.0,
    });
    graph[to].push(PreflowEdge {
        to: from,
        reverse: reverse_reverse,
        capacity: 0.0,
        flow: 0.0,
    });
}

fn preflow_push(
    graph: &mut [Vec<PreflowEdge>],
    excess: &mut [f64],
    from: usize,
    edge_index: usize,
    amount: f64,
) {
    let to = graph[from][edge_index].to;
    let reverse = graph[from][edge_index].reverse;
    graph[from][edge_index].flow += amount;
    graph[to][reverse].flow -= amount;
    excess[from] -= amount;
    excess[to] += amount;
}

fn preflow_reverse_bfs(graph: &[Vec<PreflowEdge>], target: usize) -> Vec<Option<usize>> {
    let mut heights = vec![None; graph.len()];
    let mut queue = VecDeque::from([target]);
    heights[target] = Some(0);
    while let Some(node) = queue.pop_front() {
        let following = heights[node].expect("visited node") + 1;
        // Residual edges are inserted in symmetric pairs, so the outgoing
        // neighbour order is also the predecessor insertion order used by
        // NetworkX's reverse BFS.
        for edge in &graph[node] {
            let predecessor = edge.to;
            let incoming = &graph[predecessor][edge.reverse];
            if heights[predecessor].is_none() && incoming.flow < incoming.capacity {
                heights[predecessor] = Some(following);
                queue.push_back(predecessor);
            }
        }
    }
    heights
}

fn preflow_reaches_sink_after_saturated_removal(
    graph: &[Vec<PreflowEdge>],
    sink: usize,
) -> Vec<bool> {
    let mut reached = vec![false; graph.len()];
    let mut queue = VecDeque::from([sink]);
    reached[sink] = true;
    while let Some(node) = queue.pop_front() {
        for edge in &graph[node] {
            let predecessor = edge.to;
            let incoming = &graph[predecessor][edge.reverse];
            // networkx.minimum_cut removes edges only when the two floats are
            // exactly equal.  A one-ulp over-capacity edge left by its
            // value-only preflow therefore remains traversable here even
            // though it would fail the ordinary residual `flow < capacity`
            // test used during the flow computation.
            if !reached[predecessor] && incoming.flow != incoming.capacity {
                reached[predecessor] = true;
                queue.push_back(predecessor);
            }
        }
    }
    reached
}

fn rebuild_preflow_levels(
    levels: &mut [PreflowLevel],
    heights: &[usize],
    excess: &[f64],
    source: usize,
    sink: usize,
) {
    for level in levels.iter_mut() {
        level.active.clear();
        level.inactive.clear();
    }
    for node in 0..heights.len() {
        if node == source || node == sink {
            continue;
        }
        if excess[node] > 0.0 {
            levels[heights[node]].active.insert(node);
        } else {
            levels[heights[node]].inactive.insert(node);
        }
    }
}

fn networkx_preflow_cut_assignment(
    component: &[usize],
    alpha: &[f32],
    first_seed: &[bool],
    second_seed: &[bool],
    width: usize,
) -> Vec<bool> {
    let count = component.len();
    let source = count;
    let sink = count + 1;
    let node_count = count + 2;
    let mut lookup = HashMap::<usize, usize>::with_capacity(count);
    for (local, &index) in component.iter().enumerate() {
        lookup.insert(index, local);
    }

    // Recreate DiGraph insertion order first, then build its residual graph.
    // The value-only highest-label preflow used by NetworkX intentionally
    // leaves a preflow rather than converting it to a feasible max flow, so
    // residual edge order is observable when equal cuts exist.
    let mut authored = vec![Vec::<(usize, f64)>::new(); node_count];
    let mut node_order = Vec::<usize>::with_capacity(node_count);
    let mut seen = vec![false; node_count];
    let hard = 1_000_000.0_f64;
    for (local, &index) in component.iter().enumerate() {
        let value = alpha[local].clamp(0.0, 1.0) as f64;
        add_authored_edge(
            &mut authored,
            &mut node_order,
            &mut seen,
            source,
            local,
            if first_seed[index] {
                hard
            } else {
                value * value
            },
        );
        add_authored_edge(
            &mut authored,
            &mut node_order,
            &mut seen,
            local,
            sink,
            if second_seed[index] {
                hard
            } else {
                (1.0 - value) * (1.0 - value)
            },
        );
        let x = index % width;
        for neighbour in [Some(index + width), (x + 1 < width).then_some(index + 1)]
            .into_iter()
            .flatten()
        {
            if let Some(&other) = lookup.get(&neighbour) {
                add_authored_edge(
                    &mut authored,
                    &mut node_order,
                    &mut seen,
                    local,
                    other,
                    0.15,
                );
                add_authored_edge(
                    &mut authored,
                    &mut node_order,
                    &mut seen,
                    other,
                    local,
                    0.15,
                );
            }
        }
    }
    let mut graph = vec![Vec::<PreflowEdge>::new(); node_count];
    for &from in &node_order {
        for &(to, capacity) in &authored[from] {
            if capacity > 0.0 {
                add_preflow_pair(&mut graph, from, to, capacity);
            }
        }
    }

    let initial = preflow_reverse_bfs(&graph, sink);
    if initial[source].is_none() {
        return vec![true; count];
    }
    let n = graph.len();
    let mut heights: Vec<usize> = initial
        .iter()
        .map(|height| height.unwrap_or(n + 1))
        .collect();
    let mut max_height = heights
        .iter()
        .enumerate()
        .filter(|&(node, _)| node != source)
        .map(|(_, &height)| height)
        .filter(|&height| height <= n)
        .max()
        .unwrap_or(0);
    heights[source] = n;
    let mut current_edge = vec![0_usize; n];
    let mut excess = vec![0.0_f64; n];
    let mut levels = vec![PreflowLevel::default(); 2 * n + 2];
    for edge_index in 0..graph[source].len() {
        let capacity = graph[source][edge_index].capacity;
        if capacity > 0.0 {
            preflow_push(&mut graph, &mut excess, source, edge_index, capacity);
        }
    }
    rebuild_preflow_levels(&mut levels, &heights, &excess, source, sink);
    let edge_count: usize = graph.iter().map(Vec::len).sum();
    let threshold = n + edge_count;
    let mut work = 0_usize;
    let mut height = max_height;
    while height > 0 {
        let Some(&node) = levels[height].active.iter().next() else {
            height -= 1;
            continue;
        };
        let old_height = height;
        levels[height].active.remove(&node);
        loop {
            let edge_index = current_edge[node];
            let to = graph[node][edge_index].to;
            if heights[node] == heights[to] + 1
                && graph[node][edge_index].flow < graph[node][edge_index].capacity
            {
                let amount = excess[node]
                    .min(graph[node][edge_index].capacity - graph[node][edge_index].flow);
                preflow_push(&mut graph, &mut excess, node, edge_index, amount);
                if to != source && to != sink && levels[heights[to]].inactive.remove(&to) {
                    levels[heights[to]].active.insert(to);
                }
                if excess[node] == 0.0 {
                    levels[heights[node]].inactive.insert(node);
                    break;
                }
            }
            current_edge[node] += 1;
            if current_edge[node] == graph[node].len() {
                current_edge[node] = 0;
                work += graph[node].len();
                heights[node] = graph[node]
                    .iter()
                    .filter(|edge| edge.flow < edge.capacity)
                    .map(|edge| heights[edge.to])
                    .min()
                    .expect("residual neighbour")
                    + 1;
                if heights[node] >= n - 1 {
                    levels[heights[node]].active.insert(node);
                    break;
                }
                height = heights[node];
            }
        }

        if work >= threshold {
            let relabelled = preflow_reverse_bfs(&graph, sink);
            max_height = relabelled.iter().flatten().copied().max().unwrap_or(0);
            for node in 0..n {
                if let Some(new_height) = relabelled[node] {
                    if node != sink {
                        heights[node] = new_height;
                    }
                } else if heights[node] < n {
                    heights[node] = n + 1;
                }
            }
            rebuild_preflow_levels(&mut levels, &heights, &excess, source, sink);
            height = max_height;
            work = 0;
        } else if levels[old_height].active.is_empty() && levels[old_height].inactive.is_empty() {
            for node in 0..n {
                if heights[node] > old_height && heights[node] <= max_height {
                    heights[node] = n + 1;
                }
            }
            rebuild_preflow_levels(&mut levels, &heights, &excess, source, sink);
            height = old_height - 1;
            max_height = height;
        } else {
            max_height = max_height.max(height);
        }
    }

    let reaches_sink = preflow_reaches_sink_after_saturated_removal(&graph, sink);
    let assignment: Vec<bool> = (0..count).map(|node| !reaches_sink[node]).collect();
    if let Ok(path) = std::env::var("PICVEC_PREFLOW_GRAPH_DIAGNOSTIC") {
        let diagnostic_index = 911_usize.saturating_mul(width).saturating_add(383);
        if component.contains(&diagnostic_index) {
            let adjacency: Vec<Vec<(usize, f64, f64)>> = graph
                .iter()
                .map(|edges| {
                    edges
                        .iter()
                        .map(|edge| (edge.to, edge.capacity, edge.flow))
                        .collect()
                })
                .collect();
            let value = serde_json::json!({
                "node_order": node_order,
                "adjacency": adjacency,
                "heights": heights,
                "excess": excess,
                "assignment": assignment,
            });
            let _ = std::fs::write(path, serde_json::to_vec(&value).unwrap_or_default());
        }
    }
    assignment
}

fn add_flow_edge(graph: &mut [Vec<FlowEdge>], from: usize, to: usize, capacity: f64) {
    let forward_reverse = graph[to].len();
    let reverse_reverse = graph[from].len();
    graph[from].push(FlowEdge {
        to,
        reverse: forward_reverse,
        capacity,
    });
    graph[to].push(FlowEdge {
        to: from,
        reverse: reverse_reverse,
        capacity: 0.0,
    });
}

fn add_bidirectional_flow_edge(
    graph: &mut [Vec<FlowEdge>],
    first: usize,
    second: usize,
    first_capacity: f64,
    second_capacity: f64,
) {
    let first_reverse = graph[second].len();
    let second_reverse = graph[first].len();
    graph[first].push(FlowEdge {
        to: second,
        reverse: first_reverse,
        capacity: first_capacity,
    });
    graph[second].push(FlowEdge {
        to: first,
        reverse: second_reverse,
        capacity: second_capacity,
    });
}

fn flow_levels(graph: &[Vec<FlowEdge>], source: usize) -> Vec<i32> {
    let mut levels = vec![-1_i32; graph.len()];
    let mut queue = VecDeque::from([source]);
    levels[source] = 0;
    while let Some(vertex) = queue.pop_front() {
        for edge in &graph[vertex] {
            if edge.capacity > 0.0 && levels[edge.to] < 0 {
                levels[edge.to] = levels[vertex] + 1;
                queue.push_back(edge.to);
            }
        }
    }
    levels
}

fn send_flow(
    vertex: usize,
    sink: usize,
    available: f64,
    levels: &[i32],
    offsets: &mut [usize],
    graph: &mut [Vec<FlowEdge>],
) -> f64 {
    if vertex == sink {
        return available;
    }
    while offsets[vertex] < graph[vertex].len() {
        let edge_index = offsets[vertex];
        let to = graph[vertex][edge_index].to;
        if graph[vertex][edge_index].capacity > 0.0 && levels[to] == levels[vertex] + 1 {
            let capacity = graph[vertex][edge_index].capacity;
            let sent = send_flow(to, sink, available.min(capacity), levels, offsets, graph);
            if sent > 0.0 {
                let reverse = graph[vertex][edge_index].reverse;
                graph[vertex][edge_index].capacity -= sent;
                graph[to][reverse].capacity += sent;
                return sent;
            }
        }
        offsets[vertex] += 1;
    }
    0.0
}

fn graph_cut_assignment(
    component: &[usize],
    alpha: &[f32],
    first_seed: &[bool],
    second_seed: &[bool],
    width: usize,
) -> Vec<bool> {
    let count = component.len();
    let source = count;
    let sink = count + 1;
    let mut lookup = HashMap::<usize, usize>::with_capacity(count);
    for (local, &index) in component.iter().enumerate() {
        lookup.insert(index, local);
    }
    let mut graph = vec![Vec::<FlowEdge>::new(); count + 2];
    let hard = 1_000_000.0_f64;
    for (local, &index) in component.iter().enumerate() {
        let value = alpha[local].clamp(0.0, 1.0) as f64;
        add_flow_edge(
            &mut graph,
            source,
            local,
            if first_seed[index] {
                hard
            } else {
                value * value
            },
        );
        add_flow_edge(
            &mut graph,
            local,
            sink,
            if second_seed[index] {
                hard
            } else {
                (1.0 - value) * (1.0 - value)
            },
        );
        let x = index % width;
        for neighbour in [Some(index + width), (x + 1 < width).then_some(index + 1)]
            .into_iter()
            .flatten()
        {
            if let Some(&other) = lookup.get(&neighbour) {
                // NetworkX represents the two authored directed edges as one
                // residual edge pair with capacity in both directions.  Two
                // independent forward/reverse pairs have the same nominal
                // cut energy but a different residual plateau and therefore
                // choose a different owner when several minimum cuts tie.
                add_bidirectional_flow_edge(&mut graph, local, other, 0.15, 0.15);
            }
        }
    }
    loop {
        let levels = flow_levels(&graph, source);
        if levels[sink] < 0 {
            break;
        }
        let mut offsets = vec![0_usize; graph.len()];
        loop {
            let sent = send_flow(
                source,
                sink,
                f64::INFINITY,
                &levels,
                &mut offsets,
                &mut graph,
            );
            if sent <= 0.0 {
                break;
            }
        }
    }
    // NetworkX's minimum_cut asks which nodes can still reach the sink in
    // the residual graph and returns the complementary, maximal source side.
    // Using source reachability instead selects the minimal source side; both
    // cuts have the same energy, but the latter moves a thin AA tail to the
    // opposite Paint owner on flat plateaus.
    let mut reaches_sink = vec![false; graph.len()];
    let mut queue = VecDeque::from([sink]);
    reaches_sink[sink] = true;
    while let Some(vertex) = queue.pop_front() {
        for from in 0..graph.len() {
            if !reaches_sink[from]
                && graph[from]
                    .iter()
                    .any(|edge| edge.to == vertex && edge.capacity > 0.0)
            {
                reaches_sink[from] = true;
                queue.push_back(from);
            }
        }
    }
    reaches_sink[..count].iter().map(|&value| !value).collect()
}

fn native_antialias_width(
    component: &[usize],
    labels: &[u32],
    label: u32,
    width: usize,
    height: usize,
) -> bool {
    component.iter().all(|&index| {
        let x = index % width;
        let y = index / width;
        (-3_isize..=3).any(|dy| {
            (-3_isize..=3).any(|dx| {
                if dx * dx + dy * dy > 9 {
                    return false;
                }
                let px = x as isize + dx;
                let py = y as isize + dy;
                px < 0
                    || py < 0
                    || px >= width as isize
                    || py >= height as isize
                    || labels[py as usize * width + px as usize] != label
            })
        })
    })
}

fn correct_antialias_partition(
    image: &Raster,
    labels: &[u32],
    count: usize,
) -> AntialiasCorrection {
    let mut core = vec![true; labels.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let index = y * image.width + x;
            let label = labels[index];
            'neighbourhood: for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let px = (x as isize + dx).clamp(0, image.width as isize - 1) as usize;
                    let py = (y as isize + dy).clamp(0, image.height as isize - 1) as usize;
                    if labels[py * image.width + px] != label {
                        core[index] = false;
                        break 'neighbourhood;
                    }
                }
            }
        }
    }
    let mut core_values = vec![Vec::<[f32; 3]>::new(); count];
    let mut totals = vec![0_usize; count];
    for (index, (&label, &rgb)) in labels.iter().zip(&image.pixels).enumerate() {
        totals[label as usize] += 1;
        if core[index] {
            core_values[label as usize].push(rgb);
        }
    }
    let mut stable = vec![false; count];
    let mut parents = vec![[0.0_f32; 3]; count];
    let mut parent_lab = vec![Lab::default(); count];
    for label in 0..count {
        if core_values[label].len() < 3 {
            continue;
        }
        stable[label] = true;
        for channel in 0..3 {
            let mut values: Vec<f32> = core_values[label]
                .iter()
                .map(|value| value[channel])
                .collect();
            parents[label][channel] = median_channel(&mut values);
        }
        parent_lab[label] = rgb_to_lab(parents[label]);
    }

    let source_lab = lab_pixels(image);
    let mut antialias = vec![false; labels.len()];
    let mut inspect = |first: usize, second: usize| {
        let first_label = labels[first] as usize;
        let second_label = labels[second] as usize;
        if first_label == second_label || !stable[first_label] || !stable[second_label] {
            return;
        }
        if delta_e2000(parent_lab[first_label], parent_lab[second_label]) < 4.6 {
            return;
        }
        let (alpha_first, alpha_second, error) = pair_mixture(
            image.pixels[first],
            image.pixels[second],
            parents[first_label],
            parents[second_label],
        );
        let across = delta_e2000(source_lab[first], source_lab[second]);
        let intermediate = (alpha_first > 0.05 && alpha_first < 0.95)
            || (alpha_second > 0.05 && alpha_second < 0.95);
        let explained = error <= 1.5
            && (-0.08..=1.08).contains(&alpha_first)
            && (-0.08..=1.08).contains(&alpha_second)
            && (intermediate || across <= 2.3);
        if !explained {
            return;
        }
        if delta_e2000(source_lab[first], parent_lab[first_label]) > 1.0 && alpha_first < 0.98 {
            antialias[first] = true;
        }
        if delta_e2000(source_lab[second], parent_lab[second_label]) > 1.0 && alpha_second > 0.02 {
            antialias[second] = true;
        }
    };
    for y in 0..image.height {
        for x in 0..image.width {
            let index = y * image.width + x;
            if x + 1 < image.width && labels[index] != labels[index + 1] {
                inspect(index, index + 1);
            }
            if y + 1 < image.height && labels[index] != labels[index + image.width] {
                inspect(index, index + image.width);
            }
        }
    }
    let mut antialias_counts = vec![0_usize; count];
    let mut core_counts = vec![0_usize; count];
    for (index, &label) in labels.iter().enumerate() {
        antialias_counts[label as usize] += usize::from(antialias[index]);
        core_counts[label as usize] += usize::from(core[index]);
    }
    let use_core: Vec<bool> = (0..count)
        .map(|label| {
            let boundary_count = totals[label].saturating_sub(core_counts[label]).max(1);
            stable[label]
                && antialias_counts[label] >= 3
                && antialias_counts[label] as f32 / boundary_count as f32 >= 0.02
        })
        .collect();
    let mut paint_samples: Vec<bool> = labels
        .iter()
        .enumerate()
        .map(|(index, &label)| !use_core[label as usize] || core[index])
        .collect();

    let mut contacts = vec![HashMap::<u32, usize>::new(); count];
    let mut record_contact = |first: usize, second: usize| {
        let left = labels[first] as usize;
        let right = labels[second] as usize;
        if left == right {
            return;
        }
        if stable[right] {
            *contacts[left].entry(right as u32).or_default() += 1;
        }
        if stable[left] {
            *contacts[right].entry(left as u32).or_default() += 1;
        }
    };
    for y in 0..image.height {
        for x in 0..image.width {
            let index = y * image.width + x;
            if x + 1 < image.width {
                record_contact(index, index + 1);
            }
            if y + 1 < image.height {
                record_contact(index, index + image.width);
            }
        }
    }

    let components = component_pixels(labels, count);
    let mut corrected = labels.to_vec();
    let mut split_regions = 0_usize;
    for label in 0..count {
        let component = &components[label];
        if component.is_empty()
            || !native_antialias_width(component, labels, label as u32, image.width, image.height)
        {
            continue;
        }
        let mut candidates: Vec<(u32, usize)> = contacts[label]
            .iter()
            .map(|(&owner, &amount)| (owner, amount))
            .collect();
        candidates.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        candidates.truncate(4);
        if candidates.len() < 2 {
            continue;
        }
        let mut best: Option<(f32, u32, u32, Vec<f32>)> = None;
        for first_index in 0..candidates.len() - 1 {
            let first = candidates[first_index].0;
            for &(second, _) in candidates.iter().skip(first_index + 1) {
                if delta_e2000(parent_lab[first as usize], parent_lab[second as usize]) < 4.6 {
                    continue;
                }
                let mut alpha = Vec::<f32>::with_capacity(component.len());
                let mut errors = Vec::<f32>::with_capacity(component.len());
                let mut valid = true;
                for &index in component {
                    let (amount, _, error) = pair_mixture(
                        image.pixels[index],
                        image.pixels[index],
                        parents[first as usize],
                        parents[second as usize],
                    );
                    if error > 1.5 || !(-0.08..=1.08).contains(&amount) {
                        valid = false;
                        break;
                    }
                    alpha.push(amount);
                    errors.push(error);
                }
                if !valid
                    || alpha.is_empty()
                    || percentile(alpha.clone(), 0.10) > 0.35
                    || percentile(alpha.clone(), 0.90) < 0.65
                {
                    continue;
                }
                let score = percentile(errors, 0.90);
                if best
                    .as_ref()
                    .map(|current| score < current.0)
                    .unwrap_or(true)
                {
                    best = Some((score, first, second, alpha));
                }
            }
        }
        let Some((_, first, second, alpha)) = best else {
            continue;
        };
        let mut first_seed = vec![false; labels.len()];
        let mut second_seed = vec![false; labels.len()];
        for &index in component {
            let x = index % image.width;
            let y = index / image.width;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let px = (x as isize + dx).clamp(0, image.width as isize - 1) as usize;
                    let py = (y as isize + dy).clamp(0, image.height as isize - 1) as usize;
                    let owner = corrected[py * image.width + px];
                    first_seed[index] |= owner == first;
                    second_seed[index] |= owner == second;
                }
            }
        }
        if !component.iter().any(|&index| first_seed[index])
            || !component.iter().any(|&index| second_seed[index])
        {
            continue;
        }
        let first_assignment = networkx_preflow_cut_assignment(
            component,
            &alpha,
            &first_seed,
            &second_seed,
            image.width,
        );
        if let Ok(path) = std::env::var("PICVEC_ANTIALIAS_DIAGNOSTIC") {
            let diagnostic_index = 911_usize.saturating_mul(image.width).saturating_add(383);
            if component.contains(&diagnostic_index) {
                let coordinates: Vec<[usize; 2]> = component
                    .iter()
                    .map(|&index| [index / image.width, index % image.width])
                    .collect();
                let first_seed_values: Vec<bool> =
                    component.iter().map(|&index| first_seed[index]).collect();
                let second_seed_values: Vec<bool> =
                    component.iter().map(|&index| second_seed[index]).collect();
                let value = serde_json::json!({
                    "label": label,
                    "first": first,
                    "second": second,
                    "coordinates": coordinates,
                    "alpha": alpha,
                    "first_seed": first_seed_values,
                    "second_seed": second_seed_values,
                    "assignment": first_assignment,
                });
                let _ = std::fs::write(path, serde_json::to_vec(&value).unwrap_or_default());
            }
        }
        let first_count = first_assignment.iter().filter(|&&value| value).count();
        if first_count == 0 || first_count == component.len() {
            continue;
        }
        for (offset, &index) in component.iter().enumerate() {
            corrected[index] = if first_assignment[offset] {
                first
            } else {
                second
            };
            antialias[index] = true;
            paint_samples[index] = false;
        }
        split_regions += 1;
    }
    let mut compact_map = vec![u32::MAX; count];
    let mut present = vec![false; count];
    for &label in &corrected {
        present[label as usize] = true;
    }
    let mut next = 0_u32;
    for (label, &is_present) in present.iter().enumerate() {
        if is_present {
            compact_map[label] = next;
            next += 1;
        }
    }
    for label in &mut corrected {
        *label = compact_map[*label as usize];
    }
    AntialiasCorrection {
        labels: corrected,
        paint_samples,
        antialias_pixels: antialias.iter().filter(|&&value| value).count(),
        core_sampled_regions: use_core.iter().filter(|&&value| value).count(),
        split_regions,
    }
}

/// Absolute Lab histogram quantization without transitive spatial chaining.
/// Only equal-palette four-connected samples become one geometry owner.
pub fn segment(image: &Raster, roles: &EdgeRoles, config: &Config) -> Segmentation {
    let source_lab = lab_pixels(image);
    let maximum_area = effective_minimum_area(config, image.width, image.height);
    let local_area = local_area_map(roles, config, maximum_area);
    let (mut palette_map, palette_lab, histogram_cells) = build_palette(&source_lab, config);
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let mut bytes = Vec::with_capacity(palette_map.len() * 4);
        for &palette in &palette_map {
            bytes.extend_from_slice(&palette.to_le_bytes());
        }
        let _ = std::fs::write(
            format!(
                "{prefix}-palette-map-{}x{}.u32le",
                image.width, image.height
            ),
            bytes,
        );
    }
    let (_, initial_count) = compact_connected(&palette_map, image.width, image.height);
    // The reference pass visits palette owners in stable palette order and
    // updates later owners' bounds as pixels move.  Repeating the complete
    // pass changes that ownership decision and over-erodes fine Paint.
    let (reassigned, merged_components, preserved) = merge_small_components(
        &mut palette_map,
        &palette_lab,
        &source_lab,
        image.width,
        image.height,
        &local_area,
        maximum_area,
        config,
    );
    let (labels, count) = compact_connected(&palette_map, image.width, image.height);
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let mut bytes = Vec::with_capacity(labels.len() * 4);
        for &label in &labels {
            bytes.extend_from_slice(&label.to_le_bytes());
        }
        let _ = std::fs::write(
            format!("{prefix}-raw-labels-{}x{}.u32le", image.width, image.height),
            bytes,
        );
    }
    let raw_count = count;
    let quantized = Raster::new(
        image.width,
        image.height,
        palette_map
            .iter()
            .map(|&palette| lab_to_rgb(palette_lab[palette as usize]))
            .collect(),
    );
    let correction = correct_antialias_partition(image, &labels, count);
    let labels = correction.labels;
    let count = labels
        .iter()
        .copied()
        .max()
        .map_or(0, |value| value as usize + 1);
    let regions = region_stats(image, &labels, count);
    let canonical = region_mean_raster_for(&quantized, &labels, count);
    let mut sorted_areas = local_area.clone();
    sorted_areas.sort_unstable();
    Segmentation {
        width: image.width,
        height: image.height,
        labels,
        paint_keys: (0..count as u32).collect(),
        paint_samples: correction.paint_samples,
        canonical,
        regions,
        summary: SegmentationSummary {
            graph_edges: image.width.saturating_sub(1) * image.height
                + image.height.saturating_sub(1) * image.width,
            histogram_cells,
            palette_colours: palette_lab.len(),
            initial_regions: initial_count,
            raw_regions: raw_count,
            corrected_regions: count,
            regularized_regions: count,
            merged_regions: count,
            forced_small_region_merges: reassigned,
            compatible_region_merges: merged_components,
            paint_aware_region_merge_proposals: 0,
            paint_aware_region_merges: 0,
            paint_aware_merge_render_rejected: false,
            exact_paint_region_merges: 0,
            preserved_independent_materials: preserved,
            antialias_pixels: correction.antialias_pixels,
            core_sampled_regions: correction.core_sampled_regions,
            antialias_split_regions: correction.split_regions,
            boundary_regularization_passes: 0,
            boundary_regularization_moved_pixels: 0,
            boundary_regularization_source_edges: 0,
            thin_paint_examined: 0,
            thin_paint_protected: 0,
            thin_paint_refined: 0,
            thin_paint_rollbacks: 0,
            thin_paint_reassigned_pixels: 0,
            thin_paint_removed_regions: 0,
            adaptive_patch_candidate_faces: 0,
            adaptive_patch_split_faces: 0,
            adaptive_patch_added_regions: 0,
            effective_minimum_area: maximum_area,
            local_minimum_area: sorted_areas.first().copied().unwrap_or(0),
            local_median_area: sorted_areas
                .get(sorted_areas.len() / 2)
                .copied()
                .unwrap_or(0),
            local_maximum_area: sorted_areas.last().copied().unwrap_or(0),
        },
    }
}

/// Replace the current partition after a topology-preserving merge pass.
///
/// Paint samples are pixel properties established by the anti-alias analysis,
/// so merging owners must not recompute or discard them.  This mirrors the
/// reference pipeline, which carries the correction mask unchanged from the
/// quantized partition into Paint fitting.
pub(crate) fn replace_merged_labels(
    image: &Raster,
    segmentation: &mut Segmentation,
    labels: Vec<u32>,
    accepted_merges: usize,
) {
    assert_eq!(labels.len(), segmentation.labels.len());
    let (labels, count) = compact_values(&labels);
    let canonical = region_mean_raster_for(&segmentation.canonical, &labels, count);
    segmentation.labels = labels;
    segmentation.paint_keys = (0..count as u32).collect();
    segmentation.canonical = canonical;
    segmentation.regions = region_stats(image, &segmentation.labels, count);
    segmentation.summary.merged_regions = count;
    segmentation.summary.paint_aware_region_merge_proposals += accepted_merges;
    segmentation.summary.paint_aware_region_merges += accepted_merges;
}

/// Compact a final exact-Paint ownership partition without changing the
/// canonical raster that positioned its external shared boundaries.
///
/// This is deliberately separate from `replace_merged_labels`: the latter is
/// an earlier perceptual merge and must recompute region prototypes, whereas
/// this final pass only removes interfaces between already identical Paints.
pub(crate) fn replace_final_exact_paint_labels(
    image: &Raster,
    segmentation: &mut Segmentation,
    labels: Vec<u32>,
    accepted_merges: usize,
) {
    assert_eq!(labels.len(), segmentation.labels.len());
    let (labels, count) = compact_values(&labels);
    segmentation.labels = labels;
    segmentation.paint_keys = (0..count as u32).collect();
    segmentation.regions = region_stats(image, &segmentation.labels, count);
    segmentation.summary.merged_regions = count;
    segmentation.summary.exact_paint_region_merges += accepted_merges;
}

/// Split only long Paint faces that participate in a source-smooth false
/// tone seam.  This is the Rust counterpart of
/// `_adaptive_paint_patch_partition`: topology is rebuilt afterward, so the
/// new interfaces are ordinary shared boundaries rather than overlay seams.
pub fn split_adaptive_paint_patches(
    image: &Raster,
    boundary_image: &Raster,
    segmentation: &mut Segmentation,
) {
    let count = segmentation.regions.len();
    if count == 0 || image.width != segmentation.width || image.height != segmentation.height {
        return;
    }
    let image_lab = lab_pixels(image);
    let boundary_lab = lab_pixels(boundary_image);
    let mut pixels = vec![Vec::<usize>::new(); count];
    for (index, &label) in segmentation.labels.iter().enumerate() {
        pixels[label as usize].push(index);
    }
    let median_labs: Vec<Lab> = pixels
        .iter()
        .map(|indices| {
            let mut selected: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&index| segmentation.paint_samples[index])
                .collect();
            if selected.is_empty() {
                selected.extend(indices.iter().copied());
            }
            if selected.len() > 1_024 {
                let last = (selected.len() - 1) as f64;
                let sampled = (0..1_024)
                    .map(|sample| {
                        let position = (last * sample as f64 / 1_023.0).floor() as usize;
                        selected[position]
                    })
                    .collect();
                selected = sampled;
            }
            let channel_median = |channel: usize| {
                let mut values: Vec<f32> = selected
                    .iter()
                    .map(|&index| match channel {
                        0 => image_lab[index].l,
                        1 => image_lab[index].a,
                        _ => image_lab[index].b,
                    })
                    .collect();
                values.sort_by(f32::total_cmp);
                let middle = values.len() / 2;
                if values.is_empty() {
                    0.0
                } else if values.len() % 2 == 0 {
                    (values[middle - 1] + values[middle]) * 0.5
                } else {
                    values[middle]
                }
            };
            Lab {
                l: channel_median(0),
                a: channel_median(1),
                b: channel_median(2),
            }
        })
        .collect();
    // `_smooth_paint_boundaries` first appends all horizontal interfaces,
    // then all vertical interfaces, and stable-sorts them by label pair.
    // Keep the boundary cell as well as its source error: one pair can meet
    // in distant places and Python classifies each eight-connected run
    // independently rather than pooling them into one percentile.
    let mut boundaries = BTreeMap::<(usize, usize), Vec<(usize, f32)>>::new();
    for y in 0..segmentation.height {
        for x in 0..segmentation.width.saturating_sub(1) {
            let index = y * segmentation.width + x;
            let neighbour = index + 1;
            let first = segmentation.labels[index] as usize;
            let second = segmentation.labels[neighbour] as usize;
            if first != second {
                let key = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                boundaries.entry(key).or_default().push((
                    index,
                    delta_e2000(boundary_lab[index], boundary_lab[neighbour]),
                ));
            }
        }
    }
    for y in 0..segmentation.height.saturating_sub(1) {
        for x in 0..segmentation.width {
            let index = y * segmentation.width + x;
            let neighbour = index + segmentation.width;
            let first = segmentation.labels[index] as usize;
            let second = segmentation.labels[neighbour] as usize;
            if first != second {
                let key = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                boundaries.entry(key).or_default().push((
                    index,
                    delta_e2000(boundary_lab[index], boundary_lab[neighbour]),
                ));
            }
        }
    }
    let mut candidates = HashSet::<usize>::new();
    let mut candidate_runs = 0_usize;
    let mut adaptive_run_diagnostics = Vec::new();
    for ((first, second), samples) in boundaries {
        let mut by_cell = BTreeMap::<usize, Vec<f32>>::new();
        for (cell, error) in samples {
            by_cell.entry(cell).or_default().push(error);
        }
        let cells: HashSet<usize> = by_cell.keys().copied().collect();
        let mut seen = HashSet::<usize>::new();
        for &start in by_cell.keys() {
            if !seen.insert(start) {
                continue;
            }
            let mut queue = VecDeque::from([start]);
            let mut deltas = Vec::<f32>::new();
            while let Some(cell) = queue.pop_front() {
                deltas.extend(by_cell[&cell].iter().copied());
                let x = cell % segmentation.width;
                let y = cell / segmentation.width;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx < 0
                            || ny < 0
                            || nx >= segmentation.width as isize
                            || ny >= segmentation.height as isize
                        {
                            continue;
                        }
                        let neighbour = ny as usize * segmentation.width + nx as usize;
                        if cells.contains(&neighbour) && seen.insert(neighbour) {
                            queue.push_back(neighbour);
                        }
                    }
                }
            }
            if deltas.len() < 8 {
                continue;
            }
            deltas.sort_by(f32::total_cmp);
            let middle = deltas.len() / 2;
            let median_delta = if deltas.len() % 2 == 0 {
                (deltas[middle - 1] + deltas[middle]) * 0.5
            } else {
                deltas[middle]
            };
            let p90 = percentile_f64(&deltas, 0.90);
            if median_delta <= 1.5 && p90 <= 3.0 {
                let separation = delta_e2000(median_labs[first], median_labs[second]);
                adaptive_run_diagnostics.push(serde_json::json!({
                    "left": first,
                    "right": second,
                    "length": deltas.len(),
                    "median": median_delta,
                    "p90": p90,
                    "separation": separation,
                    "left_lab": [median_labs[first].l, median_labs[first].a, median_labs[first].b],
                    "right_lab": [median_labs[second].l, median_labs[second].a, median_labs[second].b],
                }));
                if separation >= 0.75 {
                    candidates.insert(first);
                    candidates.insert(second);
                    candidate_runs += 1;
                }
            }
        }
    }
    let mut border_counts = vec![0_usize; count];
    for x in 0..segmentation.width {
        border_counts[segmentation.labels[x] as usize] += 1;
        border_counts
            [segmentation.labels[(segmentation.height - 1) * segmentation.width + x] as usize] += 1;
    }
    for y in 1..segmentation.height.saturating_sub(1) {
        border_counts[segmentation.labels[y * segmentation.width] as usize] += 1;
        border_counts
            [segmentation.labels[y * segmentation.width + segmentation.width - 1] as usize] += 1;
    }
    if let Some(background) = border_counts
        .iter()
        .enumerate()
        .fold(None, |best, (label, &area)| match best {
            Some((_, best_area)) if best_area >= area => best,
            _ => Some((label, area)),
        })
        .map(|(label, _)| label)
    {
        candidates.remove(&background);
    }
    if let Ok(path) = std::env::var("PICVEC_ADAPTIVE_DIAGNOSTICS") {
        let mut candidate_labels: Vec<usize> = candidates.iter().copied().collect();
        candidate_labels.sort_unstable();
        let value = serde_json::json!({
            "runs": adaptive_run_diagnostics,
            "candidate_runs": candidate_runs,
            "candidates": candidate_labels,
        });
        let _ = std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap_or_default());
    }
    let candidate_count = candidates.len();
    let patch_span = 128_usize.max(
        (0.20 * segmentation.width.max(segmentation.height) as f64).round_ties_even() as usize,
    );
    let source_paint_keys = if segmentation.paint_keys.len() == count {
        segmentation.paint_keys.clone()
    } else {
        (0..count as u32).collect()
    };
    let mut output = vec![u32::MAX; segmentation.labels.len()];
    let mut output_paint_keys = Vec::<u32>::new();
    let mut next_label = 0_u32;
    let mut split_faces = 0_usize;
    let mut added_regions = 0_usize;
    for (label, indices) in pixels.iter().enumerate() {
        let mut pieces = Vec::<Vec<usize>>::new();
        if candidates.contains(&label) && indices.len() >= 256 {
            let minimum_x = indices
                .iter()
                .map(|index| index % segmentation.width)
                .min()
                .unwrap();
            let maximum_x = indices
                .iter()
                .map(|index| index % segmentation.width)
                .max()
                .unwrap();
            let minimum_y = indices
                .iter()
                .map(|index| index / segmentation.width)
                .min()
                .unwrap();
            let maximum_y = indices
                .iter()
                .map(|index| index / segmentation.width)
                .max()
                .unwrap();
            let x_span = maximum_x - minimum_x + 1;
            let y_span = maximum_y - minimum_y + 1;
            let split_x = x_span >= y_span;
            let span = x_span.max(y_span);
            let part_count = span.div_ceil(patch_span).clamp(1, 8);
            if part_count > 1 {
                let mut axis: Vec<usize> = indices
                    .iter()
                    .map(|index| {
                        if split_x {
                            index % segmentation.width
                        } else {
                            index / segmentation.width
                        }
                    })
                    .collect();
                axis.sort_unstable();
                let mut thresholds = Vec::<usize>::new();
                for part in 1..part_count {
                    let position = ((axis.len() - 1) as f64 * part as f64 / part_count as f64)
                        .round_ties_even() as usize;
                    let value = axis[position];
                    if thresholds.last().copied() != Some(value) {
                        thresholds.push(value);
                    }
                }
                let mut bins = vec![Vec::<usize>::new(); thresholds.len() + 1];
                for &index in indices {
                    let coordinate = if split_x {
                        index % segmentation.width
                    } else {
                        index / segmentation.width
                    };
                    let bin = thresholds.partition_point(|&threshold| threshold <= coordinate);
                    bins[bin].push(index);
                }
                for bin in bins {
                    let bin_mask: HashSet<usize> = bin.iter().copied().collect();
                    let mut seen = HashSet::<usize>::new();
                    for start in bin {
                        if !seen.insert(start) {
                            continue;
                        }
                        let mut queue = VecDeque::from([start]);
                        let mut component = Vec::new();
                        while let Some(index) = queue.pop_front() {
                            component.push(index);
                            let x = index % segmentation.width;
                            let y = index / segmentation.width;
                            for neighbour in [
                                (x > 0).then_some(index - 1),
                                (x + 1 < segmentation.width).then_some(index + 1),
                                (y > 0).then_some(index - segmentation.width),
                                (y + 1 < segmentation.height).then_some(index + segmentation.width),
                            ]
                            .into_iter()
                            .flatten()
                            {
                                if bin_mask.contains(&neighbour) && seen.insert(neighbour) {
                                    queue.push_back(neighbour);
                                }
                            }
                        }
                        pieces.push(component);
                    }
                }
                if pieces.iter().map(Vec::len).min().unwrap_or(0) < 32 {
                    pieces.clear();
                }
            }
        }
        if pieces.len() <= 1 {
            pieces.clear();
            pieces.push(indices.clone());
        } else {
            split_faces += 1;
            added_regions += pieces.len() - 1;
        }
        for piece in pieces {
            for index in piece {
                output[index] = next_label;
            }
            output_paint_keys.push(source_paint_keys[label]);
            next_label += 1;
        }
    }
    if output.contains(&u32::MAX) || added_regions == 0 {
        segmentation.summary.adaptive_patch_candidate_faces = candidate_count;
        return;
    }
    segmentation.labels = output;
    segmentation.paint_keys = output_paint_keys;
    segmentation.regions = region_stats(image, &segmentation.labels, next_label as usize);
    segmentation.summary.merged_regions = next_label as usize;
    segmentation.summary.adaptive_patch_candidate_faces = candidate_count;
    segmentation.summary.adaptive_patch_split_faces = split_faces;
    segmentation.summary.adaptive_patch_added_regions = added_regions;
    let _ = candidate_runs;
}

fn percentile_f64(sorted: &[f32], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        sorted[low] as f64
    } else {
        let amount = position - low as f64;
        sorted[low] as f64 * (1.0 - amount) + sorted[high] as f64 * amount
    }
}

/// Remove palette-only boundary staircases while preserving measured source
/// edges and structural ownership.  Every move is synchronous, changes a
/// pixel only to an existing four-neighbour owner, and requires support on
/// both sides so topology cannot grow or detach a one-pixel island.
pub fn regularize_boundaries(
    image: &Raster,
    edge_reference: &Raster,
    segmentation: &mut Segmentation,
    roles: &EdgeRoles,
    structural_mask: &[bool],
    config: &Config,
) {
    assert_eq!(
        (image.width, image.height),
        (edge_reference.width, edge_reference.height)
    );
    assert_eq!(image.pixels.len(), segmentation.labels.len());
    let width = image.width;
    let height = image.height;
    let reference_lab = lab_pixels(edge_reference);
    let reference_at = |x: isize, y: isize| {
        let px = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let py = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        reference_lab[py * width + px]
    };
    let mut magnitude = vec![0.0_f32; width * height];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut energy = 0.0_f32;
            for channel in 0..3 {
                let value = |dx: isize, dy: isize| {
                    let sample = reference_at(x + dx, y + dy);
                    match channel {
                        0 => sample.l,
                        1 => sample.a,
                        _ => sample.b,
                    }
                };
                let gx = -value(-1, -1) + value(1, -1) - 2.0 * value(-1, 0) + 2.0 * value(1, 0)
                    - value(-1, 1)
                    + value(1, 1);
                let gy = -value(-1, -1) - 2.0 * value(0, -1) - value(1, -1)
                    + value(-1, 1)
                    + 2.0 * value(0, 1)
                    + value(1, 1);
                energy += gx * gx + gy * gy;
            }
            magnitude[y as usize * width + x as usize] = energy.sqrt();
        }
    }
    let mut local_peak = vec![0.0_f32; magnitude.len()];
    let mut local_floor = vec![0.0_f32; magnitude.len()];
    for index in 0..magnitude.len() {
        let x = index % width;
        let y = index / width;
        let mut maximum = f32::NEG_INFINITY;
        let mut minimum = f32::INFINITY;
        for dy in -2_isize..=2 {
            for dx in -2_isize..=2 {
                let px = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                let py = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                let value = magnitude[py * width + px];
                maximum = maximum.max(value);
                minimum = minimum.min(value);
            }
        }
        local_peak[index] = maximum;
        local_floor[index] = minimum;
    }
    let prominence: Vec<f32> = magnitude
        .iter()
        .zip(&local_floor)
        .map(|(&value, &floor)| (value - floor).max(0.0))
        .collect();
    let noise_floor = percentile(
        magnitude
            .iter()
            .copied()
            .filter(|value| *value > 1e-4)
            .collect(),
        0.65,
    );
    let prominence_floor = percentile(
        prominence
            .iter()
            .copied()
            .filter(|value| *value > 1e-4)
            .collect(),
        0.85,
    );
    let source_edge: Vec<bool> = (0..magnitude.len())
        .map(|index| {
            magnitude[index] >= noise_floor.max(1e-4)
                && magnitude[index] >= 0.85 * local_peak[index]
                && prominence[index] >= prominence_floor.max(0.5)
        })
        .collect();
    let supported = crate::edge::dilate_square(&source_edge, width, height, 1);
    let supported_buffer = crate::edge::dilate_square(&supported, width, height, 1);
    let face_barrier = crate::edge::dilate_square(&roles.face_barrier, width, height, 1);
    assert_eq!(structural_mask.len(), width * height);
    let frozen: Vec<bool> = structural_mask
        .iter()
        .zip(&face_barrier)
        .map(|(&line, &barrier)| line || barrier)
        .collect();

    let lab = lab_pixels(image);
    let count = segmentation.regions.len();
    let mut areas = vec![0_usize; count];
    let mut sums = vec![[0.0_f64; 3]; count];
    for (&label, sample) in segmentation.labels.iter().zip(&lab) {
        let owner = label as usize;
        areas[owner] += 1;
        sums[owner][0] += sample.l as f64;
        sums[owner][1] += sample.a as f64;
        sums[owner][2] += sample.b as f64;
    }
    let means: Vec<Lab> = (0..count)
        .map(|owner| {
            let area = areas[owner].max(1) as f64;
            Lab {
                l: (sums[owner][0] / area) as f32,
                a: (sums[owner][1] / area) as f32,
                b: (sums[owner][2] / area) as f32,
            }
        })
        .collect();
    let mut current = segmentation.labels.clone();
    let mut moved = 0_usize;
    let mut passes = 0_usize;
    for _ in 0..1 {
        let mut boundary = vec![false; current.len()];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if x + 1 < width && current[index] != current[index + 1] {
                    boundary[index] = true;
                    boundary[index + 1] = true;
                }
                if y + 1 < height && current[index] != current[index + width] {
                    boundary[index] = true;
                    boundary[index + width] = true;
                }
            }
        }
        let current_data: Vec<f32> = current
            .iter()
            .enumerate()
            .map(|(index, &owner)| delta_e2000(lab[index], means[owner as usize]))
            .collect();
        let mut output = current.clone();
        for index in 0..current.len() {
            if !boundary[index] || frozen[index] || supported_buffer[index] {
                continue;
            }
            let x = index % width;
            let y = index / width;
            let neighbour_at = |dx: isize, dy: isize| {
                let px = (x as isize + dx).clamp(0, width.saturating_sub(1) as isize) as usize;
                let py = (y as isize + dy).clamp(0, height.saturating_sub(1) as isize) as usize;
                current[py * width + px]
            };
            let neighbours = [
                neighbour_at(0, -1),
                neighbour_at(0, 1),
                neighbour_at(-1, 0),
                neighbour_at(1, 0),
            ];
            let owner = current[index];
            let current_disagreement =
                neighbours.iter().filter(|&&value| value != owner).count() as f32;
            let mut best_owner = owner;
            let mut best_energy = current_data[index] + 0.55 * current_disagreement;
            let budget = (0.45 * adaptive_tolerance(lab[index].l, config)).max(0.35);
            for &candidate in &neighbours {
                if candidate == owner {
                    continue;
                }
                let current_support = neighbours.iter().filter(|&&value| value == owner).count();
                let candidate_support = neighbours
                    .iter()
                    .filter(|&&value| value == candidate)
                    .count();
                if current_support < 2 || candidate_support < 2 {
                    continue;
                }
                let data = delta_e2000(lab[index], means[candidate as usize]);
                if data > current_data[index] + budget {
                    continue;
                }
                let disagreement = neighbours
                    .iter()
                    .filter(|&&value| value != candidate)
                    .count() as f32;
                let energy = data + 0.55 * disagreement;
                if energy < best_energy - 0.05 {
                    best_owner = candidate;
                    best_energy = energy;
                }
            }
            output[index] = best_owner;
        }
        let changed = output
            .iter()
            .zip(&current)
            .filter(|(first, second)| first != second)
            .count();
        if changed == 0 {
            break;
        }
        current = output;
        moved += changed;
        passes += 1;
    }
    // NumPy ``unique(..., return_inverse=True)`` compacts owner values but
    // deliberately does not split a temporarily disconnected owner. The SVG
    // topology stage can emit its multiple loops later under one Paint.
    let (labels, new_count) = compact_values(&current);
    let canonical = region_mean_raster_for(&segmentation.canonical, &labels, new_count);
    segmentation.labels = labels;
    segmentation.canonical = canonical;
    segmentation.regions = region_stats(image, &segmentation.labels, new_count);
    segmentation.summary.merged_regions = new_count;
    segmentation.summary.regularized_regions = new_count;
    segmentation.summary.boundary_regularization_passes = passes;
    segmentation.summary.boundary_regularization_moved_pixels = moved;
    segmentation.summary.boundary_regularization_source_edges =
        source_edge.iter().filter(|&&value| value).count();
}

/// Return unsupported dark one-pixel Paint shoulders to a neighbouring face.
/// Genuine line endpoints are protected by a one-sided rollback, and pixels
/// already transferred to the structural centre-line owner are never changed.
pub fn refine_thin_paint_ownership(
    source: &Raster,
    segmentation: &mut Segmentation,
    protected: &[bool],
) {
    assert_eq!(protected.len(), segmentation.labels.len());
    let original = segmentation.labels.clone();
    let count = segmentation.regions.len();
    let pixels = component_pixels(&original, count);
    let source_lab = lab_pixels(source);
    let mut prototype_rgb = vec![[0.0_f32; 3]; count];
    let mut prototype_seen = vec![false; count];
    for (index, &label) in original.iter().enumerate() {
        let owner = label as usize;
        if !prototype_seen[owner] {
            prototype_rgb[owner] = segmentation.canonical.pixels[index];
            prototype_seen[owner] = true;
        }
    }
    let prototypes: Vec<Lab> = prototype_rgb.iter().copied().map(rgb_to_lab).collect();
    let has_interior: Vec<bool> = (0..count)
        .map(|label| {
            has_core(
                &original,
                segmentation.width,
                segmentation.height,
                label as u32,
                &pixels[label],
            )
        })
        .collect();
    let mut output = original.clone();
    let mut examined = 0;
    let mut protected_components = 0;
    let mut refined = 0;
    let mut rollbacks = 0;
    let mut reassigned = 0;
    let mut visited = vec![false; original.len()];
    for label in 0..count {
        if prototypes[label].l > 25.0 || pixels[label].is_empty() {
            continue;
        }
        for &start in &pixels[label] {
            if visited[start] {
                continue;
            }
            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            visited[start] = true;
            while let Some(index) = queue.pop_front() {
                component.push(index);
                let x = index % segmentation.width;
                let y = index / segmentation.width;
                for neighbour in [
                    (x > 0).then(|| index - 1),
                    (x + 1 < segmentation.width).then(|| index + 1),
                    (y > 0).then(|| index - segmentation.width),
                    (y + 1 < segmentation.height).then(|| index + segmentation.width),
                ]
                .into_iter()
                .flatten()
                {
                    if !visited[neighbour] && original[neighbour] == label as u32 {
                        visited[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
            if component.iter().any(|&index| {
                let x = index % segmentation.width;
                let y = index / segmentation.width;
                x > 0
                    && y > 0
                    && x + 1 < segmentation.width
                    && y + 1 < segmentation.height
                    && original[index - 1] == label as u32
                    && original[index + 1] == label as u32
                    && original[index - segmentation.width] == label as u32
                    && original[index + segmentation.width] == label as u32
            }) {
                continue;
            }
            examined += 1;
            if component.iter().any(|&index| protected[index]) {
                protected_components += 1;
                continue;
            }
            let before: Vec<u32> = component.iter().map(|&index| output[index]).collect();
            let mut changes = 0;
            loop {
                let mut selected = Vec::<(usize, u32)>::new();
                for &index in &component {
                    if output[index] != label as u32 {
                        continue;
                    }
                    let x = index % segmentation.width;
                    let y = index / segmentation.width;
                    let mut best: Option<(u32, f32)> = None;
                    let neighbours = [
                        (y > 0).then(|| output[index - segmentation.width]),
                        (y + 1 < segmentation.height).then(|| output[index + segmentation.width]),
                        (x > 0).then(|| output[index - 1]),
                        (x + 1 < segmentation.width).then(|| output[index + 1]),
                    ];
                    for owner in neighbours.into_iter().flatten() {
                        if owner == label as u32 || !has_interior[owner as usize] {
                            continue;
                        }
                        let error = delta_e2000(source_lab[index], prototypes[owner as usize]);
                        // Python visits up, down, left, right and updates only
                        // on strict improvement; retaining the earlier owner
                        // is its deterministic tie rule.
                        if best.map(|value| error < value.1).unwrap_or(true) {
                            best = Some((owner, error));
                        }
                    }
                    if let Some((owner, error)) = best {
                        let current = delta_e2000(source_lab[index], prototypes[label]);
                        if error + 1e-4 < current {
                            selected.push((index, owner));
                        }
                    }
                }
                if selected.is_empty() {
                    break;
                }
                changes += selected.len();
                for (index, owner) in selected {
                    output[index] = owner;
                }
            }
            if changes == 0 {
                continue;
            }
            let retained = component.iter().any(|&index| output[index] == label as u32);
            let changed_owners: HashSet<u32> = component
                .iter()
                .filter_map(|&index| (output[index] != label as u32).then_some(output[index]))
                .collect();
            if retained && changed_owners.len() < 2 {
                for (&index, owner) in component.iter().zip(before) {
                    output[index] = owner;
                }
                rollbacks += 1;
            } else {
                reassigned += changes;
                refined += 1;
            }
        }
    }
    let canonical_pixels = output
        .iter()
        .map(|&label| prototype_rgb[label as usize])
        .collect();
    let (labels, new_count) = compact_values(&output);
    segmentation.labels = labels;
    segmentation.canonical = Raster::new(segmentation.width, segmentation.height, canonical_pixels);
    segmentation.regions = region_stats(source, &segmentation.labels, new_count);
    segmentation.summary.merged_regions = new_count;
    segmentation.summary.thin_paint_examined = examined;
    segmentation.summary.thin_paint_protected = protected_components;
    segmentation.summary.thin_paint_refined = refined;
    segmentation.summary.thin_paint_rollbacks = rollbacks;
    segmentation.summary.thin_paint_reassigned_pixels = reassigned;
    segmentation.summary.thin_paint_removed_regions = count.saturating_sub(new_count);
}

pub fn region_mean_raster(image: &Raster, segmentation: &Segmentation) -> Raster {
    let pixels = segmentation
        .labels
        .iter()
        .map(|&label| segmentation.regions[label as usize].mean_rgb)
        .collect();
    Raster::new(image.width, image.height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::classify;

    #[test]
    fn boundary_keeps_two_faces_separate() {
        let mut image = Raster::blank(24, 12, [0.1, 0.1, 0.1]);
        for y in 0..12 {
            for x in 12..24 {
                image.pixels[y * 24 + x] = [0.9, 0.1, 0.1];
            }
        }
        let roles = classify(&image);
        let result = segment(
            &image,
            &roles,
            &Config {
                segmentation_min_size: 2,
                ..Config::default()
            },
        );
        assert_ne!(result.labels[5 * 24 + 5], result.labels[5 * 24 + 18]);
    }

    #[test]
    fn spatially_separate_equal_colours_remain_separate_regions() {
        let mut image = Raster::blank(16, 8, [1.0, 1.0, 1.0]);
        image.pixels[2 * 16 + 2] = [0.0, 0.0, 0.0];
        image.pixels[2 * 16 + 13] = [0.0, 0.0, 0.0];
        let roles = classify(&image);
        let result = segment(
            &image,
            &roles,
            &Config {
                segmentation_min_size: 1,
                ..Config::default()
            },
        );
        assert_ne!(result.labels[2 * 16 + 2], result.labels[2 * 16 + 13]);
    }

    #[test]
    fn antialias_band_is_returned_to_its_two_parent_faces() {
        let width = 10;
        let height = 7;
        let mut image = Raster::blank(width, height, [0.0, 0.0, 0.0]);
        let mut labels = vec![0_u32; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                match x {
                    0..=2 => {}
                    3 => {
                        image.pixels[index] = [0.25; 3];
                        labels[index] = 1;
                    }
                    4 => {
                        image.pixels[index] = [0.75; 3];
                        labels[index] = 1;
                    }
                    _ => {
                        image.pixels[index] = [1.0; 3];
                        labels[index] = 2;
                    }
                }
            }
        }
        let correction = correct_antialias_partition(&image, &labels, 3);
        assert_eq!(correction.split_regions, 1);
        assert_ne!(correction.labels[3], correction.labels[4]);
        assert!(correction.paint_samples[3..=4].iter().all(|&value| !value));
    }
}
