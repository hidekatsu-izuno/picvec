use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::color::rgb_hex;
use crate::geometry::{open_path_data, Primitive, RegionGeometry};
use crate::gradient::{ColorStop, OpacityStop, Paint, PaintOverlay};
use crate::optimize::{format_number, optimize_path, separated_bboxes, OptimizedElement, PathOptimization};
use crate::structural::StructuralInk;
use crate::Result;

#[derive(Clone, Debug, Default, Serialize)]
pub struct SvgSummary {
    pub path_elements: usize,
    pub rect_elements: usize,
    pub circle_elements: usize,
    pub ellipse_elements: usize,
    pub line_elements: usize,
    pub linear_gradients: usize,
    pub radial_gradients: usize,
    pub structural_strokes: usize,
    pub structural_color_patches: usize,
    pub gradient_stops: usize,
    pub linear_cubics_to_lines: usize,
    pub redundant_segments_removed: usize,
    pub arc_segments: usize,
    pub merged_arc_segments: usize,
    pub paint_paths_merged: usize,
    pub paint_batches: usize,
    pub alpha_mask_paths: usize,
    pub outline_bands: usize,
    pub outline_color_patches: usize,
    pub outline_paint_regions_removed: usize,
    pub outline_structural_strokes_removed: usize,
    pub bytes: usize,
}

impl SvgSummary {
    pub(crate) fn add_elements_from(&mut self, other: &Self) {
        self.path_elements += other.path_elements;
        self.rect_elements += other.rect_elements;
        self.circle_elements += other.circle_elements;
        self.ellipse_elements += other.ellipse_elements;
        self.line_elements += other.line_elements;
        self.linear_gradients += other.linear_gradients;
        self.radial_gradients += other.radial_gradients;
        self.structural_strokes += other.structural_strokes;
        self.structural_color_patches += other.structural_color_patches;
        self.gradient_stops += other.gradient_stops;
        self.linear_cubics_to_lines += other.linear_cubics_to_lines;
        self.redundant_segments_removed += other.redundant_segments_removed;
        self.arc_segments += other.arc_segments;
        self.merged_arc_segments += other.merged_arc_segments;
        self.paint_paths_merged += other.paint_paths_merged;
        self.paint_batches += other.paint_batches;
        self.alpha_mask_paths += other.alpha_mask_paths;
        self.outline_bands += other.outline_bands;
        self.outline_color_patches += other.outline_color_patches;
        self.outline_paint_regions_removed += other.outline_paint_regions_removed;
        self.outline_structural_strokes_removed += other.outline_structural_strokes_removed;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AlphaMaskLayer {
    pub path_data: String,
    /// Source-over opacity chosen so this nested layer raises accumulated
    /// coverage to its corresponding alpha level. An antialiased opaque
    /// silhouette is a single full-opacity layer; durable authored
    /// transparency may use nested two-bit layers.
    pub opacity: f32,
    /// Grayscale coverage in a luminance mask; absent for flat alpha layers.
    pub paint: Option<Paint>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AlphaMask {
    pub layers: Vec<AlphaMaskLayer>,
    pub luminance: bool,
}

#[derive(Clone, Debug)]
struct PaintElement {
    geometry: OptimizedElement,
    attributes: String,
    underpaint_attributes: Option<String>,
    batchable: bool,
}

#[derive(Clone, Debug)]
struct PaintBatch {
    target: usize,
    bboxes: Vec<(f64, f64, f64, f64)>,
    merged: bool,
}

fn number(value: f32) -> String {
    let value = (value * 1000.0).round() / 1000.0;
    if (value - value.round()).abs() < 1e-5 {
        format!("{:.0}", value)
    } else {
        format!("{:.3}", value).trim_end_matches('0').to_string()
    }
}

fn stops_key(stops: &[ColorStop]) -> String {
    stops
        .iter()
        .map(|stop| {
            format!(
                "{}:{}",
                number(stop.offset as f32),
                rgb_hex(stop.color.map(|value| value as f32))
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn paint_key(paint: &Paint) -> Option<String> {
    match paint {
        Paint::Solid { .. } => None,
        Paint::Linear {
            start, end, stops, ..
        } => Some(format!(
            "L:{},{},{},{}:{}",
            number(start.x),
            number(start.y),
            number(end.x),
            number(end.y),
            stops_key(stops)
        )),
        Paint::Radial {
            rotation,
            center,
            radius,
            stops,
            ..
        } => Some(format!(
            "R:{},{},{},{},{}:{}",
            number(center.x),
            number(center.y),
            number(radius.x),
            number(radius.y),
            number(rotation.to_degrees()),
            stops_key(stops)
        )),
        Paint::Layered { .. } => None,
    }
}

fn overlay_key(overlay: &PaintOverlay) -> String {
    let opacity = overlay
        .opacity_stops
        .iter()
        .map(|stop| {
            format!(
                "{}:{}",
                number(stop.offset as f32),
                number(stop.opacity as f32)
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "O:{}:{}",
        paint_key(&overlay.paint).unwrap_or_else(|| "nested".to_string()),
        opacity
    )
}

fn opacity_at(stops: &[OpacityStop], offset: f64) -> f64 {
    if stops.is_empty() {
        return 1.0;
    }
    if offset <= stops[0].offset {
        return stops[0].opacity.clamp(0.0, 1.0);
    }
    for pair in stops.windows(2) {
        if offset <= pair[1].offset {
            let amount = ((offset - pair[0].offset) / (pair[1].offset - pair[0].offset).max(1e-12))
                .clamp(0.0, 1.0);
            return (pair[0].opacity * (1.0 - amount) + pair[1].opacity * amount).clamp(0.0, 1.0);
        }
    }
    stops
        .last()
        .map(|stop| stop.opacity)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0)
}

fn color_at(stops: &[ColorStop], offset: f64) -> [f64; 3] {
    if stops.is_empty() {
        return [0.0; 3];
    }
    if offset <= stops[0].offset {
        return stops[0].color;
    }
    for pair in stops.windows(2) {
        if offset <= pair[1].offset {
            let amount = ((offset - pair[0].offset) / (pair[1].offset - pair[0].offset).max(1e-12))
                .clamp(0.0, 1.0);
            return [0, 1, 2].map(|channel| {
                pair[0].color[channel] * (1.0 - amount) + pair[1].color[channel] * amount
            });
        }
    }
    stops.last().map(|stop| stop.color).unwrap_or([0.0; 3])
}

fn stop_elements(stops: &[ColorStop]) -> String {
    let mut output = String::new();
    for stop in stops {
        let _ = write!(
            output,
            "<stop offset=\"{}\" stop-color=\"{}\"/>",
            number(stop.offset as f32),
            rgb_hex(stop.color.map(|value| value as f32))
        );
    }
    output
}

fn overlay_stop_elements(stops: &[ColorStop], opacity_stops: &[OpacityStop]) -> String {
    let mut offsets = stops.iter().map(|stop| stop.offset).collect::<Vec<_>>();
    offsets.extend(opacity_stops.iter().map(|stop| stop.offset));
    offsets.sort_by(f64::total_cmp);
    offsets.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
    let mut output = String::new();
    for offset in offsets {
        let _ = write!(
            output,
            "<stop offset=\"{}\" stop-color=\"{}\" stop-opacity=\"{}\"/>",
            number(offset as f32),
            rgb_hex(color_at(stops, offset).map(|value| value as f32)),
            number(opacity_at(opacity_stops, offset) as f32),
        );
    }
    output
}

fn fill_value(paint: &Paint, gradient_ids: &HashMap<String, String>) -> String {
    match paint {
        Paint::Solid { color } => rgb_hex(*color),
        Paint::Linear { .. } | Paint::Radial { .. } => {
            format!("url(#{})", gradient_ids[&paint_key(paint).unwrap()])
        }
        Paint::Layered { .. } => unreachable!("layered Paint is emitted component-wise"),
    }
}

fn underpaint_attributes(fill: &str, overlap: f32) -> Option<String> {
    (overlap > 0.0).then(|| {
        format!(
            "fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\"",
            fill,
            number(overlap * 2.0)
        )
    })
}

fn write_paint_elements(
    body: &mut String,
    elements: &[Option<PaintElement>],
    summary: &mut SvgSummary,
) {
    // Seal rasterizer seams below all fills. Stroking each face in paint
    // order displaces a shared boundary by +/- overlap whenever a shade
    // changes, even if all faces reuse the very same straight/curved master.
    for element in elements.iter().flatten() {
        if let Some(attributes) = &element.underpaint_attributes {
            count_element(summary, write_geometry(body, &element.geometry, attributes));
        }
    }
    for element in elements.iter().flatten() {
        count_element(
            summary,
            write_geometry(body, &element.geometry, &element.attributes),
        );
    }
}

fn paint_path_bbox(element: &PaintElement) -> Option<(f64, f64, f64, f64)> {
    match &element.geometry {
        OptimizedElement::Path { bbox, .. } => *bbox,
        _ => None,
    }
}

fn bbox_cells(bbox: (f64, f64, f64, f64)) -> Vec<(i64, i64)> {
    const CELL: f64 = 16.0;
    const PADDING: f64 = 1.0;
    let minimum_x = ((bbox.0 - PADDING) / CELL).floor() as i64;
    let minimum_y = ((bbox.1 - PADDING) / CELL).floor() as i64;
    let maximum_x = ((bbox.2 + PADDING) / CELL).floor() as i64;
    let maximum_y = ((bbox.3 + PADDING) / CELL).floor() as i64;
    let mut cells = Vec::new();
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            cells.push((x, y));
        }
    }
    cells
}

fn batch_equal_paint_paths(elements: &mut [Option<PaintElement>], summary: &mut SvgSummary) {
    let mut signature_ids = HashMap::<String, usize>::new();
    let mut latest_spatial = HashMap::<(i64, i64), Vec<(usize, usize)>>::new();
    let mut global_blockers = Vec::<(usize, usize)>::new();
    let mut batches = HashMap::<usize, Vec<PaintBatch>>::new();
    for current in 0..elements.len() {
        let Some(element) = elements[current].as_ref() else {
            continue;
        };
        let signature = format!("{}|{:?}", element.attributes, element.underpaint_attributes);
        let next_signature = signature_ids.len();
        let signature_id = *signature_ids.entry(signature).or_insert(next_signature);
        if !element.batchable {
            // Layer components must stay adjacent and in order. Treat them as
            // global ordering barriers rather than moving equal fills across
            // another face during path batching.
            global_blockers.push((current, usize::MAX - current));
            continue;
        }
        let Some(bbox) = paint_path_bbox(element) else {
            // A primitive or a path with overlapping subpaths may cover any
            // later candidate.  Keeping it as a global ordering barrier is
            // conservative and cannot change the rendered stack.
            if let Some(last) = global_blockers.last_mut() {
                if last.1 == signature_id {
                    last.0 = current;
                    continue;
                }
            }
            global_blockers.push((current, signature_id));
            continue;
        };
        let cells = bbox_cells(bbox);
        let blocker = cells
            .iter()
            .filter_map(|cell| {
                latest_spatial.get(cell).and_then(|entries| {
                    entries
                        .iter()
                        .rev()
                        .find_map(|&(index, owner)| (owner != signature_id).then_some(index))
                })
            })
            .chain(
                global_blockers
                    .iter()
                    .rev()
                    .find_map(|&(index, owner)| (owner != signature_id).then_some(index)),
            )
            .max();
        let batch_index = batches.get(&signature_id).and_then(|values| {
            values.iter().position(|batch| {
                blocker.is_none_or(|value| batch.target >= value)
                    && batch
                        .bboxes
                        .iter()
                        .all(|&previous| separated_bboxes(bbox, previous, 1.0))
            })
        });
        if let Some(batch_index) = batch_index {
            let target = batches[&signature_id][batch_index].target;
            let current_data = match &elements[current].as_ref().unwrap().geometry {
                OptimizedElement::Path { data, .. } => data.clone(),
                _ => unreachable!(),
            };
            if let Some(PaintElement {
                geometry:
                    OptimizedElement::Path {
                        data,
                        bbox: target_bbox,
                    },
                ..
            }) = elements[target].as_mut()
            {
                data.push(' ');
                data.push_str(current_data.trim());
                *target_bbox = Some(match *target_bbox {
                    Some(previous) => (
                        previous.0.min(bbox.0),
                        previous.1.min(bbox.1),
                        previous.2.max(bbox.2),
                        previous.3.max(bbox.3),
                    ),
                    None => bbox,
                });
            }
            elements[current] = None;
            let batch = &mut batches.get_mut(&signature_id).unwrap()[batch_index];
            if !batch.merged {
                batch.merged = true;
                summary.paint_batches += 1;
            }
            batch.bboxes.push(bbox);
            summary.paint_paths_merged += 1;
        } else {
            batches.entry(signature_id).or_default().push(PaintBatch {
                target: current,
                bboxes: vec![bbox],
                merged: false,
            });
        }
        // Retain the element's original position as a conservative blocker,
        // even after its geometry has moved into an earlier equal-Paint path.
        for cell in cells {
            let entries = latest_spatial.entry(cell).or_default();
            if let Some(last) = entries.last_mut() {
                if last.1 == signature_id {
                    last.0 = current;
                    continue;
                }
            }
            entries.push((current, signature_id));
        }
    }
}

fn write_geometry(
    body: &mut String,
    geometry: &OptimizedElement,
    attributes: &str,
) -> &'static str {
    match geometry {
        OptimizedElement::Path { data, .. } => {
            let _ = write!(body, "<path d=\"{}\" {}/>", data, attributes);
            "path"
        }
        OptimizedElement::Line { x1, y1, x2, y2 } => {
            let _ = write!(
                body,
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" {}/>",
                format_number(*x1),
                format_number(*y1),
                format_number(*x2),
                format_number(*y2),
                attributes
            );
            "line"
        }
        OptimizedElement::Rect {
            x,
            y,
            width,
            height,
        } => {
            let _ = write!(
                body,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" {}/>",
                format_number(*x),
                format_number(*y),
                format_number(*width),
                format_number(*height),
                attributes
            );
            "rect"
        }
        OptimizedElement::Circle { cx, cy, radius } => {
            let _ = write!(
                body,
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" {}/>",
                format_number(*cx),
                format_number(*cy),
                format_number(*radius),
                attributes
            );
            "circle"
        }
    }
}

fn count_element(summary: &mut SvgSummary, kind: &str) {
    match kind {
        "path" => summary.path_elements += 1,
        "line" => summary.line_elements += 1,
        "rect" => summary.rect_elements += 1,
        "circle" => summary.circle_elements += 1,
        _ => {}
    }
}

fn register_gradient(
    paint: &Paint,
    opacity_stops: Option<&[OpacityStop]>,
    key: String,
    gradient_ids: &mut HashMap<String, String>,
    definitions: &mut String,
    summary: &mut SvgSummary,
) {
    if gradient_ids.contains_key(&key) {
        return;
    }
    let id = format!("paint-{}", gradient_ids.len());
    match paint {
        Paint::Linear {
            start, end, stops, ..
        } => {
            let elements = opacity_stops
                .map(|opacity| overlay_stop_elements(stops, opacity))
                .unwrap_or_else(|| stop_elements(stops));
            let _ = write!(definitions, "<linearGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\">{}</linearGradient>", id, number(start.x), number(start.y), number(end.x), number(end.y), elements);
            summary.linear_gradients += 1;
            summary.gradient_stops += if let Some(opacity) = opacity_stops {
                let mut offsets = stops.iter().map(|stop| stop.offset).collect::<Vec<_>>();
                offsets.extend(opacity.iter().map(|stop| stop.offset));
                offsets.sort_by(f64::total_cmp);
                offsets.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
                offsets.len()
            } else {
                stops.len()
            };
        }
        Paint::Radial {
            rotation,
            center,
            radius,
            stops,
            ..
        } => {
            let elements = opacity_stops
                .map(|opacity| overlay_stop_elements(stops, opacity))
                .unwrap_or_else(|| stop_elements(stops));
            let rotation = if *rotation == 0.0 {
                String::new()
            } else {
                format!(" rotate({})", number(rotation.to_degrees()))
            };
            let _ = write!(definitions, "<radialGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" cx=\"0\" cy=\"0\" r=\"1\" gradientTransform=\"translate({} {}){} scale({} {})\">{}</radialGradient>", id, number(center.x), number(center.y), rotation, number(radius.x.max(0.001)), number(radius.y.max(0.001)), elements);
            summary.radial_gradients += 1;
            summary.gradient_stops += if let Some(opacity) = opacity_stops {
                let mut offsets = stops.iter().map(|stop| stop.offset).collect::<Vec<_>>();
                offsets.extend(opacity.iter().map(|stop| stop.offset));
                offsets.sort_by(f64::total_cmp);
                offsets.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
                offsets.len()
            } else {
                stops.len()
            };
        }
        Paint::Solid { .. } | Paint::Layered { .. } => return,
    }
    gradient_ids.insert(key, id);
}

fn append_paint_elements(
    elements: &mut Vec<Option<PaintElement>>,
    geometry: OptimizedElement,
    paint: &Paint,
    gradient_ids: &HashMap<String, String>,
    paint_overlap: f32,
) {
    match paint {
        Paint::Layered { base, overlays } => {
            let base_fill = fill_value(base, gradient_ids);
            elements.push(Some(PaintElement {
                geometry: geometry.clone(),
                attributes: format!("fill=\"{base_fill}\""),
                underpaint_attributes: underpaint_attributes(&base_fill, paint_overlap),
                batchable: false,
            }));
            for overlay in overlays {
                let fill = match overlay.paint.as_ref() {
                    Paint::Solid { color } => rgb_hex(*color),
                    Paint::Linear { .. } | Paint::Radial { .. } => {
                        format!("url(#{})", gradient_ids[&overlay_key(overlay)])
                    }
                    Paint::Layered { .. } => continue,
                };
                elements.push(Some(PaintElement {
                    geometry: geometry.clone(),
                    attributes: format!("fill=\"{}\"", fill),
                    underpaint_attributes: None,
                    batchable: false,
                }));
            }
        }
        _ => {
            let fill = fill_value(paint, gradient_ids);
            elements.push(Some(PaintElement {
                geometry,
                attributes: format!("fill=\"{fill}\""),
                underpaint_attributes: underpaint_attributes(&fill, paint_overlap),
                batchable: true,
            }));
        }
    }
}

/// Serialize the complete editable document.  Gradients are restricted to
/// solid, axial linear, and elliptical radial forms with at most five stops,
/// which stay within the Office object import subset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serialize(
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
) -> (String, SvgSummary) {
    serialize_filtered(
        width,
        height,
        geometries,
        paints,
        structural,
        paint_overlap,
        final_geometry,
        &[],
    )
}

/// Serialize while omitting selected raster regions.  Geometry is retained
/// for all regions until this final stage so holes remain part of their
/// surrounding foreground paths, including disconnected keyed areas.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serialize_filtered(
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
    excluded_regions: &[bool],
) -> (String, SvgSummary) {
    serialize_filtered_with_alpha(
        width,
        height,
        geometries,
        paints,
        structural,
        paint_overlap,
        final_geometry,
        excluded_regions,
        None,
    )
}

/// A conversion-local cache of exact Paint path normalization. The input
/// string is the key, so ordinary and occlusion paths cannot share stale data.
#[derive(Default)]
pub(crate) struct GeometryCache {
    paths: HashMap<String, Option<(OptimizedElement, PathOptimization)>>,
}

impl GeometryCache {
    fn optimize(&mut self, path: &str) -> Option<(OptimizedElement, PathOptimization)> {
        if let Some(cached) = self.paths.get(path) {
            return cached.clone();
        }
        let optimized = optimize_path(path, true, false);
        self.paths.insert(path.to_owned(), optimized.clone());
        optimized
    }
}

/// Serialize selected raster regions and optionally apply an independent
/// vector alpha mask to the complete Paint and structural content.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serialize_filtered_with_alpha(
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
    excluded_regions: &[bool],
    alpha_mask: Option<&AlphaMask>,
) -> (String, SvgSummary) {
    serialize_filtered_with_alpha_cached(
        width, height, geometries, paints, structural, paint_overlap,
        final_geometry, excluded_regions, alpha_mask, &mut GeometryCache::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn serialize_filtered_with_alpha_cached(
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
    excluded_regions: &[bool],
    alpha_mask: Option<&AlphaMask>,
    geometry_cache: &mut GeometryCache,
) -> (String, SvgSummary) {
    let mut gradient_ids = HashMap::<String, String>::new();
    let mut definitions = String::new();
    let mut summary = SvgSummary::default();
    if let Some(mask) = alpha_mask {
        for (i, layer) in mask.layers.iter().enumerate() {
            if let Some(paint) = &layer.paint {
                register_gradient(
                    paint,
                    None,
                    format!("mask-{i}"),
                    &mut gradient_ids,
                    &mut definitions,
                    &mut summary,
                );
            }
        }
        let mode = if mask.luminance { "luminance" } else { "alpha" };
        let _ = write!(
            definitions,
            "<mask id=\"source-alpha-mask\" maskUnits=\"userSpaceOnUse\" maskContentUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" mask-type=\"{mode}\" style=\"mask-type:{mode}\">",
            width, height
        );
        // Seal seams underneath every mask fill. A stroke painted with its
        // own fill can overwrite a neighbouring alpha value and create a
        // bright fringe where RGB and authored transparency compensate.
        for underpass in [true, false] {
            for (i, layer) in mask.layers.iter().enumerate() {
                if layer.path_data.is_empty() || layer.opacity <= 0.0 {
                    continue;
                }
                let fill = match &layer.paint {
                    Some(Paint::Solid { color }) => rgb_hex(*color),
                    Some(_) => format!("url(#{})", gradient_ids[&format!("mask-{i}")]),
                    None => "#ffffff".to_string(),
                };
                let opacity = number(layer.opacity.clamp(0.0, 1.0));
                if underpass {
                    if !mask.luminance
                        || matches!(&layer.paint, Some(Paint::Solid {color}) if *color == [1.0;3])
                    {
                        continue;
                    }
                    let _ = write!(
                        definitions,
                        "<path d=\"{}\" fill=\"none\" stroke=\"{fill}\" stroke-width=\"1.5\" stroke-opacity=\"{opacity}\"/>",
                        layer.path_data,
                    );
                } else {
                    let _ = write!(
                        definitions,
                        "<path d=\"{}\" fill=\"{fill}\" fill-opacity=\"{opacity}\" fill-rule=\"evenodd\"/>",
                        layer.path_data,
                    );
                }
                summary.alpha_mask_paths += 1;
            }
        }
        definitions.push_str("</mask>");
    }
    for (region, paint) in paints.iter().enumerate() {
        if excluded_regions.get(region).copied().unwrap_or(false)
            || structural.outlines.iter().any(|band| {
                band.hidden.contains(&(region as u32))
                    && !band.boundary_underpaint.contains(&(region as u32))
            })
        {
            continue;
        }
        match paint {
            Paint::Layered { base, overlays } => {
                if let Some(key) = paint_key(base) {
                    register_gradient(
                        base,
                        None,
                        key,
                        &mut gradient_ids,
                        &mut definitions,
                        &mut summary,
                    );
                }
                for overlay in overlays {
                    register_gradient(
                        &overlay.paint,
                        Some(&overlay.opacity_stops),
                        overlay_key(overlay),
                        &mut gradient_ids,
                        &mut definitions,
                        &mut summary,
                    );
                }
            }
            Paint::Linear { .. } | Paint::Radial { .. } => register_gradient(
                paint,
                None,
                paint_key(paint).unwrap(),
                &mut gradient_ids,
                &mut definitions,
                &mut summary,
            ),
            Paint::Solid { .. } => {}
        }
    }
    summary.outline_bands = structural.outlines.len();
    for (i, band) in structural.outlines.iter().enumerate() {
        summary.outline_color_patches += band.patches.len();
        for paint in band
            .patches
            .iter()
            .map(|(_, paint)| paint)
            .chain(std::iter::once(&band.underpaint))
            .chain(band.inner_underpaint.iter().map(|(_, paint)| paint))
        {
            if let Some(key) = paint_key(paint) {
                register_gradient(
                    paint,
                    None,
                    key,
                    &mut gradient_ids,
                    &mut definitions,
                    &mut summary,
                );
            }
        }
        let _ = write!(definitions, "<clipPath id=\"outline-inner-{i}\"><path d=\"{}\"/></clipPath><clipPath id=\"outline-outer-{i}\"><path d=\"{}\"/></clipPath>", band.inner, band.outer);
    }
    let mut paint_elements = Vec::<Option<PaintElement>>::with_capacity(geometries.len());
    let mut band_elements: Vec<Vec<Option<PaintElement>>> =
        structural.outlines.iter().map(|_| Vec::new()).collect();
    for geometry in geometries {
        if excluded_regions
            .get(geometry.region as usize)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let band = structural
            .outlines
            .iter()
            .enumerate()
            .find(|(_, band)| band.regions.contains(&geometry.region));
        if band.is_some_and(|(_, band)| {
            band.hidden.contains(&geometry.region)
                && !band.boundary_underpaint.contains(&geometry.region)
        }) {
            summary.outline_paint_regions_removed += 1;
            continue;
        }
        let paint = &paints[geometry.region as usize];
        let optimized = match &geometry.primitive {
            Some(Primitive::Rect {
                x,
                y,
                width,
                height,
            }) => OptimizedElement::Rect {
                x: *x as f64,
                y: *y as f64,
                width: *width as f64,
                height: *height as f64,
            },
            Some(Primitive::Circle { cx, cy, radius }) => OptimizedElement::Circle {
                cx: *cx as f64,
                cy: *cy as f64,
                radius: *radius as f64,
            },
            Some(Primitive::Ellipse { cx, cy, rx, ry }) => {
                summary.ellipse_elements += 1;
                OptimizedElement::Path {
                    data: format!(
                        "M {} {} A {} {} 0 1 0 {} {} A {} {} 0 1 0 {} {} Z",
                        format_number((*cx + *rx) as f64),
                        format_number(*cy as f64),
                        format_number(*rx as f64),
                        format_number(*ry as f64),
                        format_number((*cx - *rx) as f64),
                        format_number(*cy as f64),
                        format_number(*rx as f64),
                        format_number(*ry as f64),
                        format_number((*cx + *rx) as f64),
                        format_number(*cy as f64),
                    ),
                    bbox: Some((
                        (*cx - *rx) as f64,
                        (*cy - *ry) as f64,
                        (*cx + *rx) as f64,
                        (*cy + *ry) as f64,
                    )),
                }
            }
            None => {
                let path_data = if final_geometry {
                    geometry
                        .occlusion_path_data
                        .as_deref()
                        .unwrap_or(&geometry.path_data)
                } else {
                    &geometry.path_data
                };
                if path_data.is_empty() {
                    continue;
                }
                if let Some((optimized, operations)) = geometry_cache.optimize(path_data) {
                    summary.linear_cubics_to_lines += operations.linear_cubics;
                    summary.redundant_segments_removed += operations.redundant_segments;
                    summary.arc_segments += operations.arc_segments;
                    summary.merged_arc_segments += operations.merged_arcs;
                    optimized
                } else {
                    OptimizedElement::Path {
                        data: path_data.to_string(),
                        bbox: None,
                    }
                }
            }
        };
        if let Some((_, band)) = band {
            if band.boundary_underpaint.contains(&geometry.region) {
                // Shared geometry may extend a fraction beyond the fitted
                // outer contour. Preserve its coverage underneath the new
                // opaque band, rather than cutting a hole in the backdrop.
                append_paint_elements(
                    &mut paint_elements,
                    optimized.clone(),
                    paint,
                    &gradient_ids,
                    paint_overlap,
                );
            }
            if band.hidden.contains(&geometry.region) {
                continue;
            }
        }
        let elements = if let Some((i, _)) = band {
            &mut band_elements[i]
        } else {
            &mut paint_elements
        };
        let first_element = elements.len();
        append_paint_elements(elements, optimized, paint, &gradient_ids, paint_overlap);
        if let Some((i, _)) = band {
            for element in elements[first_element..].iter_mut().flatten() {
                let _ = write!(element.attributes, " clip-path=\"url(#outline-inner-{i})\"");
                if let Some(attributes) = &mut element.underpaint_attributes {
                    let _ = write!(attributes, " clip-path=\"url(#outline-inner-{i})\"");
                }
            }
        }
    }
    batch_equal_paint_paths(&mut paint_elements, &mut summary);
    let mut body = String::new();
    body.push_str("<g id=\"paint-layer\" fill-rule=\"evenodd\">");
    write_paint_elements(&mut body, &paint_elements, &mut summary);
    for (i, (band, mut elements)) in structural.outlines.iter().zip(band_elements).enumerate() {
        // A complete underpaint prevents complementary antialias coverage
        // at the inner clip from exposing the page through a hairline seam.
        let fill = fill_value(&band.underpaint, &gradient_ids);
        let _ = write!(body, "<path d=\"{}\" fill=\"{fill}\"/>", band.outer);
        summary.path_elements += 1;
        if let Some((path, paint)) = &band.inner_underpaint {
            let fill = fill_value(paint, &gradient_ids);
            let _ = write!(body, "<path d=\"{path}\" fill=\"{fill}\"/>");
            summary.path_elements += 1;
        }
        for (path, paint) in &band.patches {
            let fill = fill_value(paint, &gradient_ids);
            let _ = write!(body, "<path data-outline-band=\"true\" d=\"{path}\" fill=\"{fill}\" stroke=\"{fill}\" stroke-width=\"0.25\" clip-path=\"url(#outline-outer-{i})\" fill-rule=\"evenodd\"/>");
            summary.path_elements += 1;
        }
        batch_equal_paint_paths(&mut elements, &mut summary);
        write_paint_elements(&mut body, &elements, &mut summary);
    }
    body.push_str("</g>");
    body.push_str("<g id=\"structural-ink-layer\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">");
    let mut patch_mask_uses = String::new();
    for (stroke_index, stroke) in structural.strokes.iter().enumerate() {
        if structural
            .outlines
            .iter()
            .any(|band| band.hides_stroke(&stroke.points, stroke.width))
        {
            summary.outline_structural_strokes_removed += 1;
            continue;
        }
        let data = stroke
            .path_data
            .clone()
            .unwrap_or_else(|| open_path_data(&stroke.points));
        if data.is_empty() {
            continue;
        }
        let mut attributes = format!(
            "data-structural-ink=\"line\" stroke=\"{}\" stroke-width=\"{}\"",
            rgb_hex(stroke.color),
            number(stroke.width)
        );
        if !structural.color_patches.is_empty() {
            let _ = write!(attributes, " id=\"ink-color-source-{stroke_index}\"");
            let _ = write!(
                patch_mask_uses,
                "<use href=\"#ink-color-source-{stroke_index}\"/>"
            );
        }
        if let Some((i, _)) = structural
            .outlines
            .iter()
            .enumerate()
            .find(|(_, band)| band.clips_stroke(&stroke.points))
        {
            let _ = write!(attributes, " clip-path=\"url(#outline-inner-{i})\"");
        }
        if matches!(stroke.role, "boundary-stroke" | "sampled-ink") {
            // Recovery owns a measured interval, not an inferred round cap.
            // The original Paint retains its tips and intentional breaks.
            attributes.push_str(" stroke-linecap=\"butt\"");
        }
        let (geometry, operations) = optimize_path(&data, true, true).unwrap_or((
            OptimizedElement::Path { data, bbox: None },
            Default::default(),
        ));
        summary.linear_cubics_to_lines += operations.linear_cubics;
        summary.redundant_segments_removed += operations.redundant_segments;
        summary.arc_segments += operations.arc_segments;
        summary.merged_arc_segments += operations.merged_arcs;
        let kind = write_geometry(&mut body, &geometry, &attributes);
        count_element(&mut summary, kind);
        summary.structural_strokes += 1;
    }
    body.push_str("</g>");
    if !structural.color_patches.is_empty() {
        let _ = write!(definitions, "<mask id=\"ink-color-coverage\" maskUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" style=\"mask-type:alpha\"><g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">{patch_mask_uses}</g></mask>");
        body.push_str("<g data-ink-color-patches=\"true\" mask=\"url(#ink-color-coverage)\">");
        let mut colors = std::collections::BTreeMap::<String, String>::new();
        for patch in &structural.color_patches {
            colors
                .entry(rgb_hex(patch.color))
                .or_default()
                .push_str(&patch.path);
        }
        for (color, path) in colors {
            let _ = write!(body, "<path d=\"{path}\" fill=\"{color}\"/>");
            summary.path_elements += 1;
            summary.structural_color_patches += 1;
        }
        body.push_str("</g>");
    }
    if alpha_mask.is_some() {
        body = format!("<g mask=\"url(#source-alpha-mask)\">{body}</g>");
    }
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><defs>{}</defs>{}</svg>\n",
        width, height, width, height, definitions, body
    );
    summary.bytes = document.len();
    (document, summary)
}

#[allow(clippy::too_many_arguments)]
pub fn write(
    output: &Path,
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    final_geometry: bool,
) -> Result<SvgSummary> {
    let (document, summary) = serialize(
        width,
        height,
        geometries,
        paints,
        structural,
        paint_overlap,
        final_geometry,
    );
    fs::write(output, document.as_bytes())?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(data: &str, bbox: (f64, f64, f64, f64), attributes: &str) -> Option<PaintElement> {
        Some(PaintElement {
            geometry: OptimizedElement::Path {
                data: data.to_string(),
                bbox: Some(bbox),
            },
            attributes: attributes.to_string(),
            underpaint_attributes: None,
            batchable: true,
        })
    }

    #[test]
    fn authored_alpha_boundary_does_not_add_a_bright_fringe() {
        for reversed in [false, true] {
            let mut layers = vec![
                AlphaMaskLayer {
                    path_data: "M0 0H20V20H0Z".into(),
                    opacity: 1.0,
                    paint: Some(Paint::Solid {
                        color: [170.0 / 255.0; 3],
                    }),
                },
                AlphaMaskLayer {
                    path_data: "M20 0H40V20H20Z".into(),
                    opacity: 1.0,
                    paint: Some(Paint::Solid {
                        color: [204.0 / 255.0; 3],
                    }),
                },
            ];
            if reversed {
                layers.reverse();
            }
            let (document, _) = serialize_filtered_with_alpha(
                40,
                20,
                &[],
                &[],
                &StructuralInk::empty(),
                0.0,
                true,
                &[],
                Some(&AlphaMask {
                    layers,
                    luminance: true,
                }),
            );
            // The two source RGBA colours have the same premultiplied RGB.
            let document = document.replace(
                "<g mask=\"url(#source-alpha-mask)\">",
                "<g mask=\"url(#source-alpha-mask)\"><path d=\"M0 0H20V20H0Z\" fill=\"#ffffff\"/><path d=\"M20 0H40V20H20Z\" fill=\"#d5d5d5\"/>",
            );
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            for background in [
                resvg::tiny_skia::Color::BLACK,
                resvg::tiny_skia::Color::WHITE,
            ] {
                let mut pixmap = resvg::tiny_skia::Pixmap::new(320, 160).unwrap();
                pixmap.fill(background);
                resvg::render(
                    &tree,
                    resvg::tiny_skia::Transform::from_scale(8.0, 8.0),
                    &mut pixmap.as_mut(),
                );
                for x in 144..176 {
                    let expected = if background == resvg::tiny_skia::Color::BLACK {
                        170
                    } else if x < 160 {
                        255
                    } else {
                        221
                    };
                    let actual = pixmap.pixel(x, 80).unwrap().red();
                    assert!(
                        (actual as i32 - expected).abs() <= 1,
                        "reversed={reversed}, x={x}: {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotated_radial_gradient_renders_its_major_axis_and_has_a_distinct_key() {
        let mut paint = Paint::Radial {
            origin: crate::gradient::RadialOrigin::Fitted,
            center: crate::geometry::Point { x: 16.0, y: 16.0 },
            radius: crate::geometry::Point { x: 12.0, y: 4.0 },
            rotation: std::f32::consts::FRAC_PI_4,
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.0; 3],
                },
                ColorStop {
                    offset: 1.0,
                    color: [1.0; 3],
                },
            ],
        };
        let key = paint_key(&paint).unwrap();
        let mut ids = HashMap::new();
        let mut definitions = String::new();
        register_gradient(
            &paint,
            None,
            key.clone(),
            &mut ids,
            &mut definitions,
            &mut SvgSummary::default(),
        );
        let document=format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"32\" height=\"32\"><defs>{definitions}</defs><rect width=\"32\" height=\"32\" fill=\"url(#{})\"/></svg>",ids[&key]);
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        assert!(pixmap.pixel(22, 22).unwrap().red() < 210);
        assert!(pixmap.pixel(22, 9).unwrap().red() > 245);
        if let Paint::Radial { rotation, .. } = &mut paint {
            *rotation = 0.0;
        }
        assert_ne!(paint_key(&paint).unwrap(), key);
        let unrotated_key = paint_key(&paint).unwrap();
        if let Paint::Radial { rotation, .. } = &mut paint {
            *rotation = 0.0004;
        }
        assert_ne!(paint_key(&paint).unwrap(), unrotated_key);
    }

    #[test]
    fn shared_edge_does_not_step_when_paint_order_changes() {
        let mut elements = Vec::new();
        for (x, y, width, height, color) in [
            (0.0, 0.0, 20.0, 10.25, [0.02; 3]),
            (0.0, 10.25, 40.0, 9.75, [1.0; 3]),
            (20.0, 0.0, 20.0, 10.25, [0.03; 3]),
        ] {
            append_paint_elements(
                &mut elements,
                OptimizedElement::Rect {
                    x,
                    y,
                    width,
                    height,
                },
                &Paint::Solid { color },
                &HashMap::new(),
                0.3,
            );
        }
        let mut body = String::new();
        write_paint_elements(&mut body, &elements, &mut SvgSummary::default());
        let document = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40\" height=\"20\">{body}</svg>"
        );
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(320, 160).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_scale(8.0, 8.0),
            &mut pixmap.as_mut(),
        );
        for x in [80, 240] {
            assert!(pixmap.pixel(x, 80).unwrap().red() < 30);
            assert!(pixmap.pixel(x, 83).unwrap().red() > 245);
            for y in 79..85 {
                assert_eq!(pixmap.pixel(x, y).unwrap().alpha(), 255);
            }
        }
    }

    #[test]
    fn equal_paint_crosses_only_spatially_disjoint_intervening_elements() {
        let mut elements = vec![
            path("M0 0L4 0L4 4L0 4Z", (0.0, 0.0, 4.0, 4.0), "red"),
            path("M20 0L24 0L24 4L20 4Z", (20.0, 0.0, 24.0, 4.0), "blue"),
            path("M40 0L44 0L44 4L40 4Z", (40.0, 0.0, 44.0, 4.0), "red"),
        ];
        let mut summary = SvgSummary::default();
        batch_equal_paint_paths(&mut elements, &mut summary);
        assert!(elements[2].is_none());
        assert_eq!(summary.paint_paths_merged, 1);

        let mut blocked = vec![
            path("M0 0L4 0L4 4L0 4Z", (0.0, 0.0, 4.0, 4.0), "red"),
            path("M10 0L16 0L16 4L10 4Z", (10.0, 0.0, 16.0, 4.0), "blue"),
            path("M8 0L12 0L12 4L8 4Z", (8.0, 0.0, 12.0, 4.0), "red"),
        ];
        let mut summary = SvgSummary::default();
        batch_equal_paint_paths(&mut blocked, &mut summary);
        assert!(blocked[2].is_some());
        assert_eq!(summary.paint_paths_merged, 0);
    }

    #[test]
    fn layered_paint_emits_ordered_transparent_gradient_components() {
        let overlay = PaintOverlay {
            paint: Box::new(Paint::Radial {
                rotation: 0.0,
                origin: crate::gradient::RadialOrigin::Fitted,
                center: crate::geometry::Point { x: 5.0, y: 5.0 },
                radius: crate::geometry::Point { x: 4.0, y: 3.0 },
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
                    opacity: 0.7,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let mut ids = HashMap::new();
        let mut definitions = String::new();
        let mut summary = SvgSummary::default();
        register_gradient(
            &overlay.paint,
            Some(&overlay.opacity_stops),
            overlay_key(&overlay),
            &mut ids,
            &mut definitions,
            &mut summary,
        );
        assert!(definitions.contains("stop-opacity=\"0.7\""));
        assert!(definitions.contains("stop-opacity=\"0\""));

        let paint = Paint::Layered {
            base: Box::new(Paint::Solid {
                color: [0.2, 0.3, 0.4],
            }),
            overlays: vec![overlay],
        };
        let mut elements = Vec::new();
        append_paint_elements(
            &mut elements,
            OptimizedElement::Path {
                data: "M0 0L10 0L10 10Z".to_string(),
                bbox: Some((0.0, 0.0, 10.0, 10.0)),
            },
            &paint,
            &ids,
            0.2,
        );
        assert_eq!(elements.len(), 2);
        assert!(elements.iter().flatten().all(|element| !element.batchable));
    }
}
