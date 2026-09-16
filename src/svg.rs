use crate::svg_document::{attrs, Attributes};
use crate::svg_document::{Document, Elements};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::color::rgb_hex;
use crate::geometry::{open_path_data, Primitive, RegionGeometry};
use crate::gradient::{ColorStop, OpacityStop, Paint, PaintOverlay};
use crate::optimize::{
    format_number, optimize_path, separated_bboxes, OptimizedElement, PathOptimization,
};
use crate::structural::StructuralInk;
use crate::Result;

#[derive(Clone, Debug, Default, Serialize)]
pub struct SvgSummary {
    /// Drawable XML elements in the final document, excluding definitions.
    pub objects: usize,
    /// Separate moveto contours inside final drawable path elements.
    pub path_subpaths: usize,
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
    pub invisible_elements_removed: usize,
    pub invisible_strokes_removed: usize,
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
        self.invisible_elements_removed += other.invisible_elements_removed;
        self.invisible_strokes_removed += other.invisible_strokes_removed;
        self.outline_bands += other.outline_bands;
        self.outline_color_patches += other.outline_color_patches;
        self.outline_paint_regions_removed += other.outline_paint_regions_removed;
        self.outline_structural_strokes_removed += other.outline_structural_strokes_removed;
    }
}

#[derive(Clone, Debug)]
struct PaintElement {
    geometry: OptimizedElement,
    attributes: Attributes,
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

/// Exact output identity, excluding fitting diagnostics and sub-byte RGB
/// differences that SVG cannot express. Used before tracing shared geometry.
pub(crate) fn appearance_key(paint: &Paint) -> Option<String> {
    match paint {
        Paint::Solid { color } => Some(format!("S:{}", rgb_hex(*color))),
        Paint::Layered { .. } => None,
        _ => paint_key(paint),
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

fn stop_elements(stops: &[ColorStop]) -> Elements {
    let mut output = Elements::new();
    for stop in stops {
        output.leaf(
            "stop",
            attrs([
                ("offset", (number(stop.offset as f32)).to_string()),
                (
                    "stop-color",
                    (rgb_hex(stop.color.map(|value| value as f32))).to_string(),
                ),
            ]),
        );
    }
    output
}

fn overlay_stop_elements(stops: &[ColorStop], opacity_stops: &[OpacityStop]) -> Elements {
    let mut offsets = stops.iter().map(|stop| stop.offset).collect::<Vec<_>>();
    offsets.extend(opacity_stops.iter().map(|stop| stop.offset));
    offsets.sort_by(f64::total_cmp);
    offsets.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
    let mut output = Elements::new();
    for offset in offsets {
        output.leaf(
            "stop",
            attrs([
                ("offset", (number(offset as f32)).to_string()),
                (
                    "stop-color",
                    format!(
                        "{0}",
                        rgb_hex(color_at(stops, offset).map(|value| value as f32))
                    ),
                ),
                (
                    "stop-opacity",
                    (number(opacity_at(opacity_stops, offset) as f32)).to_string(),
                ),
            ]),
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

fn write_paint_elements(
    body: &mut Elements,
    elements: &[Option<PaintElement>],
    summary: &mut SvgSummary,
) {
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

#[cfg(test)]
fn batch_equal_paint_paths(elements: &mut [Option<PaintElement>], summary: &mut SvgSummary) {
    batch_equal_paint_paths_impl(elements, summary, &mut Vec::new());
}

fn batch_equal_paint_paths_impl(
    elements: &mut [Option<PaintElement>],
    summary: &mut SvgSummary,
    merges: &mut Vec<(usize, usize)>,
) {
    let mut signature_ids = HashMap::<Attributes, usize>::new();
    let mut latest_spatial = HashMap::<(i64, i64), Vec<(usize, usize)>>::new();
    let mut global_blockers = Vec::<(usize, usize)>::new();
    let mut batches = HashMap::<usize, Vec<PaintBatch>>::new();
    for current in 0..elements.len() {
        let Some(element) = elements[current].as_ref() else {
            continue;
        };
        let signature = element.attributes.clone();
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
            merges.push((current, target));
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
    body: &mut Elements,
    geometry: &OptimizedElement,
    attributes: &Attributes,
) -> &'static str {
    let (kind, mut geometry_attributes) = match geometry {
        OptimizedElement::Path { data, .. } => ("path", attrs([("d", data.clone())])),
        OptimizedElement::Line { x1, y1, x2, y2 } => (
            "line",
            attrs([
                ("x1", format_number(*x1)),
                ("y1", format_number(*y1)),
                ("x2", format_number(*x2)),
                ("y2", format_number(*y2)),
            ]),
        ),
        OptimizedElement::Rect {
            x,
            y,
            width,
            height,
        } => (
            "rect",
            attrs([
                ("x", format_number(*x)),
                ("y", format_number(*y)),
                ("width", format_number(*width)),
                ("height", format_number(*height)),
            ]),
        ),
        OptimizedElement::Circle { cx, cy, radius } => (
            "circle",
            attrs([
                ("cx", format_number(*cx)),
                ("cy", format_number(*cy)),
                ("r", format_number(*radius)),
            ]),
        ),
    };
    geometry_attributes.extend(attributes.iter().cloned());
    body.leaf(kind, geometry_attributes);
    kind
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
    definitions: &mut Elements,
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
            definitions.open(
                "linearGradient",
                attrs([
                    ("id", (id).to_string()),
                    ("gradientUnits", "userSpaceOnUse".into()),
                    ("x1", (number(start.x)).to_string()),
                    ("y1", (number(start.y)).to_string()),
                    ("x2", (number(end.x)).to_string()),
                    ("y2", (number(end.y)).to_string()),
                ]),
            );
            definitions.append(elements);
            definitions.close();
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
            definitions.open(
                "radialGradient",
                attrs([
                    ("id", (id).to_string()),
                    ("gradientUnits", "userSpaceOnUse".into()),
                    ("cx", "0".into()),
                    ("cy", "0".into()),
                    ("r", "1".into()),
                    (
                        "gradientTransform",
                        format!(
                            "translate({0} {1}){2} scale({3} {4})",
                            number(center.x),
                            number(center.y),
                            rotation,
                            number(radius.x.max(0.001)),
                            number(radius.y.max(0.001))
                        ),
                    ),
                ]),
            );
            definitions.append(elements);
            definitions.close();
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

#[allow(clippy::too_many_arguments)]
fn append_rgba_elements(
    elements: &mut Vec<Option<PaintElement>>,
    geometry: OptimizedElement,
    paint: &Paint,
    alpha: &Paint,
    overlay: Option<&[OpacityStop]>,
    overlap: f32,
    gradient_ids: &mut HashMap<String, String>,
    definitions: &mut Elements,
    summary: &mut SvgSummary,
) {
    if let Paint::Layered { base, overlays } = paint {
        append_rgba_elements(
            elements,
            geometry.clone(),
            base,
            alpha,
            None,
            overlap,
            gradient_ids,
            definitions,
            summary,
        );
        for layer in overlays {
            if !layer.opacity_stops.is_empty()
                && layer.opacity_stops.iter().all(|s| s.opacity <= 0.0)
            {
                continue;
            }
            append_rgba_elements(
                elements,
                geometry.clone(),
                &layer.paint,
                alpha,
                Some(&layer.opacity_stops),
                overlap,
                gradient_ids,
                definitions,
                summary,
            );
        }
        return;
    }
    let (rgba, stops, opacity) = match alpha {
        Paint::Solid { color } => (paint.clone(), overlay.map(|s| s.to_vec()), color[0]),
        Paint::Linear { .. } | Paint::Radial { .. } => {
            let mut rgba = alpha.clone();
            let (Paint::Linear { stops, .. } | Paint::Radial { stops, .. }) = &mut rgba else {
                unreachable!()
            };
            let opacity = stops
                .iter()
                .map(|s| OpacityStop {
                    offset: s.offset,
                    opacity: s.color[0],
                })
                .collect::<Vec<_>>();
            if let Paint::Solid { color } = paint {
                for stop in stops {
                    stop.color = color.map(f64::from);
                }
            } else {
                assert!(
                    crate::gradient::same_gradient_geometry(paint, alpha),
                    "RGB and alpha must share a gradient coordinate"
                );
                rgba = paint.clone();
            }
            (rgba, Some(opacity), 1.0)
        }
        Paint::Layered { .. } => unreachable!("alpha fields have no layered paint"),
    };
    if opacity <= 0.0
        || stops
            .as_ref()
            .is_some_and(|s| !s.is_empty() && s.iter().all(|s| s.opacity <= 0.0))
    {
        return;
    }
    let fill = if let Paint::Solid { color } = rgba {
        rgb_hex(color)
    } else {
        let key = format!("rgba:{:?}:{:?}", rgba, stops);
        register_gradient(
            &rgba,
            stops.as_deref(),
            key.clone(),
            gradient_ids,
            definitions,
            summary,
        );
        format!("url(#{})", gradient_ids[&key])
    };
    let mut attributes = attrs([("fill", fill.to_string())]);
    if opacity < 1.0 {
        attributes.push(("fill-opacity".into(), number(opacity)));
    }
    elements.push(Some(PaintElement {
        geometry,
        attributes,
        batchable: true,
    }));
}

fn stroke_outline(data: &str, width: f32, butt: bool) -> Option<String> {
    use resvg::tiny_skia::{LineCap, LineJoin, PathSegment, Stroke};
    let svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"><path d=\"{data}\" fill=\"none\" stroke=\"black\"/></svg>");
    let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).ok()?;
    let resvg::usvg::Node::Path(path) = tree.root().children().first()? else {
        return None;
    };
    let outline = path.data().stroke(
        &Stroke {
            width,
            line_cap: if butt { LineCap::Butt } else { LineCap::Round },
            line_join: LineJoin::Round,
            ..Stroke::default()
        },
        1.0,
    )?;
    let mut result = String::new();
    for segment in outline.segments() {
        match segment {
            PathSegment::MoveTo(p) => {
                let _ = write!(result, "M{} {}", number(p.x), number(p.y));
            }
            PathSegment::LineTo(p) => {
                let _ = write!(result, "L{} {}", number(p.x), number(p.y));
            }
            PathSegment::QuadTo(a, b) => {
                let _ = write!(
                    result,
                    "Q{} {} {} {}",
                    number(a.x),
                    number(a.y),
                    number(b.x),
                    number(b.y)
                );
            }
            PathSegment::CubicTo(a, b, c) => {
                let _ = write!(
                    result,
                    "C{} {} {} {} {} {}",
                    number(a.x),
                    number(a.y),
                    number(b.x),
                    number(b.y),
                    number(c.x),
                    number(c.y)
                );
            }
            PathSegment::Close => result.push('Z'),
        }
    }
    Some(result)
}

fn append_paint_elements(
    elements: &mut Vec<Option<PaintElement>>,
    geometry: OptimizedElement,
    paint: &Paint,
    gradient_ids: &HashMap<String, String>,
    _paint_overlap: f32,
) {
    match paint {
        Paint::Layered { base, overlays } => {
            let base_fill = fill_value(base, gradient_ids);
            elements.push(Some(PaintElement {
                geometry: geometry.clone(),
                attributes: attrs([("fill", base_fill.to_string())]),
                batchable: false,
            }));
            for overlay in overlays {
                if !overlay.opacity_stops.is_empty()
                    && overlay.opacity_stops.iter().all(|stop| stop.opacity <= 0.0)
                {
                    continue;
                }
                let fill = match overlay.paint.as_ref() {
                    Paint::Solid { color } => rgb_hex(*color),
                    Paint::Linear { .. } | Paint::Radial { .. } => {
                        format!("url(#{})", gradient_ids[&overlay_key(overlay)])
                    }
                    Paint::Layered { .. } => continue,
                };
                elements.push(Some(PaintElement {
                    geometry: geometry.clone(),
                    attributes: attrs([("fill", (fill).to_string())]),
                    batchable: false,
                }));
            }
        }
        _ => {
            let fill = fill_value(paint, gradient_ids);
            elements.push(Some(PaintElement {
                geometry,
                attributes: attrs([("fill", fill.to_string())]),
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
) -> (Document, SvgSummary) {
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
) -> (Document, SvgSummary) {
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
    strokes: HashMap<String, Option<(OptimizedElement, PathOptimization)>>,
    stroke_outlines: HashMap<String, HashMap<(u32, bool), Option<String>>>,
    batches: std::collections::VecDeque<BatchPlan>,
}

struct BatchKey {
    attributes: Attributes,
    batchable: bool,
    bbox: Option<[u64; 4]>,
}

fn batch_bbox(element: &PaintElement) -> Option<[u64; 4]> {
    paint_path_bbox(element).map(|b| [b.0.to_bits(), b.1.to_bits(), b.2.to_bits(), b.3.to_bits()])
}

struct BatchPlan {
    keys: Vec<Option<BatchKey>>,
    merges: Vec<(usize, usize)>,
    batches: usize,
    bytes: usize,
}

impl BatchPlan {
    fn matches(&self, elements: &[Option<PaintElement>]) -> bool {
        self.keys.len() == elements.len()
            && self
                .keys
                .iter()
                .zip(elements)
                .all(|(key, element)| match (key, element) {
                    (None, None) => true,
                    (Some(k), Some(e)) => {
                        k.attributes == e.attributes
                            && k.batchable == e.batchable
                            && k.bbox == batch_bbox(e)
                    }
                    _ => false,
                })
    }
}

impl GeometryCache {
    // Batch decisions depend on exact paint attributes, batchability, and
    // original path bounds, not path commands. Reuse only those decisions;
    // append the current commands in the same order as the original loop.
    fn batch(&mut self, elements: &mut [Option<PaintElement>], summary: &mut SvgSummary) {
        if elements.is_empty() {
            return;
        }
        if let Some(index) = self.batches.iter().position(|p| p.matches(elements)) {
            let plan = self.batches.remove(index).unwrap();
            for &(current, target) in &plan.merges {
                let current = elements[current].take().unwrap();
                let OptimizedElement::Path {
                    data: source,
                    bbox: Some(b),
                    ..
                } = current.geometry
                else {
                    unreachable!()
                };
                let OptimizedElement::Path { data, bbox } =
                    &mut elements[target].as_mut().unwrap().geometry
                else {
                    unreachable!()
                };
                data.push(' ');
                data.push_str(source.trim());
                *bbox = Some(match *bbox {
                    Some(a) => (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
                    None => b,
                });
            }
            summary.paint_batches += plan.batches;
            summary.paint_paths_merged += plan.merges.len();
            self.batches.push_back(plan);
            return;
        }
        let keys: Vec<Option<BatchKey>> = elements
            .iter()
            .map(|e| {
                e.as_ref().map(|e| BatchKey {
                    attributes: e.attributes.clone(),
                    batchable: e.batchable,
                    bbox: batch_bbox(e),
                })
            })
            .collect();
        let before = summary.paint_batches;
        let mut merges = Vec::new();
        batch_equal_paint_paths_impl(elements, summary, &mut merges);
        let bytes = keys.len() * std::mem::size_of::<Option<BatchKey>>()
            + keys
                .iter()
                .flatten()
                .map(|k| {
                    k.attributes
                        .iter()
                        .map(|(name, value)| name.len() + value.len())
                        .sum::<usize>()
                })
                .sum::<usize>()
            + merges.len() * std::mem::size_of::<(usize, usize)>();
        const LIMIT: usize = 64 * 1024 * 1024;
        if bytes > LIMIT {
            return;
        }
        while self.batches.len() >= 4
            || bytes + self.batches.iter().map(|p| p.bytes).sum::<usize>() > LIMIT
        {
            self.batches.pop_front();
        }
        self.batches.push_back(BatchPlan {
            keys,
            merges,
            batches: summary.paint_batches - before,
            bytes,
        });
    }

    fn optimize(&mut self, path: &str) -> Option<(OptimizedElement, PathOptimization)> {
        if let Some(cached) = self.paths.get(path) {
            return cached.clone();
        }
        let optimized = optimize_path(path, true, false);
        self.paths.insert(path.to_owned(), optimized.clone());
        optimized
    }
    fn optimize_stroke(&mut self, path: &str) -> Option<(OptimizedElement, PathOptimization)> {
        if let Some(cached) = self.strokes.get(path) {
            return cached.clone();
        }
        let optimized = optimize_path(path, true, true);
        self.strokes.insert(path.to_owned(), optimized.clone());
        optimized
    }

    fn stroke_outline(&mut self, path: &str, width: f32, butt: bool) -> Option<String> {
        let key = (width.to_bits(), butt);
        if let Some(cached) = self.stroke_outlines.get(path).and_then(|v| v.get(&key)) {
            return cached.clone();
        }
        let outline = stroke_outline(path, width, butt);
        self.stroke_outlines
            .entry(path.to_owned())
            .or_default()
            .insert(key, outline.clone());
        outline
    }
}

/// Serialize selected raster regions and optionally apply an independent
/// intrinsic opacity to the individual colour faces and ink strokes.
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
    face_alpha: Option<&crate::face_alpha::FaceAlpha>,
) -> (Document, SvgSummary) {
    serialize_filtered_with_alpha_cached(
        width,
        height,
        geometries,
        paints,
        structural,
        paint_overlap,
        final_geometry,
        excluded_regions,
        face_alpha,
        &mut GeometryCache::default(),
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
    _final_geometry: bool,
    excluded_regions: &[bool],
    face_alpha: Option<&crate::face_alpha::FaceAlpha>,
    geometry_cache: &mut GeometryCache,
) -> (Document, SvgSummary) {
    serialize_prepared(
        width,
        height,
        geometries,
        paints,
        structural,
        paint_overlap,
        excluded_regions,
        face_alpha,
        geometry_cache,
        None,
    )
}

/// Hole trials keep the complete paint context borrowed and immutable. Only
/// geometry may change, so gradient IDs and fill attributes can be prepared
/// once without comparing approximate paint keys or reusing stale context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hole_serializer<'a>(
    width: usize,
    height: usize,
    paints: &'a [Paint],
    structural: &'a StructuralInk,
    paint_overlap: f32,
    excluded_regions: &'a [bool],
    face_alpha: Option<&'a crate::face_alpha::FaceAlpha>,
    geometry_cache: &'a mut GeometryCache,
) -> impl FnMut(&[RegionGeometry]) -> (Document, SvgSummary) + 'a {
    // RGBA fields can register gradients in geometry order and reconstruct
    // source-dependent boundary colours. Keep their full serialization path.
    let mut prepared = None;
    move |geometries| {
        if face_alpha.is_none() && prepared.is_none() {
            prepared = Some(FixedPaints::new(paints, structural, excluded_regions));
        }
        serialize_prepared(
            width,
            height,
            geometries,
            paints,
            structural,
            paint_overlap,
            excluded_regions,
            face_alpha,
            geometry_cache,
            prepared.as_ref(),
        )
    }
}

struct FixedPaints {
    gradient_ids: HashMap<String, String>,
    definitions: Elements,
    summary: SvgSummary,
    attributes: Vec<Vec<(Attributes, bool)>>,
}

impl FixedPaints {
    fn new(paints: &[Paint], structural: &StructuralInk, excluded: &[bool]) -> Self {
        let (gradient_ids, definitions, summary) =
            paint_definitions(paints, structural, excluded, None);
        let attributes = paints
            .iter()
            .enumerate()
            .map(|(region, paint)| {
                if excluded.get(region).copied().unwrap_or(false)
                    || structural.outlines.iter().any(|band| {
                        band.hidden.contains(&(region as u32))
                            && !band.boundary_underpaint.contains(&(region as u32))
                    })
                {
                    return Vec::new();
                }
                let mut elements = Vec::new();
                append_paint_elements(
                    &mut elements,
                    OptimizedElement::Path {
                        data: String::new(),
                        bbox: None,
                    },
                    paint,
                    &gradient_ids,
                    0.0,
                );
                elements
                    .into_iter()
                    .flatten()
                    .map(|e| (e.attributes, e.batchable))
                    .collect()
            })
            .collect();
        Self {
            gradient_ids,
            definitions,
            summary,
            attributes,
        }
    }

    fn append(
        &self,
        elements: &mut Vec<Option<PaintElement>>,
        geometry: OptimizedElement,
        region: usize,
    ) {
        let mut attributes = self.attributes[region].iter().peekable();
        while let Some((text, batchable)) = attributes.next() {
            if attributes.peek().is_none() {
                elements.push(Some(PaintElement {
                    geometry,
                    attributes: text.clone(),
                    batchable: *batchable,
                }));
                break;
            }
            elements.push(Some(PaintElement {
                geometry: geometry.clone(),
                attributes: text.clone(),
                batchable: *batchable,
            }));
        }
    }
}

fn paint_definitions(
    paints: &[Paint],
    structural: &StructuralInk,
    excluded_regions: &[bool],
    face_alpha: Option<&crate::face_alpha::FaceAlpha>,
) -> (HashMap<String, String>, Elements, SvgSummary) {
    let mut gradient_ids = HashMap::<String, String>::new();
    let mut definitions = Elements::new();
    let mut summary = SvgSummary::default();
    for (region, paint) in paints.iter().enumerate() {
        if face_alpha.is_some() {
            continue;
        }
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
                    if !overlay.opacity_stops.is_empty()
                        && overlay.opacity_stops.iter().all(|stop| stop.opacity <= 0.0)
                    {
                        continue;
                    }
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
        for (side, data) in [("inner", &band.inner), ("outer", &band.outer)] {
            definitions.open("clipPath", attrs([("id", format!("outline-{side}-{i}"))]));
            definitions.leaf("path", attrs([("d", format!("{data}"))]));
            definitions.close();
        }
    }
    (gradient_ids, definitions, summary)
}

#[allow(clippy::too_many_arguments)]
fn serialize_prepared(
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    paint_overlap: f32,
    excluded_regions: &[bool],
    face_alpha: Option<&crate::face_alpha::FaceAlpha>,
    geometry_cache: &mut GeometryCache,
    prepared: Option<&FixedPaints>,
) -> (Document, SvgSummary) {
    let (mut gradient_ids, mut definitions, mut summary) = match prepared {
        Some(p) => (
            p.gradient_ids.clone(),
            p.definitions.clone(),
            p.summary.clone(),
        ),
        None => paint_definitions(paints, structural, excluded_regions, face_alpha),
    };
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
                let path_data = if paint_overlap > 0.0 {
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
        if let Some(alpha) = face_alpha {
            append_rgba_elements(
                &mut paint_elements,
                optimized,
                paint,
                &alpha.fields[geometry.region as usize],
                None,
                paint_overlap,
                &mut gradient_ids,
                &mut definitions,
                &mut summary,
            );
            continue;
        }
        if let Some((_, band)) = band {
            if band.boundary_underpaint.contains(&geometry.region) {
                // Shared geometry may extend a fraction beyond the fitted
                // outer contour. Preserve its coverage underneath the new
                // opaque band, rather than cutting a hole in the backdrop.
                if let Some(prepared) = prepared {
                    prepared.append(
                        &mut paint_elements,
                        optimized.clone(),
                        geometry.region as usize,
                    );
                } else {
                    append_paint_elements(
                        &mut paint_elements,
                        optimized.clone(),
                        paint,
                        &gradient_ids,
                        paint_overlap,
                    );
                }
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
        if let Some(prepared) = prepared {
            prepared.append(elements, optimized, geometry.region as usize);
        } else {
            append_paint_elements(elements, optimized, paint, &gradient_ids, paint_overlap);
        }
        if let Some((i, _)) = band {
            for element in elements[first_element..].iter_mut().flatten() {
                element
                    .attributes
                    .push(("clip-path".into(), format!("url(#outline-inner-{i})")));
            }
        }
    }
    // Retain fitting context only in the boundary preview. Paint fully inside
    // a replacement rectangle cannot affect the final SVG and is not emitted.
    let boundary_paints = face_alpha
        .filter(|a| !a.source_fields.is_empty())
        .map(|alpha| {
            let mut context = Elements::new();
            write_paint_elements(&mut context, &paint_elements, &mut SvgSummary::default());
            paint_elements.retain(|element| {
                let Some(element) = element else {
                    return true;
                };
                let bbox = match &element.geometry {
                    OptimizedElement::Path { bbox, .. } => *bbox,
                    OptimizedElement::Rect {
                        x,
                        y,
                        width,
                        height,
                    } => Some((*x, *y, x + width, y + height)),
                    OptimizedElement::Circle { cx, cy, radius } => {
                        Some((cx - radius, cy - radius, cx + radius, cy + radius))
                    }
                    OptimizedElement::Line { .. } => None,
                };
                let hidden = bbox.is_some_and(|b| {
                    alpha.source_fields.iter().any(|p| {
                        b.0 >= p.x as f64 + 2.0
                            && b.1 >= p.y as f64 + 2.0
                            && b.2 <= (p.x + p.width) as f64 - 2.0
                            && b.3 <= (p.y + p.height) as f64 - 2.0
                    })
                });
                if hidden {
                    summary.invisible_elements_removed += 1;
                }
                !hidden
            });
            context
        });
    geometry_cache.batch(&mut paint_elements, &mut summary);
    let mut body = Elements::new();
    body.open(
        "g",
        attrs([
            ("id", "paint-layer".into()),
            // Shared contours have oriented outer loops and holes. Hidden-edge
            // expansion can make parts of one face overlap; parity would cut
            // transparent pinholes into those overlaps instead of filling them.
            ("fill-rule", "nonzero".into()),
        ]),
    );
    write_paint_elements(&mut body, &paint_elements, &mut summary);
    let paint_body_end = body.child_count();
    if let Some(alpha) = face_alpha {
        for layer in &alpha.composite_layers {
            let optimized =
                if let Some((optimized, operations)) = geometry_cache.optimize(&layer.path) {
                    summary.linear_cubics_to_lines += operations.linear_cubics;
                    summary.redundant_segments_removed += operations.redundant_segments;
                    summary.arc_segments += operations.arc_segments;
                    summary.merged_arc_segments += operations.merged_arcs;
                    optimized
                } else {
                    OptimizedElement::Path {
                        data: layer.path.clone(),
                        bbox: None,
                    }
                };
            let attributes = attrs([
                (
                    "fill",
                    (if layer.white { "#fff" } else { "#000" }).to_string(),
                ),
                ("fill-opacity", format!("{0:.7}", layer.opacity)),
                // Coverage isolines are undirected, unlike shared face loops.
                ("fill-rule", "evenodd".into()),
            ]);
            let kind = write_geometry(&mut body, &optimized, &attributes);
            count_element(&mut summary, kind);
        }
    }
    for (i, (band, mut elements)) in structural.outlines.iter().zip(band_elements).enumerate() {
        // A complete underpaint prevents complementary antialias coverage
        // at the inner clip from exposing the page through a hairline seam.
        let fill = fill_value(&band.underpaint, &gradient_ids);
        body.leaf(
            "path",
            attrs([("d", (band.outer).to_string()), ("fill", fill.to_string())]),
        );
        summary.path_elements += 1;
        if let Some((path, paint)) = &band.inner_underpaint {
            let fill = fill_value(paint, &gradient_ids);
            body.leaf(
                "path",
                attrs([("d", path.to_string()), ("fill", fill.to_string())]),
            );
            summary.path_elements += 1;
        }
        for (path, paint) in &band.patches {
            let fill = fill_value(paint, &gradient_ids);
            body.leaf(
                "path",
                attrs([
                    ("data-outline-band", "true".into()),
                    ("d", path.to_string()),
                    ("fill", fill.to_string()),
                    ("stroke", fill.to_string()),
                    ("stroke-width", "0.25".into()),
                    ("clip-path", format!("url(#outline-outer-{i})")),
                    ("fill-rule", "evenodd".into()),
                ]),
            );
            summary.path_elements += 1;
        }
        geometry_cache.batch(&mut elements, &mut summary);
        write_paint_elements(&mut body, &elements, &mut summary);
    }
    body.close();
    body.open(
        "g",
        attrs([
            ("id", "structural-ink-layer".into()),
            ("fill", "none".into()),
            ("stroke-linecap", "round".into()),
            ("stroke-linejoin", "round".into()),
        ]),
    );
    let mut patch_mask_uses = Elements::new();
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
        let mut attributes = attrs([
            ("data-structural-ink", "line".into()),
            ("stroke", (rgb_hex(stroke.color)).to_string()),
            ("stroke-width", (number(stroke.width)).to_string()),
        ]);
        if let Some(alpha) = face_alpha {
            if alpha.ink_opacity <= 0.0 {
                continue;
            }
            attributes.push(("stroke-opacity".into(), number(alpha.ink_opacity)));
        }
        if !structural.color_patches.is_empty() {
            attributes.push(("id".into(), format!("ink-color-source-{stroke_index}")));
        }
        if let Some((i, _)) = structural
            .outlines
            .iter()
            .enumerate()
            .find(|(_, band)| band.clips_stroke(&stroke.points))
        {
            attributes.push(("clip-path".into(), format!("url(#outline-inner-{i})")));
        }
        if matches!(stroke.role, "boundary-stroke" | "sampled-ink") {
            // Recovery owns a measured interval, not an inferred round cap.
            // The original Paint retains its tips and intentional breaks.
            attributes.push(("stroke-linecap".into(), "butt".into()));
        }
        if !structural.color_patches.is_empty() {
            if let Some(outline) = geometry_cache.stroke_outline(
                &data,
                stroke.width,
                matches!(stroke.role, "boundary-stroke" | "sampled-ink"),
            ) {
                let mut outline_attributes = attrs([("d", outline)]);
                if let Some(start) = attributes.iter().position(|(name, _)| name == "clip-path") {
                    outline_attributes.extend(attributes[start..].iter().cloned());
                }
                patch_mask_uses.leaf("path", outline_attributes);
            }
        }
        let (geometry, operations) = geometry_cache.optimize_stroke(&data).unwrap_or((
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
    body.close();
    if !structural.color_patches.is_empty() {
        definitions.append(patch_mask_uses.wrap(
            "clipPath",
            attrs([
                ("id", "ink-color-coverage".into()),
                ("clipPathUnits", "userSpaceOnUse".into()),
            ]),
        ));
        body.open(
            "g",
            attrs([
                ("data-ink-color-patches", "true".into()),
                ("clip-path", "url(#ink-color-coverage)".into()),
            ]),
        );
        let mut colors = std::collections::BTreeMap::<String, String>::new();
        for patch in &structural.color_patches {
            colors
                .entry(rgb_hex(patch.color))
                .or_default()
                .push_str(&patch.path);
        }
        for (color, path) in colors {
            body.leaf(
                "path",
                attrs([("d", path.to_string()), ("fill", format!("{color}"))]),
            );
            summary.path_elements += 1;
            summary.structural_color_patches += 1;
        }
        body.close();
    }
    if let Some(alpha) = face_alpha.filter(|a| !a.source_fields.is_empty()) {
        let mut base_body = body.clone();
        if let Some(context) = &boundary_paints {
            base_body.replace_prefix(paint_body_end, context.clone());
        }
        let base_document=format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><defs>{definitions}</defs>{base_body}</svg>");
        let mut base = resvg::tiny_skia::Pixmap::new(width as u32, height as u32).unwrap();
        if let Ok(tree) =
            resvg::usvg::Tree::from_str(&base_document, &resvg::usvg::Options::default())
        {
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut base.as_mut(),
            );
        }
        definitions.append(crate::colour_fields::composite_filter());
        let mut outside = format!("M0 0H{width}V{height}H0Z");
        for p in &alpha.source_fields {
            write!(
                outside,
                "M{} {}h{}v{}h-{}Z",
                p.x, p.y, p.width, p.height, p.width
            )
            .unwrap();
            // Complementary clips would attenuate the same boundary twice.
            // Retain opaque support under the patch's antialiased clip edge.
            for [x, y, w, h] in crate::colour_fields::opaque_boundary_overlap(p) {
                write!(outside, "M{x} {y}h{w}v{h}h-{w}Z").unwrap();
            }
        }
        definitions.open("clipPath", attrs([("id", "source-field-outside".into())]));
        definitions.leaf(
            "path",
            attrs([("d", format!("{outside}")), ("clip-rule", "evenodd".into())]),
        );
        definitions.close();
        body = body.wrap(
            "g",
            attrs([("clip-path", "url(#source-field-outside)".into())]),
        );
        for (index, p) in alpha.source_fields.iter().enumerate() {
            definitions.open("clipPath", attrs([("id", format!("source-field-{index}"))]));
            definitions.leaf(
                "rect",
                attrs([
                    ("x", (p.x - p.origin_x).to_string()),
                    ("y", (p.y - p.origin_y).to_string()),
                    ("width", (p.width).to_string()),
                    ("height", (p.height).to_string()),
                ]),
            );
            definitions.close();
            body.open(
                "g",
                attrs([
                    (
                        "transform",
                        format!("translate({0} {1})", p.origin_x, p.origin_y),
                    ),
                    ("clip-path", format!("url(#source-field-{index})")),
                    ("style", "isolation:isolate".into()),
                    ("filter", "url(#source-field-composite)".into()),
                ]),
            );
            let mut previous = None;
            let matched = crate::colour_fields::match_boundary(p, &base);
            for layer in &matched {
                if previous != Some(layer.color) {
                    if previous.is_some() {
                        body.close();
                    }
                    body.open("g", attrs([("style", "isolation:isolate".into())]));
                    previous = Some(layer.color);
                }
                let optimized = geometry_cache
                    .optimize(&layer.path)
                    .map(|(g, operations)| {
                        summary.linear_cubics_to_lines += operations.linear_cubics;
                        summary.redundant_segments_removed += operations.redundant_segments;
                        summary.arc_segments += operations.arc_segments;
                        summary.merged_arc_segments += operations.merged_arcs;
                        g
                    })
                    .unwrap_or_else(|| OptimizedElement::Path {
                        data: layer.path.clone(),
                        bbox: None,
                    });
                let attributes = attrs([
                    (
                        "fill",
                        format!(
                            "#{0:02x}{1:02x}{2:02x}",
                            layer.color[0], layer.color[1], layer.color[2]
                        ),
                    ),
                    ("fill-opacity", format!("{0:.7}", layer.opacity)),
                    ("fill-rule", "evenodd".into()),
                    ("filter", "url(#source-field-composite)".into()),
                ]);
                let kind = write_geometry(&mut body, &optimized, &attributes);
                count_element(&mut summary, kind);
            }
            if previous.is_some() {
                body.close();
            }
            body.close();
        }
    }
    let document = Document::from_parts(width, height, definitions, body);
    (summary.objects, summary.path_subpaths) = document.counts();
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
            attributes: attrs([("fill", attributes.into())]),
            batchable: true,
        })
    }

    #[test]
    fn hole_trials_preserve_serialization_across_geometry_and_alpha_changes() {
        use crate::geometry::Point;
        let gradient = Paint::Radial {
            origin: crate::gradient::RadialOrigin::Fitted,
            center: Point { x: 16.0, y: 16.0 },
            radius: Point { x: 12.0, y: 8.0 },
            rotation: 0.3,
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.1, 0.2, 0.3],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.7, 0.8, 0.9],
                },
            ],
        };
        let overlay = PaintOverlay {
            paint: Box::new(gradient.clone()),
            opacity_stops: vec![
                OpacityStop {
                    offset: 0.0,
                    opacity: 0.65,
                },
                OpacityStop {
                    offset: 1.0,
                    opacity: 0.0,
                },
            ],
        };
        let mut transparent_overlay = overlay.clone();
        for stop in &mut transparent_overlay.opacity_stops {
            stop.opacity = 0.0;
        }
        let paints = vec![
            gradient.clone(),
            gradient.clone(),
            Paint::Layered {
                base: Box::new(gradient.clone()),
                overlays: vec![overlay, transparent_overlay],
            },
            Paint::Solid {
                color: [0.2, 0.3, 0.4],
            },
            gradient,
        ];
        let mut geometry: Vec<_> = (0..paints.len())
            .map(|i| RegionGeometry {
                region: i as u32,
                loops: vec![],
                path_data: format!("M{} 0h24v24h-24Z", i * 32),
                occlusion_path_data: Some(format!(
                    "M{} 0h24v24h-24Z M{} 8h8v8h-8Z",
                    i * 32,
                    i * 32 + 8
                )),
                covered_hole_paths: vec![],
                primitive: None,
            })
            .collect();
        let excluded = [false, false, false, false, true];
        let alpha = crate::face_alpha::FaceAlpha {
            fields: vec![Paint::Solid { color: [0.5; 3] }; paints.len()],
            ink_opacity: 0.5,
            bands: vec![],
            composite_layers: vec![],
            source_fields: vec![],
        };
        for outlined in [false, true] {
            let mut structural = StructuralInk::empty();
            if outlined {
                structural.outlines.push(crate::outline::OutlineBand {
                    outer: "M0 0h160v32h-160Z".into(),
                    inner: "M1 1h158v30H1Z".into(),
                    underpaint: Paint::Solid { color: [0.8; 3] },
                    inner_underpaint: None,
                    patches: vec![],
                    regions: [1, 3].into_iter().collect(),
                    hidden: [3].into_iter().collect(),
                    boundary_underpaint: [1].into_iter().collect(),
                    contour: vec![],
                    width: 1.0,
                    pixels: vec![],
                });
            }
            for alpha in [None, Some(&alpha)] {
                let mut cache = GeometryCache::default();
                let mut serialize = hole_serializer(
                    160,
                    32,
                    &paints,
                    &structural,
                    0.3,
                    &excluded,
                    alpha,
                    &mut cache,
                );
                // Trials change paths, revert an earlier edit, change primitive
                // type and drawing order, and omit geometry entirely.
                for trial in 0..6 {
                    geometry[0].occlusion_path_data = Some(
                        match trial {
                            0 | 3 => "M0 0h24v24H0Z M8 8h8v8H8Z",
                            1 | 4 => "M0 0h24v24H0Z",
                            _ => "",
                        }
                        .into(),
                    );
                    geometry[2].primitive = (trial == 2).then_some(Primitive::Rect {
                        x: 64.0,
                        y: 0.0,
                        width: 24.0,
                        height: 24.0,
                    });
                    let mut g = geometry.clone();
                    if trial == 4 {
                        g.reverse();
                    }
                    if trial == 5 {
                        g.clear();
                    }
                    let expected = serialize_filtered_with_alpha(
                        160,
                        32,
                        &g,
                        &paints,
                        &structural,
                        0.3,
                        true,
                        &excluded,
                        alpha,
                    );
                    let actual = serialize(&g);
                    assert_eq!(actual.0, expected.0, "outlined={outlined}, trial={trial}");
                    assert_eq!(
                        serde_json::to_value(actual.1).unwrap(),
                        serde_json::to_value(expected.1).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn cached_stroke_geometry_and_outlines_keep_cap_and_width_context() {
        let mut cache = GeometryCache::default();
        for path in [
            "M0 0L24 0L24 24Z",
            "M1.25 2.75C13 2.75 13 19 25 19",
            "M0 0L0 0",
        ] {
            for _ in 0..2 {
                assert_eq!(
                    format!("{:?}", cache.optimize(path)),
                    format!("{:?}", optimize_path(path, true, false))
                );
                assert_eq!(
                    format!("{:?}", cache.optimize_stroke(path)),
                    format!("{:?}", optimize_path(path, true, true))
                );
                for width in [0.13, 1.0, 5.75] {
                    for butt in [false, true] {
                        assert_eq!(
                            cache.stroke_outline(path, width, butt),
                            stroke_outline(path, width, butt)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_face_contours_fill_without_closing_oriented_holes() {
        // Two expanded lobes of the same face overlap at x=15..17. The
        // counterclockwise inner loop is a real hole and must stay transparent.
        let geometry = RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: "M2 2H17V30H2Z M15 2H30V30H15Z M6 10V22H12V10Z".into(),
            occlusion_path_data: None,
            covered_hole_paths: vec![],
            primitive: None,
        };
        for opacity in [1.0, 0.5] {
            let alpha = crate::face_alpha::FaceAlpha {
                fields: vec![Paint::Solid {
                    color: [opacity; 3],
                }],
                ink_opacity: 0.0,
                bands: vec![],
                composite_layers: vec![],
                source_fields: vec![],
            };
            let (document, _) = serialize_filtered_with_alpha(
                32,
                32,
                std::slice::from_ref(&geometry),
                &[Paint::Solid {
                    color: [1.0, 0.0, 0.0],
                }],
                &StructuralInk::empty(),
                0.3,
                false,
                &[false],
                (opacity < 1.0).then_some(&alpha),
            );
            let tree =
                resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
            for scale in [1.0, 3.3, 8.0] {
                let size = (32.0_f32 * scale).ceil() as u32;
                let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size).unwrap();
                resvg::render(
                    &tree,
                    resvg::tiny_skia::Transform::from_scale(scale, scale),
                    &mut pixmap.as_mut(),
                );
                for (x, y, filled) in [
                    (16.0, 16.0, true),
                    (24.0, 16.0, true),
                    (9.0, 16.0, false),
                    (0.0, 16.0, false),
                ] {
                    let pixel = pixmap
                        .pixel((x * scale) as u32, (y * scale) as u32)
                        .unwrap();
                    let expected = if filled {
                        (opacity * 255.0).round() as u8
                    } else {
                        0
                    };
                    assert!(
                        pixel.alpha().abs_diff(expected) <= 1,
                        "scale={scale}, opacity={opacity}, x={x}: {pixel:?}"
                    );
                    assert_eq!(pixel.red(), pixel.alpha());
                    assert_eq!(pixel.green(), 0);
                    assert_eq!(pixel.blue(), 0);
                }
            }
        }
    }

    #[test]
    fn zero_alpha_objects_and_overlays_are_not_serialized() {
        let geometry = RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: "M0 0H20V20H0Z".into(),
            occlusion_path_data: None,
            covered_hole_paths: Vec::new(),
            primitive: None,
        };
        let paints = vec![Paint::Layered {
            base: Box::new(Paint::Solid {
                color: [1.0, 0.0, 0.0],
            }),
            overlays: vec![PaintOverlay {
                paint: Box::new(Paint::Solid {
                    color: [0.0, 0.0, 1.0],
                }),
                opacity_stops: vec![
                    OpacityStop {
                        offset: 0.0,
                        opacity: 0.0,
                    },
                    OpacityStop {
                        offset: 1.0,
                        opacity: 0.0,
                    },
                ],
            }],
        }];
        let mask = crate::face_alpha::FaceAlpha {
            fields: vec![Paint::Solid { color: [0.0; 3] }],
            ink_opacity: 0.0,
            bands: vec![],
            composite_layers: vec![],
            source_fields: vec![],
        };
        let (transparent, report) = serialize_filtered_with_alpha(
            24,
            24,
            std::slice::from_ref(&geometry),
            &paints,
            &StructuralInk::empty(),
            0.0,
            false,
            &[false],
            Some(&mask),
        );
        assert!(!transparent.contains("<path"));
        assert_eq!(report.path_elements, 0);
        let (opaque, report) = serialize_filtered_with_alpha(
            24,
            24,
            &[geometry],
            &paints,
            &StructuralInk::empty(),
            0.0,
            false,
            &[false],
            None,
        );
        assert_eq!(report.path_elements + report.rect_elements, 1);
        assert!(!opaque.contains("#0000ff"));
    }

    #[test]
    fn authored_alpha_boundary_does_not_add_a_bright_fringe() {
        for reversed in [false, true] {
            let mut geometries = vec![
                RegionGeometry {
                    region: 0,
                    loops: vec![],
                    path_data: "M0 0H20V20H0Z".into(),
                    occlusion_path_data: None,
                    covered_hole_paths: Vec::new(),
                    primitive: None,
                },
                RegionGeometry {
                    region: 1,
                    loops: vec![],
                    path_data: "M20 0H40V20H20Z".into(),
                    occlusion_path_data: None,
                    covered_hole_paths: Vec::new(),
                    primitive: None,
                },
            ];
            if reversed {
                geometries.reverse();
            }
            let alpha = crate::face_alpha::FaceAlpha {
                fields: vec![
                    Paint::Solid {
                        color: [170.0 / 255.0; 3],
                    },
                    Paint::Solid {
                        color: [204.0 / 255.0; 3],
                    },
                ],
                ink_opacity: 0.0,
                bands: vec![],
                composite_layers: vec![],
                source_fields: vec![],
            };
            let (document, _) = serialize_filtered_with_alpha(
                40,
                20,
                &geometries,
                &[
                    Paint::Solid { color: [1.0; 3] },
                    Paint::Solid {
                        color: [213.0 / 255.0; 3],
                    },
                ],
                &StructuralInk::empty(),
                0.0,
                false,
                &[false, false],
                Some(&alpha),
            );
            assert!(!document.contains("<mask"));
            assert!(document.contains("fill-opacity="));
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
        let mut definitions = Elements::new();
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
        let mut body = Elements::new();
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
    fn batch_plan_reuses_decisions_but_keeps_current_paths_and_paint_barriers() {
        let mut cache = GeometryCache::default();
        for version in [0, 1, 2, 3, 4, 0, 1] {
            let mut elements = vec![
                path("M0 0H4V4H0Z", (0.0, 0.0, 4.0, 4.0), "red"),
                path("M20 0H24V4H20Z", (20.0, 0.0, 24.0, 4.0), "blue"),
                path(
                    if version == 1 {
                        "M40 0H44V4H40Z M41 1V2H42V1Z"
                    } else {
                        "M40 0H44V4H40Z"
                    },
                    (40.0, 0.0, 44.0, 4.0),
                    "red",
                ),
            ];
            if version == 2 {
                elements[1].as_mut().unwrap().batchable = false;
            }
            if version == 3 {
                elements[2].as_mut().unwrap().attributes = attrs([("fill", "blue".into())]);
            }
            if version == 4 {
                elements[1].as_mut().unwrap().geometry = OptimizedElement::Rect {
                    x: 20.0,
                    y: 0.0,
                    width: 4.0,
                    height: 4.0,
                };
            }
            let mut reference = elements.clone();
            let mut expected = SvgSummary::default();
            let mut actual = SvgSummary::default();
            batch_equal_paint_paths(&mut reference, &mut expected);
            cache.batch(&mut elements, &mut actual);
            let mut a = Elements::new();
            let mut b = Elements::new();
            write_paint_elements(&mut a, &reference, &mut expected);
            write_paint_elements(&mut b, &elements, &mut actual);
            assert_eq!(a, b);
            assert_eq!(
                serde_json::to_value(expected).unwrap(),
                serde_json::to_value(actual).unwrap()
            );
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
        let mut definitions = Elements::new();
        let mut summary = SvgSummary::default();
        register_gradient(
            &overlay.paint,
            Some(&overlay.opacity_stops),
            overlay_key(&overlay),
            &mut ids,
            &mut definitions,
            &mut summary,
        );
        assert!(definitions.to_string().contains("stop-opacity=\"0.7\""));
        assert!(definitions.to_string().contains("stop-opacity=\"0\""));

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
