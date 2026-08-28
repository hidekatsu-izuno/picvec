use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::color::rgb_hex;
use crate::geometry::{open_path_data, Primitive, RegionGeometry};
use crate::gradient::{ColorStop, Paint};
use crate::optimize::{format_number, optimize_path, separated_bboxes, OptimizedElement};
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
    pub gradient_stops: usize,
    pub linear_cubics_to_lines: usize,
    pub redundant_segments_removed: usize,
    pub arc_segments: usize,
    pub merged_arc_segments: usize,
    pub paint_paths_merged: usize,
    pub paint_batches: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug)]
struct PaintElement {
    geometry: OptimizedElement,
    attributes: String,
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
            center,
            radius,
            stops,
            ..
        } => Some(format!(
            "R:{},{},{},{}:{}",
            number(center.x),
            number(center.y),
            number(radius.x),
            number(radius.y),
            stops_key(stops)
        )),
    }
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

fn fill_value(paint: &Paint, gradient_ids: &HashMap<String, String>) -> String {
    match paint {
        Paint::Solid { color } => rgb_hex(*color),
        _ => format!("url(#{})", gradient_ids[&paint_key(paint).unwrap()]),
    }
}

fn paint_attributes(fill: &str, overlap: f32) -> String {
    if overlap <= 0.0 {
        format!("fill=\"{}\"", fill)
    } else {
        format!(
            "fill=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\" paint-order=\"stroke fill\"",
            fill, fill, number(overlap * 2.0)
        )
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
        let signature = element.attributes.clone();
        let next_signature = signature_ids.len();
        let signature_id = *signature_ids.entry(signature).or_insert(next_signature);
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

/// Serialize the complete editable document.  Gradients are restricted to
/// solid, axial linear, and elliptical radial forms with at most five stops,
/// which stay within the Office object import subset.
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
    let mut gradient_ids = HashMap::<String, String>::new();
    let mut definitions = String::new();
    let mut summary = SvgSummary::default();
    for paint in paints {
        let Some(key) = paint_key(paint) else {
            continue;
        };
        if gradient_ids.contains_key(&key) {
            continue;
        }
        let id = format!("paint-{}", gradient_ids.len());
        match paint {
            Paint::Linear {
                start, end, stops, ..
            } => {
                let _ = write!(definitions, "<linearGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\">{}</linearGradient>", id, number(start.x), number(start.y), number(end.x), number(end.y), stop_elements(stops));
                summary.linear_gradients += 1;
                summary.gradient_stops += stops.len();
            }
            Paint::Radial {
                center,
                radius,
                stops,
                ..
            } => {
                let _ = write!(definitions, "<radialGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" cx=\"0\" cy=\"0\" r=\"1\" gradientTransform=\"translate({} {}) scale({} {})\">{}</radialGradient>", id, number(center.x), number(center.y), number(radius.x.max(0.001)), number(radius.y.max(0.001)), stop_elements(stops));
                summary.radial_gradients += 1;
                summary.gradient_stops += stops.len();
            }
            Paint::Solid { .. } => unreachable!(),
        }
        gradient_ids.insert(key, id);
    }
    let mut paint_elements = Vec::<Option<PaintElement>>::with_capacity(geometries.len());
    for geometry in geometries {
        let paint = &paints[geometry.region as usize];
        let fill = fill_value(paint, &gradient_ids);
        let attributes = paint_attributes(&fill, paint_overlap);
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
                paint_elements.push(Some(PaintElement {
                    geometry: OptimizedElement::Path {
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
                    },
                    attributes,
                }));
                summary.ellipse_elements += 1;
                continue;
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
                let Some((optimized, operations)) = optimize_path(path_data, true, false) else {
                    paint_elements.push(Some(PaintElement {
                        geometry: OptimizedElement::Path {
                            data: path_data.to_string(),
                            bbox: None,
                        },
                        attributes,
                    }));
                    continue;
                };
                summary.linear_cubics_to_lines += operations.linear_cubics;
                summary.redundant_segments_removed += operations.redundant_segments;
                summary.arc_segments += operations.arc_segments;
                summary.merged_arc_segments += operations.merged_arcs;
                optimized
            }
        };
        paint_elements.push(Some(PaintElement {
            geometry: optimized,
            attributes,
        }));
    }
    batch_equal_paint_paths(&mut paint_elements, &mut summary);
    let mut body = String::new();
    body.push_str("<g id=\"paint-layer\" fill-rule=\"evenodd\">");
    for element in paint_elements.into_iter().flatten() {
        let kind = write_geometry(&mut body, &element.geometry, &element.attributes);
        count_element(&mut summary, kind);
    }
    body.push_str("</g>");
    body.push_str("<g id=\"structural-ink-layer\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">");
    for stroke in &structural.strokes {
        let data = stroke
            .path_data
            .clone()
            .unwrap_or_else(|| open_path_data(&stroke.points));
        if data.is_empty() {
            continue;
        }
        let attributes = format!(
            "data-structural-ink=\"line\" stroke=\"{}\" stroke-width=\"{}\"",
            rgb_hex(stroke.color),
            number(stroke.width)
        );
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
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><defs>{}</defs>{}</svg>\n",
        width, height, width, height, definitions, body
    );
    fs::write(output, document.as_bytes())?;
    summary.bytes = document.len();
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
        })
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
}
