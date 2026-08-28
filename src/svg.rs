use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::color::rgb_hex;
use crate::config::Config;
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

fn batch_equal_paint_paths(elements: &mut [Option<PaintElement>], summary: &mut SvgSummary) {
    let mut index = 0_usize;
    while index < elements.len() {
        let Some(first) = elements[index].as_ref() else {
            index += 1;
            continue;
        };
        if !matches!(first.geometry, OptimizedElement::Path { .. }) {
            index += 1;
            continue;
        }
        let signature = first.attributes.clone();
        let mut cursor = index + 1;
        while cursor < elements.len() {
            let Some(candidate) = elements[cursor].as_ref() else {
                break;
            };
            if candidate.attributes != signature
                || !matches!(candidate.geometry, OptimizedElement::Path { .. })
            {
                break;
            }
            cursor += 1;
        }
        let mut bins = Vec::<Vec<(usize, (f64, f64, f64, f64))>>::new();
        for current in index..cursor {
            let Some(bbox) = elements[current].as_ref().and_then(paint_path_bbox) else {
                continue;
            };
            if let Some(batch) = bins.iter_mut().find(|batch| {
                batch
                    .iter()
                    .all(|(_, previous)| separated_bboxes(bbox, *previous, 1.0))
            }) {
                batch.push((current, bbox));
            } else {
                bins.push(vec![(current, bbox)]);
            }
        }
        for batch in bins.into_iter().filter(|batch| batch.len() > 1) {
            let target = batch[0].0;
            let merged_data = batch
                .iter()
                .filter_map(|(element_index, _)| elements[*element_index].as_ref())
                .filter_map(|element| match &element.geometry {
                    OptimizedElement::Path { data, .. } => Some(data.trim()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            if let Some(element) = elements[target].as_mut() {
                element.geometry = OptimizedElement::Path {
                    data: merged_data,
                    bbox: None,
                };
            }
            for (element_index, _) in batch.iter().skip(1) {
                elements[*element_index] = None;
            }
            summary.paint_paths_merged += batch.len() - 1;
            summary.paint_batches += 1;
        }
        index = cursor;
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
pub fn write(
    output: &Path,
    width: usize,
    height: usize,
    geometries: &[RegionGeometry],
    paints: &[Paint],
    structural: &StructuralInk,
    config: &Config,
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
        let attributes = paint_attributes(&fill, config.shared_boundary_overlap);
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
                if geometry.path_data.is_empty() {
                    continue;
                }
                let Some((optimized, operations)) = optimize_path(&geometry.path_data, true, false)
                else {
                    paint_elements.push(Some(PaintElement {
                        geometry: OptimizedElement::Path {
                            data: geometry.path_data.clone(),
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
