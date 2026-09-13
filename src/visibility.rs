//! Remove draw operations only when their omission leaves the final RGBA
//! raster unchanged at both source scale and 4x. Work on spatial tiles, parse
//! once, and stop at the first changed tile rather than rerendering the image
//! once per object. Partial occlusion is deliberately left alone.
use resvg::{
    tiny_skia::{Pixmap, Transform},
    usvg::{self, Node},
};
use std::collections::HashMap;

const TILE: usize = 64;
const PREFIX: &str = "picvec-visibility-";

#[derive(Default, Debug)]
pub(crate) struct Removed {
    pub shapes: usize,
    pub strokes: usize,
    pub ink: usize,
    pub paths: usize,
    pub rects: usize,
    pub circles: usize,
    pub ellipses: usize,
    pub lines: usize,
}

struct Candidate {
    start: usize,
    end: usize,
    stroke: bool,
    ink: bool,
}

// This is intentionally a scanner for our own serializer's self-closing
// geometry, not a general-purpose XML editor. Definitions/referenced objects
// and complex compositing groups are never independently removed.
fn annotate(document: &str) -> (String, Vec<Candidate>) {
    let mut result = String::new();
    let mut candidates = Vec::new();
    let mut cursor = 0;
    let mut definitions = false;
    while let Some(offset) = document[cursor..].find('<') {
        let start = cursor + offset;
        let Some(end) = document[start..].find('>').map(|n| start + n + 1) else {
            break;
        };
        let tag = &document[start..end];
        result.push_str(&document[cursor..start]);
        if tag.starts_with("<defs") {
            definitions = true;
        }
        let geometry = [
            "<path ",
            "<rect ",
            "<circle ",
            "<ellipse ",
            "<line ",
            "<polygon ",
            "<polyline ",
        ]
        .iter()
        .any(|p| tag.starts_with(p));
        if !definitions && geometry && tag.ends_with("/>") && !tag.contains(" id=") {
            let id = candidates.len();
            result.push_str(&format!("<g id=\"{PREFIX}{id}\">{tag}</g>"));
            candidates.push(Candidate {
                start,
                end,
                stroke: tag.contains("fill=\"none\""),
                ink: tag.contains("data-structural-ink="),
            });
        } else {
            result.push_str(tag);
        }
        if tag.starts_with("</defs") {
            definitions = false;
        }
        cursor = end;
    }
    result.push_str(&document[cursor..]);
    (result, candidates)
}

struct Draw<'a> {
    node: &'a Node,
    candidate: Option<usize>,
    bounds: [f32; 4],
}
fn collect<'a>(group: &'a usvg::Group, draws: &mut Vec<Draw<'a>>) {
    for node in group.children() {
        let candidate = node.id().strip_prefix(PREFIX).and_then(|s| s.parse().ok());
        if let Node::Group(g) = node {
            if candidate.is_none() && g.transform().is_identity() && !g.should_isolate() {
                collect(g, draws);
                continue;
            }
        }
        let Some(b) = node.abs_layer_bounding_box() else {
            continue;
        };
        draws.push(Draw {
            node,
            candidate,
            bounds: [
                b.left() - 1.0,
                b.top() - 1.0,
                b.right() + 1.0,
                b.bottom() + 1.0,
            ],
        });
    }
}

fn draw_tile(
    draws: &[Draw<'_>],
    members: &[usize],
    active: &[bool],
    skip: Option<usize>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    scale: usize,
) -> Pixmap {
    let mut pixmap = Pixmap::new((w * scale) as u32, (h * scale) as u32).unwrap();
    for &i in members {
        if !active[i] || Some(i) == skip {
            continue;
        }
        let node = draws[i].node;
        let b = node.abs_layer_bounding_box().unwrap();
        // render_node subtracts its absolute bounding-box origin internally.
        let transform = Transform::from_scale(scale as f32, scale as f32)
            .pre_translate(b.x() - x as f32, b.y() - y as f32);
        resvg::render_node(node, transform, &mut pixmap.as_mut());
    }
    pixmap
}

pub(crate) fn prune(document: &str, width: usize, height: usize) -> (String, Removed) {
    // A <use> can render the same source element again through a filter or
    // transform. Removing its source would alter both instances; do not treat
    // those instances as independently removable draw operations.
    if document.contains("<use ") {
        return (document.into(), Removed::default());
    }
    let (annotated, candidates) = annotate(document);
    if candidates.is_empty() {
        return (document.into(), Removed::default());
    }
    let Ok(tree) = usvg::Tree::from_str(&annotated, &usvg::Options::default()) else {
        return (document.into(), Removed::default());
    };
    // The core serializer has no viewport transform. Leave other documents
    // intact rather than detaching children from their inherited transform.
    if !tree.root().transform().is_identity() {
        return (document.into(), Removed::default());
    }
    let mut draws = Vec::new();
    collect(tree.root(), &mut draws);
    let columns = width.div_ceil(TILE);
    let rows = height.div_ceil(TILE);
    let mut tiles = vec![Vec::new(); columns * rows];
    let mut memberships = vec![Vec::new(); draws.len()];
    for (i, draw) in draws.iter().enumerate() {
        let [left, top, right, bottom] = draw.bounds;
        let x0 = (left.max(0.0) as usize / TILE).min(columns);
        let y0 = (top.max(0.0) as usize / TILE).min(rows);
        let x1 = ((right.max(0.0).ceil() as usize).div_ceil(TILE)).min(columns);
        let y1 = ((bottom.max(0.0).ceil() as usize).div_ceil(TILE)).min(rows);
        for y in y0..y1 {
            for x in x0..x1 {
                let tile = y * columns + x;
                tiles[tile].push(i);
                memberships[i].push(tile);
            }
        }
    }
    let mut active = vec![true; draws.len()];
    let mut removed: Vec<_> = (0..candidates.len())
        .map(|i| tree.node_by_id(&format!("{PREFIX}{i}")).is_none())
        .collect();
    let mut baseline = HashMap::<(usize, usize), Pixmap>::new();
    for (i, draw) in draws.iter().enumerate() {
        let Some(candidate) = draw.candidate else {
            continue;
        };
        let mut unchanged = true;
        'scales: for scale in [1, 4] {
            for &tile in &memberships[i] {
                let x = tile % columns * TILE;
                let y = tile / columns * TILE;
                let w = TILE.min(width - x);
                let h = TILE.min(height - y);
                // Bound the scratch cache independently of source dimensions.
                if baseline.len() >= 128 {
                    baseline.clear();
                }
                let before = baseline.entry((tile, scale)).or_insert_with(|| {
                    draw_tile(&draws, &tiles[tile], &active, None, x, y, w, h, scale)
                });
                let after = draw_tile(&draws, &tiles[tile], &active, Some(i), x, y, w, h, scale);
                if before.data() != after.data() {
                    unchanged = false;
                    break 'scales;
                }
            }
        }
        if unchanged {
            active[i] = false;
            removed[candidate] = true;
        }
    }
    let mut output = String::new();
    let mut cursor = 0;
    let mut report = Removed::default();
    for (candidate, remove) in candidates.iter().zip(removed) {
        if !remove {
            continue;
        }
        output.push_str(&document[cursor..candidate.start]);
        cursor = candidate.end;
        let tag = &document[candidate.start..candidate.end];
        report.paths += usize::from(tag.starts_with("<path "));
        report.rects += usize::from(tag.starts_with("<rect "));
        report.circles += usize::from(tag.starts_with("<circle "));
        report.ellipses += usize::from(tag.starts_with("<ellipse "));
        report.lines += usize::from(tag.starts_with("<line "));
        report.shapes += 1;
        report.strokes += usize::from(candidate.stroke);
        report.ink += usize::from(candidate.ink);
    }
    output.push_str(&document[cursor..]);
    (output, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(svg: &str, scale: u32) -> Vec<u8> {
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let mut p = Pixmap::new(
            tree.size().width() as u32 * scale,
            tree.size().height() as u32 * scale,
        )
        .unwrap();
        resvg::render(
            &tree,
            Transform::from_scale(scale as f32, scale as f32),
            &mut p.as_mut(),
        );
        p.take()
    }
    #[test]
    fn removes_hidden_faces_and_redundant_lines_but_keeps_partial_overlap() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><g fill="red"><rect x="5" y="5" width="5" height="5"/><rect width="20" height="20"/><path d="M5 5L15 15" fill="none" stroke="red" stroke-width="2"/><rect x="15" y="15" width="10" height="10" fill="blue"/></g></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 2);
        assert_eq!(report.strokes, 1);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn tile_boundaries_and_multiple_removals_preserve_the_complete_render() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="32"><rect width="256" height="32" fill="red"/><rect x="60.2" y="4" width="120" height="16" fill="red"/><path d="M10 15.3H200" fill="none" stroke="red" stroke-width="2"/><rect x="128" y="16" width="80" height="16" fill="blue"/></svg>"##;
        let (output, report) = prune(svg, 256, 32);
        assert_eq!(report.shapes, 2);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn zero_area_paint_is_removed_but_its_stroked_counterpart_is_kept() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M2 2H20" fill="red"/><path d="M2 2H20" fill="none" stroke="red"/></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 1);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
    #[test]
    fn preserves_alpha_accumulation_gradients_and_subpixel_lines() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect x="10" width="10" height="25" fill="url(#g)"/><path d="M1 1L30 30" fill="none" stroke="black" stroke-width="0.1"/></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(output, svg);
    }
    #[test]
    fn preserves_clip_context_and_referenced_geometry() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><clipPath id="c"><rect width="10" height="10"/></clipPath></defs><rect id="base" width="32" height="32" fill="red"/><g clip-path="url(#c)"><rect width="32" height="32" fill="blue"/></g></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
}
