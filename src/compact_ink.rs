//! Source-supported local replacement of thin two-colour shapes.
//! The ordinary vectorizer always sees the unmodified source. Each replacement
//! is checked against that final rendering; pixels elsewhere are untouched.
use crate::adaptive::SourceRect;
use crate::chroma::AlphaMatte;
use crate::geometry::fitted_colour_contour_with_smoothing;
use crate::raster::{RasterSource, SourceRaster};
use crate::svg_document::{attrs, Document, Element, Elements};

#[derive(Clone)]
struct Region {
    rect: SourceRect,
    foreground: [f32; 3],
    background: [f32; 3],
}

fn overlaps(a: SourceRect, b: SourceRect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

fn intensity(rgb: [f32; 3]) -> f32 {
    (rgb[0] + rgb[1] + rgb[2]) / 3.0
}

fn extrema(values: &[f32], w: usize, h: usize, maximum: bool) -> Vec<f32> {
    let mut horizontal = Vec::with_capacity(values.len());
    for row in values.chunks_exact(w) {
        horizontal.extend(crate::extrema::sliding(row, 5, maximum, false));
    }
    let mut output = vec![0.0; values.len()];
    for x in 0..w {
        let column: Vec<_> = (0..h).map(|y| horizontal[y * w + x]).collect();
        for (y, v) in crate::extrema::sliding(&column, 5, maximum, false)
            .into_iter()
            .enumerate()
        {
            output[y * w + x] = v;
        }
    }
    output
}

fn median(samples: &[[f32; 3]]) -> [f32; 3] {
    std::array::from_fn(|c| {
        let mut values: Vec<_> = samples.iter().map(|p| p[c]).collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    })
}

fn hex(rgb: [f32; 3]) -> String {
    let [r, g, b] = rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn regions(source: &SourceRaster) -> Vec<Region> {
    let (w, h) = (source.width, source.height);
    if w < 8 || h < 8 {
        return Vec::new();
    }
    let values: Vec<_> = (0..w * h)
        .map(|i| intensity(source.get(i % w, i / w)))
        .collect();
    let mut candidates = Vec::new();
    for dark in [true, false] {
        let limit = extrema(&values, w, h, dark);
        let support: Vec<_> = values
            .iter()
            .zip(limit)
            .map(|(&v, b)| if dark { b - v > 0.22 } else { v - b > 0.22 })
            .collect();
        let mut labels = vec![u32::MAX; w * h];
        let mut components = Vec::<Vec<usize>>::new();
        for seed in 0..labels.len() {
            if !support[seed] || labels[seed] != u32::MAX {
                continue;
            }
            let id = components.len() as u32;
            let mut pending = vec![seed];
            let mut pixels = Vec::new();
            labels[seed] = id;
            while let Some(i) = pending.pop() {
                // Mark the entire component, but retain at most one more
                // sample than the admissible size for photographic regions.
                if pixels.len() <= 2048 {
                    pixels.push(i);
                }
                let (x, y) = (i % w, i / w);
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let j = yy * w + xx;
                        if support[j] && labels[j] == u32::MAX {
                            labels[j] = id;
                            pending.push(j);
                        }
                    }
                }
            }
            components.push(pixels);
        }
        for (id, pixels) in components.iter().enumerate() {
            let id = id as u32;
            if pixels.len() < 4 || pixels.len() > 2048 {
                continue;
            }
            let left = pixels.iter().map(|i| i % w).min().unwrap();
            let top = pixels.iter().map(|i| i / w).min().unwrap();
            let right = pixels.iter().map(|i| i % w).max().unwrap() + 1;
            let bottom = pixels.iter().map(|i| i / w).max().unwrap() + 1;
            // Adjacent small marks can touch in a scan. Their union may be
            // wide even though each stroke is narrow; bound both total ink
            // and the short side instead of cutting such a contour apart.
            if (right - left).max(bottom - top) > 256
                || (right - left).min(bottom - top) > 64
                || left < 3
                || top < 3
                || right + 3 > w
                || bottom + 3 > h
            {
                continue;
            }
            let perimeter: usize = pixels
                .iter()
                .map(|&i| {
                    [i - 1, i + 1, i - w, i + w]
                        .iter()
                        .filter(|&&j| labels[j] != id)
                        .count()
                })
                .sum();
            if 2 * pixels.len() > 5 * perimeter {
                continue;
            }
            let (x0, y0) = (left - 2, top - 2);
            let (cw, ch) = (right - left + 4, bottom - top + 4);
            let mut halo = Vec::new();
            for y in y0..y0 + ch {
                for x in x0..x0 + cw {
                    if x == x0 || y == y0 || x + 1 == x0 + cw || y + 1 == y0 + ch {
                        let beside_other = (y - 1..=y + 1).any(|yy| {
                            (x - 1..=x + 1).any(|xx| {
                                labels[yy * w + xx] != u32::MAX && labels[yy * w + xx] != id
                            })
                        });
                        if labels[y * w + x] == u32::MAX && !beside_other {
                            halo.push(source.get(x, y));
                        }
                    }
                }
            }
            if halo.len() < 8 {
                continue;
            }
            let background = median(&halo);
            if halo
                .iter()
                .filter(|p| (0..3).any(|c| (p[c] - background[c]).abs() > 0.06))
                .count()
                * 10
                > halo.len()
            {
                continue;
            }
            let mut ink: Vec<_> = pixels.iter().map(|&i| source.get(i % w, i / w)).collect();
            ink.sort_by(|a, b| {
                if dark {
                    intensity(*a).total_cmp(&intensity(*b))
                } else {
                    intensity(*b).total_cmp(&intensity(*a))
                }
            });
            let foreground = median(&ink[..ink.len().div_ceil(4)]);
            let axis = std::array::from_fn::<_, 3, _>(|c| foreground[c] - background[c]);
            let norm = axis.iter().map(|v| v * v).sum::<f32>();
            if norm < 0.25 {
                continue;
            }
            candidates.push(Region {
                rect: SourceRect {
                    x: left.saturating_sub(4),
                    y: top.saturating_sub(4),
                    width: (right + 4).min(w) - left.saturating_sub(4),
                    height: (bottom + 4).min(h) - top.saturating_sub(4),
                },
                foreground,
                background,
            });
        }
    }
    // Keep individual supports. Joining overlapping boxes here can turn a
    // paragraph into an oversized component and lose all of its small glyphs.
    // Alternatives are allowed to overlap until their actual gain is known.
    let mut groups: Vec<Region> = Vec::new();
    let mut ordered = candidates.clone();
    ordered.sort_by_key(|r| std::cmp::Reverse(r.rect.area()));
    for mut r in ordered {
        let mut i = 0;
        while i < groups.len() {
            let b = groups[i].rect;
            if !overlaps(r.rect, b) {
                i += 1;
                continue;
            }
            let x = r.rect.x.min(b.x);
            let y = r.rect.y.min(b.y);
            let right = (r.rect.x + r.rect.width).max(b.x + b.width);
            let bottom = (r.rect.y + r.rect.height).max(b.y + b.height);
            let merged = SourceRect {
                x,
                y,
                width: right - x,
                height: bottom - y,
            };
            // Keep the original alternatives when a join becomes too large.
            // This permits adjacent letters without swallowing a paragraph.
            if merged.width.max(merged.height) > 512 || merged.width.min(merged.height) > 64 {
                i += 1;
                continue;
            }
            let other = groups.remove(i);
            if other.rect.area() > r.rect.area() {
                r.foreground = other.foreground;
                r.background = other.background;
            }
            r.rect = merged;
            i = 0;
        }
        groups.push(r);
    }
    for group in groups {
        if !candidates.iter().any(|r| {
            r.rect == group.rect
                && r.foreground == group.foreground
                && r.background == group.background
        }) {
            candidates.push(group);
        }
    }
    candidates
}

struct Patch {
    rect: SourceRect,
    document: Document,
    pixels: Vec<[f32; 4]>,
}

fn render(document: &Document, width: usize, height: usize) -> Option<resvg::tiny_skia::Pixmap> {
    let tree = resvg::usvg::Tree::from_str(document, &resvg::usvg::Options::default()).ok()?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width as u32, height as u32)?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(
            width as f32 / tree.size().width(),
            height as f32 / tree.size().height(),
        ),
        &mut pixmap.as_mut(),
    );
    Some(pixmap)
}

fn patch(source: &SourceRaster, region: &Region) -> Option<Patch> {
    let rect = region.rect;
    let (w, h) = (rect.width, rect.height);
    let bg = region.background;
    let fg = region.foreground;
    let axis: [f32; 3] = std::array::from_fn(|c| fg[c] - bg[c]);
    let norm = axis.iter().map(|v| v * v).sum::<f32>();
    let mut coverage = Vec::with_capacity(w * h);
    let mut residuals = Vec::with_capacity(w * h);
    let mut perimeter_bad = 0;
    for y in 0..h {
        for x in 0..w {
            let p = source.get(rect.x + x, rect.y + y);
            let a = ((0..3).map(|c| (p[c] - bg[c]) * axis[c]).sum::<f32>() / norm).clamp(0.0, 1.0);
            let residual = (0..3)
                .map(|c| (p[c] - bg[c] - a * axis[c]).abs())
                .fold(0.0_f32, f32::max);
            let amount = (0..3).map(|c| (p[c] - bg[c]) * axis[c]).sum::<f32>() / norm;
            if (0..3).any(|c| (p[c] - bg[c] - amount * axis[c]).abs() > 0.1) {
                return None;
            }
            residuals.push(residual);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                perimeter_bad += usize::from(residual > 0.06);
            }
            coverage.push(a);
        }
    }
    residuals.sort_by(f32::total_cmp);
    // A complete crop must be explained, including neighbouring objects and
    // the joining boundary. A good fit to one glyph alone is insufficient.
    if perimeter_bad > (w + h) / 20 {
        return None;
    }
    if residuals[residuals.len() * 9 / 10] > 0.04 {
        return None;
    }
    let mut area = 0;
    let mut perimeter = 0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if coverage[i] >= 0.5 {
                area += 1;
                perimeter += [i - 1, i + 1, i - w, i + w]
                    .iter()
                    .filter(|&&j| coverage[j] < 0.5)
                    .count();
            }
        }
    }
    if area * 2 > perimeter * 5 {
        return None;
    }
    let mass = coverage.iter().sum::<f32>();
    if mass < 3.0 {
        return None;
    }
    let matte = AlphaMatte::from_u8(
        w,
        h,
        coverage.iter().map(|v| (v * 255.0).round() as u8).collect(),
    );
    let contours = matte.isocontours(0.5);
    if contours.is_empty() {
        return None;
    }
    for sigma in [0.35, 0.1, 0.0] {
        let path: String = contours
            .iter()
            .map(|p| fitted_colour_contour_with_smoothing(p, sigma))
            .collect();
        let mut body = Elements::new();
        body.leaf(
            "path",
            attrs([
                ("d", path.clone()),
                ("fill", "white".into()),
                ("fill-rule", "evenodd".into()),
            ]),
        );
        let silhouette = Document::from_parts(w, h, Elements::new(), body);
        let pixmap = render(&silhouette, w, h)?;
        let rendered: Vec<f32> = pixmap
            .pixels()
            .iter()
            .map(|p| p.alpha() as f32 / 255.0)
            .collect();
        let error = coverage
            .iter()
            .zip(&rendered)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        if error > mass * 0.25
            || coverage
                .iter()
                .zip(&rendered)
                .any(|(&a, &b)| (a < 0.15 && b > 0.65) || (a > 0.85 && b < 0.35))
        {
            continue;
        }
        let mut body = Elements::new();
        body.leaf(
            "rect",
            attrs([
                ("width", w.to_string()),
                ("height", h.to_string()),
                ("fill", hex(bg)),
            ]),
        );
        body.leaf(
            "path",
            attrs([
                ("d", path),
                ("fill", hex(fg)),
                ("fill-rule", "evenodd".into()),
            ]),
        );
        let document = Document::from_parts(w, h, Elements::new(), body);
        let pixmap = render(&document, w, h)?;
        let pixels = pixmap
            .pixels()
            .iter()
            .map(|p| {
                [
                    p.red() as f32 / 255.0,
                    p.green() as f32 / 255.0,
                    p.blue() as f32 / 255.0,
                    p.alpha() as f32 / 255.0,
                ]
            })
            .collect();
        return Some(Patch {
            rect,
            document,
            pixels,
        });
    }
    None
}

fn improvement(source: &SourceRaster, baseline: &resvg::tiny_skia::Pixmap, patch: &Patch) -> f32 {
    let rect = patch.rect;
    let mut old_error = 0.0;
    let mut new_error = 0.0;
    let mut old_edge = 0.0;
    let mut new_edge = 0.0;
    let mut edge_count = 0;
    // Validate small tiles too: an improvement in a dark stroke must not pay
    // for erasing a counter, changing a nearby mark, or moving the join.
    for ty in (0..rect.height).step_by(8) {
        for tx in (0..rect.width).step_by(8) {
            let mut old_tile = 0.0;
            let mut new_tile = 0.0;
            for y in ty..(ty + 8).min(rect.height) {
                for x in tx..(tx + 8).min(rect.width) {
                    let source_rgb = source.get(rect.x + x, rect.y + y);
                    let old = baseline.pixels()[(rect.y + y) * source.width + rect.x + x];
                    let old = [
                        old.red() as f32 / 255.0,
                        old.green() as f32 / 255.0,
                        old.blue() as f32 / 255.0,
                        old.alpha() as f32 / 255.0,
                    ];
                    let new = patch.pixels[y * rect.width + x];
                    // Premultiplied RGB and alpha jointly account for either
                    // black or white backing, even for an opaque input.
                    let a = (0..3).map(|c| (old[c] - source_rgb[c]).abs()).sum::<f32>() / 3.0
                        + (1.0 - old[3]);
                    let b = (0..3).map(|c| (new[c] - source_rgb[c]).abs()).sum::<f32>() / 3.0
                        + (1.0 - new[3]);
                    // Do not trade an already-correct fine mark or counter
                    // for an improvement elsewhere in the same tile.
                    if a < 0.04 && b > 0.45 {
                        return 0.0;
                    }
                    old_tile += a;
                    new_tile += b;
                    if x == 0 || y == 0 || x + 1 == rect.width || y + 1 == rect.height {
                        old_edge += a;
                        new_edge += b;
                        edge_count += 1;
                    }
                }
            }
            if new_tile > old_tile * 1.03 + 0.64 {
                return 0.0;
            }
            old_error += old_tile;
            new_error += new_tile;
        }
    }
    if new_error < old_error * 0.8
        && old_error - new_error > 0.5
        && old_edge / (edge_count as f32) < 0.08
        && new_edge <= old_edge + edge_count as f32 * 0.015
    {
        old_error - new_error
    } else {
        0.0
    }
}

fn proposals(source: &SourceRaster) -> Vec<Patch> {
    let mut output = Vec::new();
    for seed in regions(source) {
        for margin in [0, 4, 8, 16] {
            let mut region = seed.clone();
            region.rect = region.rect.expanded(margin, source.width, source.height);
            if region.rect.width.max(region.rect.height) > 512
                || region.rect.width.min(region.rect.height) > 96
            {
                continue;
            }
            if let Some(p) = patch(source, &region) {
                output.push(p);
            }
        }
    }
    output
}

fn select<'a>(
    source: &SourceRaster,
    baseline: &resvg::tiny_skia::Pixmap,
    patches: &'a [Patch],
) -> Vec<&'a Patch> {
    let mut scored: Vec<_> = patches
        .iter()
        .filter_map(|p| {
            let gain = improvement(source, baseline, p);
            (gain > 0.0).then_some((gain - p.rect.area() as f32 * 0.001, p))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.rect.area().cmp(&b.1.rect.area()))
    });
    let mut selected: Vec<&Patch> = Vec::new();
    for (_, p) in scored {
        if !selected.iter().any(|a| overlaps(a.rect, p.rect)) {
            selected.push(p);
        }
    }
    selected
}

/// Refine independently of global segmentation and only after comparing with
/// its final SVG. No OCR, source edits, masks, or opacity clips are involved.
pub(crate) fn refine(
    source: &SourceRaster,
    document: &mut Document,
    dimensions: (usize, usize),
) -> usize {
    let patches = proposals(source);
    if patches.is_empty() {
        return 0;
    }
    let mut count = 0;
    // A repaired join can make a neighbouring crop safe. Reuse the source
    // proposals, but compare against the actual updated SVG for each pass.
    // Two passes bound runtime and avoid repeatedly fitting the same source.
    for _ in 0..2 {
        let Some(baseline) = render(document, source.width, source.height) else {
            break;
        };
        let selected = select(source, &baseline, &patches);
        if selected.is_empty() {
            break;
        }
        compose(
            document,
            dimensions,
            (source.width, source.height),
            &selected,
        );
        count += selected.len();
    }
    count
}

fn remove_replaced_shapes(
    document: &mut Document,
    dimensions: (usize, usize),
    source_dimensions: (usize, usize),
    patches: &[&Patch],
) {
    use std::collections::HashSet;
    const PREFIX: &str = "compact-ink-prune-";
    if document.root().contains_name("filter") || document.root().contains_name("use") {
        return;
    }
    fn annotate(node: &mut Element, hidden: bool, count: &mut usize) {
        let hidden = hidden
            || matches!(
                node.name.as_str(),
                "defs" | "clipPath" | "symbol" | "pattern" | "marker"
            );
        if !hidden
            && matches!(
                node.name.as_str(),
                "path" | "rect" | "circle" | "ellipse" | "line" | "polygon" | "polyline"
            )
            && node.attr("id").is_none()
        {
            node.attributes
                .push(("id".into(), format!("{PREFIX}{}", *count)));
            *count += 1;
        }
        for child in &mut node.children {
            annotate(child, hidden, count);
        }
    }
    let mut annotated = document.clone();
    annotate(annotated.root_mut(), false, &mut 0);
    let Ok(tree) = resvg::usvg::Tree::from_str(&annotated, &resvg::usvg::Options::default()) else {
        return;
    };
    let mut removed = HashSet::new();
    fn collect(
        group: &resvg::usvg::Group,
        patches: &[&Patch],
        scale: [f32; 2],
        removed: &mut HashSet<String>,
    ) {
        for node in group.children() {
            if node.id().starts_with(PREFIX) {
                if let Some(b) = node.abs_layer_bounding_box() {
                    // Require a full source-pixel clearance at the crop edge.
                    // The padded geometric bound covers AA at native and 4x.
                    if patches.iter().any(|p| {
                        b.left() * scale[0] >= p.rect.x as f32 + 1.0
                            && b.top() * scale[1] >= p.rect.y as f32 + 1.0
                            && b.right() * scale[0] <= (p.rect.x + p.rect.width) as f32 - 1.0
                            && b.bottom() * scale[1] <= (p.rect.y + p.rect.height) as f32 - 1.0
                    }) {
                        removed.insert(node.id().to_string());
                    }
                }
            }
            if let resvg::usvg::Node::Group(g) = node {
                collect(g, patches, scale, removed);
            }
        }
    }
    collect(
        tree.root(),
        patches,
        [
            source_dimensions.0 as f32 / dimensions.0 as f32,
            source_dimensions.1 as f32 / dimensions.1 as f32,
        ],
        &mut removed,
    );
    fn omit(node: &mut Element, removed: &HashSet<String>) {
        node.children
            .retain(|c| !c.attr("id").is_some_and(|id| removed.contains(id)));
        node.attributes
            .retain(|(k, v)| !(k == "id" && v.starts_with(PREFIX)));
        for child in &mut node.children {
            omit(child, removed);
        }
    }
    omit(annotated.root_mut(), &removed);
    *document = annotated;
}

fn compose(
    document: &mut Document,
    dimensions: (usize, usize),
    source_dimensions: (usize, usize),
    patches: &[&Patch],
) {
    if patches.iter().any(|p| {
        p.rect.x == 0
            && p.rect.y == 0
            && p.rect.width == source_dimensions.0
            && p.rect.height == source_dimensions.1
    }) {
        document.root_mut().children.clear();
    } else {
        remove_replaced_shapes(document, dimensions, source_dimensions, patches);
    }
    let (w, h) = source_dimensions;
    let sx = dimensions.0 as f32 / w as f32;
    let sy = dimensions.1 as f32 / h as f32;
    let mut layer = Element::new(
        "g",
        attrs([
            ("data-compact-ink", "true".into()),
            ("transform", format!("scale({sx} {sy})")),
        ]),
    );
    for patch in patches {
        let r = patch.rect;
        let mut group = Element::new(
            "svg",
            attrs([
                ("x", r.x.to_string()),
                ("y", r.y.to_string()),
                ("width", r.width.to_string()),
                ("height", r.height.to_string()),
                ("viewBox", format!("0 0 {} {}", r.width, r.height)),
                ("preserveAspectRatio", "none".into()),
                ("overflow", "hidden".into()),
            ]),
        );
        group.children = patch.document.root().children.clone();
        layer.children.push(group);
    }
    document.root_mut().children.push(layer);
}

#[cfg(test)]
include!("../tests/unit/compact_ink.rs");
