//! Source-over factorisation into the eight RGB cube vertices.
use crate::svg_document::attrs;
use crate::{chroma::AlphaMatte, raster::Raster};
use std::fmt::Write;

// Materialize RGBA paint before source-over and the complete group before its
// clip. CSS isolation alone is not a compositing boundary in Chromium's SVG
// image renderer. This neutral sRGB stage changes neither colour nor alpha;
// it adds no blur, mask, fixed raster resolution or embedded bitmap.
#[cfg(test)]
pub(crate) const COMPOSITE_FILTER: &str = r#"<filter id="source-field-composite" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0"/></filter>"#;

pub(crate) fn composite_filter() -> crate::svg_document::Elements {
    let mut elements = crate::svg_document::Elements::new();
    elements.open(
        "filter",
        attrs([
            ("id", "source-field-composite".into()),
            ("x", "-10%".into()),
            ("y", "-10%".into()),
            ("width", "120%".into()),
            ("height", "120%".into()),
            ("color-interpolation-filters", "sRGB".into()),
        ]),
    );
    elements.leaf(
        "feColorMatrix",
        attrs([
            ("type", "matrix".into()),
            ("values", "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0".into()),
        ]),
    );
    elements.close();
    elements
}

#[derive(Clone, Debug)]
pub(crate) struct Layer {
    pub path: String,
    pub opacity: f32,
    pub color: [u8; 3],
}
#[derive(Clone, Debug)]
pub(crate) struct Patch {
    pub origin_x: usize,
    pub origin_y: usize,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub layers: Vec<Layer>,
    source: Raster,
    matte: AlphaMatte,
}

/// The same opaque-underpaint rule used for ordinary paint seams: retain
/// support beneath a later opaque face, never beneath authored transparency.
/// Returned rectangles are disjoint and remain inside the replacement core.
pub(crate) fn opaque_boundary_overlap(p: &Patch) -> Vec<[usize; 4]> {
    let mut rectangles: Vec<[usize; 4]> = Vec::new();
    let mut active = std::collections::BTreeMap::<(usize, usize), usize>::new();
    for y in p.y..p.y + p.height {
        let supported = |x: usize| {
            if x >= p.x + 2 && x + 2 < p.x + p.width && y >= p.y + 2 && y + 2 < p.y + p.height {
                return false;
            }
            let (lx, ly) = (x - p.origin_x, y - p.origin_y);
            p.matte.get(ly * p.source.width + lx) >= 254.5 / 255.0
        };
        let mut next = std::collections::BTreeMap::new();
        let mut x = p.x;
        while x < p.x + p.width {
            if !supported(x) {
                x += 1;
                continue;
            }
            let start = x;
            while x < p.x + p.width && supported(x) {
                x += 1;
            }
            let key = (start, x - start);
            let index = if let Some(&index) = active.get(&key) {
                rectangles[index][3] += 1;
                index
            } else {
                let index = rectangles.len();
                rectangles.push([start, y, x - start, 1]);
                index
            };
            next.insert(key, index);
        }
        active = next;
    }
    rectangles
}

/// Match the replacement boundary to the surrounding vector reconstruction.
/// Keep authored coverage authoritative; a hole in the coarse preview is not
/// evidence that the source should become transparent.
pub(crate) fn match_boundary(p: &Patch, base: &resvg::tiny_skia::Pixmap) -> Vec<Layer> {
    let mut source = p.source.clone();
    let mut alpha = Vec::with_capacity(source.pixels.len());
    for i in 0..source.pixels.len() {
        let x = p.origin_x + i % source.width;
        let y = p.origin_y + i / source.width;
        let distance = (x as f32 - p.x as f32)
            .min(y as f32 - p.y as f32)
            .min((p.x + p.width - 1) as f32 - x as f32)
            .min((p.y + p.height - 1) as f32 - y as f32);
        let t = (distance / 4.0).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        let pixel = &base.data()[(y * base.width() as usize + x) * 4..][..4];
        let a = p.matte.get(i);
        if pixel[3] > 0 {
            source.pixels[i] = std::array::from_fn(|k| {
                source.pixels[i][k] * t + pixel[k] as f32 / pixel[3] as f32 * (1.0 - t)
            });
        }
        alpha.push((a * 255.0).round() as u8);
    }
    let matte = AlphaMatte::from_u8(source.width, source.height, alpha);
    generate_window(
        &source,
        &matte,
        32,
        3_000_000,
        Some([p.x - p.origin_x, p.y - p.origin_y, p.width, p.height]),
    )
    .unwrap_or_else(|| p.layers.clone())
}

fn weights(rgb: [f32; 3], alpha: f32) -> [f32; 8] {
    let mut channels = [0, 1, 2];
    channels.sort_by(|&a, &b| rgb[a].total_cmp(&rgb[b]));
    let [lo, mid, hi] = channels;
    let mut w = [0.0; 8];
    w[0] = (1.0 - rgb[hi]) * alpha;
    w[7] = rgb[lo] * alpha;
    w[1 << hi] = (rgb[hi] - rgb[mid]) * alpha;
    w[(1 << hi) | (1 << mid)] = (rgb[mid] - rgb[lo]) * alpha;
    let mut above = 0.0;
    for k in (0..8).rev() {
        let mass = w[k];
        w[k] = if k == 0 && alpha >= 254.0 / 255.0 {
            1.0
        } else if 1.0 - above > 1e-6 {
            (mass / (1.0 - above)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        above += mass;
    }
    w
}

fn generate(
    source: &Raster,
    matte: &AlphaMatte,
    levels: usize,
    budget: usize,
) -> Option<Vec<Layer>> {
    generate_window(source, matte, levels, budget, None)
}

fn generate_window(
    source: &Raster,
    matte: &AlphaMatte,
    levels: usize,
    budget: usize,
    window: Option<[usize; 4]>,
) -> Option<Vec<Layer>> {
    let (w, h) = (source.width, source.height);
    let mut fields: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0; (w + 2) * (h + 2)]);
    for (i, &rgb) in source.pixels.iter().enumerate() {
        // Counter a small part of the second area filter introduced when the
        // already-antialiased source is interpolated and then rasterized again.
        let a = matte.get(i);
        let (x, y) = (i % w, i / w);
        let neighbours = [
            y.saturating_sub(1) * w + x,
            (y + 1).min(h - 1) * w + x,
            y * w + x.saturating_sub(1),
            y * w + (x + 1).min(w - 1),
        ];
        let average_alpha = neighbours.iter().map(|&j| matte.get(j)).sum::<f32>() * 0.25;
        let sharpened_alpha = (a + 0.25 * (a - average_alpha)).clamp(0.0, 1.0);
        let sharpened = std::array::from_fn(|k| {
            let value = rgb[k] * a;
            let mean = neighbours
                .iter()
                .map(|&j| source.pixels[j][k] * matte.get(j))
                .sum::<f32>()
                * 0.25;
            (value + 0.25 * (value - mean)).clamp(0.0, sharpened_alpha) / sharpened_alpha.max(1e-8)
        });
        let values = weights(sharpened, sharpened_alpha);
        for k in 0..8 {
            fields[k][(i / w + 1) * (w + 2) + i % w + 1] = (values[k] * 255.0).round() as u8;
        }
    }
    let mut layers = Vec::new();
    let mut bytes = 0;
    for (k, values) in fields.into_iter().enumerate() {
        // Keep common constant coverages exact while retaining every AA level.
        let mut counts = [0usize; 256];
        for y in 1..=h {
            for x in 1..=w {
                let i = y * (w + 2) + x;
                let v = values[i];
                if [i - 1, i + 1, i - w - 2, i + w + 2]
                    .iter()
                    .all(|&j| values[j] == v)
                {
                    counts[v as usize] += 1;
                }
            }
        }
        let mut targets: Vec<f32> = (1..=levels).map(|v| v as f32 / levels as f32).collect();
        targets.extend(
            (1..255)
                .filter(|&v| counts[v] >= 16)
                .map(|v| v as f32 / 255.0),
        );
        targets.sort_by(f32::total_cmp);
        targets.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        let field = AlphaMatte::from_u8(w + 2, h + 2, values);
        let mut previous = 0.0;
        for target in targets {
            let code = (255.0 * (target - previous) / (1.0 - previous)).round();
            if code < 1.0 {
                continue;
            }
            let mut path = String::new();
            for mut contour in field.isocontours((previous + target) * 0.5) {
                for p in &mut contour {
                    p.x -= 1.0;
                    p.y -= 1.0;
                }
                if let Some([x, y, w, h]) = window {
                    if contour.iter().all(|p| p.x < x as f32)
                        || contour.iter().all(|p| p.y < y as f32)
                        || contour.iter().all(|p| p.x > (x + w) as f32)
                        || contour.iter().all(|p| p.y > (y + h) as f32)
                    {
                        continue;
                    }
                }
                path.push_str(&crate::geometry::fitted_colour_contour_path_data(&contour));
                if bytes + path.len() > budget {
                    return None;
                }
            }
            let a = ((code + 0.0001) / 255.0).min(1.0);
            previous = target;
            if path.is_empty() {
                continue;
            }
            bytes += path.len();
            layers.push(Layer {
                path,
                opacity: a,
                color: [
                    if k & 1 > 0 { 255 } else { 0 },
                    if k & 2 > 0 { 255 } else { 0 },
                    if k & 4 > 0 { 255 } else { 0 },
                ],
            });
        }
    }
    (!layers.is_empty()).then_some(layers)
}

fn crop(source: &Raster, matte: &AlphaMatte, r: [usize; 4]) -> (Raster, AlphaMatte) {
    let [x, y, w, h] = r;
    (
        Raster::new(
            w,
            h,
            (y..y + h)
                .flat_map(|yy| (x..x + w).map(move |xx| source.pixels[yy * source.width + xx]))
                .collect(),
        ),
        matte.crop(x, y, w, h),
    )
}
fn supported(source: &Raster, matte: &AlphaMatte, layers: &[Layer], limits: [f32; 2]) -> bool {
    let (w, h) = (source.width, source.height);
    let mut document =
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">");
    let mut previous = None;
    for l in layers {
        if previous != Some(l.color) {
            if previous.is_some() {
                document.push_str("</g>");
            }
            document.push_str("<g style=\"isolation:isolate\">");
            previous = Some(l.color);
        }
        write!(document,"<path d=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\"/>",l.path,l.color[0],l.color[1],l.color[2],l.opacity).unwrap();
    }
    if previous.is_some() {
        document.push_str("</g>");
    }
    document.push_str("</svg>");
    let Ok(tree) = resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()) else {
        return false;
    };
    let Some(mut rendered) = resvg::tiny_skia::Pixmap::new(w as u32, h as u32) else {
        return false;
    };
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut rendered.as_mut(),
    );
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS") {
        let prefix = prefix.to_string_lossy();
        let _ = std::fs::write(format!("{prefix}-field-{w}-{h}.svg"), &document);
        let _ = rendered.save_png(format!("{prefix}-field-{w}-{h}.png"));
    }
    let (mut rgb, mut alpha, mut n) = (0.0, 0.0, 0usize);
    for (i, p) in rendered.data().chunks_exact(4).enumerate() {
        let a = matte.get(i);
        if a == 0.0 && p[3] == 0 {
            continue;
        }
        n += 1;
        alpha += (a * 255.0 - p[3] as f32).abs();
        rgb += (0..3)
            .map(|k| (source.pixels[i][k] * a * 255.0 - p[k] as f32).abs())
            .sum::<f32>()
            / 3.0;
    }
    #[cfg(feature = "diagnostics")]
    if std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some() {
        eprintln!(
            "colour field quality: rgb={} alpha={}",
            rgb / n.max(1) as f32,
            alpha / n.max(1) as f32
        );
    }
    rgb / (n.max(1) as f32) < limits[0] && alpha / (n.max(1) as f32) < limits[1]
}

/// Locate narrow chromatic materials with opaque support and a large
/// antialias shoulder. These are candidates for sharp material ownership.
fn narrow_components(source: &Raster, matte: &AlphaMatte) -> Vec<Vec<usize>> {
    let (w, h) = (source.width, source.height);
    let keys: Vec<_> = source
        .pixels
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let max = c.iter().copied().fold(0.0_f32, f32::max);
            let min = c.iter().copied().fold(1.0_f32, f32::min);
            if matte.get(i) <= 0.06 || max <= 0.2 || max - min <= 0.6 * max {
                return None;
            }
            let q = c.map(|v| (6.0 * v / max).round() as u16);
            Some(q[0] * 49 + q[1] * 7 + q[2])
        })
        .collect();
    let mut seen = vec![false; w * h];
    let mut proposals = Vec::new();
    for seed in 0..w * h {
        let Some(key) = keys[seed] else {
            continue;
        };
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        let mut pixels = vec![seed];
        let mut cursor = 0;
        while cursor < pixels.len() {
            let i = pixels[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            for j in [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then_some(i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then_some(i + w),
            ]
            .into_iter()
            .flatten()
            {
                if !seen[j] && keys[j] == Some(key) {
                    seen[j] = true;
                    pixels.push(j);
                }
            }
        }
        if !(8..=160).contains(&pixels.len()) {
            continue;
        }
        let fractional = pixels.iter().filter(|&&i| matte.get(i) < 0.98).count();
        if fractional < 12 || fractional * 2 < pixels.len() || pixels.len() - fractional < 3 {
            continue;
        }
        let n = pixels.len() as f32;
        let cx = pixels.iter().map(|&i| (i % w) as f32).sum::<f32>() / n;
        let cy = pixels.iter().map(|&i| (i / w) as f32).sum::<f32>() / n;
        let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
        for &i in &pixels {
            let (x, y) = (i % w, i / w);
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            xx += dx * dx / n;
            yy += dy * dy / n;
            xy += dx * dy / n;
        }
        let d = ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt();
        let short = (6.0 * (xx + yy - d).max(0.0)).sqrt();
        let long = (6.0 * (xx + yy + d)).sqrt();
        if short > 4.5 || !(6.0..=40.0).contains(&long) || long < 2.2 * short.max(1.0) {
            continue;
        }
        let clear = pixels.iter().any(|&i| {
            let (x, y) = (i % w, i / w);
            (y.saturating_sub(2)..=(y + 2).min(h - 1)).any(|yy| {
                (x.saturating_sub(2)..=(x + 2).min(w - 1)).any(|xx| matte.get(yy * w + xx) == 0.0)
            })
        });
        if !clear {
            continue;
        }
        proposals.push(pixels);
    }
    proposals
}

/// Recover material ownership, not independent vector overlays. The caller
/// feeds these samples back into the normal shared-boundary partition.
pub(crate) fn narrow_material_pixels(
    source: &Raster,
    matte: &AlphaMatte,
) -> Vec<(usize, [u8; 3], bool)> {
    let (w, h) = (source.width, source.height);
    let mut recovered = std::collections::BTreeMap::new();
    for pixels in narrow_components(source, matte) {
        let mut opaque: Vec<_> = pixels
            .iter()
            .copied()
            .filter(|&i| matte.get(i) >= 0.98)
            .collect();
        opaque.sort_by(|&a, &b| source.pixels[a][0].total_cmp(&source.pixels[b][0]));
        let colour = source.pixels[opaque[opaque.len() / 2]];
        let norm = colour.iter().map(|c| c * c).sum::<f32>();
        let mut selected = std::collections::BTreeSet::new();
        for &i in &pixels {
            let (x, y) = (i % w, i / w);
            for yy in y.saturating_sub(2)..=(y + 2).min(h - 1) {
                for xx in x.saturating_sub(2)..=(x + 2).min(w - 1) {
                    selected.insert(yy * w + xx);
                }
            }
        }
        for i in selected {
            let a = matte.get(i);
            if a == 0.0 {
                continue;
            }
            let rgb = source.pixels[i];
            let t = (0..3).map(|k| rgb[k] * colour[k]).sum::<f32>() / norm;
            if !(0.0..=if a < 0.5 { 1.5 } else { 1.08 }).contains(&t)
                || (0..3).any(|k| (rgb[k] - colour[k] * t).abs() > 8.0 / 255.0)
            {
                continue;
            }
            let paint = if t >= 0.5 { colour } else { [0.0; 3] };
            let (x, y) = (i % w, i / w);
            let core = (y.saturating_sub(2)..=(y + 2).min(h - 1)).any(|yy| {
                (x.saturating_sub(2)..=(x + 2).min(w - 1)).any(|xx| {
                    let j = yy * w + xx;
                    matte.get(j) >= 0.98
                        && (0..3).all(|k| (source.pixels[j][k] - paint[k]).abs() < 8.0 / 255.0)
                })
            });
            if !core && !(t >= 0.5 && pixels.contains(&i)) {
                continue;
            }
            let visible = a >= 0.5;
            recovered.insert(
                i,
                (
                    if visible {
                        paint.map(|c| (c * 255.0).round() as u8)
                    } else {
                        [0; 3]
                    },
                    visible,
                ),
            );
        }
    }
    recovered.into_iter().map(|(i, (c, a))| (i, c, a)).collect()
}

/// Recover bounded, predominantly opaque illustrations with subpixel ink.
/// Accepted source rectangles leave segmentation before alpha-band splitting.
pub(crate) fn extract(
    source: &mut Raster,
    matte: &AlphaMatte,
    backing: [f32; 3],
) -> (AlphaMatte, Vec<Patch>) {
    let (w, h) = (source.width, source.height);
    let mut pending = vec![false; w * h];
    let mut proposals = Vec::new();
    let luma = |i: usize| {
        let c = source.pixels[i];
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    };
    // Compact flat fills anchor the scope; opposed bright samples distinguish
    // under-resolved dark ink from broad shadows or ordinary silhouettes.
    let keys: Vec<_> = source
        .pixels
        .iter()
        .map(|c| c.map(|v| (v * 63.0).round() as u8))
        .collect();
    for seed in 0..pending.len() {
        let c = source.pixels[seed];
        if pending[seed]
            || matte.get(seed) < 0.99
            || c.iter().copied().fold(0.0_f32, f32::max) < 0.8
            || c.iter().copied().fold(0.0_f32, f32::max) - c.iter().copied().fold(1.0_f32, f32::min)
                < 0.15
        {
            continue;
        }
        let key = keys[seed];
        let mut pixels = vec![seed];
        pending[seed] = true;
        let mut cursor = 0;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        while cursor < pixels.len() {
            let i = pixels[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            for j in [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then_some(i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then_some(i + w),
            ]
            .into_iter()
            .flatten()
            {
                if !pending[j] && matte.get(j) >= 0.99 && keys[j] == key {
                    pending[j] = true;
                    pixels.push(j);
                }
            }
        }
        let bw = x1 - x0;
        let bh = y1 - y0;
        if !(512..=8000).contains(&pixels.len())
            || pixels.len() * 100 < bw * bh * 15
            || pixels.len() * 100 > bw * bh * 95
            || bw.min(bh) < 16
            || bw.max(bh) > 128
        {
            continue;
        }
        let mut weak = 0;
        for y in y0.saturating_sub(2)..(y1 + 2).min(h) {
            for x in x0.saturating_sub(2)..(x1 + 2).min(w) {
                let i = y * w + x;
                if x < 2
                    || y < 2
                    || x + 2 >= w
                    || y + 2 >= h
                    || matte.get(i) < 0.9
                    || !(0.05..0.55).contains(&luma(i))
                {
                    continue;
                }
                let opaque_core = (y - 1..=y + 1).any(|yy| {
                    (x - 1..=x + 1)
                        .any(|xx| matte.get(yy * w + xx) >= 0.99 && luma(yy * w + xx) < 0.03)
                });
                if opaque_core {
                    continue;
                }
                if [
                    (i - 2, i + 2),
                    (i - 2 * w, i + 2 * w),
                    (i - 2 * w - 2, i + 2 * w + 2),
                    (i - 2 * w + 2, i + 2 * w - 2),
                ]
                .iter()
                .any(|&(a, b)| {
                    matte.get(a) >= 0.99
                        && matte.get(b) >= 0.99
                        && luma(a) > luma(i) + 0.18
                        && luma(b) > luma(i) + 0.18
                }) {
                    weak += 1;
                }
            }
        }
        if weak < 32 {
            continue;
        }
        let x = x0.saturating_sub(6);
        let y = y0.saturating_sub(6);
        let right = (x1 + 6).min(w);
        let bottom = (y1 + 6).min(h);
        proposals.push((
            [x, y, right - x, bottom - y],
            (right - x) * (bottom - y) * 192,
        ));
    }
    let mut patches: Vec<Patch> = Vec::new();
    let mut cleared = Vec::new();
    for (r, budget) in proposals {
        let [x, y, pw, ph] = r;
        if patches
            .iter()
            .any(|p| x < p.x + p.width && p.x < x + pw && y < p.y + p.height && p.y < y + ph)
        {
            continue;
        }
        let origin_x = x.saturating_sub(4);
        let origin_y = y.saturating_sub(4);
        let expanded = [
            origin_x,
            origin_y,
            (x + pw + 4).min(w) - origin_x,
            (y + ph + 4).min(h) - origin_y,
        ];
        let (image, alpha) = crop(source, matte, expanded);
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some() {
            eprintln!("colour field proposal {r:?} budget={budget}");
        }
        let Some(layers) = generate(&image, &alpha, 32, budget.min(2_000_000)) else {
            continue;
        };
        if !supported(&image, &alpha, &layers, [7.0, 3.5]) {
            continue;
        }
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "colour field accepted {r:?}: {} bytes",
                layers.iter().map(|l| l.path.len()).sum::<usize>()
            );
        }
        for yy in y + 4..(y + ph).saturating_sub(4) {
            for xx in x + 4..(x + pw).saturating_sub(4) {
                let i = yy * w + xx;
                source.pixels[i] = backing;
                cleared.push(i);
            }
        }
        patches.push(Patch {
            origin_x,
            origin_y,
            x,
            y,
            width: pw,
            height: ph,
            layers,
            source: image,
            matte: alpha,
        });
    }
    (
        if cleared.is_empty() {
            matte.clone()
        } else {
            matte.cleared(&cleared)
        },
        patches,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(bytes: &[u8]) -> (Raster, AlphaMatte) {
        let image = image::load_from_memory(bytes).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        (
            Raster::new(
                w as usize,
                h as usize,
                image
                    .pixels()
                    .map(|p| std::array::from_fn(|k| p[k] as f32 / 255.0))
                    .collect(),
            ),
            AlphaMatte::from_u8(
                w as usize,
                h as usize,
                image.pixels().map(|p| p[3]).collect(),
            ),
        )
    }

    fn flat_patch(alpha: u8) -> Patch {
        let source = Raster::blank(32, 32, [230.0 / 255.0, 204.0 / 255.0, 51.0 / 255.0]);
        let matte = AlphaMatte::from_u8(32, 32, vec![alpha; 32 * 32]);
        let layers = generate(&source, &matte, 32, 2_000_000).unwrap_or_default();
        Patch {
            origin_x: 0,
            origin_y: 0,
            x: 4,
            y: 4,
            width: 24,
            height: 24,
            source,
            matte,
            layers,
        }
    }

    #[test]
    fn authored_transparency_never_receives_boundary_underpaint() {
        for alpha in [0, 128, 253, 254] {
            assert!(opaque_boundary_overlap(&flat_patch(alpha)).is_empty());
        }
        assert!(!opaque_boundary_overlap(&flat_patch(255)).is_empty());
    }

    #[test]
    fn boundary_matching_does_not_copy_preview_coverage_errors() {
        for (source_alpha, preview_alpha) in [(255, 0), (128, 255), (0, 255)] {
            let p = flat_patch(source_alpha);
            let mut base = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
            base.fill(resvg::tiny_skia::Color::from_rgba8(
                230,
                204,
                51,
                preview_alpha,
            ));
            let mut svg =
                String::from(r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">"#);
            for layer in match_boundary(&p, &base) {
                write!(svg, "<path d=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\"/>", layer.path, layer.color[0], layer.color[1], layer.color[2], layer.opacity).unwrap();
            }
            svg.push_str("</svg>");
            let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
            let mut rendered = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut rendered.as_mut(),
            );
            for (x, y) in [(4, 16), (5, 16), (16, 16), (27, 16)] {
                let actual = rendered.pixel(x, y).unwrap().alpha();
                // Eight independently quantized source-over factors accumulate rounding.
                assert!(
                    (actual as i16 - source_alpha as i16).abs() <= 5,
                    "source={source_alpha} preview={preview_alpha} ({x},{y}) actual={actual}"
                );
            }
        }
    }

    #[test]
    fn isolated_patch_keeps_an_opaque_boundary_at_fractional_zoom() {
        let p = flat_patch(255);
        let mut base = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
        base.fill(resvg::tiny_skia::Color::from_rgba8(230, 204, 51, 255));
        let mut outside = String::from("M0 0H32V32H0ZM4 4h24v24h-24Z");
        for [x, y, w, h] in opaque_boundary_overlap(&p) {
            write!(outside, "M{x} {y}h{w}v{h}h-{w}Z").unwrap();
        }
        let mut svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs>{COMPOSITE_FILTER}<clipPath id="outside"><path d="{outside}" clip-rule="evenodd"/></clipPath><clipPath id="inside"><rect x="4" y="4" width="24" height="24"/></clipPath></defs><g clip-path="url(#outside)"><rect width="32" height="32" fill="#e6cc33"/></g><g clip-path="url(#inside)" style="isolation:isolate" filter="url(#source-field-composite)">"##
        );
        for layer in match_boundary(&p, &base) {
            write!(svg, "<path d=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\" filter=\"url(#source-field-composite)\"/>", layer.path, layer.color[0], layer.color[1], layer.color[2], layer.opacity).unwrap();
        }
        svg.push_str("</g></svg>");
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        for scale in [1.0_f32, 3.3, 8.0] {
            let size = (32.0 * scale).ceil() as u32 + 2;
            let mut rendered = resvg::tiny_skia::Pixmap::new(size, size).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, 0.37, 0.21),
                &mut rendered.as_mut(),
            );
            for y in (2.0 * scale).ceil() as u32..(30.0 * scale).floor() as u32 {
                for x in (2.0 * scale).ceil() as u32..(30.0 * scale).floor() as u32 {
                    let pixel = rendered.pixel(x, y).unwrap();
                    assert_eq!(pixel.alpha(), 255, "scale={scale}, x={x}, y={y}");
                    for (actual, expected) in [pixel.red(), pixel.green(), pixel.blue()]
                        .into_iter()
                        .zip([230, 204, 51])
                    {
                        assert!(
                            (actual as i16 - expected as i16).abs() <= 4,
                            "scale={scale}, x={x}, y={y}, actual={actual}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn source_over_reconstructs_translucent_chromatic_colours() {
        for rgb in [
            [0.8, 0.2, 0.4],
            [0.1, 0.9, 0.3],
            [0.3, 0.4, 0.95],
            [0.0; 3],
            [1.0; 3],
        ] {
            for alpha in [0.0, 0.2, 0.7, 1.0] {
                let mut pixel = [0.0; 4];
                for (k, a) in weights(rgb, alpha).into_iter().enumerate() {
                    for c in 0..3 {
                        pixel[c] = pixel[c] * (1.0 - a) + if k & (1 << c) > 0 { a } else { 0.0 };
                    }
                    pixel[3] = pixel[3] * (1.0 - a) + a;
                }
                for c in 0..3 {
                    assert!((pixel[c] - rgb[c] * alpha).abs() < 1e-5);
                }
                assert!((pixel[3] - alpha).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn transparent_source_and_exceeded_budget_emit_no_paint() {
        let source = Raster::new(16, 16, vec![[1.0, 0.7, 0.2]; 256]);
        assert!(generate(
            &source,
            &AlphaMatte::from_u8(16, 16, vec![0; 256]),
            32,
            100_000
        )
        .is_none());
        assert!(generate(&source, &AlphaMatte::from_u8(16, 16, vec![255; 256]), 32, 1).is_none());
    }

    #[test]
    fn fine_face_ink_and_dotted_rim_preserve_source_rgba() {
        for bytes in [
            include_bytes!("test-data/cliparts-face-ink.png").as_slice(),
            include_bytes!("test-data/cliparts-dotted-rim.png").as_slice(),
        ] {
            let (source, alpha) = fixture(bytes);
            let layers = generate(&source, &alpha, 32, 2_000_000).unwrap();
            assert!(supported(&source, &alpha, &layers, [7.0, 3.5]));
            assert!(layers.iter().all(|l| l.opacity > 0.0 && l.opacity <= 1.0));
        }
    }

    #[test]
    fn thin_materials_supply_ownership_including_the_mixed_junction() {
        for bytes in [
            include_bytes!("test-data/cactus-spine-tip.png").as_slice(),
            include_bytes!("test-data/cactus-spine-side.png").as_slice(),
        ] {
            let (source, alpha) = fixture(bytes);
            let pixels = narrow_material_pixels(&source, &alpha);
            assert!(
                pixels
                    .iter()
                    .filter(|&&(i, _, a)| a && alpha.get(i) < 0.98)
                    .count()
                    >= 8
            );
            assert!(pixels.iter().all(|&(i, _, _)| alpha.get(i) > 0.0));
            if source.width == 39 {
                assert!(
                    pixels.iter().any(|&(i, _, a)| i == 12 * 39 + 24 && a),
                    "opaque RGB(85,59,7) junction must remain material"
                );
            }
        }
    }

    #[test]
    fn narrow_rim_detector_requires_chroma_and_mixed_coverage() {
        let (source, alpha) = fixture(include_bytes!("test-data/cactus-spine-tip.png"));
        assert!(!narrow_components(&source, &alpha).is_empty());
        let opaque =
            AlphaMatte::from_u8(source.width, source.height, vec![255; source.pixels.len()]);
        assert!(narrow_components(&source, &opaque).is_empty());
        let translucent =
            AlphaMatte::from_u8(source.width, source.height, vec![128; source.pixels.len()]);
        assert!(narrow_components(&source, &translucent).is_empty());
        let neutral = Raster::new(
            source.width,
            source.height,
            vec![[0.3; 3]; source.pixels.len()],
        );
        assert!(narrow_components(&neutral, &alpha).is_empty());
    }

    #[test]
    fn broad_flat_fill_without_weak_ink_is_not_extracted() {
        let mut source = Raster::new(64, 64, vec![[1.0, 0.8, 0.0]; 64 * 64]);
        let original = source.pixels.clone();
        let alpha = AlphaMatte::from_u8(64, 64, vec![255; 64 * 64]);
        let (remaining, patches) = extract(&mut source, &alpha, [1.0; 3]);
        assert!(patches.is_empty());
        assert_eq!(source.pixels, original);
        assert!((0..64 * 64).all(|i| remaining.get(i) == 1.0));
    }
}
