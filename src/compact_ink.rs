//! Recover small, thin, two-colour silhouettes before reduction and palette
//! segmentation. These are geometric primitives, not detected or decoded text.
use std::collections::BTreeMap;

use crate::chroma::AlphaMatte;
use crate::geometry::{fitted_colour_contour_with_smoothing, Point};
use crate::raster::{RasterSource, SourceRaster};
use crate::svg_document::{attrs, Document, Element, Elements};

#[derive(Default)]
pub(crate) struct CompactInk {
    paths: BTreeMap<String, String>,
    pub components: usize,
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

pub(crate) fn extract(source: &SourceRaster) -> (CompactInk, Option<SourceRaster>) {
    let (w, h) = (source.width, source.height);
    let mut result = CompactInk::default();
    if w < 8 || h < 8 {
        return (result, None);
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
            let mut coverage = vec![0.0; cw * ch];
            let mut erase = Vec::new();
            let mut valid = true;
            let mut residuals = Vec::new();
            for y in 0..ch {
                for x in 0..cw {
                    let i = (y0 + y) * w + x0 + x;
                    // A neighbouring object keeps its own pixels and fringe.
                    if labels[i] != id && labels[i] != u32::MAX {
                        continue;
                    }
                    let near_other = (y0 + y - 1..=y0 + y + 1).any(|yy| {
                        (x0 + x - 1..=x0 + x + 1).any(|xx| {
                            let owner = labels[yy * w + xx];
                            owner != u32::MAX && owner != id
                        })
                    });
                    if near_other && labels[i] != id {
                        continue;
                    }
                    let p = source.get(x0 + x, y0 + y);
                    let amount = (0..3)
                        .map(|c| (p[c] - background[c]) * axis[c])
                        .sum::<f32>()
                        / norm;
                    let a = amount.clamp(0.0, 1.0);
                    let residual = (0..3)
                        .map(|c| (p[c] - background[c] - a * axis[c]).abs())
                        .fold(0.0_f32, f32::max);
                    residuals.push(residual);
                    // Sharpening may overshoot the robust ink endpoint while
                    // remaining on the same colour axis. A distinct paint
                    // deviates from that axis even before coverage clamping.
                    let chromatic_residual = (0..3)
                        .map(|c| (p[c] - background[c] - amount * axis[c]).abs())
                        .fold(0.0_f32, f32::max);
                    if a > 0.1 && chromatic_residual > 0.1 {
                        valid = false;
                    }
                    if x == 0 || y == 0 || x + 1 == cw || y + 1 == ch {
                        if a > 0.15 {
                            valid = false;
                        }
                        continue;
                    }
                    // Never remove a neighbour's AA fringe or a previously
                    // recovered object just because it lies in this box.
                    coverage[y * cw + x] = a;
                    if a > 0.02 {
                        erase.push(i);
                    }
                }
            }
            residuals.sort_by(f32::total_cmp);
            if !valid || residuals.is_empty() || residuals[residuals.len() * 9 / 10] > 0.045 {
                continue;
            }
            let mut area = 0;
            let mut perimeter = 0;
            for y in 1..ch - 1 {
                for x in 1..cw - 1 {
                    let i = y * cw + x;
                    if coverage[i] >= 0.5 {
                        area += 1;
                        perimeter += [i - 1, i + 1, i - cw, i + cw]
                            .iter()
                            .filter(|&&j| coverage[j] < 0.5)
                            .count();
                    }
                }
            }
            if area * 2 > perimeter * 5 {
                continue;
            }
            let matte = AlphaMatte::from_u8(
                cw,
                ch,
                coverage.iter().map(|v| (v * 255.0).round() as u8).collect(),
            );
            let contours = matte.isocontours(0.5);
            if contours.is_empty() {
                continue;
            }
            let local_path: String = contours
                .iter()
                .map(|p| fitted_colour_contour_with_smoothing(p, 0.35))
                .collect();
            // Measure the fitted, serialized silhouette. Tight colour-model
            // residual alone cannot detect a fitter that closes a small hole.
            let mut body = Elements::new();
            body.leaf(
                "path",
                attrs([
                    ("d", local_path),
                    ("fill", "white".into()),
                    ("fill-rule", "evenodd".into()),
                ]),
            );
            let document = Document::from_parts(cw, ch, Elements::new(), body);
            let Ok(tree) = resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default())
            else {
                continue;
            };
            let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(cw as u32, ch as u32) else {
                continue;
            };
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            let rendered: Vec<_> = pixmap
                .pixels()
                .iter()
                .map(|p| p.alpha() as f32 / 255.0)
                .collect();
            let mass = coverage.iter().sum::<f32>();
            let error = coverage
                .iter()
                .zip(&rendered)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            if mass < 3.0 || error > mass * 0.28 {
                continue;
            }
            if coverage
                .iter()
                .zip(&rendered)
                .any(|(&a, &b)| (a < 0.15 && b > 0.65) || (a > 0.85 && b < 0.35))
            {
                continue;
            }
            let path: String = contours
                .iter()
                .map(|p| {
                    let global: Vec<_> = p
                        .iter()
                        .map(|p| Point {
                            x: p.x + x0 as f32,
                            y: p.y + y0 as f32,
                        })
                        .collect();
                    fitted_colour_contour_with_smoothing(&global, 0.35)
                })
                .collect();
            candidates.push((mass, pixels[0], foreground, background, path, erase));
        }
    }
    // Opposite-polarity proposals can describe a ring and its counter.
    // Resolve competing coverage by source support, not by a dark-first bias.
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut cleared = BTreeMap::<usize, [f32; 3]>::new();
    for (_, _, foreground, background, path, erase) in candidates {
        if erase.iter().any(|i| cleared.contains_key(i)) {
            continue;
        }
        result
            .paths
            .entry(hex(foreground))
            .or_default()
            .push_str(&path);
        result.components += 1;
        for i in erase {
            cleared.insert(i, background);
        }
    }
    let cleaned = (!cleared.is_empty()).then(|| {
        SourceRaster::from_unorm16_fn(w, h, |i| {
            cleared
                .get(&i)
                .copied()
                .unwrap_or_else(|| source.get(i % w, i / w))
        })
    });
    (result, cleaned)
}

impl CompactInk {
    pub fn path_elements(&self) -> usize {
        self.paths.len()
    }

    pub fn append(&self, document: &mut Document, scale: [f32; 2]) {
        if self.paths.is_empty() {
            return;
        }
        let mut group = Element::new(
            "g",
            attrs([
                ("data-compact-ink", "true".into()),
                ("transform", format!("scale({} {})", scale[0], scale[1])),
            ]),
        );
        for (color, path) in &self.paths {
            group.children.push(Element::new(
                "path",
                attrs([
                    ("fill", color.clone()),
                    ("fill-rule", "evenodd".into()),
                    ("d", path.clone()),
                ]),
            ));
        }
        document.root_mut().children.push(group);
    }
}

#[cfg(test)]
include!("../tests/unit/compact_ink.rs");
