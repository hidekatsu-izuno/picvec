//! Source-supported pale separator bands behind touching foreground figures.
//! Factoring a band before fitting prevents it from joining a whole sheet to
//! one object, and keeps its geometry identical across refinement boundaries.

use crate::chroma::AlphaMatte;
use crate::raster::{RasterSource, SourceRaster};
use crate::svg_document::attrs;
use crate::svg_document::{Document, Elements};

pub(crate) struct Separator {
    pub rect: [f32; 4],
    pub color: [f32; 3],
    pub opacity: f32,
}

pub(crate) fn extract(
    source: &SourceRaster,
    matte: &AlphaMatte,
    minimum_length: usize,
) -> (Vec<Separator>, Option<AlphaMatte>) {
    let (width, height) = (source.width, source.height);
    let mut separators = Vec::new();
    let mut cleared = Vec::new();
    let pale = |rgb: [f32; 3]| {
        let lo = rgb.into_iter().fold(1.0_f32, f32::min);
        let hi = rgb.into_iter().fold(0.0_f32, f32::max);
        lo >= 0.8 && hi - lo <= 0.2
    };
    for vertical in [false, true] {
        let (along, across) = if vertical {
            (height, width)
        } else {
            (width, height)
        };
        if along <= minimum_length {
            continue;
        }
        let at = |a: usize, b: usize| if vertical { (a, b) } else { (b, a) };
        let counts: Vec<usize> = (0..across)
            .map(|a| {
                (0..along)
                    .filter(|&b| {
                        let (x, y) = at(a, b);
                        matte.get(y * width + x) >= 0.5 && pale(source.get(x, y))
                    })
                    .count()
            })
            .collect();
        let mut i = 0;
        while i < across {
            if counts[i] * 5 < along * 4 {
                i += 1;
                continue;
            }
            let start = i;
            while i < across && counts[i] * 5 >= along * 4 {
                i += 1;
            }
            let end = i;
            if end - start > 16 || start < 3 || end + 3 >= across {
                continue;
            }
            // Only factor a band that really passes behind another object.
            // A weak band, broad white panel, or isolated line stays
            // in the normal source model.
            let crosses = (0..along)
                .filter(|&b| {
                    [start - 2, (start + end) / 2, end + 1]
                        .into_iter()
                        .all(|a| {
                            let (x, y) = at(a, b);
                            matte.get(y * width + x) >= 0.9
                                && source.get(x, y).into_iter().any(|v| v < 0.7)
                        })
                })
                .count();
            if crosses < 8 {
                continue;
            }
            let mut colors: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
            for b in 0..along {
                let (x, y) = at((start + end) / 2, b);
                let rgb = source.get(x, y);
                if matte.get(y * width + x) >= 0.85 && pale(rgb) {
                    for c in 0..3 {
                        colors[c].push(rgb[c]);
                    }
                }
            }
            if colors[0].len() * 5 < along * 4 {
                continue;
            }
            let color = colors.map(|mut values| {
                values.sort_by(f32::total_cmp);
                values[values.len() / 2]
            });
            let profile: Vec<f32> = (start - 3..end + 3)
                .map(|a| {
                    let mut values: Vec<f32> = (0..along)
                        .map(|b| {
                            let (x, y) = at(a, b);
                            if pale(source.get(x, y)) {
                                matte.get(y * width + x)
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    values.sort_by(f32::total_cmp);
                    values[values.len() / 2]
                })
                .collect();
            let opacity = profile.iter().copied().fold(0.0_f32, f32::max);
            if opacity < 0.85 {
                continue;
            }
            let half = 0.5 * opacity;
            let first = profile.iter().position(|&v| v >= half).unwrap();
            let last = profile.iter().rposition(|&v| v >= half).unwrap();
            if first == 0 || last + 1 == profile.len() {
                continue;
            }
            let crossing = |a: usize, b: usize| {
                (start - 3 + a) as f32
                    + 0.5
                    + (half - profile[a]) / (profile[b] - profile[a]) * (b as f32 - a as f32)
            };
            let lo = crossing(first - 1, first);
            let hi = crossing(last, last + 1);
            let rect = if vertical {
                [lo, 0.0, hi - lo, height as f32]
            } else {
                [0.0, lo, width as f32, hi - lo]
            };
            separators.push(Separator {
                rect,
                color,
                opacity,
            });
            let occluded: Vec<bool> = (0..along)
                .map(|b| {
                    let (x0, y0) = at(start - 3, b);
                    let (x1, y1) = at(end + 2, b);
                    matte.get(y0 * width + x0) >= 0.5
                        && matte.get(y1 * width + x1) >= 0.5
                        && (!pale(source.get(x0, y0)) || !pale(source.get(x1, y1)))
                })
                .collect();
            for a in start - 3..end + 3 {
                for (b, &covered) in occluded.iter().enumerate() {
                    let (x, y) = at(a, b);
                    let rgb = source.get(x, y);
                    let faint_shoulder = matte.get(y * width + x) < 0.5
                        && rgb.into_iter().fold(0.0_f32, f32::max) > 0.8;
                    if !covered && (pale(rgb) || faint_shoulder) {
                        cleared.push(y * width + x);
                    }
                }
            }
        }
    }
    let cleaned = (!cleared.is_empty()).then(|| matte.cleared(&cleared));
    (separators, cleaned)
}

pub(crate) fn prepend(document: &mut Document, separators: &[Separator], scale: [f32; 2]) {
    if separators.is_empty() {
        return;
    }
    let mut layer = Elements::new();
    layer.open("g", attrs([("data-source-separators", "true".into())]));
    for s in separators {
        let [x, y, w, h] = s.rect;
        let [r, g, b] = s.color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        layer.leaf(
            "rect",
            attrs([
                ("x", format!("{0:.4}", x * scale[0])),
                ("y", format!("{0:.4}", y * scale[1])),
                ("width", format!("{0:.4}", w * scale[0])),
                ("height", format!("{0:.4}", h * scale[1])),
                ("fill", format!("#{r:02x}{g:02x}{b:02x}")),
                ("fill-opacity", format!("{0:.4}", s.opacity)),
            ]),
        );
    }
    layer.close();
    document.root_mut().children.splice(..0, layer.roots);
}

#[cfg(test)]
include!("../tests/unit/separators.rs");
