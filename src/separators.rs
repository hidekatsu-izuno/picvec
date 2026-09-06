//! Source-supported pale separator bands behind touching foreground figures.
//! Factoring a band before fitting prevents it from joining a whole sheet to
//! one object, and keeps its geometry identical across refinement boundaries.

use crate::chroma::AlphaMatte;
use crate::raster::{RasterSource, SourceRaster};

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

pub(crate) fn prepend(document: &mut String, separators: &[Separator], scale: [f32; 2]) {
    if separators.is_empty() {
        return;
    }
    let Some(root) = document.find("<svg") else {
        return;
    };
    let Some(end) = document[root..].find('>') else {
        return;
    };
    let mut layer = String::from("<g data-source-separators=\"true\">");
    for s in separators {
        let [x, y, w, h] = s.rect;
        let [r, g, b] = s.color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        layer.push_str(&format!("<rect x=\"{:.4}\" y=\"{:.4}\" width=\"{:.4}\" height=\"{:.4}\" fill=\"#{r:02x}{g:02x}{b:02x}\" opacity=\"{:.4}\"/>",
            x*scale[0],y*scale[1],w*scale[0],h*scale[1],s.opacity));
    }
    layer.push_str("</g>");
    document.insert_str(root + end + 1, &layer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occluded_band_keeps_its_subpixel_edges_and_foreground_coverage() {
        for vertical in [false, true] {
            let (width, height) = if vertical { (96, 320) } else { (320, 96) };
            let coordinates = |i: usize| {
                let (x, y) = (i % width, i / width);
                if vertical {
                    (y, x)
                } else {
                    (x, y)
                }
            };
            let source = SourceRaster::from_rgb8_fn(width, height, |i| {
                let (x, y) = coordinates(i);
                if (145..175).contains(&x) && (20..80).contains(&y) {
                    [0.1; 3]
                } else if (29..=36).contains(&y) {
                    [1.0; 3]
                } else {
                    [0.0, 1.0, 0.0]
                }
            });
            let alpha: Vec<_> = (0..width * height)
                .map(|i| {
                    let (x, y) = coordinates(i);
                    if ((145..175).contains(&x) && (20..80).contains(&y)) || (30..36).contains(&y) {
                        255
                    } else if y == 29 || y == 36 {
                        128
                    } else {
                        0
                    }
                })
                .collect();
            let matte = AlphaMatte::from_u8(width, height, alpha.clone());
            let (bands, cleaned) = extract(&source, &matte, 128);
            assert_eq!(bands.len(), 1);
            let band = &bands[0];
            let start = band.rect[usize::from(!vertical)];
            let span = band.rect[2 + usize::from(!vertical)];
            assert!((start - 29.5).abs() < 0.01 && (span - 7.0).abs() < 0.02);
            let cleaned = cleaned.unwrap();
            for (i, &old) in alpha.iter().enumerate() {
                let (x, y) = coordinates(i);
                if (145..175).contains(&x) && (20..80).contains(&y) {
                    assert_eq!(cleaned.get(i), old as f32 / 255.0);
                } else if (29..=36).contains(&y) {
                    assert_eq!(cleaned.get(i), 0.0);
                }
            }
            let translucent =
                AlphaMatte::from_u8(width, height, alpha.iter().map(|&v| v / 2).collect());
            assert!(extract(&source, &translucent, 128).0.is_empty());
        }
    }

    #[test]
    fn isolated_line_and_broad_panel_are_not_factored() {
        for end in [36, 60] {
            let source = SourceRaster::from_rgb8_fn(320, 96, |i| {
                if (30..end).contains(&(i / 320)) {
                    [1.0; 3]
                } else {
                    [0.0, 1.0, 0.0]
                }
            });
            let matte = AlphaMatte::from_u8(
                320,
                96,
                (0..320 * 96)
                    .map(|i| {
                        if (30..end).contains(&(i / 320)) {
                            255
                        } else {
                            0
                        }
                    })
                    .collect(),
            );
            assert!(extract(&source, &matte, 128).0.is_empty());
        }
    }
}
