//! Keep source-supported soft details without blurring sharp lettering.
use crate::{chroma::AlphaMatte, raster::Raster, Result};
use std::fmt::Write;

#[derive(Clone, Copy)]
struct Patch {
    x: usize,
    y: usize,
    size: usize,
    sigma: f32,
    gain: f32,
}

fn error(source: &Raster, render: &Raster, patch: Patch) -> f32 {
    let mut total = 0.0;
    for y in patch.y..patch.y + patch.size {
        for x in patch.x..patch.x + patch.size {
            let i = y * source.width + x;
            total += (0..3)
                .map(|c| (source.pixels[i][c] - render.pixels[i][c]).powi(2))
                .sum::<f32>()
                / 3.0;
        }
    }
    total / (patch.size * patch.size) as f32
}

fn propose(source: &Raster, render: &Raster, matte: Option<&AlphaMatte>) -> Vec<Patch> {
    let alpha = |x: isize, y: isize| {
        matte.map_or(1.0, |m| {
            m.get(
                y.clamp(0, source.height as isize - 1) as usize * source.width
                    + x.clamp(0, source.width as isize - 1) as usize,
            )
        })
    };
    let size = 24;
    let mut proposals = Vec::new();
    if source.width < size || source.height < size {
        return proposals;
    }
    for y in (0..=source.height - size).step_by(size) {
        for x in (0..=source.width - size).step_by(size) {
            let patch = Patch {
                x,
                y,
                size,
                sigma: 0.0,
                gain: 0.0,
            };
            let mut sharp = 0;
            let mut colored = 0;
            for yy in y..y + size {
                for xx in x..x + size {
                    let pixel = source.get(xx, yy);
                    let lo = pixel.into_iter().fold(1.0_f32, f32::min);
                    let hi = pixel.into_iter().fold(0.0_f32, f32::max);
                    sharp += usize::from(lo < 0.12);
                    colored += usize::from(hi - lo > 0.025);
                }
            }
            if sharp > 2 || colored > 2 {
                continue;
            }
            let baseline = error(source, render, patch);
            if baseline < 0.0005 {
                continue;
            }
            let mut best = baseline;
            let mut selected = 0.0;
            for sigma in [0.6_f32, 1.0, 1.5, 2.0] {
                let radius = (sigma * 3.0).ceil() as isize;
                let mut kernel: Vec<f32> = (-radius..=radius)
                    .map(|d| (-0.5 * (d as f32 / sigma).powi(2)).exp())
                    .collect();
                let sum: f32 = kernel.iter().sum();
                for v in &mut kernel {
                    *v /= sum;
                }
                let mut score = 0.0;
                for yy in y..y + size {
                    for xx in x..x + size {
                        let mut filtered = 0.0;
                        let mut coverage = 0.0;
                        for (dy, &wy) in (-radius..=radius).zip(&kernel) {
                            for (dx, &wx) in (-radius..=radius).zip(&kernel) {
                                let weight = wx * wy * alpha(xx as isize + dx, yy as isize + dy);
                                filtered += render.get_clamped(xx as isize + dx, yy as isize + dy)
                                    [0]
                                    * weight;
                                coverage += weight;
                            }
                        }
                        // SVG filters blur premultiplied colour and coverage,
                        // not the white preview backing. Composite the filtered
                        // copy over the original and retain the source mask.
                        filtered += render.get(xx, yy)[0] * (1.0 - coverage);
                        // Fade into unchanged geometry at the patch boundary.
                        let d = (xx - x)
                            .min(x + size - 1 - xx)
                            .min(yy - y)
                            .min(y + size - 1 - yy) as f32;
                        let weight = (d / 4.0).min(1.0) * alpha(xx as isize, yy as isize);
                        let predicted = filtered * weight + render.get(xx, yy)[0] * (1.0 - weight);
                        score += (source.get(xx, yy)[0] - predicted).powi(2);
                    }
                }
                score /= (size * size) as f32;
                if score < best {
                    best = score;
                    selected = sigma;
                }
            }
            if best < baseline * 0.9 {
                proposals.push(Patch {
                    sigma: selected,
                    gain: baseline - best,
                    ..patch
                });
            }
        }
    }
    // Bound patch complexity and address the strongest source evidence first.
    proposals.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    proposals.truncate(512);
    proposals
}

fn document_with_patches(document: &str, patches: &[Patch]) -> String {
    if patches.is_empty() {
        return document.into();
    }
    let Some(body_start) = document.find("</defs>").map(|p| p + 7) else {
        return document.into();
    };
    let Some(body_end) = document.rfind("</svg>") else {
        return document.into();
    };
    let mut result = document[..body_start].to_string();
    result.push_str("<defs>");
    let sigmas = [0.6_f32, 1.0, 1.5, 2.0];
    for (i, &sigma) in sigmas.iter().enumerate() {
        let group: Vec<_> = patches.iter().filter(|p| p.sigma == sigma).collect();
        if group.is_empty() {
            continue;
        }
        let x0 = group.iter().map(|p| p.x).min().unwrap();
        let y0 = group.iter().map(|p| p.y).min().unwrap();
        let x1 = group.iter().map(|p| p.x + p.size).max().unwrap();
        let y1 = group.iter().map(|p| p.y + p.size).max().unwrap();
        // One reused vector instance per blur scale, not per patch. This
        // keeps the renderer's expanded instance tree bounded.
        let _ = write!(result, "<filter id=\"soft-filter-{i}\" x=\"-2%\" y=\"-2%\" width=\"104%\" height=\"104%\" color-interpolation-filters=\"sRGB\"><feGaussianBlur stdDeviation=\"{sigma}\"/></filter><mask id=\"soft-mask-{i}\" maskUnits=\"userSpaceOnUse\" x=\"{x0}\" y=\"{y0}\" width=\"{}\" height=\"{}\" style=\"mask-type:luminance\">", x1-x0,y1-y0);
        // Join adjacent equal-scale patches before feathering. An internal
        // tile edge must not leave a stripe of the original hard geometry.
        let mut rows: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut sorted = group;
        sorted.sort_by_key(|p| (p.y, p.x));
        for p in sorted {
            if let Some(last) = rows.last_mut() {
                if last.1 == p.y && last.0 + last.2 == p.x {
                    last.2 += p.size;
                    continue;
                }
            }
            rows.push((p.x, p.y, p.size, p.size));
        }
        rows.sort_by_key(|r| (r.0, r.2, r.1));
        let mut rectangles: Vec<(usize, usize, usize, usize)> = Vec::new();
        for row in rows {
            if let Some(last) = rectangles.last_mut() {
                if last.0 == row.0 && last.2 == row.2 && last.1 + last.3 == row.1 {
                    last.3 += row.3;
                    continue;
                }
            }
            rectangles.push(row);
        }
        for (x, y, w, h) in rectangles {
            for d in 1..=4 {
                let grey = (255 * d / 4) as u8;
                let _ = write!(result, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#{grey:02x}{grey:02x}{grey:02x}\"/>",x+d,y+d,w-2*d,h-2*d);
            }
        }
        result.push_str("</mask>");
    }
    result.push_str("</defs><g id=\"soft-source\">");
    result.push_str(&document[body_start..body_end]);
    result.push_str("</g>");
    for (i, &sigma) in sigmas.iter().enumerate() {
        if !patches.iter().any(|p| p.sigma == sigma) {
            continue;
        }
        let _ = write!(result, "<g mask=\"url(#soft-mask-{i})\">");
        let source_mask = document.contains("id=\"source-alpha-mask\"");
        if source_mask {
            result.push_str("<g mask=\"url(#source-alpha-mask)\">");
        }
        let _ = write!(
            result,
            "<use href=\"#soft-source\" filter=\"url(#soft-filter-{i})\"/>"
        );
        if source_mask {
            result.push_str("</g>");
        }
        result.push_str("</g>");
    }
    result.push_str("</svg>");
    result
}

pub(crate) fn refine(
    document: &str,
    source: &Raster,
    matte: Option<&AlphaMatte>,
    render: impl Fn(&str) -> Result<Raster>,
) -> Result<String> {
    // Coloured illustrations cannot supply neutral shading evidence here.
    if source
        .pixels
        .iter()
        .filter(|p| {
            p.iter().copied().fold(1.0_f32, f32::min) + 0.025
                < p.iter().copied().fold(0.0_f32, f32::max)
        })
        .count()
        > source.pixels.len() / 20
    {
        return Ok(document.into());
    }
    let before = render(document)?;
    let candidates = propose(source, &before, matte);
    if candidates.is_empty() {
        return Ok(document.into());
    }
    let trial = document_with_patches(document, &candidates);
    let after = render(&trial)?;
    let accepted: Vec<_> = candidates
        .into_iter()
        .filter(|p| error(source, &after, *p) < error(source, &before, *p) * 0.95)
        .collect();
    Ok(document_with_patches(document, &accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wikipedia_soft_tip_is_detected_without_mixing_the_transparent_backing() {
        let input = image::load_from_memory(include_bytes!("test-data/wiki-soft-source.png"))
            .unwrap()
            .to_rgba8();
        let previous = image::load_from_memory(include_bytes!("test-data/wiki-soft-before.png"))
            .unwrap()
            .to_rgba8();
        let matte = AlphaMatte::from_u8(96, 96, input.pixels().map(|p| p[3]).collect());
        let composite = |image: &image::RgbaImage| {
            Raster::new(
                96,
                96,
                image
                    .pixels()
                    .map(|p| {
                        let a = p[3] as f32 / 255.0;
                        [0, 1, 2].map(|c| p[c] as f32 / 255.0 * a + 1.0 - a)
                    })
                    .collect(),
            )
        };
        let source = composite(&input);
        let render = composite(&previous);
        let proposals = propose(&source, &render, Some(&matte));
        assert!(
            proposals
                .iter()
                .any(|p| p.x == 48 && p.y == 24 && p.sigma <= 1.5),
            "blurred source tip was missed"
        );
    }

    #[test]
    fn filtered_svg_improves_soft_detail_without_raster_embedding() {
        let document = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"72\" height=\"72\"><defs></defs><rect width=\"72\" height=\"72\" fill=\"#ccc\"/><path d=\"M43 30H54V42H43Z\" fill=\"#5a5a5a\"/></svg>";
        let render = |document: &str| -> Result<Raster> {
            let tree = resvg::usvg::Tree::from_str(document, &resvg::usvg::Options::default())?;
            let mut pixmap = resvg::tiny_skia::Pixmap::new(72, 72).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            Ok(Raster::new(
                72,
                72,
                pixmap
                    .pixels()
                    .iter()
                    .map(|p| {
                        [
                            p.red() as f32 / 255.0,
                            p.green() as f32 / 255.0,
                            p.blue() as f32 / 255.0,
                        ]
                    })
                    .collect(),
            ))
        };
        let before = render(document).unwrap();
        let input = image::RgbImage::from_fn(72, 72, |x, y| {
            image::Rgb(
                before
                    .get(x as usize, y as usize)
                    .map(|v| (v * 255.0).round() as u8),
            )
        });
        let soft = image::imageops::blur(&input, 1.4);
        let source = Raster::new(
            72,
            72,
            soft.pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let updated = refine(document, &source, None, render).unwrap();
        assert!(updated.contains("feGaussianBlur"));
        assert!(!updated.contains("<image"));
        let after = render(&updated).unwrap();
        let patch = Patch {
            x: 24,
            y: 24,
            size: 24,
            sigma: 0.0,
            gain: 0.0,
        };
        assert!(error(&source, &after, patch) < 0.9 * error(&source, &before, patch));
        assert!(
            (after.get(48, 30)[0] - source.get(48, 30)[0]).abs()
                < (before.get(48, 30)[0] - source.get(48, 30)[0]).abs() * 0.8,
            "internal tile edge retained a hard stripe"
        );
        assert_eq!(before.get(10, 10), after.get(10, 10));
        assert_eq!(refine(document, &before, None, render).unwrap(), document);
    }

    #[test]
    fn soft_source_detail_is_distinguished_from_a_sharp_detail() {
        let mut image = image::RgbImage::from_pixel(72, 72, image::Rgb([204; 3]));
        for y in 30..42 {
            for x in 31..38 {
                image.put_pixel(x, y, image::Rgb([90; 3]));
            }
        }
        let raster = |image: &image::RgbImage| {
            Raster::new(
                72,
                72,
                image
                    .pixels()
                    .map(|p| p.0.map(|v| v as f32 / 255.0))
                    .collect(),
            )
        };
        let render = raster(&image);
        let source = raster(&image::imageops::blur(&image, 1.4));
        let proposals = propose(&source, &render, None);
        assert_eq!(proposals.len(), 1);
        assert!((proposals[0].sigma - 1.5).abs() < 0.1);
        assert!(propose(&render, &render, None).is_empty());
    }
}
