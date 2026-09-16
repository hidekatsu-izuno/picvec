//! Preserve coupled neutral RGB/alpha fields without adjacent translucent cells.
//!
//! For premultiplied grey P = A*C, white opacity W=P over black opacity
//! B=(A-P)/(1-P) reconstructs both P and A. Nested level sets keep each scalar
//! field continuous across its support: ordinary source-over AA between two
//! adjacent half-covered translucent cells would instead lose coverage.
//! Only complex neutral materials with a dominant native opacity qualify.
//! Outer dark ink stays in the normal pipeline; accepted fields are removed
//! before RGB segmentation, alpha-code splitting and structural recovery.
use crate::{chroma::AlphaMatte, raster::Raster};

const NEAR_BLACK: f32 = 0.02;
#[derive(Clone, Debug)]
pub(crate) struct Layer {
    pub path: String,
    pub opacity: f32,
    pub white: bool,
}

fn factors(alpha: f32, grey: f32) -> [f32; 2] {
    let white = alpha * grey;
    // An opaque white area can hide any black underpaint. Snap at most one
    // alpha code here so quantisation cannot create unstable black holes near
    // white; preserve the premultiplied colour exactly in W.
    let black = if alpha >= 254.0 / 255.0 {
        1.0
    } else {
        (alpha - white) / (1.0 - white)
    };
    [black.clamp(0.0, 1.0), white.clamp(0.0, 1.0)]
}

pub(crate) fn extract(
    source: &mut Raster,
    matte: &AlphaMatte,
    backing: [f32; 3],
) -> (AlphaMatte, Vec<Layer>) {
    let (w, h) = (source.width, source.height);
    let neutral: Vec<_> = source
        .pixels
        .iter()
        .enumerate()
        .map(|(i, c)| {
            matte.get(i) > 0.0
                && (matte.get(i) < 0.999 || c[0] > NEAR_BLACK)
                && c.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    - c.iter().copied().fold(f32::INFINITY, f32::min)
                    <= 2.0 / 255.0
        })
        .collect();
    let mut seen = vec![false; w * h];
    let mut cleared = Vec::new();
    let mut layers = Vec::new();
    for seed in 0..w * h {
        if seen[seed] || !neutral[seed] {
            continue;
        }
        let mut pixels = vec![seed];
        seen[seed] = true;
        let mut cursor = 0;
        while cursor < pixels.len() {
            let i = pixels[cursor];
            cursor += 1;
            for j in [
                i.checked_sub(w),
                (i + w < w * h).then_some(i + w),
                (i % w > 0).then(|| i - 1),
                (i % w + 1 < w).then_some(i + 1),
            ]
            .into_iter()
            .flatten()
            {
                if neutral[j] && !seen[j] {
                    seen[j] = true;
                    pixels.push(j);
                }
            }
        }
        if pixels.len() < 512 {
            continue;
        }
        let (low, high) = pixels.iter().fold((1.0_f32, 0.0_f32), |(low, high), &i| {
            (low.min(source.pixels[i][0]), high.max(source.pixels[i][0]))
        });
        // A constant colour already has one scalar opacity paint; it does not
        // need two compositing fields.
        if high - low <= 4.0 / 255.0 {
            continue;
        }
        let mut histogram = [0usize; 256];
        let mut interior = 0;
        for &i in &pixels {
            let (x, y) = (i % w, i / w);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                continue;
            }
            if (y - 1..=y + 1).all(|yy| {
                (x - 1..=x + 1)
                    .all(|xx| neutral[yy * w + xx] && (0.5..0.97).contains(&matte.get(yy * w + xx)))
            }) {
                histogram[(matte.get(i) * 255.0).round() as usize] += 1;
                interior += 1;
            }
        }
        let code = (0..256).max_by_key(|&a| histogram[a]).unwrap();
        if !(128..=230).contains(&code)
            || interior < 512
            || histogram[code] * 5 < interior * 3
            || histogram.iter().filter(|&&n| n > 0).count() < 16
        {
            continue;
        }
        let x0 = pixels.iter().map(|i| i % w).min().unwrap();
        let x1 = pixels.iter().map(|i| i % w).max().unwrap();
        let y0 = pixels.iter().map(|i| i / w).min().unwrap();
        let y1 = pixels.iter().map(|i| i / w).max().unwrap();
        let (cw, ch) = (x1 - x0 + 3, y1 - y0 + 3);
        let mut fields = [vec![0u8; cw * ch], vec![0u8; cw * ch]];
        for &i in &pixels {
            let grey = source.pixels[i].iter().sum::<f32>() / 3.0;
            let f = factors(matte.get(i), grey);
            let j = (i / w - y0 + 1) * cw + i % w - x0 + 1;
            for k in 0..2 {
                fields[k][j] = (f[k] * 255.0).round() as u8;
            }
        }
        // Opaque near-black already drawn by the remaining scene can also serve
        // as black underpaint. Filling those hidden holes avoids tracing the
        // same pupil/ink boundary separately at every opacity level. Include
        // the full dark range excluded above, not just exact black: otherwise
        // isolated 1..5-code source pixels become holes in the underpaint.
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * w + x;
                if matte.get(i) >= 0.999 && source.pixels[i].iter().all(|&c| c <= NEAR_BLACK) {
                    let j = (y - y0 + 1) * cw + x - x0 + 1;
                    let grey = source.pixels[i].iter().sum::<f32>() / 3.0;
                    let f = factors(matte.get(i), grey);
                    fields[0][j] = (f[0] * 255.0).round() as u8;
                    fields[1][j] = (f[1] * 255.0).round() as u8;
                }
            }
        }
        let mut candidate = Vec::new();
        let budget = pixels.len() * 32;
        let mut bytes = 0;
        let mut exhausted = false;
        // Quantise the opacity field, not the RGB/alpha pair. Round incremental
        // paint opacity to 8-bit coverage explicitly; renderer truncation
        // otherwise accumulates a dark bias through many small source-over fills.
        const LEVELS: usize = 64;
        'fields: for (k, values) in fields.into_iter().enumerate() {
            let field = AlphaMatte::from_u8(cw, ch, values);
            for level in 1..=LEVELS {
                let mut path = String::new();
                for mut contour in field.isocontours((level as f32 - 0.5) / LEVELS as f32) {
                    for p in &mut contour {
                        p.x += x0 as f32 - 1.0;
                        p.y += y0 as f32 - 1.0;
                    }
                    path.push_str(&crate::geometry::fitted_alpha_contour_path_data(&contour));
                    if bytes + path.len() > budget {
                        exhausted = true;
                        break 'fields;
                    }
                }
                if !path.is_empty() {
                    bytes += path.len();
                    candidate.push(Layer {
                        path,
                        white: k == 1,
                        opacity: (((255.0 / (LEVELS + 1 - level) as f32).round() + 0.0001) / 255.0)
                            .min(1.0),
                    });
                }
            }
        }
        if exhausted || candidate.is_empty() {
            continue;
        }
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_PIPELINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "neutral fields {x0},{y0}..{x1},{y1}: {} pixels {} paths {} bytes",
                pixels.len(),
                candidate.len(),
                candidate.iter().map(|l| l.path.len()).sum::<usize>()
            );
        }
        for &i in &pixels {
            source.pixels[i] = backing;
        }
        cleared.extend(pixels);
        layers.extend(candidate);
    }
    (
        if cleared.is_empty() {
            matte.clone()
        } else {
            matte.cleared(&cleared)
        },
        layers,
    )
}

#[cfg(test)]
include!("../tests/unit/neutral_fields.rs");
