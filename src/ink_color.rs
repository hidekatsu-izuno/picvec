//! Recover independent interior colours lost by a single-colour stroke.
use crate::color::{delta_e_ok, rgb_to_oklab};
use crate::raster::Raster;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub(crate) struct ColorPatch {
    pub path: String,
    pub color: [f32; 3],
    pixels: Vec<usize>,
}

impl ColorPatch {
    pub fn improves(&self, source: &Raster, before: &Raster, after: &Raster) -> bool {
        let gain: f32 = self
            .pixels
            .iter()
            .map(|&i| {
                let lab = rgb_to_oklab(source.pixels[i]);
                delta_e_ok(lab, rgb_to_oklab(before.pixels[i]))
                    - delta_e_ok(lab, rgb_to_oklab(after.pixels[i]))
            })
            .sum();
        gain > 2.0 * self.pixels.len() as f32
    }
}

pub(crate) fn propose(
    source: &Raster,
    paint: &Raster,
    before: &Raster,
    white: &Raster,
    black: &Raster,
) -> Vec<ColorPatch> {
    let width = source.width;
    let height = source.height;
    let mut candidates = vec![false; width * height];
    let mut foreground = vec![false; width * height];
    for (i, candidate) in candidates.iter_mut().enumerate() {
        // White/black renders measure the actual serialized stroke coverage,
        // including its fitted curve, caps, outline clips and alpha mask.
        let coverage = (0..3)
            .map(|c| white.pixels[i][c] - black.pixels[i][c])
            .fold(1.0_f32, f32::min);
        if coverage < 0.9 {
            continue;
        }
        let observed = source.pixels[i];
        let base = paint.pixels[i];
        let ink = before.pixels[i];
        let direction = std::array::from_fn::<_, 3, _>(|c| ink[c] - base[c]);
        let denominator: f32 = direction.iter().map(|v| v * v).sum();
        if denominator < 1e-6 {
            foreground[i] = delta_e_ok(rgb_to_oklab(observed), rgb_to_oklab(ink)) <= 5.0;
            continue;
        }
        let alpha = (0..3)
            .map(|c| (observed[c] - base[c]) * direction[c])
            .sum::<f32>()
            / denominator;
        // The source must also belong to the foreground half of this profile.
        // Background pixels under an over-wide fitted stroke are a geometry
        // error; colouring isolated cells there would punch speckled holes.
        if alpha < 0.5 {
            continue;
        }
        foreground[i] = true;
        // Keep the projection unbounded when testing for a third colour.
        // A too-light fitted ink yields alpha > 1 for its dark core. Clamping
        // here turns that same colour direction into a false residual and
        // paints the raster's alternating core coverage as square patches.
        let residual = (0..3)
            .map(|c| (observed[c] - base[c] - alpha * direction[c]).powi(2))
            .sum::<f32>()
            .sqrt();
        // Coverage changes of the existing two colours are not missing paint.
        *candidate = residual > 0.06 && delta_e_ok(rgb_to_oklab(observed), rgb_to_oklab(ink)) > 5.0;
    }
    // Independent interior paint needs a foreground collar. Coverage of the
    // fitted stroke alone also admits source/background antialias pixels
    // when the stroke is too wide or slightly displaced. Their square runs
    // would reinstate the raster staircase along a smooth vector boundary.
    for (i, candidate) in candidates.iter_mut().enumerate() {
        let (x, y) = (i % width, i / width);
        *candidate &= x > 0
            && y > 0
            && x + 1 < width
            && y + 1 < height
            && (y - 1..=y + 1).all(|py| (x - 1..=x + 1).all(|px| foreground[py * width + px]));
    }
    // Spatial support belongs to the material, not to an exactly uniform
    // colour bucket: a tiny shaded interior can contain several colours.
    let supported: Vec<bool> = candidates
        .iter()
        .enumerate()
        .map(|(i, &candidate)| {
            candidate
                && ((i % width > 0 && candidates[i - 1])
                    || (i % width + 1 < width && candidates[i + 1])
                    || (i >= width && candidates[i - width])
                    || (i + width < candidates.len() && candidates[i + width]))
        })
        .collect();
    let candidates = supported;
    let mut visited = vec![false; candidates.len()];
    let mut result = Vec::new();
    for start in 0..candidates.len() {
        if !candidates[start] || visited[start] {
            continue;
        }
        let anchor = rgb_to_oklab(source.pixels[start]);
        let mut pixels = vec![start];
        visited[start] = true;
        let mut cursor = 0;
        while cursor < pixels.len() {
            let i = pixels[cursor];
            cursor += 1;
            let x = i % width;
            let y = i / width;
            let neighbours = [
                x.checked_sub(1).map(|x| y * width + x),
                (x + 1 < width).then_some(i + 1),
                y.checked_sub(1).map(|y| y * width + x),
                (y + 1 < height).then_some(i + width),
            ];
            for j in neighbours.into_iter().flatten() {
                if !visited[j]
                    && candidates[j]
                    && delta_e_ok(anchor, rgb_to_oklab(source.pixels[j])) <= 4.0
                {
                    visited[j] = true;
                    pixels.push(j);
                }
            }
        }
        let color = std::array::from_fn(|c| {
            pixels.iter().map(|&i| source.pixels[i][c]).sum::<f32>() / pixels.len() as f32
        });
        // Integer scanline runs cannot paint neighbouring pixels. They remain
        // editable vector paths and are clipped to the original stroke union.
        pixels.sort_unstable();
        let mut path = String::new();
        let mut cursor = 0;
        while cursor < pixels.len() {
            let first = pixels[cursor];
            let y = first / width;
            let mut end = cursor + 1;
            while end < pixels.len()
                && pixels[end] == pixels[end - 1] + 1
                && pixels[end] / width == y
            {
                end += 1;
            }
            let x = first % width;
            let length = end - cursor;
            let _ = write!(path, "M{x} {y}h{length}v1h-{length}z");
            cursor = end;
        }
        result.push(ColorPatch {
            path,
            color,
            pixels,
        });
    }
    // Share a small correction palette. The same perceptual corridor used
    // within a region also bounds sharing between disconnected regions.
    // Largest regions choose representatives first; rendering validates every
    // region again after this quantization.
    result.sort_by(|a, b| {
        b.pixels
            .len()
            .cmp(&a.pixels.len())
            .then(a.pixels[0].cmp(&b.pixels[0]))
    });
    let mut palette = Vec::new();
    for patch in &mut result {
        let lab = rgb_to_oklab(patch.color);
        let nearest = palette
            .iter()
            .enumerate()
            .map(|(i, &(rgb, representative))| (i, rgb, delta_e_ok(lab, representative)))
            .min_by(|a, b| a.2.total_cmp(&b.2));
        if let Some((_, rgb, error)) = nearest.filter(|v| v.2 <= 4.0) {
            let _ = error;
            patch.color = rgb;
        } else {
            palette.push((patch.color, lab));
        }
    }
    result
}

#[cfg(test)]
include!("../tests/unit/ink_color.rs");
