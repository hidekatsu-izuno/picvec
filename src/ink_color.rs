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
mod tests {
    use super::*;

    #[test]
    fn antialias_at_a_displaced_stroke_edge_is_not_an_interior_colour() {
        let paint = Raster::new(24, 16, vec![[0.9, 0.8, 0.7]; 384]);
        let mut source = paint.clone();
        let mut before = paint.clone();
        let mut white = paint.clone();
        let mut black = paint.clone();
        for x in 3..21 {
            for y in 6..10 {
                let i = y * 24 + x;
                source.pixels[i] = [0.05; 3];
                before.pixels[i] = [0.05; 3];
                white.pixels[i] = [1.0; 3];
                black.pixels[i] = [0.0; 3];
            }
            // The other incident paint differs from the paint underlay.
            // This 70%-covered edge consequently fails the two-colour fit,
            // despite not being an independent mark inside the source ink.
            source.pixels[6 * 24 + x] = [0.065, 0.275, 0.335];
            source.pixels[5 * 24 + x] = [0.1, 0.8, 1.0];
        }
        assert!(propose(&source, &paint, &before, &white, &black).is_empty());
    }

    #[test]
    fn darker_ink_on_the_same_colour_axis_is_not_a_patch_material() {
        let paint = Raster::new(12, 8, vec![[0.6; 3]; 96]);
        let before = Raster::new(12, 8, vec![[0.12; 3]; 96]);
        let white = Raster::new(12, 8, vec![[1.0; 3]; 96]);
        let black = Raster::new(12, 8, vec![[0.0; 3]; 96]);
        let mut source = before.clone();
        // Raster phase alternates between a dark core and partial coverage.
        // A brighter uniform stroke is a colour-estimation error, not a
        // connected secondary material to be traced as pixel rectangles.
        for x in 2..10 {
            source.pixels[3 * 12 + x] = [0.01; 3];
            source.pixels[4 * 12 + x] = [0.04; 3];
        }
        assert!(propose(&source, &paint, &before, &white, &black).is_empty());
        for x in 3..8 {
            source.pixels[3 * 12 + x] = [0.02, 0.04, 0.4];
        }
        assert!(!propose(&source, &paint, &before, &white, &black).is_empty());
    }

    #[test]
    fn interior_colours_are_distinct_from_coverage_and_isolated_noise() {
        for vertical in [false, true] {
            for color in [[0.9, 0.05, 0.1], [0.05, 0.2, 0.9], [0.1, 0.8, 0.2]] {
                let paint = Raster::new(24, 24, vec![[0.9, 0.8, 0.7]; 576]);
                let mut source = paint.clone();
                let mut before = paint.clone();
                let mut white = paint.clone();
                let mut black = paint.clone();
                let index = |t: usize, n: usize| if vertical { t * 24 + n } else { n * 24 + t };
                for t in 3..21 {
                    for n in 10..14 {
                        let i = index(t, n);
                        source.pixels[i] = [0.05; 3];
                        before.pixels[i] = [0.05; 3];
                        white.pixels[i] = [1.0; 3];
                        black.pixels[i] = [0.0; 3];
                    }
                }
                for t in 8..13 {
                    source.pixels[index(t, 11)] = color;
                }
                // A lone mismatching pixel is insufficient region evidence.
                source.pixels[index(20, 12)] = color;
                // A coverage mixture of the existing paint and ink is not a
                // third material, even when it differs strongly from the line.
                source.pixels[index(4, 11)] = [0.475, 0.425, 0.375];
                source.pixels[index(5, 11)] = [0.475, 0.425, 0.375];
                source.pixels[index(15, 12)] = [0.85, 0.78, 0.8];
                source.pixels[index(16, 12)] = [0.85, 0.78, 0.8];
                let patches = propose(&source, &paint, &before, &white, &black);
                assert_eq!(patches.len(), 1);
                assert_eq!(patches[0].pixels.len(), 5);
                let mut after = before.clone();
                for &i in &patches[0].pixels {
                    after.pixels[i] = patches[0].color;
                }
                assert!(patches[0].improves(&source, &before, &after));
                assert!(!patches[0].improves(&source, &before, &before));
                // A distinct shade inside a supported material is not an
                // isolated noise pixel and must not disappear during grouping.
                source.pixels[index(10, 11)] = [0.45, 0.15, 0.4];
                let shaded = propose(&source, &paint, &before, &white, &black);
                assert_eq!(shaded.iter().map(|p| p.pixels.len()).sum::<usize>(), 5);
                // Outside actual stroke coverage, the same colour is paint's
                // responsibility and cannot trigger a repair overlay.
                for &i in &patches[0].pixels {
                    white.pixels[i] = black.pixels[i];
                }
                assert!(propose(&source, &paint, &before, &white, &black).is_empty());
            }
        }
    }
}
