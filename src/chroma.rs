//! Automatic constant-colour keying for the six saturated RGB cube corners.
//!
//! A single backing image does not uniquely determine foreground colour and
//! coverage in the general matting equation.  The restricted inputs handled
//! here use the classical colour-difference assumption: at least one of the
//! backing's low channels remains no brighter than the foreground while every
//! high channel participates in the key.  This is the six-corner analogue of
//! the Vlahos form discussed by Smith and Blinn, "Blue Screen Matting",
//! SIGGRAPH 1996, DOI 10.1145/237170.237263.

use image::{imageops::FilterType, ImageBuffer, Luma};
use serde::Serialize;

use crate::geometry::Point;
use crate::raster::Raster;

const KEY_CORNERS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
];
const KEY_SAMPLE_DISTANCE: f32 = 56.0 / 255.0;
const MINIMUM_BORDER_COVERAGE: f32 = 0.20;
const MAXIMUM_BORDER_BAND_DEPTH: usize = 64;
const BACKGROUND_OWNERSHIP_ALPHA: f32 = 0.50;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChromaKey {
    pub corner: [f32; 3],
    pub sampled: [f32; 3],
    pub border_coverage: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct AlphaMatte {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ChromaKeySummary {
    pub enabled: bool,
    pub detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_color: Option<[u8; 3]>,
    pub border_coverage: f32,
    pub removed_regions: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AlphaTransparencySummary {
    pub detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporary_backing_color: Option<[u8; 3]>,
    pub removed_regions: usize,
}

fn squared_distance(first: [f32; 3], second: [f32; 3]) -> f32 {
    (first[0] - second[0]).powi(2) + (first[1] - second[1]).powi(2) + (first[2] - second[2]).powi(2)
}

fn border_band_depth(width: usize, height: usize) -> usize {
    let minimum_dimension = width.min(height);
    if minimum_dimension == 0 {
        return 0;
    }
    // Sample roughly the outer 1.5% so a narrow neutral frame or a few rows
    // of antialiasing do not hide the backing.  Keep the band shallow enough
    // that saturated colours well inside the subject cannot drive detection.
    minimum_dimension
        .div_ceil(64)
        .clamp(2, MAXIMUM_BORDER_BAND_DEPTH)
        .min(minimum_dimension.div_ceil(2))
}

fn border_indices(width: usize, height: usize) -> Vec<usize> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let depth = border_band_depth(width, height);
    let interior_width = width.saturating_sub(2 * depth);
    let interior_height = height.saturating_sub(2 * depth);
    let mut indices = Vec::with_capacity(
        width
            .saturating_mul(height)
            .saturating_sub(interior_width.saturating_mul(interior_height)),
    );
    for y in 0..height {
        if y < depth || y >= height - depth {
            indices.extend(y * width..(y + 1) * width);
        } else {
            indices.extend(y * width..y * width + depth);
            indices.extend((y + 1) * width - depth..(y + 1) * width);
        }
    }
    indices
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

/// Pick a key only when a saturated, non-white/non-black RGB corner occupies
/// a meaningful part of a shallow outer band.  Sampling a band instead of the
/// exact perimeter tolerates narrow frames and edge resampling.  The
/// component-wise median then represents small encoder, lighting, or
/// quantisation errors in the backing.
pub(crate) fn detect(image: &Raster) -> Option<ChromaKey> {
    let border = border_indices(image.width, image.height);
    if border.is_empty() {
        return None;
    }
    let maximum_squared_distance = KEY_SAMPLE_DISTANCE * KEY_SAMPLE_DISTANCE;
    let mut best = None::<([f32; 3], Vec<[f32; 3]>)>;
    for corner in KEY_CORNERS {
        let samples = border
            .iter()
            .filter_map(|&index| {
                let pixel = image.pixels[index];
                (squared_distance(pixel, corner) <= maximum_squared_distance).then_some(pixel)
            })
            .collect::<Vec<_>>();
        if best
            .as_ref()
            .is_none_or(|(_, current)| samples.len() > current.len())
        {
            best = Some((corner, samples));
        }
    }
    let (corner, samples) = best?;
    let coverage = samples.len() as f32 / border.len() as f32;
    if samples.len() < 4 || coverage < MINIMUM_BORDER_COVERAGE {
        return None;
    }
    let sampled = [0, 1, 2].map(|channel| {
        median(
            samples
                .iter()
                .map(|pixel| pixel[channel])
                .collect::<Vec<_>>(),
        )
    });
    Some(ChromaKey {
        corner,
        sampled,
        border_coverage: coverage,
    })
}

fn minimum_channel(pixel: [f32; 3], selected: impl Iterator<Item = usize>) -> f32 {
    selected
        .map(|channel| pixel[channel])
        .fold(f32::INFINITY, f32::min)
}

fn maximum_channel(pixel: [f32; 3], selected: impl Iterator<Item = usize>) -> f32 {
    selected
        .map(|channel| pixel[channel])
        .fold(f32::NEG_INFINITY, f32::max)
}

fn coverage(pixel: [f32; 3], key: ChromaKey) -> f32 {
    let high = |channel: &usize| key.corner[*channel] > 0.5;
    let low = |channel: &usize| key.corner[*channel] < 0.5;
    let channels = [0_usize, 1, 2];
    // min(high)-max(low) measures only the chroma unique to the selected RGB
    // corner.  For example, yellow and cyan are foreground under a green key
    // because one of green's low channels is also high.  Values outside the
    // backing-to-neutral interval are foreground, hence the final clamp.
    let key_signal = minimum_channel(key.sampled, channels.iter().filter(|c| high(c)).copied())
        - maximum_channel(key.sampled, channels.iter().filter(|c| low(c)).copied());
    if !key_signal.is_finite() || key_signal <= 0.25 {
        return 1.0;
    }
    let signal = minimum_channel(pixel, channels.iter().filter(|c| high(c)).copied())
        - maximum_channel(pixel, channels.iter().filter(|c| low(c)).copied());
    (1.0 - signal / key_signal).clamp(0.0, 1.0)
}

/// Pull a soft matte from an already keyed opaque raster.  The raster itself
/// remains on its detected backing throughout vectorization; only matching
/// vector regions are omitted from the final SVG.
pub(crate) fn pull_matte(image: &Raster, key: ChromaKey) -> AlphaMatte {
    let mut values = Vec::with_capacity(image.pixels.len());
    for &pixel in &image.pixels {
        values.push(coverage(pixel, key));
    }
    AlphaMatte {
        width: image.width,
        height: image.height,
        values,
    }
}

/// Choose the saturated RGB corner farthest, on average, from pixels with
/// source coverage.  This temporary backing makes alpha boundaries visible to
/// the ordinary RGB vectorizer and is removed again after geometry fitting.
pub(crate) fn select_alpha_backing(image: &Raster, matte: &AlphaMatte) -> [f32; 3] {
    KEY_CORNERS
        .into_iter()
        .max_by(|&first, &second| {
            let score = |corner| {
                image
                    .pixels
                    .iter()
                    .zip(&matte.values)
                    .map(|(&pixel, &alpha)| squared_distance(pixel, corner) * alpha)
                    .sum::<f32>()
            };
            score(first).total_cmp(&score(second))
        })
        .unwrap_or([0.0, 1.0, 0.0])
}

pub(crate) fn composite_over(image: &Raster, matte: &AlphaMatte, backing: [f32; 3]) -> Raster {
    if image.pixels.len() != matte.values.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(&matte.values)
            .map(|(&pixel, &alpha)| {
                let alpha = alpha.clamp(0.0, 1.0);
                [0, 1, 2].map(|channel| pixel[channel] * alpha + backing[channel] * (1.0 - alpha))
            })
            .collect(),
    )
}

/// Turn the soft keyed boundary into vector ownership while removing backing
/// contamination from the retained side.  Pixels below the half-coverage
/// crossing become uniform backing; pixels above it are unmixed with the
/// standard compositing equation before ordinary colour segmentation.
pub(crate) fn separate_foreground(image: &Raster, matte: &AlphaMatte, backing: [f32; 3]) -> Raster {
    if image.pixels.len() != matte.values.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(&matte.values)
            .map(|(&pixel, &alpha)| {
                let alpha = alpha.clamp(0.0, 1.0);
                if alpha < BACKGROUND_OWNERSHIP_ALPHA {
                    backing
                } else {
                    [0, 1, 2].map(|channel| {
                        ((pixel[channel] - backing[channel] * (1.0 - alpha)) / alpha.max(1e-6))
                            .clamp(0.0, 1.0)
                    })
                }
            })
            .collect(),
    )
}

impl AlphaMatte {
    pub(crate) fn new(width: usize, height: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), width * height);
        Self {
            width,
            height,
            values,
        }
    }

    pub(crate) fn resized(&self, width: usize, height: usize) -> Self {
        if self.width == width && self.height == height {
            return self.clone();
        }
        let source = ImageBuffer::<Luma<u8>, Vec<u8>>::from_fn(
            self.width as u32,
            self.height as u32,
            |x, y| {
                Luma([
                    (self.values[y as usize * self.width + x as usize].clamp(0.0, 1.0) * 255.0)
                        .round() as u8,
                ])
            },
        );
        let resized =
            image::imageops::resize(&source, width as u32, height as u32, FilterType::Lanczos3);
        let values = resized
            .pixels()
            .map(|pixel| pixel[0] as f32 / 255.0)
            .collect();
        Self {
            width,
            height,
            values,
        }
    }

    pub(crate) fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        assert!(x <= self.width && y <= self.height);
        let width = width.min(self.width - x);
        let height = height.min(self.height - y);
        let mut values = Vec::with_capacity(width * height);
        for row in y..y + height {
            let start = row * self.width + x;
            values.extend_from_slice(&self.values[start..start + width]);
        }
        Self {
            width,
            height,
            values,
        }
    }

    fn sample_nearest(&self, point: Point) -> f32 {
        if self.width == 0 || self.height == 0 || self.values.is_empty() {
            return 1.0;
        }
        let x = point
            .x
            .round()
            .clamp(0.0, self.width.saturating_sub(1) as f32) as usize;
        let y = point
            .y
            .round()
            .clamp(0.0, self.height.saturating_sub(1) as f32) as usize;
        self.values[y * self.width + x]
    }

    pub(crate) fn retains_stroke(&self, points: &[Point]) -> bool {
        if points.is_empty() {
            return true;
        }
        let mut background_owned = 0_usize;
        let mut alpha_sum = 0.0_f64;
        for &point in points {
            let alpha = self.sample_nearest(point);
            background_owned += usize::from(alpha < BACKGROUND_OWNERSHIP_ALPHA);
            alpha_sum += f64::from(alpha);
        }
        background_owned * 2 < points.len()
            || alpha_sum / points.len() as f64 >= f64::from(BACKGROUND_OWNERSHIP_ALPHA)
    }
}

pub(crate) fn background_regions(
    labels: &[u32],
    region_count: usize,
    matte: &AlphaMatte,
) -> Vec<bool> {
    if labels.len() != matte.values.len() {
        return vec![false; region_count];
    }
    let mut areas = vec![0_usize; region_count];
    let mut background_owned = vec![0_usize; region_count];
    let mut alpha_sum = vec![0.0_f64; region_count];
    for (&label, &alpha) in labels.iter().zip(&matte.values) {
        let label = label as usize;
        if label >= region_count {
            continue;
        }
        areas[label] += 1;
        alpha_sum[label] += alpha as f64;
        if alpha < BACKGROUND_OWNERSHIP_ALPHA {
            background_owned[label] += 1;
        }
    }
    (0..region_count)
        .map(|label| {
            areas[label] > 0
                && background_owned[label] * 2 > areas[label]
                && alpha_sum[label] / (areas[label] as f64) < f64::from(BACKGROUND_OWNERSHIP_ALPHA)
        })
        .collect()
}

impl ChromaKey {
    pub(crate) fn summary(self, enabled: bool, removed_regions: usize) -> ChromaKeySummary {
        ChromaKeySummary {
            enabled,
            detected: true,
            key_color: Some(
                self.sampled
                    .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8),
            ),
            border_coverage: self.border_coverage,
            removed_regions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_accepts_six_keys_but_not_white_or_black() {
        for corner in KEY_CORNERS {
            let image = Raster::blank(8, 8, corner);
            assert_eq!(detect(&image).unwrap().corner, corner);
        }
        assert!(detect(&Raster::blank(8, 8, [1.0; 3])).is_none());
        assert!(detect(&Raster::blank(8, 8, [0.0; 3])).is_none());
    }

    #[test]
    fn detection_looks_past_a_narrow_neutral_outer_frame() {
        let mut image = Raster::blank(512, 384, [0.0, 1.0, 0.0]);
        for y in 0..image.height {
            for x in 0..image.width {
                if x < 3 || y < 3 || x + 3 >= image.width || y + 3 >= image.height {
                    image.pixels[y * image.width + x] = [0.9; 3];
                }
            }
        }
        // An unrelated saturated colour in the interior must not become the
        // key merely because the exact perimeter is neutral.
        for y in 96..288 {
            for x in 128..384 {
                image.pixels[y * image.width + x] = [1.0, 0.0, 0.0];
            }
        }
        let key = detect(&image).expect("green backing behind the trim");
        assert_eq!(key.corner, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn colour_difference_matte_removes_antialias_key_contamination() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        // Half-covered neutral gray and black over green.
        let source = Raster::new(2, 1, vec![[0.25, 0.75, 0.25], [0.0, 0.5, 0.0]]);
        let matte = pull_matte(&source, key);
        assert!((matte.values[0] - 0.5).abs() < 1e-6);
        assert!((matte.values[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn green_key_does_not_remove_yellow_or_cyan_foreground() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        let source = Raster::new(2, 1, vec![[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]);
        assert_eq!(pull_matte(&source, key).values, vec![1.0, 1.0]);
    }

    #[test]
    fn source_alpha_is_composited_over_a_saturated_temporary_backing() {
        let image = Raster::new(2, 1, vec![[1.0, 0.0, 0.0], [0.5; 3]]);
        let matte = AlphaMatte::new(2, 1, vec![0.0, 0.5]);
        let composed = composite_over(&image, &matte, [0.0, 1.0, 0.0]);
        assert_eq!(composed.pixels[0], [0.0, 1.0, 0.0]);
        assert_eq!(composed.pixels[1], [0.25, 0.75, 0.25]);
    }

    #[test]
    fn foreground_side_is_unmixed_and_background_side_is_normalized() {
        let image = Raster::new(2, 1, vec![[0.3, 0.7, 0.3], [0.2, 0.8, 0.2]]);
        let matte = AlphaMatte::new(2, 1, vec![0.6, 0.4]);
        let separated = separate_foreground(&image, &matte, [0.0, 1.0, 0.0]);
        assert!(separated.pixels[0]
            .into_iter()
            .all(|channel| (channel - 0.5).abs() < 1e-6));
        assert_eq!(separated.pixels[1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn structural_stroke_on_background_side_is_rejected() {
        let matte = AlphaMatte::new(3, 1, vec![0.1, 0.2, 0.9]);
        assert!(!matte.retains_stroke(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
        ]));
        assert!(matte.retains_stroke(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
        ]));
    }

    #[test]
    fn disconnected_clear_labels_are_all_background() {
        let matte = AlphaMatte {
            width: 5,
            height: 1,
            values: vec![0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let removed = background_regions(&[0, 0, 1, 2, 2], 3, &matte);
        assert_eq!(removed, vec![true, false, true]);
    }

    #[test]
    fn background_region_tolerates_many_antialiased_boundary_samples() {
        let matte = AlphaMatte {
            width: 10,
            height: 1,
            values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.20, 0.30],
        };
        assert_eq!(background_regions(&[0; 10], 1, &matte), vec![true]);

        let foreground = AlphaMatte {
            width: 10,
            height: 1,
            values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0, 1.0],
        };
        assert_eq!(background_regions(&[0; 10], 1, &foreground), vec![false]);
    }
}
