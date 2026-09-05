//! Automatic constant-colour keying for the six saturated RGB cube corners.
//!
//! A single backing image does not uniquely determine foreground colour and
//! coverage in the general matting equation.  The restricted inputs handled
//! here use the classical colour-difference assumption: at least one of the
//! backing's low channels remains no brighter than the foreground while every
//! high channel participates in the key.  This is the six-corner analogue of
//! the Vlahos form discussed by Smith and Blinn, "Blue Screen Matting",
//! SIGGRAPH 1996, DOI 10.1145/237170.237263.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use image::{imageops::FilterType, ImageBuffer, Luma};
use serde::Serialize;

use crate::geometry::Point;
use crate::raster::{Raster, RasterSource, SourceRaster};

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
pub(crate) const SOURCE_ALPHA_QUANTIZATION_BITS: u8 = 2;
const SOURCE_ALPHA_MAXIMUM_LEVEL: u8 = (1 << SOURCE_ALPHA_QUANTIZATION_BITS) - 1;

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
    values: AlphaValues,
}

#[derive(Clone, Debug)]
enum AlphaValues {
    /// Exact synthetic coverage used by focused unit tests.
    #[cfg(test)]
    Float(Vec<f32>),
    /// Q0.16 inferred coverage retained at full source resolution.
    Unorm16(Vec<u16>),
    /// Exact source alpha. The decoder has already reduced the input to eight
    /// bits, so retaining bytes is lossless relative to the previous `f32`
    /// representation.
    Byte(Vec<u8>),
    /// Four exact source-alpha levels, packed four pixels per byte.
    Packed2 { bytes: Vec<u8>, len: usize },
}

/// A marching-squares crossing is identified by the source-sample edge, not
/// by a rounded coordinate. This makes adjacent cells share the exact same
/// vertex even when interpolation produces a non-terminating `f32` value.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AlphaContourVertex {
    Horizontal(i32, i32),
    Vertical(i32, i32),
    CanvasCorner(i32, i32),
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
    /// Maximum bit depth used for durable authored transparency. An
    /// antialiased opaque silhouette may collapse to one binary mask path.
    /// Zero means that source alpha was not present.
    pub quantization_bits: u8,
    /// Number of vector paths used by the resulting alpha mask.
    pub mask_paths: usize,
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
pub(crate) fn detect<R: RasterSource + ?Sized>(image: &R) -> Option<ChromaKey> {
    let border = border_indices(image.width(), image.height());
    if border.is_empty() {
        return None;
    }
    let maximum_squared_distance = KEY_SAMPLE_DISTANCE * KEY_SAMPLE_DISTANCE;
    let mut best = None::<([f32; 3], Vec<[f32; 3]>)>;
    for corner in KEY_CORNERS {
        let samples = border
            .iter()
            .filter_map(|&index| {
                let pixel = image.get(index % image.width(), index / image.width());
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
pub(crate) fn pull_matte<R: RasterSource + ?Sized>(image: &R, key: ChromaKey) -> AlphaMatte {
    let len = image.width() * image.height();
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        let alpha = coverage(image.get(index % image.width(), index / image.width()), key);
        values.push((alpha * 65_535.0).round() as u16);
    }
    AlphaMatte {
        width: image.width(),
        height: image.height(),
        values: AlphaValues::Unorm16(values),
    }
}

/// Choose the saturated RGB corner farthest, on average, from pixels with
/// source coverage.  This temporary backing makes alpha boundaries visible to
/// the ordinary RGB vectorizer and is removed again after geometry fitting.
pub(crate) fn select_alpha_backing<R: RasterSource + ?Sized>(
    image: &R,
    matte: &AlphaMatte,
) -> [f32; 3] {
    KEY_CORNERS
        .into_iter()
        .max_by(|&first, &second| {
            let score = |corner| {
                (0..image.width() * image.height())
                    .zip(matte.iter())
                    .map(|(index, alpha)| {
                        squared_distance(
                            image.get(index % image.width(), index / image.width()),
                            corner,
                        ) * alpha
                    })
                    .sum::<f32>()
            };
            score(first).total_cmp(&score(second))
        })
        .unwrap_or([0.0, 1.0, 0.0])
}

pub(crate) fn composite_over(image: &Raster, matte: &AlphaMatte, backing: [f32; 3]) -> Raster {
    if image.pixels.len() != matte.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(matte.iter())
            .map(|(&pixel, alpha)| {
                let alpha = alpha.clamp(0.0, 1.0);
                [0, 1, 2].map(|channel| pixel[channel] * alpha + backing[channel] * (1.0 - alpha))
            })
            .collect(),
    )
}

/// Build the retained alpha-composited reference directly in Q0.16 storage.
/// It is sampled for adaptive scoring but expanded to `f32` only for pixels
/// that are actually visited or cropped.
pub(crate) fn composite_source_over(
    image: &SourceRaster,
    matte: &AlphaMatte,
    backing: [f32; 3],
) -> SourceRaster {
    assert_eq!(image.width * image.height, matte.len());
    SourceRaster::from_unorm16_fn(image.width, image.height, |index| {
        let pixel = image.get(index % image.width, index / image.width);
        let alpha = matte.get(index).clamp(0.0, 1.0);
        [0, 1, 2].map(|channel| pixel[channel] * alpha + backing[channel] * (1.0 - alpha))
    })
}

fn quantized_alpha_level(alpha: f32) -> u8 {
    (alpha.clamp(0.0, 1.0) * f32::from(SOURCE_ALPHA_MAXIMUM_LEVEL))
        .round()
        .clamp(0.0, f32::from(SOURCE_ALPHA_MAXIMUM_LEVEL)) as u8
}

/// Preserve straight foreground RGB independently of source coverage.
///
/// Pixels with exactly zero source coverage are normalized to a stable colour
/// so arbitrary hidden PNG RGB does not create vector complexity. Every
/// nonzero sample retains straight RGB as underpaint, including samples below
/// the first output-alpha threshold. The interpolated SVG contour can then
/// move within that sample without exposing the normalization colour.
#[cfg(test)]
pub(crate) fn prepare_source_alpha(
    image: &Raster,
    matte: &AlphaMatte,
    transparent_color: [f32; 3],
) -> Raster {
    if image.pixels.len() != matte.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(matte.iter())
            .map(|(&pixel, alpha)| {
                if alpha <= 0.0 {
                    transparent_color
                } else {
                    pixel
                }
            })
            .collect(),
    )
}

pub(crate) fn prepare_compact_source_alpha(
    image: &SourceRaster,
    matte: &AlphaMatte,
    transparent_color: [f32; 3],
) -> SourceRaster {
    assert_eq!(image.width * image.height, matte.len());
    SourceRaster::from_rgb8_fn(image.width, image.height, |index| {
        if matte.get(index) <= 0.0 {
            transparent_color
        } else {
            image.get(index % image.width, index / image.width)
        }
    })
}

/// Turn the soft keyed boundary into vector ownership while removing backing
/// contamination from the retained side.  Pixels below the half-coverage
/// crossing become uniform backing; pixels above it are unmixed with the
/// standard compositing equation before ordinary colour segmentation.
#[cfg(test)]
pub(crate) fn separate_foreground(image: &Raster, matte: &AlphaMatte, backing: [f32; 3]) -> Raster {
    if image.pixels.len() != matte.len() {
        return image.clone();
    }
    Raster::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(matte.iter())
            .map(|(&pixel, alpha)| {
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

pub(crate) fn separate_compact_foreground(
    image: &SourceRaster,
    matte: &AlphaMatte,
    backing: [f32; 3],
) -> SourceRaster {
    assert_eq!(image.width * image.height, matte.len());
    SourceRaster::from_unorm16_fn(image.width, image.height, |index| {
        let pixel = image.get(index % image.width, index / image.width);
        let alpha = matte.get(index).clamp(0.0, 1.0);
        if alpha < BACKGROUND_OWNERSHIP_ALPHA {
            backing
        } else {
            [0, 1, 2].map(|channel| {
                ((pixel[channel] - backing[channel] * (1.0 - alpha)) / alpha.max(1e-6))
                    .clamp(0.0, 1.0)
            })
        }
    })
}

impl AlphaMatte {
    #[cfg(test)]
    pub(crate) fn new(width: usize, height: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), width * height);
        Self {
            width,
            height,
            values: AlphaValues::Float(values),
        }
    }

    pub(crate) fn from_u8(width: usize, height: usize, values: Vec<u8>) -> Self {
        assert_eq!(values.len(), width * height);
        Self {
            width,
            height,
            values: AlphaValues::Byte(values),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.values {
            #[cfg(test)]
            AlphaValues::Float(values) => values.len(),
            AlphaValues::Unorm16(values) => values.len(),
            AlphaValues::Byte(values) => values.len(),
            AlphaValues::Packed2 { len, .. } => *len,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> f32 {
        match &self.values {
            #[cfg(test)]
            AlphaValues::Float(values) => values[index],
            AlphaValues::Unorm16(values) => f32::from(values[index]) / 65_535.0,
            AlphaValues::Byte(values) => f32::from(values[index]) / 255.0,
            AlphaValues::Packed2 { bytes, len } => {
                assert!(index < *len);
                let shift = (index & 3) * SOURCE_ALPHA_QUANTIZATION_BITS as usize;
                f32::from((bytes[index / 4] >> shift) & SOURCE_ALPHA_MAXIMUM_LEVEL)
                    / f32::from(SOURCE_ALPHA_MAXIMUM_LEVEL)
            }
        }
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        (0..self.len()).map(|index| self.get(index))
    }

    #[cfg(test)]
    fn storage_bytes(&self) -> usize {
        match &self.values {
            #[cfg(test)]
            AlphaValues::Float(values) => std::mem::size_of_val(values.as_slice()),
            AlphaValues::Unorm16(values) => std::mem::size_of_val(values.as_slice()),
            AlphaValues::Byte(values) => values.len(),
            AlphaValues::Packed2 { bytes, .. } => bytes.len(),
        }
    }

    #[inline]
    fn quantized_level_at(&self, index: usize) -> u8 {
        match &self.values {
            AlphaValues::Packed2 { bytes, len } => {
                assert!(index < *len);
                let shift = (index & 3) * SOURCE_ALPHA_QUANTIZATION_BITS as usize;
                (bytes[index / 4] >> shift) & SOURCE_ALPHA_MAXIMUM_LEVEL
            }
            _ => quantized_alpha_level(self.get(index)),
        }
    }

    fn packed_2bit_from_levels(
        width: usize,
        height: usize,
        levels: impl IntoIterator<Item = u8>,
    ) -> Self {
        let len = width * height;
        let mut bytes = vec![0_u8; len.div_ceil(4)];
        let mut count = 0_usize;
        for (index, level) in levels.into_iter().enumerate() {
            assert!(index < len);
            let shift = (index & 3) * SOURCE_ALPHA_QUANTIZATION_BITS as usize;
            bytes[index / 4] |= (level & SOURCE_ALPHA_MAXIMUM_LEVEL) << shift;
            count += 1;
        }
        assert_eq!(count, len);
        Self {
            width,
            height,
            values: AlphaValues::Packed2 { bytes, len },
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
                Luma([(self
                    .get(y as usize * self.width + x as usize)
                    .clamp(0.0, 1.0)
                    * 255.0)
                    .round() as u8])
            },
        );
        let resized =
            image::imageops::resize(&source, width as u32, height as u32, FilterType::Lanczos3);
        Self::from_u8(
            width,
            height,
            resized.pixels().map(|pixel| pixel[0]).collect(),
        )
    }

    /// Quantize exact source coverage to the four values representable by a
    /// two-bit alpha channel: 0, 1/3, 2/3, and 1.
    pub(crate) fn quantized_2bit(&self) -> Self {
        if matches!(self.values, AlphaValues::Packed2 { .. }) {
            return self.clone();
        }
        Self::packed_2bit_from_levels(
            self.width,
            self.height,
            (0..self.len()).map(|index| self.quantized_level_at(index)),
        )
    }

    #[cfg(test)]
    pub(crate) fn quantized_levels(&self) -> Vec<u8> {
        (0..self.len())
            .map(|index| self.quantized_level_at(index))
            .collect()
    }

    /// Reduce raster coverage shoulders to durable vector alpha regions.
    ///
    /// A thin connected run of intermediate alpha between clear and opaque is
    /// sampling coverage, not an authored translucent object. Keeping it as
    /// two extra SVG bands creates a visible grey outline at large zoom. Such
    /// pixels are reassigned to their nearest durable alpha region. A partial
    /// component touching only one extreme is retained, as is the eroded core
    /// of a broad transition, so intentional translucent fills and gradients
    /// still survive the conversion.
    pub(crate) fn vectorized_levels(&self) -> Vec<u8> {
        let levels = (0..self.len())
            .map(|index| self.quantized_level_at(index))
            .collect::<Vec<_>>();
        if levels.is_empty() {
            return levels;
        }
        let width = self.width;
        let height = self.height;
        let mut durable = levels
            .iter()
            .map(|&level| level == 0 || level == SOURCE_ALPHA_MAXIMUM_LEVEL)
            .collect::<Vec<_>>();
        let mut visited = vec![false; levels.len()];

        for start in 0..levels.len() {
            if visited[start] || levels[start] == 0 || levels[start] == SOURCE_ALPHA_MAXIMUM_LEVEL {
                continue;
            }
            visited[start] = true;
            let mut pending = vec![start];
            let mut component = Vec::<usize>::new();
            let mut touches_clear = false;
            let mut touches_opaque = false;
            while let Some(index) = pending.pop() {
                component.push(index);
                let x = index % width;
                let y = index / width;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let px = x as isize + dx;
                        let py = y as isize + dy;
                        if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                            touches_clear = true;
                            continue;
                        }
                        let neighbour = py as usize * width + px as usize;
                        match levels[neighbour] {
                            0 => touches_clear = true,
                            SOURCE_ALPHA_MAXIMUM_LEVEL => touches_opaque = true,
                            _ if !visited[neighbour] => {
                                visited[neighbour] = true;
                                pending.push(neighbour);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if !touches_clear || !touches_opaque {
                for index in component {
                    durable[index] = true;
                }
                continue;
            }

            // A 3x3 same-level interior is wider than an ordinary one-pixel
            // coverage shoulder. Retain it as evidence of authored opacity;
            // its boundary samples will be assigned from these core seeds.
            for index in component {
                let x = index % width;
                let y = index / width;
                let level = levels[index];
                durable[index] = x > 0
                    && y > 0
                    && x + 1 < width
                    && y + 1 < height
                    && (y - 1..=y + 1)
                        .all(|py| (x - 1..=x + 1).all(|px| levels[py * width + px] == level));
            }
        }

        let mut distance = vec![usize::MAX; levels.len()];
        let mut owners = vec![u8::MAX; levels.len()];
        let mut queue = VecDeque::<usize>::new();
        for index in 0..levels.len() {
            if durable[index] {
                distance[index] = 0;
                owners[index] = levels[index];
                queue.push_back(index);
            }
        }
        if queue.is_empty() {
            return levels;
        }

        while let Some(index) = queue.pop_front() {
            let x = index % width;
            let y = index / width;
            for neighbour in [
                (x > 0).then(|| index - 1),
                (x + 1 < width).then_some(index + 1),
                (y > 0).then(|| index - width),
                (y + 1 < height).then_some(index + width),
            ]
            .into_iter()
            .flatten()
            {
                if durable[neighbour] {
                    continue;
                }
                let candidate_distance = distance[index].saturating_add(1);
                let candidate_owner = owners[index];
                let target = self.get(neighbour) * f32::from(SOURCE_ALPHA_MAXIMUM_LEVEL);
                let candidate_error = (target - f32::from(candidate_owner)).abs();
                let current_error = (target - f32::from(owners[neighbour])).abs();
                let better_tie = candidate_distance == distance[neighbour]
                    && (candidate_error < current_error
                        || (candidate_error == current_error
                            && candidate_owner < owners[neighbour]));
                if candidate_distance < distance[neighbour] || better_tie {
                    distance[neighbour] = candidate_distance;
                    owners[neighbour] = candidate_owner;
                    queue.push_back(neighbour);
                }
            }
        }
        owners
    }

    fn contour_sample(&self, x: i32, y: i32) -> f32 {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            0.0
        } else {
            self.get(y as usize * self.width + x as usize)
        }
    }

    fn contour_point(&self, vertex: AlphaContourVertex, threshold: f32) -> Point {
        let interpolate = |first: f32, second: f32| {
            if (second - first).abs() <= 1e-8 {
                0.5
            } else {
                ((threshold - first) / (second - first)).clamp(0.0, 1.0)
            }
        };
        match vertex {
            AlphaContourVertex::Horizontal(x, y) => {
                // Crossings against the virtual clear ring belong exactly on
                // the SVG viewport edge. Letting the interpolation threshold
                // move these crossings inward would incorrectly feather an
                // opaque object merely because it touches the canvas border.
                let position = if x < 0 {
                    0.0
                } else if x + 1 >= self.width as i32 {
                    self.width as f32
                } else {
                    x as f32
                        + 0.5
                        + interpolate(self.contour_sample(x, y), self.contour_sample(x + 1, y))
                };
                Point {
                    x: position,
                    y: y as f32 + 0.5,
                }
            }
            AlphaContourVertex::Vertical(x, y) => {
                let position = if y < 0 {
                    0.0
                } else if y + 1 >= self.height as i32 {
                    self.height as f32
                } else {
                    y as f32
                        + 0.5
                        + interpolate(self.contour_sample(x, y), self.contour_sample(x, y + 1))
                };
                Point {
                    x: x as f32 + 0.5,
                    y: position,
                }
            }
            AlphaContourVertex::CanvasCorner(x, y) => Point {
                x: x as f32,
                y: y as f32,
            },
        }
    }

    /// Recover closed alpha isolines in pixel-centre coordinates.
    ///
    /// Exact source alpha is a coverage sample, not a label located at a pixel
    /// edge. Interpolating the crossing between neighbouring samples avoids
    /// baking the source raster staircase into an otherwise resolution-
    /// independent SVG mask. A clear virtual ring closes contours at the
    /// viewport; corner cells are routed through the exact canvas corner so a
    /// fully covered image remains fully covered up to its clip boundary.
    pub(crate) fn isocontours(&self, threshold: f32) -> Vec<Vec<Point>> {
        if self.width == 0 || self.height == 0 || !(0.0..=1.0).contains(&threshold) {
            return Vec::new();
        }
        type Segment = (AlphaContourVertex, AlphaContourVertex);
        let normalized = |first: AlphaContourVertex, second: AlphaContourVertex| -> Segment {
            if first < second {
                (first, second)
            } else {
                (second, first)
            }
        };
        let mut segments = BTreeSet::<Segment>::new();
        let mut add = |first: AlphaContourVertex, second: AlphaContourVertex| {
            if first != second {
                segments.insert(normalized(first, second));
            }
        };

        let width = self.width as i32;
        let height = self.height as i32;
        for y in -1..height {
            for x in -1..width {
                let top_left = self.contour_sample(x, y);
                let top_right = self.contour_sample(x + 1, y);
                let bottom_right = self.contour_sample(x + 1, y + 1);
                let bottom_left = self.contour_sample(x, y + 1);
                let case = u8::from(top_left >= threshold)
                    | (u8::from(top_right >= threshold) << 1)
                    | (u8::from(bottom_right >= threshold) << 2)
                    | (u8::from(bottom_left >= threshold) << 3);
                if case == 0 || case == 15 {
                    continue;
                }
                let top = AlphaContourVertex::Horizontal(x, y);
                let right = AlphaContourVertex::Vertical(x + 1, y);
                let bottom = AlphaContourVertex::Horizontal(x, y + 1);
                let left = AlphaContourVertex::Vertical(x, y);

                // The four padded corner cells otherwise cut diagonally
                // between boundary midpoints and leave a transparent triangle
                // in an opaque canvas corner.
                if x == -1 && y == -1 && case == 4 {
                    let corner = AlphaContourVertex::CanvasCorner(0, 0);
                    add(right, corner);
                    add(corner, bottom);
                    continue;
                }
                if x == width - 1 && y == -1 && case == 8 {
                    let corner = AlphaContourVertex::CanvasCorner(width, 0);
                    add(left, corner);
                    add(corner, bottom);
                    continue;
                }
                if x == -1 && y == height - 1 && case == 2 {
                    let corner = AlphaContourVertex::CanvasCorner(0, height);
                    add(top, corner);
                    add(corner, right);
                    continue;
                }
                if x == width - 1 && y == height - 1 && case == 1 {
                    let corner = AlphaContourVertex::CanvasCorner(width, height);
                    add(top, corner);
                    add(corner, left);
                    continue;
                }

                let centre_inside =
                    0.25 * (top_left + top_right + bottom_right + bottom_left) >= threshold;
                match case {
                    1 => add(top, left),
                    2 => add(top, right),
                    3 => add(left, right),
                    4 => add(right, bottom),
                    5 if centre_inside => {
                        add(top, right);
                        add(bottom, left);
                    }
                    5 => {
                        add(top, left);
                        add(right, bottom);
                    }
                    6 => add(top, bottom),
                    7 => add(left, bottom),
                    8 => add(bottom, left),
                    9 => add(top, bottom),
                    10 if centre_inside => {
                        add(top, left);
                        add(right, bottom);
                    }
                    10 => {
                        add(top, right);
                        add(bottom, left);
                    }
                    11 => add(right, bottom),
                    12 => add(left, right),
                    13 => add(top, right),
                    14 => add(top, left),
                    _ => unreachable!("all marching-squares cases are covered"),
                }
            }
        }

        let mut adjacency = BTreeMap::<AlphaContourVertex, Vec<AlphaContourVertex>>::new();
        for &(first, second) in &segments {
            adjacency.entry(first).or_default().push(second);
            adjacency.entry(second).or_default().push(first);
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }

        let mut remaining = segments;
        let mut contours = Vec::<Vec<Point>>::new();
        while let Some(&(start, following)) = remaining.iter().next() {
            remaining.remove(&normalized(start, following));
            let mut vertices = vec![start];
            let mut current = following;
            let mut closed = false;
            while vertices.len() <= adjacency.len() + 1 {
                if current == start {
                    closed = true;
                    break;
                }
                vertices.push(current);
                let next = adjacency
                    .get(&current)
                    .into_iter()
                    .flatten()
                    .copied()
                    .find(|&candidate| remaining.contains(&normalized(current, candidate)));
                let Some(next) = next else {
                    break;
                };
                remaining.remove(&normalized(current, next));
                current = next;
            }
            if !closed {
                continue;
            }
            let mut points = vertices
                .into_iter()
                .map(|vertex| self.contour_point(vertex, threshold))
                .collect::<Vec<_>>();
            points.dedup_by(|left, right| left.distance(*right) <= 1e-5);
            if points.len() >= 3 {
                contours.push(points);
            }
        }
        contours
    }

    pub(crate) fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        assert!(x <= self.width && y <= self.height);
        let width = width.min(self.width - x);
        let height = height.min(self.height - y);
        if matches!(self.values, AlphaValues::Packed2 { .. }) {
            return Self::packed_2bit_from_levels(
                width,
                height,
                (y..y + height).flat_map(|row| {
                    let start = row * self.width + x;
                    (start..start + width).map(|index| self.quantized_level_at(index))
                }),
            );
        }
        let values = match &self.values {
            AlphaValues::Unorm16(source) => {
                let mut values = Vec::with_capacity(width * height);
                for row in y..y + height {
                    let start = row * self.width + x;
                    values.extend_from_slice(&source[start..start + width]);
                }
                AlphaValues::Unorm16(values)
            }
            AlphaValues::Byte(source) => {
                let mut values = Vec::with_capacity(width * height);
                for row in y..y + height {
                    let start = row * self.width + x;
                    values.extend_from_slice(&source[start..start + width]);
                }
                AlphaValues::Byte(values)
            }
            #[cfg(test)]
            AlphaValues::Float(source) => {
                let mut values = Vec::with_capacity(width * height);
                for row in y..y + height {
                    let start = row * self.width + x;
                    values.extend_from_slice(&source[start..start + width]);
                }
                AlphaValues::Float(values)
            }
            AlphaValues::Packed2 { .. } => unreachable!("packed matte returned above"),
        };
        Self {
            width,
            height,
            values,
        }
    }

    fn sample_nearest(&self, point: Point) -> f32 {
        if self.width == 0 || self.height == 0 || self.is_empty() {
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
        self.get(y * self.width + x)
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
    if labels.len() != matte.len() {
        return vec![false; region_count];
    }
    let mut areas = vec![0_usize; region_count];
    let mut background_owned = vec![0_usize; region_count];
    let mut alpha_sum = vec![0.0_f64; region_count];
    for (&label, alpha) in labels.iter().zip(matte.iter()) {
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
        assert!((matte.get(0) - 0.5).abs() <= 0.5 / 65_535.0);
        assert!((matte.get(1) - 0.5).abs() <= 0.5 / 65_535.0);
    }

    #[test]
    fn green_key_does_not_remove_yellow_or_cyan_foreground() {
        let key = ChromaKey {
            corner: [0.0, 1.0, 0.0],
            sampled: [0.0, 1.0, 0.0],
            border_coverage: 1.0,
        };
        let source = Raster::new(2, 1, vec![[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]);
        assert_eq!(
            pull_matte(&source, key).iter().collect::<Vec<_>>(),
            vec![1.0, 1.0]
        );
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
    fn source_alpha_is_quantized_to_four_uniform_levels() {
        let matte = AlphaMatte::new(8, 1, vec![0.0, 0.10, 0.17, 0.49, 0.50, 0.82, 0.84, 1.0]);
        let quantized = matte.quantized_2bit();
        assert_eq!(
            quantized.iter().collect::<Vec<_>>(),
            vec![
                0.0,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                2.0 / 3.0,
                1.0,
                1.0
            ]
        );
        assert_eq!(quantized.storage_bytes(), 2);
        assert_eq!(matte.quantized_levels(), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn narrow_source_antialias_shoulders_are_not_vector_alpha_regions() {
        let matte = AlphaMatte::new(
            5,
            3,
            [0.0, 0.25, 0.75, 1.0, 1.0]
                .into_iter()
                .cycle()
                .take(15)
                .collect(),
        );
        assert_eq!(
            matte.vectorized_levels(),
            [0, 0, 3, 3, 3]
                .into_iter()
                .cycle()
                .take(15)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn authored_translucent_area_is_retained() {
        let mut values = vec![0.0; 25];
        for y in 1..4 {
            for x in 1..4 {
                values[y * 5 + x] = 0.25;
            }
        }
        let matte = AlphaMatte::new(5, 5, values);
        let levels = matte.vectorized_levels();
        for y in 1..4 {
            for x in 1..4 {
                assert_eq!(levels[y * 5 + x], 1);
            }
        }
    }

    #[test]
    fn broad_partial_alpha_transition_keeps_a_durable_core() {
        let matte = AlphaMatte::new(
            5,
            5,
            [0.0, 0.25, 0.25, 0.25, 1.0]
                .into_iter()
                .cycle()
                .take(25)
                .collect(),
        );
        let levels = matte.vectorized_levels();
        assert_eq!(levels[2 * 5 + 2], 1);
        assert!(levels.contains(&1));
    }

    #[test]
    fn source_alpha_isoline_uses_the_interpolated_subpixel_crossing() {
        let matte = AlphaMatte::new(2, 2, vec![0.0, 0.75, 0.0, 0.75]);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);

        // Pixel centres are x=0.5 and x=1.5. Alpha=0.5 crosses two thirds of
        // the way between them, at x=7/6, rather than at their grid edge x=1.
        let expected = 7.0 / 6.0;
        let crossings = contours[0]
            .iter()
            .filter(|point| (point.x - expected).abs() < 1e-5)
            .count();
        assert_eq!(crossings, 2, "contour was snapped back to pixel edges");
    }

    #[test]
    fn opaque_alpha_isoline_covers_all_four_canvas_corners() {
        let matte = AlphaMatte::new(2, 2, vec![1.0; 4]);
        let contours = matte.isocontours(0.5);
        assert_eq!(contours.len(), 1);
        for corner in [
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 0.0, y: 2.0 },
        ] {
            assert!(contours[0].contains(&corner), "missing corner {corner:?}");
        }
    }

    #[test]
    fn source_alpha_rgb_is_retained_for_every_nonzero_coverage_sample() {
        let image = Raster::new(
            3,
            1,
            vec![[0.8, 0.1, 0.2], [0.1, 0.8, 0.2], [0.1, 0.2, 0.8]],
        );
        let matte = AlphaMatte::new(3, 1, vec![0.0, 0.1, 1.0]);
        let prepared = prepare_source_alpha(&image, &matte, [1.0, 0.0, 1.0]);
        assert_eq!(prepared.pixels[0], [1.0, 0.0, 1.0]);
        assert_eq!(prepared.pixels[1], image.pixels[1]);
        assert_eq!(prepared.pixels[2], image.pixels[2]);
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
        let matte = AlphaMatte::new(5, 1, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let removed = background_regions(&[0, 0, 1, 2, 2], 3, &matte);
        assert_eq!(removed, vec![true, false, true]);
    }

    #[test]
    fn background_region_tolerates_many_antialiased_boundary_samples() {
        let matte = AlphaMatte::new(
            10,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.20, 0.30],
        );
        assert_eq!(background_regions(&[0; 10], 1, &matte), vec![true]);

        let foreground = AlphaMatte::new(
            10,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(background_regions(&[0; 10], 1, &foreground), vec![false]);
    }
}
