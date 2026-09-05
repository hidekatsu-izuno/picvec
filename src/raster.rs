use std::path::Path;
use std::sync::Arc;

use image::{imageops::FilterType, DynamicImage, ImageBuffer, ImageReader, Limits, Rgb};

use crate::Result;

#[derive(Clone, Debug)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[f32; 3]>,
}

/// Read-only full-resolution source retained for adaptive refinement.
///
/// Working images stay in `f32`, but decoded source pixels spend most of
/// their lifetime waiting for a small crop to be selected. Keeping those
/// pixels packed cuts that retained allocation without introducing fixed
/// point into colour, filtering, or geometry calculations.
#[derive(Clone, Debug)]
pub(crate) struct SourceRaster {
    pub width: usize,
    pub height: usize,
    pixels: SourcePixels,
}

#[derive(Clone, Debug)]
enum SourcePixels {
    /// Exact representation of the decoder's RGBA8 output.
    Rgb8(Arc<Vec<u8>>),
    /// Q0.16 storage for chroma separation and alpha-composited references.
    Unorm16(Arc<Vec<u16>>),
}

pub(crate) trait RasterSource: Sync {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn get(&self, x: usize, y: usize) -> [f32; 3];
    fn resize_max(&self, maximum: u32) -> Raster;
}

impl Raster {
    pub fn new(width: usize, height: usize, pixels: Vec<[f32; 3]>) -> Self {
        assert_eq!(pixels.len(), width * height);
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn blank(width: usize, height: usize, color: [f32; 3]) -> Self {
        Self::new(width, height, vec![color; width * height])
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 3] {
        self.pixels[self.index(x.min(self.width - 1), y.min(self.height - 1))]
    }

    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> [f32; 3] {
        let px = x.clamp(0, self.width.saturating_sub(1) as isize) as usize;
        let py = y.clamp(0, self.height.saturating_sub(1) as isize) as usize;
        self.get(px, py)
    }

    pub fn load(
        path: &Path,
        maximum_dimension: u32,
        maximum_pixels: u64,
        maximum_decode_bytes: u64,
    ) -> Result<Self> {
        let decoded = Self::decode_limited(
            path,
            maximum_dimension,
            maximum_pixels,
            maximum_decode_bytes,
        )?;
        Ok(Self::from_dynamic(&decoded))
    }

    fn decode_limited(
        path: &Path,
        maximum_dimension: u32,
        maximum_pixels: u64,
        maximum_decode_bytes: u64,
    ) -> Result<DynamicImage> {
        let (width, height) = ImageReader::open(path)?.into_dimensions()?;
        let pixels = u64::from(width) * u64::from(height);
        if pixels > maximum_pixels {
            return Err(format!(
                "input raster has {pixels} pixels ({width}x{height}); limit is {maximum_pixels}"
            )
            .into());
        }
        let mut reader = ImageReader::open(path)?;
        let mut limits = Limits::default();
        limits.max_image_width = Some(maximum_dimension);
        limits.max_image_height = Some(maximum_dimension);
        limits.max_alloc = Some(maximum_decode_bytes);
        reader.limits(limits);
        Ok(reader.decode()?)
    }

    pub fn from_dynamic(image: &DynamicImage) -> Self {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixels = rgba
            .pixels()
            .map(|pixel| {
                let alpha = pixel[3] as f32 / 255.0;
                let inv = 1.0 - alpha;
                [
                    (pixel[0] as f32 / 255.0) * alpha + inv,
                    (pixel[1] as f32 / 255.0) * alpha + inv,
                    (pixel[2] as f32 / 255.0) * alpha + inv,
                ]
            })
            .collect();
        Self::new(width as usize, height as usize, pixels)
    }

    pub fn to_rgb8(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        ImageBuffer::from_fn(self.width as u32, self.height as u32, |x, y| {
            let rgb = self.get(x as usize, y as usize);
            Rgb([
                (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ])
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.to_rgb8().save(path)?;
        Ok(())
    }

    pub fn resize_max(&self, maximum: u32) -> Self {
        let current = self.width.max(self.height) as u32;
        if current <= maximum || maximum == 0 {
            return self.clone();
        }
        let scale = maximum as f64 / current as f64;
        let width = (self.width as f64 * scale).round().max(1.0) as u32;
        let height = (self.height as f64 * scale).round().max(1.0) as u32;
        let resized = image::imageops::resize(&self.to_rgb8(), width, height, FilterType::Lanczos3);
        Self::from_dynamic(&DynamicImage::ImageRgb8(resized))
    }

    pub fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        assert!(x <= self.width && y <= self.height);
        let width = width.min(self.width - x);
        let height = height.min(self.height - y);
        let mut pixels = Vec::with_capacity(width * height);
        for row in y..y + height {
            let start = row * self.width + x;
            pixels.extend_from_slice(&self.pixels[start..start + width]);
        }
        Self::new(width, height, pixels)
    }

    /// Bilinear sampling in pixel-centre coordinates. Coordinates outside the
    /// image are clamped so source/refinement mappings share identical edge
    /// behaviour.
    pub fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x.clamp(0.0, self.width.saturating_sub(1) as f32);
        let y = y.clamp(0.0, self.height.saturating_sub(1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let top = [0, 1, 2]
            .map(|channel| self.get(x0, y0)[channel] * (1.0 - tx) + self.get(x1, y0)[channel] * tx);
        let bottom = [0, 1, 2]
            .map(|channel| self.get(x0, y1)[channel] * (1.0 - tx) + self.get(x1, y1)[channel] * tx);
        [0, 1, 2].map(|channel| top[channel] * (1.0 - ty) + bottom[channel] * ty)
    }

    pub fn from_rgb8(path: &Path) -> Result<Self> {
        Self::load(path, 32_768, 32_000_000, 512 * 1024 * 1024)
    }
}

impl RasterSource for Raster {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> [f32; 3] {
        Raster::get(self, x, y)
    }

    fn resize_max(&self, maximum: u32) -> Raster {
        Raster::resize_max(self, maximum)
    }
}

impl SourceRaster {
    pub(crate) fn load_with_alpha(
        path: &Path,
        maximum_dimension: u32,
        maximum_pixels: u64,
        maximum_decode_bytes: u64,
    ) -> Result<(Self, Option<Vec<u8>>)> {
        let decoded = Raster::decode_limited(
            path,
            maximum_dimension,
            maximum_pixels,
            maximum_decode_bytes,
        )?;
        let rgba = decoded.into_rgba8();
        let (width, height) = rgba.dimensions();
        let len = width as usize * height as usize;
        let mut pixels = Vec::with_capacity(len * 3);
        let mut alpha = None::<Vec<u8>>;
        for (index, pixel) in rgba.pixels().enumerate() {
            pixels.extend([pixel[0], pixel[1], pixel[2]]);
            if pixel[3] != 255 && alpha.is_none() {
                let mut values = Vec::with_capacity(len);
                values.resize(index, 255);
                alpha = Some(values);
            }
            if let Some(values) = &mut alpha {
                values.push(pixel[3]);
            }
        }
        Ok((
            Self {
                width: width as usize,
                height: height as usize,
                pixels: SourcePixels::Rgb8(Arc::new(pixels)),
            },
            alpha,
        ))
    }

    pub(crate) fn from_unorm16_fn(
        width: usize,
        height: usize,
        pixel: impl Fn(usize) -> [f32; 3],
    ) -> Self {
        let len = width * height;
        let mut packed = Vec::with_capacity(len * 3);
        for index in 0..len {
            packed.extend(
                pixel(index).map(|channel| (channel.clamp(0.0, 1.0) * 65_535.0).round() as u16),
            );
        }
        Self {
            width,
            height,
            pixels: SourcePixels::Unorm16(Arc::new(packed)),
        }
    }

    pub(crate) fn from_rgb8_fn(
        width: usize,
        height: usize,
        pixel: impl Fn(usize) -> [f32; 3],
    ) -> Self {
        let len = width * height;
        let mut packed = Vec::with_capacity(len * 3);
        for index in 0..len {
            packed.extend(
                pixel(index).map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8),
            );
        }
        Self {
            width,
            height,
            pixels: SourcePixels::Rgb8(Arc::new(packed)),
        }
    }

    pub(crate) fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Raster {
        assert!(x <= self.width && y <= self.height);
        let width = width.min(self.width - x);
        let height = height.min(self.height - y);
        let mut pixels = Vec::with_capacity(width * height);
        for row in y..y + height {
            for column in x..x + width {
                pixels.push(self.get(column, row));
            }
        }
        Raster::new(width, height, pixels)
    }

    fn to_rgb8(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        if let SourcePixels::Rgb8(values) = &self.pixels {
            return ImageBuffer::from_raw(
                self.width as u32,
                self.height as u32,
                values.as_ref().clone(),
            )
            .expect("packed RGB source length matches its dimensions");
        }
        ImageBuffer::from_fn(self.width as u32, self.height as u32, |x, y| {
            let pixel = self.get(x as usize, y as usize);
            Rgb(pixel.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8))
        })
    }

    fn to_raster(&self) -> Raster {
        Raster::new(
            self.width,
            self.height,
            (0..self.width * self.height)
                .map(|index| self.get(index % self.width, index / self.width))
                .collect(),
        )
    }
}

impl RasterSource for SourceRaster {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> [f32; 3] {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let offset = (y * self.width + x) * 3;
        match &self.pixels {
            SourcePixels::Rgb8(values) => {
                [0, 1, 2].map(|channel| f32::from(values[offset + channel]) / 255.0)
            }
            SourcePixels::Unorm16(values) => {
                [0, 1, 2].map(|channel| f32::from(values[offset + channel]) / 65_535.0)
            }
        }
    }

    fn resize_max(&self, maximum: u32) -> Raster {
        let current = self.width.max(self.height) as u32;
        if current <= maximum || maximum == 0 {
            return self.to_raster();
        }
        let scale = maximum as f64 / current as f64;
        let width = (self.width as f64 * scale).round().max(1.0) as u32;
        let height = (self.height as f64 * scale).round().max(1.0) as u32;
        let resized = image::imageops::resize(&self.to_rgb8(), width, height, FilterType::Lanczos3);
        Raster::from_dynamic(&DynamicImage::ImageRgb8(resized))
    }
}

pub fn percentile(mut values: Vec<f32>, quantile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        values[low]
    } else {
        let amount = position - low as f32;
        values[low] * (1.0 - amount) + values[high] * amount
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_dimensions_are_limited_before_full_decode() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("large.png");
        ImageBuffer::from_pixel(16, 8, Rgb([0_u8, 0, 0]))
            .save(&path)
            .unwrap();

        assert!(Raster::load(&path, 8, 1_000, 64 * 1024 * 1024).is_err());
        assert!(Raster::load(&path, 16, 100, 64 * 1024 * 1024).is_err());
        let loaded = Raster::load(&path, 16, 1_000, 64 * 1024 * 1024).unwrap();
        assert_eq!((loaded.width, loaded.height), (16, 8));
    }

    #[test]
    fn alpha_aware_load_does_not_flatten_rgb_onto_white() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("alpha.png");
        let image = image::RgbaImage::from_pixel(2, 1, image::Rgba([12, 34, 56, 0]));
        image.save(&path).unwrap();
        let (raster, alpha) =
            SourceRaster::load_with_alpha(&path, 64, 64 * 64, 64 * 1024 * 1024).unwrap();
        assert_eq!(raster.get(0, 0), [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]);
        assert_eq!(alpha.unwrap(), vec![0, 0]);
    }

    #[test]
    fn retained_source_uses_rgb8_or_unorm16_storage() {
        let rgb8 = SourceRaster::from_rgb8_fn(2, 1, |index| {
            if index == 0 {
                [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]
            } else {
                [1.0, 0.0, 0.5]
            }
        });
        let SourcePixels::Rgb8(values) = &rgb8.pixels else {
            panic!("decoded source should use RGB8 storage");
        };
        assert_eq!(values.len(), 6);
        assert_eq!(rgb8.get(0, 0), [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]);

        let expected = [[0.123_456, 0.5, 0.987_654], [0.0, 1.0, 0.25]];
        let unorm16 = SourceRaster::from_unorm16_fn(2, 1, |index| expected[index]);
        let SourcePixels::Unorm16(values) = &unorm16.pixels else {
            panic!("derived source should use Q0.16 storage");
        };
        assert_eq!(values.len() * std::mem::size_of::<u16>(), 12);
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = unorm16.get(index, 0);
            for channel in 0..3 {
                assert!((actual[channel] - expected[channel]).abs() <= 0.5 / 65_535.0);
            }
        }
    }

    #[test]
    fn crop_and_bilinear_sampling_preserve_source_coordinates() {
        let raster = Raster::new(
            3,
            2,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        );
        let crop = raster.crop(1, 0, 2, 2);
        assert_eq!((crop.width, crop.height), (2, 2));
        assert_eq!(crop.get(0, 0), [1.0, 0.0, 0.0]);
        assert_eq!(crop.get(1, 1), [1.0, 1.0, 1.0]);
        assert_eq!(raster.sample_bilinear(0.5, 0.0), [0.5, 0.0, 0.0]);
    }
}
