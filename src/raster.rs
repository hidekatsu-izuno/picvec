use std::path::Path;

use image::{imageops::FilterType, DynamicImage, ImageBuffer, ImageReader, Limits, Rgb};

use crate::Result;

#[derive(Clone, Debug)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[f32; 3]>,
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

    /// Decode straight RGB and return the source alpha separately.  Keeping
    /// alpha out of `Raster` avoids making every opaque pipeline allocation
    /// four-channel while allowing the converter entry point to preserve
    /// transparent PNG input without first flattening it onto white.
    pub(crate) fn load_with_alpha(
        path: &Path,
        maximum_dimension: u32,
        maximum_pixels: u64,
        maximum_decode_bytes: u64,
    ) -> Result<(Self, Option<Vec<f32>>)> {
        let decoded = Self::decode_limited(
            path,
            maximum_dimension,
            maximum_pixels,
            maximum_decode_bytes,
        )?;
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        let mut has_transparency = false;
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        let mut alpha = Vec::with_capacity(pixels.capacity());
        for pixel in rgba.pixels() {
            pixels.push([
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ]);
            let value = pixel[3] as f32 / 255.0;
            has_transparency |= pixel[3] != 255;
            alpha.push(value);
        }
        Ok((
            Self::new(width as usize, height as usize, pixels),
            has_transparency.then_some(alpha),
        ))
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
            Raster::load_with_alpha(&path, 64, 64 * 64, 64 * 1024 * 1024).unwrap();
        assert_eq!(raster.pixels[0], [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]);
        assert_eq!(alpha.unwrap(), vec![0.0, 0.0]);
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
