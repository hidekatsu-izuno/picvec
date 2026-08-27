use std::path::Path;

use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};

use crate::{Error, Result};

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

    pub fn load(path: &Path) -> Result<Self> {
        let decoded = image::open(path)?;
        Ok(Self::from_dynamic(&decoded))
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

    pub fn from_rgb8(path: &Path) -> Result<Self> {
        Self::load(path).map_err(|error| -> Error { error })
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
