use serde::{Deserialize, Serialize};

/// Reproducible controls corresponding to the current raster2svg defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Strict per-axis limit applied before an input raster is decoded.
    pub maximum_input_dimension: u32,
    /// Strict total-pixel limit checked from the raster header before decode.
    pub maximum_input_pixels: u64,
    /// Best-effort allocation limit passed to the image decoder.
    pub maximum_decode_bytes: u64,
    pub maximum_dimension: u32,
    pub auto_dimension: bool,
    pub auto_minimum_dimension: u32,
    pub auto_maximum_dimension: u32,
    pub smoothing_radius: u32,
    pub smoothing_spatial_sigma: f64,
    pub smoothing_dark_delta_e: f32,
    pub smoothing_light_delta_e: f32,
    pub dark_knee_lstar: f32,
    pub segmentation_min_size: u32,
    pub segmentation_reference_dimension: u32,
    pub local_detail_adaptation: bool,
    pub local_detail_window: f32,
    pub local_detail_density_pivot: f32,
    pub quantization_dark_delta_e: f32,
    pub quantization_light_delta_e: f32,
    pub gradient_merge_error: f32,
    /// Maximum within-region DeltaE00 range treated unconditionally as Solid.
    pub solid_color_max_delta_e: f32,
    pub minimum_gradient_area: u32,
    pub shared_boundary_overlap: f32,
    pub maximum_gradient_stops: usize,
    /// Maximum source samples used by the inexpensive Paint coherence gate.
    pub paint_primary_sample_budget: usize,
    /// Minimum final-region density (regions per processing pixel) at which
    /// the Paint coherence gate is enabled.
    pub paint_primary_min_region_density: f32,
    /// Minimum spatially explained RGB variance before a normal face runs the
    /// complete linear/radial Paint search.
    pub paint_primary_min_explained_variance: f32,
    /// Stricter coherence threshold for faces below `minimum_gradient_area`.
    pub paint_primary_small_min_explained_variance: f32,
    /// Rayon worker count. Zero selects physical cores when discoverable.
    pub rayon_threads: usize,
    /// Render the complete SVG in memory and calculate report-only quality
    /// metrics. Requires the `diagnostics` Cargo feature.
    pub compute_quality_metrics: bool,
    /// Print in-memory progress diagnostics. Requires the `diagnostics` Cargo
    /// feature.
    pub retain_diagnostics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            maximum_input_dimension: 32_768,
            maximum_input_pixels: 32_000_000,
            maximum_decode_bytes: 512 * 1024 * 1024,
            maximum_dimension: 1600,
            auto_dimension: true,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            smoothing_radius: 4,
            smoothing_spatial_sigma: 1.15,
            smoothing_dark_delta_e: 1.8,
            smoothing_light_delta_e: 4.5,
            dark_knee_lstar: 45.0,
            segmentation_min_size: 24,
            segmentation_reference_dimension: 384,
            local_detail_adaptation: true,
            local_detail_window: 12.0,
            local_detail_density_pivot: 0.35,
            quantization_dark_delta_e: 2.5,
            quantization_light_delta_e: 5.0,
            gradient_merge_error: 2.3,
            solid_color_max_delta_e: 1.5,
            minimum_gradient_area: 64,
            shared_boundary_overlap: 0.2,
            maximum_gradient_stops: 5,
            paint_primary_sample_budget: 64,
            paint_primary_min_region_density: 0.015,
            paint_primary_min_explained_variance: 0.08,
            paint_primary_small_min_explained_variance: 0.24,
            rayon_threads: 0,
            compute_quality_metrics: false,
            retain_diagnostics: false,
        }
    }
}

impl Config {
    /// Reject invalid programmatic or deserialized settings before conversion
    /// allocates memory or creates output directories.
    pub fn validate(&self) -> crate::Result<()> {
        fn require(condition: bool, message: &'static str) -> crate::Result<()> {
            if condition {
                Ok(())
            } else {
                Err(message.into())
            }
        }

        require(
            self.maximum_input_dimension >= 64,
            "maximum_input_dimension must be at least 64",
        )?;
        require(
            self.maximum_input_pixels >= 64 * 64,
            "maximum_input_pixels must be at least 4096",
        )?;
        require(
            self.maximum_decode_bytes >= 1024 * 1024,
            "maximum_decode_bytes must be at least 1 MiB",
        )?;
        require(
            self.maximum_dimension >= 64,
            "maximum_dimension must be at least 64",
        )?;
        require(
            self.auto_minimum_dimension >= 64,
            "auto_minimum_dimension must be at least 64",
        )?;
        require(
            self.auto_maximum_dimension >= 64,
            "auto_maximum_dimension must be at least 64",
        )?;
        require(
            self.smoothing_spatial_sigma.is_finite() && self.smoothing_spatial_sigma > 0.0,
            "smoothing_spatial_sigma must be finite and positive",
        )?;
        for (name, value) in [
            ("smoothing_dark_delta_e", self.smoothing_dark_delta_e),
            ("smoothing_light_delta_e", self.smoothing_light_delta_e),
            ("quantization_dark_delta_e", self.quantization_dark_delta_e),
            (
                "quantization_light_delta_e",
                self.quantization_light_delta_e,
            ),
        ] {
            require(
                value.is_finite() && value > 0.0,
                match name {
                    "smoothing_dark_delta_e" => {
                        "smoothing_dark_delta_e must be finite and positive"
                    }
                    "smoothing_light_delta_e" => {
                        "smoothing_light_delta_e must be finite and positive"
                    }
                    "quantization_dark_delta_e" => {
                        "quantization_dark_delta_e must be finite and positive"
                    }
                    _ => "quantization_light_delta_e must be finite and positive",
                },
            )?;
        }
        require(
            self.dark_knee_lstar.is_finite() && (0.0..=100.0).contains(&self.dark_knee_lstar),
            "dark_knee_lstar must be finite and between 0 and 100",
        )?;
        require(
            self.segmentation_min_size >= 1,
            "segmentation_min_size must be at least 1",
        )?;
        require(
            self.segmentation_reference_dimension >= 1,
            "segmentation_reference_dimension must be at least 1",
        )?;
        require(
            self.local_detail_window.is_finite() && self.local_detail_window >= 1.0,
            "local_detail_window must be finite and at least 1",
        )?;
        require(
            self.local_detail_density_pivot.is_finite() && self.local_detail_density_pivot > 0.0,
            "local_detail_density_pivot must be finite and positive",
        )?;
        require(
            self.gradient_merge_error.is_finite() && self.gradient_merge_error >= 0.0,
            "gradient_merge_error must be finite and non-negative",
        )?;
        require(
            self.solid_color_max_delta_e.is_finite() && self.solid_color_max_delta_e >= 0.0,
            "solid_color_max_delta_e must be finite and non-negative",
        )?;
        require(
            self.minimum_gradient_area >= 1,
            "minimum_gradient_area must be at least 1",
        )?;
        require(
            self.shared_boundary_overlap.is_finite() && self.shared_boundary_overlap >= 0.0,
            "shared_boundary_overlap must be finite and non-negative",
        )?;
        require(
            (2..=5).contains(&self.maximum_gradient_stops),
            "maximum_gradient_stops must be between 2 and 5",
        )?;
        require(
            self.paint_primary_sample_budget >= 8,
            "paint_primary_sample_budget must be at least 8",
        )?;
        require(
            self.paint_primary_min_region_density.is_finite()
                && self.paint_primary_min_region_density >= 0.0,
            "paint_primary_min_region_density must be finite and non-negative",
        )?;
        require(
            self.paint_primary_min_explained_variance.is_finite()
                && (0.0..=1.0).contains(&self.paint_primary_min_explained_variance),
            "paint_primary_min_explained_variance must be finite and between 0 and 1",
        )?;
        require(
            self.paint_primary_small_min_explained_variance.is_finite()
                && (0.0..=1.0).contains(&self.paint_primary_small_min_explained_variance),
            "paint_primary_small_min_explained_variance must be finite and between 0 and 1",
        )?;
        Ok(())
    }

    /// Effective automatic bounds. `maximum_dimension` is always a hard
    /// upper bound; the legacy automatic maximum can only lower it.
    pub(crate) fn automatic_dimension_bounds(&self) -> (u32, u32) {
        let maximum = self.maximum_dimension.min(self.auto_maximum_dimension);
        let minimum = self.auto_minimum_dimension.min(maximum);
        (minimum, maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn maximum_dimension_remains_a_hard_automatic_bound() {
        let config = Config {
            maximum_dimension: 320,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            ..Config::default()
        };
        assert_eq!(config.automatic_dimension_bounds(), (320, 320));
    }

    #[test]
    fn nonfinite_threshold_is_rejected() {
        let config = Config {
            paint_primary_min_explained_variance: f32::NAN,
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            solid_color_max_delta_e: f32::INFINITY,
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }
}
