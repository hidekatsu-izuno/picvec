use serde::{Deserialize, Serialize};

/// Reproducible controls corresponding to the current raster2svg defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
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
    pub minimum_gradient_area: u32,
    pub shared_boundary_overlap: f32,
    pub maximum_gradient_stops: usize,
    pub retain_diagnostics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            maximum_dimension: 1600,
            auto_dimension: true,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            smoothing_radius: 2,
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
            minimum_gradient_area: 64,
            shared_boundary_overlap: 0.2,
            maximum_gradient_stops: 5,
            retain_diagnostics: false,
        }
    }
}
