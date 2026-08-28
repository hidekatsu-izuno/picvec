//! Perceptual, structure-aware raster-to-SVG vectorisation.
//!
//! The implementation is native Rust.  It does not shell out to the former
//! Python vectorizer; librsvg is used only for in-memory native-resolution
//! ownership validation, and no rendered sidecar is written.

pub mod color;
pub mod config;
pub mod edge;
pub mod geometry;
pub mod gradient;
pub mod metrics;
pub mod optimize;
pub mod ownership;
pub mod pipeline;
pub mod raster;
pub mod ridge;
pub mod segment;
pub mod structural;
pub mod svg;
mod svml;
mod union_find;

pub use config::Config;
pub use pipeline::{vectorize, Summary};

pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;
