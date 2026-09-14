//! Perceptual, structure-aware raster-to-SVG vectorisation.
//!
//! The implementation is native Rust. It does not shell out to the former
//! Python vectorizer or an external SVG renderer. Preview rendering for
//! ownership validation is performed in memory by the embedded `resvg` crate.

pub mod adaptive;
mod alpha_coverage;
mod alpha_lines;
mod alpha_paint;
pub mod chroma;
pub mod color;
mod colour_fields;
pub mod config;
pub mod edge;
mod elementary;
mod extrema;
mod face_alpha;
pub mod geometry;
pub mod gradient;
pub mod hierarchy;
mod ink_color;
pub mod metrics;
mod neutral_fields;
pub mod optimize;
mod outline;
pub mod ownership;
mod paint_order;
pub mod pipeline;
pub mod raster;
pub mod ridge;
pub mod segment;
mod separators;
mod soft_edges;
pub mod structural;
pub mod svg;
mod svg_document;
mod svg_fragments;
mod union_find;
mod visibility;

pub use chroma::{AlphaTransparencySummary, ChromaKeySummary};
pub use config::Config;
pub use pipeline::{vectorize, Summary};

pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;

mod occlusion;
