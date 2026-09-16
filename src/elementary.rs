//! Portable SIMD elementary-function helpers.
//!
//! Vectors are evaluated by `wide`, which selects the SIMD representation
//! available for the compilation target. Partial final vectors are padded and
//! evaluated through the same path. This keeps one implementation across
//! architectures without target-specific ABI shims or vendored assembly.

use wide::{f32x8, f64x4};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

pub fn exp_f32_in_place(values: &mut [f32]) {
    for chunk in values.chunks_mut(F32_LANES) {
        let mut input = [0.0_f32; F32_LANES];
        input[..chunk.len()].copy_from_slice(chunk);
        let output = f32x8::new(input).exp().to_array();
        chunk.copy_from_slice(&output[..chunk.len()]);
    }
}

pub fn exp_f64(value: f64) -> f64 {
    value.exp()
}

pub fn exp_f64_in_place(values: &mut [f64]) {
    for chunk in values.chunks_mut(F64_LANES) {
        let mut input = [0.0_f64; F64_LANES];
        input[..chunk.len()].copy_from_slice(chunk);
        let output = f64x4::new(input).exp().to_array();
        chunk.copy_from_slice(&output[..chunk.len()]);
    }
}

pub fn pow_f32_in_place(values: &mut [f32], exponent: f32) {
    for chunk in values.chunks_mut(F32_LANES) {
        let mut input = [1.0_f32; F32_LANES];
        input[..chunk.len()].copy_from_slice(chunk);
        let lanes = f32x8::new(input);
        let output = if exponent == 3.0 {
            lanes * lanes * lanes
        } else {
            lanes.powf_simd(f32x8::splat(exponent))
        };
        let output = output.to_array();
        chunk.copy_from_slice(&output[..chunk.len()]);
    }
}

#[cfg(test)]
include!("../tests/unit/elementary.rs");
