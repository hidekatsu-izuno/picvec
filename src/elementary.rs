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

pub fn cbrt_f32_in_place(values: &mut [f32]) {
    for chunk in values.chunks_mut(F32_LANES) {
        let mut input = [0.0_f32; F32_LANES];
        input[..chunk.len()].copy_from_slice(chunk);
        let output = f32x8::new(input).cbrt().to_array();
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

pub fn atan2_f32(first: &[f32], second: &[f32]) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    let mut result = Vec::with_capacity(first.len());
    atan2_f32_into(first, second, &mut result);
    result
}

pub fn atan2_f32_into(first: &[f32], second: &[f32], result: &mut Vec<f32>) {
    assert_eq!(first.len(), second.len());
    result.clear();
    result.reserve(first.len());
    for (first_chunk, second_chunk) in first.chunks(F32_LANES).zip(second.chunks(F32_LANES)) {
        let mut y = [1.0_f32; F32_LANES];
        let mut x = [1.0_f32; F32_LANES];
        y[..first_chunk.len()].copy_from_slice(first_chunk);
        x[..second_chunk.len()].copy_from_slice(second_chunk);
        let output = f32x8::new(y).atan2(f32x8::new(x)).to_array();
        result.extend_from_slice(&output[..first_chunk.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_f32(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} differs from {expected}"
        );
    }

    #[test]
    fn portable_simd_functions_cover_vectors_and_tails() {
        for length in [0_usize, 1, 7, 8, 15, 16, 17, 31] {
            let source = (0..length)
                .map(|index| 0.01 + index as f32 * 0.17)
                .collect::<Vec<_>>();

            let mut roots = source.clone();
            cbrt_f32_in_place(&mut roots);
            for (&actual, &value) in roots.iter().zip(&source) {
                close_f32(actual, value.cbrt(), 2e-5);
            }

            let mut powers = source.clone();
            pow_f32_in_place(&mut powers, 2.4);
            for (&actual, &value) in powers.iter().zip(&source) {
                close_f32(actual, value.powf(2.4), 2e-5);
            }

            let second = source
                .iter()
                .enumerate()
                .map(|(index, &value)| value + 0.3 + index as f32 * 0.01)
                .collect::<Vec<_>>();
            let angles = atan2_f32(&source, &second);
            for ((&actual, &y), &x) in angles.iter().zip(&source).zip(&second) {
                close_f32(actual, y.atan2(x), 2e-5);
            }

            let mut exponentials = (0..length)
                .map(|index| index as f64 * 0.125 - 2.0)
                .collect::<Vec<_>>();
            let expected = exponentials
                .iter()
                .map(|value| value.exp())
                .collect::<Vec<_>>();
            exp_f64_in_place(&mut exponentials);
            for (&actual, &expected) in exponentials.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
                    "{actual} differs from {expected}"
                );
            }
        }
    }

    #[test]
    fn exp_f32_covers_full_and_partial_vectors() {
        let mut values = [-10.0_f32, -2.0, -0.0, 0.25, 1.0, 4.0, 16.0, 80.0, 0.75];
        let expected = values.map(f32::exp);
        exp_f32_in_place(&mut values);
        for (&actual, &expected) in values.iter().zip(&expected) {
            close_f32(actual, expected, 2e-5);
        }
    }
}
