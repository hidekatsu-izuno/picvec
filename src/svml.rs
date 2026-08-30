//! NumPy-compatible dispatched elementary functions.
//!
//! NumPy 2.4.4 dispatches contiguous float32 `cbrt` arrays to Intel's
//! low-accuracy AVX-512 SVML routine when AVX512_SKX is available.  The
//! Python reference consequently uses those exact results in `rgb2lab`.
//! Keep the same dispatch here and retain the ordinary scalar fallback on
//! other CPUs.  The vendored routine is BSD-3-Clause, copyright Intel.

// The vendored entry shims use the SysV x86-64 calling convention and ELF
// assembler directives. Keep them Linux-only until another target has a
// native ABI shim of its own.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
    include_str!("svml/cbrt_f32_avx512.s"),
    options(raw, att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
    include_str!("svml/pow_f32_avx512.s"),
    options(raw, att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
    include_str!("svml/atan2_f32_avx512.s"),
    options(raw, att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
    include_str!("svml/exp_f64_avx512.s"),
    options(raw, att_syntax)
);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
unsafe extern "C" {
    fn picvec_svml_cbrtf16(input: *const f32, output: *mut f32);
    fn picvec_svml_powf16(first: *const f32, second: *const f32, output: *mut f32);
    fn picvec_svml_atan2f16(first: *const f32, second: *const f32, output: *mut f32);
    fn picvec_svml_exp8(input: *const f64, output: *mut f64);
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[inline]
fn avx512_svml_available() -> bool {
    // Match NumPy's AVX512_SKX dispatch level. Checking the complete
    // Skylake-X feature set also keeps future upstream routine replacements
    // safe instead of relying on today's instruction mix.
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

pub fn exp_f32_in_place(values: &mut [f32]) {
    values
        .iter_mut()
        .for_each(|value| *value = numpy_exp_f32(*value));
}

pub fn exp_f64(value: f64) -> f64 {
    value.exp()
}

pub fn exp_f64_in_place(values: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if avx512_svml_available() {
        for chunk in values.chunks_mut(8) {
            let mut input = [0.0_f64; 8];
            let mut output = [0.0_f64; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: runtime detection guarantees AVX512_SKX on the
            // SysV target expected by the shim; both arrays provide all lanes.
            unsafe {
                picvec_svml_exp8(input.as_ptr(), output.as_mut_ptr());
            }
            chunk.copy_from_slice(&output[..chunk.len()]);
        }
        return;
    }
    values.iter_mut().for_each(|value| *value = value.exp());
}

fn numpy_exp_f32(mut value: f32) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    if value >= 88.722_84 {
        return f32::INFINITY;
    }
    if value <= -103.972_084 {
        return 0.0;
    }
    let mut quadrant = value * std::f32::consts::LOG2_E;
    quadrant = (quadrant + 12_582_912.0) - 12_582_912.0;
    value = quadrant.mul_add(-6.931_457_5e-1, value);
    value = quadrant.mul_add(-1.428_606_8e-6, value);

    let mut numerator = 5.082_763e-4_f32.mul_add(value, 6.757_897e-3);
    numerator = numerator.mul_add(value, 5.114_512e-2);
    numerator = numerator.mul_add(value, 2.473_615_4e-1);
    numerator = numerator.mul_add(value, 7.257_665e-1);
    numerator = numerator.mul_add(value, 1.0);
    let mut denominator = 2.159_509_4e-2_f32.mul_add(value, -2.742_335_5e-1);
    denominator = denominator.mul_add(value, 1.0);
    let polynomial = numerator / denominator;
    let exponent = quadrant as i32;
    let scale = if exponent >= -126 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else if exponent >= -149 {
        f32::from_bits(1_u32 << (exponent + 149))
    } else {
        0.0
    };
    polynomial * scale
}

pub fn cbrt_f32_in_place(values: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if avx512_svml_available() {
        for chunk in values.chunks_mut(16) {
            let mut input = [0.0_f32; 16];
            let mut output = [0.0_f32; 16];
            input[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: runtime detection guarantees AVX512_SKX on the
            // SysV target expected by the shim; both arrays provide all lanes.
            unsafe {
                picvec_svml_cbrtf16(input.as_ptr(), output.as_mut_ptr());
            }
            chunk.copy_from_slice(&output[..chunk.len()]);
        }
        return;
    }

    values.iter_mut().for_each(|value| *value = value.cbrt());
}

pub fn pow_f32_in_place(values: &mut [f32], exponent: f32) {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if avx512_svml_available() {
        for chunk in values.chunks_mut(16) {
            let mut input = [1.0_f32; 16];
            let powers = [exponent; 16];
            let mut output = [1.0_f32; 16];
            input[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: see `cbrt_f32_in_place`; all three arrays have the
            // full sixteen lanes expected by the SVML ABI.
            unsafe {
                picvec_svml_powf16(input.as_ptr(), powers.as_ptr(), output.as_mut_ptr());
            }
            chunk.copy_from_slice(&output[..chunk.len()]);
        }
        return;
    }

    values
        .iter_mut()
        .for_each(|value| *value = value.powf(exponent));
}

pub fn atan2_f32(first: &[f32], second: &[f32]) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    let mut result = vec![0.0_f32; first.len()];
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if avx512_svml_available() {
        for (chunk_index, (first_chunk, second_chunk)) in
            first.chunks(16).zip(second.chunks(16)).enumerate()
        {
            let mut first_input = [1.0_f32; 16];
            let mut second_input = [1.0_f32; 16];
            let mut output = [0.0_f32; 16];
            first_input[..first_chunk.len()].copy_from_slice(first_chunk);
            second_input[..second_chunk.len()].copy_from_slice(second_chunk);
            // SAFETY: runtime detection guarantees AVX512_SKX on the
            // SysV target expected by the shim; all arrays provide every lane.
            unsafe {
                picvec_svml_atan2f16(
                    first_input.as_ptr(),
                    second_input.as_ptr(),
                    output.as_mut_ptr(),
                );
            }
            let start = chunk_index * 16;
            result[start..start + first_chunk.len()].copy_from_slice(&output[..first_chunk.len()]);
        }
        return result;
    }

    result
        .iter_mut()
        .zip(first.iter().zip(second))
        .for_each(|(output, (&y, &x))| *output = y.atan2(x));
    result
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
    fn dispatched_elementary_functions_cover_partial_vectors() {
        for length in [1_usize, 7, 8, 15, 16, 17, 31] {
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

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn svml_gate_requires_every_instruction_subset() {
        if avx512_svml_available() {
            assert!(std::arch::is_x86_feature_detected!("avx512f"));
            assert!(std::arch::is_x86_feature_detected!("avx512dq"));
            assert!(std::arch::is_x86_feature_detected!("avx512cd"));
            assert!(std::arch::is_x86_feature_detected!("avx512bw"));
            assert!(std::arch::is_x86_feature_detected!("avx512vl"));
        }
    }
}
