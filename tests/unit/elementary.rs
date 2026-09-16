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

            let mut powers = source.clone();
            pow_f32_in_place(&mut powers, 2.4);
            for (&actual, &value) in powers.iter().zip(&source) {
                close_f32(actual, value.powf(2.4), 2e-5);
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
