fn meijering(input: &[f32], width: usize, height: usize, sigmas: &[f64]) -> Vec<f32> {
    let mut filtered = vec![0.0_f32; input.len()];
    for &sigma in sigmas {
        let gradient_row = gaussian_filter(input, width, height, sigma, [1, 0]);
        let gradient_column = gaussian_filter(input, width, height, sigma, [0, 1]);
        let hrr = gaussian_filter(&gradient_row, width, height, sigma, [1, 0]);
        let hrc = gaussian_filter(&gradient_row, width, height, sigma, [0, 1]);
        let hcc = gaussian_filter(&gradient_column, width, height, sigma, [0, 1]);
        let values: Vec<f32> = (0..input.len())
            .into_par_iter()
            .map(|index| {
                let centre = (hrr[index] + hcc[index]) / 2.0;
                let difference = (hrr[index] - hcc[index]) / 2.0;
                let half_root = (hrc[index] * hrc[index] + difference * difference).sqrt();
                let first = centre + half_root;
                let second = centre - half_root;
                let first_normalized = first + (1.0_f32 / 3.0) * second;
                let second_normalized = (1.0_f32 / 3.0) * first + second;
                if first_normalized.abs() >= second_normalized.abs() {
                    first_normalized
                } else {
                    second_normalized
                }
                .max(0.0)
            })
            .collect();
        let maximum = values.par_iter().copied().reduce(|| 0.0, f32::max);
        if maximum > 0.0 {
            filtered
                .par_iter_mut()
                .zip(values.into_par_iter())
                .for_each(|(destination, value)| {
                    *destination = destination.max(value / maximum);
                });
        }
    }
    filtered
}

fn correlate_axis_reference(
    input: &[f32],
    width: usize,
    height: usize,
    axis: usize,
    weights: &[f64],
) -> Vec<f32> {
    let radius = (weights.len() / 2) as isize;
    let mut output = vec![0.0_f32; input.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, output) in row.iter_mut().enumerate() {
                let mut sum = 0.0_f64;
                for (position, &weight) in weights.iter().enumerate() {
                    let offset = position as isize - radius;
                    let (sample_x, sample_y) = if axis == 0 {
                        (x, reflect_index(y as isize + offset, height))
                    } else {
                        (reflect_index(x as isize + offset, width), y)
                    };
                    sum += input[sample_y * width + sample_x] as f64 * weight;
                }
                *output = sum as f32;
            }
        });
    output
}

mod tests {

    #[test]
    fn vector_correlation_matches_scalar_bits_at_reflected_edges() {
        for (width, height) in [(1, 1), (2, 7), (3, 2), (4, 5), (7, 9), (39, 17)] {
            let input: Vec<f32> = (0..width * height)
                .map(|i| ((i * 7919 + 13) % 997) as f32 / 997.0 - 0.5)
                .collect();
            for sigma in [0.5, 1.5, 6.0, 23.0] {
                for order in [0, 1] {
                    let weights = gaussian_kernel(sigma, order, 8.0);
                    for axis in [0, 1] {
                        let expected =
                            correlate_axis_reference(&input, width, height, axis, &weights);
                        let actual = correlate_axis(&input, width, height, axis, &weights);
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            "{width}x{height}, sigma={sigma}, order={order}, axis={axis}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn disk_morphology_matches_brute_force() {
        for (width, height) in [(1, 1), (2, 7), (37, 19)] {
            let input: Vec<f32> = (0..width * height)
                .map(|i| ((i * 7919 + 13) % 997) as f32 / 997.0)
                .collect();
            for radius in [0, 1, 3, 11, 17, 23, 34] {
                let offsets = disk_offsets(radius);
                for dilate in [false, true] {
                    let expected: Vec<f32> = (0..input.len())
                        .map(|i| {
                            offsets
                                .iter()
                                .map(|&(dx, dy)| {
                                    input[reflect_index((i / width) as isize + dy, height) * width
                                        + reflect_index((i % width) as isize + dx, width)]
                                })
                                .fold(
                                    if dilate {
                                        f32::NEG_INFINITY
                                    } else {
                                        f32::INFINITY
                                    },
                                    |a, b| if dilate { a.max(b) } else { a.min(b) },
                                )
                        })
                        .collect();
                    assert_eq!(
                        morphology(&input, width, height, &offsets, dilate),
                        expected
                    );
                }
            }
        }
    }

    use super::*;

    #[test]
    fn shared_hessian_preserves_both_independent_polarities() {
        for (width, height) in [(1, 17), (23, 1), (23, 19)] {
            let input: Vec<f32> = (0..width * height)
                .map(|i| ((i * 71 + i * i * 17) % 251) as f32 / 250.0)
                .collect();
            let sigmas = [0.5, 1.5, 3.0];
            let (dark, bright) = meijering_polarities(&input, width, height, &sigmas);
            let inverted: Vec<f32> = input.iter().map(|v| -v).collect();
            for (actual, expected) in [
                (dark, meijering(&input, width, height, &sigmas)),
                (bright, meijering(&inverted, width, height, &sigmas)),
            ] {
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 1e-6));
            }
        }
    }

    #[test]
    fn canonical_ridge_extrema_require_source_support_to_restore_samples() {
        let mut image = Raster::blank(32, 24, [0.82; 3]);
        for x in 4..28 {
            image.pixels[7 * image.width + x] = [0.05; 3];
            image.pixels[16 * image.width + x] = [0.96; 3];
        }
        let analysis = analyze(&image);
        let valid = vec![true; image.pixels.len()];
        let selected = adjust_paint_samples_from_analysis(&image, &valid, &analysis);
        let excluded = [7 * image.width + 16, 16 * image.width + 16];
        let mut mask = valid;
        let mut source = image.clone();
        source.pixels[excluded[0]] = [0.4; 3];
        source.pixels[excluded[1]] = [0.85; 3];
        for i in excluded {
            assert!(selected[i], "a valid ridge core should remain usable");
            mask[i] = false;
            mask[i - 1] = false;
        }
        let selected = adjust_paint_samples_from_analysis(&source, &mask, &analysis);
        for i in excluded {
            assert!(!selected[i], "canonical extrema are not source evidence");
            assert!(selected[i - 1], "restore actual source ridge cores");
        }
    }

    #[test]
    fn shared_analysis_matches_independent_ridge_consumers() {
        let mut image = Raster::blank(32, 24, [0.82, 0.82, 0.82]);
        for x in 4..28 {
            image.pixels[7 * image.width + x] = [0.05, 0.08, 0.12];
            image.pixels[16 * image.width + x] = [0.96, 0.75, 0.12];
        }
        let original = vec![true; image.pixels.len()];
        let analysis = analyze(&image);

        assert_eq!(
            adjust_paint_samples_from_analysis(&image, &original, &analysis),
            adjust_paint_samples(&image, &original)
        );
        let shared = strong_branches_from_analysis(&image, &analysis);
        let independent = strong_branches(&image);
        assert_eq!(shared.dark, independent.dark);
        assert_eq!(shared.bright, independent.bright);
    }
}
