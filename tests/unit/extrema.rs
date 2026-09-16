mod tests {
    use super::*;
    #[test]
    fn sliding_matches_full_window_with_both_boundary_modes() {
        for n in [1, 2, 7, 31] {
            let input: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 23) as f32 - 10.0).collect();
            for radius in [0, 1, 4, 39] {
                for dilate in [false, true] {
                    for reflect in [false, true] {
                        let expected: Vec<f32> = (0..n)
                            .map(|x| {
                                (-(radius as isize)..=radius as isize)
                                    .map(|dx| {
                                        let mut p = x as isize + dx;
                                        if reflect {
                                            while p < 0 || p >= n as isize {
                                                p = if p < 0 {
                                                    -p - 1
                                                } else {
                                                    2 * n as isize - p - 1
                                                };
                                            }
                                        } else {
                                            p = p.clamp(0, n as isize - 1);
                                        }
                                        input[p as usize]
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
                        assert_eq!(sliding(&input, radius, dilate, reflect), expected);
                    }
                }
            }
        }
    }
}
