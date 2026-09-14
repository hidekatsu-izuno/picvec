//! Exact sliding extrema for finite image samples.
use std::collections::VecDeque;

pub(crate) fn sliding(input: &[f32], radius: usize, dilate: bool, reflect: bool) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let mut queue: VecDeque<(usize, f32)> = VecDeque::new();
    let mut output = Vec::with_capacity(n);
    for i in 0..n + 2 * radius {
        let position = i as isize - radius as isize;
        let index = if reflect {
            let period = 2 * n as isize;
            let folded = position.rem_euclid(period) as usize;
            if folded < n {
                folded
            } else {
                2 * n - folded - 1
            }
        } else {
            position.clamp(0, n as isize - 1) as usize
        };
        let value = input[index];
        while queue.front().is_some_and(|&(j, _)| j + 2 * radius < i) {
            queue.pop_front();
        }
        while queue
            .back()
            .is_some_and(|&(_, v)| if dilate { v <= value } else { v >= value })
        {
            queue.pop_back();
        }
        queue.push_back((i, value));
        if i >= 2 * radius {
            output.push(queue.front().unwrap().1);
        }
    }
    output
}

#[cfg(test)]
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
