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
include!("../tests/unit/extrema.rs");
