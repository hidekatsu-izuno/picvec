//! Exact observations shared by residual-layer candidates.
use super::*;
use std::cell::OnceCell;

// Squared errors are nonnegative. Any completed subtree of NumPy's pairwise
// sum is a lower bound on the final sum, even with rounded f32 additions.
// Retain that exact reduction tree so surviving candidates keep identical MSE.
pub(super) fn mse_below(
    source: &Raster,
    samples: &[usize],
    mut predict: impl FnMut(usize) -> [f32; 3],
    limit: f32,
) -> Option<f32> {
    fn sum(
        start: usize,
        len: usize,
        divisor: f32,
        limit: f32,
        value: &mut impl FnMut(usize) -> f32,
    ) -> Option<f32> {
        let result = if len <= 128 {
            let mut values = [0.0_f32; 128];
            for (offset, target) in values[..len].iter_mut().enumerate() {
                *target = value(start + offset);
            }
            numpy_sum_f32(&values[..len])
        } else {
            let mut middle = len / 2;
            middle -= middle % 8;
            let left = sum(start, middle, divisor, limit, value)?;
            let right = sum(start + middle, len - middle, divisor, limit, value)?;
            left + right
        };
        (result / divisor < limit).then_some(result)
    }
    if samples.is_empty() {
        return (0.0 < limit).then_some(0.0);
    }
    let mut previous = usize::MAX;
    let mut squared = [0.0; 3];
    let mut value = |offset: usize| {
        let sample = offset / 3;
        if previous != sample {
            let index = samples[sample];
            let predicted = predict(index);
            squared = [0, 1, 2].map(|c| {
                let difference = predicted[c] - source.pixels[index][c];
                difference * difference
            });
            previous = sample;
        }
        squared[offset % 3]
    };
    let len = samples.len() * 3;
    sum(0, len, len as f32, limit, &mut value).map(|total| total / len as f32)
}

pub(super) struct SubsetSamples {
    base: Vec<[f32; 3]>,
    overlays: Vec<Vec<([f32; 3], f32)>>,
    targets: Vec<[f32; 3]>,
    squared: Vec<f32>,
}

impl SubsetSamples {
    pub(super) fn new(
        source: &Raster,
        samples: &[usize],
        base: &Paint,
        overlays: &[PaintOverlay],
    ) -> Self {
        Self {
            base: samples
                .iter()
                .map(|&i| paint_at(base, i, source.width))
                .collect(),
            overlays: overlays
                .iter()
                .map(|overlay| {
                    samples
                        .iter()
                        .map(|&i| sample_overlay(overlay, i, source.width))
                        .collect()
                })
                .collect(),
            targets: samples.iter().map(|&i| source.pixels[i]).collect(),
            squared: Vec::with_capacity(samples.len() * 3),
        }
    }

    pub(super) fn mse(&mut self, mask: usize) -> f32 {
        self.squared.clear();
        for (i, &base) in self.base.iter().enumerate() {
            let mut under = base;
            for (layer, samples) in self.overlays.iter().enumerate() {
                if layer >= 12 || mask & (1 << layer) != 0 {
                    let (over, alpha) = samples[i];
                    // Keep the same layer order and arithmetic as paint_at.
                    under = [0, 1, 2].map(|c| under[c] * (1.0 - alpha) + over[c] * alpha);
                }
            }
            for (c, &value) in under.iter().enumerate() {
                let difference = value - self.targets[i][c];
                self.squared.push(difference * difference);
            }
        }
        if self.squared.is_empty() {
            0.0
        } else {
            numpy_sum_f32(&self.squared) / self.squared.len() as f32
        }
    }
}

pub(super) struct SlopeSupport<'a> {
    source: &'a Raster,
    base: &'a Paint,
    pixels: &'a [usize],
    positions: HashMap<usize, usize>,
    observations: Vec<OnceCell<(Oklab, Oklab)>>,
}

impl<'a> SlopeSupport<'a> {
    pub(super) fn new(source: &'a Raster, base: &'a Paint, pixels: &'a [usize]) -> Self {
        Self {
            source,
            base,
            pixels,
            positions: pixels.iter().enumerate().map(|(p, &i)| (i, p)).collect(),
            observations: (0..pixels.len()).map(|_| OnceCell::new()).collect(),
        }
    }

    fn observation(&self, position: usize) -> (Oklab, Oklab) {
        *self.observations[position].get_or_init(|| {
            let i = self.pixels[position];
            (
                rgb_to_oklab(self.source.pixels[i]),
                rgb_to_oklab(paint_at(self.base, i, self.source.width)),
            )
        })
    }

    pub(super) fn accepts(&self, candidate: &Paint) -> bool {
        let difference = |a: Oklab, b: Oklab| Oklab {
            l: b.l - a.l,
            a: b.a - a.a,
            b: b.b - a.b,
        };
        self.pixels.iter().enumerate().all(|(position, &i)| {
            [i + 6, i + 6 * self.source.width].into_iter().all(|j| {
                if j == i + 6 && i % self.source.width + 6 >= self.source.width {
                    return true;
                }
                let Some(&other) = self.positions.get(&j) else {
                    return true;
                };
                let (source_a, base_a) = self.observation(position);
                let (source_b, base_b) = self.observation(other);
                residual_slope_supported(
                    difference(source_a, source_b),
                    difference(base_a, base_b),
                    difference(
                        rgb_to_oklab(paint_at(candidate, i, self.source.width)),
                        rgb_to_oklab(paint_at(candidate, j, self.source.width)),
                    ),
                )
            })
        })
    }

    pub(super) fn preserves_error(&self, candidate: &Paint, extra: f32) -> bool {
        self.pixels.iter().enumerate().all(|(position, &i)| {
            let (target, base) = self.observation(position);
            delta_e_ok(
                target,
                rgb_to_oklab(paint_at(candidate, i, self.source.width)),
            ) <= delta_e_ok(target, base) + extra
        })
    }
}

#[cfg(test)]
include!("../tests/unit/gradient_residual.rs");
