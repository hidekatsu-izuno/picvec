//! Replace redundant colour fragments with shared source-alpha geometry.
use crate::{chroma::AlphaMatte, gradient::Paint, raster::Raster};

pub(crate) fn consolidate(
    matte: &AlphaMatte,
    source: &Raster,
    labels: &[u32],
    paints: &mut [Paint],
) -> usize {
    // Sample the exterior colour inside its antialias fringe.
    let mut boundary_weight = std::collections::BTreeMap::<String, (usize, [f32; 3])>::new();
    for y in 0..matte.height {
        for x in 0..matte.width {
            let i = y * matte.width + x;
            if matte.get(i) < 0.5 {
                continue;
            }
            let boundary = [
                (x.saturating_sub(4), y),
                ((x + 4).min(matte.width - 1), y),
                (x, y.saturating_sub(4)),
                (x, (y + 4).min(matte.height - 1)),
            ]
            .iter()
            .any(|&(xx, yy)| matte.get(yy * matte.width + xx) < 0.5);
            if !boundary {
                continue;
            }
            let color = source.pixels[i];
            boundary_weight
                .entry(crate::color::rgb_hex(color))
                .or_insert((0, color))
                .0 += 1;
        }
    }
    let Some((_, color)) = boundary_weight
        .into_values()
        .max_by_key(|(weight, _)| *weight)
    else {
        return 0;
    };
    let close = |other: [f32; 3]| (0..3).all(|c| (other[c] - color[c]).abs() <= 6.0 / 255.0);
    let candidates: Vec<_> = paints
        .iter()
        .enumerate()
        .filter_map(|(id, paint)| {
            let similar = match paint {
                Paint::Solid { color } => close(*color),
                Paint::Linear { stops, .. } | Paint::Radial { stops, .. } => {
                    !stops.is_empty() && stops.iter().all(|s| close(s.color.map(|v| v as f32)))
                }
                Paint::Layered { .. } => false,
            };
            similar.then_some(id)
        })
        .collect();
    let mut eligible = vec![false; paints.len()];
    for &id in &candidates {
        eligible[id] = true;
    }
    let mut error = vec![0.0_f64; paints.len()];
    let mut weight = vec![0.0_f64; paints.len()];
    for (i, &label) in labels.iter().enumerate() {
        let id = label as usize;
        if !eligible[id] {
            continue;
        }
        let alpha = f64::from(matte.get(i));
        if alpha == 0.0 {
            continue;
        }
        let previous = crate::gradient::paint_at(&paints[id], i, source.width);
        for c in 0..3 {
            let old = f64::from(source.pixels[i][c] - previous[c]);
            let new = f64::from(source.pixels[i][c] - color[c]);
            error[id] += alpha * alpha * (new * new - old * old);
            weight[id] += alpha * alpha;
        }
    }
    let mut count = 0;
    for id in candidates {
        // Allow at most one 8-bit code value of additional RMS error. This
        // rejects an authored low-contrast gradient even when its extrema
        // happen to be close to the exterior colour.
        if error[id] <= weight[id] / (255.0 * 255.0) && paints[id] != (Paint::Solid { color }) {
            paints[id] = Paint::Solid { color };
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geometry::Point,
        gradient::{ColorStop, LinearPreset},
    };

    #[test]
    fn source_gate_merges_rim_noise_but_retains_a_real_shallow_gradient() {
        let width = 48;
        let base = [50.0 / 255.0, 80.0 / 255.0, 110.0 / 255.0];
        let mut source = Raster::blank(width, width, base);
        let mut values = vec![0.0; width * width];
        let mut labels = vec![0; values.len()];
        let gradient = Paint::Linear {
            preset: LinearPreset::LeftToRight,
            start: Point { x: 12.0, y: 0.0 },
            end: Point { x: 36.0, y: 0.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: base.map(|v| f64::from(v - 5.0 / 255.0)),
                },
                ColorStop {
                    offset: 1.0,
                    color: base.map(|v| f64::from(v + 5.0 / 255.0)),
                },
            ],
        };
        for y in 4..44 {
            for x in 4..44 {
                let i = y * width + x;
                values[i] = 1.0;
                if (12..36).contains(&x) && (12..36).contains(&y) {
                    labels[i] = 1;
                    source.pixels[i] = crate::gradient::paint_at(&gradient, i, width);
                }
            }
        }
        let matte = AlphaMatte::new(width, width, values);
        let mut paints = vec![
            Paint::Solid {
                color: base.map(|v| v - 2.0 / 255.0),
            },
            gradient.clone(),
        ];
        assert_eq!(consolidate(&matte, &source, &labels, &mut paints), 1);
        assert_eq!(paints[0], Paint::Solid { color: base });
        assert_eq!(paints[1], gradient);
    }

    #[test]
    fn opaque_input_has_no_exterior_colour_proposal() {
        let source = Raster::blank(16, 16, [0.5; 3]);
        let matte = AlphaMatte::new(16, 16, vec![1.0; 256]);
        let mut paints = vec![Paint::Solid { color: [0.49; 3] }];
        let before = paints.clone();
        assert_eq!(consolidate(&matte, &source, &[0; 256], &mut paints), 0);
        assert_eq!(paints, before);
    }
}
