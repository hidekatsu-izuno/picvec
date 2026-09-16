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
include!("../tests/unit/alpha_paint.rs");
