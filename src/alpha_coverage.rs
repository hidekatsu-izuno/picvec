//! Separate raster edge coverage from authored opacity before colour tracing.
use crate::{chroma::AlphaMatte, raster::Raster};
use std::collections::VecDeque;

pub(crate) struct Coverage {
    pub opacity: f32,
    owners: Vec<usize>,
}

pub(crate) fn detect(matte: &AlphaMatte) -> Option<Coverage> {
    detect_with_subpixel_slits(matte, false)
}

fn detect_with_subpixel_slits(matte: &AlphaMatte, allow_slits: bool) -> Option<Coverage> {
    let (w, h) = (matte.width, matte.height);
    let mut histogram = [0_usize; 256];
    for i in 0..matte.len() {
        histogram[(matte.get(i) * 255.0).round() as usize] += 1;
    }
    let visible: usize = histogram[2..].iter().sum();
    let core: usize = histogram[250..].iter().sum();
    if visible == 0 || core * 100 < visible * 97 {
        return None;
    }
    let peak = (250..256).max_by_key(|&a| histogram[a])?;
    let variance = (250..256)
        .map(|a| histogram[a] as f64 * (a as f64 - peak as f64).powi(2))
        .sum::<f64>()
        / core.max(1) as f64;
    if variance > 1.0 {
        return None;
    }
    let mut owners = vec![usize::MAX; matte.len()];
    let mut distance = vec![u8::MAX; matte.len()];
    let mut pending = VecDeque::new();
    for i in 0..matte.len() {
        if matte.get(i) >= 250.0 / 255.0 {
            owners[i] = i;
            distance[i] = 0;
            pending.push_back(i);
        }
    }
    while let Some(i) = pending.pop_front() {
        if distance[i] >= 12 {
            continue;
        }
        for j in neighbours(i, w, h) {
            if distance[j] == u8::MAX {
                distance[j] = distance[i] + 1;
                owners[j] = owners[i];
                pending.push_back(j);
            }
        }
    }
    for i in 0..matte.len() {
        let a = matte.get(i);
        if a <= 1.0 / 255.0 || a >= 250.0 / 255.0 {
            continue;
        }
        if owners[i] == usize::MAX {
            return None;
        }
        let (x, y) = (i % w, i / w);
        // A constant partial-opacity interior is material, not edge coverage.
        if x > 0
            && x + 1 < w
            && y > 0
            && y + 1 < h
            && (y - 1..=y + 1).all(|yy| {
                (x - 1..=x + 1).all(|xx| (matte.get(yy * w + xx) - a).abs() < 0.5 / 255.0)
            })
        {
            return None;
        }
        // Broad fades and authored opacity ramps have an interior well away
        // from transparency; a raster coverage shoulder does not.
        if x >= 7
            && y >= 7
            && x + 7 < w
            && y + 7 < h
            && (y - 7..=y + 7).all(|yy| (x - 7..=x + 7).all(|xx| matte.get(yy * w + xx) > 0.0))
        {
            // A transparent slit can taper below one pixel before its
            // centre reaches zero alpha. Require opaque material on both
            // sides and nearby clear support; an interior opacity stroke or
            // broad fade does not supply this evidence.
            let narrow = (1..=2).any(|r| {
                (x >= r
                    && x + r < w
                    && matte.get(i - r) >= 250.0 / 255.0
                    && matte.get(i + r) >= 250.0 / 255.0)
                    || (y >= r
                        && y + r < h
                        && matte.get(i - r * w) >= 250.0 / 255.0
                        && matte.get(i + r * w) >= 250.0 / 255.0)
            });
            let near_clear = (y.saturating_sub(12)..=(y + 12).min(h - 1)).any(|yy| {
                (x.saturating_sub(12)..=(x + 12).min(w - 1))
                    .any(|xx| matte.get(yy * w + xx) <= 1.0 / 255.0)
            });
            if !allow_slits || !narrow || !near_clear {
                return None;
            }
        }
    }
    Some(Coverage {
        opacity: peak as f32 / 255.0,
        owners,
    })
}

/// Coverage classification is local to disconnected artwork. A translucent
/// gradient in one illustration must not turn another illustration's AA rim
/// into hundreds of authored-opacity faces.
pub(crate) struct LocalCoverage {
    pub opacity: Vec<f32>,
}

pub(crate) fn detect_components(matte: &AlphaMatte) -> Option<LocalCoverage> {
    let (w, h) = (matte.width, matte.height);
    let mut labels = vec![usize::MAX; matte.len()];
    let mut components = Vec::<Vec<usize>>::new();
    for seed in 0..matte.len() {
        if matte.get(seed) <= 0.0 || labels[seed] != usize::MAX {
            continue;
        }
        let id = components.len();
        labels[seed] = id;
        let mut pixels = vec![seed];
        let mut cursor = 0;
        while cursor < pixels.len() {
            let i = pixels[cursor];
            cursor += 1;
            for j in neighbours(i, w, h) {
                if matte.get(j) > 0.0 && labels[j] == usize::MAX {
                    labels[j] = id;
                    pixels.push(j);
                }
            }
        }
        components.push(pixels);
    }
    let mut local = LocalCoverage {
        opacity: vec![0.0; matte.len()],
    };
    let mut accepted = false;
    for (id, pixels) in components.iter().enumerate() {
        let core = pixels
            .iter()
            .filter(|&&i| matte.get(i) >= 250.0 / 255.0)
            .count();
        if core * 100 < pixels.len() * 97 {
            continue;
        }
        let x0 = pixels
            .iter()
            .map(|i| i % w)
            .min()
            .unwrap()
            .saturating_sub(13);
        let y0 = pixels
            .iter()
            .map(|i| i / w)
            .min()
            .unwrap()
            .saturating_sub(13);
        let x1 = (pixels.iter().map(|i| i % w).max().unwrap() + 14).min(w);
        let y1 = (pixels.iter().map(|i| i / w).max().unwrap() + 14).min(h);
        let cw = x1 - x0;
        let ch = y1 - y0;
        let indices: Vec<_> = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| y * w + x))
            .collect();
        let cropped = AlphaMatte::from_u8(
            cw,
            ch,
            indices
                .iter()
                .map(|&i| {
                    if labels[i] == id {
                        (matte.get(i) * 255.0).round() as u8
                    } else {
                        0
                    }
                })
                .collect(),
        );
        let Some(coverage) = detect_with_subpixel_slits(&cropped, true) else {
            continue;
        };
        accepted = true;
        for &i in pixels {
            local.opacity[i] = coverage.opacity;
        }
    }
    accepted.then_some(local)
}

fn neighbours(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    [
        i.checked_sub(1).filter(|_| i % w > 0),
        (i % w + 1 < w).then_some(i + 1),
        i.checked_sub(w),
        (i + w < w * h).then_some(i + w),
    ]
    .into_iter()
    .flatten()
}

impl Coverage {
    pub fn exterior_colour(&self, source: &Raster, matte: &AlphaMatte) -> Option<[f32; 3]> {
        let mut colours = std::collections::BTreeMap::<String, (usize, [f32; 3])>::new();
        let (w, h) = (matte.width, matte.height);
        for i in 0..matte.len() {
            if matte.get(i) < 250.0 / 255.0 {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let near_edge = [
                (x.saturating_sub(4), y),
                ((x + 4).min(w - 1), y),
                (x, y.saturating_sub(4)),
                (x, (y + 4).min(h - 1)),
            ]
            .iter()
            .any(|&(xx, yy)| matte.get(yy * w + xx) < self.opacity * 0.5);
            if near_edge {
                let colour = source.pixels[i];
                colours
                    .entry(crate::color::rgb_hex(colour))
                    .or_insert((0, colour))
                    .0 += 1;
            }
        }
        let &(_, colour) = colours.values().max_by_key(|(n, _)| *n)?;
        colours
            .values()
            .all(|(_, other)| (0..3).all(|c| (other[c] - colour[c]).abs() <= 6.0 / 255.0))
            .then_some(colour)
    }

    pub fn extend_colour(&self, source: &Raster, matte: &AlphaMatte) -> Raster {
        let mut image = source.clone();
        for (i, &owner) in self.owners.iter().enumerate() {
            if owner != usize::MAX && matte.get(i) < self.opacity * 0.9 {
                image.pixels[i] = source.pixels[owner];
            }
        }
        if let Some(colour) = self.exterior_colour(source, matte) {
            let (w, h) = (matte.width, matte.height);
            for i in 0..matte.len() {
                let (x, y) = (i % w, i / w);
                if [
                    (x.saturating_sub(4), y),
                    ((x + 4).min(w - 1), y),
                    (x, y.saturating_sub(4)),
                    (x, (y + 4).min(h - 1)),
                ]
                .iter()
                .any(|&(xx, yy)| matte.get(yy * w + xx) < self.opacity * 0.5)
                {
                    image.pixels[i] = colour;
                }
            }
        }
        image
    }
}

#[cfg(test)]
include!("../tests/unit/alpha_coverage.rs");
