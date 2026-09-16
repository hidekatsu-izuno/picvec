//! Recover long, isolated thin bands from alpha coverage before contour fitting.
use crate::chroma::AlphaMatte;
#[derive(Clone, Debug)]
pub(crate) struct AlphaBand {
    pub path_data: String,
    pub opacity: f32,
    pub pixels: Vec<usize>,
}

#[derive(Clone)]
struct Section {
    row: usize,
    left: usize,
    right: usize,
    center: f32,
    mass: f32,
    peak: f32,
}

pub(crate) fn extract(matte: &AlphaMatte) -> (AlphaMatte, Vec<AlphaBand>) {
    let mut cleared = Vec::new();
    let mut layers = Vec::new();
    for vertical in [true, false] {
        let (across, along) = if vertical {
            (matte.width, matte.height)
        } else {
            (matte.height, matte.width)
        };
        let index = |x: usize, y: usize| {
            if vertical {
                y * matte.width + x
            } else {
                x * matte.width + y
            }
        };
        let mut active: Vec<Vec<Section>> = Vec::new();
        let mut finished = Vec::new();
        for row in 0..along {
            let mut sections = Vec::new();
            let mut x = 0;
            while x < across {
                if matte.get(index(x, row)) == 0.0 {
                    x += 1;
                    continue;
                }
                let left = x;
                let (mut mass, mut moment, mut peak) = (0.0, 0.0, 0.0_f32);
                while x < across && matte.get(index(x, row)) > 0.0 {
                    let a = matte.get(index(x, row));
                    mass += a;
                    moment += a * (x as f32 + 0.5);
                    peak = peak.max(a);
                    x += 1;
                }
                if left > 0 && x < across && x - left <= 4 && (0.15..=3.0).contains(&mass) {
                    sections.push(Section {
                        row,
                        left,
                        right: x,
                        center: moment / mass,
                        mass,
                        peak,
                    });
                }
            }
            let mut next = Vec::new();
            for section in sections {
                let candidate = active.iter().position(|track| {
                    let last = track.last().unwrap();
                    (last.center - section.center).abs() < 0.75
                        && (last.mass - section.mass).abs() < 0.15 * last.mass
                });
                let mut track = candidate.map(|i| active.swap_remove(i)).unwrap_or_default();
                track.push(section);
                next.push(track);
            }
            finished.extend(active);
            active = next;
        }
        finished.extend(active);
        // Only isolated consecutive profiles form a track. Junctions and
        // empty source rows stop it, so no invented bridges are needed.
        for track in finished {
            // Short straight spans also need a filled band now that coverage
            // is carried by the face rather than an image-wide alpha mask.
            if track.len() < 8 {
                continue;
            }
            let n = track.len() as f32;
            let mean_y = track.iter().map(|s| s.row as f32 + 0.5).sum::<f32>() / n;
            let mean_x = track.iter().map(|s| s.center).sum::<f32>() / n;
            let variance = track
                .iter()
                .map(|s| (s.row as f32 + 0.5 - mean_y).powi(2))
                .sum::<f32>();
            let slope = track
                .iter()
                .map(|s| (s.row as f32 + 0.5 - mean_y) * (s.center - mean_x))
                .sum::<f32>()
                / variance;
            let mass = track.iter().map(|s| s.mass).sum::<f32>() / n;
            if slope.abs() > 0.05
                || track.iter().any(|s| {
                    (s.center - mean_x - slope * (s.row as f32 + 0.5 - mean_y)).abs() > 0.15
                        || (s.mass - mass).abs() > 0.08 * mass
                })
            {
                continue;
            }
            let peak = track.iter().map(|s| s.peak).fold(0.0_f32, f32::max);
            let half_width = 0.5 * mass / peak;
            let start = track.first().unwrap().row as f32;
            let end = (track.last().unwrap().row + 1) as f32;
            let x0 = mean_x + slope * (start - mean_y);
            let x1 = mean_x + slope * (end - mean_y);
            let point = |x: f32, y: f32| {
                if vertical {
                    format!("{x:.4} {y:.4}")
                } else {
                    format!("{y:.4} {x:.4}")
                }
            };
            let path_data = format!(
                "M {} L {} L {} L {} Z",
                point(x0 - half_width, start),
                point(x1 - half_width, end),
                point(x1 + half_width, end),
                point(x0 + half_width, start)
            );
            let mut pixels = Vec::new();
            for section in &track {
                for x in section.left..section.right {
                    cleared.push(index(x, section.row));
                    pixels.push(index(x, section.row));
                }
            }
            layers.push(AlphaBand {
                path_data,
                opacity: peak,
                pixels,
            });
        }
    }
    (
        if cleared.is_empty() {
            matte.clone()
        } else {
            matte.cleared(&cleared)
        },
        layers,
    )
}

#[cfg(test)]
include!("../tests/unit/alpha_lines.rs");
