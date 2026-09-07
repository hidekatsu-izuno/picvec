//! Recover long, isolated thin bands from alpha coverage before contour fitting.
use crate::chroma::AlphaMatte;
use crate::gradient::Paint;
use crate::svg::AlphaMaskLayer;

#[derive(Clone)]
struct Section {
    row: usize,
    left: usize,
    right: usize,
    center: f32,
    mass: f32,
    peak: f32,
}

pub(crate) fn extract(matte: &AlphaMatte) -> (AlphaMatte, Vec<AlphaMaskLayer>) {
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
            if track.len() < 64 {
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
            for section in &track {
                for x in section.left..section.right {
                    cleared.push(index(x, section.row));
                }
            }
            layers.push(AlphaMaskLayer {
                path_data,
                opacity: peak,
                paint: Some(Paint::Solid { color: [1.0; 3] }),
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
mod tests {
    use super::*;

    fn band(curved: bool, gap: bool) -> AlphaMatte {
        let mut values = vec![0.0; 32 * 256];
        for y in 0..256 {
            if gap && (124..128).contains(&y) {
                continue;
            }
            let center = 12.3
                + if curved {
                    (y as f32 / 40.0).sin()
                } else {
                    y as f32 * 0.003
                };
            for x in 0..32 {
                let coverage =
                    ((x as f32 + 1.0).min(center + 0.8) - (x as f32).max(center - 0.8)).max(0.0);
                values[y * 32 + x] = 0.7 * coverage;
            }
        }
        AlphaMatte::new(32, 256, values)
    }

    #[test]
    fn straight_alpha_band_keeps_coverage_and_authored_breaks() {
        for gap in [false, true] {
            let source = band(false, gap);
            let (remaining, lines) = extract(&source);
            assert_eq!(lines.len(), if gap { 2 } else { 1 });
            assert!((0..remaining.len()).all(|i| remaining.get(i) == 0.0));
            for line in lines {
                assert!(!line.path_data.contains('C'));
                assert!((line.opacity - 0.7).abs() < 0.001);
                let coords = line
                    .path_data
                    .split_whitespace()
                    .filter_map(|v| v.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                let width = coords[6] - coords[0];
                assert!((width - 1.6).abs() < 0.01);
                if gap {
                    assert!(coords[3] <= 124.0 || coords[1] >= 128.0);
                }
            }
        }
    }

    #[test]
    fn horizontal_and_vertical_bands_keep_their_crossing() {
        let mut values = vec![0.0; 192 * 192];
        for t in 2..190 {
            values[t * 192 + 95] = 0.8;
            values[95 * 192 + t] = 0.8;
        }
        let source = AlphaMatte::new(192, 192, values);
        let (remaining, lines) = extract(&source);
        assert_eq!(lines.len(), 4);
        assert!((remaining.get(95 * 192 + 95) - 0.8).abs() < 0.001);
        assert_eq!(remaining.get(50 * 192 + 95), 0.0);
        assert_eq!(remaining.get(95 * 192 + 50), 0.0);
    }

    #[test]
    fn curved_alpha_band_is_not_forced_straight() {
        let source = band(true, false);
        let (remaining, lines) = extract(&source);
        assert!(lines.is_empty());
        assert!((0..source.len()).all(|i| source.get(i) == remaining.get(i)));
    }

    #[test]
    fn native_cliparts_grid_uses_long_straight_bands() {
        let input = image::load_from_memory(include_bytes!("test-data/cliparts-grid-alpha.png"))
            .unwrap()
            .to_luma8();
        let matte = AlphaMatte::from_u8(
            input.width() as usize,
            input.height() as usize,
            input.into_raw(),
        );
        let (remaining, lines) = extract(&matte);
        assert!(lines.len() >= 3, "grid was not recovered: {}", lines.len());
        let source_mass: f32 = (0..matte.len()).map(|i| matte.get(i)).sum();
        let remaining_mass: f32 = (0..remaining.len()).map(|i| remaining.get(i)).sum();
        assert!(
            remaining_mass < 0.1 * source_mass,
            "grid remained in contour fitting"
        );
    }
}
