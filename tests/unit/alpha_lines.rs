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
        let input = image::load_from_memory(include_bytes!("../data/cliparts-grid-alpha.png"))
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
