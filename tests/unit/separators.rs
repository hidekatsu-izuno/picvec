mod tests {
    use super::*;

    #[test]
    fn occluded_band_keeps_its_subpixel_edges_and_foreground_coverage() {
        for vertical in [false, true] {
            let (width, height) = if vertical { (96, 320) } else { (320, 96) };
            let coordinates = |i: usize| {
                let (x, y) = (i % width, i / width);
                if vertical {
                    (y, x)
                } else {
                    (x, y)
                }
            };
            let source = SourceRaster::from_rgb8_fn(width, height, |i| {
                let (x, y) = coordinates(i);
                if (145..175).contains(&x) && (20..80).contains(&y) {
                    [0.1; 3]
                } else if (29..=36).contains(&y) {
                    [1.0; 3]
                } else {
                    [0.0, 1.0, 0.0]
                }
            });
            let alpha: Vec<_> = (0..width * height)
                .map(|i| {
                    let (x, y) = coordinates(i);
                    if ((145..175).contains(&x) && (20..80).contains(&y)) || (30..36).contains(&y) {
                        255
                    } else if y == 29 || y == 36 {
                        128
                    } else {
                        0
                    }
                })
                .collect();
            let matte = AlphaMatte::from_u8(width, height, alpha.clone());
            let (bands, cleaned) = extract(&source, &matte, 128);
            assert_eq!(bands.len(), 1);
            let band = &bands[0];
            let start = band.rect[usize::from(!vertical)];
            let span = band.rect[2 + usize::from(!vertical)];
            assert!((start - 29.5).abs() < 0.01 && (span - 7.0).abs() < 0.02);
            let cleaned = cleaned.unwrap();
            for (i, &old) in alpha.iter().enumerate() {
                let (x, y) = coordinates(i);
                if (145..175).contains(&x) && (20..80).contains(&y) {
                    assert_eq!(cleaned.get(i), old as f32 / 255.0);
                } else if (29..=36).contains(&y) {
                    assert_eq!(cleaned.get(i), 0.0);
                }
            }
            let translucent =
                AlphaMatte::from_u8(width, height, alpha.iter().map(|&v| v / 2).collect());
            assert!(extract(&source, &translucent, 128).0.is_empty());
        }
    }

    #[test]
    fn isolated_line_and_broad_panel_are_not_factored() {
        for end in [36, 60] {
            let source = SourceRaster::from_rgb8_fn(320, 96, |i| {
                if (30..end).contains(&(i / 320)) {
                    [1.0; 3]
                } else {
                    [0.0, 1.0, 0.0]
                }
            });
            let matte = AlphaMatte::from_u8(
                320,
                96,
                (0..320 * 96)
                    .map(|i| {
                        if (30..end).contains(&(i / 320)) {
                            255
                        } else {
                            0
                        }
                    })
                    .collect(),
            );
            assert!(extract(&source, &matte, 128).0.is_empty());
        }
    }
}
