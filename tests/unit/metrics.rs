mod tests {
    use super::*;

    #[test]
    fn sliding_ssim_matches_independent_window_statistics() {
        for (width, height) in [(1, 1), (2, 9), (3, 2), (9, 11), (21, 17), (131, 69)] {
            let a: Vec<_> = (0..width * height)
                .map(|i| ((i * 19 % 103) as f32 / 103.0).powi(2))
                .collect();
            let b: Vec<_> = a
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 17 == 0 { v * 0.7 } else { v })
                .collect();
            let (actual, window, _) = local_similarity(&a, &b, width, height);
            let mut expected = 0.0;
            let mut n = 0;
            for y in 0..=height - window {
                for x in 0..=width - window {
                    let mut values = Vec::new();
                    for dy in 0..window {
                        for dx in 0..window {
                            let i = (y + dy) * width + x + dx;
                            values.push((a[i] as f64, b[i] as f64));
                        }
                    }
                    let count = values.len() as f64;
                    let mx = values.iter().map(|p| p.0).sum::<f64>() / count;
                    let my = values.iter().map(|p| p.1).sum::<f64>() / count;
                    let divisor = (count - 1.0).max(1.0);
                    let vx = values.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>() / divisor;
                    let vy = values.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>() / divisor;
                    let cov = values.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / divisor;
                    expected += ((2.0 * mx * my + 0.0001) * (2.0 * cov + 0.0009))
                        / ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009));
                    n += 1;
                }
            }
            assert!(
                (actual as f64 - expected / n as f64).abs() < 1e-6,
                "{width}x{height}: {actual} vs {}",
                expected / n as f64
            );
            let identity = local_similarity(&a, &a, width, height).0;
            assert!((identity - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn diagnostic_tiles_locate_damage_and_clip_to_image_bounds() {
        let reference = Raster::blank(130, 70, [0.5; 3]);
        let mut candidate = reference.clone();
        for y in 64..70 {
            for x in 128..130 {
                candidate.pixels[y * 130 + x] = [0.0; 3];
            }
        }
        let quality = compare_on(&reference, &candidate, [1.0; 3]);
        assert_eq!((quality.width, quality.height), (130, 70));
        assert_eq!(quality.comparison_background, Some([1.0; 3]));
        assert_eq!(quality.local_ssim_window, 7);
        assert!(quality.local_ssim < 1.0);
        let worst = &quality.worst_tiles[0];
        assert_eq!(
            (worst.x, worst.y, worst.width, worst.height),
            (128, 64, 2, 6)
        );
        assert!(worst.delta_e_ok_mean > quality.delta_e_ok_mean * 100.0);
        // This narrow edge tile has no valid 7x7 window centre.
        assert!(worst.local_ssim.is_none());
        let identical = compare(&reference, &reference);
        assert_eq!(identical.delta_e_ok_mean, 0.0);
        assert_eq!(identical.local_ssim, 1.0);
        assert!(identical
            .worst_tiles
            .iter()
            .all(|t| t.delta_e_ok_max == 0.0));
    }
}
