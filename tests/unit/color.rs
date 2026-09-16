mod tests {
    use super::*;

    #[test]
    fn published_oklab_primaries_and_neutrals() {
        for (rgb, expected) in [
            ([0.0; 3], [0.0, 0.0, 0.0]),
            ([1.0; 3], [100.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [62.7955, 22.4863, 12.5846]),
            ([0.0, 1.0, 0.0], [86.6440, -23.3888, 17.9498]),
            ([0.0, 0.0, 1.0], [45.2014, -3.2457, -31.1528]),
        ] {
            let actual = rgb_to_oklab(rgb);
            for (a, b) in [actual.l, actual.a, actual.b].into_iter().zip(expected) {
                assert!((a - b).abs() < 0.002, "{rgb:?}: {actual:?}");
            }
        }
    }

    #[test]
    fn oklab_round_trip_preserves_gamut_and_dark_branch() {
        for r in [0.0, 0.001, 0.04, 0.1, 0.5, 1.0] {
            for g in [0.0, 0.003, 0.04045, 0.5, 1.0] {
                for b in [0.0, 0.02, 0.1, 0.5, 1.0] {
                    let rgb = [r, g, b];
                    let restored = oklab_to_rgb(rgb_to_oklab(rgb));
                    for c in 0..3 {
                        assert!(
                            (rgb[c] - restored[c]).abs() < 0.000_03,
                            "{rgb:?}: {restored:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn oklab_distance_units_and_batches() {
        let a = Oklab {
            l: 50.0,
            a: 0.0,
            b: 0.0,
        };
        let b = Oklab {
            l: 50.0,
            a: 3.0,
            b: 4.0,
        };
        assert_eq!(delta_e_ok(a, b), 5.0);
        assert_eq!(delta_e_ok(b, a), 5.0);
        assert_eq!(delta_e_ok(a, a), 0.0);
        assert_eq!(delta_e_ok_pairs(&[a, b], &[b, a]), vec![5.0, 5.0]);
        assert_eq!(delta_e_ok_to_many(&[a, b], a), vec![0.0, 5.0]);
    }
}
