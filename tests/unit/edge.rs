mod tests {
    #[test]
    fn small_color_batches_match_parallel_transform_bit_for_bit() {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let values = [-0.1, -0.0, 0.0, 0.001, 0.04045, 0.04046, 0.5, 1.0, 1.1];
            for length in [
                0, 1, 31, 128, 255, 256, 257, 1023, 1024, 1025, 8191, 8192, 8193,
            ] {
                let pixels: Vec<_> = (0..length)
                    .map(|i| [values[i % 9], values[(i / 9) % 9], values[(i / 81) % 9]])
                    .collect();
                let expected: Vec<_> = pixels
                    .par_iter()
                    .copied()
                    .map(crate::color::rgb_to_oklab)
                    .collect();
                for (a, b) in super::oklab_values(&pixels).iter().zip(&expected) {
                    assert_eq!(
                        [a.l.to_bits(), a.a.to_bits(), a.b.to_bits()],
                        [b.l.to_bits(), b.a.to_bits(), b.b.to_bits()]
                    );
                }
                assert_eq!(super::oklab_values(&pixels).len(), expected.len());
            }
        });
    }

    #[test]
    fn dark_quantization_noise_smooths_without_erasing_a_black_line() {
        let config = crate::config::Config::default();
        let source = Raster::new(
            32,
            32,
            (0..1024)
                .map(|i| [((i % 32 + i / 32) % 2) as f32 / 255.0; 3])
                .collect(),
        );
        let filtered = super::perceptual_smooth(&source, &config);
        assert!((filtered.get(15, 15)[0] - filtered.get(16, 15)[0]).abs() < 0.1 / 255.0);

        let mut source = Raster::blank(32, 32, [20.0 / 255.0; 3]);
        for y in 0..32 {
            source.pixels[y * 32 + 15] = [0.0; 3];
            source.pixels[y * 32 + 16] = [0.0; 3];
        }
        let filtered = super::perceptual_smooth(&source, &config);
        assert!(filtered.get(15, 15)[0] < 1.0 / 255.0);
        assert!(filtered.get(14, 15)[0] > 18.0 / 255.0);
    }

    use super::{
        classify, classify_profile, nonoverlapping_extensions, point_tangents,
        width_profile_requires_paint, NumpyPcg64, ProfileRole, SourceEdge,
        PROFILE_OVERLAP_DISTANCE,
    };
    use crate::raster::Raster;

    fn edge(points: Vec<[f64; 2]>) -> SourceEdge {
        SourceEdge {
            points,
            width: 1.0,
            role: "test",
            width_samples: Vec::new(),
        }
    }

    fn direct_remove(candidates: Vec<SourceEdge>, owners: &[SourceEdge]) -> Vec<SourceEdge> {
        if owners.is_empty() {
            return candidates;
        }
        let owner_samples: Vec<([f64; 2], [f64; 2])> = owners
            .iter()
            .flat_map(|owner| {
                owner
                    .points
                    .iter()
                    .copied()
                    .zip(point_tangents(&owner.points))
            })
            .collect();
        let cosine = 20.0_f64.to_radians().cos();
        let mut result = Vec::new();
        for candidate in candidates {
            if candidate.points.len() < 2 {
                continue;
            }
            let tangents = point_tangents(&candidate.points);
            let keep: Vec<bool> = candidate
                .points
                .iter()
                .zip(&tangents)
                .map(|(point, tangent)| {
                    !owner_samples.iter().any(|(owner_point, owner_tangent)| {
                        (point[0] - owner_point[0]).hypot(point[1] - owner_point[1])
                            <= PROFILE_OVERLAP_DISTANCE
                            && (tangent[0] * owner_tangent[0] + tangent[1] * owner_tangent[1]).abs()
                                >= cosine
                    })
                })
                .collect();
            let mut index = 0_usize;
            while index < keep.len() {
                if !keep[index] {
                    index += 1;
                    continue;
                }
                let mut first = index;
                while index + 1 < keep.len() && keep[index + 1] {
                    index += 1;
                }
                let mut last = index;
                index += 1;
                first = first.saturating_sub(1);
                if last + 1 < candidate.points.len() {
                    last += 1;
                }
                if last > first {
                    result.push(SourceEdge {
                        points: candidate.points[first..=last].to_vec(),
                        width: candidate.width,
                        role: candidate.role,
                        width_samples: candidate.width_samples.clone(),
                    });
                }
            }
        }
        result
    }

    fn direct_extensions(
        mut candidates: Vec<SourceEdge>,
        owners: &[SourceEdge],
    ) -> Vec<SourceEdge> {
        candidates.sort_by(|first, second| {
            let length = |value: &SourceEdge| {
                value
                    .points
                    .windows(2)
                    .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
                    .sum::<f64>()
            };
            length(second).total_cmp(&length(first))
        });
        let mut accepted = owners.to_vec();
        let mut extensions = Vec::new();
        for candidate in candidates {
            let remaining = direct_remove(vec![candidate], &accepted);
            accepted.extend(remaining.iter().cloned());
            extensions.extend(remaining);
        }
        extensions
    }

    #[test]
    fn numpy_seed_zero_permutation_matches_reference() {
        assert_eq!(
            NumpyPcg64::permutation(10),
            vec![4, 6, 2, 7, 3, 5, 9, 0, 8, 1]
        );
    }

    #[test]
    fn indexed_profile_overlap_matches_direct_scan_and_incremental_ownership() {
        let owners = vec![edge(
            (-8..=8).map(|x| [x as f64 * 0.75 - 0.25, 2.25]).collect(),
        )];
        let candidates = vec![
            edge((-12..=12).map(|x| [x as f64 * 0.5, 3.1]).collect()),
            edge((-8..=8).map(|y| [0.25, y as f64 * 0.75]).collect()),
            edge((-8..=8).map(|x| [x as f64 * 0.75, -0.1]).collect()),
            edge((-8..=8).map(|x| [x as f64 * 0.75, 4.0]).collect()),
        ];
        let expected = direct_extensions(candidates.clone(), &owners);
        let actual = nonoverlapping_extensions(candidates, &owners);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.points.as_slice())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.points.as_slice())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn flat_image_stays_empty_without_a_secondary_classifier() {
        let image = Raster::blank(32, 24, [0.5, 0.5, 0.5]);
        let roles = classify(&image);
        assert_eq!(roles.summary.face_barrier_pixels, 0);
        assert_eq!(roles.summary.visible_ridge_coverage_pixels, 0);
        assert_eq!(roles.summary.dark_boundary_pixels, 0);
    }

    #[test]
    fn adaptive_profile_seeds_keep_low_contrast_step_evidence() {
        let mut image = Raster::blank(32, 24, [0.45, 0.45, 0.45]);
        for y in 0..24 {
            for x in 16..32 {
                image.pixels[y * 32 + x] = [0.55, 0.55, 0.55];
            }
        }
        let roles = classify(&image);
        assert!(roles.summary.skeleton_pixels > 0, "{:?}", roles.summary);
        assert!(roles.summary.shading_pixels > 0, "{:?}", roles.summary);
    }

    #[test]
    fn narrow_bright_extremum_keeps_boundary_polarity() {
        let offsets = (-28..=28)
            .map(|value| value as f32 * 0.25)
            .collect::<Vec<_>>();
        let samples = offsets
            .iter()
            .map(|&offset| {
                let side = if offset < 0.0 { 0.22 } else { 0.48 };
                let peak = (1.0 - offset.abs() / 1.25).max(0.0);
                [side + peak * (0.94 - side), 0.0, 0.0]
            })
            .collect::<Vec<_>>();

        let profile = classify_profile(&samples, &offsets, 7.0);

        assert_eq!(profile.role, ProfileRole::RidgeOnBoundary);
        assert!(profile.dark_contrast < -0.06);
        assert!(profile.width < 2.5);
    }

    #[test]
    fn variable_width_medial_mark_stays_paint_owned() {
        assert!(width_profile_requires_paint(
            &[6.0, 5.66, 4.47, 4.0, 2.83, 2.0, 2.0, 2.0, 2.0],
            2,
            1,
            1.0,
        ));
        assert!(!width_profile_requires_paint(
            &[2.0, 2.0, 2.0, 2.83, 2.0, 2.0, 2.0],
            1,
            1,
            1.0,
        ));
        assert!(!width_profile_requires_paint(
            &[
                6.32, 6.0, 6.0, 6.0, 5.66, 4.47, 4.0, 4.47, 4.0, 4.0, 4.0, 4.0, 2.83, 2.83, 2.83,
                2.83, 2.83,
            ],
            3,
            3,
            1.0,
        ));
    }
}
