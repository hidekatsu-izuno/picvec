mod tests {
    use super::*;

    #[test]
    fn percentile_selection_matches_sorted_interpolation() {
        for length in 1..258 {
            let values = (0..length)
                .map(|i| ((i * 73 + length * 31) % 101) as f32 - 50.0)
                .collect::<Vec<_>>();
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            for q in [-1.0_f32, 0.0, 0.01, 0.25, 0.5, 0.9, 0.99, 1.0, 2.0] {
                let position = q.clamp(0.0, 1.0) * (length - 1) as f32;
                let low = position.floor() as usize;
                let high = position.ceil() as usize;
                let amount = position - low as f32;
                let expected = if low == high {
                    sorted[low]
                } else {
                    sorted[low] * (1.0 - amount) + sorted[high] * amount
                };
                assert_eq!(percentile(values.clone(), q).to_bits(), expected.to_bits());
            }
        }
        for values in [
            vec![],
            vec![-0.0, 0.0],
            vec![f32::NEG_INFINITY, 1.0, f32::INFINITY],
        ] {
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            for q in [0.0, 0.5, 1.0] {
                let expected = if sorted.is_empty() {
                    0.0
                } else {
                    let position = q * (sorted.len() - 1) as f32;
                    let low = position.floor() as usize;
                    let high = position.ceil() as usize;
                    if low == high {
                        sorted[low]
                    } else {
                        let amount = position - low as f32;
                        sorted[low] * (1.0 - amount) + sorted[high] * amount
                    }
                };
                assert_eq!(percentile(values.clone(), q).to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn input_dimensions_are_limited_before_full_decode() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("large.png");
        ImageBuffer::from_pixel(16, 8, Rgb([0_u8, 0, 0]))
            .save(&path)
            .unwrap();

        assert!(Raster::load(&path, 8, 1_000, 64 * 1024 * 1024).is_err());
        assert!(Raster::load(&path, 16, 100, 64 * 1024 * 1024).is_err());
        let loaded = Raster::load(&path, 16, 1_000, 64 * 1024 * 1024).unwrap();
        assert_eq!((loaded.width, loaded.height), (16, 8));
    }

    #[test]
    fn checked_decoder_enforces_pixel_and_output_buffer_limits() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("limits.png");
        ImageBuffer::from_pixel(1024, 1024, Rgb([0_u8, 0, 0]))
            .save(&path)
            .unwrap();
        let error = Raster::load(&path, 1024, 1024 * 1024 - 1, 64 * 1024 * 1024).unwrap_err();
        assert!(error.to_string().contains("1048576 pixels"), "{error}");
        assert!(Raster::load(&path, 1024, 1024 * 1024, 1024 * 1024).is_err());
        assert!(Raster::load(&path, 1024, 1024 * 1024, 64 * 1024 * 1024).is_ok());
    }

    #[test]
    fn alpha_aware_load_does_not_flatten_rgb_onto_white() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("alpha.png");
        let image = image::RgbaImage::from_pixel(2, 1, image::Rgba([12, 34, 56, 0]));
        image.save(&path).unwrap();
        let (raster, alpha) =
            SourceRaster::load_with_alpha(&path, 64, 64 * 64, 64 * 1024 * 1024).unwrap();
        assert_eq!(raster.get(0, 0), [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]);
        assert_eq!(alpha.unwrap(), vec![0, 0]);
    }

    #[test]
    fn retained_source_uses_rgb8_or_unorm16_storage() {
        let rgb8 = SourceRaster::from_rgb8_fn(2, 1, |index| {
            if index == 0 {
                [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]
            } else {
                [1.0, 0.0, 0.5]
            }
        });
        let SourcePixels::Rgb8(values) = &rgb8.pixels else {
            panic!("decoded source should use RGB8 storage");
        };
        assert_eq!(values.len(), 6);
        assert_eq!(rgb8.get(0, 0), [12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0]);

        let expected = [[0.123_456, 0.5, 0.987_654], [0.0, 1.0, 0.25]];
        let unorm16 = SourceRaster::from_unorm16_fn(2, 1, |index| expected[index]);
        let SourcePixels::Unorm16(values) = &unorm16.pixels else {
            panic!("derived source should use Q0.16 storage");
        };
        assert_eq!(values.len() * std::mem::size_of::<u16>(), 12);
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = unorm16.get(index, 0);
            for channel in 0..3 {
                assert!((actual[channel] - expected[channel]).abs() <= 0.5 / 65_535.0);
            }
        }
    }

    #[test]
    fn crop_and_bilinear_sampling_preserve_source_coordinates() {
        let raster = Raster::new(
            3,
            2,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        );
        let crop = raster.crop(1, 0, 2, 2);
        assert_eq!((crop.width, crop.height), (2, 2));
        assert_eq!(crop.get(0, 0), [1.0, 0.0, 0.0]);
        assert_eq!(crop.get(1, 1), [1.0, 1.0, 1.0]);
        assert_eq!(raster.sample_bilinear(0.5, 0.0), [0.5, 0.0, 0.0]);
    }
}
