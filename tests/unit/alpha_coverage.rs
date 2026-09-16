mod tests {

    #[test]
    fn local_coverage_keeps_a_thin_authored_interior_opacity_ramp() {
        let (w, h) = (80, 80);
        let mut values = vec![255; w * h];
        for y in 30..50 {
            for x in 39..41 {
                values[y * w + x] = (80 + 5 * (y - 30)) as u8;
            }
        }
        let matte = AlphaMatte::from_u8(w, h, values);
        assert!(
            detect_components(&matte).is_none(),
            "opaque neighbours alone must not flatten a translucent interior line"
        );
    }

    #[test]
    fn disconnected_gradient_does_not_reclassify_an_antialiased_illustration() {
        let source = image::load_from_memory(include_bytes!("../data/cliparts-woman-coverage.png"))
            .unwrap()
            .to_rgba8();
        let (cw, h) = (source.width() as usize, source.height() as usize);
        let w = cw + 120;
        let mut values = vec![0; w * h];
        for y in 0..h {
            for x in 0..cw {
                values[y * w + x] = source.get_pixel(x as u32, y as u32)[3];
            }
            for x in cw + 20..w - 10 {
                values[y * w + x] = (80 + (x - cw) / 2) as u8;
            }
        }
        let matte = AlphaMatte::from_u8(w, h, values);
        assert!(
            detect(&matte).is_none(),
            "mixed artwork has authored transparency"
        );
        let local = detect_components(&matte).expect("opaque illustration has local coverage");
        let face = (510 - 458) * w + (315 - 26);
        assert_eq!(local.opacity[face], 1.0);
        assert_eq!(
            local.opacity[50 * w + cw + 50],
            0.0,
            "preserve the neighbouring authored fade"
        );
    }
    use super::*;

    #[test]
    fn remojii_edge_samples_are_coverage_not_a_stack_of_opacity_bands() {
        let alpha = image::load_from_memory(include_bytes!("../data/remojii-rim-alpha.png"))
            .unwrap()
            .to_luma8();
        let matte = AlphaMatte::from_u8(
            alpha.width() as usize,
            alpha.height() as usize,
            alpha.into_raw(),
        );
        let coverage = detect(&matte).expect("nearly opaque artwork should have one silhouette");
        assert!((coverage.opacity - 254.0 / 255.0).abs() < 0.0001);
    }

    #[test]
    fn small_authored_translucent_patch_is_not_absorbed_into_an_opaque_object() {
        let mut values = vec![0; 128 * 128];
        for y in 8..120 {
            for x in 8..120 {
                values[y * 128 + x] = 255;
            }
        }
        for y in 60..64 {
            for x in 8..12 {
                values[y * 128 + x] = 128;
            }
        }
        let matte = AlphaMatte::from_u8(128, 128, values);
        assert!(detect(&matte).is_none());
    }

    #[test]
    fn multi_colour_exterior_does_not_get_a_uniform_rgb_support() {
        let mut values = vec![0; 64 * 64];
        let mut source = Raster::blank(64, 64, [1.0, 0.0, 0.0]);
        for y in 8..56 {
            for x in 8..56 {
                values[y * 64 + x] = 255;
                if x > 32 {
                    source.pixels[y * 64 + x] = [0.0, 0.0, 1.0];
                }
            }
        }
        let matte = AlphaMatte::from_u8(64, 64, values);
        let coverage = detect(&matte).unwrap();
        assert!(coverage.exterior_colour(&source, &matte).is_none());
    }
}
