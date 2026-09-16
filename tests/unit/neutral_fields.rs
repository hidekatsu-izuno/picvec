mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn source_over_factors_preserve_premultiplied_colour_and_alpha() {
        for alpha in [
            0.0,
            0.2,
            0.5,
            191.0 / 255.0,
            226.0 / 255.0,
            254.0 / 255.0,
            1.0,
        ] {
            for grey in [0.0, 0.1, 0.5, 0.8, 1.0] {
                let [black, white] = factors(alpha, grey);
                assert!((white - alpha * grey).abs() < 1e-6);
                assert!((white + black * (1.0 - white) - alpha).abs() <= 1.001 / 255.0);
            }
        }
        assert_eq!(
            factors(1.0, 1.0),
            [1.0, 1.0],
            "opaque white needs no black underpaint hole"
        );
        assert_eq!(factors(0.0, 1.0), [0.0, 0.0]);
    }

    #[test]
    fn flat_opacity_and_chromatic_materials_keep_the_existing_pipeline() {
        for colour in [[0.4; 3], [0.8, 0.2, 0.1]] {
            let mut source = Raster::blank(64, 64, colour);
            let matte = AlphaMatte::new(
                64,
                64,
                (0..4096)
                    .map(|i| {
                        if colour[0] == colour[1] {
                            0.73123
                        } else {
                            0.5 + (i % 64) as f32 / 128.0
                        }
                    })
                    .collect(),
            );
            let (remaining, layers) = extract(&mut source, &matte, [0.0; 3]);
            assert!(layers.is_empty());
            assert_eq!(source.pixels[123], colour);
            assert_eq!(remaining.get(123), matte.get(123));
        }
    }

    #[test]
    fn varying_opacity_on_a_constant_colour_does_not_create_two_fields() {
        let mut source = Raster::blank(96, 80, [0.4; 3]);
        let values = (0..80)
            .flat_map(|y| {
                (0..96).map(move |x| {
                    let r = ((x as f32 - 40.0) / 22.0).powi(2) + ((y as f32 - 30.0) / 14.0).powi(2);
                    191.0 / 255.0 + 0.15 * (1.0 - r).max(0.0)
                })
            })
            .collect();
        let matte = AlphaMatte::new(96, 80, values);
        let (remaining, layers) = extract(&mut source, &matte, [0.0; 3]);
        assert!(layers.is_empty());
        assert_eq!(remaining.get(30 * 96 + 40), matte.get(30 * 96 + 40));
        assert_eq!(source.pixels[30 * 96 + 40], [0.4; 3]);
    }

    #[test]
    fn near_black_source_pixels_do_not_leave_underpaint_pinholes() {
        let (w, h) = (96, 80);
        let mut source = Raster::blank(w, h, [0.0; 3]);
        let mut alpha = vec![0.0; w * h];
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let glow =
                    (1.0 - ((x as f32 - 36.0) / 23.0).powi(2) - ((y as f32 - 23.0) / 14.0).powi(2))
                        .max(0.0)
                        * 0.4;
                alpha[y * w + x] = 191.0 / 255.0 + (1.0 - 191.0 / 255.0) * glow;
                source.pixels[y * w + x] = [0.2 + 0.4 * x as f32 / w as f32; 3];
            }
        }
        for y in 44..60 {
            for x in 40..56 {
                alpha[y * w + x] = 1.0;
                source.pixels[y * w + x] = [0.0; 3];
            }
        }
        for code in 1..=5 {
            source.pixels[51 * w + 42 + code * 2] = [code as f32 / 255.0; 3];
        }
        let matte = AlphaMatte::new(w, h, alpha);
        let (_, layers) = extract(&mut source, &matte, [0.0; 3]);
        assert!(!layers.is_empty());
        let mut svg =
            format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">");
        for layer in layers {
            write!(
                svg,
                "<path d=\"{}\" fill=\"{}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\"/>",
                layer.path,
                if layer.white { "white" } else { "black" },
                layer.opacity
            )
            .unwrap();
        }
        svg.push_str("</svg>");
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut rendered = resvg::tiny_skia::Pixmap::new((w * 4) as u32, (h * 4) as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_scale(4.0, 4.0),
            &mut rendered.as_mut(),
        );
        for code in 1..=5 {
            let j = ((51 * 4 + 2) * w * 4 + (42 + code * 2) * 4 + 2) * 4;
            assert!(
                rendered.data()[j + 3] >= 254,
                "coverage hole at grey code {code}"
            );
            assert!(rendered.data()[j] <= 8, "bright speck at grey code {code}");
        }
    }

    #[test]
    fn coupled_fields_with_an_opaque_face_render_without_separate_alpha_cells() {
        let (w, h) = (96, 80);
        let mut pixels = Vec::new();
        let mut alphas = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let xx = x as f32 + 0.5;
                let yy = y as f32 + 0.5;
                let glow = (1.0 - ((xx - 36.0) / 23.0).powi(2) - ((yy - 23.0) / 14.0).powi(2))
                    .max(0.0)
                    * 0.5;
                let radius = (((xx - 53.0) / 24.0).powi(2) + ((yy - 51.0) / 19.0).powi(2)).sqrt();
                let face = ((1.0 - radius) * 19.0 + 0.5).clamp(0.0, 1.0);
                let white = 1.0 - (1.0 - glow) * (1.0 - face);
                let base = 191.0 / 255.0;
                let grey = 0.2 + 0.4 * xx / w as f32;
                let alpha = if x < 4 || y < 4 || x >= w - 4 || y >= h - 4 {
                    0.0
                } else {
                    white + base * (1.0 - white)
                };
                pixels.push(
                    [(white + grey * base * (1.0 - white)) / (white + base * (1.0 - white)); 3],
                );
                alphas.push(alpha);
            }
        }
        let original = Raster::new(w, h, pixels);
        let matte = AlphaMatte::new(w, h, alphas);
        let mut processing = original.clone();
        let (remaining, layers) = extract(&mut processing, &matte, [0.1, 0.2, 0.3]);
        assert!(!layers.is_empty());
        assert!(layers.len() <= 128);
        assert_eq!(
            remaining.get(51 * w + 53),
            0.0,
            "the component must leave segmentation before curve fitting"
        );
        assert_eq!(processing.pixels[51 * w + 53], [0.1, 0.2, 0.3]);
        let mut svg =
            format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">");
        for layer in &layers {
            assert!(layer.opacity > 0.0 && layer.opacity <= 1.0);
            let _ = write!(
                svg,
                "<path d=\"{}\" fill=\"{}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\"/>",
                layer.path,
                if layer.white { "white" } else { "black" },
                layer.opacity
            );
        }
        svg.push_str("</svg>");
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        for scale in [1, 4] {
            let mut rendered =
                resvg::tiny_skia::Pixmap::new((w * scale) as u32, (h * scale) as u32).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
                &mut rendered.as_mut(),
            );
            let (mut alpha_error, mut colour_error, mut count) = (0.0, 0.0, 0);
            // Compare the coupled field itself, away from the fixture's hard
            // rectangular outer crop. The opaque face edge is included.
            for y in 7..h - 7 {
                for x in 7..w - 7 {
                    let i = y * w + x;
                    let j = ((y * scale + scale / 2) * w * scale + x * scale + scale / 2) * 4;
                    alpha_error += (rendered.data()[j + 3] as f32 - matte.get(i) * 255.0).abs();
                    colour_error += (rendered.data()[j] as f32
                        - original.pixels[i][0] * matte.get(i) * 255.0)
                        .abs();
                    count += 1;
                }
            }
            assert!(
                alpha_error / (count as f32) < 4.0,
                "alpha at {scale}x: {}",
                alpha_error / count as f32
            );
            assert!(
                colour_error / (count as f32) < 4.0,
                "premultiplied colour at {scale}x: {}",
                colour_error / count as f32
            );
        }
    }
}
