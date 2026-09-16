// Materialize RGBA paint before source-over and the complete group before its
// clip. CSS isolation alone is not a compositing boundary in Chromium's SVG
// image renderer. This neutral sRGB stage changes neither colour nor alpha;
// it adds no blur, mask, fixed raster resolution or embedded bitmap.
pub(crate) const COMPOSITE_FILTER: &str = r#"<filter id="source-field-composite" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0"/></filter>"#;

mod tests {
    use super::*;

    fn fixture(bytes: &[u8]) -> (Raster, AlphaMatte) {
        let image = image::load_from_memory(bytes).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        (
            Raster::new(
                w as usize,
                h as usize,
                image
                    .pixels()
                    .map(|p| std::array::from_fn(|k| p[k] as f32 / 255.0))
                    .collect(),
            ),
            AlphaMatte::from_u8(
                w as usize,
                h as usize,
                image.pixels().map(|p| p[3]).collect(),
            ),
        )
    }

    fn flat_patch(alpha: u8) -> Patch {
        let source = Raster::blank(32, 32, [230.0 / 255.0, 204.0 / 255.0, 51.0 / 255.0]);
        let matte = AlphaMatte::from_u8(32, 32, vec![alpha; 32 * 32]);
        let layers = generate(&source, &matte, 32, 2_000_000).unwrap_or_default();
        Patch {
            origin_x: 0,
            origin_y: 0,
            x: 4,
            y: 4,
            width: 24,
            height: 24,
            source,
            matte,
            layers,
        }
    }

    #[test]
    fn authored_transparency_never_receives_boundary_underpaint() {
        for alpha in [0, 128, 253, 254] {
            assert!(opaque_boundary_overlap(&flat_patch(alpha)).is_empty());
        }
        assert!(!opaque_boundary_overlap(&flat_patch(255)).is_empty());
    }

    #[test]
    fn boundary_matching_does_not_copy_preview_coverage_errors() {
        for (source_alpha, preview_alpha) in [(255, 0), (128, 255), (0, 255)] {
            let p = flat_patch(source_alpha);
            let mut base = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
            base.fill(resvg::tiny_skia::Color::from_rgba8(
                230,
                204,
                51,
                preview_alpha,
            ));
            let mut svg =
                String::from(r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">"#);
            for layer in match_boundary(&p, &base) {
                write!(svg, "<path d=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\"/>", layer.path, layer.color[0], layer.color[1], layer.color[2], layer.opacity).unwrap();
            }
            svg.push_str("</svg>");
            let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
            let mut rendered = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut rendered.as_mut(),
            );
            for (x, y) in [(4, 16), (5, 16), (16, 16), (27, 16)] {
                let actual = rendered.pixel(x, y).unwrap().alpha();
                // Eight independently quantized source-over factors accumulate rounding.
                assert!(
                    (actual as i16 - source_alpha as i16).abs() <= 5,
                    "source={source_alpha} preview={preview_alpha} ({x},{y}) actual={actual}"
                );
            }
        }
    }

    #[test]
    fn isolated_patch_keeps_an_opaque_boundary_at_fractional_zoom() {
        let p = flat_patch(255);
        let mut base = resvg::tiny_skia::Pixmap::new(32, 32).unwrap();
        base.fill(resvg::tiny_skia::Color::from_rgba8(230, 204, 51, 255));
        let mut outside = String::from("M0 0H32V32H0ZM4 4h24v24h-24Z");
        for [x, y, w, h] in opaque_boundary_overlap(&p) {
            write!(outside, "M{x} {y}h{w}v{h}h-{w}Z").unwrap();
        }
        let mut svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs>{COMPOSITE_FILTER}<clipPath id="outside"><path d="{outside}" clip-rule="evenodd"/></clipPath><clipPath id="inside"><rect x="4" y="4" width="24" height="24"/></clipPath></defs><g clip-path="url(#outside)"><rect width="32" height="32" fill="#e6cc33"/></g><g clip-path="url(#inside)" style="isolation:isolate" filter="url(#source-field-composite)">"##
        );
        for layer in match_boundary(&p, &base) {
            write!(svg, "<path d=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.7}\" fill-rule=\"evenodd\" filter=\"url(#source-field-composite)\"/>", layer.path, layer.color[0], layer.color[1], layer.color[2], layer.opacity).unwrap();
        }
        svg.push_str("</g></svg>");
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        for scale in [1.0_f32, 3.3, 8.0] {
            let size = (32.0 * scale).ceil() as u32 + 2;
            let mut rendered = resvg::tiny_skia::Pixmap::new(size, size).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, 0.37, 0.21),
                &mut rendered.as_mut(),
            );
            for y in (2.0 * scale).ceil() as u32..(30.0 * scale).floor() as u32 {
                for x in (2.0 * scale).ceil() as u32..(30.0 * scale).floor() as u32 {
                    let pixel = rendered.pixel(x, y).unwrap();
                    assert_eq!(pixel.alpha(), 255, "scale={scale}, x={x}, y={y}");
                    for (actual, expected) in [pixel.red(), pixel.green(), pixel.blue()]
                        .into_iter()
                        .zip([230, 204, 51])
                    {
                        assert!(
                            (actual as i16 - expected as i16).abs() <= 4,
                            "scale={scale}, x={x}, y={y}, actual={actual}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn source_over_reconstructs_translucent_chromatic_colours() {
        for rgb in [
            [0.8, 0.2, 0.4],
            [0.1, 0.9, 0.3],
            [0.3, 0.4, 0.95],
            [0.0; 3],
            [1.0; 3],
        ] {
            for alpha in [0.0, 0.2, 0.7, 1.0] {
                let mut pixel = [0.0; 4];
                for (k, a) in weights(rgb, alpha).into_iter().enumerate() {
                    for c in 0..3 {
                        pixel[c] = pixel[c] * (1.0 - a) + if k & (1 << c) > 0 { a } else { 0.0 };
                    }
                    pixel[3] = pixel[3] * (1.0 - a) + a;
                }
                for c in 0..3 {
                    assert!((pixel[c] - rgb[c] * alpha).abs() < 1e-5);
                }
                assert!((pixel[3] - alpha).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn transparent_source_and_exceeded_budget_emit_no_paint() {
        let source = Raster::new(16, 16, vec![[1.0, 0.7, 0.2]; 256]);
        assert!(generate(
            &source,
            &AlphaMatte::from_u8(16, 16, vec![0; 256]),
            32,
            100_000
        )
        .is_none());
        assert!(generate(&source, &AlphaMatte::from_u8(16, 16, vec![255; 256]), 32, 1).is_none());
    }

    #[test]
    fn fine_face_ink_and_dotted_rim_preserve_source_rgba() {
        for bytes in [
            include_bytes!("../data/cliparts-face-ink.png").as_slice(),
            include_bytes!("../data/cliparts-dotted-rim.png").as_slice(),
        ] {
            let (source, alpha) = fixture(bytes);
            let layers = generate(&source, &alpha, 32, 2_000_000).unwrap();
            assert!(supported(&source, &alpha, &layers, [7.0, 3.5]));
            assert!(layers.iter().all(|l| l.opacity > 0.0 && l.opacity <= 1.0));
        }
    }

    #[test]
    fn thin_materials_supply_ownership_including_the_mixed_junction() {
        for bytes in [
            include_bytes!("../data/cactus-spine-tip.png").as_slice(),
            include_bytes!("../data/cactus-spine-side.png").as_slice(),
        ] {
            let (source, alpha) = fixture(bytes);
            let pixels = narrow_material_pixels(&source, &alpha);
            assert!(
                pixels
                    .iter()
                    .filter(|&&(i, _, a)| a && alpha.get(i) < 0.98)
                    .count()
                    >= 8
            );
            assert!(pixels.iter().all(|&(i, _, _)| alpha.get(i) > 0.0));
            if source.width == 39 {
                assert!(
                    pixels.iter().any(|&(i, _, a)| i == 12 * 39 + 24 && a),
                    "opaque RGB(85,59,7) junction must remain material"
                );
            }
        }
    }

    #[test]
    fn narrow_rim_detector_requires_chroma_and_mixed_coverage() {
        let (source, alpha) = fixture(include_bytes!("../data/cactus-spine-tip.png"));
        assert!(!narrow_components(&source, &alpha).is_empty());
        let opaque =
            AlphaMatte::from_u8(source.width, source.height, vec![255; source.pixels.len()]);
        assert!(narrow_components(&source, &opaque).is_empty());
        let translucent =
            AlphaMatte::from_u8(source.width, source.height, vec![128; source.pixels.len()]);
        assert!(narrow_components(&source, &translucent).is_empty());
        let neutral = Raster::new(
            source.width,
            source.height,
            vec![[0.3; 3]; source.pixels.len()],
        );
        assert!(narrow_components(&neutral, &alpha).is_empty());
    }

    #[test]
    fn broad_flat_fill_without_weak_ink_is_not_extracted() {
        let mut source = Raster::new(64, 64, vec![[1.0, 0.8, 0.0]; 64 * 64]);
        let original = source.pixels.clone();
        let alpha = AlphaMatte::from_u8(64, 64, vec![255; 64 * 64]);
        let (remaining, patches) = extract(&mut source, &alpha, [1.0; 3]);
        assert!(patches.is_empty());
        assert_eq!(source.pixels, original);
        assert!((0..64 * 64).all(|i| remaining.get(i) == 1.0));
    }
}
