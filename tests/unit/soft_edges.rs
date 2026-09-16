mod tests {
    use super::*;
    #[test]
    fn wikipedia_soft_tip_is_detected_without_mixing_the_transparent_backing() {
        let input = image::load_from_memory(include_bytes!("../data/wiki-soft-source.png"))
            .unwrap()
            .to_rgba8();
        let previous = image::load_from_memory(include_bytes!("../data/wiki-soft-before.png"))
            .unwrap()
            .to_rgba8();
        let matte = AlphaMatte::from_u8(96, 96, input.pixels().map(|p| p[3]).collect());
        let composite = |image: &image::RgbaImage| {
            Raster::new(
                96,
                96,
                image
                    .pixels()
                    .map(|p| {
                        let a = p[3] as f32 / 255.0;
                        [0, 1, 2].map(|c| p[c] as f32 / 255.0 * a + 1.0 - a)
                    })
                    .collect(),
            )
        };
        let source = composite(&input);
        let render = composite(&previous);
        let proposals = propose(&source, &render, Some(&matte));
        assert!(
            proposals
                .iter()
                .any(|p| p.x == 48 && p.y == 24 && p.sigma <= 1.5),
            "blurred source tip was missed"
        );
    }

    #[test]
    fn filtered_svg_improves_soft_detail_without_raster_embedding() {
        let document = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"72\" height=\"72\"><defs></defs><rect width=\"72\" height=\"72\" fill=\"#ccc\"/><path d=\"M43 30H54V42H43Z\" fill=\"#5a5a5a\"/></svg>";
        let render = |document: &str| -> Result<Raster> {
            let tree = resvg::usvg::Tree::from_str(document, &resvg::usvg::Options::default())?;
            let mut pixmap = resvg::tiny_skia::Pixmap::new(72, 72).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );
            Ok(Raster::new(
                72,
                72,
                pixmap
                    .pixels()
                    .iter()
                    .map(|p| {
                        [
                            p.red() as f32 / 255.0,
                            p.green() as f32 / 255.0,
                            p.blue() as f32 / 255.0,
                        ]
                    })
                    .collect(),
            ))
        };
        let before = render(&Document::from((document).to_string())).unwrap();
        let input = image::RgbImage::from_fn(72, 72, |x, y| {
            image::Rgb(
                before
                    .get(x as usize, y as usize)
                    .map(|v| (v * 255.0).round() as u8),
            )
        });
        let soft = image::imageops::blur(&input, 1.4);
        let source = Raster::new(
            72,
            72,
            soft.pixels()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        );
        let updated = refine(
            &Document::from((document).to_string()),
            &source,
            None,
            render,
        )
        .unwrap();
        assert!(updated.contains("feGaussianBlur"));
        assert!(!updated.contains("<image"));
        let after = render(&updated).unwrap();
        let patch = Patch {
            x: 24,
            y: 24,
            size: 24,
            sigma: 0.0,
            gain: 0.0,
        };
        assert!(error(&source, &after, patch) < 0.9 * error(&source, &before, patch));
        assert!(
            (after.get(48, 30)[0] - source.get(48, 30)[0]).abs()
                < (before.get(48, 30)[0] - source.get(48, 30)[0]).abs() * 0.8,
            "internal tile edge retained a hard stripe"
        );
        assert_eq!(before.get(10, 10), after.get(10, 10));
        assert_eq!(
            refine(
                &Document::from((document).to_string()),
                &before,
                None,
                render
            )
            .unwrap(),
            Document::from(document)
        );
    }

    #[test]
    fn soft_source_detail_is_distinguished_from_a_sharp_detail() {
        let mut image = image::RgbImage::from_pixel(72, 72, image::Rgb([204; 3]));
        for y in 30..42 {
            for x in 31..38 {
                image.put_pixel(x, y, image::Rgb([90; 3]));
            }
        }
        let raster = |image: &image::RgbImage| {
            Raster::new(
                72,
                72,
                image
                    .pixels()
                    .map(|p| p.0.map(|v| v as f32 / 255.0))
                    .collect(),
            )
        };
        let render = raster(&image);
        let source = raster(&image::imageops::blur(&image, 1.4));
        let proposals = propose(&source, &render, None);
        assert_eq!(proposals.len(), 1);
        assert!((proposals[0].sigma - 1.5).abs() < 0.1);
        assert!(propose(&render, &render, None).is_empty());
    }
}
