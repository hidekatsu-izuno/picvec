mod tests {
    use super::*;

    #[test]
    fn real_small_glyphs_keep_their_counters_and_leave_no_old_ink_in_the_base() {
        // Unmodified 80x40 crop at (420, 860) in sample/input/catwhale.png.
        let (source, alpha) = SourceRaster::from_decoded(
            image::load_from_memory(include_bytes!("../data/catwhale-small-text.png")).unwrap(),
        );
        assert!(alpha.is_none());
        let (ink, cleaned) = extract(&source);
        assert!(ink.components > 0);
        let cleaned = cleaned.unwrap();
        // Stroke interiors from the three connected glyphs, not the accent.
        for (x, y) in [(16, 16), (38, 20), (61, 26)] {
            assert!(intensity(source.get(x, y)) < 0.65);
            assert!(
                intensity(cleaned.get(x, y)) > 0.8,
                "old ink remains at {x},{y}"
            );
        }
        let mut document = Document::from_parts(80, 40, Elements::new(), Elements::new());
        ink.append(&mut document, [1.0, 1.0]);
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        for scale in [1, 4] {
            let mut pixmap = resvg::tiny_skia::Pixmap::new(80 * scale, 40 * scale).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
                &mut pixmap.as_mut(),
            );
            for (x, y) in [(23, 25), (35, 22), (59, 26)] {
                let mut alpha = 0;
                for yy in y * scale..(y + 1) * scale {
                    for xx in x * scale..(x + 1) * scale {
                        alpha += pixmap.pixels()[(yy * 80 * scale + xx) as usize].alpha() as u32;
                    }
                }
                assert!(
                    alpha < 170 * scale * scale,
                    "counter lost at {x},{y}, {scale}x"
                );
            }
        }
        assert!(!document.contains("<mask"));
        assert!(!document.contains("<image"));
    }

    #[test]
    fn broad_objects_and_nonuniform_backgrounds_are_not_removed() {
        for gradient in [false, true] {
            let source = SourceRaster::from_unorm16_fn(100, 100, |i| {
                let (x, y) = (i % 100, i / 100);
                if (20..80).contains(&x) && (20..80).contains(&y) {
                    [0.1; 3]
                } else if gradient {
                    [x as f32 / 100.0, y as f32 / 100.0, 0.8]
                } else {
                    [0.9; 3]
                }
            });
            let (ink, cleaned) = extract(&source);
            assert_eq!(ink.components, 0);
            assert!(cleaned.is_none());
        }
    }

    #[test]
    fn scanned_annotations_have_source_supported_compact_silhouettes() {
        let (source, _) = SourceRaster::from_decoded(
            image::load_from_memory(include_bytes!("../data/booster-annotations.png")).unwrap(),
        );
        let (ink, cleaned) = extract(&source);
        assert!(
            ink.components >= 20,
            "only {} supported silhouettes",
            ink.components
        );
        assert!(cleaned.is_some());
        let cleaned = cleaned.unwrap();
        let mut document = Document::from_parts(
            source.width,
            source.height,
            Elements::new(),
            Elements::new(),
        );
        ink.append(&mut document, [1.0, 1.0]);
        let tree =
            resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).unwrap();
        let mut pixmap =
            resvg::tiny_skia::Pixmap::new(source.width as u32, source.height as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let mut error = 0.0;
        let mut ink_mass = 0.0;
        for (i, p) in pixmap.pixels().iter().enumerate() {
            let a = p.alpha() as f32 / 255.0;
            let bg = cleaned.get(i % source.width, i / source.width);
            let original = source.get(i % source.width, i / source.width);
            let rgb = [p.red(), p.green(), p.blue()];
            ink_mass += a;
            for c in 0..3 {
                error += (rgb[c] as f32 / 255.0 + bg[c] * (1.0 - a) - original[c]).abs();
            }
        }
        assert!(
            error / (3.0 * ink_mass) < 0.18,
            "serialized ink error: {}",
            error / (3.0 * ink_mass)
        );
    }

    #[test]
    fn two_colour_rings_work_in_both_polarities_but_do_not_erase_a_coloured_mark() {
        for light in [false, true] {
            for marked in [false, true] {
                let background = if light { [0.08; 3] } else { [0.94; 3] };
                let foreground = if light { [0.94; 3] } else { [0.1, 0.2, 0.3] };
                let source = SourceRaster::from_unorm16_fn(64, 64, |i| {
                    let (x, y) = ((i % 64) as f32 + 0.5, (i / 64) as f32 + 0.5);
                    let r = (x - 32.0).hypot(y - 32.0);
                    let a = ((r - 6.0).min(9.0 - r) + 0.5).clamp(0.0, 1.0);
                    if marked && i == 32 * 64 + 39 {
                        return [0.9, 0.05, 0.05];
                    }
                    std::array::from_fn(|c| background[c] + a * (foreground[c] - background[c]))
                });
                let (ink, cleaned) = extract(&source);
                if marked {
                    assert_eq!(ink.components, 0, "independent paint was erased");
                } else {
                    assert!(ink.components > 0);
                    let clean = cleaned.unwrap();
                    for c in 0..3 {
                        assert!((clean.get(39, 32)[c] - background[c]).abs() < 0.01);
                    }
                }
            }
        }
    }
}
