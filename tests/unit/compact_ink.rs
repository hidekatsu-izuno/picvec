mod tests {
    use super::*;

    #[test]
    fn uniform_ring_repair_keeps_the_counter_in_both_polarities() {
        for light in [false, true] {
            let (bg, fg) = if light { (0.08, 0.94) } else { (0.94, 0.1) };
            let source = SourceRaster::from_unorm16_fn(64, 64, |i| {
                let r = ((i % 64) as f32 + 0.5 - 32.0).hypot((i / 64) as f32 + 0.5 - 32.0);
                let a = ((r - 6.0).min(9.0 - r) + 0.5).clamp(0.0, 1.0);
                [bg + a * (fg - bg); 3]
            });
            let mut body = Elements::new();
            body.leaf(
                "rect",
                attrs([
                    ("width", "64".into()),
                    ("height", "64".into()),
                    ("fill", hex([bg; 3])),
                ]),
            );
            body.leaf(
                "circle",
                attrs([
                    ("cx", "32".into()),
                    ("cy", "32".into()),
                    ("r", "9".into()),
                    ("fill", hex([fg; 3])),
                ]),
            );
            let mut doc = Document::from_parts(64, 64, Elements::new(), body);
            assert!(refine(&source, &mut doc, (64, 64)) > 0);
            for scale in [1, 4] {
                let p = render(&doc, 64 * scale, 64 * scale).unwrap();
                let c = p.pixels()[32 * scale * 64 * scale + 32 * scale];
                assert!((c.red() as f32 / 255.0 - bg).abs() < 0.02);
            }
            assert!(!doc.contains("<circle"), "fully replaced old ink retained");
            assert!(!doc.contains("<mask"));
            assert!(!doc.contains("<image"));
        }
    }

    #[test]
    fn real_glyph_contours_keep_their_small_counters() {
        let (source, _) = SourceRaster::from_decoded(
            image::load_from_memory(include_bytes!("../data/catwhale-small-text.png")).unwrap(),
        );
        let regions = regions(&source);
        let patches: Vec<_> = regions.iter().filter_map(|r| patch(&source, r)).collect();
        assert!(!patches.is_empty());
        for (x, y) in [(23, 25), (35, 22), (59, 26)] {
            let p = patches
                .iter()
                .find(|p| {
                    x >= p.rect.x
                        && y >= p.rect.y
                        && x < p.rect.x + p.rect.width
                        && y < p.rect.y + p.rect.height
                })
                .expect("glyph patch");
            let c = p.pixels[(y - p.rect.y) * p.rect.width + x - p.rect.x];
            assert!(intensity([c[0], c[1], c[2]]) > 0.65, "counter at {x},{y}");
        }
    }

    #[test]
    fn broad_shapes_are_not_local_ink() {
        let source = SourceRaster::from_unorm16_fn(100, 100, |i| {
            if (20..80).contains(&(i % 100)) && (20..80).contains(&(i / 100)) {
                [0.1; 3]
            } else {
                [0.9; 3]
            }
        });
        assert!(regions(&source).iter().all(|r| patch(&source, r).is_none()));
    }

    #[test]
    fn a_separate_colour_is_not_erased_as_antialiasing() {
        let source = SourceRaster::from_unorm16_fn(64, 64, |i| {
            if i == 32 * 64 + 39 {
                return [0.9, 0.05, 0.05];
            }
            let r = ((i % 64) as f32 + 0.5 - 32.0).hypot((i / 64) as f32 + 0.5 - 32.0);
            let a = ((r - 6.0).min(9.0 - r) + 0.5).clamp(0.0, 1.0);
            [0.94 - a * 0.84; 3]
        });
        let region = Region {
            rect: SourceRect {
                x: 18,
                y: 18,
                width: 28,
                height: 28,
            },
            foreground: [0.1; 3],
            background: [0.94; 3],
        };
        assert!(patch(&source, &region).is_none());
    }

    #[test]
    fn distant_rgba_is_unchanged_on_an_unevenly_scaled_canvas() {
        let source = SourceRaster::from_unorm16_fn(128, 64, |i| {
            let r = ((i % 128) as f32 + 0.5 - 32.0).hypot((i / 128) as f32 + 0.5 - 32.0);
            let a = ((r - 6.0).min(9.0 - r) + 0.5).clamp(0.0, 1.0);
            [0.94 - a * 0.84; 3]
        });
        let mut body = Elements::new();
        body.open("g", attrs([("transform", "scale(0.625 0.640625)".into())]));
        body.leaf(
            "rect",
            attrs([
                ("width", "128".into()),
                ("height", "64".into()),
                ("fill", hex([0.94; 3])),
            ]),
        );
        body.leaf(
            "circle",
            attrs([
                ("cx", "32".into()),
                ("cy", "32".into()),
                ("r", "9".into()),
                ("fill", hex([0.1; 3])),
            ]),
        );
        body.leaf(
            "path",
            attrs([
                ("d", "M90 8L120 57".into()),
                ("stroke", "red".into()),
                ("stroke-width", "0.4".into()),
                ("stroke-opacity", "0.3".into()),
                ("fill", "none".into()),
            ]),
        );
        body.close();
        let original = Document::from_parts(80, 41, Elements::new(), body);
        let mut changed = original.clone();
        assert!(refine(&source, &mut changed, (80, 41)) > 0);
        for scale in [1, 4] {
            let a = render(&original, 128 * scale, 64 * scale).unwrap();
            let b = render(&changed, 128 * scale, 64 * scale).unwrap();
            for y in 0..64 * scale {
                for x in 80 * scale..128 * scale {
                    let i = y * 128 * scale + x;
                    assert_eq!(
                        a.pixels()[i],
                        b.pixels()[i],
                        "outside crop at {x},{y}, scale {scale}"
                    );
                }
            }
        }
        assert!(changed.contains("stroke-opacity"));
    }

    #[test]
    fn adjacent_paragraph_rows_remain_available_as_local_candidates() {
        // Unmodified source crop: catwhale.png, (1070, 830), 270x190.
        let (source, _) = SourceRaster::from_decoded(
            image::load_from_memory(include_bytes!("../data/catwhale-paragraph.png")).unwrap(),
        );
        let patches = proposals(&source);
        assert!(
            patches.iter().any(|p| p.rect.y < 30),
            "first line lost through grouping"
        );
        assert!(
            patches.iter().any(|p| p.rect.y > 130),
            "last lines lost through grouping"
        );
    }
}
