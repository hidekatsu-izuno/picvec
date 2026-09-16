mod tests {
    use super::*;

    #[test]
    fn lazy_resources_preserve_inherited_context_and_ignore_unused_definitions() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="139" height="193"><defs><g id="unused"><use href="#missing"/></g></defs><g id="parent" fill="#39a" fill-rule="evenodd"><path id="shape" d="M4 4H28V38H4Z M10 10H20V20H10Z"/></g><use href="#shape" x="50"/><use href="#parent" y="100"/></svg>"##;
        let cache = Cache::new(&svg.into()).unwrap();
        compare(&cache, svg);
        compare(&cache, &svg.replace("#39a", "#c52"));
        compare(&cache, &svg.replace("H28", "H32"));
    }

    fn compare(cache: &Cache, svg: &str) {
        let scene = cache.scene(&Document::from((svg).to_string())).unwrap();
        let full = Tree::from_str(svg, &Options::default()).unwrap();
        for scale in [1, 4] {
            for y in (0..193).step_by(64) {
                let mut expected =
                    Pixmap::new(139 * scale as u32, (193 - y).min(64) as u32 * scale as u32)
                        .unwrap();
                let mut actual = expected.clone();
                let transform = Transform::from_row(
                    scale as f32,
                    0.0,
                    0.0,
                    scale as f32,
                    0.0,
                    -((y * scale) as f32),
                );
                resvg::render(&full, transform, &mut expected.as_mut());
                scene.render(scale, y, &mut actual);
                assert_eq!(actual.data(), expected.data(), "scale={scale}, y={y}");
            }
        }
    }

    #[test]
    fn cached_fragments_preserve_clips_strokes_filters_and_changed_definitions() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="139" height="193"><defs>
<linearGradient id="g"><stop stop-color="#d24"/><stop offset="1" stop-color="#2ad" stop-opacity=".3"/></linearGradient>
<linearGradient id="linked" xlink:href="#g"/>
<clipPath id="c"><path d="M0 50L130 50L130 150L0 150Z"/></clipPath>
<path id="shape" d="M1 1L110 1L110 190L1 190Z"/>
<filter id="f" x="-.2" y="-.2" width="1.4" height="1.4"><feGaussianBlur stdDeviation="2"/></filter>
</defs><g id="paint-layer" fill-rule="evenodd"><path d="M0 0L139 0L139 193L0 193Z M10 90L20 90L20 100L10 100Z" fill="url(#g)"/><g fill="url(#linked)"><rect x="10.3" y="21.7" width="91.4" height="75.7"/></g><g clip-path="url(#c)"><use xlink:href="#shape" fill="#282" fill-opacity=".38"/></g><g transform="translate(1.7 14.3) rotate(3)"><rect width="30" height="60" fill="#39a" fill-opacity=".7"/></g><g filter="url(#f)"><rect x="65" y="57" width="20" height="30" fill="#eee" fill-opacity=".3"/></g></g><g fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="60" x2="131" y2="60" stroke="#182" stroke-width="21" stroke-opacity=".6"/><path d="M68 2L68 190" stroke="url(#g)" stroke-width=".31"/><path d="M0 64.13L139 127.57" stroke="#382" stroke-width="1.15"/></g></svg>"##;
        let cache = Cache::new(&Document::from((svg).to_string())).unwrap();
        for changed in [
            svg.to_owned(),
            svg.replace(" M10 90L20 90L20 100L10 100Z", ""),
            svg.replace("#2ad", "#bac"),
            svg.replace("M0 50L130 50L130 150L0 150Z", "M0 70L130 70L130 151L0 151Z"),
        ] {
            compare(&cache, &changed);
        }
        let scene = cache.scene(&Document::from((svg).to_string())).unwrap();
        assert!(scene.draws.iter().all(|draw| cache
            .trees
            .values()
            .any(|tree| Arc::ptr_eq(tree, &draw.tree))));
    }

    #[test]
    fn clip_references_keep_inherited_attributes_of_visible_strokes() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="139" height="193"><defs><clipPath id="coverage"><use xlink:href="#ink"/></clipPath></defs><g fill-rule="evenodd"><rect width="139" height="193" fill="#abc"/></g><g id="ink-layer" fill="none" stroke-linecap="round" stroke-linejoin="round"><path id="ink" d="M14 30L120 160L20 160" stroke="#123" stroke-width="13"/></g><g clip-path="url(#coverage)"><rect width="139" height="100" fill="#f20"/></g></svg>"##;
        let cache = Cache::new(&Document::from((svg).to_string())).unwrap();
        compare(&cache, svg);
        assert!(!cache.layers.lock().unwrap().entries.is_empty());
        compare(&cache, &svg.replace("L120 160", "L122 160"));
    }

    #[test]
    #[ignore = "reads an explicitly supplied emitted SVG"]
    fn emitted_core_fragments_match_full_renderer() {
        let path = std::env::var("PICVEC_FRAGMENT_SVG").unwrap();
        let svg = std::fs::read_to_string(path).unwrap();
        let cache = Cache::new(&Document::from((&svg).to_string()))
            .expect("emitted core should support cached parsing");
        let scene = cache.scene(&Document::from((&svg).to_string())).unwrap();
        let tree = Tree::from_str(&svg, &Options::default()).unwrap();
        let w = tree.size().width().ceil() as u32;
        let h = tree.size().height().ceil() as usize;
        for scale in [1usize, 4] {
            for y in [0, (h / 2 / 64) * 64, ((h - 1) / 64) * 64] {
                let mut expected =
                    Pixmap::new(w * scale as u32, (h - y).min(64) as u32 * scale as u32).unwrap();
                let mut actual = expected.clone();
                let t = std::time::Instant::now();
                resvg::render(
                    &tree,
                    Transform::from_row(
                        scale as f32,
                        0.0,
                        0.0,
                        scale as f32,
                        0.0,
                        -((y * scale) as f32),
                    ),
                    &mut expected.as_mut(),
                );
                let full = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                scene.render(scale, y, &mut actual);
                eprintln!(
                    "fragment render scale={scale} y={y}: full={full:.3}s cached={:.3}s",
                    t.elapsed().as_secs_f64()
                );
                assert_eq!(actual.data(), expected.data(), "scale={scale}, y={y}");
                let mut warmed = Pixmap::new(actual.width(), actual.height()).unwrap();
                let t = std::time::Instant::now();
                scene.render(scale, y, &mut warmed);
                eprintln!(
                    "fragment warm scale={scale} y={y}: {:.3}s",
                    t.elapsed().as_secs_f64()
                );
                assert_eq!(warmed.data(), expected.data(), "warm scale={scale}, y={y}");
            }
        }
    }

    #[test]
    fn path_only_trials_preserve_context_and_rebuild_referenced_ancestors() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="139" height="193"><defs><linearGradient id="g"><stop stop-color="#f23"/><stop offset="1" stop-color="#29a" stop-opacity=".3"/></linearGradient></defs><g fill-rule="evenodd"><path d="M0 0H139V193H0Z M10 10H20V20H10Z" fill="url(#g)"/><path d="M30 30H50V50H30Z" fill="#237"/></g></svg>"##;
        let cache = Cache::new(&Document::from((svg).to_string())).unwrap();
        let changed = svg.replace(" M10 10H20V20H10Z", "");
        assert!(cache.scene(&Document::from(changed.clone())).is_some());
        compare(&cache, &changed);
        for changed in [
            svg.replace("#f23", "#000"),
            svg.replace("evenodd", "nonzero"),
            svg.replace("fill=\"#237\"", "fill=\"#234\""),
        ] {
            assert!(cache
                .scene(&Document::from(changed.clone()))
                .unwrap()
                .draws
                .iter()
                .any(|draw| !cache
                    .trees
                    .values()
                    .any(|tree| Arc::ptr_eq(tree, &draw.tree))));
            compare(&cache, &changed);
        }
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="139" height="193"><g id="outline"><path d="M10 10H40V40H10Z" fill="#f23"/></g><use href="#outline" x="70"/></svg>"##;
        let cache = Cache::new(&Document::from((svg).to_string())).unwrap();
        let changed = svg.replace("H40", "H50");
        assert!(cache
            .scene(&Document::from(changed.clone()))
            .unwrap()
            .draws
            .iter()
            .any(|draw| !cache
                .trees
                .values()
                .any(|tree| Arc::ptr_eq(tree, &draw.tree))));
        compare(&cache, &changed);
    }

    #[test]
    fn unsupported_document_context_falls_back() {
        for body in [
            "<style>path { fill: red }</style><path d='M0 0L1 1L0 1Z'/>",
            "<svg width='5' height='5'><rect width='5' height='5'/></svg>",
            "<use href='#missing'/>",
        ] {
            let svg = format!(
                "<svg xmlns='http://www.w3.org/2000/svg' width='139' height='193'>{body}</svg>"
            );
            assert!(Cache::new(&Document::from((&svg).to_string())).is_none());
        }
    }
    #[test]
    #[ignore = "manual profiling of an emitted SVG; set PICVEC_FRAGMENT_SVG"]
    fn profile_scene_preparation() {
        use std::{hint::black_box, time::Instant};
        let svg = std::fs::read_to_string(std::env::var("PICVEC_FRAGMENT_SVG").unwrap()).unwrap();
        let doc = Document::from(svg);
        black_box(doc.root());
        for _ in 0..3 {
            let start = Instant::now();
            let extracted = sources(&doc).unwrap();
            eprintln!(
                "source preparation: {:.6}s, {} draws",
                start.elapsed().as_secs_f64(),
                black_box(&extracted).len()
            );
        }
        let cache = Cache::new(&doc).unwrap();
        for _ in 0..3 {
            let start = Instant::now();
            let scene = cache.scene(&doc).unwrap();
            eprintln!(
                "scene preparation: {:.6}s, {} draws",
                start.elapsed().as_secs_f64(),
                black_box(&scene).draws.len()
            );
        }
    }
}
