fn equivalent(before: &Document, after: &Document, completion: Option<&Completion<'_>>) -> bool {
    Baseline::new(before).is_some_and(|baseline| baseline.equivalent(after, completion))
}

pub(crate) fn simplify<F>(
    geometry: &mut [RegionGeometry],
    original: (Document, SvgSummary),
    labels: &[u32],
    width: usize,
    alpha: Option<&crate::chroma::AlphaMatte>,
    serialize: F,
) -> (Document, SvgSummary, usize)
where
    F: FnMut(&[RegionGeometry]) -> (Document, SvgSummary),
{
    simplify_cached(geometry, original, labels, width, alpha, None, serialize)
}

#[path = "../reference/occlusion_reference.rs"]
mod reference;

mod tests {
    #[test]
    fn order_cache_preserves_hole_checks_after_adoption_or_rejection() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="137" height="193"><defs><clipPath id="c"><path d="M1 1H135V191H1Z"/></clipPath><linearGradient id="p"><stop stop-color="#579"/><stop offset="1" stop-color="#ace" stop-opacity=".5"/></linearGradient></defs><g clip-path="url(#c)"><path fill="url(#p)" fill-rule="evenodd" d="M2 2H134V190H2Z M20 20H40V40H20Z"/><path d="M3 96H133" fill="none" stroke="#123" stroke-width=".3" stroke-opacity=".6"/></g></svg>"##;
        let original = Document::from(before.to_owned());
        let source = crate::raster::Raster::blank(137, 193, [0.5; 3]);
        let labels = vec![0; 137 * 193];
        for adopted in [before.to_owned(), before.replace("#579", "#975")] {
            let mut cache = None;
            crate::paint_order::validate_cached(
                &original,
                &original,
                &source,
                None,
                &labels,
                &mut crate::paint_order::Summary::default(),
                &mut cache,
            );
            assert!(cache.is_some());
            let document = Document::from(adopted.clone());
            let cached = Baseline::with_fragments(&document, cache).unwrap();
            let fresh = Baseline::new(&document).unwrap();
            for (trial, expected) in [
                (adopted.clone(), true),
                (adopted.replace(" M20 20H40V40H20Z", ""), false),
                (adopted.replace(".6", ".2"), false),
                (adopted.replace("H135", "H35"), false),
            ] {
                let trial = Document::from(trial);
                assert_eq!(fresh.equivalent(&trial, None), expected);
                assert_eq!(cached.equivalent(&trial, None), expected);
            }
        }
    }

    use super::*;
    #[test]
    fn reused_checks_match_full_final_render_for_partial_and_complete_acceptance() {
        for mode in 0..4 {
            let mut geometry: Vec<_> = (0..6)
                .map(|i| {
                    let y = i * 64;
                    // Keep removal profitable even when the adversarial
                    // serializer below adds an unrelated rectangle.
                    let hole = format!(
                        "M8 {}H24V{}H8{}Z",
                        y + 8,
                        y + 24,
                        format!("L8 {}", y + 24).repeat(12)
                    );
                    let path = format!("M0 {y}H96V{}H0Z {hole}", y + 64);
                    RegionGeometry {
                        region: i,
                        loops: vec![],
                        path_data: path.clone(),
                        occlusion_path_data: Some(path),
                        covered_hole_paths: vec![hole],
                        primitive: None,
                    }
                })
                .collect();
            let serialize = |g: &[RegionGeometry]| {
                let mut svg = String::from(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" width="96" height="384">"#,
                );
                for (i, face) in g.iter().enumerate() {
                    let alpha = if mode == 1 && i % 2 == 0 { 0.5 } else { 1.0 };
                    let width = if mode == 2 && i == 5 { 8 } else { 18 };
                    let y = i * 64;
                    svg.push_str(&format!(r##"<path fill="#fff" fill-rule="evenodd" d="{}"/><rect x="7" y="{}" width="{width}" height="18" fill="#123456" fill-opacity="{alpha}"/>"##,face.occlusion_path_data.as_ref().unwrap(), y+7));
                }
                // A later candidate changes a band checked by an earlier
                // accepted candidate. Accumulating old check bits is unsafe.
                if mode == 3
                    && !g[5]
                        .occlusion_path_data
                        .as_ref()
                        .unwrap()
                        .contains(&g[5].covered_hole_paths[0])
                {
                    svg.push_str(r##"<rect y="40" width="96" height="8" fill="#f00"/>"##);
                }
                svg.push_str("</svg>");
                (Document::from(svg), SvgSummary::default())
            };
            let original = serialize(&geometry);
            let mut expected_geometry = geometry.clone();
            let labels = vec![0; 96 * 384];
            let expected = reference::simplify(
                &mut expected_geometry,
                original.clone(),
                &labels,
                96,
                None,
                serialize,
            );
            let actual = simplify(&mut geometry, original, &labels, 96, None, serialize);
            assert_eq!(actual.0, expected.0, "mode={mode}");
            assert_eq!(actual.2, expected.2, "mode={mode}");
            assert_eq!(
                serde_json::to_string(&actual.1).unwrap(),
                serde_json::to_string(&expected.1).unwrap()
            );
            assert_eq!(
                geometry
                    .iter()
                    .map(|g| &g.occlusion_path_data)
                    .collect::<Vec<_>>(),
                expected_geometry
                    .iter()
                    .map(|g| &g.occlusion_path_data)
                    .collect::<Vec<_>>()
            );
            if mode == 0 {
                assert_eq!(actual.2, 6);
            }
            if mode == 1 {
                assert_eq!(actual.2, 3);
            }
            if mode == 2 {
                assert_eq!(actual.2, 5);
            }
            if mode == 3 {
                assert_eq!(actual.2, 0);
            }
        }
    }

    #[test]
    fn final_full_render_rolls_back_changes_outside_candidate_bounds() {
        let hole = "M8 8H24V24H8Z";
        let path = format!("M0 0H96V192H0Z {hole}");
        let mut geometry = vec![RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: path.clone(),
            occlusion_path_data: Some(path.clone()),
            covered_hole_paths: vec![hole.into()],
            primitive: None,
        }];
        let serialize = |g: &[RegionGeometry]| {
            let changed = !g[0].occlusion_path_data.as_ref().unwrap().contains(hole);
            // Deliberately simulate an unrelated serializer change outside the
            // candidate band. The final whole-image guard must catch it.
            let colour = if changed { "#f00" } else { "#000" };
            (
                format!(
                    r##"<svg xmlns="http://www.w3.org/2000/svg" width="96" height="192"><path fill="#fff" fill-rule="evenodd" d="{}"/><rect x="7" y="7" width="18" height="18" fill="#123456"/><rect y="160" width="96" height="32" fill="{colour}"/></svg>"##,
                    g[0].occlusion_path_data.as_ref().unwrap()
                ).into(),
                SvgSummary::default(),
            )
        };
        let original = serialize(&geometry);
        let (output, _, removed) = simplify(
            &mut geometry,
            original.clone(),
            &vec![0; 96 * 192],
            96,
            None,
            serialize,
        );
        assert_eq!(removed, 0);
        assert_eq!(output, Document::from(original.0));
        assert_eq!(
            geometry[0].occlusion_path_data.as_deref(),
            Some(path.as_str())
        );
    }

    #[test]
    fn hole_bounds_check_both_sides_of_a_render_band_boundary() {
        let hole = "M8 62H24V66H8Z";
        let before = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="96" height="192"><path fill="#fff" fill-rule="evenodd" d="M0 0H96V192H0Z {hole}"/><rect x="7" y="61" width="18" height="6" fill="#123456"/></svg>"##
        );
        let range = affected_rows(hole, 96, 192);
        assert_eq!(range, (60.0, 68.0));
        for candidate in [
            before.clone(),
            before.replace(hole, ""),
            before
                .replace(hole, "")
                .replace("height=\"6\"", "height=\"2\""),
        ] {
            let baseline = Baseline::new(&Document::from((&before).to_string())).unwrap();
            assert_eq!(
                baseline.equivalent_in(&Document::from((&candidate).to_string()), None, &[range]),
                baseline.equivalent(&Document::from((&candidate).to_string()), None)
            );
        }
        assert_eq!(
            affected_rows("m8 62h16v4z", 96, 192),
            (f32::NEG_INFINITY, f32::INFINITY)
        );
    }

    #[test]
    fn cached_baseline_does_not_accumulate_candidate_tolerances() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="96" height="80"><path fill="#000000" d="M0 0H96V80H0Z"/></svg>"##;
        let baseline = Baseline::new(&Document::from((before).to_string())).unwrap();
        assert!(baseline.equivalent(
            &Document::from((&before.replace("#000000", "#010000")).to_string()),
            None
        ));
        assert!(!baseline.equivalent(
            &Document::from((&before.replace("#000000", "#020000")).to_string()),
            None
        ));
        assert!(baseline.equivalent(&Document::from((before).to_string()), None));
    }

    #[test]
    fn boundary_completion_does_not_erase_authored_transparency() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path fill="#fff" fill-rule="evenodd" d="M0 0H32V32H0Z M8.2 8.2H23.8V23.8H8.2Z"/><rect x="8.2" y="8.2" width="15.6" height="15.6" fill="#123456"/></svg>"##;
        let after = before.replace(" M8.2 8.2H23.8V23.8H8.2Z", "");
        let mut boundary = vec![false; 32 * 32];
        for y in 7..25 {
            for x in 7..25 {
                boundary[y * 32 + x] = x <= 9 || x >= 22 || y <= 9 || y >= 22;
            }
        }
        let context = Completion {
            width: 32,
            boundary,
            alpha: None,
        };
        assert!(!equivalent(
            &Document::from((before).to_string()),
            &Document::from((&after).to_string()),
            None
        ));
        assert!(equivalent(
            &Document::from((before).to_string()),
            &Document::from((&after).to_string()),
            Some(&context)
        ));
        let matte = crate::chroma::AlphaMatte::new(32, 32, vec![0.5; 32 * 32]);
        let translucent = Completion {
            alpha: Some(&matte),
            ..context
        };
        assert!(!equivalent(
            &Document::from((before).to_string()),
            &Document::from((&after).to_string()),
            Some(&translucent)
        ));
    }

    #[test]
    fn only_opaque_cover_allows_a_hole_to_be_filled() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path fill="#fff" fill-rule="evenodd" d="M0 0H32V32H0Z M8 8H24V24H8Z"/><rect x="7" y="7" width="18" height="18" fill="#123456"/></svg>"##;
        let after = before.replace(" M8 8H24V24H8Z", "");
        assert!(equivalent(
            &Document::from((before).to_string()),
            &Document::from((&after).to_string()),
            None
        ));
        let translucent =
            before.replace("fill=\"#123456\"", "fill=\"#123456\" fill-opacity=\"0.5\"");
        assert!(!equivalent(
            &Document::from((&translucent).to_string()),
            &Document::from((&translucent.replace(" M8 8H24V24H8Z", "")).to_string()),
            None
        ));
        let uncovered = before.replace("width=\"18\"", "width=\"8\"");
        assert!(!equivalent(
            &Document::from((&uncovered).to_string()),
            &Document::from((&uncovered.replace(" M8 8H24V24H8Z", "")).to_string()),
            None
        ));
    }
}

impl Baseline {
    fn new(svg: &Document) -> Option<Self> {
        Self::with_fragments(svg, None)
    }

    fn equivalent(&self, after: &Document, completion: Option<&Completion<'_>>) -> bool {
        self.equivalent_in(after, completion, &[])
    }

    fn equivalent_in(
        &self,
        after: &Document,
        completion: Option<&Completion<'_>>,
        ranges: &[(f32, f32)],
    ) -> bool {
        self.equivalent_bands(after, completion, &self.selected_bands(ranges))
    }
}
