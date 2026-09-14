//! Fill only covered holes whose removal stays within one premultiplied RGBA
//! code value at native size and 4x, except for filling AA gaps along opaque
//! material boundaries. All accepted changes share one baseline,
//! so tolerances cannot accumulate across overlapping faces.
use crate::{geometry::RegionGeometry, svg::SvgSummary};
use rayon::prelude::*;
use resvg::{
    tiny_skia::{Pixmap, Transform},
    usvg::{Options, Tree},
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Completion<'a> {
    width: usize,
    boundary: Vec<bool>,
    alpha: Option<&'a crate::chroma::AlphaMatte>,
}

struct BaselineBand {
    scale: usize,
    y: usize,
    pixels: Pixmap,
}

struct Baseline {
    size: resvg::usvg::Size,
    bands: Vec<BaselineBand>,
    fragments: Option<crate::svg_fragments::Cache>,
    last_failure: AtomicUsize,
}

impl Baseline {
    fn new(svg: &str) -> Option<Self> {
        let tree = Tree::from_str(svg, &Options::default()).ok()?;
        let size = tree.size();
        let (w, h) = (size.width().ceil() as usize, size.height().ceil() as usize);
        let coordinates: Vec<_> = [1usize, 4]
            .into_iter()
            .flat_map(|scale| (0..h).step_by(64).map(move |y| (scale, y)))
            .collect();
        let bands = coordinates
            .into_par_iter()
            .map(|(scale, y)| {
                let mut pixels =
                    Pixmap::new((w * scale) as u32, ((h - y).min(64) * scale) as u32).unwrap();
                resvg::render(&tree, band_transform(scale, y), &mut pixels.as_mut());
                BaselineBand { scale, y, pixels }
            })
            .collect();
        Some(Self {
            size,
            bands,
            fragments: crate::svg_fragments::Cache::new(svg),
            last_failure: AtomicUsize::new(usize::MAX),
        })
    }

    #[cfg(test)]
    fn equivalent(&self, after: &str, completion: Option<&Completion<'_>>) -> bool {
        self.equivalent_in(after, completion, &[])
    }

    fn selected_bands(&self, ranges: &[(f32, f32)]) -> Vec<bool> {
        self.bands
            .iter()
            .map(|band| {
                ranges.is_empty()
                    || ranges.iter().any(|&(top, bottom)| {
                        (band.y as f32) < bottom && ((band.y + 64) as f32) > top
                    })
            })
            .collect()
    }

    #[cfg(test)]
    fn equivalent_in(
        &self,
        after: &str,
        completion: Option<&Completion<'_>>,
        ranges: &[(f32, f32)],
    ) -> bool {
        self.equivalent_bands(after, completion, &self.selected_bands(ranges))
    }

    fn equivalent_bands(
        &self,
        after: &str,
        completion: Option<&Completion<'_>>,
        selected: &[bool],
    ) -> bool {
        self.equivalent_bands_impl(after, completion, selected, true)
    }

    fn equivalent_bands_impl(
        &self,
        after: &str,
        completion: Option<&Completion<'_>>,
        selected: &[bool],
        accelerate: bool,
    ) -> bool {
        self.last_failure.store(usize::MAX, Ordering::Relaxed);
        assert_eq!(selected.len(), self.bands.len());
        let scene = if accelerate {
            self.fragments.as_ref().and_then(|cache| cache.scene(after))
        } else {
            None
        };
        let tree = if scene.is_none() {
            let Ok(tree) = Tree::from_str(after, &Options::default()) else {
                return false;
            };
            if self.size != tree.size() {
                return false;
            }
            Some(tree)
        } else {
            None
        };
        // Every candidate uses the same original pixels; accepted changes never
        // become the baseline, so neither rounding nor tolerances accumulate.
        // Do not start expensive 4x jobs while native pixels can already
        // reject a trial. Both scales still use the same complete RGBA gates.
        let check = |band_index: usize| {
            if !selected[band_index] {
                return true;
            }
            let band = &self.bands[band_index];
            let passed = (|| {
                let p = &band.pixels;
                let tw = p.width() as usize;
                let mut q = Pixmap::new(p.width(), p.height()).unwrap();
                if let Some(scene) = &scene {
                    scene.render(band.scale, band.y, &mut q);
                } else {
                    resvg::render(
                        tree.as_ref().unwrap(),
                        band_transform(band.scale, band.y),
                        &mut q.as_mut(),
                    );
                }
                for (k, (p, q)) in p
                    .data()
                    .chunks_exact(4)
                    .zip(q.data().chunks_exact(4))
                    .enumerate()
                {
                    if p.iter().zip(q).all(|(&p, &q)| p.abs_diff(q) <= 1) {
                        continue;
                    }
                    let Some(context) = completion else {
                        return false;
                    };
                    let i = (band.y + k / tw / band.scale) * context.width + k % tw / band.scale;
                    let target_alpha = context.alpha.map_or(1.0, |a| a.get(i));
                    let added_alpha = q[3] as i16 - p[3] as i16;
                    // Boundary-only completion: retain the old premultiplied
                    // foreground and add at most the formerly uncovered alpha.
                    // This is not permission to repaint an opaque pixel.
                    if !context.boundary[i]
                        || target_alpha < 254.0 / 255.0 - 1e-6
                        || added_alpha < 0
                        || q[3] as f32 > target_alpha * 255.0 + 1.01
                        || (0..3).any(|c| {
                            let delta = q[c] as i16 - p[c] as i16;
                            delta < -1 || delta > added_alpha + 1
                        })
                    {
                        return false;
                    }
                }
                true
            })();
            if !passed {
                let _ = self.last_failure.compare_exchange(
                    usize::MAX,
                    band_index,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
            passed
        };
        [1usize, 4].into_iter().all(|scale| {
            self.bands
                .par_iter()
                .enumerate()
                .filter(|(_, band)| band.scale == scale)
                .all(|(i, _)| check(i))
        })
    }
}

fn band_transform(scale: usize, y: usize) -> Transform {
    Transform::from_row(
        scale as f32,
        0.0,
        0.0,
        scale as f32,
        0.0,
        -((y * scale) as f32),
    )
}

#[cfg(test)]
fn equivalent(before: &str, after: &str, completion: Option<&Completion<'_>>) -> bool {
    Baseline::new(before).is_some_and(|baseline| baseline.equivalent(after, completion))
}

// These are standalone, absolute, closed contours produced by geometry.
// Removing a closed subpath can change winding only inside its bounds. Keep
// two native pixels around it for antialiasing at both validation scales.
// Unrecognised contours conservatively require all bands.
fn affected_rows(path: &str, width: usize, height: usize) -> (f32, f32) {
    let full = (f32::NEG_INFINITY, f32::INFINITY);
    if !path.trim_start().starts_with('M')
        || !path.trim_end().ends_with('Z')
        || path.chars().any(|c| c.is_ascii_lowercase())
    {
        return full;
    }
    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><path d="{path}"/></svg>"#
    );
    let Ok(tree) = Tree::from_str(&svg, &Options::default()) else {
        return full;
    };
    let Some(node) = tree.root().children().first() else {
        return full;
    };
    let bounds = node.abs_bounding_box();
    (bounds.top() - 2.0, bounds.bottom() + 2.0)
}

/// Bisection often retries an ancestor's rejected geometry after accepting its
/// left half. Keep exact path strings, not probabilistic hashes. A cached bad
/// pixel is reusable only if its band is selected by the new trial too.
#[derive(Default)]
struct RejectedTrials {
    entries: std::collections::VecDeque<(Vec<String>, usize, usize)>,
    bytes: usize,
    hits: usize,
}

impl RejectedTrials {
    fn contains(&mut self, paths: &[&str], selected: &[bool]) -> bool {
        let found = self.entries.iter().position(|(key, band, _)| {
            selected[*band] && key.iter().map(String::as_str).eq(paths.iter().copied())
        });
        if let Some(index) = found {
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
            self.hits += 1;
            true
        } else {
            false
        }
    }

    fn insert(&mut self, paths: &[&str], band: usize) {
        const LIMIT: usize = 64 * 1024 * 1024;
        let bytes = paths
            .iter()
            .map(|p| p.len() + std::mem::size_of::<String>())
            .sum::<usize>();
        if band == usize::MAX || bytes > LIMIT {
            return;
        }
        while !self.entries.is_empty() && (self.entries.len() >= 8 || self.bytes + bytes > LIMIT) {
            self.bytes -= self.entries.pop_front().unwrap().2;
        }
        self.entries
            .push_back((paths.iter().map(|p| p.to_string()).collect(), band, bytes));
        self.bytes += bytes;
    }
}

pub(crate) fn simplify<F>(
    geometry: &mut [RegionGeometry],
    original: (String, SvgSummary),
    labels: &[u32],
    width: usize,
    alpha: Option<&crate::chroma::AlphaMatte>,
    mut serialize: F,
) -> (String, SvgSummary, usize)
where
    F: FnMut(&[RegionGeometry]) -> (String, SvgSummary),
{
    let height = labels.len() / width;
    let candidates: Vec<_> = geometry
        .iter()
        .enumerate()
        .flat_map(|(i, g)| {
            g.covered_hole_paths
                .iter()
                .map(move |path| (i, path.clone(), affected_rows(path, width, height)))
        })
        .collect();
    if candidates.is_empty() {
        return (original.0, original.1, 0);
    }
    let Some(baseline) = Baseline::new(&original.0) else {
        return (original.0, original.1, 0);
    };
    let mut boundary = vec![false; labels.len()];
    for i in 0..labels.len() {
        let (x, y) = (i % width, i / width);
        boundary[i] = [
            (x > 0).then(|| i - 1),
            (x + 1 < width).then(|| i + 1),
            (y > 0).then(|| i - width),
            (y + 1 < height).then(|| i + width),
        ]
        .into_iter()
        .flatten()
        .any(|j| labels[j] != labels[i]);
    }
    let completion = Completion {
        width,
        boundary: crate::edge::dilate_square(&boundary, width, height, 1),
        alpha,
    };
    let saved_paths: Vec<_> = geometry
        .iter()
        .map(|g| g.occlusion_path_data.clone())
        .collect();
    let mut current = original.clone();
    let mut removed = 0;
    let mut watched: Vec<_> = candidates.iter().map(|item| item.0).collect();
    watched.sort_unstable();
    watched.dedup();
    let mut rejections = RejectedTrials::default();
    let mut timings = (0usize, std::time::Duration::ZERO, std::time::Duration::ZERO);
    fn visit<F>(
        items: &[(usize, String, (f32, f32))],
        geometry: &mut [RegionGeometry],
        baseline: &Baseline,
        completion: &Completion<'_>,
        current: &mut (String, SvgSummary),
        removed: &mut usize,
        watched: &[usize],
        rejections: &mut RejectedTrials,
        timings: &mut (usize, std::time::Duration, std::time::Duration),
        serialize: &mut F,
    ) where
        F: FnMut(&[RegionGeometry]) -> (String, SvgSummary),
    {
        if items.is_empty() {
            return;
        }
        let mut backup = std::collections::BTreeMap::new();
        for (i, path, _) in items {
            backup
                .entry(*i)
                .or_insert_with(|| geometry[*i].occlusion_path_data.clone());
            let data = geometry[*i]
                .occlusion_path_data
                .as_mut()
                .expect("covered hole has overlap geometry");
            *data = data.replacen(path, "", 1);
        }
        let ranges: Vec<_> = items.iter().map(|item| item.2).collect();
        let selected = baseline.selected_bands(&ranges);
        let paths: Vec<_> = watched
            .iter()
            .map(|&i| geometry[i].occlusion_path_data.as_deref().unwrap())
            .collect();
        timings.0 += 1;
        if !rejections.contains(&paths, &selected) {
            let started = std::time::Instant::now();
            let candidate = serialize(geometry);
            timings.1 += started.elapsed();
            let started = std::time::Instant::now();
            let profitable = candidate.0.len() < current.0.len();
            let accepted =
                profitable && baseline.equivalent_bands(&candidate.0, Some(completion), &selected);
            timings.2 += started.elapsed();
            if accepted {
                *current = candidate;
                *removed += items.len();
                return;
            }
            if profitable {
                rejections.insert(&paths, baseline.last_failure.load(Ordering::Relaxed));
            }
        }
        for (i, data) in backup {
            geometry[i].occlusion_path_data = data;
        }
        if items.len() > 1 {
            let middle = items.len() / 2;
            visit(
                &items[..middle],
                geometry,
                baseline,
                completion,
                current,
                removed,
                watched,
                rejections,
                timings,
                serialize,
            );
            visit(
                &items[middle..],
                geometry,
                baseline,
                completion,
                current,
                removed,
                watched,
                rejections,
                timings,
                serialize,
            );
        }
    }
    visit(
        &candidates,
        geometry,
        &baseline,
        &completion,
        &mut current,
        &mut removed,
        &watched,
        &mut rejections,
        &mut timings,
        &mut serialize,
    );
    if cfg!(feature = "diagnostics") && std::env::var_os("PICVEC_OCCLUSION_DIAGNOSTICS").is_some() {
        eprintln!(
            "picvec hole trials: {} calls, {} cached rejections, serialization {:.3}s, validation {:.3}s",
            timings.0, rejections.hits,
            timings.1.as_secs_f64(),
            timings.2.as_secs_f64()
        );
    }
    // Independently validate the complete accumulated SVG with the original
    // parser and renderer at both scales, even bands checked through cached
    // fragments. Trial caches never become the final authority.
    if removed > 0
        && !baseline.equivalent_bands_impl(
            &current.0,
            Some(&completion),
            &vec![true; baseline.bands.len()],
            false,
        )
    {
        for (g, saved) in geometry.iter_mut().zip(saved_paths) {
            g.occlusion_path_data = saved;
        }
        return (original.0, original.1, 0);
    }
    (current.0, current.1, removed)
}

#[cfg(test)]
#[path = "occlusion_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
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
                (svg, SvgSummary::default())
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
                ),
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
        assert_eq!(output, original.0);
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
            let baseline = Baseline::new(&before).unwrap();
            assert_eq!(
                baseline.equivalent_in(&candidate, None, &[range]),
                baseline.equivalent(&candidate, None)
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
        let baseline = Baseline::new(before).unwrap();
        assert!(baseline.equivalent(&before.replace("#000000", "#010000"), None));
        assert!(!baseline.equivalent(&before.replace("#000000", "#020000"), None));
        assert!(baseline.equivalent(before, None));
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
        assert!(!equivalent(before, &after, None));
        assert!(equivalent(before, &after, Some(&context)));
        let matte = crate::chroma::AlphaMatte::new(32, 32, vec![0.5; 32 * 32]);
        let translucent = Completion {
            alpha: Some(&matte),
            ..context
        };
        assert!(!equivalent(before, &after, Some(&translucent)));
    }

    #[test]
    fn only_opaque_cover_allows_a_hole_to_be_filled() {
        let before = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path fill="#fff" fill-rule="evenodd" d="M0 0H32V32H0Z M8 8H24V24H8Z"/><rect x="7" y="7" width="18" height="18" fill="#123456"/></svg>"##;
        let after = before.replace(" M8 8H24V24H8Z", "");
        assert!(equivalent(before, &after, None));
        let translucent =
            before.replace("fill=\"#123456\"", "fill=\"#123456\" fill-opacity=\"0.5\"");
        assert!(!equivalent(
            &translucent,
            &translucent.replace(" M8 8H24V24H8Z", ""),
            None
        ));
        let uncovered = before.replace("width=\"18\"", "width=\"8\"");
        assert!(!equivalent(
            &uncovered,
            &uncovered.replace(" M8 8H24V24H8Z", ""),
            None
        ));
    }
}
