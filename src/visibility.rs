//! Remove draw operations only when their omission leaves the final RGBA
//! raster unchanged at both source scale and 4x. Work on spatial tiles, parse
//! once, and stop at the first changed tile rather than rerendering the image
//! once per object. Partial occlusion is deliberately left alone.
use resvg::{
    tiny_skia::{Pixmap, Transform},
    usvg::{self, Node},
};
use std::collections::HashMap;

const TILE: usize = 64;
const PREFIX: &str = "picvec-visibility-";
// Each entry owns a baseline and a prefix framebuffer. Together they use at
// most the previous cache's 128 pixmaps (32 MiB at 4x), plus one scratch tile.
const CACHE_TILES: usize = 128 / 2;

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Removed {
    pub shapes: usize,
    pub strokes: usize,
    pub ink: usize,
    pub paths: usize,
    pub rects: usize,
    pub circles: usize,
    pub ellipses: usize,
    pub lines: usize,
}

struct Candidate {
    start: usize,
    end: usize,
    stroke: bool,
    ink: bool,
}

// This is intentionally a scanner for our own serializer's self-closing
// geometry, not a general-purpose XML editor. Definitions/referenced objects
// and complex compositing groups are never independently removed.
fn annotate(document: &str) -> (String, Vec<Candidate>) {
    let mut result = String::new();
    let mut candidates = Vec::new();
    let mut cursor = 0;
    let mut definitions = false;
    while let Some(offset) = document[cursor..].find('<') {
        let start = cursor + offset;
        let Some(end) = document[start..].find('>').map(|n| start + n + 1) else {
            break;
        };
        let tag = &document[start..end];
        result.push_str(&document[cursor..start]);
        if tag.starts_with("<defs") {
            definitions = true;
        }
        let geometry = [
            "<path ",
            "<rect ",
            "<circle ",
            "<ellipse ",
            "<line ",
            "<polygon ",
            "<polyline ",
        ]
        .iter()
        .any(|p| tag.starts_with(p));
        if !definitions && geometry && tag.ends_with("/>") && !tag.contains(" id=") {
            let id = candidates.len();
            result.push_str(&format!("<g id=\"{PREFIX}{id}\">{tag}</g>"));
            candidates.push(Candidate {
                start,
                end,
                stroke: tag.contains("fill=\"none\""),
                ink: tag.contains("data-structural-ink="),
            });
        } else {
            result.push_str(tag);
        }
        if tag.starts_with("</defs") {
            definitions = false;
        }
        cursor = end;
    }
    result.push_str(&document[cursor..]);
    (result, candidates)
}

struct Draw<'a> {
    node: &'a Node,
    candidate: Option<usize>,
    bounds: [f32; 4],
}

// Match Tree::node_by_id's search scope, including nodes inside isolated
// groups, without restarting a full tree traversal for every candidate.
fn missing_candidates(group: &usvg::Group, count: usize) -> Vec<bool> {
    fn visit(group: &usvg::Group, missing: &mut [bool]) {
        for node in group.children() {
            if let Some(suffix) = node.id().strip_prefix(PREFIX) {
                if let Ok(id) = suffix.parse::<usize>() {
                    // Existing noncanonical IDs such as "01" or "+1" must
                    // not count as the generated ID "1".
                    if suffix == id.to_string() {
                        if let Some(missing) = missing.get_mut(id) {
                            *missing = false;
                        }
                    }
                }
            }
            if let Node::Group(group) = node {
                visit(group, missing);
            }
        }
    }
    let mut missing = vec![true; count];
    visit(group, &mut missing);
    missing
}

fn collect<'a>(group: &'a usvg::Group, draws: &mut Vec<Draw<'a>>) {
    for node in group.children() {
        let candidate = node.id().strip_prefix(PREFIX).and_then(|s| s.parse().ok());
        if let Node::Group(g) = node {
            if candidate.is_none() && g.transform().is_identity() && !g.should_isolate() {
                collect(g, draws);
                continue;
            }
        }
        let Some(b) = node.abs_layer_bounding_box() else {
            continue;
        };
        draws.push(Draw {
            node,
            candidate,
            bounds: [
                b.left() - 1.0,
                b.top() - 1.0,
                b.right() + 1.0,
                b.bottom() + 1.0,
            ],
        });
    }
}

#[derive(Clone, Copy)]
struct TileView {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    scale: usize,
}

impl TileView {
    fn blank(self) -> Pixmap {
        Pixmap::new((self.w * self.scale) as u32, (self.h * self.scale) as u32).unwrap()
    }

    fn render(
        self,
        draws: &[Draw<'_>],
        members: &[usize],
        active: &[bool],
        skip: Option<usize>,
        pixels: &mut Pixmap,
    ) {
        for &i in members {
            if !active[i] || Some(i) == skip {
                continue;
            }
            let node = draws[i].node;
            let b = node.abs_layer_bounding_box().unwrap();
            let transform = Transform::from_scale(self.scale as f32, self.scale as f32)
                .pre_translate(b.x() - self.x as f32, b.y() - self.y as f32);
            resvg::render_node(node, transform, &mut pixels.as_mut());
        }
    }
}

struct TileRaster {
    baseline: Pixmap,
    prefix: Pixmap,
    // First member not yet included in prefix. Candidates are visited in
    // increasing draw order, so accepted removals never invalidate a prefix.
    next: usize,
}

impl TileRaster {
    fn new(
        view: TileView,
        draws: &[Draw<'_>],
        members: &[usize],
        active: &[bool],
        candidate: usize,
    ) -> Self {
        let next = members.partition_point(|&i| i < candidate);
        let mut prefix = view.blank();
        view.render(draws, &members[..next], active, None, &mut prefix);
        let mut baseline = prefix.clone();
        view.render(draws, &members[next..], active, None, &mut baseline);
        Self {
            baseline,
            prefix,
            next,
        }
    }

    fn unchanged_without(
        &mut self,
        view: TileView,
        draws: &[Draw<'_>],
        members: &[usize],
        active: &[bool],
        candidate: usize,
    ) -> bool {
        let end = self.next + members[self.next..].partition_point(|&i| i < candidate);
        view.render(
            draws,
            &members[self.next..end],
            active,
            None,
            &mut self.prefix,
        );
        self.next = end;
        // Clone the exact framebuffer, then execute the original suffix.
        // Compositing a separately rendered suffix would change alpha rounding.
        let mut after = self.prefix.clone();
        view.render(draws, &members[end..], active, Some(candidate), &mut after);
        self.baseline.data() == after.data()
    }
}

struct CacheEntry {
    raster: TileRaster,
    last_used: u64,
}

struct TileCache {
    entries: HashMap<(usize, usize), CacheEntry>,
    capacity: usize,
    clock: u64,
}

impl TileCache {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0 && capacity <= CACHE_TILES);
        Self {
            entries: HashMap::new(),
            capacity,
            clock: 0,
        }
    }

    fn get_or_insert(
        &mut self,
        key: (usize, usize),
        create: impl FnOnce() -> TileRaster,
    ) -> &mut TileRaster {
        self.clock += 1;
        if !self.entries.contains_key(&key) {
            if self.entries.len() == self.capacity {
                // At most 64 entries: a bounded scan on misses avoids an
                // unbounded access-history queue or an additional LRU index.
                let oldest = *self
                    .entries
                    .iter()
                    .min_by_key(|(_, v)| v.last_used)
                    .unwrap()
                    .0;
                // Free both pixmaps before allocating their replacements.
                self.entries.remove(&oldest);
            }
            self.entries.insert(
                key,
                CacheEntry {
                    raster: create(),
                    last_used: self.clock,
                },
            );
        }
        let entry = self.entries.get_mut(&key).unwrap();
        entry.last_used = self.clock;
        &mut entry.raster
    }
}

pub(crate) fn prune(document: &str, width: usize, height: usize) -> (String, Removed) {
    prune_cached(document, width, height, CACHE_TILES)
}

fn prune_cached(document: &str, width: usize, height: usize, capacity: usize) -> (String, Removed) {
    // A <use> can render the same source element again through a filter or
    // transform. Removing its source would alter both instances; do not treat
    // those instances as independently removable draw operations.
    if document.contains("<use ") {
        return (document.into(), Removed::default());
    }
    let (annotated, candidates) = annotate(document);
    if candidates.is_empty() {
        return (document.into(), Removed::default());
    }
    let Ok(tree) = usvg::Tree::from_str(&annotated, &usvg::Options::default()) else {
        return (document.into(), Removed::default());
    };
    // The core serializer has no viewport transform. Leave other documents
    // intact rather than detaching children from their inherited transform.
    if !tree.root().transform().is_identity() {
        return (document.into(), Removed::default());
    }
    let mut draws = Vec::new();
    collect(tree.root(), &mut draws);
    let columns = width.div_ceil(TILE);
    let rows = height.div_ceil(TILE);
    let mut tiles = vec![Vec::new(); columns * rows];
    let mut memberships = vec![Vec::new(); draws.len()];
    for (i, draw) in draws.iter().enumerate() {
        let [left, top, right, bottom] = draw.bounds;
        let x0 = (left.max(0.0) as usize / TILE).min(columns);
        let y0 = (top.max(0.0) as usize / TILE).min(rows);
        let x1 = ((right.max(0.0).ceil() as usize).div_ceil(TILE)).min(columns);
        let y1 = ((bottom.max(0.0).ceil() as usize).div_ceil(TILE)).min(rows);
        for y in y0..y1 {
            for x in x0..x1 {
                let tile = y * columns + x;
                tiles[tile].push(i);
                memberships[i].push(tile);
            }
        }
    }
    let mut active = vec![true; draws.len()];
    let mut removed = missing_candidates(tree.root(), candidates.len());
    let mut cache = TileCache::new(capacity);
    for (i, draw) in draws.iter().enumerate() {
        let Some(candidate) = draw.candidate else {
            continue;
        };
        let mut unchanged = true;
        'scales: for scale in [1, 4] {
            for &tile in &memberships[i] {
                let x = tile % columns * TILE;
                let y = tile / columns * TILE;
                let w = TILE.min(width - x);
                let h = TILE.min(height - y);
                let view = TileView { x, y, w, h, scale };
                let raster = cache.get_or_insert((tile, scale), || {
                    TileRaster::new(view, &draws, &tiles[tile], &active, i)
                });
                if !raster.unchanged_without(view, &draws, &tiles[tile], &active, i) {
                    unchanged = false;
                    break 'scales;
                }
            }
        }
        if unchanged {
            active[i] = false;
            removed[candidate] = true;
        }
    }
    let mut output = String::new();
    let mut cursor = 0;
    let mut report = Removed::default();
    for (candidate, remove) in candidates.iter().zip(removed) {
        if !remove {
            continue;
        }
        output.push_str(&document[cursor..candidate.start]);
        cursor = candidate.end;
        let tag = &document[candidate.start..candidate.end];
        report.paths += usize::from(tag.starts_with("<path "));
        report.rects += usize::from(tag.starts_with("<rect "));
        report.circles += usize::from(tag.starts_with("<circle "));
        report.ellipses += usize::from(tag.starts_with("<ellipse "));
        report.lines += usize::from(tag.starts_with("<line "));
        report.shapes += 1;
        report.strokes += usize::from(candidate.stroke);
        report.ink += usize::from(candidate.ink);
    }
    output.push_str(&document[cursor..]);
    (output, report)
}

#[cfg(test)]
#[path = "visibility_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;

    fn layered_scene() -> String {
        let mut svg = String::from(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80"><defs><linearGradient id="g"><stop stop-color="#315a91" stop-opacity="0.3"/><stop offset="1" stop-color="#ef9021" stop-opacity="0.8"/></linearGradient><clipPath id="c"><circle cx="63.5" cy="35" r="29"/></clipPath></defs><rect id="fixed" width="160" height="80" fill="#254a81"/><rect x="3" y="3" width="12" height="12" fill="red"/><rect x="2" y="2" width="20" height="20" fill="#254a81"/><rect x="2" y="2" width="20" height="20" fill="#254a81"/><g clip-path="url(#c)"><rect x="10" y="4" width="110" height="70" fill="url(#g)"/></g>"##,
        );
        for i in 0..36 {
            let x = (i * 17 % 150) as f32 + 0.13;
            let y = (i * 13 % 68) as f32 + 0.37;
            svg.push_str(&format!(r##"<rect x="{x}" y="{y}" width="19.8" height="10.6" fill="#985c31" fill-opacity="0.17"/><path d="M{x} {y}l14 6" fill="none" stroke="#1246ab" stroke-opacity="0.31" stroke-width="0.13"/>"##));
        }
        svg.push_str("</svg>");
        svg
    }

    #[test]
    fn indexed_presence_matches_recursive_lookup_even_inside_isolated_groups() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><defs><clipPath id="c"><circle cx="32" cy="32" r="20"/></clipPath></defs><rect id="picvec-visibility-0" width="10" height="10"/><g clip-path="url(#c)"><rect id="picvec-visibility-1" width="64" height="64"/></g><path id="picvec-visibility-2" d="M0 0H20"/><rect id="picvec-visibility-03" x="3" width="5" height="5"/><rect id="picvec-visibility-+4" x="4" width="5" height="5"/><rect id="picvec-visibility-99" width="5" height="5"/></svg>"##;
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let indexed = missing_candidates(tree.root(), 8);
        let scanned: Vec<_> = (0..8)
            .map(|i| tree.node_by_id(&format!("{PREFIX}{i}")).is_none())
            .collect();
        assert_eq!(indexed, scanned);
        assert!(!indexed[0] && !indexed[1]);
        assert!(indexed[3] && indexed[4]);
    }

    #[test]
    fn incremental_framebuffers_match_full_redraw_at_every_candidate() {
        let (svg, _) = annotate(&layered_scene());
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
        let mut draws = Vec::new();
        collect(tree.root(), &mut draws);
        let members: Vec<_> = (0..draws.len()).collect();
        for scale in [1, 4, 8] {
            for (x, y, w, h) in [(0, 0, 64, 64), (64, 0, 64, 64), (128, 64, 32, 16)] {
                let view = TileView { x, y, w, h, scale };
                let mut active = vec![true; draws.len()];
                let mut raster = TileRaster::new(view, &draws, &members, &active, 0);
                for i in 0..draws.len() {
                    let before =
                        reference::draw_tile(&draws, &members, &active, None, x, y, w, h, scale);
                    let after =
                        reference::draw_tile(&draws, &members, &active, Some(i), x, y, w, h, scale);
                    assert_eq!(raster.baseline.data(), before.data());
                    let unchanged = raster.unchanged_without(view, &draws, &members, &active, i);
                    assert_eq!(unchanged, before.data() == after.data());
                    let prefix = reference::draw_tile(
                        &draws,
                        &members[..i],
                        &active,
                        None,
                        x,
                        y,
                        w,
                        h,
                        scale,
                    );
                    assert_eq!(raster.prefix.data(), prefix.data());
                    // Include consecutive removals, then advance through only
                    // the retained nodes on the next visit to this tile.
                    if unchanged {
                        active[i] = false;
                    }
                }
            }
        }
    }

    #[test]
    fn lru_evicts_only_the_cold_entry_and_stays_within_the_pixmap_budget() {
        let mut cache = TileCache::new(3);
        let create = || TileRaster {
            baseline: Pixmap::new(256, 256).unwrap(),
            prefix: Pixmap::new(256, 256).unwrap(),
            next: 0,
        };
        for i in 0..3 {
            cache.get_or_insert((i, 4), create);
        }
        cache.get_or_insert((0, 4), || panic!("hot entry was lost"));
        cache.get_or_insert((3, 4), create);
        assert_eq!(cache.entries.len(), 3);
        assert!(!cache.entries.contains_key(&(1, 4)));
        for key in [(0, 4), (2, 4), (3, 4)] {
            cache.get_or_insert(key, || panic!("unrelated entry was evicted"));
        }
        for i in 4..140 {
            cache.get_or_insert((i, 4), create);
        }
        let bytes: usize = cache
            .entries
            .values()
            .map(|v| v.raster.baseline.data().len() + v.raster.prefix.data().len())
            .sum();
        assert_eq!(bytes, cache.capacity * 2 * 256 * 256 * 4);
        assert!(CACHE_TILES * 2 * 256 * 256 * 4 <= 32 * 1024 * 1024);
    }

    #[test]
    fn eviction_and_transparency_preserve_the_reference_removal_decisions() {
        let width = (CACHE_TILES + 5) * TILE;
        let mut wide = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="72"><rect id="fixed" width="{width}" height="72" fill="#125641"/>"##
        );
        // Cross the cache limit, then return to tiles with retained and
        // removed candidates at both scales, including partial edge tiles.
        for pass in 0..3 {
            for i in 0..CACHE_TILES + 5 {
                let x = i * TILE + 1;
                let opacity = if pass == 2 { 0.2 } else { 1.0 };
                if pass == 0 {
                    // Fully covered interior mark; the fractional outer rims
                    // themselves must stay because their AA accumulates.
                    wide.push_str(&format!(
                        r##"<rect x="{}" y="10" width="4" height="4" fill="#ff0077"/>"##,
                        x + 5
                    ));
                }
                wide.push_str(&format!(r##"<rect x="{x}" y="3.25" width="66" height="64.5" fill="#192581" fill-opacity="{opacity}"/>"##));
            }
        }
        wide.push_str("</svg>");
        for (svg, w, h) in [(layered_scene(), 160, 80), (wide, width, 72)] {
            let expected = reference::prune(&svg, w, h);
            assert!(expected.1.shapes > 0, "fixture {w}x{h} needs covered marks");
            for capacity in [1, 3, CACHE_TILES] {
                let actual = prune_cached(&svg, w, h, capacity);
                assert_eq!(actual, expected);
            }
            for scale in [1, 4] {
                assert_eq!(image(&svg, scale), image(&expected.0, scale));
            }
        }
    }

    #[test]
    #[ignore = "stage benchmark; set PICVEC_VISIBILITY_BENCH_INPUTS and PICVEC_VISIBILITY_BENCH_OUTPUT"]
    fn benchmark_emitted_svgs_against_full_redraw_reference() {
        let paths = std::env::var_os("PICVEC_VISIBILITY_BENCH_INPUTS").unwrap();
        let output = std::env::var_os("PICVEC_VISIBILITY_BENCH_OUTPUT").unwrap();
        let repeats = std::env::var("PICVEC_VISIBILITY_BENCH_REPEATS")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(repeats > 0);
        let mut rows = Vec::new();
        for path in std::env::split_paths(&paths) {
            let svg = std::fs::read_to_string(&path).unwrap();
            let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
            let (w, h) = (
                tree.size().width().ceil() as usize,
                tree.size().height().ceil() as usize,
            );
            drop(tree);
            let measure = |reference_version| {
                let start = std::time::Instant::now();
                let result = if reference_version {
                    reference::prune(&svg, w, h)
                } else {
                    prune(&svg, w, h)
                };
                (result, start.elapsed().as_secs_f64())
            };
            for repeat in 0..repeats {
                let (baseline, candidate) = if repeat % 2 == 0 {
                    (measure(true), measure(false))
                } else {
                    let candidate = measure(false);
                    (measure(true), candidate)
                };
                assert_eq!(baseline.0, candidate.0);
                eprintln!(
                    "visibility {} trial {}: {:.3}s -> {:.3}s",
                    path.display(),
                    repeat + 1,
                    baseline.1,
                    candidate.1
                );
                rows.push(serde_json::json!({"input": path, "repeat": repeat, "baseline_seconds": baseline.1, "candidate_seconds": candidate.1, "input_bytes": svg.len(), "output_bytes": candidate.0.0.len(), "removed_shapes": candidate.0.1.shapes, "outputs_equal": true}));
                std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
            }
        }
    }

    fn image(svg: &str, scale: u32) -> Vec<u8> {
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let mut p = Pixmap::new(
            tree.size().width() as u32 * scale,
            tree.size().height() as u32 * scale,
        )
        .unwrap();
        resvg::render(
            &tree,
            Transform::from_scale(scale as f32, scale as f32),
            &mut p.as_mut(),
        );
        p.take()
    }
    #[test]
    fn removes_hidden_faces_and_redundant_lines_but_keeps_partial_overlap() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><g fill="red"><rect x="5" y="5" width="5" height="5"/><rect width="20" height="20"/><path d="M5 5L15 15" fill="none" stroke="red" stroke-width="2"/><rect x="15" y="15" width="10" height="10" fill="blue"/></g></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 2);
        assert_eq!(report.strokes, 1);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn tile_boundaries_and_multiple_removals_preserve_the_complete_render() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="32"><rect width="256" height="32" fill="red"/><rect x="60.2" y="4" width="120" height="16" fill="red"/><path d="M10 15.3H200" fill="none" stroke="red" stroke-width="2"/><rect x="128" y="16" width="80" height="16" fill="blue"/></svg>"##;
        let (output, report) = prune(svg, 256, 32);
        assert_eq!(report.shapes, 2);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn zero_area_paint_is_removed_but_its_stroked_counterpart_is_kept() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M2 2H20" fill="red"/><path d="M2 2H20" fill="none" stroke="red"/></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 1);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
    #[test]
    fn preserves_alpha_accumulation_gradients_and_subpixel_lines() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect x="10" width="10" height="25" fill="url(#g)"/><path d="M1 1L30 30" fill="none" stroke="black" stroke-width="0.1"/></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(output, svg);
    }
    #[test]
    fn preserves_clip_context_and_referenced_geometry() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><clipPath id="c"><rect width="10" height="10"/></clipPath></defs><rect id="base" width="32" height="32" fill="red"/><g clip-path="url(#c)"><rect width="32" height="32" fill="blue"/></g></svg>"##;
        let (output, report) = prune(svg, 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
}
