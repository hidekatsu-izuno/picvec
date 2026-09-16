//! Remove draw operations only when their omission leaves native RGBA unchanged
//! and either a later opaque fill covers their padded bounds or the 4x raster
//! is unchanged too. Work on spatial tiles, parse once, and stop at the first
//! changed tile. Partial occlusion is deliberately left alone.
use crate::svg_document::attrs;
use crate::svg_document::Document;
use rayon::prelude::*;
use resvg::{
    tiny_skia::{Pixmap, PixmapPaint, Transform},
    usvg::{self, Node},
};
use std::collections::HashMap;
use std::sync::Mutex;

#[path = "visibility_coverage.rs"]
mod coverage;

const TILE: usize = 64;
const PREFIX: &str = "picvec-visibility-";
// Each tile owns exact baseline/prefix pixels and up to four cropped isolated
// layers. Stable tile stripes share them across workers, with a 512 MiB budget
// (plus at most one growing tile per stripe until its next access).
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
    path: Vec<usize>,
    name: String,
    stroke: bool,
    ink: bool,
}

fn annotate(document: &Document) -> (Document, Vec<Candidate>) {
    fn visit(
        node: &mut crate::svg_document::Element,
        path: &mut Vec<usize>,
        hidden: bool,
        candidates: &mut Vec<Candidate>,
    ) {
        let hidden = hidden
            || matches!(
                node.name.as_str(),
                "defs" | "clipPath" | "symbol" | "pattern" | "marker"
            );
        let geometry = matches!(
            node.name.as_str(),
            "path" | "rect" | "circle" | "ellipse" | "line" | "polygon" | "polyline"
        );
        if !hidden && geometry && node.children.is_empty() && node.attr("id").is_none() {
            let id = candidates.len();
            candidates.push(Candidate {
                path: path.clone(),
                name: node.name.clone(),
                stroke: node.attr("fill") == Some("none"),
                ink: node.attr("data-structural-ink").is_some(),
            });
            let mut group =
                crate::svg_document::Element::new("g", attrs([("id", format!("{PREFIX}{id}"))]));
            group.children.push(node.clone());
            *node = group;
        } else {
            for (i, child) in node.children.iter_mut().enumerate() {
                path.push(i);
                visit(child, path, hidden, candidates);
                path.pop();
            }
        }
    }
    let mut annotated = document.clone();
    let mut candidates = Vec::new();
    visit(
        annotated.root_mut(),
        &mut Vec::new(),
        false,
        &mut candidates,
    );
    (annotated, candidates)
}

fn omit(document: &Document, candidates: &[Candidate], removed: Vec<bool>) -> (Document, Removed) {
    let mut output = document.clone();
    let mut report = Removed::default();
    // Reverse document order keeps sibling indices valid after removal.
    for (candidate, remove) in candidates.iter().zip(removed).rev() {
        if !remove {
            continue;
        }
        let mut parent = output.root_mut();
        for &i in &candidate.path[..candidate.path.len() - 1] {
            parent = &mut parent.children[i];
        }
        parent.children.remove(*candidate.path.last().unwrap());
        report.paths += usize::from(candidate.name == "path");
        report.rects += usize::from(candidate.name == "rect");
        report.circles += usize::from(candidate.name == "circle");
        report.ellipses += usize::from(candidate.name == "ellipse");
        report.lines += usize::from(candidate.name == "line");
        report.shapes += 1;
        report.strokes += usize::from(candidate.stroke);
        report.ink += usize::from(candidate.ink);
    }
    (output, report)
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

// Restrict tile membership only for an untransformed, directly clipped draw.
// Keep render_node's original layer bounds and transform: changing its canvas
// or origin changes antialias rounding. This only avoids tiles outside the
// clip's support, with two source pixels of outward padding at both scales.
fn clipped_tile_bounds(node: &Node, original: [f32; 4]) -> [f32; 4] {
    let Node::Group(group) = node else {
        return original;
    };
    let mut group = group;
    loop {
        if !group.abs_transform().is_identity()
            || !group.filters().is_empty()
            || group.blend_mode() != usvg::BlendMode::Normal
        {
            return original;
        }
        if let Some(clip) = group.clip_path() {
            if !clip.transform().is_identity() {
                return original;
            }
            let b = clip.root().bounding_box();
            return [
                original[0].max(b.left() - 2.0),
                original[1].max(b.top() - 2.0),
                original[2].min(b.right() + 2.0),
                original[3].min(b.bottom() + 2.0),
            ];
        }
        let [Node::Group(child)] = group.children() else {
            return original;
        };
        group = child;
    }
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
            bounds: clipped_tile_bounds(
                node,
                [
                    b.left() - 1.0,
                    b.top() - 1.0,
                    b.right() + 1.0,
                    b.bottom() + 1.0,
                ],
            ),
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
        layers: &mut HashMap<usize, (crate::svg_fragments::Layer, u128)>,
    ) {
        for &i in members {
            if !active[i] || Some(i) == skip {
                continue;
            }
            let node = draws[i].node;
            let b = node.abs_layer_bounding_box().unwrap();
            let transform = Transform::from_scale(self.scale as f32, self.scale as f32)
                .pre_translate(b.x() - self.x as f32, b.y() - self.y as f32);
            // An isolated normal, opacity-one group already renders into an
            // independent premultiplied layer in resvg. Reuse that exact draw
            // operation; never flatten a sequence of translucent operations.
            let isolated = matches!(node, Node::Group(g) if crate::svg_fragments::isolated_group_source_over(g));
            if isolated {
                if let Some((layer, _)) = layers.get(&i) {
                    pixels.draw_pixmap(
                        layer.x,
                        layer.y,
                        layer.pixels.as_ref(),
                        &PixmapPaint::default(),
                        Transform::identity(),
                        None,
                    );
                    continue;
                }
                let started = crate::time::Instant::now();
                let mut raster = self.blank();
                resvg::render_node(node, transform, &mut raster.as_mut());
                let cost = started.elapsed().as_nanos();
                let layer = crate::svg_fragments::Layer::compact(raster);
                pixels.draw_pixmap(
                    layer.x,
                    layer.y,
                    layer.pixels.as_ref(),
                    &PixmapPaint::default(),
                    Transform::identity(),
                    None,
                );
                // The first layers are often cheap tiny clips. Retain the
                // expensive independent operations, wherever they occur.
                let cheapest = layers
                    .iter()
                    .min_by_key(|entry| (entry.1 .1, *entry.0))
                    .map(|(&id, (_, cost))| (id, *cost));
                if layers.len() < 4 || cheapest.is_some_and(|(_, old)| cost > old) {
                    if layers.len() == 4 {
                        layers.remove(&cheapest.unwrap().0);
                    }
                    layers.insert(i, (layer, cost));
                }
            } else {
                resvg::render_node(node, transform, &mut pixels.as_mut());
            }
        }
    }
}

struct TileRaster {
    baseline: Pixmap,
    prefix: Pixmap,
    // First member not yet included in prefix. Candidates are visited in
    // increasing draw order, so accepted removals never invalidate a prefix.
    next: usize,
    layers: HashMap<usize, (crate::svg_fragments::Layer, u128)>,
}

impl TileRaster {
    fn bytes(&self) -> usize {
        self.baseline.data().len()
            + self.prefix.data().len()
            + self
                .layers
                .values()
                .map(|(layer, _)| layer.pixels.data().len())
                .sum::<usize>()
    }

    fn new(
        view: TileView,
        draws: &[Draw<'_>],
        members: &[usize],
        active: &[bool],
        candidate: usize,
    ) -> Self {
        let next = members.partition_point(|&i| i < candidate);
        let mut prefix = view.blank();
        let mut layers = HashMap::new();
        view.render(
            draws,
            &members[..next],
            active,
            None,
            &mut prefix,
            &mut layers,
        );
        let mut baseline = prefix.clone();
        view.render(
            draws,
            &members[next..],
            active,
            None,
            &mut baseline,
            &mut layers,
        );
        Self {
            baseline,
            prefix,
            next,
            layers,
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
            &mut self.layers,
        );
        self.next = end;
        // Clone the exact framebuffer, then execute the original suffix.
        // Compositing a separately rendered suffix would change alpha rounding.
        let mut after = self.prefix.clone();
        view.render(
            draws,
            &members[end..],
            active,
            Some(candidate),
            &mut after,
            &mut self.layers,
        );
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
        // A previous query may have populated another exact isolated layer.
        while self.entries.len() > 1
            && self
                .entries
                .values()
                .map(|e| e.raster.bytes())
                .sum::<usize>()
                > 16 * 1024 * 1024
        {
            let oldest = *self
                .entries
                .iter()
                .filter(|(k, _)| **k != key)
                .min_by_key(|(_, e)| e.last_used)
                .unwrap()
                .0;
            self.entries.remove(&oldest);
        }
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

struct VisibilityEvidence {
    witnesses: Vec<Vec<usize>>,
    prefix_noop: Vec<bool>,
}

/// Each pixel in `alternative` follows one draw's omission through the same
/// subsequent render operations. Resvg composites nodes pointwise onto the
/// destination (filters use independent source layers), so pixels can track
/// different omissions in one framebuffer. A surviving difference is an exact
/// native-scale witness. Prior accepted removals invalidate affected witnesses.
fn visibility_evidence(
    draws: &[Draw<'_>],
    tiles: &[Vec<usize>],
    width: usize,
    height: usize,
    columns: usize,
    covered: &[bool],
) -> VisibilityEvidence {
    let evidence: Vec<_> = tiles
        .par_iter()
        .enumerate()
        .map(|(tile, members)| {
            let x = tile % columns * TILE;
            let y = tile / columns * TILE;
            let w = TILE.min(width - x);
            let h = TILE.min(height - y);
            let view = TileView {
                x,
                y,
                w,
                h,
                scale: 1,
            };
            let mut pixels = view.blank();
            let mut alternative = view.blank();
            let mut owners = vec![usize::MAX; w * h];
            let mut counts = vec![0usize; members.len()];
            let mut changed = vec![false; members.len()];
            for (local, &i) in members.iter().enumerate() {
                let before = pixels.clone();
                let node = draws[i].node;
                let b = node.abs_layer_bounding_box().unwrap();
                let transform = Transform::from_translate(b.x() - x as f32, b.y() - y as f32);
                if matches!(node, Node::Group(g) if crate::svg_fragments::isolated_group_source_over(g)) {
                    let mut layer = view.blank();
                    resvg::render_node(node, transform, &mut layer.as_mut());
                    for target in [&mut pixels, &mut alternative] {
                        target.draw_pixmap(0, 0, layer.as_ref(), &PixmapPaint::default(), Transform::identity(), None);
                    }
                } else {
                    resvg::render_node(node, transform, &mut pixels.as_mut());
                    resvg::render_node(node, transform, &mut alternative.as_mut());
                }
                changed[local] = before.data() != pixels.data();
                for (k, owner) in owners.iter_mut().enumerate() {
                    if *owner != usize::MAX
                        && pixels.data()[k * 4..k * 4 + 4] == alternative.data()[k * 4..k * 4 + 4]
                    {
                        counts[*owner] -= 1;
                        *owner = usize::MAX;
                    }
                }
                if draws[i].candidate.is_none() {
                    continue;
                }
                // Prefer strong, spatially spread differences so a later thin
                // boundary does not erase every witness for a broad fill. Keep
                // at least one existing witness for each earlier draw.
                let mut best = [None::<(u16, usize)>; 8];
                for (k, &owner) in owners.iter().enumerate() {
                    if owner != usize::MAX && counts[owner] <= 1 {
                        continue;
                    }
                    let score: u16 = before.data()[k * 4..k * 4 + 4]
                        .iter()
                        .zip(&pixels.data()[k * 4..k * 4 + 4])
                        .map(|(&a, &b)| a.abs_diff(b) as u16)
                        .sum();
                    if score == 0 {
                        continue;
                    }
                    let bin = (k % w) * 4 / w + 4 * ((k / w) * 2 / h);
                    if best[bin].is_none_or(|(s, _)| score > s) {
                        best[bin] = Some((score, k));
                    }
                }
                for (_, k) in best.into_iter().flatten() {
                    let owner = owners[k];
                    if owner != usize::MAX && counts[owner] <= 1 {
                        continue;
                    }
                    if owner != usize::MAX {
                        counts[owner] -= 1;
                    }
                    owners[k] = local;
                    counts[local] += 1;
                    alternative.data_mut()[k * 4..k * 4 + 4]
                        .copy_from_slice(&before.data()[k * 4..k * 4 + 4]);
                }
            }
            let points = owners.into_iter().enumerate().filter_map(|(k, owner)| {
                (owner != usize::MAX).then(|| (members[owner], (y + k / w) * width + x + k % w))
            }).collect::<Vec<_>>();
            (points, members.iter().copied().zip(changed).collect::<Vec<_>>())
        }).collect();
    let mut witnesses = vec![Vec::new(); draws.len()];
    let mut prefix_noop = vec![true; draws.len()];
    for (points, changes) in evidence {
        for (i, changed) in changes {
            prefix_noop[i] &= !changed;
        }
        for (i, pixel) in points {
            if witnesses[i].len() < 16 {
                witnesses[i].push(pixel);
            }
        }
    }
    // A draw that changes no prefix pixel remains a no-op after every later
    // draw. Check 4x too, only in tiles containing a surviving native no-op.
    let changed: Vec<Vec<usize>> = tiles
        .par_iter()
        .enumerate()
        .map(|(tile, members)| {
            let Some(last) = members
                .iter()
                .rposition(|&i| !covered[i] && prefix_noop[i] && draws[i].candidate.is_some())
            else {
                return Vec::new();
            };
            let x = tile % columns * TILE;
            let y = tile / columns * TILE;
            let view = TileView {
                x,
                y,
                w: TILE.min(width - x),
                h: TILE.min(height - y),
                scale: 4,
            };
            let mut pixels = view.blank();
            let mut changed = Vec::new();
            // Only the prefix up to the last queried draw matters. Later
            // source-field composites cannot affect a prefix no-op proof.
            for &i in &members[..=last] {
                let before = (!covered[i] && prefix_noop[i] && draws[i].candidate.is_some())
                    .then(|| pixels.clone());
                let node = draws[i].node;
                let b = node.abs_layer_bounding_box().unwrap();
                let transform = Transform::from_scale(4.0, 4.0)
                    .pre_translate(b.x() - x as f32, b.y() - y as f32);
                resvg::render_node(node, transform, &mut pixels.as_mut());
                if before.is_some_and(|p| p.data() != pixels.data()) {
                    changed.push(i);
                }
            }
            changed
        })
        .collect();
    for i in changed.into_iter().flatten() {
        prefix_noop[i] = false;
    }
    for (i, &hidden) in covered.iter().enumerate() {
        if hidden {
            prefix_noop[i] = false;
        }
    }
    VisibilityEvidence {
        witnesses,
        prefix_noop,
    }
}

#[cfg(test)]
fn visibility_witnesses(
    draws: &[Draw<'_>],
    tiles: &[Vec<usize>],
    width: usize,
    height: usize,
    columns: usize,
) -> Vec<Vec<usize>> {
    visibility_evidence(
        draws,
        tiles,
        width,
        height,
        columns,
        &vec![false; draws.len()],
    )
    .witnesses
}

pub(crate) fn prune(document: &Document, width: usize, height: usize) -> (Document, Removed) {
    #[cfg(feature = "diagnostics")]
    if let Some(prefix) = std::env::var_os("PICVEC_VISIBILITY_DUMP") {
        let _ = std::fs::write(
            format!("{}-{width}x{height}.svg", prefix.to_string_lossy()),
            document,
        );
    }
    prune_cached(document, width, height, CACHE_TILES)
}

fn prune_cached(
    document: &Document,
    width: usize,
    height: usize,
    capacity: usize,
) -> (Document, Removed) {
    // Prefer geometric coverage when it can be proved; retain the raster
    // fallback for all other draws. Diagnostics can reproduce the old path.
    let geometric = !(cfg!(feature = "diagnostics")
        && std::env::var_os("PICVEC_VISIBILITY_GEOMETRY").is_some_and(|v| v == "0"));
    prune_impl(document, width, height, capacity, geometric)
}

fn prune_impl(
    document: &Document,
    width: usize,
    height: usize,
    capacity: usize,
    geometric: bool,
) -> (Document, Removed) {
    // A <use> can render the same source element again through a filter or
    // transform. Removing its source would alter both instances; do not treat
    // those instances as independently removable draw operations.
    if document.root().contains_name("use") {
        return (document.clone(), Removed::default());
    }
    let (annotated, candidates) = annotate(document);
    if candidates.is_empty() {
        return (document.clone(), Removed::default());
    }
    let Ok(tree) = usvg::Tree::from_str(&annotated, &usvg::Options::default()) else {
        return (document.clone(), Removed::default());
    };
    // The core serializer has no viewport transform. Leave other documents
    // intact rather than detaching children from their inherited transform.
    if !tree.root().transform().is_identity() {
        return (document.clone(), Removed::default());
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
                if !crate::svg_fragments::has_paint_in(
                    draw.node,
                    [
                        (x * TILE) as f32,
                        (y * TILE) as f32,
                        ((x + 1) * TILE).min(width) as f32,
                        ((y + 1) * TILE).min(height) as f32,
                    ],
                ) {
                    continue;
                }
                tiles[tile].push(i);
                memberships[i].push(tile);
            }
        }
    }
    let geometry_started = crate::time::Instant::now();
    let mut covered = vec![false; draws.len()];
    if geometric {
        let covers: Vec<_> = draws.iter().map(|d| coverage::Cover::new(d.node)).collect();
        covered.par_iter_mut().enumerate().for_each(|(i, hidden)| {
            if draws[i].candidate.is_none() {
                return;
            }
            // Any single covering draw must appear in every occupied tile.
            // Search the shortest list and keep the cost bounded on dense art.
            let Some(members) = memberships[i]
                .iter()
                .map(|&t| &tiles[t])
                .min_by_key(|m| m.len())
            else {
                return;
            };
            *hidden = members
                .iter()
                .rev()
                .copied()
                .take_while(|&j| j > i)
                .take(128)
                .any(|j| {
                    covers[j]
                        .as_ref()
                        .is_some_and(|c| c.contains(draws[i].bounds))
                });
        });
    }
    if cfg!(any(test, feature = "diagnostics"))
        && std::env::var_os("PICVEC_VISIBILITY_DIAGNOSTICS").is_some()
    {
        eprintln!(
            "picvec visibility geometric covers: {}/{}, {:.3}s",
            covered.iter().filter(|&&b| b).count(),
            draws.len(),
            geometry_started.elapsed().as_secs_f64()
        );
    }
    let witness_started = crate::time::Instant::now();
    let VisibilityEvidence {
        witnesses,
        prefix_noop,
    } = visibility_evidence(&draws, &tiles, width, height, columns, &covered);
    let mut prefix_dirty = Vec::<[f32; 4]>::new();
    let mut prefix_shortcuts = 0usize;
    if cfg!(feature = "diagnostics") && std::env::var_os("PICVEC_VISIBILITY_DIAGNOSTICS").is_some()
    {
        eprintln!(
            "picvec visibility witnesses: {}/{} draws, {:.3}s",
            witnesses.iter().filter(|v| !v.is_empty()).count(),
            draws.len(),
            witness_started.elapsed().as_secs_f64()
        );
    }
    let mut shortcuts = 0usize;
    let mut slow_reports = 0usize;
    let mut invalidated = vec![false; width * height];
    let mut active = vec![true; draws.len()];
    let mut removed = missing_candidates(tree.root(), candidates.len());
    // Tile identity, not the current Rayon worker, selects the cache. Tasks
    // can move between workers without throwing away their exact prefix.
    let caches: Vec<_> = (0..32)
        .map(|_| Mutex::new(TileCache::new(capacity.min(32))))
        .collect();
    for (i, draw) in draws.iter().enumerate() {
        let Some(candidate) = draw.candidate else {
            continue;
        };
        if prefix_noop[i]
            && prefix_dirty.iter().all(|b| {
                draw.bounds[0] > b[2]
                    || b[0] > draw.bounds[2]
                    || draw.bounds[1] > b[3]
                    || b[1] > draw.bounds[3]
            })
        {
            active[i] = false;
            removed[candidate] = true;
            prefix_shortcuts += 1;
            // Removing an exact prefix no-op cannot change any later prefix.
            continue;
        }
        if witnesses[i].iter().any(|&pixel| !invalidated[pixel]) {
            shortcuts += 1;
            continue;
        }
        let check_started = crate::time::Instant::now();
        let check = |scale, tile: usize| {
            let x = tile % columns * TILE;
            let y = tile / columns * TILE;
            let w = TILE.min(width - x);
            let h = TILE.min(height - y);
            let view = TileView { x, y, w, h, scale };
            let mut cache = caches[tile % caches.len()].lock().unwrap();
            let raster = cache.get_or_insert((tile, scale), || {
                TileRaster::new(view, &draws, &tiles[tile], &active, i)
            });
            raster.unchanged_without(view, &draws, &tiles[tile], &active, i)
        };
        let unchanged = [1, 4]
            .into_iter()
            .filter(|&scale| scale == 1 || !covered[i])
            .all(|scale| {
                // Most visible objects fail on the first tile. Preserve that cheap
                // rejection before scheduling the rest of a large footprint.
                let Some((&first, rest)) = memberships[i].split_first() else {
                    return true;
                };
                check(scale, first)
                    && if rest.len() >= 4 {
                        rest.par_iter().all(|&tile| check(scale, tile))
                    } else {
                        rest.iter().all(|&tile| check(scale, tile))
                    }
            });
        if cfg!(feature = "diagnostics")
            && std::env::var_os("PICVEC_VISIBILITY_DIAGNOSTICS").is_some()
            && check_started.elapsed().as_secs_f64() > 0.5
            && slow_reports < 8
        {
            eprintln!("picvec visibility slow omission: draw {i}, {} tiles, unchanged {unchanged}, {:.3}s, bounds {:?}", memberships[i].len(), check_started.elapsed().as_secs_f64(), draw.bounds);
            slow_reports += 1;
        }
        if unchanged {
            active[i] = false;
            removed[candidate] = true;
            prefix_dirty.push(draw.bounds);
            let [left, top, right, bottom] = draw.bounds;
            let x0 = (left.floor().max(0.0) as usize).min(width);
            let y0 = (top.floor().max(0.0) as usize).min(height);
            let x1 = (right.ceil().max(0.0) as usize).min(width);
            let y1 = (bottom.ceil().max(0.0) as usize).min(height);
            for y in y0..y1 {
                invalidated[y * width + x0..y * width + x1].fill(true);
            }
        }
    }
    if cfg!(feature = "diagnostics") && std::env::var_os("PICVEC_VISIBILITY_DIAGNOSTICS").is_some()
    {
        eprintln!("picvec visibility proof shortcuts: {shortcuts} visible, {prefix_shortcuts} prefix no-op");
    }
    omit(document, &candidates, removed)
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
    fn every_mixed_counterfactual_witness_matches_an_independent_omission() {
        let svg = layered_scene().replace("</svg>", r##"<defs><filter id="blur"><feGaussianBlur stdDeviation="1.3"/></filter></defs><g filter="url(#blur)"><rect x="13" y="9" width="77" height="35" fill="#892" fill-opacity=".3"/></g><g style="mix-blend-mode:multiply"><rect x="55" y="32" width="44" height="31" fill="#498" fill-opacity=".2"/></g></svg>"##);
        let (svg, _) = annotate(&Document::from((&svg).to_string()));
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
        let mut draws = Vec::new();
        collect(tree.root(), &mut draws);
        let members: Vec<_> = (0..draws.len()).collect();
        let tiles = vec![members.clone(); 6];
        let witnesses = visibility_witnesses(&draws, &tiles, 160, 80, 3);
        let active = vec![true; draws.len()];
        assert!(witnesses.iter().filter(|p| !p.is_empty()).count() > 8);
        for (i, points) in witnesses.iter().enumerate() {
            for &pixel in points {
                let x = pixel % 160 / TILE * TILE;
                let y = pixel / 160 / TILE * TILE;
                let w = TILE.min(160 - x);
                let h = TILE.min(80 - y);
                let before = reference::draw_tile(&draws, &members, &active, None, x, y, w, h, 1);
                let after = reference::draw_tile(&draws, &members, &active, Some(i), x, y, w, h, 1);
                let k = ((pixel / 160 - y) * w + pixel % 160 - x) * 4;
                assert_ne!(
                    &before.data()[k..k + 4],
                    &after.data()[k..k + 4],
                    "draw {i}, pixel {pixel}"
                );
            }
        }
    }

    #[test]
    fn distant_clipped_patches_and_filtered_spill_match_full_reference() {
        for filter in ["", " filter=\"url(#f)\""] {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="128"><defs><clipPath id="c"><rect width="256" height="128"/></clipPath><filter id="f" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="3"/></filter></defs><rect width="256" height="128" fill="white"/><path d="M125 62H132V67H125Z" fill="red"/><g clip-path="url(#c)"><path d="M2 2H4V4H2Z" fill="blue"/><path d="M244 119H247V123H244Z" fill="green"/><g{filter}><path d="M62 63H63V64H62Z" fill="black"/></g></g></svg>"##
            );
            assert_eq!(
                prune(&Document::from((&svg).to_string()), 256, 128),
                reference::prune(&Document::from((&svg).to_string()), 256, 128)
            );
        }
    }

    #[test]
    fn clip_tile_membership_matches_original_omission_renderer() {
        for transform in ["", " transform=\"translate(0.25 0.5)\""] {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="192" height="128"><defs><clipPath id="c"><path d="M65.1 62.7H80.3V65.2H65.1Z"/></clipPath><linearGradient id="g"><stop stop-color="red" stop-opacity="0.2"/><stop offset="1" stop-color="blue" stop-opacity="0.8"/></linearGradient></defs><rect width="192" height="128" fill="white"/><g{transform}><rect width="192" height="128" fill="url(#g)" clip-path="url(#c)"/><rect width="192" height="128" fill="url(#g)" clip-path="url(#c)"/></g><path d="M64 62H85V63H64Z" fill="white"/></svg>"##
            );
            assert_eq!(
                prune(&Document::from((&svg).to_string()), 192, 128),
                reference::prune(&Document::from((&svg).to_string()), 192, 128)
            );
        }
    }

    #[test]
    fn a_general_removal_invalidates_a_later_prefix_noop() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect id="fixed" width="64" height="64" fill="white"/><rect x="8" y="8" width="16" height="16" fill="red"/><rect x="8" y="8" width="16" height="16" fill="red"/></svg>"##;
        let expected = reference::prune(&Document::from((svg).to_string()), 64, 64);
        assert_eq!(expected.1.shapes, 1);
        assert_eq!(prune(&Document::from((svg).to_string()), 64, 64), expected);
    }

    #[test]
    fn removing_a_covered_draw_invalidates_the_covering_draws_witness() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect id="fixed" width="64" height="64" fill="white"/><rect x="8" y="8" width="16" height="16" fill="red"/><rect x="8" y="8" width="16" height="16" fill="white"/></svg>"##;
        let expected = reference::prune(&Document::from((svg).to_string()), 64, 64);
        assert_eq!(expected.1.shapes, 2);
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    assert_eq!(prune(&Document::from((svg).to_string()), 64, 64), expected);
                });
        }
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
        let (svg, _) = annotate(&Document::from((&layered_scene()).to_string()));
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
            layers: HashMap::new(),
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
            let expected = reference::prune(&Document::from((&svg).to_string()), w, h);
            assert!(expected.1.shapes > 0, "fixture {w}x{h} needs covered marks");
            for capacity in [1, 3, CACHE_TILES] {
                let actual = prune_cached(&Document::from((&svg).to_string()), w, h, capacity);
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
                let start = crate::time::Instant::now();
                let result = if reference_version {
                    reference::prune(&Document::from((&svg).to_string()), w, h)
                } else {
                    prune(&Document::from((&svg).to_string()), w, h)
                };
                (result, start.elapsed().as_secs_f64())
            };
            if std::env::var_os("PICVEC_VISIBILITY_BENCH_CANDIDATE_ONLY").is_some() {
                // Profiling only; this mode makes no reference-equivalence claim.
                for repeat in 0..repeats {
                    let (candidate, seconds) = measure(false);
                    std::fs::write(
                        std::path::Path::new(&output).with_extension("svg"),
                        &candidate.0,
                    )
                    .unwrap();
                    rows.push(serde_json::json!({"input": path, "repeat": repeat, "candidate_seconds": seconds, "reference_compared": false, "removed_shapes": candidate.1.shapes}));
                    std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
                    eprintln!("visibility profiling {}: {seconds:.3}s", path.display());
                }
                continue;
            }
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

    #[test]
    fn geometric_covers_preserve_holes_alpha_and_subpixel_exposure() {
        let cases = [
            // Opaque cubic encloses the mark with a wide margin.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M4 32C4 4 60 4 60 32C60 60 4 60 4 32Z"/>"##,
                true,
            ),
            // A curved notch enters the query despite the broad outer bounds.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M2 2H62V62H2V40C50 40 50 24 2 24Z"/>"##,
                false,
            ),
            // Opposite winding makes a nonzero-rule hole too.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" d="M2 2H62V62H2Z M30 30V34H34V30Z"/>"##,
                false,
            ),
            // A hole inside the queried bounds defeats boundary-only containment.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path fill="blue" fill-rule="evenodd" d="M2 2H62V62H2Z M30 30H34V34H30Z"/>"##,
                false,
            ),
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><rect x="2" y="2" width="60" height="60" fill="blue" fill-opacity=".5"/>"##,
                false,
            ),
            // A broad object with only a subpixel strip exposed.
            (
                r##"<rect x="8" y="8" width="40.1" height="40" fill="red"/><rect x="2" y="2" width="46" height="60" fill="blue"/>"##,
                false,
            ),
            // Implicit closure and a transformed quadratic boundary.
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><path transform="translate(2 2)" fill="blue" d="M0 0H60V60H0 Q-5 30 0 0"/>"##,
                true,
            ),
            (
                r##"<rect x="28" y="28" width="8" height="8" fill="red"/><defs><clipPath id="c"><rect width="30" height="64"/></clipPath></defs><rect width="64" height="64" fill="blue" clip-path="url(#c)"/>"##,
                false,
            ),
            // Shared boundaries are excluded even when both fills are opaque.
            (
                r##"<rect x="8" y="8" width="40" height="40" fill="red"/><rect x="8" y="8" width="40" height="40" fill="blue"/>"##,
                false,
            ),
        ];
        for (body, expected_cover) in cases {
            let svg = format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">{body}</svg>"##
            );
            let document = Document::from(svg.clone());
            let (annotated, _) = annotate(&document);
            let tree = usvg::Tree::from_str(&annotated, &usvg::Options::default()).unwrap();
            let mut draws = Vec::new();
            collect(tree.root(), &mut draws);
            let covered = coverage::Cover::new(draws.last().unwrap().node)
                .is_some_and(|cover| cover.contains(draws[0].bounds));
            assert_eq!(covered, expected_cover, "{body}");
            let baseline = prune_impl(&document, 64, 64, CACHE_TILES, false);
            let candidate = prune_impl(&document, 64, 64, CACHE_TILES, true);
            assert_eq!(baseline, candidate, "{body}");
            for scale in [1, 4, 8, 16] {
                assert!(
                    image(&svg, scale) == image(&candidate.0, scale),
                    "RGBA differs at scale {scale}: {body}"
                );
            }
        }
    }

    #[test]
    fn unresolved_subpixel_exposure_retains_the_legacy_fallback() {
        // Characterize a pre-existing limitation, not a geometry shortcut:
        // 1x/4x equality does not imply equality at arbitrary SVG zoom levels.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect x="8" y="8" width="40.01" height="40" fill="red"/><rect x="2" y="2" width="46" height="60" fill="blue"/></svg>"##;
        let document = Document::from(svg);
        let baseline = prune_impl(&document, 64, 64, CACHE_TILES, false);
        let candidate = prune_impl(&document, 64, 64, CACHE_TILES, true);
        assert_eq!(baseline, candidate);
        assert_eq!(candidate.1.shapes, 1);
        for scale in [1, 4, 8] {
            assert!(image(svg, scale) == image(&candidate.0, scale));
        }
        assert!(image(svg, 16) != image(&candidate.0, 16));
    }

    #[test]
    #[ignore = "prototype A/B benchmark; set PICVEC_VISIBILITY_BENCH_INPUTS and PICVEC_VISIBILITY_BENCH_OUTPUT"]
    fn benchmark_geometric_coverage() {
        let paths = std::env::var_os("PICVEC_VISIBILITY_BENCH_INPUTS").unwrap();
        let output = std::env::var_os("PICVEC_VISIBILITY_BENCH_OUTPUT").unwrap();
        let repeats = std::env::var("PICVEC_VISIBILITY_BENCH_REPEATS")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3);
        let mut rows = Vec::new();
        for path in std::env::split_paths(&paths) {
            let document = Document::from(std::fs::read_to_string(&path).unwrap());
            let tree = usvg::Tree::from_str(&document, &usvg::Options::default()).unwrap();
            let (w, h) = (
                tree.size().width().ceil() as usize,
                tree.size().height().ceil() as usize,
            );
            drop(tree);
            let measure = |geometric| {
                let start = crate::time::Instant::now();
                let result = prune_impl(&document, w, h, CACHE_TILES, geometric);
                (result, start.elapsed().as_secs_f64())
            };
            // Warm both implementations; alternate order in measured trials.
            assert_eq!(measure(false).0, measure(true).0);
            for repeat in 0..repeats {
                let (baseline, candidate) = if repeat % 2 == 0 {
                    (measure(false), measure(true))
                } else {
                    let candidate = measure(true);
                    (measure(false), candidate)
                };
                assert_eq!(baseline.0, candidate.0, "{}", path.display());
                eprintln!(
                    "geometry {} trial {}: {:.3}s -> {:.3}s",
                    path.display(),
                    repeat + 1,
                    baseline.1,
                    candidate.1
                );
                rows.push(
                    serde_json::json!({"input": path, "repeat": repeat, "width": w, "height": h,
                    "baseline_seconds": baseline.1, "candidate_seconds": candidate.1,
                    "removed_shapes": candidate.0.1.shapes, "outputs_equal": true}),
                );
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
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 2);
        assert_eq!(report.strokes, 1);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn tile_boundaries_and_multiple_removals_preserve_the_complete_render() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="256" height="32"><rect width="256" height="32" fill="red"/><rect x="60.2" y="4" width="120" height="16" fill="red"/><path d="M10 15.3H200" fill="none" stroke="red" stroke-width="2"/><rect x="128" y="16" width="80" height="16" fill="blue"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 256, 32);
        assert_eq!(report.shapes, 2);
        for scale in [1, 4, 8] {
            assert_eq!(image(svg, scale), image(&output, scale));
        }
    }
    #[test]
    fn zero_area_paint_is_removed_but_its_stroked_counterpart_is_kept() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><path d="M2 2H20" fill="red"/><path d="M2 2H20" fill="none" stroke="red"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 1);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
    #[test]
    fn preserves_alpha_accumulation_gradients_and_subpixel_lines() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect width="25" height="25" fill="red" fill-opacity="0.5"/><rect x="10" width="10" height="25" fill="url(#g)"/><path d="M1 1L30 30" fill="none" stroke="black" stroke-width="0.1"/></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(output, Document::from(svg));
    }
    #[test]
    fn preserves_clip_context_and_referenced_geometry() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><clipPath id="c"><rect width="10" height="10"/></clipPath></defs><rect id="base" width="32" height="32" fill="red"/><g clip-path="url(#c)"><rect width="32" height="32" fill="blue"/></g></svg>"##;
        let (output, report) = prune(&Document::from((svg).to_string()), 32, 32);
        assert_eq!(report.shapes, 0);
        assert_eq!(image(svg, 4), image(&output, 4));
    }
}
