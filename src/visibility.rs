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
include!("../tests/unit/visibility.rs");
