//! Test-only reference: the original full redraw and clear-all cache.
//! Keep this independent of the incremental renderer for exact comparisons.
use super::*;

pub(super) fn draw_tile(
    draws: &[Draw<'_>],
    members: &[usize],
    active: &[bool],
    skip: Option<usize>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    scale: usize,
) -> Pixmap {
    let mut pixmap = Pixmap::new((w * scale) as u32, (h * scale) as u32).unwrap();
    for &i in members {
        if !active[i] || Some(i) == skip {
            continue;
        }
        let node = draws[i].node;
        let b = node.abs_layer_bounding_box().unwrap();
        // render_node subtracts its absolute bounding-box origin internally.
        let transform = Transform::from_scale(scale as f32, scale as f32)
            .pre_translate(b.x() - x as f32, b.y() - y as f32);
        resvg::render_node(node, transform, &mut pixmap.as_mut());
    }
    pixmap
}

pub(super) fn prune(document: &Document, width: usize, height: usize) -> (Document, Removed) {
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
                tiles[tile].push(i);
                memberships[i].push(tile);
            }
        }
    }
    let mut active = vec![true; draws.len()];
    let mut removed: Vec<_> = (0..candidates.len())
        .map(|i| tree.node_by_id(&format!("{PREFIX}{i}")).is_none())
        .collect();
    let mut baseline = HashMap::<(usize, usize), Pixmap>::new();
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
                // Bound the scratch cache independently of source dimensions.
                if baseline.len() >= 128 {
                    baseline.clear();
                }
                let before = baseline.entry((tile, scale)).or_insert_with(|| {
                    draw_tile(&draws, &tiles[tile], &active, None, x, y, w, h, scale)
                });
                let after = draw_tile(&draws, &tiles[tile], &active, Some(i), x, y, w, h, scale);
                if before.data() != after.data() {
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
    omit(document, &candidates, removed)
}
