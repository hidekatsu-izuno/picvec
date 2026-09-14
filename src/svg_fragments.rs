//! Reuse unchanged parsed draw operations during covered-hole trials.
//! Only inert groups are split. Clips, filters and transforms stay attached to
//! their entire group. The final hole check still uses the full SVG renderer.
use resvg::{
    tiny_skia::{IntRect, Pixmap, PixmapPaint, Transform},
    usvg::{Options, Tree},
};
use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct Fragment {
    tree: Arc<Tree>,
    top: f32,
    bottom: f32,
    cache_layer: bool,
}

pub(crate) struct Scene {
    draws: Vec<Fragment>,
    layers: Arc<Mutex<Layers>>,
}

type LayerKey = (usize, usize, usize, u32, u32);

pub(crate) struct Layer {
    pub(crate) pixels: Pixmap,
    pub(crate) x: i32,
    pub(crate) y: i32,
}

impl Layer {
    pub(crate) fn compact(pixels: Pixmap) -> Self {
        // Strip only exactly zero RGBA padding after rendering in the original
        // full-band context. Rendering on a smaller canvas could change filter
        // clipping; cropping existing integer pixels cannot.
        let w = pixels.width() as usize;
        let (mut left, mut top, mut right, mut bottom) = (w, pixels.height() as usize, 0, 0);
        for (i, p) in pixels.data().chunks_exact(4).enumerate() {
            if p.iter().any(|&v| v != 0) {
                left = left.min(i % w);
                right = right.max(i % w + 1);
                top = top.min(i / w);
                bottom = bottom.max(i / w + 1);
            }
        }
        if right == 0 {
            return Self {
                pixels: Pixmap::new(1, 1).unwrap(),
                x: 0,
                y: 0,
            };
        }
        let rect = IntRect::from_xywh(
            left as i32,
            top as i32,
            (right - left) as u32,
            (bottom - top) as u32,
        )
        .unwrap();
        Self {
            pixels: pixels.clone_rect(rect).unwrap(),
            x: left as i32,
            y: top as i32,
        }
    }
}

#[derive(Default)]
struct Layers {
    entries: HashMap<LayerKey, (Arc<Layer>, u64)>,
    bytes: usize,
    tick: u64,
}

impl Layers {
    fn get(&mut self, key: LayerKey) -> Option<Arc<Layer>> {
        self.tick += 1;
        let (pixels, age) = self.entries.get_mut(&key)?;
        *age = self.tick;
        Some(Arc::clone(pixels))
    }

    fn insert(&mut self, key: LayerKey, pixels: Arc<Layer>) {
        const LIMIT: usize = 256 * 1024 * 1024;
        let bytes = pixels.pixels.data().len();
        if bytes > LIMIT || self.entries.contains_key(&key) {
            return;
        }
        while self.bytes + bytes > LIMIT {
            let oldest = *self
                .entries
                .iter()
                .min_by_key(|(_, (_, age))| age)
                .unwrap()
                .0;
            self.bytes -= self.entries.remove(&oldest).unwrap().0.pixels.data().len();
        }
        self.tick += 1;
        self.bytes += bytes;
        self.entries.insert(key, (pixels, self.tick));
    }
}

fn isolated_source_over(tree: &Tree) -> bool {
    isolated_group_source_over(tree.root())
}

pub(crate) fn isolated_group_source_over(mut group: &resvg::usvg::Group) -> bool {
    loop {
        if group.should_isolate() {
            return group.opacity().get() == 1.0
                && group.blend_mode() == resvg::usvg::BlendMode::Normal;
        }
        let [resvg::usvg::Node::Group(child)] = group.children() else {
            return false;
        };
        group = child;
    }
}

/// A normal, unfiltered group cannot paint where none of its children can
/// paint. Use their union, not the large rectangle spanning distant patches.
/// Clip/mask operations only reduce support. Filters/blends stay conservative.
/// This does not change render context, origins, paths, or clip geometry.
pub(crate) fn has_paint_in(node: &resvg::usvg::Node, rect: [f32; 4]) -> bool {
    // usvg's Node::abs_layer_bounding_box omits stroke expansion for leaf
    // paths (and returns None for horizontal/vertical centre lines). Groups
    // need filter-expanded layer bounds; leaves need their actual stroke bounds.
    let b = match node {
        resvg::usvg::Node::Group(g) => g.abs_layer_bounding_box().to_rect(),
        _ => node.abs_stroke_bounding_box(),
    };
    if b.right() + 2.0 < rect[0]
        || b.left() - 2.0 > rect[2]
        || b.bottom() + 2.0 < rect[1]
        || b.top() - 2.0 > rect[3]
    {
        return false;
    }
    if let resvg::usvg::Node::Group(g) = node {
        if g.filters().is_empty() && g.blend_mode() == resvg::usvg::BlendMode::Normal {
            return g.children().iter().any(|child| has_paint_in(child, rect));
        }
    }
    true
}

impl Scene {
    pub(crate) fn render(&self, scale: usize, y: usize, pixels: &mut Pixmap) {
        let transform = Transform::from_row(
            scale as f32,
            0.0,
            0.0,
            scale as f32,
            0.0,
            -((y * scale) as f32),
        );
        let bottom = y as f32 + pixels.height() as f32 / scale as f32;
        for draw in &self.draws {
            if draw.bottom + 2.0 < y as f32 || draw.top - 2.0 > bottom {
                continue;
            }
            if !draw.tree.root().children().iter().any(|node| {
                has_paint_in(node, [f32::NEG_INFINITY, y as f32, f32::INFINITY, bottom])
            }) {
                continue;
            }
            if draw.cache_layer {
                // This fragment is exactly one isolated, opacity-one,
                // source-over operation. Its premultiplied pixels already
                // exist independently of the destination in resvg. Cache that
                // operation, not a flattened sequence of translucent draws.
                let key = (
                    Arc::as_ptr(&draw.tree) as usize,
                    scale,
                    y,
                    pixels.width(),
                    pixels.height(),
                );
                let cached = self.layers.lock().unwrap().get(key);
                let layer = cached.unwrap_or_else(|| {
                    let mut layer = Pixmap::new(pixels.width(), pixels.height()).unwrap();
                    resvg::render(&draw.tree, transform, &mut layer.as_mut());
                    let layer = Arc::new(Layer::compact(layer));
                    self.layers.lock().unwrap().insert(key, Arc::clone(&layer));
                    layer
                });
                pixels.draw_pixmap(
                    layer.x,
                    layer.y,
                    layer.pixels.as_ref(),
                    &PixmapPaint::default(),
                    Transform::identity(),
                    None,
                );
            } else {
                resvg::render(&draw.tree, transform, &mut pixels.as_mut());
            }
        }
    }
}

struct Template {
    range: std::ops::Range<usize>,
    path: Option<std::ops::Range<usize>>,
    document: String,
    document_body: usize,
    draw: Fragment,
}

pub(crate) struct Cache {
    source: String,
    templates: Vec<Template>,
    root: String,
    trees: HashMap<String, Arc<Tree>>,
    layers: Arc<Mutex<Layers>>,
}

fn refs(node: roxmltree::Node<'_, '_>, ids: &mut BTreeSet<String>) -> Option<()> {
    for n in node.descendants().filter(|n| n.is_element()) {
        for attr in n.attributes() {
            let mut value = attr.value();
            if attr.name() == "href" {
                ids.insert(value.strip_prefix('#')?.to_owned());
            }
            while let Some(start) = value.find("url(") {
                value = &value[start + 4..];
                let end = value.find(')')?;
                let id = value[..end]
                    .trim()
                    .trim_matches(['\'', '"'])
                    .strip_prefix('#')?;
                ids.insert(id.to_owned());
                value = &value[end + 1..];
            }
        }
    }
    Some(())
}

fn documents(svg: &str) -> Option<(String, Vec<(String, std::ops::Range<usize>, usize)>)> {
    let document = roxmltree::Document::parse(svg).ok()?;
    let root = document.root_element();
    if root.tag_name().name() != "svg"
        || root.descendants().any(|n| n.has_tag_name("style"))
        || root.attributes().any(|a| a.value().contains("url("))
    {
        return None;
    }
    let begin = root.range().start;
    let open = svg[begin..begin + svg[begin..].find('>')? + 1].to_owned();
    let mut resources = HashMap::new();
    for defs in root.children().filter(|n| n.has_tag_name("defs")) {
        for node in defs.children().filter(|n| n.is_element()) {
            resources.insert(node.attribute("id")?, node);
        }
    }
    // Colour-patch clips reference strokes in the rendered ink layer. Keep
    // those definitions too, with their original inherited paint attributes.
    for node in root.descendants().filter(|n| n.is_element()) {
        if let Some(id) = node.attribute("id") {
            resources.entry(id).or_insert(node);
        }
    }
    fn visit<'a>(
        node: roxmltree::Node<'a, 'a>,
        svg: &str,
        open: &str,
        wrappers: &str,
        depth: usize,
        resources: &HashMap<&'a str, roxmltree::Node<'a, 'a>>,
        out: &mut Vec<(String, std::ops::Range<usize>, usize)>,
    ) -> Option<()> {
        if node.has_tag_name("defs") {
            return Some(());
        }
        if node.has_tag_name("g")
            && !node.attributes().any(|a| a.value().contains("url("))
            && node.attributes().all(|a| {
                matches!(
                    a.name(),
                    "id" | "fill-rule" | "fill" | "stroke-linecap" | "stroke-linejoin"
                )
            })
        {
            let start = node.range().start;
            let prefix = format!(
                "{wrappers}{}",
                &svg[start..start + svg[start..].find('>')? + 1]
            );
            for child in node.children().filter(|n| n.is_element()) {
                visit(child, svg, open, &prefix, depth + 1, resources, out)?;
            }
            return Some(());
        }
        // Nested viewports and definitions outside the root need their original
        // document context; use the existing parser for those documents.
        if node
            .descendants()
            .any(|n| n.has_tag_name("svg") || n.has_tag_name("defs"))
        {
            return None;
        }
        let mut needed = BTreeSet::new();
        refs(node, &mut needed)?;
        let mut done = BTreeSet::new();
        while let Some(id) = needed.iter().find(|id| !done.contains(*id)).cloned() {
            let resource = *resources.get(id.as_str())?;
            refs(resource, &mut needed)?;
            done.insert(id);
        }
        let mut fragment = String::from(open);
        fragment.push_str("<defs>");
        for id in needed {
            let resource = resources[id.as_str()];
            let outside_defs = !resource.ancestors().any(|n| n.has_tag_name("defs"));
            let mut ancestors = Vec::new();
            if outside_defs {
                for parent in resource.ancestors().skip(1) {
                    if parent.has_tag_name("svg") {
                        break;
                    }
                    if !parent.has_tag_name("g")
                        || !parent.attributes().all(|a| {
                            matches!(
                                a.name(),
                                "id" | "fill-rule" | "fill" | "stroke-linecap" | "stroke-linejoin"
                            ) && !a.value().contains("url(")
                        })
                    {
                        return None;
                    }
                    ancestors.push(parent);
                }
                for parent in ancestors.iter().rev() {
                    let start = parent.range().start;
                    fragment.push_str(&svg[start..start + svg[start..].find('>')? + 1]);
                }
            }
            fragment.push_str(&svg[resource.range()]);
            for _ in ancestors {
                fragment.push_str("</g>");
            }
        }
        fragment.push_str("</defs>");
        fragment.push_str(wrappers);
        let document_body = fragment.len();
        fragment.push_str(&svg[node.range()]);
        for _ in 0..depth {
            fragment.push_str("</g>");
        }
        fragment.push_str("</svg>");
        out.push((fragment, node.range(), document_body));
        Some(())
    }
    let mut fragments = Vec::new();
    for node in root.children().filter(|n| n.is_element()) {
        visit(node, svg, &open, "", 0, &resources, &mut fragments)?;
    }
    Some((open, fragments))
}

impl Cache {
    pub(crate) fn new(svg: &str) -> Option<Self> {
        let (root, documents) = documents(svg)?;
        // Bound retained source text. Candidate-specific parses are temporary,
        // so rejected trials cannot grow the cache without limit.
        if documents.iter().map(|d| d.0.len()).sum::<usize>() > 64 * 1024 * 1024 {
            return None;
        }
        let dom = roxmltree::Document::parse(svg).ok()?;
        // A named ancestor can be referenced even when its child path has no
        // ID. Such edits must rebuild all dependent resource documents.
        let mut referenced_ids = BTreeSet::new();
        refs(dom.root_element(), &mut referenced_ids)?;
        let referenced: Vec<_> = dom
            .descendants()
            .filter(|n| n.is_element())
            .filter(|n| !n.ancestors().any(|p| p.has_tag_name("defs")))
            .filter(|n| {
                n.attribute("id")
                    .is_some_and(|id| referenced_ids.contains(id))
            })
            .map(|n| n.range())
            .collect();
        let mut trees = HashMap::new();
        let mut templates = Vec::new();
        for (document, range, document_body) in documents {
            let tree = match trees.entry(document.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let tree = Arc::new(Tree::from_str(&document, &Options::default()).ok()?);
                    entry.insert(Arc::clone(&tree));
                    tree
                }
            };
            let raw = &svg[range.clone()];
            // Hole simplification changes only the d attribute of anonymous
            // emitted paths. Named paths may be referenced by another draw.
            let path = if raw.starts_with("<path ")
                && raw.ends_with("/>")
                && !raw.contains(" id=")
                && !referenced
                    .iter()
                    .any(|r| r.start <= range.start && range.end <= r.end)
            {
                raw.find(" d=\"").and_then(|i| {
                    let start = i + 4;
                    Some(start..start + raw[start..].find('"')?)
                })
            } else {
                None
            };
            templates.push(Template {
                range,
                path,
                document,
                document_body,
                draw: fragment(tree, true),
            });
        }
        Some(Self {
            source: svg.to_owned(),
            templates,
            root,
            trees,
            layers: Arc::new(Mutex::new(Layers::default())),
        })
    }

    // Check every unchanged byte, including definitions, group attributes and
    // non-path draws. Only anonymous path data can vary. This avoids rebuilding
    // a DOM and thousands of standalone documents for each small hole trial.
    fn path_scene(&self, svg: &str) -> Option<Scene> {
        let mut position = 0;
        let mut previous = 0;
        let mut draws = Vec::with_capacity(self.templates.len());
        for template in &self.templates {
            let gap = &self.source[previous..template.range.start];
            if !svg.get(position..)?.starts_with(gap) {
                return None;
            }
            position += gap.len();
            let raw = &self.source[template.range.clone()];
            if let Some(path) = &template.path {
                if !svg.get(position..)?.starts_with(&raw[..path.start]) {
                    return None;
                }
                position += path.start;
                let end = position + svg.get(position..)?.find('"')?;
                let data = &svg[position..end];
                if !svg.get(end..)?.starts_with(&raw[path.end..]) {
                    return None;
                }
                position = end + raw.len() - path.end;
                if data == &raw[path.clone()] {
                    draws.push(template.draw.clone());
                } else {
                    let start = template.document_body + path.start;
                    let end = template.document_body + path.end;
                    let mut document = String::with_capacity(template.document.len() + data.len());
                    document.push_str(&template.document[..start]);
                    document.push_str(data);
                    document.push_str(&template.document[end..]);
                    let tree = Arc::new(Tree::from_str(&document, &Options::default()).ok()?);
                    draws.push(fragment(tree, false));
                }
            } else {
                if !svg.get(position..)?.starts_with(raw) {
                    return None;
                }
                position += raw.len();
                draws.push(template.draw.clone());
            }
            previous = template.range.end;
        }
        if svg.get(position..)? != &self.source[previous..] {
            return None;
        }
        Some(Scene {
            draws,
            layers: Arc::clone(&self.layers),
        })
    }

    pub(crate) fn scene(&self, svg: &str) -> Option<Scene> {
        if let Some(scene) = self.path_scene(svg) {
            return Some(scene);
        }
        let (root, documents) = documents(svg)?;
        if root != self.root {
            return None;
        }
        let mut draws = Vec::new();
        for (document, _, _) in documents {
            let unchanged = self.trees.contains_key(&document);
            let tree = match self.trees.get(&document) {
                Some(tree) => Arc::clone(tree),
                None => Arc::new(Tree::from_str(&document, &Options::default()).ok()?),
            };
            draws.push(fragment(tree, unchanged));
        }
        Some(Scene {
            draws,
            layers: Arc::clone(&self.layers),
        })
    }
}

fn fragment(tree: Arc<Tree>, unchanged: bool) -> Fragment {
    let bounds = tree.root().abs_layer_bounding_box();
    // Isolated layer/filter clipping depends on the complete rendering
    // context. Do not infer a smaller draw extent from a clip rectangle.
    let isolated = tree
        .root()
        .children()
        .iter()
        .any(|node| matches!(node, resvg::usvg::Node::Group(g) if g.should_isolate()));
    let (top, bottom) = if isolated {
        (f32::NEG_INFINITY, f32::INFINITY)
    } else {
        (bounds.top(), bounds.bottom())
    };
    let cache_layer = unchanged && isolated_source_over(&tree);
    Fragment {
        tree,
        top,
        bottom,
        cache_layer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(cache: &Cache, svg: &str) {
        let scene = cache.scene(svg).unwrap();
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
        let cache = Cache::new(svg).unwrap();
        for changed in [
            svg.to_owned(),
            svg.replace(" M10 90L20 90L20 100L10 100Z", ""),
            svg.replace("#2ad", "#bac"),
            svg.replace("M0 50L130 50L130 150L0 150Z", "M0 70L130 70L130 151L0 151Z"),
        ] {
            compare(&cache, &changed);
        }
        let scene = cache.scene(svg).unwrap();
        assert!(scene.draws.iter().all(|draw| cache
            .trees
            .values()
            .any(|tree| Arc::ptr_eq(tree, &draw.tree))));
    }

    #[test]
    fn clip_references_keep_inherited_attributes_of_visible_strokes() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="139" height="193"><defs><clipPath id="coverage"><use xlink:href="#ink"/></clipPath></defs><g fill-rule="evenodd"><rect width="139" height="193" fill="#abc"/></g><g id="ink-layer" fill="none" stroke-linecap="round" stroke-linejoin="round"><path id="ink" d="M14 30L120 160L20 160" stroke="#123" stroke-width="13"/></g><g clip-path="url(#coverage)"><rect width="139" height="100" fill="#f20"/></g></svg>"##;
        let cache = Cache::new(svg).unwrap();
        compare(&cache, svg);
        assert!(!cache.layers.lock().unwrap().entries.is_empty());
        compare(&cache, &svg.replace("L120 160", "L122 160"));
    }

    #[test]
    #[ignore = "reads an explicitly supplied emitted SVG"]
    fn emitted_core_fragments_match_full_renderer() {
        let path = std::env::var("PICVEC_FRAGMENT_SVG").unwrap();
        let svg = std::fs::read_to_string(path).unwrap();
        let cache = Cache::new(&svg).expect("emitted core should support cached parsing");
        let scene = cache.scene(&svg).unwrap();
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
        let cache = Cache::new(svg).unwrap();
        let changed = svg.replace(" M10 10H20V20H10Z", "");
        assert!(cache.path_scene(&changed).is_some());
        compare(&cache, &changed);
        for changed in [
            svg.replace("#f23", "#000"),
            svg.replace("evenodd", "nonzero"),
            svg.replace("fill=\"#237\"", "fill=\"#234\""),
        ] {
            assert!(cache.path_scene(&changed).is_none());
            compare(&cache, &changed);
        }
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="139" height="193"><g id="outline"><path d="M10 10H40V40H10Z" fill="#f23"/></g><use href="#outline" x="70"/></svg>"##;
        let cache = Cache::new(svg).unwrap();
        let changed = svg.replace("H40", "H50");
        assert!(cache.path_scene(&changed).is_none());
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
            assert!(Cache::new(&svg).is_none());
        }
    }
}
