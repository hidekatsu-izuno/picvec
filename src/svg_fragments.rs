//! Reuse unchanged parsed draw operations during covered-hole trials.
//! Only inert groups are split. Clips, filters and transforms stay attached to
//! their entire group. The final hole check still uses the full SVG renderer.
use crate::svg_document::Document;
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

#[derive(Clone, PartialEq, Eq, Hash)]
struct DrawSource {
    root: Vec<(String, String)>,
    wrappers: Vec<(String, Vec<(String, String)>)>,
    resources: Vec<crate::svg_document::Element>,
    element: crate::svg_document::Element,
}
impl DrawSource {
    fn document(&self) -> Document {
        use crate::svg_document::Element;
        let mut element = self.element.clone();
        for (name, attributes) in self.wrappers.iter().rev() {
            element = Element {
                name: name.clone(),
                attributes: attributes.clone(),
                children: vec![element],
            };
        }
        Document::new(Element {
            name: "svg".into(),
            attributes: self.root.clone(),
            children: vec![
                Element {
                    name: "defs".into(),
                    attributes: vec![],
                    children: self.resources.clone(),
                },
                element,
            ],
        })
    }
}

pub(crate) struct Cache {
    root: Vec<(String, String)>,
    trees: HashMap<DrawSource, Arc<Tree>>,
    layers: Arc<Mutex<Layers>>,
}

fn refs(node: &crate::svg_document::Element, ids: &mut BTreeSet<String>) -> Option<()> {
    for (name, value) in &node.attributes {
        let mut value = value.as_str();
        if matches!(name.as_str(), "href" | "xlink:href") {
            ids.insert(value.strip_prefix('#')?.to_owned());
        }
        while let Some(start) = value.find("url(") {
            value = &value[start + 4..];
            let end = value.find(')')?;
            ids.insert(
                value[..end]
                    .trim()
                    .trim_matches(['\'', '"'])
                    .strip_prefix('#')?
                    .to_owned(),
            );
            value = &value[end + 1..];
        }
    }
    for child in &node.children {
        refs(child, ids)?;
    }
    Some(())
}

fn inert(node: &crate::svg_document::Element) -> bool {
    node.name == "g"
        && node.attributes.iter().all(|(name, value)| {
            matches!(
                name.as_str(),
                "id" | "fill-rule" | "fill" | "stroke-linecap" | "stroke-linejoin"
            ) && !value.contains("url(")
        })
}

fn sources(svg: &Document) -> Option<Vec<DrawSource>> {
    use crate::svg_document::Element;
    // Index borrowed resources first. Named paint/ink groups can contain most
    // of the drawing, yet are often never referenced. Cloning them eagerly
    // duplicates the whole scene on every trial. Materialize only actual refs.
    struct Resource<'a> {
        node: &'a Element,
        wrappers: Vec<&'a Element>,
    }
    impl Resource<'_> {
        fn materialize(&self) -> Element {
            let mut resource = self.node.clone();
            for parent in &self.wrappers {
                resource = Element {
                    name: parent.name.clone(),
                    attributes: parent.attributes.clone(),
                    children: vec![resource],
                };
            }
            resource
        }
    }
    fn contains(node: &Element, names: &[&str]) -> bool {
        names.contains(&node.name.as_str()) || node.children.iter().any(|n| contains(n, names))
    }
    if svg.root().name != "svg"
        || contains(&svg.root(), &["style"])
        || svg
            .root()
            .attributes
            .iter()
            .any(|(_, v)| v.contains("url("))
    {
        return None;
    }
    // Definitions and their inherited context come directly from generated
    // elements. Never parse the emitted XML to recover ownership or references.
    fn index<'a>(
        node: &'a Element,
        ancestors: &mut Vec<&'a Element>,
        defs: bool,
        out: &mut HashMap<&'a str, Option<Resource<'a>>>,
    ) {
        let defs = defs || node.name == "defs";
        if let Some(id) = node.attr("id") {
            let resource = if defs {
                Some(Resource {
                    node,
                    wrappers: Vec::new(),
                })
            } else if ancestors.iter().all(|p| p.name == "svg" || inert(p)) {
                Some(Resource {
                    node,
                    wrappers: ancestors
                        .iter()
                        .rev()
                        .take_while(|p| p.name != "svg")
                        .copied()
                        .collect(),
                })
            } else {
                None
            };
            out.insert(id, resource);
        }
        ancestors.push(node);
        for child in &node.children {
            index(child, ancestors, defs, out);
        }
        ancestors.pop();
    }
    let mut resources = HashMap::new();
    index(&svg.root(), &mut Vec::new(), false, &mut resources);
    fn visit(
        node: &Element,
        root: &Element,
        wrappers: &mut Vec<(String, Vec<(String, String)>)>,
        resources: &HashMap<&str, Option<Resource<'_>>>,
        out: &mut Vec<DrawSource>,
    ) -> Option<()> {
        if node.name == "defs" {
            return Some(());
        }
        if inert(node) {
            wrappers.push((node.name.clone(), node.attributes.clone()));
            for child in &node.children {
                visit(child, root, wrappers, resources, out)?;
            }
            wrappers.pop();
            return Some(());
        }
        if contains(node, &["svg", "defs"]) {
            return None;
        }
        let mut needed = BTreeSet::new();
        refs(node, &mut needed)?;
        let mut done = BTreeSet::new();
        while let Some(id) = needed.iter().find(|id| !done.contains(*id)).cloned() {
            // Eligible wrappers are inert and contain no referenced paints;
            // all transitive references are in the borrowed node itself.
            refs(resources.get(id.as_str())?.as_ref()?.node, &mut needed)?;
            done.insert(id);
        }
        out.push(DrawSource {
            root: root.attributes.clone(),
            wrappers: wrappers.clone(),
            resources: needed
                .into_iter()
                .map(|id| resources[id.as_str()].as_ref().unwrap().materialize())
                .collect(),
            element: node.clone(),
        });
        Some(())
    }
    let mut out = Vec::new();
    for child in &svg.root().children {
        visit(child, &svg.root(), &mut Vec::new(), &resources, &mut out)?;
    }
    Some(out)
}
impl Cache {
    pub(crate) fn new(svg: &Document) -> Option<Self> {
        let sources = sources(svg)?;
        let mut trees = HashMap::new();
        let mut bytes = 0;
        for source in sources {
            if trees.contains_key(&source) {
                continue;
            }
            let document = source.document();
            bytes += document.len();
            if bytes > 64 * 1024 * 1024 {
                return None;
            }
            let tree = Arc::new(Tree::from_str(&document, &Options::default()).ok()?);
            trees.insert(source, tree);
        }
        Some(Self {
            root: svg.root().attributes.clone(),
            trees,
            layers: Arc::new(Mutex::new(Layers::default())),
        })
    }
    pub(crate) fn scene(&self, svg: &Document) -> Option<Scene> {
        if svg.root().attributes != self.root {
            return None;
        }
        let mut draws = Vec::new();
        for source in sources(svg)? {
            // Hashing and comparing the full source is substantial for large
            // paths and definitions. Obtain both results with one lookup.
            let (tree, unchanged) = match self.trees.get(&source) {
                Some(tree) => (Arc::clone(tree), true),
                None => (
                    Arc::new(Tree::from_str(&source.document(), &Options::default()).ok()?),
                    false,
                ),
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
