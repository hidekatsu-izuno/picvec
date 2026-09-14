//! Generated SVG structure. XML is materialized only for rendering or output.
use std::{fmt, ops::Deref, sync::OnceLock};

pub(crate) type Attributes = Vec<(String, String)>;
pub(crate) fn attrs<const N: usize>(values: [(&str, String); N]) -> Attributes {
    values
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Element>,
}
impl Element {
    pub fn new(name: &str, attributes: Attributes) -> Self {
        Self {
            name: name.into(),
            attributes,
            children: Vec::new(),
        }
    }
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
    #[cfg(test)]
    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        if let Some((_, v)) = self.attributes.iter_mut().find(|(k, _)| k == name) {
            *v = value.into();
        } else {
            self.attributes.push((name.into(), value.into()));
        }
    }
    pub fn namespace_ids(&mut self, prefix: &str) {
        for (key, value) in &mut self.attributes {
            if key == "id" {
                *value = format!("{prefix}{value}");
            } else if matches!(key.as_str(), "href" | "xlink:href") && value.starts_with('#') {
                *value = format!("#{prefix}{}", &value[1..]);
            } else {
                *value = value.replace("url(#", &format!("url(#{prefix}"));
            }
        }
        for child in &mut self.children {
            child.namespace_ids(prefix);
        }
    }
    pub fn contains_name(&self, name: &str) -> bool {
        self.name == name || self.children.iter().any(|child| child.contains_name(name))
    }
    pub fn contains_attribute(&self, name: &str) -> bool {
        self.attr(name).is_some()
            || self
                .children
                .iter()
                .any(|child| child.contains_attribute(name))
    }
    fn emit(&self, out: &mut String, counts: &mut (usize, usize), hidden: bool) {
        let hidden = hidden
            || matches!(
                self.name.as_str(),
                "defs" | "clipPath" | "mask" | "symbol" | "pattern" | "marker"
            );
        if !hidden
            && matches!(
                self.name.as_str(),
                "path"
                    | "rect"
                    | "circle"
                    | "ellipse"
                    | "line"
                    | "polyline"
                    | "polygon"
                    | "use"
                    | "image"
                    | "text"
            )
        {
            counts.0 += 1;
            if self.name == "path" {
                counts.1 += self
                    .attr("d")
                    .unwrap_or_default()
                    .bytes()
                    .filter(|c| matches!(c, b'M' | b'm'))
                    .count();
            }
        }
        out.push('<');
        out.push_str(&self.name);
        for (key, value) in &self.attributes {
            out.push(' ');
            out.push_str(key);
            out.push_str("=\"");
            for c in value.chars() {
                match c {
                    '&' => out.push_str("&amp;"),
                    '<' => out.push_str("&lt;"),
                    '"' => out.push_str("&quot;"),
                    _ => out.push(c),
                }
            }
            out.push('"');
        }
        if self.children.is_empty()
            && !matches!(
                self.name.as_str(),
                "svg" | "g" | "defs" | "clipPath" | "linearGradient" | "radialGradient" | "filter"
            )
        {
            out.push_str("/>");
        } else {
            out.push('>');
            for child in &self.children {
                child.emit(out, counts, hidden);
            }
            out.push_str("</");
            out.push_str(&self.name);
            out.push('>');
        }
    }
}
impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        self.emit(&mut s, &mut (0, 0), false);
        f.write_str(&s)
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Elements {
    pub roots: Vec<Element>,
    stack: Vec<Element>,
}
impl Elements {
    pub fn new() -> Self {
        Self::default()
    }
    fn current(&mut self) -> &mut Vec<Element> {
        self.stack
            .last_mut()
            .map(|e| &mut e.children)
            .unwrap_or(&mut self.roots)
    }
    pub fn open(&mut self, name: &str, attrs: Attributes) {
        self.stack.push(Element::new(name, attrs));
    }
    pub fn close(&mut self) {
        let node = self.stack.pop().expect("balanced SVG groups");
        self.current().push(node);
    }
    pub fn leaf(&mut self, name: &str, attrs: Attributes) {
        self.current().push(Element::new(name, attrs));
    }
    pub fn append(&mut self, other: Self) {
        assert!(other.stack.is_empty());
        self.current().extend(other.roots);
    }
    pub fn wrap(self, name: &str, attrs: Attributes) -> Self {
        assert!(self.stack.is_empty());
        let mut e = Element::new(name, attrs);
        e.children = self.roots;
        Self {
            roots: vec![e],
            stack: Vec::new(),
        }
    }
    pub fn child_count(&mut self) -> usize {
        self.current().len()
    }
    pub fn replace_prefix(&mut self, count: usize, replacement: Self) {
        self.roots[0].children.splice(..count, replacement.roots);
    }
}
impl fmt::Display for Elements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(self.stack.is_empty());
        for e in &self.roots {
            write!(f, "{e}")?;
        }
        Ok(())
    }
}
#[derive(Debug)]
pub(crate) struct Document {
    root: Element,
    emitted: OnceLock<(String, (usize, usize))>,
}
impl Clone for Document {
    fn clone(&self) -> Self {
        Self::new(self.root.clone())
    }
}
impl Document {
    pub fn new(root: Element) -> Self {
        Self {
            root,
            emitted: OnceLock::new(),
        }
    }
    pub fn from_parts(width: usize, height: usize, defs: Elements, body: Elements) -> Self {
        let mut root = Element::new(
            "svg",
            attrs([
                ("xmlns", "http://www.w3.org/2000/svg".into()),
                ("width", format!("{width}")),
                ("height", format!("{height}")),
                ("viewBox", format!("0 0 {width} {height}")),
            ]),
        );
        root.children = defs.wrap("defs", vec![]).roots;
        root.children.extend(body.roots);
        Self::new(root)
    }
    pub fn root(&self) -> &Element {
        &self.root
    }
    pub fn root_mut(&mut self) -> &mut Element {
        self.emitted.take();
        &mut self.root
    }
    fn emitted(&self) -> &(String, (usize, usize)) {
        self.emitted.get_or_init(|| {
            let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
            let mut counts = (0, 0);
            self.root.emit(&mut s, &mut counts, false);
            s.push('\n');
            (s, counts)
        })
    }
    pub fn counts(&self) -> (usize, usize) {
        self.emitted().1
    }
}
impl Deref for Document {
    type Target = str;
    fn deref(&self) -> &str {
        &self.emitted().0
    }
}
impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}
impl AsRef<[u8]> for Document {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}
impl PartialEq for Document {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

// XML fixtures are test inputs, never a production source of SVG structure.
#[cfg(test)]
impl From<&str> for Document {
    fn from(s: &str) -> Self {
        fn element(n: resvg::usvg::roxmltree::Node<'_, '_>) -> Element {
            Element {
                name: n.tag_name().name().into(),
                attributes: n
                    .attributes()
                    .map(|a| {
                        (
                            if a.namespace() == Some("http://www.w3.org/1999/xlink") {
                                format!("xlink:{}", a.name())
                            } else {
                                a.name().into()
                            },
                            a.value().into(),
                        )
                    })
                    .collect(),
                children: n
                    .children()
                    .filter(|n| n.is_element())
                    .map(element)
                    .collect(),
            }
        }
        let dom = resvg::usvg::roxmltree::Document::parse(s).unwrap();
        let mut root = element(dom.root_element());
        root.set("xmlns", "http://www.w3.org/2000/svg");
        root.set("xmlns:xlink", "http://www.w3.org/1999/xlink");
        Self::new(root)
    }
}
#[cfg(test)]
impl From<String> for Document {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emission_counts_drawables_and_contours_after_structural_edits() {
        let mut definitions = Elements::new();
        definitions.leaf(
            "path",
            attrs([("id", "source".into()), ("d", "M0 0H10V10Z".into())]),
        );
        definitions.open("clipPath", attrs([("id", "clip".into())]));
        definitions.leaf(
            "rect",
            attrs([("width", "10".into()), ("height", "10".into())]),
        );
        definitions.close();
        let mut body = Elements::new();
        body.open("g", attrs([("clip-path", "url(#clip)".into())]));
        body.leaf(
            "path",
            attrs([
                ("d", "M0 0H10V10Z m2 2h3v3z".into()),
                ("fill-opacity", "0.25".into()),
            ]),
        );
        body.leaf("use", attrs([("href", "#source".into())]));
        body.close();
        let mut document = Document::from_parts(10, 10, definitions, body);
        assert_eq!(document.counts(), (2, 2));
        let before = document.to_string();
        document.root_mut().children[1].children.remove(0);
        assert_eq!(document.counts(), (1, 0));
        assert_ne!(document.to_string(), before);
        let dom = resvg::usvg::roxmltree::Document::parse(&document).unwrap();
        assert_eq!(
            dom.descendants().filter(|n| n.has_tag_name("use")).count(),
            1
        );
    }

    #[test]
    fn attributes_are_escaped_once_and_id_references_follow_nested_elements() {
        let mut root = Element::new("svg", vec![]);
        root.set("xmlns", "http://www.w3.org/2000/svg");
        let mut group = Element::new("g", attrs([("id", "source".into())]));
        group.set("data-note", "a < b & \"c\"");
        group.children.push(Element::new(
            "use",
            attrs([
                ("href", "#shape".into()),
                ("clip-path", "url(#clip)".into()),
            ]),
        ));
        root.children.push(group);
        root.namespace_ids("child-");
        let document = Document::new(root);
        let dom = resvg::usvg::roxmltree::Document::parse(&document).unwrap();
        let group = dom.descendants().find(|n| n.has_tag_name("g")).unwrap();
        assert_eq!(group.attribute("data-note"), Some("a < b & \"c\""));
        assert_eq!(group.attribute("id"), Some("child-source"));
        let node = group.first_element_child().unwrap();
        assert_eq!(node.attribute("href"), Some("#child-shape"));
        assert_eq!(node.attribute("clip-path"), Some("url(#child-clip)"));
    }
}
