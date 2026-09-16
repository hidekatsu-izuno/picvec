// XML fixtures are test inputs, never a production source of SVG structure.
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

impl From<String> for Document {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

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

impl Element {
    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        if let Some((_, v)) = self.attributes.iter_mut().find(|(k, _)| k == name) {
            *v = value.into();
        } else {
            self.attributes.push((name.into(), value.into()));
        }
    }
}
