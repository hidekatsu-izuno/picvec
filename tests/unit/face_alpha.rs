mod tests {
    use super::*;
    use crate::geometry::{Point, RegionGeometry};
    use crate::gradient::{ColorStop, LinearPreset};

    #[test]
    fn opaque_ink_edge_over_translucent_material_is_coverage() {
        let (w, h) = (40, 40);
        let mut source = Raster::blank(w, h, [0.7; 3]);
        let mut values = vec![0.6; w * h];
        let mut labels = vec![1; w * h];
        // No transparent canvas is nearby. The fractional sample mixes black
        // opaque ink with the translucent clock face in premultiplied RGBA.
        let i = 20 * w + 20;
        source.pixels[i - 1] = [0.0; 3];
        values[i - 1] = 1.0;
        labels[i - 1] = 0;
        values[i] = 0.8;
        source.pixels[i] = [0.7 * 0.6 * 0.5 / 0.8; 3];
        labels[i] = 0;
        let matte = AlphaMatte::new(w, h, values.clone());
        assert!(opaque_rim_sample(&source, &matte, i, 0, &labels, [0.0; 3]));
        // Similar alpha alone is insufficient: an incompatible colour must
        // remain a separate material, and a same-colour fade must remain alpha.
        source.pixels[i] = [0.5, 0.0, 0.0];
        assert!(!opaque_rim_sample(&source, &matte, i, 0, &labels, [0.0; 3]));
        source.pixels.fill([0.0; 3]);
        assert!(!opaque_rim_sample(&source, &matte, i, 0, &labels, [0.0; 3]));
    }

    #[test]
    fn one_owner_can_have_an_opaque_rim_and_a_translucent_interior() {
        check_rim_and_interior(Paint::Solid { color: [0.0; 3] });
    }

    #[test]
    fn gradient_paint_coverage_is_not_split_into_alpha_code_bands() {
        check_rim_and_interior(Paint::Linear {
            preset: LinearPreset::Fitted,
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 80.0, y: 48.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.0; 3],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.0; 3],
                },
            ],
        });
    }

    fn check_rim_and_interior(paint: Paint) {
        let (w, h) = (80, 48);
        let source = Raster::blank(w, h, [0.0; 3]);
        let mut values = vec![0.0; w * h];
        for x in 2..78 {
            values[5 * w + x] = 0.2 + (x % 7) as f32 * 0.1;
            for y in 6..9 {
                values[y * w + x] = 1.0;
            }
        }
        for y in 6..44 {
            for x in 68..71 {
                values[y * w + x] = 1.0;
            }
        }
        for y in 25..44 {
            for x in 30..68 {
                values[y * w + x] = 0.3;
            }
        }
        let matte = AlphaMatte::new(w, h, values);
        let mut segmentation = segment::segment(
            &source,
            &crate::edge::classify(&source),
            &crate::Config::default(),
        );
        let mut paints = vec![paint; segmentation.regions.len()];
        let alpha = prepare(&source, &matte, None, None, &mut segmentation, &mut paints);
        let rim = segmentation.labels[5 * w + 20] as usize;
        let translucent = segmentation.labels[34 * w + 45] as usize;
        assert_eq!(alpha.fields[rim], Paint::Solid { color: [1.0; 3] });
        assert!(
            (gradient::paint_at(&alpha.fields[translucent], 34 * w + 45, w)[0] - 0.3).abs()
                < 1.0 / 255.0
        );
        assert!(
            paints.len() <= 6,
            "AA isobands were retained: {}",
            paints.len()
        );
    }

    #[test]
    fn fractional_rim_can_use_opaque_ink_owned_by_the_structural_layer() {
        let (w, h) = (40, 20);
        let source = Raster::blank(w, h, [0.0; 3]);
        let mut values = vec![0.0; w * h];
        let mut labels = vec![1; w * h];
        let pixels: Vec<_> = (2..38).map(|x| 5 * w + x).collect();
        for &i in &pixels {
            labels[i] = 0;
            values[i] = 0.35;
            values[i + w] = 1.0;
            values[i + 2 * w] = 1.0;
        }
        let paint = Paint::Solid { color: [0.0; 3] };
        let matte = AlphaMatte::new(w, h, values.clone());
        let alpha = Raster::new(w, h, values.iter().map(|&a| [a; 3]).collect());
        let (_, field) =
            material_rgba_field(&source, &matte, &alpha, &pixels, 0, &labels, &paint).unwrap();
        assert_eq!(field, Paint::Solid { color: [1.0; 3] });
        for &i in &pixels {
            values[i + w] = 0.35;
            values[i + 2 * w] = 0.35;
        }
        let matte = AlphaMatte::new(w, h, values.clone());
        let alpha = Raster::new(w, h, values.iter().map(|&a| [a; 3]).collect());
        assert!(
            material_rgba_field(&source, &matte, &alpha, &pixels, 0, &labels, &paint).is_none(),
            "a genuinely translucent band must keep its own alpha"
        );
    }

    #[test]
    fn material_opacity_excludes_only_explainable_boundary_coverage() {
        let (w, h) = (34, 34);
        let source = Raster::blank(w, h, [0.3; 3]);
        let mut values = vec![0.0; w * h];
        let mut labels = vec![1; w * h];
        let mut pixels = Vec::new();
        for y in 1..33 {
            for x in 1..33 {
                let i = y * w + x;
                pixels.push(i);
                labels[i] = 0;
                values[i] = if x == 1 || x == 32 || y == 1 || y == 32 {
                    0.35
                } else {
                    0.7
                };
            }
        }
        let matte = AlphaMatte::new(w, h, values.clone());
        let alpha = Raster::new(w, h, values.iter().map(|&a| [a; 3]).collect());
        let paint = Paint::Solid { color: [0.3; 3] };
        let (_, field) =
            material_rgba_field(&source, &matte, &alpha, &pixels, 0, &labels, &paint).unwrap();
        assert!((gradient::paint_at(&field, 17 * w + 17, w)[0] - 0.7).abs() < 1.0 / 255.0);
        // An opacity line inside the same material is not boundary coverage.
        for y in 5..29 {
            values[y * w + 17] = 0.2;
        }
        let matte = AlphaMatte::new(w, h, values.clone());
        let alpha = Raster::new(w, h, values.iter().map(|&a| [a; 3]).collect());
        assert!(
            material_rgba_field(&source, &matte, &alpha, &pixels, 0, &labels, &paint).is_none()
        );
    }

    #[test]
    fn slightly_different_gradient_axes_use_few_bounded_fields() {
        let (w, h) = (64, 64);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| [0.2 + 0.6 * (i % w) as f32 / 63.0; 3])
                .collect(),
        );
        let alpha = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| [0.3 + 0.4 * (i % w) as f32 / 63.0 + 0.08 * (i / w) as f32 / 63.0; 3])
                .collect(),
        );
        let pixels: Vec<_> = (0..w * h).collect();
        let colour = gradient::fit_alpha_field(&source, &pixels).unwrap();
        let parts = local_gradient_fields(&source, &alpha, &pixels, &colour, 0).unwrap();
        assert!(parts.len() <= 8);
        for (indices, _, field) in parts {
            let mean = indices
                .iter()
                .map(|&i| (gradient::paint_at(&field, i, w)[0] - alpha.pixels[i][0]).abs())
                .sum::<f32>()
                / indices.len() as f32;
            assert!(mean <= 2.0 / 255.0);
        }
    }

    #[test]
    fn shared_colour_and_opacity_ramp_remains_one_rgba_face() {
        let (w, h) = (64, 32);
        let source = Raster::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let t = (i % w) as f32 / (w - 1) as f32;
                    [0.2 + t * 0.6, 0.3 + t * 0.2, 0.5 - t * 0.3]
                })
                .collect(),
        );
        let matte = AlphaMatte::new(
            w,
            h,
            (0..w * h)
                .map(|i| 0.2 + 0.6 * (i % w) as f32 / (w - 1) as f32)
                .collect(),
        );
        let blank = Raster::blank(w, h, [0.5; 3]);
        let mut segmentation = segment::segment(
            &blank,
            &crate::edge::classify(&blank),
            &crate::Config::default(),
        );
        let mut paints = vec![Paint::Linear {
            preset: LinearPreset::Fitted,
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: 63.0, y: 0.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: [0.2, 0.3, 0.5],
                },
                ColorStop {
                    offset: 1.0,
                    color: [0.8, 0.5, 0.2],
                },
            ],
        }];
        let alpha = prepare(&source, &matte, None, None, &mut segmentation, &mut paints);
        assert_eq!(paints.len(), 1, "must not make one face per alpha value");
        assert!(gradient::same_gradient_geometry(
            &paints[0],
            &alpha.fields[0]
        ));
        let geometry = RegionGeometry {
            region: 0,
            loops: vec![],
            path_data: "M0 0H64V32H0Z".into(),
            occlusion_path_data: None,
            covered_hole_paths: vec![],
            primitive: None,
        };
        let (svg, summary) = crate::svg::serialize_filtered_with_alpha(
            w,
            h,
            &[geometry],
            &paints,
            &crate::structural::StructuralInk::empty(),
            0.0,
            false,
            &[false],
            Some(&alpha),
        );
        assert_eq!(summary.path_elements + summary.rect_elements, 1);
        assert!(svg.contains("stop-opacity"));
        assert!(!svg.contains("<mask"));
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        for scale in [1usize, 4] {
            let mut image =
                resvg::tiny_skia::Pixmap::new((w * scale) as u32, (h * scale) as u32).unwrap();
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
                &mut image.as_mut(),
            );
            for y in 1..h * scale - 1 {
                for x in 1..(w - 1) * scale {
                    let t = (x as f32 + 0.5) / scale as f32 / 63.0;
                    let opacity = 0.2 + 0.6 * t;
                    let colour = [0.2 + t * 0.6, 0.3 + t * 0.2, 0.5 - t * 0.3];
                    let pixel = image.pixels()[y * w * scale + x];
                    assert!((pixel.alpha() as f32 - opacity * 255.0).abs() <= 3.0);
                    for (actual, expected) in [pixel.red(), pixel.green(), pixel.blue()]
                        .into_iter()
                        .zip(colour)
                    {
                        assert!((actual as f32 - expected * opacity * 255.0).abs() <= 3.0);
                    }
                }
            }
        }
    }
}
