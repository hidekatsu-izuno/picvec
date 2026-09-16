mod tests {
    use super::*;
    use crate::{
        geometry::Point,
        gradient::{ColorStop, LinearPreset},
    };

    #[test]
    fn source_gate_merges_rim_noise_but_retains_a_real_shallow_gradient() {
        let width = 48;
        let base = [50.0 / 255.0, 80.0 / 255.0, 110.0 / 255.0];
        let mut source = Raster::blank(width, width, base);
        let mut values = vec![0.0; width * width];
        let mut labels = vec![0; values.len()];
        let gradient = Paint::Linear {
            preset: LinearPreset::LeftToRight,
            start: Point { x: 12.0, y: 0.0 },
            end: Point { x: 36.0, y: 0.0 },
            stops: vec![
                ColorStop {
                    offset: 0.0,
                    color: base.map(|v| f64::from(v - 5.0 / 255.0)),
                },
                ColorStop {
                    offset: 1.0,
                    color: base.map(|v| f64::from(v + 5.0 / 255.0)),
                },
            ],
        };
        for y in 4..44 {
            for x in 4..44 {
                let i = y * width + x;
                values[i] = 1.0;
                if (12..36).contains(&x) && (12..36).contains(&y) {
                    labels[i] = 1;
                    source.pixels[i] = crate::gradient::paint_at(&gradient, i, width);
                }
            }
        }
        let matte = AlphaMatte::new(width, width, values);
        let mut paints = vec![
            Paint::Solid {
                color: base.map(|v| v - 2.0 / 255.0),
            },
            gradient.clone(),
        ];
        assert_eq!(consolidate(&matte, &source, &labels, &mut paints), 1);
        assert_eq!(paints[0], Paint::Solid { color: base });
        assert_eq!(paints[1], gradient);
    }

    #[test]
    fn opaque_input_has_no_exterior_colour_proposal() {
        let source = Raster::blank(16, 16, [0.5; 3]);
        let matte = AlphaMatte::new(16, 16, vec![1.0; 256]);
        let mut paints = vec![Paint::Solid { color: [0.49; 3] }];
        let before = paints.clone();
        assert_eq!(consolidate(&matte, &source, &[0; 256], &mut paints), 0);
        assert_eq!(paints, before);
    }
}
