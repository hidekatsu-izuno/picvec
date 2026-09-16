mod tests {
    use super::*;

    #[test]
    fn antialias_at_a_displaced_stroke_edge_is_not_an_interior_colour() {
        let paint = Raster::new(24, 16, vec![[0.9, 0.8, 0.7]; 384]);
        let mut source = paint.clone();
        let mut before = paint.clone();
        let mut white = paint.clone();
        let mut black = paint.clone();
        for x in 3..21 {
            for y in 6..10 {
                let i = y * 24 + x;
                source.pixels[i] = [0.05; 3];
                before.pixels[i] = [0.05; 3];
                white.pixels[i] = [1.0; 3];
                black.pixels[i] = [0.0; 3];
            }
            // The other incident paint differs from the paint underlay.
            // This 70%-covered edge consequently fails the two-colour fit,
            // despite not being an independent mark inside the source ink.
            source.pixels[6 * 24 + x] = [0.065, 0.275, 0.335];
            source.pixels[5 * 24 + x] = [0.1, 0.8, 1.0];
        }
        assert!(propose(&source, &paint, &before, &white, &black).is_empty());
    }

    #[test]
    fn darker_ink_on_the_same_colour_axis_is_not_a_patch_material() {
        let paint = Raster::new(12, 8, vec![[0.6; 3]; 96]);
        let before = Raster::new(12, 8, vec![[0.12; 3]; 96]);
        let white = Raster::new(12, 8, vec![[1.0; 3]; 96]);
        let black = Raster::new(12, 8, vec![[0.0; 3]; 96]);
        let mut source = before.clone();
        // Raster phase alternates between a dark core and partial coverage.
        // A brighter uniform stroke is a colour-estimation error, not a
        // connected secondary material to be traced as pixel rectangles.
        for x in 2..10 {
            source.pixels[3 * 12 + x] = [0.01; 3];
            source.pixels[4 * 12 + x] = [0.04; 3];
        }
        assert!(propose(&source, &paint, &before, &white, &black).is_empty());
        for x in 3..8 {
            source.pixels[3 * 12 + x] = [0.02, 0.04, 0.4];
        }
        assert!(!propose(&source, &paint, &before, &white, &black).is_empty());
    }

    #[test]
    fn interior_colours_are_distinct_from_coverage_and_isolated_noise() {
        for vertical in [false, true] {
            for color in [[0.9, 0.05, 0.1], [0.05, 0.2, 0.9], [0.1, 0.8, 0.2]] {
                let paint = Raster::new(24, 24, vec![[0.9, 0.8, 0.7]; 576]);
                let mut source = paint.clone();
                let mut before = paint.clone();
                let mut white = paint.clone();
                let mut black = paint.clone();
                let index = |t: usize, n: usize| if vertical { t * 24 + n } else { n * 24 + t };
                for t in 3..21 {
                    for n in 10..14 {
                        let i = index(t, n);
                        source.pixels[i] = [0.05; 3];
                        before.pixels[i] = [0.05; 3];
                        white.pixels[i] = [1.0; 3];
                        black.pixels[i] = [0.0; 3];
                    }
                }
                for t in 8..13 {
                    source.pixels[index(t, 11)] = color;
                }
                // A lone mismatching pixel is insufficient region evidence.
                source.pixels[index(20, 12)] = color;
                // A coverage mixture of the existing paint and ink is not a
                // third material, even when it differs strongly from the line.
                source.pixels[index(4, 11)] = [0.475, 0.425, 0.375];
                source.pixels[index(5, 11)] = [0.475, 0.425, 0.375];
                source.pixels[index(15, 12)] = [0.85, 0.78, 0.8];
                source.pixels[index(16, 12)] = [0.85, 0.78, 0.8];
                let patches = propose(&source, &paint, &before, &white, &black);
                assert_eq!(patches.len(), 1);
                assert_eq!(patches[0].pixels.len(), 5);
                let mut after = before.clone();
                for &i in &patches[0].pixels {
                    after.pixels[i] = patches[0].color;
                }
                assert!(patches[0].improves(&source, &before, &after));
                assert!(!patches[0].improves(&source, &before, &before));
                // A distinct shade inside a supported material is not an
                // isolated noise pixel and must not disappear during grouping.
                source.pixels[index(10, 11)] = [0.45, 0.15, 0.4];
                let shaded = propose(&source, &paint, &before, &white, &black);
                assert_eq!(shaded.iter().map(|p| p.pixels.len()).sum::<usize>(), 5);
                // Outside actual stroke coverage, the same colour is paint's
                // responsibility and cannot trigger a repair overlay.
                for &i in &patches[0].pixels {
                    white.pixels[i] = black.pixels[i];
                }
                assert!(propose(&source, &paint, &before, &white, &black).is_empty());
            }
        }
    }
}
