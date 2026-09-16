mod tests {
    use super::*;

    #[test]
    fn tree_matches_exhaustive_segments_bit_for_bit() {
        for count in [0, 1, 2, 9, 33, 257] {
            for origin in [0.0, -300.25, 1_000_000.0] {
                let points: Vec<_> = (0..count)
                    .map(|i| Point {
                        x: origin + ((i * 37) % 101) as f32 * 0.25,
                        y: origin + ((i * 19) % 61) as f32 * 0.125,
                    })
                    .collect();
                let tree = SegmentIndex::new(&points);
                for i in 0..300 {
                    let p = Point {
                        x: origin - 3.0 + i as f32 * 0.1,
                        y: origin + (i % 21) as f32 * 0.7,
                    };
                    // Keep the original expression as an independent oracle.
                    let expected = points
                        .windows(2)
                        .map(|pair| {
                            let dx = pair[1].x - pair[0].x;
                            let dy = pair[1].y - pair[0].y;
                            let t = (((p.x - pair[0].x) * dx + (p.y - pair[0].y) * dy)
                                / (dx * dx + dy * dy).max(1e-12))
                            .clamp(0.0, 1.0);
                            p.distance(Point {
                                x: pair[0].x + dx * t,
                                y: pair[0].y + dy * t,
                            })
                        })
                        .fold(f32::INFINITY, f32::min);
                    assert_eq!(tree.distance(p).to_bits(), expected.to_bits());
                }
            }
        }
    }
}
