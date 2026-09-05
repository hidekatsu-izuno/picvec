//! Ordered correspondence between raster graph nodes and a continuous master.
//! Equal fractions of raster arclength and Bezier parameter are not equivalent:
//! raster backtracking and nonuniform cubic speed otherwise move junctions off
//! their local source support.

use super::{cubic_point, sample_curve_sequence, AdaptiveCurveSpan, CurveSegment, Point};
use std::collections::HashMap;

pub(super) struct Mapping {
    pub positions: Vec<Point>,
    pub edges: Vec<Vec<AdaptiveCurveSpan>>,
}

pub(super) fn map(
    source: &[Point],
    curves: &[CurveSegment],
    corridor: f32,
    next_master: &mut usize,
) -> Option<Mapping> {
    if source.len() < 2 || curves.is_empty() {
        return None;
    }
    let mut samples = vec![(curves[0].start(), 0.0_f64, 0.0_f64)];
    for (i, curve) in curves.iter().enumerate() {
        let points = sample_curve_sequence(&[*curve], 0.25);
        for (j, &point) in points.iter().enumerate().skip(1) {
            let previous = samples.last().unwrap();
            let length = previous.2 + previous.0.distance(point) as f64;
            samples.push((
                point,
                i as f64 + j as f64 / (points.len() - 1) as f64,
                length,
            ));
        }
    }
    let total = samples.last()?.2;
    if total <= 1e-6 {
        return None;
    }
    const CELL: f32 = 2.0;
    let cell = |p: Point| ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32);
    let mut spatial = HashMap::<(i32, i32), Vec<usize>>::new();
    for (i, pair) in samples.windows(2).enumerate() {
        let middle = super::interpolate_point(pair[0].0, pair[1].0, 0.5);
        spatial.entry(cell(middle)).or_default().push(i);
    }
    let radius = (corridor / CELL).ceil() as i32 + 1;
    let mut distances = Vec::with_capacity(source.len());
    for &point in source {
        let (cx, cy) = cell(point);
        let mut best = (f32::INFINITY, 0.0_f64);
        for y in cy - radius..=cy + radius {
            for x in cx - radius..=cx + radius {
                for &i in spatial.get(&(x, y)).into_iter().flatten() {
                    let (a, _, start) = samples[i];
                    let (b, _, end) = samples[i + 1];
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let t = (((point.x - a.x) * dx + (point.y - a.y) * dy)
                        / (dx * dx + dy * dy).max(1e-12))
                    .clamp(0.0, 1.0);
                    let nearest = Point {
                        x: a.x + t * dx,
                        y: a.y + t * dy,
                    };
                    let error = point.distance(nearest);
                    let distance = start + t as f64 * (end - start);
                    if error < best.0 || (error == best.0 && distance < best.1) {
                        best = (error, distance);
                    }
                }
            }
        }
        if best.0 > corridor {
            return None;
        }
        distances.push(best.1);
    }
    if source.first() == source.last() && curves[0].start() == curves.last()?.end() {
        // The first vertex of a closed raster loop is only a storage seam.
        // A one-pixel backward step there projects near the end of the same
        // ellipse. Unwrap before isotonic pooling, otherwise that harmless
        // step pools an entire loop across the artificial parameter break.
        let mut previous = 0.0;
        for distance in &mut distances {
            *distance += ((previous - *distance) / total).round() * total;
            previous = *distance;
            *distance = distance.clamp(0.0, total);
        }
    }
    distances[0] = 0.0;
    *distances.last_mut()? = total;
    // Isotonic regression pools raster excursions instead of following them
    // backwards along the master. It preserves the ordering of every half-edge.
    let mut blocks = Vec::<(usize, usize, f64)>::new();
    for (i, &distance) in distances.iter().enumerate() {
        blocks.push((i, i + 1, distance));
        while blocks.len() >= 2 {
            let b = blocks[blocks.len() - 1];
            let a = blocks[blocks.len() - 2];
            if a.2 / (a.1 - a.0) as f64 <= b.2 / (b.1 - b.0) as f64 {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push((a.0, b.1, a.2 + b.2));
        }
    }
    for (start, end, sum) in blocks {
        distances[start..end].fill(sum / (end - start) as f64);
    }
    let parameters: Vec<_> = distances
        .iter()
        .map(|&distance| {
            let index = samples
                .partition_point(|sample| sample.2 < distance)
                .clamp(1, samples.len() - 1);
            let (_, first, start) = samples[index - 1];
            let (_, second, end) = samples[index];
            first
                + (second - first) * ((distance - start) / (end - start).max(1e-12)).clamp(0.0, 1.0)
        })
        .collect();
    let positions: Vec<_> = parameters
        .iter()
        .map(|&parameter| {
            let index = (parameter.floor() as usize).min(curves.len() - 1);
            cubic_point(curves[index], (parameter - index as f64) as f32)
        })
        .collect();
    if positions
        .iter()
        .zip(source)
        .any(|(&a, &b)| a.distance(b) > corridor)
    {
        return None;
    }
    let master_start = *next_master;
    *next_master += curves.len();
    let edges = parameters
        .windows(2)
        .map(|pair| {
            let first = (pair[0].floor() as usize).min(curves.len() - 1);
            let last = (pair[1].floor() as usize).min(curves.len() - 1);
            let mut spans = Vec::new();
            for (index, &curve) in curves.iter().enumerate().take(last + 1).skip(first) {
                let start_parameter = (pair[0] - index as f64).clamp(0.0, 1.0);
                let end_parameter = (pair[1] - index as f64).clamp(0.0, 1.0);
                if end_parameter > start_parameter || pair[0] == pair[1] {
                    spans.push(AdaptiveCurveSpan {
                        master_id: master_start + index,
                        curve,
                        start_parameter,
                        end_parameter,
                    });
                }
            }
            spans
        })
        .collect();
    Some(Mapping { positions, edges })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_nodes_follow_position_not_uniform_bezier_parameter() {
        let curve = CurveSegment::Cubic {
            start: Point { x: 0.0, y: 0.0 },
            first: Point { x: 0.1, y: 0.1 },
            second: Point { x: 0.2, y: 0.2 },
            end: Point { x: 20.0, y: 20.0 },
        };
        let source: Vec<_> = (0..=20)
            .map(|i| Point {
                x: i as f32,
                y: i as f32,
            })
            .collect();
        let mapping = map(&source, &[curve], 0.1, &mut 0).unwrap();
        assert!(mapping
            .positions
            .iter()
            .zip(&source)
            .all(|(&a, &b)| a.distance(b) < 0.01));
        assert!(mapping
            .edges
            .iter()
            .flatten()
            .all(|span| span.master_id == 0));
        assert_eq!(mapping.positions[0], source[0]);
        assert_eq!(*mapping.positions.last().unwrap(), *source.last().unwrap());
    }

    #[test]
    fn raster_backtracking_is_pooled_without_reversing_the_shared_master() {
        let source = [0.0, 5.0, 4.5, 10.0, 20.0].map(|x| Point { x, y: x });
        let curve = CurveSegment::Line {
            start: source[0],
            end: source[4],
        };
        let mapping = map(&source, &[curve], 0.5, &mut 0).unwrap();
        assert_eq!(mapping.positions[1], mapping.positions[2]);
        assert!(mapping
            .positions
            .windows(2)
            .all(|pair| pair[0].x <= pair[1].x));
        assert!(mapping
            .edges
            .iter()
            .flatten()
            .all(|span| span.start_parameter <= span.end_parameter));
        assert!(
            map(&source, &[curve], 0.1, &mut 0).is_none(),
            "an unsupported correspondence must reject the whole fit"
        );
    }
}
