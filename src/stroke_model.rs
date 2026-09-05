//! Recover a narrow, uniform ink band jointly with its two incident paints.
//! A step edge or a broad shadow cannot supply two supported band edges.

use super::{Point, StructuralStroke};
use crate::edge::SourceEdge;
use crate::raster::Raster;
use rayon::prelude::*;

type PaintUpdate = (usize, [f32; 3]);
type StrokeCandidate = (StructuralStroke, Vec<PaintUpdate>);

struct Profile {
    center: Point,
    normal: Point,
    width: f32,
    ink: [f32; 3],
    sides: [[f32; 3]; 2],
}

pub(super) struct Recovery {
    pub strokes: Vec<StructuralStroke>,
    pub updates: Vec<PaintUpdate>,
    pub mask: Vec<bool>,
}

fn sample(image: &Raster, p: Point) -> [f32; 3] {
    let x = (p.x - 0.5).clamp(0.0, (image.width - 1) as f32);
    let y = (p.y - 0.5).clamp(0.0, (image.height - 1) as f32);
    let ix = x as usize;
    let iy = y as usize;
    let tx = x - ix as f32;
    let ty = y - iy as f32;
    std::array::from_fn(|c| {
        let a = image.pixels[iy * image.width + ix][c];
        let b = image.pixels[iy * image.width + (ix + 1).min(image.width - 1)][c];
        let d = image.pixels[(iy + 1).min(image.height - 1) * image.width + ix][c];
        let e = image.pixels
            [(iy + 1).min(image.height - 1) * image.width + (ix + 1).min(image.width - 1)][c];
        (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (d * (1.0 - tx) + e * tx) * ty
    })
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
fn offset(p: Point, n: Point, d: f32) -> Point {
    Point {
        x: p.x + n.x * d,
        y: p.y + n.y * d,
    }
}

fn profile(image: &Raster, edge: &SourceEdge, i: usize) -> Option<Profile> {
    let points = &edge.points;
    let closed = points.first() == points.last();
    let n = points.len() - usize::from(closed);
    let (before, after) = if closed {
        ((i + n - 3) % n, (i + 3) % n)
    } else {
        (i.saturating_sub(3), (i + 3).min(n - 1))
    };
    let a = points[before];
    let b = points[after];
    let length = (b[0] - a[0]).hypot(b[1] - a[1]) as f32;
    if length < 0.5 {
        return None;
    }
    let normal = Point {
        x: (a[1] - b[1]) as f32 / length,
        y: (b[0] - a[0]) as f32 / length,
    };
    let p = Point {
        x: points[i][0] as f32,
        y: points[i][1] as f32,
    };
    // A dark-boundary seed lies near one side of the band. Its provisional
    // overlay width is not the band width, so search far enough to see both.
    let scale = (image.width.max(image.height) as f32 / 1024.0).max(1.0);
    let reach = (edge.width as f32 + 2.5).max(12.0 * scale).min(24.0);
    let count = (reach * 2.0).ceil() as isize;
    let colors: Vec<_> = (-count..=count)
        .map(|k| sample(image, offset(p, normal, k as f32 * 0.5)))
        .collect();
    let middle = count as usize;
    let search = ((edge.width as f32 * 0.5 + 0.75) * 2.0).ceil() as usize;
    let core = (middle.saturating_sub(search)..=(middle + search).min(colors.len() - 1)).min_by(
        |&a, &b| {
            let score = |j: usize| luma(colors[j]) + 0.001 * (j as f32 - middle as f32).powi(2);
            score(a).total_cmp(&score(b))
        },
    )?;
    let ink = colors[core];
    let sides = [colors[0], *colors.last()?];
    let dark = luma(ink);
    if sides.iter().any(|&c| luma(c) - dark < 0.12) {
        return None;
    }
    // Require an actual flat ink core, not the minimum of a smooth shadow.
    if core == 0
        || core + 1 >= colors.len()
        || (luma(colors[core - 1]) - dark).max(luma(colors[core + 1]) - dark) > 0.06
    {
        return None;
    }
    let threshold = [0.5 * (dark + luma(sides[0])), 0.5 * (dark + luma(sides[1]))];
    let mut left = core;
    while left > 0 && luma(colors[left]) < threshold[0] {
        left -= 1;
    }
    let mut right = core;
    while right + 1 < colors.len() && luma(colors[right]) < threshold[1] {
        right += 1;
    }
    if left == 0 || right + 1 == colors.len() {
        return None;
    }
    let crossing = |a: usize, b: usize, threshold: f32| {
        let t = (threshold - luma(colors[a])) / (luma(colors[b]) - luma(colors[a]));
        (a as f32 + t * (b as f32 - a as f32) - middle as f32) * 0.5
    };
    let low = crossing(left, left + 1, threshold[0]);
    let high = crossing(right - 1, right, threshold[1]);
    let width = high - low;
    if !(1.5..=16.0).contains(&width) {
        return None;
    }
    let center = offset(p, normal, 0.5 * (low + high));
    let contrast = (luma(sides[0]) - dark).min(luma(sides[1]) - dark);
    if [-1.0_f32, 1.0].iter().any(|&sign| {
        luma(sample(
            image,
            offset(center, normal, sign * (0.25 * width - 0.5).max(0.25)),
        )) - dark
            > 0.05 * contrast
    }) {
        return None;
    }
    Some(Profile {
        center,
        normal,
        width,
        ink,
        sides,
    })
}

fn candidate(image: &Raster, profiles: Vec<Profile>) -> Option<StrokeCandidate> {
    if profiles.len() < 12 {
        return None;
    }
    let mut widths: Vec<_> = profiles.iter().map(|p| p.width).collect();
    widths.sort_by(f32::total_cmp);
    let width = widths[widths.len() / 2];
    if widths[widths.len() * 9 / 10] - widths[widths.len() / 10] > (0.25 * width).max(0.7) {
        return None;
    }
    let color = std::array::from_fn(|c| {
        let mut values: Vec<_> = profiles.iter().map(|p| p.ink[c]).collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    });
    if profiles
        .iter()
        .filter(|p| distance(p.ink, color) > 0.12)
        .count()
        * 10
        > profiles.len()
    {
        return None;
    }
    let points: Vec<_> = profiles.iter().map(|p| p.center).collect();
    let length: f32 = points.windows(2).map(|p| p[0].distance(p[1])).sum();
    if length < (6.0 * width).max(16.0) {
        return None;
    }
    let mut updates = std::collections::BTreeMap::new();
    for (i, p) in profiles.iter().enumerate() {
        // Diagonal samples are farther apart than horizontal samples. Cover
        // the interval between centres so original ink cannot survive in gaps.
        let previous = profiles[i.saturating_sub(1)].center;
        let next = profiles[(i + 1).min(profiles.len() - 1)].center;
        let half_step =
            (0.5 * p.center.distance(previous).max(p.center.distance(next)) + 0.1).max(0.6);
        let radius = (0.5 * p.width + 1.25).ceil() as isize;
        let x = p.center.x.floor() as isize;
        let y = p.center.y.floor() as isize;
        for py in (y - radius).max(0)..=(y + radius).min(image.height as isize - 1) {
            for px in (x - radius).max(0)..=(x + radius).min(image.width as isize - 1) {
                let delta = Point {
                    x: px as f32 + 0.5 - p.center.x,
                    y: py as f32 + 0.5 - p.center.y,
                };
                let across = delta.x * p.normal.x + delta.y * p.normal.y;
                let along = -delta.x * p.normal.y + delta.y * p.normal.x;
                if across.abs() > 0.5 * p.width + 1.0 || along.abs() > half_step {
                    continue;
                }
                let paint = p.sides[usize::from(across >= 0.0)];
                let index = py as usize * image.width + px as usize;
                let observed = image.pixels[index];
                let direction: [f32; 3] = std::array::from_fn(|c| color[c] - paint[c]);
                let denominator = direction.iter().map(|x| x * x).sum::<f32>().max(1e-6);
                let alpha = (0..3)
                    .map(|c| (observed[c] - paint[c]) * direction[c])
                    .sum::<f32>()
                    / denominator;
                let predicted = std::array::from_fn(|c| paint[c] + alpha * direction[c]);
                if alpha > 0.04 && alpha <= 1.15 && distance(predicted, observed) < 0.10 {
                    updates.entry(index).or_insert(paint);
                }
            }
        }
    }
    if (updates.len() as f32) < length * width * 0.6 {
        return None;
    }
    Some((
        StructuralStroke {
            points,
            path_data: None,
            precise_points: None,
            color,
            width,
            role: "boundary-stroke",
            width_samples: vec![(width, profiles.len())],
        },
        updates.into_iter().collect(),
    ))
}

pub(super) fn recover(image: &Raster, edges: &[SourceEdge]) -> Recovery {
    let candidates: Vec<_> = edges
        .par_iter()
        .flat_map_iter(|edge| {
            let mut candidates = Vec::new();
            if edge.points.len() < 12 || !matches!(edge.role, "ridge-on-boundary" | "dark-boundary")
            {
                return candidates;
            }
            let mut run = Vec::new();
            for i in 0..edge.points.len() {
                if let Some(p) = profile(image, edge, i) {
                    run.push(p);
                } else {
                    if let Some(candidate) = candidate(image, std::mem::take(&mut run)) {
                        candidates.push(candidate);
                    }
                }
            }
            if let Some(candidate) = candidate(image, run) {
                candidates.push(candidate);
            }
            candidates
        })
        .collect();
    let mut result = Recovery {
        strokes: Vec::new(),
        updates: Vec::new(),
        mask: vec![false; image.pixels.len()],
    };
    for (stroke, updates) in candidates {
        // Opposite detected edges can describe the same ink band.
        if updates.iter().filter(|(i, _)| result.mask[*i]).count() * 4 > updates.len() {
            continue;
        }
        for &(i, _) in &updates {
            result.mask[i] = true;
        }
        result.updates.extend(updates);
        result.strokes.push(stroke);
    }
    result
}

// A narrow colour band next to transparency has only one incident Paint.
// Recover its coverage along the alpha contour instead of quantizing each
// antialiased edge pixel into an independent face.
pub(super) fn recover_alpha_boundary(
    image: &Raster,
    matte: &crate::chroma::AlphaMatte,
) -> Vec<StructuralStroke> {
    let mut strokes = Vec::new();
    // Authored partial-alpha areas have a different coverage model.
    if (0..matte.len()).any(|i| {
        let a = matte.get(i);
        a > 0.0 && a < 1.0
    }) {
        return strokes;
    }
    for contour in matte.isocontours(0.5) {
        let spans = crate::geometry::alpha_contour_spans(&contour);
        if spans.len() < 8 {
            continue;
        }
        let coverage = |p: Point| {
            let x = p.x.floor().clamp(0.0, (image.width - 1) as f32) as usize;
            let y = p.y.floor().clamp(0.0, (image.height - 1) as f32) as usize;
            matte.get(y * image.width + x)
        };
        let mut observations = Vec::new();
        for span in &spans {
            let p = span.points[1];
            let a = span.points[0];
            let b = span.points[2];
            let length = a.distance(b).max(1e-6);
            let mut normal = Point {
                x: (a.y - b.y) / length,
                y: (b.x - a.x) / length,
            };
            if coverage(offset(p, normal, 2.0)) < coverage(offset(p, normal, -2.0)) {
                normal.x = -normal.x;
                normal.y = -normal.y;
            }
            if coverage(offset(p, normal, 3.0)) < 1.0 || coverage(offset(p, normal, -2.0)) > 0.0 {
                observations.push(None);
                continue;
            }
            let paint = sample(image, offset(p, normal, 3.5));
            let peak = (0..=6)
                .map(|i| sample(image, offset(p, normal, i as f32 * 0.25)))
                .max_by(|a, b| luma(*a).total_cmp(&luma(*b)))
                .unwrap();
            observations.push(Some((normal, paint, peak)));
        }
        let count = spans.len();
        let mut widths = vec![0.0; count];
        let mut inks = vec![[0.0; 3]; count];
        for i in 0..count {
            let Some((normal, paint, _)) = observations[i] else {
                continue;
            };
            let ink = (-3..=3)
                .filter_map(|d| observations[(i as isize + d).rem_euclid(count as isize) as usize])
                .map(|(_, _, peak)| peak)
                .max_by(|a, b| luma(*a).total_cmp(&luma(*b)))
                .unwrap();
            let contrast = luma(ink) - luma(paint);
            if contrast < 0.06
                || distance(
                    sample(image, offset(spans[i].points[1], normal, 2.5)),
                    sample(image, offset(spans[i].points[1], normal, 4.5)),
                ) > 0.5 * distance(ink, paint)
            {
                continue;
            }
            let width: f32 = (0..12)
                .map(|j| {
                    let observed = sample(
                        image,
                        offset(spans[i].points[1], normal, (j as f32 + 0.5) * 0.25),
                    );
                    ((luma(observed) - luma(paint)) / contrast).clamp(0.0, 1.0) * 0.25
                })
                .sum();
            inks[i] = ink;
            if width > 0.05 && width < 1.8 {
                widths[i] = width;
            }
        }
        for i in 0..count {
            let Some((_, paint, _)) = observations[i] else {
                continue;
            };
            let mut local: Vec<_> = (-2..=2)
                .map(|d| widths[(i as isize + d).rem_euclid(count as isize) as usize])
                .collect();
            local.sort_by(f32::total_cmp);
            let width = local[2];
            if width < 0.12 || luma(inks[i]) - luma(paint) < 0.06 {
                continue;
            }
            let points = spans[i].points.to_vec();
            // The stroke lies on the shared mask curve; clipping its outside
            // half leaves exactly the measured inward band width.
            strokes.push(StructuralStroke {
                points: points.clone(),
                path_data: Some(spans[i].path_data.clone()),
                precise_points: None,
                color: inks[i],
                width: 2.0 * width,
                role: "alpha-boundary-stroke",
                width_samples: vec![(2.0 * width, 3)],
            });
            strokes.push(StructuralStroke {
                points,
                path_data: Some(spans[i].path_data.clone()),
                precise_points: None,
                color: paint,
                width: 2.0 * (width + 0.75).min(2.5),
                role: "alpha-boundary-underpaint",
                width_samples: Vec::new(),
            });
        }
    }
    // All incident-paint restoration precedes all edge ink; interleaving
    // them would let the next span erase the previous span's round cap.
    strokes.sort_by_key(|s| s.role == "alpha-boundary-stroke");
    let mut joined: Vec<StructuralStroke> = Vec::new();
    for stroke in strokes {
        if let Some(previous) = joined.last_mut() {
            let connected = previous.points.last().unwrap().distance(stroke.points[0]) < 1e-3;
            if connected
                && previous.role == stroke.role
                && distance(previous.color, stroke.color) < 0.06
                && (previous.width - stroke.width).abs() < 0.25 * previous.width.max(0.5)
            {
                let path = stroke.path_data.as_ref().unwrap();
                if let Some(start) = path.find(" C").into_iter().chain(path.find(" L")).min() {
                    previous
                        .path_data
                        .as_mut()
                        .unwrap()
                        .push_str(&path[start..]);
                    previous.points.extend(stroke.points.into_iter().skip(1));
                    continue;
                }
            }
        }
        joined.push(stroke);
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> SourceEdge {
        SourceEdge {
            points: (6..58).map(|x| [x as f64 + 0.5, 20.5]).collect(),
            width: 3.2,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        }
    }

    #[test]
    fn alpha_rim_uses_continuous_mask_curves_and_keeps_black_silhouettes() {
        let width = 96;
        for mode in 0..3 {
            let white_rim = mode != 0;
            let mut image = Raster::blank(width, width, [0.0; 3]);
            let mut alpha = vec![0.0; width * width];
            for y in 0..width {
                for x in 0..width {
                    let radius = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                    if radius < 38.0 {
                        alpha[y * width + x] = 1.0;
                        let angle = (y as f32 + 0.5 - 48.0).atan2(x as f32 + 0.5 - 48.0);
                        let band = if white_rim && !(mode == 2 && (0.3..1.1).contains(&angle)) {
                            ((radius - 36.5) / 1.0).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        image.pixels[y * width + x] = [0.3 + 0.7 * band; 3];
                    }
                }
            }
            let matte = crate::chroma::AlphaMatte::new(width, width, alpha);
            let source = crate::chroma::prepare_source_alpha(&image, &matte);
            let recovered = recover_alpha_boundary(&source, &matte);
            if !white_rim {
                assert!(recovered.is_empty());
                continue;
            }
            assert!(!recovered.is_empty());
            let mut covered = [false; 72];
            for stroke in &recovered {
                assert!(stroke.path_data.as_ref().unwrap().contains(" C"));
                for pair in stroke.points.windows(2) {
                    for i in 0..=8 {
                        let t = i as f32 / 8.0;
                        let x = pair[0].x * (1.0 - t) + pair[1].x * t - 48.0;
                        let y = pair[0].y * (1.0 - t) + pair[1].y * t - 48.0;
                        let angle = y.atan2(x).rem_euclid(std::f32::consts::TAU);
                        covered[(angle / std::f32::consts::TAU * 72.0) as usize % 72] = true;
                    }
                }
            }
            if mode == 2 {
                assert!(!covered[7], "an intentional rim gap was bridged");
                assert!(covered[36], "the supported opposite rim disappeared");
                continue;
            }
            assert!(
                covered.iter().all(|&b| b),
                "alpha rim has angular gaps: {covered:?}"
            );
        }
    }

    #[test]
    fn paired_band_recovers_width_and_both_incident_colors() {
        let mut image = Raster::blank(64, 40, [1.0; 3]);
        let paints = [[0.9, 0.8, 0.7], [0.6, 0.8, 1.0]];
        for y in 0..image.height {
            for x in 0..image.width {
                let d = y as f32 + 0.5 - 20.5;
                let coverage = (2.1 - d.abs()).clamp(0.0, 1.0);
                let paint = paints[usize::from(d >= 0.0)];
                image.pixels[y * 64 + x] = paint.map(|v| 0.1 * coverage + v * (1.0 - coverage));
            }
        }
        let recovered = recover(&image, &[edge(), edge()]);
        assert_eq!(
            recovered.strokes.len(),
            1,
            "duplicate paired edges must share one owner"
        );
        assert!((recovered.strokes[0].width - 3.2).abs() < 0.15);
        for &(i, c) in &recovered.updates {
            let side = usize::from(i / 64 >= 20);
            assert!(
                distance(c, paints[side]) < 1e-5,
                "incident colors must not be averaged"
            );
        }
        assert!(recovered.mask[20 * 64 + 30]);
        let mut roles = crate::edge::classify(&image);
        let (_, ink) = super::super::analyse(&image, &mut roles);
        assert!(
            ink.summary.recovered_boundary_strokes > 0,
            "the detector must feed the model"
        );
    }

    #[test]
    fn steps_and_diffuse_shadows_remain_paint_owned() {
        for shadow in [false, true] {
            let mut image = Raster::blank(64, 40, [1.0; 3]);
            for y in 0..40 {
                let value = if shadow {
                    1.0 - 0.8 * (-0.5 * ((y as f32 - 20.0) / 4.0).powi(2)).exp()
                } else if y < 21 {
                    0.1
                } else {
                    1.0
                };
                for x in 0..64 {
                    image.pixels[y * 64 + x] = [value; 3];
                }
            }
            assert!(
                recover(&image, &[edge()]).strokes.is_empty(),
                "shadow={shadow}"
            );
        }
    }

    #[test]
    fn diagonal_band_has_continuous_underpaint_ownership() {
        let mut image = Raster::blank(64, 64, [1.0; 3]);
        for y in 0..64 {
            for x in 0..64 {
                let d = (y as f32 - x as f32) * std::f32::consts::FRAC_1_SQRT_2;
                let coverage = (2.1 - d.abs()).clamp(0.0, 1.0);
                image.pixels[y * 64 + x] = [1.0 - 0.9 * coverage; 3];
            }
        }
        let mut diagonal = edge();
        diagonal.points = (14..50).map(|x| [x as f64 + 0.5; 2]).collect();
        let recovered = recover(&image, &[diagonal]);
        assert_eq!(recovered.strokes.len(), 1);
        for x in 20..44 {
            for y in x - 1..=x + 1 {
                assert!(recovered.mask[y * 64 + x], "unowned ink at ({x}, {y})");
            }
        }
    }
}
