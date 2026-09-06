//! Recover a narrow, uniform ink band jointly with its two incident paints.
//! A step edge or a broad shadow cannot supply two supported band edges.

use super::{Point, StructuralStroke};
use crate::edge::SourceEdge;
use crate::raster::Raster;
use rayon::prelude::*;

type PaintUpdate = (usize, [f32; 3]);
type StrokeCandidate = (StructuralStroke, Vec<PaintUpdate>);

#[derive(Clone)]
struct Profile {
    bright: bool,
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

fn profile(image: &Raster, edge: &SourceEdge, i: usize, bright: bool) -> Option<Profile> {
    let value = |color| {
        if bright {
            1.0 - luma(color)
        } else {
            luma(color)
        }
    };
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
    let reach = (edge.width as f32 + 2.5).max(20.0 * scale).min(24.0);
    let count = (reach * 2.0).ceil() as isize;
    let colors: Vec<_> = (-count..=count)
        .map(|k| sample(image, offset(p, normal, k as f32 * 0.5)))
        .collect();
    let middle = count as usize;
    let search = ((edge.width as f32 * 0.5 + 0.75).max(8.0) * 2.0).ceil() as usize;
    let core = (middle.saturating_sub(search)..=(middle + search).min(colors.len() - 1))
        .filter(|&j| {
            j > 0
                && j + 1 < colors.len()
                && (value(colors[j - 1]) - value(colors[j]))
                    .max(value(colors[j + 1]) - value(colors[j]))
                    <= 0.06
        })
        .min_by(|&a, &b| {
            // The seed can be a band edge, several pixels away from the
            // ink core. Distance only breaks near-ties; a strong penalty
            // selects the antialiased edge instead of the actual core.
            let score = |j: usize| value(colors[j]) + 0.00001 * (j as f32 - middle as f32).powi(2);
            score(a).total_cmp(&score(b))
        })?;
    let ink = colors[core];
    let sides = [colors[0], *colors.last()?];
    let dark = value(ink);
    if sides.iter().any(|&c| value(c) - dark < 0.12) {
        return None;
    }
    let threshold = [
        0.5 * (dark + value(sides[0])),
        0.5 * (dark + value(sides[1])),
    ];
    let mut left = core;
    while left > 0 && value(colors[left]) < threshold[0] {
        left -= 1;
    }
    let mut right = core;
    while right + 1 < colors.len() && value(colors[right]) < threshold[1] {
        right += 1;
    }
    if left == 0 || right + 1 == colors.len() {
        return None;
    }
    let crossing = |a: usize, b: usize, threshold: f32| {
        let t = (threshold - value(colors[a])) / (value(colors[b]) - value(colors[a]));
        (a as f32 + t * (b as f32 - a as f32) - middle as f32) * 0.5
    };
    let low = crossing(left, left + 1, threshold[0]);
    let high = crossing(right - 1, right, threshold[1]);
    let width = high - low;
    if !(1.5..=16.0).contains(&width) || low > 1.0 || high < -1.0 {
        return None;
    }
    // A lightly shaded ink core is still an authored band when both of its
    // transitions are sharp. A Gaussian shadow can have a flat minimum too,
    // but its 20--80% transitions are much broader than raster antialiasing.
    for (side, direction) in [(0, -1_isize), (1, 1)] {
        let level = |fraction: f32| dark + fraction * (value(sides[side]) - dark);
        let mut j = core;
        let mut positions = [0.0; 2];
        for (k, fraction) in [0.2, 0.8].into_iter().enumerate() {
            while j > 0 && j + 1 < colors.len() && value(colors[j]) < level(fraction) {
                j = (j as isize + direction) as usize;
            }
            if j == 0 || j + 1 == colors.len() {
                return None;
            }
            let previous = (j as isize - direction) as usize;
            positions[k] = crossing(previous, j, level(fraction));
        }
        if (positions[1] - positions[0]).abs() > (1.5 + 0.15 * width).min(3.0) {
            return None;
        }
    }
    let center = offset(p, normal, 0.5 * (low + high));
    let contrast = (value(sides[0]) - dark).min(value(sides[1]) - dark);
    if [-1.0_f32, 1.0].iter().any(|&sign| {
        value(sample(
            image,
            offset(center, normal, sign * (0.25 * width - 0.5).max(0.25)),
        )) - dark
            > 0.10 * contrast
    }) {
        return None;
    }
    // The extremum locates the band but is a biased ink estimate when its
    // core has mild shading. Use the robust interior colour for the shared
    // width/underpaint model rather than exporting its darkest/lightest pixel.
    let core_colors: Vec<_> = (0..=8)
        .map(|j| {
            sample(
                image,
                offset(center, normal, (j as f32 / 8.0 - 0.5) * 0.5 * width),
            )
        })
        .collect();
    let ink = std::array::from_fn(|c| {
        let mut values: Vec<_> = core_colors.iter().map(|color| color[c]).collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    });
    Some(Profile {
        bright,
        center,
        normal,
        width,
        ink,
        sides,
    })
}

fn band_parameters(profiles: &[Profile]) -> Option<(f32, [f32; 3])> {
    if profiles.len() < 3 {
        return None;
    }
    let mut widths: Vec<_> = profiles.iter().map(|p| p.width).collect();
    widths.sort_by(f32::total_cmp);
    let width = widths[widths.len() / 2];
    // A percentile test can hide a taper or a junction in the discarded
    // tails. Every section must agree before a whole region loses its two
    // independently fitted contours.
    if widths.iter().any(|w| (w - width).abs() > 0.5) {
        return None;
    }
    let color = std::array::from_fn(|c| {
        let mut values: Vec<_> = profiles.iter().map(|p| p.ink[c]).collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    });
    if profiles.iter().any(|p| distance(p.ink, color) > 0.08) {
        return None;
    }
    Some((width, color))
}

fn candidate(image: &Raster, profiles: Vec<Profile>) -> Option<StrokeCandidate> {
    if profiles.len() < 12 {
        return None;
    }
    let (width, color) = band_parameters(&profiles)?;
    let points: Vec<_> = profiles.iter().map(|p| p.center).collect();
    let length: f32 = points.windows(2).map(|p| p[0].distance(p[1])).sum();
    if length < (6.0 * width).max(16.0) {
        return None;
    }
    let mut updates = std::collections::BTreeMap::new();
    let closed = points.first() == points.last();
    for (i, p) in profiles.iter().enumerate() {
        // Keep a short original-Paint overlap at an open interval's ends.
        // The butt-capped model does not fully cover their antialiased pixels.
        if !closed
            && (p.center.distance(points[0]) < 1.5
                || p.center.distance(*points.last().unwrap()) < 1.5)
        {
            continue;
        }
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
            path_data: Some(crate::geometry::fitted_structural_open_path_data(
                &points, 0.35, 0.25,
            )),
            points,
            precise_points: None,
            color,
            width,
            role: "boundary-stroke",
            width_samples: vec![(width, profiles.len())],
        },
        updates.into_iter().collect(),
    ))
}

fn profile_endpoint(run: &[Profile], start: bool) -> (&Profile, Point) {
    let index = if start { 0 } else { run.len() - 1 };
    let interior = if start {
        3.min(run.len() - 1)
    } else {
        run.len().saturating_sub(4)
    };
    let point = &run[index];
    let delta = Point {
        x: point.center.x - run[interior].center.x,
        y: point.center.y - run[interior].center.y,
    };
    let length = delta.x.hypot(delta.y).max(1e-6);
    (
        point,
        Point {
            x: delta.x / length,
            y: delta.y / length,
        },
    )
}

fn supported_bridge(
    image: &Raster,
    a: (&Profile, Point),
    b: (&Profile, Point),
) -> Option<Vec<Profile>> {
    let (first, outgoing) = a;
    let (last, incoming) = b;
    let length = first.center.distance(last.center);
    if first.bright != last.bright
        || distance(first.ink, last.ink) > 0.06
        || (first.width - last.width).abs() > (0.1 * first.width).max(0.5)
        || outgoing.x * incoming.x + outgoing.y * incoming.y > -0.94
    {
        return None;
    }
    if length < 0.25 {
        return Some(Vec::new());
    }
    let direction = Point {
        x: (last.center.x - first.center.x) / length,
        y: (last.center.y - first.center.y) / length,
    };
    if outgoing.x * direction.x + outgoing.y * direction.y < 0.87
        || incoming.x * direction.x + incoming.y * direction.y > -0.87
    {
        return None;
    }
    let count = (length / 0.5).ceil() as usize;
    let points: Vec<_> = [-3.0, -2.0, -1.0]
        .into_iter()
        .chain((0..=count).map(|i| length * i as f32 / count as f32))
        .chain([length + 1.0, length + 2.0, length + 3.0])
        .map(|d| {
            let p = offset(first.center, direction, d);
            [p.x as f64, p.y as f64]
        })
        .collect();
    let edge = SourceEdge {
        points,
        width: first.width as f64,
        role: "band-boundary",
        width_samples: Vec::new(),
    };
    let mut bridge = Vec::new();
    for i in 1..count {
        let p = profile(image, &edge, i + 3, first.bright)?;
        let expected = offset(first.center, direction, length * i as f32 / count as f32);
        if p.center.distance(expected) > 0.5
            || distance(p.ink, first.ink) > 0.06
            || (p.width - first.width).abs() > (0.1 * first.width).max(0.5)
        {
            return None;
        }
        bridge.push(p);
    }
    Some(bridge)
}

/// Join detector fragments only after every intervening cross-section proves
/// the same band. Width/colour agreement and tangent continuity alone cannot
/// distinguish an authored break from a missing graph sample.
fn join_profile_runs(image: &Raster, runs: Vec<Vec<Profile>>) -> Vec<Vec<Profile>> {
    let mut runs: Vec<_> = runs
        .into_iter()
        .filter(|r| r.len() >= 3)
        .map(Some)
        .collect();
    for _ in 0..8 {
        let mut endpoints = Vec::new();
        let mut owners = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            let Some(run) = run else {
                continue;
            };
            if run[0].center == run.last().unwrap().center {
                continue;
            }
            for start in [true, false] {
                endpoints.push(profile_endpoint(run, start).0.center);
                owners.push((i, start));
            }
        }
        let mut pairs = super::nearby_point_pairs(&endpoints, 2.5);
        pairs.sort_by(|&(a, b), &(c, d)| {
            endpoints[a]
                .distance(endpoints[b])
                .total_cmp(&endpoints[c].distance(endpoints[d]))
                .then((a, b).cmp(&(c, d)))
        });
        let mut changed = vec![false; runs.len()];
        for (a, b) in pairs {
            let (first, first_start) = owners[a];
            let (last, last_start) = owners[b];
            if first == last || changed[first] || changed[last] {
                continue;
            }
            let (Some(left), Some(right)) = (&runs[first], &runs[last]) else {
                continue;
            };
            let Some(bridge) = supported_bridge(
                image,
                profile_endpoint(left, first_start),
                profile_endpoint(right, last_start),
            ) else {
                continue;
            };
            let mut joined = left.clone();
            if first_start {
                joined.reverse();
            }
            joined.extend(bridge);
            if last_start {
                joined.extend(right.iter().cloned());
            } else {
                joined.extend(right.iter().rev().cloned());
            }
            if band_parameters(&joined).is_none() {
                continue;
            }
            runs[first] = Some(joined);
            runs[last] = None;
            changed[first] = true;
            changed[last] = true;
        }
        if !changed.iter().any(|v| *v) {
            break;
        }
    }
    runs.into_iter().flatten().collect()
}

/// A fitted interval must account for the complete connected ink region.
/// Width/profile support alone accepts fragments of an outline network; its
/// junctions then remain in Paint and can separate from the new stroke after
/// segmentation. Follow the source ink before removing any pixels, and reject
/// a model if that ink continues beyond its footprint and small AA/cap collar.
fn owns_complete_band(image: &Raster, stroke: &StructuralStroke, updates: &[PaintUpdate]) -> bool {
    use std::collections::HashSet;

    let footprint: HashSet<_> = updates.iter().map(|&(i, _)| i).collect();
    let ink = |i: usize| distance(image.pixels[i], stroke.color) <= 0.10;
    let mut visited: HashSet<_> = footprint.iter().copied().filter(|&i| ink(i)).collect();
    if visited.is_empty() {
        return false;
    }
    let mut pending: Vec<_> = visited.iter().copied().collect();
    while let Some(i) = pending.pop() {
        let x = i % image.width;
        let y = i / image.width;
        for py in y.saturating_sub(1)..=(y + 1).min(image.height - 1) {
            for px in x.saturating_sub(1)..=(x + 1).min(image.width - 1) {
                let next = py * image.width + px;
                if !ink(next) || !visited.insert(next) {
                    continue;
                }
                // Three pixels allow the retained 1.5px end overlap, detector
                // spacing and raster AA. This collar is bounded for a wide band;
                // it cannot grow with the connected source outline.
                if !footprint.contains(&next)
                    && !(py.saturating_sub(3)..=(py + 3).min(image.height - 1)).any(|fy| {
                        (px.saturating_sub(3)..=(px + 3).min(image.width - 1))
                            .any(|fx| footprint.contains(&(fy * image.width + fx)))
                    })
                {
                    return false;
                }
                pending.push(next);
            }
        }
    }
    true
}

pub(super) fn recover(image: &Raster, edges: &[SourceEdge], boundaries: &[SourceEdge]) -> Recovery {
    let runs: Vec<_> = edges
        .par_iter()
        .chain(boundaries.par_iter())
        .flat_map_iter(|edge| {
            let mut runs = Vec::new();
            if edge.points.len() < 12 {
                return runs;
            }
            for bright in [false, true] {
                let mut run = Vec::new();
                for i in 0..edge.points.len() {
                    if let Some(p) = profile(image, edge, i, bright) {
                        run.push(p);
                    } else if !run.is_empty() {
                        runs.push(std::mem::take(&mut run));
                    }
                }
                if !run.is_empty() {
                    runs.push(run);
                }
            }
            runs
        })
        .collect();
    let mut candidates: Vec<_> = join_profile_runs(image, runs)
        .into_par_iter()
        .filter_map(|run| candidate(image, run))
        .collect();
    // Let the most complete observation own a band before rejecting its
    // opposite-edge duplicates. A short fragment must not veto a long fit.
    candidates.sort_by_key(|(_, updates)| std::cmp::Reverse(updates.len()));
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
        if !owns_complete_band(image, &stroke, &updates) {
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

/// Re-measure repeated subpixel ink after graph joining. Connectivity is only
/// a geometric hypothesis: it must not turn a dotted source into a solid band.
pub(super) fn refine_interrupted(
    source: &Raster,
    stroke: &StructuralStroke,
    matte: Option<&crate::chroma::AlphaMatte>,
) -> Option<Vec<StructuralStroke>> {
    if stroke.width > 6.0
        || stroke.points.len() < 2
        || matches!(
            stroke.role,
            "boundary-stroke"
                | "bright-ridge-on-boundary"
                | "alpha-boundary-stroke"
                | "alpha-boundary-underpaint"
        )
    {
        return None;
    }
    let covered_sample = |p: Point| {
        let Some(matte) = matte else {
            return (luma(sample(source, p)), 1.0);
        };
        let x = (p.x - 0.5).clamp(0.0, (source.width - 1) as f32);
        let y = (p.y - 0.5).clamp(0.0, (source.height - 1) as f32);
        let (ix, iy) = (x as usize, y as usize);
        let (tx, ty) = (x - ix as f32, y - iy as f32);
        let mut visible = 0.0;
        let mut alpha = 0.0;
        for (x, y, weight) in [
            (ix, iy, (1.0 - tx) * (1.0 - ty)),
            ((ix + 1).min(source.width - 1), iy, tx * (1.0 - ty)),
            (ix, (iy + 1).min(source.height - 1), (1.0 - tx) * ty),
            (
                (ix + 1).min(source.width - 1),
                (iy + 1).min(source.height - 1),
                tx * ty,
            ),
        ] {
            let i = y * source.width + x;
            let a = matte.get(i);
            // Interpolate composited samples, never straight RGB and alpha
            // independently: invisible black must contribute no visible ink.
            visible += weight * (luma(source.pixels[i]) * a + 1.0 - a);
            alpha += weight * a;
        }
        (visible, alpha)
    };
    let visible = |p| covered_sample(p).0;
    let reach = (stroke.width * 0.5 + 1.0).max(2.0);
    let count = (reach * 4.0).ceil() as i32;
    let mut measured = Vec::new();
    for pair in stroke.points.windows(2) {
        let length = pair[0].distance(pair[1]);
        if length < 1e-5 {
            continue;
        }
        let normal = Point {
            x: (pair[0].y - pair[1].y) / length,
            y: (pair[1].x - pair[0].x) / length,
        };
        let steps = (length * 2.0).ceil() as usize;
        for j in 0..steps {
            let t = j as f32 / steps as f32;
            let p = Point {
                x: pair[0].x + t * (pair[1].x - pair[0].x),
                y: pair[0].y + t * (pair[1].y - pair[0].y),
            };
            let background =
                visible(offset(p, normal, -reach)).min(visible(offset(p, normal, reach)));
            let contrast = background - luma(stroke.color);
            let mut mass = 0.0;
            let mut moment = 0.0;
            if contrast > 0.15 {
                for k in -count..=count {
                    let d = k as f32 * 0.25;
                    let weight = ((background - visible(offset(p, normal, d))) / contrast)
                        .clamp(0.0, 1.0)
                        * 0.25;
                    mass += weight;
                    moment += weight * d;
                }
            }
            let center = offset(
                p,
                normal,
                if mass > 0.05 {
                    (moment / mass).clamp(-1.5, 1.5)
                } else {
                    0.0
                },
            );
            let width = mass / covered_sample(center).1.max(0.25);
            measured.push((center, width));
        }
    }
    if measured.len() < 8 {
        return None;
    }
    let mut widths: Vec<_> = measured.iter().map(|(_, w)| *w).collect();
    widths.sort_by(f32::total_cmp);
    let low = widths[widths.len() / 5];
    let high = widths[widths.len() * 4 / 5];
    let median = widths[widths.len() / 2];
    let crossings = measured
        .windows(2)
        .filter(|p| p[0].1 < 0.6 * high && p[1].1 >= 0.6 * high)
        .count();
    if !(0.2..=1.8).contains(&high)
        || low > 0.6 * high
        || median > 0.85 * stroke.width
        || crossings < 2
    {
        return None;
    }
    Some(
        measured
            .windows(2)
            .filter_map(|p| {
                let width = 0.5 * (p[0].1 + p[1].1);
                if width < 0.08 || p[0].0.distance(p[1].0) > 2.0 {
                    return None;
                }
                Some(StructuralStroke {
                    points: vec![p[0].0, p[1].0],
                    path_data: None,
                    precise_points: None,
                    color: stroke.color,
                    width: width.min(1.8),
                    role: "sampled-ink",
                    width_samples: Vec::new(),
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotted_ink_keeps_its_gaps_and_ignores_invisible_black() {
        for hidden_black in [false, true] {
            let mut source = Raster::blank(96, 40, [1.0; 3]);
            let mut alpha = vec![1.0; 96 * 40];
            for x in 8..88 {
                if x % 4 < 2 {
                    source.pixels[20 * 96 + x] = [0.0; 3];
                }
                if hidden_black {
                    for y in 17..20 {
                        source.pixels[y * 96 + x] = [0.0; 3];
                        alpha[y * 96 + x] = 0.0;
                    }
                }
            }
            let matte = crate::chroma::AlphaMatte::new(96, 40, alpha);
            let stroke = StructuralStroke {
                points: vec![Point { x: 8.5, y: 20.5 }, Point { x: 87.5, y: 20.5 }],
                path_data: None,
                precise_points: None,
                color: [0.0; 3],
                width: 4.0,
                role: "ridge-on-boundary",
                width_samples: Vec::new(),
            };
            let restored = refine_interrupted(&source, &stroke, Some(&matte)).unwrap();
            assert!(!restored.is_empty());
            assert!(restored.iter().all(|s| s.width <= 1.01));
            assert!(restored
                .iter()
                .flat_map(|s| &s.points)
                .all(|p| (p.y - 20.5).abs() < 0.01));
            for x in (10..86).step_by(4) {
                let gap = x as f32 + 1.0;
                assert!(
                    restored
                        .iter()
                        .all(|s| !(s.points[0].x < gap && s.points[1].x > gap)),
                    "closed source gap at {gap}"
                );
            }
        }
    }

    #[test]
    fn a_continuous_thin_line_is_not_replaced_by_dashes() {
        let mut source = Raster::blank(96, 40, [1.0; 3]);
        for x in 8..88 {
            source.pixels[20 * 96 + x] = [0.0; 3];
        }
        let stroke = StructuralStroke {
            points: vec![Point { x: 10.5, y: 20.5 }, Point { x: 85.5, y: 20.5 }],
            path_data: None,
            precise_points: None,
            color: [0.0; 3],
            width: 1.2,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        };
        assert!(refine_interrupted(&source, &stroke, None).is_none());
    }

    fn edge() -> SourceEdge {
        SourceEdge {
            points: (6..58).map(|x| [x as f64 + 0.5, 20.5]).collect(),
            width: 3.2,
            role: "ridge-on-boundary",
            width_samples: Vec::new(),
        }
    }

    fn wide_band(phase: f32, gap: bool, wobble: bool) -> (Raster, SourceEdge) {
        let mut image = Raster::blank(240, 80, [1.0; 3]);
        for y in 0..80 {
            for x in 0..240 {
                let d = y as f32 + 0.5 - (40.0 + phase);
                let width = 12.0
                    + if wobble {
                        0.3 * (x as f32 * 0.09).sin()
                    } else {
                        0.0
                    };
                let coverage = if !(12..228).contains(&x) || (gap && (116..124).contains(&x)) {
                    0.0
                } else {
                    (0.5 * width + 0.5 - d.abs()).clamp(0.0, 1.0)
                };
                let paint = if d < 0.0 {
                    [0.9, 0.8, 0.7]
                } else {
                    [0.6, 0.8, 1.0]
                };
                image.pixels[y * 240 + x] = paint.map(|v| 0.1 * coverage + v * (1.0 - coverage));
            }
        }
        let seed = SourceEdge {
            // A detector's 1.2px inset lies near the outside of a 12px band.
            points: (12..228)
                .map(|x| [x as f64 + 0.5, (34.5 + phase) as f64])
                .collect(),
            width: 1.2,
            role: "dark-boundary",
            width_samples: Vec::new(),
        };
        (image, seed)
    }

    fn render_recovered(
        width: usize,
        height: usize,
        recovered: &Recovery,
    ) -> resvg::tiny_skia::Pixmap {
        let mut ink = super::super::StructuralInk::empty();
        ink.strokes = recovered.strokes.clone();
        let (svg, _) = crate::svg::serialize(width, height, &[], &[], &ink, 0.0, true);
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut pixels = resvg::tiny_skia::Pixmap::new(width as u32, height as u32).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixels.as_mut(),
        );
        pixels
    }

    #[test]
    fn outside_seeds_recover_one_complete_band_at_different_pixel_phases() {
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let (image, seed) = wide_band(phase, false, false);
            let mut opposite = seed.clone();
            for p in &mut opposite.points {
                p[1] += 11.0;
            }
            let recovered = recover(&image, &[seed, opposite], &[]);
            assert_eq!(recovered.strokes.len(), 1, "phase {phase}");
            let stroke = &recovered.strokes[0];
            assert!((stroke.width - 12.0).abs() < 0.15);
            assert!(stroke
                .points
                .iter()
                .all(|p| (p.y - 40.0 - phase).abs() < 0.15));
            assert!(stroke.path_data.as_ref().unwrap().contains(" L"));
            for x in 20..220 {
                for y in 35..45 {
                    assert!(recovered.mask[y * 240 + x], "unremoved core at {x}, {y}");
                }
            }
        }
    }

    #[test]
    fn partial_outline_intervals_keep_the_connected_region_in_paint() {
        for bright in [false, true] {
            let (mut image, seed) = wide_band(0.25, false, false);
            if bright {
                for pixel in &mut image.pixels {
                    *pixel = pixel.map(|c| 1.0 - c);
                }
            }
            assert_eq!(
                recover(&image, std::slice::from_ref(&seed), &[])
                    .strokes
                    .len(),
                1
            );
            let partial = SourceEdge {
                points: seed.points[20..seed.points.len() - 20].to_vec(),
                ..seed
            };
            let rejected = recover(&image, &[partial], &[]);
            assert!(rejected.strokes.is_empty());
            assert!(rejected.updates.is_empty());
            assert!(rejected.mask.iter().all(|&owned| !owned));
        }
    }

    #[test]
    fn connected_branches_and_short_width_outliers_stay_in_paint() {
        for bright in [false, true] {
            for branch in [false, true] {
                let (mut image, seed) = wide_band(0.0, false, false);
                if branch {
                    // A branch is part of the same ink region even when a
                    // detector reports just the long horizontal interval.
                    for y in 42..66 {
                        for x in 118..124 {
                            image.pixels[y * image.width + x] = [0.1; 3];
                        }
                    }
                } else {
                    // Fewer than 10% of sections widen. The previous
                    // percentile gate discarded these observations.
                    for y in 32..48 {
                        for x in 214..224 {
                            image.pixels[y * image.width + x] = [0.1; 3];
                        }
                    }
                }
                if bright {
                    for pixel in &mut image.pixels {
                        *pixel = pixel.map(|c| 1.0 - c);
                    }
                }
                let recovered = recover(&image, &[seed], &[]);
                assert!(
                    recovered.strokes.is_empty(),
                    "bright={bright}, branch={branch}"
                );
                assert!(recovered.updates.is_empty());
            }
        }
    }

    #[test]
    fn recovered_width_is_uniform_and_original_paint_is_removed_under_it() {
        let (image, seed) = wide_band(0.25, false, true);
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        let raster = render_recovered(240, 80, &recovered);
        let mut widths = Vec::new();
        let mut paint = image.clone();
        for &(i, color) in &recovered.updates {
            paint.pixels[i] = color;
        }
        for x in 20..220 {
            widths.push(
                (20..60)
                    .map(|y| raster.pixels()[y * 240 + x].alpha() as f32 / 255.0)
                    .sum::<f32>(),
            );
            for y in 36..44 {
                let expected = if (y as f32 + 0.5) < 40.25 {
                    [0.9, 0.8, 0.7]
                } else {
                    [0.6, 0.8, 1.0]
                };
                assert!(
                    distance(paint.pixels[y * 240 + x], expected) < 1e-5,
                    "source ink remains under the fitted stroke"
                );
            }
        }
        let min = widths.iter().copied().fold(f32::INFINITY, f32::min);
        let max = widths.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max - min < 0.05,
            "rendered width still wobbles: {min}..{max}"
        );
        assert!((0.5 * (min + max) - 12.0).abs() < 0.2);
    }

    #[test]
    fn mildly_shaded_core_uses_interior_colour_instead_of_the_extremum() {
        let (mut image, seed) = wide_band(0.0, false, false);
        for y in 0..80 {
            let d = y as f32 + 0.5 - 40.0;
            let coverage = (6.5 - d.abs()).clamp(0.0, 1.0);
            for x in 0..240 {
                image.pixels[y * 240 + x] =
                    image.pixels[y * 240 + x].map(|v| v + 0.005 * d * coverage);
            }
        }
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        for c in recovered.strokes[0].color {
            assert!((c - 0.1).abs() < 0.005, "biased core colour {c}");
        }
    }

    #[test]
    fn bright_band_between_different_paints_uses_the_same_joint_model() {
        let (mut image, seed) = wide_band(0.5, false, false);
        for color in &mut image.pixels {
            *color = color.map(|v| 1.0 - v);
        }
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        assert!(recovered.strokes[0]
            .color
            .iter()
            .all(|v| (*v - 0.9).abs() < 1e-5));
        assert!((recovered.strokes[0].width - 12.0).abs() < 0.15);
        let mut roles = crate::edge::classify(&image);
        let (paint, ink) = super::super::analyse(&image, &mut roles);
        // Here the detector fragments at the caps. The complete explicit
        // seed above is recoverable; partial detector runs must keep Paint.
        assert_eq!(ink.summary.recovered_boundary_strokes, 0);
        let middle = 40 * image.width + 120;
        assert_eq!(paint.pixels[middle], image.pixels[middle]);
        assert!(!ink.paint_ownership_mask[middle]);
    }

    #[test]
    fn closed_circular_band_has_one_width_and_no_artificial_storage_gap() {
        let mut image = Raster::blank(160, 160, [0.8; 3]);
        for y in 0..160 {
            for x in 0..160 {
                let r = (x as f32 + 0.5 - 80.0).hypot(y as f32 + 0.5 - 80.0);
                let coverage = (4.5 - (r - 50.0).abs()).clamp(0.0, 1.0);
                let paint = if r < 50.0 { 0.6 } else { 0.8 };
                image.pixels[y * 160 + x] = [0.1 * coverage + paint * (1.0 - coverage); 3];
            }
        }
        let mut points: Vec<_> = (0..360)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 360.0;
                [80.0 + 46.5 * a.cos(), 80.0 + 46.5 * a.sin()]
            })
            .collect();
        points.push(points[0]);
        let seed = SourceEdge {
            points,
            width: 1.2,
            role: "dark-boundary",
            width_samples: Vec::new(),
        };
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        let stroke = &recovered.strokes[0];
        assert!((stroke.width - 8.0).abs() < 0.2);
        for bright in [false, true] {
            let mut source = image.clone();
            if bright {
                for pixel in &mut source.pixels {
                    *pixel = pixel.map(|c| 1.0 - c);
                }
            }
            let (_, detected) = super::super::analyse(&source, &mut crate::edge::classify(&source));
            assert!(
                detected.summary.recovered_boundary_strokes > 0,
                "complete circles should be detected automatically, bright={bright}"
            );
        }
        assert!(stroke.path_data.as_ref().unwrap().ends_with(" Z"));
        let raster = render_recovered(160, 160, &recovered);
        for degrees in 0..360 {
            let a = degrees as f32 * std::f32::consts::TAU / 360.0;
            let x = (80.0 + 50.0 * a.cos()).floor() as usize;
            let y = (80.0 + 50.0 * a.sin()).floor() as usize;
            assert!(raster.pixels()[y * 160 + x].alpha() > 250);
            assert!(recovered.mask[y * 160 + x]);
        }
    }

    #[test]
    fn short_detector_fragments_join_only_over_source_supported_ink() {
        let (image, seed) = wide_band(0.0, false, false);
        let fragments: Vec<_> = seed
            .points
            .chunks(54)
            .map(|points| SourceEdge {
                points: points.to_vec(),
                ..seed.clone()
            })
            .collect();
        let recovered = recover(&image, &fragments, &[]);
        assert_eq!(
            recovered.strokes.len(),
            1,
            "four short fragments should become one editable stroke"
        );
        assert!(recovered.strokes[0].points.len() >= seed.points.len());
        let mut broken = image.clone();
        for y in 0..80 {
            broken.pixels[y * 240 + 120] = if y < 40 {
                [0.9, 0.8, 0.7]
            } else {
                [0.6, 0.8, 1.0]
            };
        }
        let recovered = recover(&broken, &[seed], &[]);
        assert_eq!(
            recovered.strokes.len(),
            2,
            "a one-pixel source gap must veto joining"
        );
        let raster = render_recovered(240, 80, &recovered);
        for y in 32..48 {
            assert_eq!(raster.pixels()[y * 240 + 120].alpha(), 0);
        }
    }

    #[test]
    fn interval_caps_do_not_bridge_an_authored_gap() {
        let (image, seed) = wide_band(0.0, true, false);
        let recovered = recover(&image, &[seed], &[]);
        assert_eq!(recovered.strokes.len(), 2);
        let raster = render_recovered(240, 80, &recovered);
        for y in 32..48 {
            for x in 116..124 {
                assert_eq!(
                    raster.pixels()[y * 240 + x].alpha(),
                    0,
                    "invented ink in the gap at {x}, {y}"
                );
                assert!(!recovered.mask[y * 240 + x]);
            }
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
                let coverage = if (6..58).contains(&x) {
                    (2.1 - d.abs()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let paint = paints[usize::from(d >= 0.0)];
                image.pixels[y * 64 + x] = paint.map(|v| 0.1 * coverage + v * (1.0 - coverage));
            }
        }
        let recovered = recover(&image, &[edge(), edge()], &[]);
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
            for bright in [false, true] {
                if bright {
                    for color in &mut image.pixels {
                        *color = color.map(|v| 1.0 - v);
                    }
                }
                assert!(
                    recover(&image, &[edge()], &[]).strokes.is_empty(),
                    "shadow={shadow}, bright={bright}"
                );
            }
        }
    }

    #[test]
    fn diagonal_band_has_continuous_underpaint_ownership() {
        let mut image = Raster::blank(64, 64, [1.0; 3]);
        for y in 0..64 {
            for x in 0..64 {
                let d = (y as f32 - x as f32) * std::f32::consts::FRAC_1_SQRT_2;
                let coverage = if (28..=98).contains(&(x + y)) {
                    (2.1 - d.abs()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                image.pixels[y * 64 + x] = [1.0 - 0.9 * coverage; 3];
            }
        }
        let mut diagonal = edge();
        // Sample just inside the antialiased cap; its retained Paint fits
        // within the bounded collar around the complete stroke model.
        diagonal.points = (15..49).map(|x| [x as f64 + 0.5; 2]).collect();
        let recovered = recover(&image, &[diagonal], &[]);
        assert_eq!(recovered.strokes.len(), 1);
        for x in 20..44 {
            for y in x - 1..=x + 1 {
                assert!(recovered.mask[y * 64 + x], "unowned ink at ({x}, {y})");
            }
        }
    }
}
