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

/// A ridge seed is medial, unlike a boundary seed. Refine only a clearly
/// over-wide, isolated straight ridge; searching the wide boundary profile
/// can jump to a neighbouring parallel ink band instead of this one.
pub(super) fn refine_isolated_ridge(
    image: &Raster,
    start: Point,
    end: Point,
    width: f32,
) -> Option<(Point, Point, f32)> {
    let length = start.distance(end);
    if width < 3.0 || length < width * 6.0 {
        return None;
    }
    let normal = Point {
        x: (start.y - end.y) / length,
        y: (end.x - start.x) / length,
    };
    let reach = width * 0.75 + 0.5;
    let count = (reach * 4.0).ceil() as i32;
    let samples = ((length / 2.0) as usize).clamp(8, 64);
    let mut measured = Vec::new();
    for i in 1..samples {
        let t = i as f32 / samples as f32;
        let p = Point {
            x: start.x + (end.x - start.x) * t,
            y: start.y + (end.y - start.y) * t,
        };
        let values: Vec<_> = (-count..=count)
            .map(|j| luma(sample(image, offset(p, normal, j as f32 * 0.25))))
            .collect();
        let middle = count as usize;
        let search = (width * 2.0).ceil() as usize;
        let core = (middle.saturating_sub(search)..=(middle + search).min(values.len() - 1))
            .min_by(|&a, &b| values[a].total_cmp(&values[b]))
            .unwrap();
        let ink = values[core];
        if ink > 0.15 || values[0].min(*values.last().unwrap()) - ink < 0.08 {
            continue;
        }
        // Match outline recovery's persistent-core crossing. Half-height
        // also includes the broad shaded shoulder beside a narrow dark rim.
        let thresholds = [
            ink + 0.35 * (values[0] - ink),
            ink + 0.35 * (values[values.len() - 1] - ink),
        ];
        let Some(left) = (0..core).rev().find(|&j| values[j] >= thresholds[0]) else {
            continue;
        };
        let Some(right) = (core + 1..values.len()).find(|&j| values[j] >= thresholds[1]) else {
            continue;
        };
        let crossing = |a: usize, b: usize, threshold: f32| {
            let t = (threshold - values[a]) / (values[b] - values[a]);
            (a as f32 + t * (b as f32 - a as f32) - middle as f32) * 0.25
        };
        let low = crossing(left, left + 1, thresholds[0]);
        let high = crossing(right - 1, right, thresholds[1]);
        measured.push((high - low, 0.5 * (low + high)));
    }
    if measured.len() * 4 < (samples - 1) * 3 {
        return None;
    }
    let mut widths: Vec<_> = measured.iter().map(|p| p.0).collect();
    let mut shifts: Vec<_> = measured.iter().map(|p| p.1).collect();
    widths.sort_by(f32::total_cmp);
    shifts.sort_by(f32::total_cmp);
    let n = measured.len();
    let refined = widths[n / 2];
    let shift = shifts[n / 2];
    if refined < 0.5
        || refined > width * 0.65
        || widths[n * 9 / 10] - widths[n / 10] > 0.75
        || shifts[n * 9 / 10] - shifts[n / 10] > 0.75
        || shift.abs() > width * 0.5
    {
        return None;
    }
    Some((
        offset(start, normal, shift),
        offset(end, normal, shift),
        refined,
    ))
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
    let mut strokes = recover_alpha_boundary_polarity(image, matte, true, false);
    strokes.extend(recover_alpha_boundary_polarity(image, matte, false, false));
    strokes.sort_by_key(|s| s.role == "alpha-boundary-stroke");
    strokes
}

pub(super) fn recover_local_coverage_boundary(
    image: &Raster,
    matte: &crate::chroma::AlphaMatte,
    local: &crate::alpha_coverage::LocalCoverage,
) -> Vec<StructuralStroke> {
    let support = crate::chroma::AlphaMatte::from_u8(
        matte.width,
        matte.height,
        (0..matte.len())
            .map(|i| {
                if local.opacity[i] > 0.0 {
                    (matte.get(i) * 255.0).round() as u8
                } else {
                    0
                }
            })
            .collect(),
    );
    recover_alpha_boundary_polarity(image, &support, false, true)
}

fn recover_alpha_boundary_polarity(
    image: &Raster,
    matte: &crate::chroma::AlphaMatte,
    bright: bool,
    inward: bool,
) -> Vec<StructuralStroke> {
    let polarity = if bright { 1.0 } else { -1.0 };
    let mut strokes = Vec::new();
    // Bright partial-alpha areas retain their existing coverage model.
    if bright
        && (0..matte.len()).any(|i| {
            let a = matte.get(i);
            a > 0.0 && a < 1.0
        })
    {
        return strokes;
    }
    // Partial alpha at an antialiased dark rim is valid evidence. Exclude
    // invisible RGB from its profiles; the shared mask supplies coverage.
    let edge_sample = |p: Point, paint: [f32; 3]| {
        if bright {
            return sample(image, p);
        }
        let x = (p.x - 0.5).clamp(0.0, (image.width - 1) as f32);
        let y = (p.y - 0.5).clamp(0.0, (image.height - 1) as f32);
        let (ix, iy) = (x as usize, y as usize);
        let (tx, ty) = (x - ix as f32, y - iy as f32);
        let mut mass = 0.0;
        let mut color = [0.0; 3];
        for (sx, sy, weight) in [
            (ix, iy, (1.0 - tx) * (1.0 - ty)),
            ((ix + 1).min(image.width - 1), iy, tx * (1.0 - ty)),
            (ix, (iy + 1).min(image.height - 1), (1.0 - tx) * ty),
            (
                (ix + 1).min(image.width - 1),
                (iy + 1).min(image.height - 1),
                tx * ty,
            ),
        ] {
            let i = sy * image.width + sx;
            let weight = weight * matte.get(i);
            mass += weight;
            for (c, channel) in color.iter_mut().enumerate() {
                *channel += weight * image.pixels[i][c];
            }
        }
        if mass > 1e-5 {
            color.map(|c| c / mass)
        } else {
            paint
        }
    };
    for contour in matte.isocontours(0.5) {
        // Dark rims may be subpixel dots. Sample them more finely than
        // the bright, continuous rim so a span cannot bridge several dots.
        let spans = if bright {
            crate::geometry::alpha_contour_spans(&contour)
        } else {
            crate::geometry::alpha_contour_spans_with_step(&contour, 0.5)
        };
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
                .map(|i| edge_sample(offset(p, normal, i as f32 * 0.25), paint))
                .max_by(|a, b| (polarity * luma(*a)).total_cmp(&(polarity * luma(*b))))
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
                .max_by(|a, b| (polarity * luma(*a)).total_cmp(&(polarity * luma(*b))))
                .unwrap();
            let contrast = polarity * (luma(ink) - luma(paint));
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
                    let observed = edge_sample(
                        offset(spans[i].points[1], normal, (j as f32 + 0.5) * 0.25),
                        paint,
                    );
                    (polarity * (luma(observed) - luma(paint)) / contrast).clamp(0.0, 1.0) * 0.25
                })
                .sum();
            inks[i] = ink;
            if width > 0.05 && width < 1.8 {
                widths[i] = width;
            }
        }
        for i in 0..count {
            let Some((normal, paint, _)) = observations[i] else {
                continue;
            };
            let mut local: Vec<_> = (-2..=2)
                .map(|d| widths[(i as isize + d).rem_euclid(count as isize) as usize])
                .collect();
            local.sort_by(f32::total_cmp);
            // Keep the measured dark-dot gaps instead of median-filling them.
            let width = if bright { local[2] } else { widths[i] };
            if width < 0.12 || polarity * (luma(inks[i]) - luma(paint)) < 0.06 {
                continue;
            }
            let points = spans[i].points.to_vec();
            if inward {
                // Paint the measured band inside the silhouette. A stroke of
                // twice this width on the edge relied on a now-forbidden mask.
                for (colour, band_width, role) in [
                    (inks[i], width, "alpha-boundary-stroke"),
                    (paint, (width + 0.75).min(2.5), "alpha-boundary-underpaint"),
                ] {
                    strokes.push(StructuralStroke {
                        points: points
                            .iter()
                            .map(|&p| offset(p, normal, band_width * 0.5))
                            .collect(),
                        path_data: None,
                        precise_points: None,
                        color: colour,
                        width: band_width,
                        role,
                        width_samples: Vec::new(),
                    });
                }
                continue;
            }
            // Legacy uniform-coverage recovery retains its existing geometry.
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
            let connected = previous.points.last().unwrap().distance(stroke.points[0])
                < if inward { 0.35 } else { 1e-3 };
            if connected
                && previous.role == stroke.role
                && distance(previous.color, stroke.color) < 0.06
                && (previous.width - stroke.width).abs() < 0.25 * previous.width.max(0.5)
            {
                if inward {
                    previous.points.extend(stroke.points.into_iter().skip(1));
                    continue;
                }
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
    if inward {
        for stroke in &mut joined {
            stroke.path_data = Some(crate::geometry::fitted_structural_open_path_data(
                &stroke.points,
                0.15,
                0.5,
            ));
        }
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
            // Insufficient contrast is an unknown width, not evidence of
            // missing ink. In particular, dark shading can cross this gate
            // repeatedly along a continuous line. Keep the existing stroke
            // unless every profile can support the interruption decision.
            if contrast <= 0.15 {
                return None;
            }
            let mut mass = 0.0;
            let mut moment = 0.0;
            for k in -count..=count {
                let d = k as f32 * 0.25;
                let weight = ((background - visible(offset(p, normal, d))) / contrast)
                    .clamp(0.0, 1.0)
                    * 0.25;
                mass += weight;
                moment += weight * d;
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
include!("../tests/unit/stroke_model.rs");
