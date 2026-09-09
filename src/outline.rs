//! Reconstruct narrow outline bands from a common pair of smooth boundaries.
use crate::{
    chroma::AlphaMatte, geometry::Point, gradient::Paint, raster::Raster, segment::Segmentation,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub(crate) struct OutlineBand {
    pub outer: String,
    pub inner: String,
    pub underpaint: Paint,
    pub inner_underpaint: Option<(String, Paint)>,
    pub patches: Vec<(String, Paint)>,
    pub regions: HashSet<u32>,
    pub hidden: HashSet<u32>,
    pub boundary_underpaint: HashSet<u32>,
    pub contour: Vec<Point>,
    /// Depth from the outer repair boundary through the ink and inner fill.
    pub width: f32,
    pub pixels: Vec<(usize, bool)>,
}

fn inside(p: Point, polygon: &[Point]) -> bool {
    let mut result = false;
    for edge in polygon.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            result = !result;
        }
    }
    result
}
fn index(image: &Raster, p: Point) -> usize {
    (p.y.floor().clamp(0.0, image.height as f32 - 1.0) as usize) * image.width
        + p.x.floor().clamp(0.0, image.width as f32 - 1.0) as usize
}
fn luminance(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}
fn distance(p: Point, contour: &[Point]) -> f32 {
    contour
        .iter()
        .map(|q| p.distance(*q))
        .fold(f32::INFINITY, f32::min)
}
impl OutlineBand {
    pub fn hides_stroke(&self, points: &[Point], width: f32) -> bool {
        !points.is_empty()
            && points
                .iter()
                .all(|&p| distance(p, &self.contour) + width * 0.5 < self.width - 0.25)
    }
    pub fn supported_by_render(&self, source: &Raster, before: &Raster, after: &Raster) -> bool {
        use crate::color::{delta_e_ok, rgb_to_oklab};
        let mut sums = [0.0; 4];
        let mut rim_count = 0;
        for &(i, rim) in &self.pixels {
            let lab = rgb_to_oklab(source.pixels[i]);
            let old = delta_e_ok(lab, rgb_to_oklab(before.pixels[i]));
            let new = delta_e_ok(lab, rgb_to_oklab(after.pixels[i]));
            sums[0] += old;
            sums[1] += new;
            if rim {
                sums[2] += old;
                sums[3] += new;
                rim_count += 1;
            }
        }
        let count = self.pixels.len().max(1) as f32;
        let rims = rim_count.max(1) as f32;
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "outline render at {:?}: mean {} -> {}, rim {} -> {}",
                self.contour[0],
                sums[0] / count,
                sums[1] / count,
                sums[2] / rims,
                sums[3] / rims
            );
        }
        sums[1] / count <= sums[0] / count + 0.75 && sums[3] / rims <= sums[2] / rims + 3.0
    }
    pub fn clips_stroke(&self, points: &[Point]) -> bool {
        !points.is_empty()
            && points
                .iter()
                .all(|&p| inside(p, &self.contour) || distance(p, &self.contour) < 3.0)
    }
}

// A shaded round button can have a thicker rim on one side. Fit the small
// translation of its inner ellipse from width = base + normal dot offset.
// Keep a positive rim and a bounded displacement; final source-render checks
// remain responsible for accepting the complete reconstructed band.
fn ellipse_inset(profiles: &[(Point, f32)], initial_width: f32) -> (f32, Point) {
    let mut width = initial_width;
    let mut offset = Point::default();
    if profiles.len() < 6 {
        return (width, offset);
    }
    let count = profiles.len() as f32;
    let mx = profiles.iter().map(|(n, _)| n.x).sum::<f32>() / count;
    let my = profiles.iter().map(|(n, _)| n.y).sum::<f32>() / count;
    let mw = profiles.iter().map(|(_, w)| w).sum::<f32>() / count;
    let (mut xx, mut xy, mut yy, mut xw, mut yw) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (normal, w) in profiles {
        let (x, y, w) = (normal.x - mx, normal.y - my, w - mw);
        xx += x * x;
        xy += x * y;
        yy += y * y;
        xw += x * w;
        yw += y * w;
    }
    let determinant = xx * yy - xy * xy;
    if determinant > 1e-6 {
        let fitted = Point {
            x: (yy * xw - xy * yw) / determinant,
            y: (xx * yw - xy * xw) / determinant,
        };
        let length = fitted.x.hypot(fitted.y);
        if length > 0.25 && length <= 1.5_f32.min(width * 0.45) {
            offset = fitted;
            let mut adjusted: Vec<_> = profiles
                .iter()
                .map(|(n, w)| w - n.x * offset.x - n.y * offset.y)
                .collect();
            adjusted.sort_by(f32::total_cmp);
            width = adjusted[adjusted.len() * 3 / 8];
        }
    }
    (width, offset)
}

// A closed paint face can describe either side of its surrounding ink.
// Locate the outer crossing before measuring inward from a glass/fill edge;
// otherwise a real outline has a negative measured width and is discarded.
fn source_outer_offset(source: &Raster, points: &[Point]) -> Option<f32> {
    let n = points.len().checked_sub(1)?;
    if n < 7 {
        return None;
    }
    let sign = points
        .windows(2)
        .map(|p| p[0].x * p[1].y - p[1].x * p[0].y)
        .sum::<f32>()
        .signum();
    let mut offsets = Vec::new();
    let mut total = 0;
    for i in (0..n).step_by(3) {
        total += 1;
        let a = points[(i + n - 3) % n];
        let b = points[(i + 3) % n];
        let length = a.distance(b).max(0.001);
        let normal = Point {
            x: sign * (a.y - b.y) / length,
            y: sign * (b.x - a.x) / length,
        };
        let values = (0..41)
            .map(|j| {
                let t = j as f32 * 0.25 - 5.0;
                luminance(
                    source.pixels[index(
                        source,
                        Point {
                            x: points[i].x + normal.x * t,
                            y: points[i].y + normal.y * t,
                        },
                    )],
                )
            })
            .collect::<Vec<_>>();
        let core = (8..33).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
        let ink = values[core];
        if values[0] - ink < 0.04 || values[40] - ink < 0.08 {
            continue;
        }
        let threshold = ink + 0.35 * (values[0] - ink);
        if let Some(j) = (0..core).rev().find(|&j| values[j] > threshold) {
            let amount = (values[j] - threshold) / (values[j] - values[j + 1]).max(1e-6);
            offsets.push((j as f32 + amount) * 0.25 - 5.0);
        }
    }
    if offsets.is_empty() || offsets.len() * 4 < total * 3 {
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "outline outer crossing at {:?}: {}/{} supported",
                points[0],
                offsets.len(),
                total
            );
        }
        return None;
    }
    offsets.sort_by(f32::total_cmp);
    let shift = -offsets[offsets.len() / 2];
    #[cfg(feature = "diagnostics")]
    if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
        eprintln!(
            "outline outer crossing at {:?}: shift={shift}, spread={}",
            points[0],
            offsets[offsets.len() * 9 / 10] - offsets[offsets.len() / 10]
        );
    }
    ((0.5..=4.0).contains(&shift)
        && offsets[offsets.len() * 9 / 10] - offsets[offsets.len() / 10] <= 3.5)
        .then_some(shift)
}

pub(crate) fn propose(
    source: &Raster,
    matte: Option<&AlphaMatte>,
    segmentation: &Segmentation,
    contours: &[crate::geometry::ClosedContour],
) -> Vec<OutlineBand> {
    propose_with_alignment(source, matte, segmentation, contours, false)
}

fn propose_with_alignment(
    source: &Raster,
    matte: Option<&AlphaMatte>,
    segmentation: &Segmentation,
    contours: &[crate::geometry::ClosedContour],
    realigned: bool,
) -> Vec<OutlineBand> {
    let mut result: Vec<OutlineBand> = Vec::new();
    let repair_error = 6.2;
    let coverage = 0.35;
    // A geometric ellipse can be valid while its shaded band cannot meet the
    // color/path budget. Retain the bounded cubic alternative in that case;
    // an already proposed primary band owns its regions and excludes it.
    for model in contours
        .iter()
        .flat_map(|model| std::iter::once(model).chain(model.fallback.as_deref()))
    {
        let contour = &model.points;
        let n = contour.len().saturating_sub(1);
        if n < 80 {
            continue;
        }
        let area: f32 = contour
            .windows(2)
            .map(|e| e[0].x * e[1].y - e[1].x * e[0].y)
            .sum();
        let sign = area.signum();
        let mut widths = Vec::new();
        let mut profiles = Vec::new();
        let mut colors = Vec::new();
        let mut total = 0;
        for i in (0..n).step_by(3) {
            total += 1;
            let a = contour[(i + n - 3) % n];
            let b = contour[(i + 3) % n];
            let len = a.distance(b).max(0.001);
            let normal = Point {
                x: sign * (a.y - b.y) / len,
                y: sign * (b.x - a.x) / len,
            };
            let p = contour[i];
            let indices: Vec<_> = (0..33)
                .map(|j| {
                    index(
                        source,
                        Point {
                            x: p.x + normal.x * (j as f32 * 0.25 - 2.0),
                            y: p.y + normal.y * (j as f32 * 0.25 - 2.0),
                        },
                    )
                })
                .collect();
            let Some(core) = (0..20)
                .filter(|&j| matte.is_none_or(|m| m.get(indices[j]) > 0.9))
                .min_by(|&a, &b| {
                    luminance(source.pixels[indices[a]])
                        .total_cmp(&luminance(source.pixels[indices[b]]))
                })
            else {
                continue;
            };
            let ink = luminance(source.pixels[indices[core]]);
            let fill = luminance(source.pixels[*indices.last().unwrap()]);
            let outside = index(
                source,
                Point {
                    x: p.x - normal.x * 5.0,
                    y: p.y - normal.y * 5.0,
                },
            );
            if matte.is_none_or(|m| m.get(outside) > 0.1)
                && luminance(source.pixels[outside]) - ink < 0.04
            {
                continue;
            }

            if fill - ink < 0.08 {
                continue;
            }
            let threshold = ink + coverage * (fill - ink);
            let Some(end) =
                (core + 1..33).find(|&j| luminance(source.pixels[indices[j]]) > threshold)
            else {
                continue;
            };
            let width = end as f32 * 0.25 - 2.0;
            if !(0.5..=5.0).contains(&width) {
                continue;
            }
            widths.push(width);
            profiles.push((normal, width));
            colors.push((i, indices[core]));
        }
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "outline profile n={} support={}/{} start={:?}",
                n,
                widths.len(),
                total,
                contour[0]
            );
        }
        if widths.len() * 4 < total * 3 {
            if !realigned && matte.is_none() {
                if let Some(shift) = source_outer_offset(source, contour) {
                    if let Some(aligned) = crate::geometry::offset_outline(model, shift) {
                        #[cfg(feature = "diagnostics")]
                        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
                            eprintln!("outline align start={:?} shift={shift}", contour[0]);
                        }
                        for band in
                            propose_with_alignment(source, matte, segmentation, &[aligned], true)
                        {
                            if result.iter().all(|b| b.regions.is_disjoint(&band.regions)) {
                                result.push(band);
                            }
                        }
                    }
                }
            }
            continue;
        }
        widths.sort_by(f32::total_cmp);
        // Prefer the persistent dark core to the wider, bright bevel. The
        // complete rendered result is checked before this estimate is adopted.
        let width = widths[widths.len() * 3 / 8];
        let (width, offset) = if model.is_ellipse {
            ellipse_inset(&profiles, width)
        } else {
            (width, Point::default())
        };
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "outline width={} p10={} p90={} start={:?}",
                width,
                widths[widths.len() / 10],
                widths[widths.len() * 9 / 10],
                contour[0]
            );
        }
        if widths[widths.len() * 9 / 10] - widths[widths.len() / 10] > 3.5 {
            continue;
        }
        let shape = if realigned {
            crate::geometry::offset_outline(model, -width)
                .and_then(|inner| crate::geometry::outline_between(model, &inner))
        } else {
            crate::geometry::inset_outline(model, width, offset)
        };
        let Some(shape) = shape else {
            #[cfg(feature = "diagnostics")]
            if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
                eprintln!("outline inset rejected");
            }
            continue;
        };
        // Merely clipping at the ink edge leaves protrusions from the old
        // colour faces. Rebuild a narrow strip of the adjacent fill as well.
        let clearance = width + 1.25;
        let deeper = if realigned {
            crate::geometry::offset_outline(model, -clearance)
                .and_then(|inner| crate::geometry::outline_between(model, &inner))
        } else {
            crate::geometry::inset_outline(model, clearance, offset)
        };
        let Some(deeper) = deeper else {
            continue;
        };
        let exterior = if realigned {
            let Some(extended) = crate::geometry::offset_outline(model, 1.25) else {
                continue;
            };
            let Some(band) = crate::geometry::outline_between_on(model, &extended, model) else {
                continue;
            };
            Some(band)
        } else {
            None
        };
        let inner_backdrop = if realigned {
            let Some(middle) = crate::geometry::offset_outline(model, -width * 0.5)
                .and_then(|inner| crate::geometry::outline_between(model, &inner))
            else {
                continue;
            };
            Some(middle.inner)
        } else {
            None
        };
        let repair_colors: Vec<_> = colors
            .iter()
            .map(|&(i, _)| {
                let a = contour[(i + n - 3) % n];
                let b = contour[(i + 3) % n];
                let len = a.distance(b).max(0.001);
                let pixel = index(
                    source,
                    Point {
                        x: contour[i].x + sign * (a.y - b.y) / len * clearance + offset.x,
                        y: contour[i].y + sign * (b.x - a.x) / len * clearance + offset.y,
                    },
                );
                (i, pixel)
            })
            .collect();
        let exterior_colors: Vec<_> = colors
            .iter()
            .map(|&(i, _)| {
                let a = contour[(i + n - 3) % n];
                let b = contour[(i + 3) % n];
                let len = a.distance(b).max(0.001);
                (
                    i,
                    index(
                        source,
                        Point {
                            x: contour[i].x - sign * (a.y - b.y) / len * 1.25,
                            y: contour[i].y - sign * (b.x - a.x) / len * 1.25,
                        },
                    ),
                )
            })
            .collect();
        // Single-pixel ink cores alternate with antialias coverage. Estimate
        // their persistent colour along the contour before fitting a field.
        let smooth_reference = |samples: &[(usize, usize)]| {
            let mut image = source.clone();
            // Preserve localized bevel highlights on small round controls.
            let window = if model.is_ellipse { 3 } else { 7 };
            for (j, &(_, pixel)) in samples.iter().enumerate() {
                image.pixels[pixel] = [0, 1, 2].map(|channel| {
                    let mut values: Vec<_> = (0..window)
                        .map(|k| {
                            source.pixels
                                [samples[(j + samples.len() + k - window / 2) % samples.len()].1]
                                [channel]
                        })
                        .collect();
                    values.sort_by(f32::total_cmp);
                    values[window / 2]
                });
            }
            image
        };
        let color_reference = smooth_reference(&colors);
        let repair_reference = smooth_reference(&repair_colors);
        let exterior_reference = smooth_reference(&exterior_colors);
        // At black, even one 8-bit RGB level spans about 6.7 scaled OKLab
        // units. A mixed 0/1 core can exceed the old mean-error budget solely
        // through quantization. Keep the tighter budget for coloured rims.
        let mut ink_lightness: Vec<_> = colors
            .iter()
            .map(|&(_, i)| crate::color::rgb_to_oklab(color_reference.pixels[i]).l)
            .collect();
        ink_lightness.sort_by(f32::total_cmp);
        let ink_error = if ink_lightness[ink_lightness.len() / 2] < 20.0 {
            4.0
        } else {
            3.1
        };
        let mut patches = Vec::new();
        let mut underpaint = None;
        let mut inner_underpaint = None;
        let mut intervals = vec![(0.0_f32, 1.0_f32, 0_u8)];
        while let Some((a, b, depth)) = intervals.pop() {
            let mut pixels: Vec<_> = colors
                .iter()
                .filter(|(i, _)| (*i as f32 / n as f32) >= a && (*i as f32 / n as f32) < b)
                .map(|&(_, pixel)| pixel)
                .collect();
            pixels.sort_unstable();
            pixels.dedup();
            let mut repair_pixels: Vec<_> = repair_colors
                .iter()
                .filter(|(i, _)| (*i as f32 / n as f32) >= a && (*i as f32 / n as f32) < b)
                .map(|&(_, i)| i)
                .collect();
            repair_pixels.sort_unstable();
            repair_pixels.dedup();
            let ink = crate::gradient::fit_outline_field(&color_reference, &pixels, ink_error);
            let repair =
                crate::gradient::fit_outline_field(&repair_reference, &repair_pixels, repair_error);
            let outside = if exterior.is_some() {
                let mut pixels: Vec<_> = exterior_colors
                    .iter()
                    .filter(|(i, _)| (*i as f32 / n as f32) >= a && (*i as f32 / n as f32) < b)
                    .map(|&(_, i)| i)
                    .collect();
                pixels.sort_unstable();
                pixels.dedup();
                crate::gradient::fit_outline_field(&exterior_reference, &pixels, repair_error)
                    .map(Some)
            } else {
                Some(None)
            };
            if let (Some(paint), Some(repair), Some(outside)) = (ink, repair, outside) {
                if underpaint.is_none() {
                    // Each side of the ink needs its own backdrop. Extending
                    // the glass colour beneath the exterior clip leaves a
                    // blue hairline where complementary antialias meets.
                    underpaint = Some(outside.as_ref().unwrap_or(&repair).clone());
                    inner_underpaint = inner_backdrop
                        .as_ref()
                        .map(|path| (path.clone(), repair.clone()));
                }
                let path = if depth == 0 {
                    format!("{} {}", shape.outer, shape.inner)
                } else {
                    shape.patch(a, b)
                };
                let repair_path = if depth == 0 {
                    format!("{} {}", shape.inner, deeper.inner)
                } else {
                    shape.repair_patch(&deeper, a, b)
                };
                patches.push((repair_path, repair));
                if let (Some(exterior), Some(outside)) = (&exterior, outside) {
                    let path = if depth == 0 {
                        format!("{} {}", exterior.outer, exterior.inner)
                    } else {
                        exterior.patch(a, b)
                    };
                    patches.push((path, outside));
                }
                patches.push((path, paint));
            } else if depth < 5
                && patches.len() / if exterior.is_some() { 3 } else { 2 } + intervals.len() + 2
                    <= 16
            {
                // Split at observed color transitions instead of arbitrary
                // loop fractions, under the same depth and path budgets.
                let middle = colors
                    .windows(2)
                    .zip(repair_colors.windows(2))
                    .zip(exterior_colors.windows(2))
                    .filter_map(|((ink, repair), outside)| {
                        let t = (ink[0].0 + ink[1].0) as f32 / (2.0 * n as f32);
                        if t < a + (b - a) * 0.25 || t > b - (b - a) * 0.25 {
                            return None;
                        }
                        let difference = |image: &Raster, samples: &[(usize, usize)]| {
                            crate::color::delta_e_ok(
                                crate::color::rgb_to_oklab(image.pixels[samples[0].1]),
                                crate::color::rgb_to_oklab(image.pixels[samples[1].1]),
                            )
                        };
                        Some((
                            t,
                            (difference(&color_reference, ink) / ink_error)
                                .max(difference(&repair_reference, repair) / repair_error)
                                .max(if exterior.is_some() {
                                    difference(&exterior_reference, outside) / repair_error
                                } else {
                                    0.0
                                }),
                        ))
                    })
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .map_or((a + b) * 0.5, |(t, _)| t);
                intervals.push((middle, b, depth + 1));
                intervals.push((a, middle, depth + 1));
            } else {
                patches.clear();
                break;
            }
        }
        if patches.is_empty() {
            continue;
        }
        let contour = exterior
            .as_ref()
            .map_or(shape.contour, |e| e.contour.clone());
        let outer = exterior.as_ref().map_or(shape.outer, |e| e.outer.clone());
        let inner = deeper.inner;
        let clearance = clearance + if exterior.is_some() { 1.25 } else { 0.0 };
        let width = clearance - offset.x.hypot(offset.y);
        let minx = contour
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as usize;
        let maxx = contour
            .iter()
            .map(|p| p.x)
            .fold(0.0, f32::max)
            .ceil()
            .min(source.width as f32) as usize;
        let miny = contour
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as usize;
        let maxy = contour
            .iter()
            .map(|p| p.y)
            .fold(0.0, f32::max)
            .ceil()
            .min(source.height as f32) as usize;
        let mut counts = HashMap::<u32, (usize, usize)>::new();
        let mut boundary_underpaint = HashSet::new();
        let mut measured_pixels = Vec::new();
        for y in miny..maxy {
            for x in minx..maxx {
                let p = Point {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                };
                if !inside(p, &contour) && distance(p, &contour) > 2.5 {
                    continue;
                }
                let pixel = y * source.width + x;
                if distance(p, &contour) <= 2.5 {
                    boundary_underpaint.insert(segmentation.labels[pixel]);
                }
                if inside(p, &contour) {
                    measured_pixels.push((
                        pixel,
                        distance(p, &contour) <= clearance + offset.x.hypot(offset.y) + 1.0,
                    ));
                }
                let e = counts.entry(segmentation.labels[pixel]).or_default();
                e.0 += 1;
                if inside(p, &contour) && distance(p, &contour) > width + 0.5 {
                    e.1 += 1;
                }
            }
        }
        let regions: HashSet<_> = counts
            .iter()
            .filter(|(id, (count, _))| {
                *count as f32 >= segmentation.regions[**id as usize].area as f32 * 0.85
            })
            .map(|(&id, _)| id)
            .collect();
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
            eprintln!(
                "outline ownership at {:?}: {} regions",
                contour[0],
                regions.len()
            );
        }
        if regions.len() < 3 || result.iter().any(|b| !b.regions.is_disjoint(&regions)) {
            continue;
        }
        let hidden = counts
            .iter()
            .filter(|(id, (_, interior))| regions.contains(id) && *interior == 0)
            .map(|(&id, _)| id)
            .collect();
        let pixels = measured_pixels
            .into_iter()
            .filter(|(i, _)| exterior.is_some() || regions.contains(&segmentation.labels[*i]))
            .collect();
        result.push(OutlineBand {
            outer,
            inner,
            underpaint: underpaint.unwrap(),
            inner_underpaint,
            patches,
            regions,
            hidden,
            boundary_underpaint,
            contour,
            width,
            pixels,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        color::rgb_to_oklab,
        segment::{RegionStats, SegmentationSummary},
    };

    #[test]
    fn ellipse_inset_recovers_bounded_asymmetric_rims() {
        let profile = |shift: Point| {
            (0..64)
                .map(|i| {
                    let angle = i as f32 * std::f32::consts::TAU / 64.0;
                    let normal = Point {
                        x: angle.cos(),
                        y: angle.sin(),
                    };
                    (normal, 3.0 + normal.x * shift.x + normal.y * shift.y)
                })
                .collect::<Vec<_>>()
        };
        let expected = Point { x: -0.5, y: 0.6 };
        let (width, offset) = ellipse_inset(&profile(expected), 3.0);
        assert!((width - 3.0).abs() < 1e-4);
        assert!(offset.distance(expected) < 1e-4);
        assert_eq!(
            ellipse_inset(&profile(Point { x: 2.0, y: 0.0 }), 3.0),
            (3.0, Point::default())
        );
        assert_eq!(
            ellipse_inset(&[(Point { x: 1.0, y: 0.0 }, 3.0); 8], 3.0),
            (3.0, Point::default())
        );
    }

    fn disc(background: [f32; 3]) -> (Raster, Segmentation, Vec<Point>) {
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        for y in 0..96 {
            for x in 0..96 {
                let radius = (x as f32 + 0.5 - 48.0).hypot(y as f32 + 0.5 - 48.0);
                let id = if radius >= 30.0 {
                    0
                } else if radius >= 28.0 {
                    1
                } else if x < 48 {
                    2
                } else {
                    3
                };
                labels.push(id);
                pixels.push(match id {
                    0 => background,
                    1 => [0.05, 0.1, 0.2],
                    2 => [0.4, 0.65, 0.9],
                    _ => [0.42, 0.67, 0.9],
                });
            }
        }
        let source = Raster::new(96, 96, pixels);
        let regions = (0..4)
            .map(|id| {
                let owned: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, label)| **label == id)
                    .map(|(i, _)| i)
                    .collect();
                let color = source.pixels[owned[0]];
                RegionStats {
                    id,
                    area: owned.len(),
                    min_x: 0,
                    min_y: 0,
                    max_x: 95,
                    max_y: 95,
                    mean_rgb: color,
                    mean_lab: rgb_to_oklab(color),
                }
            })
            .collect();
        let segmentation = Segmentation {
            width: 96,
            height: 96,
            labels,
            paint_keys: vec![0, 1, 2, 3],
            paint_samples: vec![true; 96 * 96],
            canonical: source.clone(),
            regions,
            summary: SegmentationSummary::default(),
        };
        let mut contour: Vec<_> = (0..360)
            .map(|i| {
                let theta = i as f32 * std::f32::consts::TAU / 360.0;
                Point {
                    x: 48.0 + 30.0 * theta.cos(),
                    y: 48.0 + 30.0 * theta.sin(),
                }
            })
            .collect();
        contour.push(contour[0]);
        (source, segmentation, contour)
    }

    #[test]
    fn reconstructs_ink_outside_a_fill_contour_in_both_directions() {
        let (source, segmentation, outer) = disc([1.0; 3]);
        let inner: Vec<_> = outer
            .iter()
            .map(|p| Point {
                x: 48.0 + (p.x - 48.0) * 28.0 / 30.0,
                y: 48.0 + (p.y - 48.0) * 28.0 / 30.0,
            })
            .collect();
        for reverse in [false, true] {
            let mut points = inner.clone();
            if reverse {
                points.reverse();
            }
            let shift = source_outer_offset(&source, &points).unwrap();
            assert!((1.5..=2.5).contains(&shift), "outer offset: {shift}");
            let bands = propose(
                &source,
                None,
                &segmentation,
                &[crate::geometry::ClosedContour::from_points(&points)],
            );
            assert_eq!(bands.len(), 1, "reversed={reverse}");
            assert!(bands[0].regions.contains(&1));
            // The band owns the ink and both adjacent 1.25px fill collars.
            assert!((3.5..=5.5).contains(&bands[0].width));
        }
        assert!(source_outer_offset(&source, &outer).is_none());
        let (plain, _, _) = disc([0.0; 3]);
        assert!(source_outer_offset(&plain, &inner).is_none());
    }

    #[test]
    fn reconstructs_a_ridge_but_not_a_plain_brightness_edge() {
        let (source, segmentation, contour) = disc([1.0; 3]);
        let bands = propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)],
        );
        assert_eq!(bands.len(), 1);
        assert!((2.0..=4.5).contains(&bands[0].width));
        assert!(bands[0].regions.contains(&1));
        let (source, segmentation, contour) = disc([0.0; 3]);
        assert!(propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)]
        )
        .is_empty());
    }

    #[test]
    fn rendered_colour_changes_and_erased_detail_are_rejected() {
        let (source, segmentation, contour) = disc([1.0; 3]);
        let band = propose(
            &source,
            None,
            &segmentation,
            &[crate::geometry::ClosedContour::from_points(&contour)],
        )
        .remove(0);
        assert!(band.supported_by_render(&source, &source, &source));
        let mut damaged = source.clone();
        for &(i, _) in &band.pixels {
            damaged.pixels[i] = [1.0, 0.0, 0.0];
        }
        assert!(!band.supported_by_render(&source, &source, &damaged));
        let mut gap_source = source.clone();
        for &(i, rim) in &band.pixels {
            if rim && i % 96 > 48 {
                gap_source.pixels[i] = [1.0; 3];
            }
        }
        assert!(!band.supported_by_render(&gap_source, &gap_source, &source));
    }
}
