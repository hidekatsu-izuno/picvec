//! Reconstruct narrow outline bands from a common pair of smooth boundaries.
use crate::{
    chroma::AlphaMatte, geometry::Point, gradient::Paint, raster::Raster, segment::Segmentation,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub(crate) struct OutlineBand {
    pub outer: String,
    pub inner: String,
    pub patches: Vec<(String, Paint)>,
    pub regions: HashSet<u32>,
    pub hidden: HashSet<u32>,
    pub boundary_underpaint: HashSet<u32>,
    pub contour: Vec<Point>,
    /// Depth of the repaired collar, including the ink and its inner fill.
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
        use crate::color::{delta_e2000, rgb_to_lab};
        let mut sums = [0.0; 4];
        let mut rim_count = 0;
        for &(i, rim) in &self.pixels {
            let lab = rgb_to_lab(source.pixels[i]);
            let old = delta_e2000(lab, rgb_to_lab(before.pixels[i]));
            let new = delta_e2000(lab, rgb_to_lab(after.pixels[i]));
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

pub(crate) fn propose(
    source: &Raster,
    matte: Option<&AlphaMatte>,
    segmentation: &Segmentation,
    contours: &[crate::geometry::ClosedContour],
) -> Vec<OutlineBand> {
    let mut result: Vec<OutlineBand> = Vec::new();
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
            let threshold = ink + 0.35 * (fill - ink);
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
            eprintln!("outline profile n={} support={}/{}", n, widths.len(), total);
        }
        if widths.len() * 4 < total * 3 {
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
        let Some(shape) = crate::geometry::inset_outline(model, width, offset) else {
            #[cfg(feature = "diagnostics")]
            if std::env::var_os("PICVEC_OUTLINE_DIAGNOSTICS").is_some() {
                eprintln!("outline inset rejected");
            }
            continue;
        };
        // Merely clipping at the ink edge leaves protrusions from the old
        // colour faces. Rebuild a narrow strip of the adjacent fill as well.
        let clearance = width + 1.25;
        let Some(deeper) = crate::geometry::inset_outline(model, clearance, offset) else {
            continue;
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
        let mut patches = Vec::new();
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
            let ink = crate::gradient::fit_outline_field(&color_reference, &pixels, 5.0);
            let repair =
                crate::gradient::fit_outline_field(&repair_reference, &repair_pixels, 10.0);
            if let (Some(paint), Some(repair)) = (ink, repair) {
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
                patches.push((path, paint));
            } else if depth < 4 && patches.len() / 2 + intervals.len() + 2 <= 8 {
                // Split at observed color transitions instead of arbitrary
                // loop fractions, under the same depth and path budgets.
                let middle = colors
                    .windows(2)
                    .zip(repair_colors.windows(2))
                    .filter_map(|(ink, repair)| {
                        let t = (ink[0].0 + ink[1].0) as f32 / (2.0 * n as f32);
                        if t < a + (b - a) * 0.25 || t > b - (b - a) * 0.25 {
                            return None;
                        }
                        let difference = |image: &Raster, samples: &[(usize, usize)]| {
                            crate::color::delta_e2000(
                                crate::color::rgb_to_lab(image.pixels[samples[0].1]),
                                crate::color::rgb_to_lab(image.pixels[samples[1].1]),
                            )
                        };
                        Some((
                            t,
                            (difference(&color_reference, ink) / 5.0)
                                .max(difference(&repair_reference, repair) / 10.0),
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
        let contour = shape.contour;
        let outer = shape.outer;
        let inner = deeper.inner;
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
            .filter(|(i, _)| regions.contains(&segmentation.labels[*i]))
            .collect();
        result.push(OutlineBand {
            outer,
            inner,
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
        color::rgb_to_lab,
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
                    mean_lab: rgb_to_lab(color),
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
