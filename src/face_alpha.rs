//! Native opacity belongs to each colour face, never to a masking layer.
use crate::{
    alpha_coverage::Coverage,
    chroma::AlphaMatte,
    gradient::{self, Paint},
    raster::Raster,
    segment::{self, Segmentation},
};

#[derive(Clone, Debug)]
pub(crate) struct FaceAlpha {
    pub fields: Vec<Paint>,
    pub ink_opacity: f32,
    pub composite_layers: Vec<crate::neutral_fields::Layer>,
    pub source_fields: Vec<crate::colour_fields::Patch>,
    pub bands: Vec<(crate::alpha_lines::AlphaBand, [f32; 3])>,
}

pub(crate) fn uniform_opacity(matte: &AlphaMatte, coverage: Option<&Coverage>) -> Option<f32> {
    coverage.map(|c| c.opacity).or_else(|| {
        let visible = (0..matte.len()).filter(|&i| matte.get(i) > 0.5).count();
        let opaque = (0..matte.len())
            .filter(|&i| matte.get(i) >= 254.5 / 255.0)
            .count();
        (opaque * 2 >= visible && matte.vectorized_levels().iter().all(|&a| a == 0 || a == 3))
            .then_some(1.0)
    })
}

pub(crate) fn prepare(
    source: &Raster,
    matte: &AlphaMatte,
    coverage: Option<&Coverage>,
    local: Option<&crate::alpha_coverage::LocalCoverage>,
    segmentation: &mut Segmentation,
    paints: &mut Vec<Paint>,
) -> FaceAlpha {
    let mut bands = Vec::new();
    let mut cleared = Vec::new();
    if coverage.is_none() {
        for band in crate::alpha_lines::extract(matte).1 {
            if local.is_some_and(|local| band.pixels.iter().all(|&i| local.opacity[i] > 0.0)) {
                continue;
            }
            let mut colour = [0.0; 3];
            for &i in &band.pixels {
                for c in 0..3 {
                    colour[c] += source.pixels[i][c] / band.pixels.len() as f32;
                }
            }
            if band
                .pixels
                .iter()
                .all(|&i| (0..3).all(|c| (source.pixels[i][c] - colour[c]).abs() <= 4.0 / 255.0))
            {
                cleared.extend(&band.pixels);
                bands.push((band, colour));
            }
        }
    }
    let remaining = matte.cleared(&cleared);
    let matte = &remaining;
    let flat = uniform_opacity(matte, coverage);
    let support: Vec<_> = (0..matte.len())
        .map(|i| {
            // Keep visible rim colours, including subpixel ink below half
            // coverage. Geometry fits the common alpha isocontour later.
            matte.get(i) > flat.map_or(0.0, |a| a * 0.5)
        })
        .collect();
    let (parents, visible) = segment::split_partition_by_values(source, segmentation, &support);
    *paints = parents.iter().map(|&p| paints[p].clone()).collect();
    if let Some(opacity) = flat {
        return FaceAlpha {
            fields: visible
                .iter()
                .map(|&yes| Paint::Solid {
                    color: [if yes { opacity } else { 0.0 }; 3],
                })
                .collect(),
            ink_opacity: opacity,
            composite_layers: Vec::new(),
            source_fields: Vec::new(),
            bands,
        };
    }
    let alpha = Raster::new(
        source.width,
        source.height,
        (0..matte.len()).map(|i| [matte.get(i); 3]).collect(),
    );
    let mut pixels = vec![Vec::new(); paints.len()];
    for (i, &label) in segmentation.labels.iter().enumerate() {
        pixels[label as usize].push(i);
    }
    let mut core_fields = vec![None; paints.len()];
    for (id, indices) in pixels.iter().enumerate() {
        if visible[id] {
            if let Some((colour, opacity)) = material_rgba_field(
                source,
                matte,
                &alpha,
                indices,
                id as u32,
                &segmentation.labels,
                &paints[id],
            ) {
                paints[id] = colour;
                core_fields[id] = Some(opacity);
            }
        }
    }
    for (id, indices) in pixels.iter().enumerate() {
        if core_fields[id].is_some() || local_face_opacity(local, matte, indices).is_some() {
            continue;
        }
        if visible[id]
            && matches!(
                paints[id],
                Paint::Linear { .. } | Paint::Radial { .. } | Paint::Layered { .. }
            )
            && gradient::fit_alpha_on_paint(&alpha, indices, &paints[id]).is_none()
        {
            if let Some(field) = gradient::fit_alpha_field(&alpha, indices) {
                if let Some(colour) =
                    gradient::fit_colour_on_alpha(source, indices, &paints[id], &field)
                {
                    paints[id] = colour;
                }
            }
        }
    }
    let mut fields: Vec<_> = pixels
        .iter()
        .enumerate()
        .map(|(id, indices)| {
            if !visible[id] {
                return Some(Paint::Solid { color: [0.0; 3] });
            }
            if let Some(field) = &core_fields[id] {
                return Some(field.clone());
            }
            if let Some(opacity) = local_face_opacity(local, matte, indices) {
                return Some(Paint::Solid {
                    color: [opacity; 3],
                });
            }
            gradient::fit_alpha_on_paint(&alpha, indices, &paints[id]).or_else(|| {
                gradient::fit_alpha_field(&alpha, indices).filter(|field| {
                    matches!(field, Paint::Solid { .. })
                        || matches!(paints[id], Paint::Solid { .. })
                })
            })
        })
        .collect();
    // Bound refinement before geometry fitting. Commit only if every local
    // field passes the same error gates and fewer faces replace alpha bands.
    let mut patch_keys = vec![0usize; source.pixels.len()];
    let mut patches = vec![None];
    for (id, indices) in pixels.iter().enumerate() {
        if matches!(paints[id], Paint::Layered { .. }) || fields[id].is_some() || indices.len() < 32
        {
            continue;
        }
        let mut eligible = std::collections::HashSet::new();
        for &i in indices {
            if opaque_rim_sample(
                source,
                matte,
                i,
                id as u32,
                &segmentation.labels,
                gradient::paint_at(&paints[id], i, source.width),
            ) {
                eligible.insert(i);
            }
        }
        if eligible.len() < 32 {
            continue;
        }
        let mut old_edges = 0;
        let mut new_edges = 0;
        for &i in indices {
            for j in [
                (i % source.width + 1 < source.width).then_some(i + 1),
                (i + source.width < source.pixels.len()).then_some(i + source.width),
            ]
            .into_iter()
            .flatten()
            {
                if segmentation.labels[j] != id as u32 {
                    continue;
                }
                let a = (matte.get(i) * 255.0).round() as u16;
                let b = (matte.get(j) * 255.0).round() as u16;
                old_edges += usize::from(a != b);
                new_edges += usize::from(
                    (if eligible.contains(&i) { 256 } else { a })
                        != (if eligible.contains(&j) { 256 } else { b }),
                );
            }
        }
        if old_edges == 0 || new_edges * 4 >= old_edges * 3 {
            continue;
        }
        let key = patches.len();
        patches.push(Some((paints[id].clone(), Paint::Solid { color: [1.0; 3] })));
        let clear_key = patches.len();
        patches.push(Some((paints[id].clone(), Paint::Solid { color: [0.0; 3] })));
        for i in eligible {
            // Once coverage is explained, use its half-coverage silhouette.
            // Keeping every nonzero AA sample and assigning opacity one
            // expands the edge and turns shallow shoe rims into polygons.
            let outside = matte.get(i) < 0.5 && {
                let (x, y) = (i % source.width, i / source.width);
                (y.saturating_sub(2)..=(y + 2).min(source.height - 1)).any(|yy| {
                    (x.saturating_sub(2)..=(x + 2).min(source.width - 1))
                        .any(|xx| matte.get(yy * source.width + xx) == 0.0)
                })
            };
            patch_keys[i] = if outside { clear_key } else { key };
        }
    }
    for (id, indices) in pixels.iter().enumerate() {
        if fields[id].is_some()
            || indices.len() < 32
            || !matches!(paints[id], Paint::Linear { .. } | Paint::Radial { .. })
            || indices.iter().any(|&i| patch_keys[i] != 0)
        {
            continue;
        }
        let codes: std::collections::BTreeSet<_> = indices
            .iter()
            .map(|&i| (matte.get(i) * 255.0).round() as u8)
            .collect();
        if codes.len() < 8 {
            continue;
        }
        if let Some(parts) = local_gradient_fields(source, &alpha, indices, &paints[id], 0) {
            if parts.len() > 16 || parts.len() >= codes.len() {
                continue;
            }
            for (indices, paint, opacity) in parts {
                let key = patches.len();
                for i in indices {
                    patch_keys[i] = key;
                }
                patches.push(Some((paint, opacity)));
            }
        }
    }
    if patches.len() > 1 {
        let (parents, keys) = segment::split_partition_by_values(source, segmentation, &patch_keys);
        let previous_paints = std::mem::take(paints);
        let previous_fields = std::mem::take(&mut fields);
        for (&parent, key) in parents.iter().zip(keys) {
            if let Some((paint, opacity)) = &patches[key] {
                paints.push(paint.clone());
                fields.push(Some(opacity.clone()));
            } else {
                paints.push(previous_paints[parent].clone());
                fields.push(previous_fields[parent].clone());
            }
        }
    }
    // Correct ownership before exact-alpha splitting, using the same Paint
    // faces and shared geometry as every other material. No late overlay or
    // rectangular replacement can open a seam with an incident face.
    let recovered = crate::colour_fields::narrow_material_pixels(source, matte);
    if !recovered.is_empty() {
        let mut material_ids = std::collections::BTreeMap::new();
        let mut labels = segmentation.labels.clone();
        for (i, rgb, visible) in recovered {
            let id = *material_ids.entry((rgb, visible)).or_insert_with(|| {
                let id = paints.len() as u32;
                paints.push(Paint::Solid {
                    color: rgb.map(|c| c as f32 / 255.0),
                });
                fields.push(Some(Paint::Solid {
                    color: [if visible { 1.0 } else { 0.0 }; 3],
                }));
                id
            });
            labels[i] = id;
            segmentation.canonical.pixels[i] = if visible {
                rgb.map(|c| c as f32 / 255.0)
            } else {
                [1.0; 3]
            };
        }
        let mut parents = labels.clone();
        parents.sort_unstable();
        parents.dedup();
        let previous_paints = std::mem::take(paints);
        let previous_fields = std::mem::take(&mut fields);
        *paints = parents
            .iter()
            .map(|&p| previous_paints[p as usize].clone())
            .collect();
        fields = parents
            .iter()
            .map(|&p| previous_fields[p as usize].clone())
            .collect();
        segment::replace_source_supported_paint_labels(source, segmentation, labels, 0);
    }
    let values: Vec<_> = segmentation
        .labels
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            let id = label as usize;
            let layered = matches!(paints[id], Paint::Layered { .. })
                && !matches!(&fields[id], Some(Paint::Solid {color}) if color[0] >= 0.99);
            if fields[id].is_none() || layered {
                let rgb = if layered {
                    gradient::paint_at(&paints[id], i, source.width)
                        .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
                } else {
                    [0; 3]
                };
                ((matte.get(i) * 255.0).round() as u8, rgb, layered)
            } else {
                (0, [0; 3], false)
            }
        })
        .collect();
    let (parents, values) = segment::split_partition_by_values(source, segmentation, &values);
    let new_fields = parents
        .iter()
        .zip(&values)
        .map(|(&parent, &(a, _, layered))| {
            if fields[parent].is_none() || layered {
                Paint::Solid {
                    color: [a as f32 / 255.0; 3],
                }
            } else {
                fields[parent].clone().unwrap()
            }
        })
        .collect();
    *paints = parents
        .iter()
        .zip(values)
        .map(|(&parent, (_, rgb, layered))| {
            if layered {
                Paint::Solid {
                    color: rgb.map(|c| c as f32 / 255.0),
                }
            } else {
                paints[parent].clone()
            }
        })
        .collect();
    FaceAlpha {
        fields: new_fields,
        // Structural analysis protects every non-opaque source pixel before
        // transferring ink. Remaining strokes belong to opaque material even
        // when another face elsewhere has authored variable transparency.
        ink_opacity: 1.0,
        composite_layers: Vec::new(),
        source_fields: Vec::new(),
        bands,
    }
}

/// Fit material alpha on interior samples. Boundary samples also contain
/// raster coverage from neighbouring materials, not just the face's opacity.
#[allow(clippy::too_many_arguments)]
fn material_rgba_field(
    source: &Raster,
    matte: &AlphaMatte,
    alpha: &Raster,
    pixels: &[usize],
    owner: u32,
    labels: &[u32],
    paint: &Paint,
) -> Option<(Paint, Paint)> {
    if pixels.len() < 32 || matches!(paint, Paint::Layered { .. }) {
        return None;
    }
    let w = source.width;
    let h = source.height;
    let core: Vec<_> = pixels
        .iter()
        .copied()
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            x > 0
                && y > 0
                && x + 1 < w
                && y + 1 < h
                && [i - 1, i + 1, i - w, i + w]
                    .iter()
                    .all(|&j| labels[j] == owner)
        })
        .collect();
    let (colour, field) = if core.len() < 16 || core.len() * 4 < pixels.len() {
        let Paint::Solid { color } = paint else {
            return None;
        };
        // Structural ownership can remove the opaque centre of a rim from
        // this Paint face. Borrow only same-colour opaque source support
        // within two pixels, and only near transparency. The RGBA mixture
        // gate below still has to explain every fractional boundary sample.
        for &i in pixels {
            if matte.get(i) >= 250.0 / 255.0 {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let opaque_ink = (y.saturating_sub(2)..=(y + 2).min(h - 1)).any(|yy| {
                (x.saturating_sub(2)..=(x + 2).min(w - 1)).any(|xx| {
                    let j = yy * w + xx;
                    matte.get(j) >= 250.0 / 255.0
                        && (0..3).all(|c| (source.pixels[j][c] - color[c]).abs() <= 4.0 / 255.0)
                })
            });
            let near_clear = (y.saturating_sub(7)..=(y + 7).min(h - 1)).any(|yy| {
                (x.saturating_sub(7)..=(x + 7).min(w - 1)).any(|xx| matte.get(yy * w + xx) <= 0.0)
            });
            if !opaque_ink || !near_clear {
                return None;
            }
        }
        (paint.clone(), Paint::Solid { color: [1.0; 3] })
    } else {
        let fitted = if let Some(field) = gradient::fit_alpha_on_paint(alpha, &core, paint) {
            (paint.clone(), field)
        } else {
            let field = gradient::fit_alpha_field(alpha, &core)?;
            let colour =
                if matches!(field, Paint::Solid { .. }) || matches!(paint, Paint::Solid { .. }) {
                    paint.clone()
                } else {
                    gradient::fit_colour_on_alpha(source, &core, paint, &field)?
                };
            (colour, field)
        };
        fitted
    };
    for &i in pixels {
        let predicted_alpha = gradient::paint_at(&field, i, w)[0];
        let actual_alpha = matte.get(i);
        if (predicted_alpha - actual_alpha).abs() <= 4.0 / 255.0 {
            continue;
        }
        let predicted = gradient::paint_at(&colour, i, w).map(|c| c * predicted_alpha);
        let actual = source.pixels[i].map(|c| c * actual_alpha);
        let (x, y) = (i % w, i / w);
        let covered = (y.saturating_sub(1)..=(y + 1).min(h - 1)).any(|yy| {
            (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| {
                let j = yy * w + xx;
                if labels[j] == owner {
                    return false;
                }
                let other_alpha = matte.get(j);
                let delta = other_alpha - predicted_alpha;
                if delta.abs() < 1e-6 {
                    return false;
                }
                let t = (actual_alpha - predicted_alpha) / delta;
                (0.0..=1.0).contains(&t)
                    && (0..3).all(|c| {
                        let mixed =
                            predicted[c] * (1.0 - t) + source.pixels[j][c] * other_alpha * t;
                        (mixed - actual[c]).abs() <= 4.0 / 255.0
                    })
            })
        });
        if !covered {
            return None;
        }
    }
    Some((colour, field))
}

fn opaque_rim_sample(
    source: &Raster,
    matte: &AlphaMatte,
    i: usize,
    owner: u32,
    labels: &[u32],
    colour: [f32; 3],
) -> bool {
    let a = matte.get(i);
    if a >= 254.5 / 255.0 {
        return true;
    }
    let (w, h) = (source.width, source.height);
    let (x, y) = (i % w, i / w);
    let core = (y.saturating_sub(2)..=(y + 2).min(h - 1)).any(|yy| {
        (x.saturating_sub(2)..=(x + 2).min(w - 1)).any(|xx| {
            let j = yy * w + xx;
            matte.get(j) >= 254.5 / 255.0
                && (0..3).all(|c| (source.pixels[j][c] - colour[c]).abs() <= 4.0 / 255.0)
        })
    });
    if !core {
        return false;
    }
    let clear = (y.saturating_sub(7)..=(y + 7).min(h - 1)).any(|yy| {
        (x.saturating_sub(7)..=(x + 7).min(w - 1)).any(|xx| matte.get(yy * w + xx) == 0.0)
    });
    (y.saturating_sub(1)..=(y + 1).min(h - 1)).any(|yy| {
        (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| {
            let j = yy * w + xx;
            let b = matte.get(j);
            // An internal edge may border translucent paint rather than clear
            // canvas. Require a different material there; a same-colour alpha
            // ramp is authored opacity, not evidence of boundary coverage.
            let material_contrast =
                (0..3).any(|c| (source.pixels[j][c] - colour[c]).abs() > 24.0 / 255.0);
            if labels[j] == owner || b >= a || (!clear && !material_contrast) {
                return false;
            }
            let t = (1.0 - a) / (1.0 - b);
            (0..3).all(|c| {
                (source.pixels[i][c] * a - (colour[c] * (1.0 - t) + source.pixels[j][c] * b * t))
                    .abs()
                    <= 4.0 / 255.0
            })
        })
    })
}

fn local_face_opacity(
    local: Option<&crate::alpha_coverage::LocalCoverage>,
    matte: &AlphaMatte,
    pixels: &[usize],
) -> Option<f32> {
    let local = local?;
    let opacity = local.opacity[*pixels.first()?];
    if opacity <= 0.0 || !pixels.iter().all(|&i| local.opacity[i] == opacity) {
        return None;
    }
    // A subpixel coloured rim can be entirely fractional even when the
    // neighbouring material is opaque. Its own alpha must remain in paint.
    let core = pixels
        .iter()
        .filter(|&&i| matte.get(i) >= opacity * 250.0 / 255.0)
        .count();
    (core * 100 >= pixels.len() * 97).then_some(opacity)
}

type GradientPatch = (Vec<usize>, Paint, Paint);

fn local_gradient_fields(
    source: &Raster,
    alpha: &Raster,
    pixels: &[usize],
    paint: &Paint,
    depth: usize,
) -> Option<Vec<GradientPatch>> {
    if let Some(field) = gradient::fit_alpha_on_paint(alpha, pixels, paint) {
        return Some(vec![(pixels.to_vec(), paint.clone(), field)]);
    }
    if let Some(field) = gradient::fit_alpha_field(alpha, pixels) {
        if matches!(field, Paint::Solid { .. }) {
            return Some(vec![(pixels.to_vec(), paint.clone(), field)]);
        }
        if let Some(colour) = gradient::fit_colour_on_alpha(source, pixels, paint, &field) {
            return Some(vec![(pixels.to_vec(), colour, field)]);
        }
    }
    if depth >= 6 || pixels.len() < 4 {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for &i in pixels {
        x0 = x0.min(i % source.width);
        x1 = x1.max(i % source.width);
        y0 = y0.min(i / source.width);
        y1 = y1.max(i / source.width);
    }
    let horizontal = x1 - x0 > y1 - y0;
    let mid = if horizontal {
        (x0 + x1) / 2
    } else {
        (y0 + y1) / 2
    };
    let (a, b): (Vec<_>, Vec<_>) = pixels.iter().copied().partition(|&i| {
        (if horizontal {
            i % source.width
        } else {
            i / source.width
        }) <= mid
    });
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut parts = local_gradient_fields(source, alpha, &a, paint, depth + 1)?;
    parts.extend(local_gradient_fields(source, alpha, &b, paint, depth + 1)?);
    (parts.len() <= 16).then_some(parts)
}

#[cfg(test)]
include!("../tests/unit/face_alpha.rs");
