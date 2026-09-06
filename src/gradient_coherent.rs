//! Fit continuous source fields before committing to quantizer/patch boundaries.
use super::*;
use std::collections::VecDeque;

pub(crate) fn reconstruct(
    source: &Raster,
    boundary_source: &Raster,
    segmentation: &mut Segmentation,
    config: &Config,
) -> Vec<Option<Paint>> {
    let mut hints = reconstruct_domains(source, boundary_source, segmentation, config);
    let mut faces = vec![Vec::new(); segmentation.regions.len()];
    for (i, &label) in segmentation.labels.iter().enumerate() {
        faces[label as usize].push(i);
    }
    let labs = lab_pixels(source);
    hints.par_iter_mut().enumerate().for_each(|(label, hint)| {
        let pixels = &faces[label];
        let region = &segmentation.regions[label];
        if hint.is_some()
            || pixels.len() < 2048
            || region.min_x == 0
            || region.min_y == 0
            || region.max_x + 1 == source.width
            || region.max_y + 1 == source.height
        {
            return;
        }
        // Test the existing face before introducing spatial patches. A valid
        // field needs neither additional owners nor subsequent seam coupling.
        let interior = pixels
            .iter()
            .copied()
            .filter(|&i| {
                let w = source.width;
                i % w > 0
                    && i % w + 1 < w
                    && i >= w
                    && i + w < source.pixels.len()
                    && [i - 1, i + 1, i - w, i + w]
                        .iter()
                        .all(|&j| segmentation.labels[j] as usize == label)
            })
            .collect::<Vec<_>>();
        if interior.len() < 2048 {
            return;
        }
        let samples = sampled_indices(&interior, 2048);
        let mut tensor = [0.0_f64; 3];
        for &i in &samples {
            for c in 0..3 {
                let dx = (source.pixels[i + 1][c] - source.pixels[i - 1][c]) as f64;
                let dy = (source.pixels[i + source.width][c] - source.pixels[i - source.width][c])
                    as f64;
                tensor[0] += dx * dx;
                tensor[1] += dx * dy;
                tensor[2] += dy * dy;
            }
        }
        let angle = 0.5 * (2.0 * tensor[1]).atan2(tensor[0] - tensor[2]);
        let mut best = None;
        let mut best_error = f32::INFINITY;
        for direction in [
            (angle.cos() as f32, angle.sin() as f32),
            (1.0, 0.0),
            (0.0, 1.0),
        ] {
            let paint = profile_paint(source, &interior, direction, config.maximum_gradient_stops);
            let errors = errors_for_indices(&labs, &samples, source.width, &paint);
            let mean = numpy_sum_f32(&errors) / errors.len() as f32;
            if mean < best_error && mean <= 1.8 && percentile(errors, 0.90) <= 4.0 {
                best_error = mean;
                best = Some(paint);
            }
        }
        *hint = best;
    });
    hints
}

fn reconstruct_domains(
    source: &Raster,
    boundary_source: &Raster,
    segmentation: &mut Segmentation,
    config: &Config,
) -> Vec<Option<Paint>> {
    let n = segmentation.regions.len();
    let len = source.pixels.len();
    let w = source.width;
    if n < 3 {
        return vec![None; n];
    }
    let protected = segmentation
        .regions
        .iter()
        .map(|region| {
            let span = (region.max_x - region.min_x + 1).max(region.max_y - region.min_y + 1);
            region.area as f32 / span.max(1) as f32 <= 2.5
        })
        .collect::<Vec<_>>();
    let neighbours = |i: usize| {
        [
            if !i.is_multiple_of(w) { i - 1 } else { i },
            if i % w + 1 < w { i + 1 } else { i },
            if i >= w { i - w } else { i },
            if i + w < len { i + w } else { i },
        ]
    };
    // A colour change across a pixel edge is not itself a new Paint owner.
    // Extract smooth interiors directly from the source, independently of the
    // quantizer labels (which can contain unrelated boundary fragments).
    let quiet = (0..len)
        .map(|i| {
            neighbours(i).iter().all(|&j| {
                (0..3).all(|c| {
                    (boundary_source.pixels[i][c] - boundary_source.pixels[j][c]).abs()
                        <= 3.0 / 255.0
                })
            })
        })
        .collect::<Vec<_>>();
    let mut seen = vec![false; len];
    let mut components = Vec::new();
    for i in 0..len {
        if seen[i] || !quiet[i] {
            continue;
        }
        seen[i] = true;
        let mut queue = VecDeque::from([i]);
        let mut pixels = Vec::new();
        while let Some(j) = queue.pop_front() {
            pixels.push(j);
            for k in neighbours(j) {
                if quiet[k] && !seen[k] {
                    seen[k] = true;
                    queue.push_back(k);
                }
            }
        }
        if pixels.len() >= 2048 {
            pixels.sort_unstable();
            components.push(pixels);
        }
    }
    if components.is_empty() {
        return vec![None; n];
    }
    // A quiet path around an edge endpoint does not make the two materials
    // interchangeable. Preserve measured interfaces anywhere in the union,
    // including interfaces far from the path that connected its interiors.
    let boundary_labs = lab_pixels(boundary_source);
    let barriers = smooth_paint_boundaries(&boundary_labs, segmentation, 8, true)
        .into_iter()
        .map(|mut boundary| {
            measure_boundary_material_step(&boundary_labs, segmentation, &mut boundary);
            boundary
        })
        .filter(|boundary| {
            boundary_has_material_step(boundary)
                || (boundary.median_delta_e >= 3.0 && !boundary_has_continuous_gradient(boundary))
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let report = barriers
            .iter()
            .map(|b| {
                serde_json::json!({
                    "left": b.left, "right": b.right, "length": b.length,
                    "median_delta_e": b.median_delta_e,
                    "p90_delta_e": b.percentile_delta_e,
                    "material_step_fraction": b.material_step_fraction,
                })
            })
            .collect::<Vec<_>>();
        let _ = std::fs::write(
            format!("{prefix}-coherent-barriers.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        );
    }
    let mut region_pixels = vec![Vec::new(); n];
    for (i, &label) in segmentation.labels.iter().enumerate() {
        region_pixels[label as usize].push(i);
    }
    let labs = lab_pixels(source);
    let fits = components
        .par_iter()
        .map(|pixels| {
            let samples = sampled_indices(pixels, 2048);
            let children = pixels
                .iter()
                .map(|&i| segmentation.labels[i])
                .collect::<HashSet<_>>();
            if children.len() < 3 {
                return (None, 0.0, 0.0, children.len(), Vec::new(), 0, 0);
            }
            let mut tensor = [0.0_f64; 3];
            for &i in &samples {
                if i.is_multiple_of(w) || i % w + 1 == w || i < w || i + w >= len {
                    continue;
                }
                for c in 0..3 {
                    let x = (source.pixels[i + 1][c] - source.pixels[i - 1][c]) as f64;
                    let y = (source.pixels[i + w][c] - source.pixels[i - w][c]) as f64;
                    tensor[0] += x * x;
                    tensor[1] += x * y;
                    tensor[2] += y * y;
                }
            }
            let trace = tensor[0] + tensor[2];
            let disc = ((tensor[0] - tensor[2]).powi(2) + 4.0 * tensor[1] * tensor[1]).sqrt();
            let coherence = if trace > 1e-14 {
                (trace + disc) / (2.0 * trace)
            } else {
                1.0
            };
            if coherence < 0.53 {
                return (None, coherence, 0.0, children.len(), Vec::new(), 0, 0);
            }
            let angle = 0.5 * (2.0 * tensor[1]).atan2(tensor[0] - tensor[2]);
            let mut directions = vec![
                (angle.cos() as f32, angle.sin() as f32),
                (1.0, 0.0),
                (0.0, 1.0),
            ];
            directions.dedup();
            let solid = Paint::Solid {
                color: mean_color(source, &samples),
            };
            let mut best = solid;
            let score = |paint: &Paint| {
                let mut errors = errors_for_indices(&labs, &samples, w, paint);
                errors.sort_by(f32::total_cmp);
                let keep = (errors.len() * 3 / 4).max(1);
                numpy_sum_f32(&errors[..keep]) / keep as f32
            };
            let mut best_score = score(&best);
            for &direction in &directions {
                let candidate =
                    profile_paint(source, pixels, direction, config.maximum_gradient_stops);
                let error = score(&candidate);
                if error < best_score {
                    best = candidate;
                    best_score = error;
                }
            }
            // Transfer complete existing faces. Selecting a pixel mask, even
            // from smooth interiors, creates new jagged perimeter fragments.
            // A face must predominantly belong to this continuous field.
            let mut coverage = HashMap::<u32, usize>::new();
            for &i in pixels {
                *coverage.entry(segmentation.labels[i]).or_default() += 1;
            }
            let mut eligible = (0..n)
                .map(|label| {
                    !protected[label]
                        && coverage.get(&(label as u32)).copied().unwrap_or(0) * 5
                            >= region_pixels[label].len() * 4
                })
                .collect::<Vec<_>>();
            let mut pruned = false;
            for boundary in &barriers {
                if eligible[boundary.left] && eligible[boundary.right] {
                    // Keep the larger supported face; never let another path
                    // through the component bypass this cannot-merge pair.
                    let excluded = if region_pixels[boundary.left].len()
                        <= region_pixels[boundary.right].len()
                    {
                        boundary.left
                    } else {
                        boundary.right
                    };
                    eligible[excluded] = false;
                    pruned = true;
                }
            }
            let members = eligible
                .iter()
                .enumerate()
                .filter(|(_, yes)| **yes)
                .flat_map(|(label, _)| region_pixels[label].iter().copied())
                .collect::<Vec<_>>();
            if members.len() < 2048 || eligible.iter().filter(|&&yes| yes).count() < 3 {
                return (None, coherence, 0.0, children.len(), Vec::new(), 0, 0);
            }
            if pruned {
                // The old profile included both sides of the forbidden edge.
                // Refit after exclusion rather than retaining that blurred paint.
                let supported = pixels
                    .iter()
                    .copied()
                    .filter(|&i| eligible[segmentation.labels[i] as usize])
                    .collect::<Vec<_>>();
                let selected_samples = sampled_indices(&supported, 2048);
                let selected_score = |paint: &Paint| {
                    let mut errors = errors_for_indices(&labs, &selected_samples, w, paint);
                    errors.sort_by(f32::total_cmp);
                    let keep = (errors.len() * 3 / 4).max(1);
                    numpy_sum_f32(&errors[..keep]) / keep as f32
                };
                best = Paint::Solid {
                    color: mean_color(source, &selected_samples),
                };
                best_score = selected_score(&best);
                for &direction in &directions {
                    let candidate =
                        profile_paint(source, &supported, direction, config.maximum_gradient_stops);
                    let error = selected_score(&candidate);
                    if error < best_score {
                        best = candidate;
                        best_score = error;
                    }
                }
            }
            let errors = errors_for_indices(&labs, &members, w, &best);
            let mean = numpy_sum_f32(&errors) / errors.len() as f32;
            let p90 = percentile(errors, 0.90);
            let removed = members
                .iter()
                .map(|&i| {
                    neighbours(i)
                        .iter()
                        .filter(|&&j| {
                            eligible[segmentation.labels[j] as usize]
                                && segmentation.labels[i] != segmentation.labels[j]
                        })
                        .count()
                })
                .sum::<usize>()
                / 2;
            let introduced = 0;
            let crosses_interface = barriers
                .iter()
                .any(|boundary| eligible[boundary.left] && eligible[boundary.right]);
            // Label contours need not coincide with the source transition:
            // quantization can place a contour well inside a blurred step.
            // Check the proposed interior itself before discarding its owners.
            let internal_steps = sampled_indices(&members, 4096)
                .iter()
                .filter(|&&i| {
                    let x = i % w;
                    let y = i / w;
                    [1_usize, w].iter().any(|&stride| {
                        if (stride == 1 && (x < 24 || x + 24 >= w))
                            || (stride == w && (y < 24 || y + 24 >= source.height))
                        {
                            return false;
                        }
                        [4_usize, 8].iter().any(|&r| {
                            let indices = [
                                i - 3 * r * stride,
                                i - r * stride,
                                i + r * stride,
                                i + 3 * r * stride,
                            ];
                            if indices
                                .iter()
                                .any(|&j| !eligible[segmentation.labels[j] as usize])
                            {
                                return false;
                            }
                            let centre = delta_e2000(labs[indices[1]], labs[indices[2]]);
                            let outside = 0.5
                                * (delta_e2000(labs[indices[0]], labs[indices[1]])
                                    + delta_e2000(labs[indices[2]], labs[indices[3]]));
                            centre >= 2.0 && centre > 3.0 * outside.max(0.25)
                        })
                    })
                })
                .take(12)
                .count();
            let accepted = !crosses_interface && internal_steps < 12 && mean <= 1.8 && p90 <= 4.0;

            (
                accepted.then_some(best),
                coherence,
                mean,
                children.len(),
                members,
                removed,
                introduced,
            )
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "diagnostics")]
    if let Ok(prefix) = std::env::var("PICVEC_PIPELINE_DIAGNOSTICS") {
        let report = components
            .iter()
            .zip(&fits)
            .map(
                |(pixels, (paint, coherence, mean, children, members, removed, introduced))| {
                    serde_json::json!({
                        "removed_edges": removed,
                        "introduced_edges": introduced,
                        "area": pixels.len(),
                        "children": children,
                        "member_pixels": members.len(),
                        "coherence": coherence,
                        "mean": mean,
                        "accepted": paint.is_some(),
                        "bounds": [
                            pixels.iter().map(|i| i % w).min(),
                            pixels.iter().map(|i| i / w).min(),
                            pixels.iter().map(|i| i % w).max(),
                            pixels.iter().map(|i| i / w).max(),
                        ],
                    })
                },
            )
            .collect::<Vec<_>>();
        let _ = std::fs::write(
            format!("{prefix}-coherent.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        );
    }
    let mut labels = segmentation.labels.clone();
    let mut hints = vec![None; n];
    let mut accepted = 0;
    for (_, (paint, _, _, _, pixels, _, _)) in components.iter().zip(fits) {
        let Some(paint) = paint else {
            continue;
        };
        let id = hints.len() as u32;
        hints.push(Some(paint.clone()));
        accepted += 1;
        for &i in &pixels {
            labels[i] = id;
        }
    }
    if accepted == 0 {
        return vec![None; n];
    }
    let mut used = labels.clone();
    used.sort_unstable();
    used.dedup();
    let result = used.iter().map(|&i| hints[i as usize].clone()).collect();
    let removed = n.saturating_sub(used.len());
    replace_source_supported_paint_labels(source, segmentation, labels, 0);
    segmentation.summary.coherent_domains = accepted;
    segmentation.summary.coherent_regions_removed = removed;
    result
}

pub(super) fn profile_paint(
    source: &Raster,
    pixels: &[usize],
    direction: (f32, f32),
    maximum_stops: usize,
) -> Paint {
    let project = |i: usize| {
        (i % source.width) as f32 * direction.0 + (i / source.width) as f32 * direction.1
    };
    let lo = pixels
        .iter()
        .map(|&i| project(i))
        .fold(f32::INFINITY, f32::min);
    let hi = pixels
        .iter()
        .map(|&i| project(i))
        .fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1.0);
    let bins = (span.ceil() as usize + 1).clamp(2, 256);
    let mut rows = vec![Vec::<[f32; 3]>::new(); bins];
    for &i in pixels {
        let b = (((project(i) - lo) / span * (bins - 1) as f32).round() as usize).min(bins - 1);
        rows[b].push(source.pixels[i]);
    }
    let mut colours = Vec::new();
    let mut parameters = Vec::new();
    for (b, row) in rows.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let colour = [0, 1, 2].map(|c| {
            let mut values = row.iter().map(|v| v[c]).collect::<Vec<_>>();
            values.sort_by(f32::total_cmp);
            values[values.len() / 2]
        });
        colours.push(colour);
        parameters.push(b as f32 / (bins - 1) as f32);
    }
    let profile = Raster::new(colours.len(), 1, colours);
    let samples = (0..profile.pixels.len()).collect::<Vec<_>>();
    let stops = profile_stops(&profile, &samples, &parameters, maximum_stops);
    let start = Point {
        x: direction.0 * lo,
        y: direction.1 * lo,
    };
    let end = Point {
        x: direction.0 * hi,
        y: direction.1 * hi,
    };
    Paint::Linear {
        preset: LinearPreset::Fitted,
        start,
        end,
        stops,
    }
}

// Profile samples are already denoised across the field. Curvature regularization
// here would suppress real narrow highlights and depends on the bin count.
fn profile_colours(
    source: &Raster,
    samples: &[usize],
    parameters: &[f32],
    offsets: &[f64],
) -> Vec<ColorStop> {
    let n = offsets.len();
    let mut normal = vec![vec![0.0; n]; n];
    let mut targets = vec![vec![0.0; n]; 3];
    for (&i, &t) in samples.iter().zip(parameters) {
        let (l, r, a) = interpolation_weights(t, offsets);
        for (j, v) in [(l, 1.0 - a), (r, a)] {
            for (k, u) in [(l, 1.0 - a), (r, a)] {
                normal[j][k] += v as f64 * u as f64;
            }
            for (c, target) in targets.iter_mut().enumerate() {
                target[j] += v as f64 * source.pixels[i][c] as f64;
            }
        }
    }
    let colours = targets
        .into_iter()
        .map(|t| solve_system(normal.clone(), t))
        .collect::<Vec<_>>();
    offsets
        .iter()
        .enumerate()
        .map(|(i, &offset)| ColorStop {
            offset,
            color: [0, 1, 2].map(|c| colours[c][i].clamp(0.0, 1.0)),
        })
        .collect()
}

// Optimize knot locations against the whole colour profile. A fixed tenth-of-
// span grid cannot represent narrow highlights in a broad continuous field.
fn profile_stops(
    profile: &Raster,
    samples: &[usize],
    parameters: &[f32],
    maximum: usize,
) -> Vec<ColorStop> {
    let mse = |stops: &[ColorStop]| {
        parameters
            .iter()
            .zip(&profile.pixels)
            .map(|(&t, p)| {
                let q = interpolate(stops, t);
                (0..3).map(|c| (p[c] - q[c]).powi(2) as f64).sum::<f64>()
            })
            .sum::<f64>()
    };
    let mut offsets = vec![0.0, 1.0];
    let mut stops = profile_colours(profile, samples, parameters, &offsets);
    let mut error = mse(&stops);
    while offsets.len() < maximum.clamp(2, 5) {
        let mut best = None;
        for &t in parameters.iter().step_by(4) {
            let t = t as f64;
            if offsets.iter().any(|&v| (v - t).abs() < 0.01) {
                continue;
            }
            let mut knots = offsets.clone();
            knots.push(t);
            knots.sort_by(f64::total_cmp);
            let candidate = profile_colours(profile, samples, parameters, &knots);
            let score = mse(&candidate);
            if score < error && best.as_ref().is_none_or(|(e, _, _)| score < *e) {
                best = Some((score, knots, candidate));
            }
        }
        let Some((score, knots, candidate)) = best else {
            break;
        };
        error = score;
        offsets = knots;
        stops = candidate;
    }
    for _ in 0..3 {
        for j in 1..offsets.len() - 1 {
            let old = offsets[j];
            for &t in parameters {
                let t = t as f64;
                if (t - old).abs() > 0.05
                    || t <= offsets[j - 1] + 0.002
                    || t >= offsets[j + 1] - 0.002
                {
                    continue;
                }
                let mut knots = offsets.clone();
                knots[j] = t;
                let candidate = profile_colours(profile, samples, parameters, &knots);
                let score = mse(&candidate);
                if score < error {
                    error = score;
                    offsets = knots;
                    stops = candidate;
                }
            }
        }
    }
    stops
}

#[cfg(test)]
mod tests {
    use super::*;
    fn partition(source: &Raster, labels: Vec<u32>) -> Segmentation {
        let mut segmentation = Segmentation {
            width: source.width,
            height: source.height,
            labels: vec![0; source.pixels.len()],
            paint_keys: vec![0],
            paint_samples: vec![true; source.pixels.len()],
            canonical: source.clone(),
            regions: vec![],
            summary: Default::default(),
        };
        replace_source_supported_paint_labels(source, &mut segmentation, labels, 0);
        segmentation
    }
    #[test]
    fn quiet_bridge_cannot_erase_a_measured_material_interface() {
        let source = Raster::new(
            128,
            96,
            (0..128 * 96)
                .map(|i| {
                    let x = i % 128;
                    let y = i / 128;
                    let step = if y < 8 {
                        x as f32 / 127.0
                    } else if x < 64 {
                        0.0
                    } else {
                        1.0
                    };
                    [0.4 + 0.04 * step; 3]
                })
                .collect(),
        );
        let labels = (0..128 * 96)
            .map(|i| ((i / 128 / 16) * 8 + i % 128 / 16) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_ne!(
            segmentation.labels[48 * 128 + 63],
            segmentation.labels[48 * 128 + 64]
        );
    }
    #[test]
    fn a_single_fittable_face_is_not_cut_into_spatial_patches() {
        for noise in [0.0, 0.006] {
            let source = Raster::new(
                256,
                256,
                (0..256 * 256)
                    .map(|i| {
                        [0.3 + 0.3 * (i % 256) as f32 / 255.0
                            + noise * ((i / 256 % 7) as f32 - 3.0); 3]
                    })
                    .collect(),
            );
            let labels = (0..256 * 256)
                .map(|i| {
                    u32::from((32..224).contains(&(i % 256)) && (32..224).contains(&(i / 256)))
                })
                .collect();
            let mut segmentation = partition(&source, labels);
            let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
            assert!(hints[1].is_some());
            let protected = hints.iter().map(Option::is_some).collect::<Vec<_>>();
            crate::segment::split_adaptive_paint_patches_with_protected(
                &source,
                &source,
                &mut segmentation,
                &protected,
            );
            assert_eq!(segmentation.regions.len(), 2);
        }
    }

    #[test]
    fn a_local_highlight_is_refined_without_subdividing_its_field() {
        let source = Raster::new(
            256,
            256,
            (0..256 * 256)
                .map(|i| {
                    let x = (i % 256) as f32;
                    let y = (i / 256) as f32;
                    let r = ((x - 110.0) / 28.0).hypot((y - 120.0) / 28.0);
                    [0.3 + 0.3 * x / 255.0 + 0.2 * (-r * r).exp(); 3]
                })
                .collect(),
        );
        let labels = (0..256 * 256)
            .map(|i| u32::from((32..224).contains(&(i % 256)) && (32..224).contains(&(i / 256))))
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert!(hints[1].is_some());
        let protected = hints.iter().map(Option::is_some).collect::<Vec<_>>();
        crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut segmentation,
            &protected,
        );
        assert_eq!(segmentation.regions.len(), 2);
        let branches = crate::ridge::StrongRidgeBranches {
            dark: vec![false; 256 * 256],
            bright: vec![false; 256 * 256],
        };
        let (paints, _) = fit_all_without_topology(
            &hints,
            &source,
            &source,
            &segmentation,
            &branches,
            &Config::default(),
        );
        let centre = 120 * 256 + 110;
        let original = paint_at(hints[1].as_ref().unwrap(), centre, 256)[0];
        let corrected = paint_at(&paints[1], centre, 256)[0];
        let target = source.pixels[centre][0];
        assert!(
            (target - corrected).abs() < (target - original).abs() * 0.6,
            "{original} -> {corrected}, source {target}"
        );
    }

    #[test]
    fn blurred_step_inside_quantizer_faces_is_not_merged_away() {
        let source = Raster::new(
            256,
            128,
            (0..256 * 128)
                .map(|i| {
                    let x = (i % 256) as f32;
                    let step = ((x - 124.0) / 8.0).clamp(0.0, 1.0);
                    [0.4 + 0.04 * step; 3]
                })
                .collect(),
        );
        // The source step falls inside a face, not on a label interface.
        let labels = (0..256 * 128)
            .map(|i| ((i % 256 + 16) / 32) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_ne!(
            segmentation.labels[64 * 256 + 96],
            segmentation.labels[64 * 256 + 160]
        );
    }

    #[test]
    fn perforated_smooth_interior_does_not_create_extra_boundaries() {
        let source = Raster::new(
            128,
            128,
            (0..128 * 128)
                .map(|i| {
                    let spot = if i % 128 % 8 == 4 && i / 128 % 8 == 4 {
                        0.1
                    } else {
                        0.0
                    };
                    [0.3 + 0.3 * (i % 128) as f32 / 127.0 + spot; 3]
                })
                .collect(),
        );
        let labels = (0..128 * 128)
            .map(|i| {
                if i % 128 % 8 == 4 && i / 128 % 8 == 4 {
                    3
                } else {
                    (i % 128 / 43).min(2) as u32
                }
            })
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_eq!(hints.iter().flatten().count(), 1);
        assert_eq!(segmentation.regions.len(), 2);
        assert_ne!(
            segmentation.labels[4 * 128 + 4],
            segmentation.labels[4 * 128 + 3]
        );
    }
    #[test]
    fn continuous_ramp_replaces_artificial_grid_and_retains_thin_line() {
        for line in [false, true] {
            let source = Raster::new(
                96,
                64,
                (0..96 * 64)
                    .map(|i| {
                        [0.3 + 0.4 * (i % 96) as f32 / 95.0
                            + if line && i % 96 == 48 { 0.005 } else { 0.0 };
                            3]
                    })
                    .collect(),
            );
            let labels = (0..96 * 64)
                .map(|i| {
                    if line && i % 96 == 48 {
                        24
                    } else {
                        ((i / 96 / 16) * 6 + (i % 96 / 16)) as u32
                    }
                })
                .collect();
            let mut segmentation = partition(&source, labels);
            let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
            assert_eq!(hints.iter().flatten().count(), 1);
            assert_eq!(segmentation.regions.len(), if line { 2 } else { 1 });
            if line {
                assert_ne!(
                    segmentation.labels[32 * 96 + 48],
                    segmentation.labels[32 * 96 + 47]
                );
            }
        }
    }
    #[test]
    fn material_transition_separates_coherent_fields() {
        let source = Raster::new(
            128,
            64,
            (0..128 * 64)
                .map(|i| [if i % 128 < 64 { 0.2 } else { 0.7 } + 0.1 * (i % 64) as f32 / 63.0; 3])
                .collect(),
        );
        let labels = (0..128 * 64)
            .map(|i| ((i / 128 / 16) * 8 + i % 128 / 16) as u32)
            .collect();
        let mut segmentation = partition(&source, labels);
        let hints = reconstruct(&source, &source, &mut segmentation, &Config::default());
        assert_eq!(hints.iter().flatten().count(), 2);
        assert_ne!(
            segmentation.labels[32 * 128 + 63],
            segmentation.labels[32 * 128 + 64]
        );
    }
    #[test]
    fn fitted_field_is_not_subdivided_into_long_axis_patches() {
        let source = Raster::new(
            256,
            192,
            (0..256 * 192)
                .map(|i| [0.2 + 0.6 * (i % 256) as f32 / 255.0; 3])
                .collect(),
        );
        let labels = (0..256 * 192)
            .map(|i| (i % 256 / 86).min(2) as u32)
            .collect();
        let segmentation = partition(&source, labels);
        let mut split = segmentation.clone();
        let parents = crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut split,
            &[false, true, false],
        );
        assert_eq!(parents.iter().filter(|&&i| i == 1).count(), 1);
        let mut legacy = segmentation;
        let parents = crate::segment::split_adaptive_paint_patches_with_protected(
            &source,
            &source,
            &mut legacy,
            &[],
        );
        assert!(parents.iter().filter(|&&i| i == 1).count() > 1);
    }
    #[test]
    fn colour_knots_follow_a_non_tenth_highlight() {
        let input = vec![
            ColorStop {
                offset: 0.0,
                color: [0.2; 3],
            },
            ColorStop {
                offset: 0.173,
                color: [0.9; 3],
            },
            ColorStop {
                offset: 1.0,
                color: [0.3; 3],
            },
        ];
        let parameters = (0..256).map(|i| i as f32 / 255.0).collect::<Vec<_>>();
        let profile = Raster::new(
            256,
            1,
            parameters.iter().map(|&t| interpolate(&input, t)).collect(),
        );
        let fitted = profile_stops(&profile, &(0..256).collect::<Vec<_>>(), &parameters, 3);
        assert!((fitted[1].offset - 0.173).abs() < 0.006);
        let error = parameters
            .iter()
            .zip(&profile.pixels)
            .map(|(&t, p)| (interpolate(&fitted, t)[0] - p[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 0.01, "{error}");
    }
}
