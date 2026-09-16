use super::*;

/// Merge source-supported colour fragments before fitting their own paints.
/// Use local source colours: global means fail on shaded faces and fragmented
/// outlines. Near-identical patches may share a material; coverage mixtures
/// use measured paint mixtures and keep ownership in incident faces.
pub(crate) fn absorb_micro_regions(
    image: &Raster,
    segmentation: &mut Segmentation,
    matte: Option<&crate::chroma::AlphaMatte>,
) -> usize {
    let (w, h) = (image.width, image.height);
    let mut total = 0;
    for _ in 0..8 {
        let count = segmentation.regions.len();
        let labels = &segmentation.labels;
        let largest = segmentation
            .regions
            .iter()
            .map(|r| r.area)
            .max()
            .unwrap_or(0);
        let mut pixels = vec![Vec::new(); count];
        for (i, &id) in labels.iter().enumerate() {
            // Ownership only moves into a larger face. Avoid analysing the
            // largest field, whose source distributions cannot yield a merge.
            if segmentation.regions[id as usize].area < largest {
                pixels[id as usize].push(i);
            }
        }
        // Bound scratch allocation to a few chunks per worker, and collect in
        // label order so chained unions and tie-breaking remain deterministic.
        let chunk_size = count.div_ceil(rayon::current_num_threads() * 4).max(1);
        let proposals: Vec<Vec<MicroProposal>> = pixels
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk, components)| {
                let mut parent_slots = vec![usize::MAX; count];
                components
                    .iter()
                    .enumerate()
                    .map(|(offset, component)| {
                        micro_region_proposal(
                            image,
                            segmentation,
                            matte,
                            chunk * chunk_size + offset,
                            component,
                            &mut parent_slots,
                        )
                    })
                    .collect()
            })
            .collect();
        let mut changes = Vec::new();
        let mut material_unions = vec![false; count];
        for (id, proposal) in proposals.into_iter().flatten().enumerate() {
            total += proposal.merges;
            material_unions[id] = proposal.material_union;
            changes.extend(proposal.changes);
        }
        if changes.is_empty() {
            break;
        }
        // Resolve whole-face unions before committing synchronous pixel moves.
        // Otherwise A -> B and B -> C leave A's pixels behind under the old B
        // label, recreating a small object for a paint that was just removed.
        // All targets were larger in this pass, so these links are acyclic.
        let mut redirects = vec![None::<(u32, [f32; 3])>; count];
        let mut split = vec![false; count];
        for &(i, owner, rgb) in &changes {
            let old = labels[i] as usize;
            if let Some((previous, _)) = redirects[old] {
                if previous != owner {
                    split[old] = true;
                }
            } else {
                redirects[old] = Some((owner, rgb));
            }
        }
        let mut split_assignments = vec![Vec::new(); count];
        for &(i, owner, rgb) in &changes {
            if split[labels[i] as usize] {
                split_assignments[labels[i] as usize].push((i, owner, rgb));
            }
        }
        for id in 0..count {
            if split[id] {
                redirects[id] = None;
            }
        }
        // A material union preserves interior colour evidence. Coverage
        // reassignment does not: those pixels mix different paints. Follow the
        // entire redirect chain before deciding which evidence survives.
        let preserve_material: Vec<_> = (0..count)
            .map(|id| {
                let mut owner = id;
                while let Some((next, _)) = redirects[owner] {
                    if !material_unions[owner] {
                        return false;
                    }
                    owner = next as usize;
                }
                material_unions[id] && !split[owner]
            })
            .collect();
        let preserve_samples: Vec<_> = (0..labels.len())
            .map(|i| {
                let (x, y) = (i % w, i / w);
                preserve_material[labels[i] as usize]
                    && x > 0
                    && x + 1 < w
                    && y > 0
                    && y + 1 < h
                    && [i - 1, i + 1, i - w, i + w]
                        .iter()
                        .all(|&j| labels[j] == labels[i])
            })
            .collect();
        for id in 0..count {
            if let Some(mut target) = redirects[id] {
                while let Some(next) = redirects[target.0 as usize] {
                    target = next;
                }
                redirects[id] = Some(target);
            }
        }
        segmentation.summary.micro_pixels_reassigned += changes.len();
        for (i, mut owner, mut rgb) in changes {
            let mut position = i;
            loop {
                if let Some(target) = redirects[owner as usize] {
                    (owner, rgb) = target;
                } else if split[owner as usize] {
                    // A coverage parent can retire into several paints. Follow
                    // the nearest native part of that parent's assignment,
                    // rather than retaining its now-empty intermediate label.
                    let &(next_position, next_owner, next_rgb) = split_assignments[owner as usize]
                        .iter()
                        .min_by_key(|&&(j, _, _)| {
                            let dx = (j % w).abs_diff(position % w);
                            let dy = (j / w).abs_diff(position / w);
                            (dx * dx + dy * dy, j)
                        })
                        .unwrap();
                    position = next_position;
                    owner = next_owner;
                    rgb = next_rgb;
                } else {
                    break;
                }
            }
            segmentation.labels[i] = owner;
            if !preserve_samples[i] {
                segmentation.canonical.pixels[i] = rgb;
                segmentation.paint_samples[i] = false;
            }
        }
        let (labels, count) = compact_values(&segmentation.labels);
        segmentation.labels = labels;
        segmentation.regions = region_stats(image, &segmentation.labels, count);
        segmentation.paint_keys = (0..count as u32).collect();
        segmentation.summary.merged_regions = count;
    }
    segmentation.summary.micro_region_merges += total;
    total
}

#[derive(Default)]
struct MicroProposal {
    changes: Vec<(usize, u32, [f32; 3])>,
    material_union: bool,
    merges: usize,
}

// Read the same ownership snapshot for every region. Scratch slots are private
// to a chunk and reset before any return that follows their population.
fn micro_region_proposal(
    image: &Raster,
    segmentation: &Segmentation,
    matte: Option<&crate::chroma::AlphaMatte>,
    id: usize,
    component: &[usize],
    parent_slots: &mut [usize],
) -> MicroProposal {
    let (w, h) = (image.width, image.height);
    let labels = &segmentation.labels;
    let mut proposal = MicroProposal::default();
    if component.is_empty() {
        return proposal;
    }
    let region = &segmentation.regions[id];
    let neutral = component.iter().all(|&i| {
        let p = image.pixels[i];
        p.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - p.iter().copied().fold(f32::INFINITY, f32::min)
            <= 48.0 / 255.0
    });
    let radius = if neutral { 8 } else { 4 };
    let large = component.len() > 128;
    let interior: Vec<_> = if large {
        component
            .iter()
            .copied()
            .filter(|&i| {
                let (x, y) = (i % w, i / w);
                x > 0
                    && x + 1 < w
                    && y > 0
                    && y + 1 < h
                    && [i - 1, i + 1, i - w, i + w]
                        .iter()
                        .all(|&j| labels[j] as usize == id)
            })
            .collect()
    } else {
        Vec::new()
    };
    let material_distribution =
        (interior.len() >= 8).then(|| colour_distribution(image, &interior));
    let resolved_shading = large && has_resolved_shading(image, &interior);
    if large && material_distribution.is_none() {
        return proposal;
    }
    // Only an incident, larger owner may receive this region.
    let mut contacts = BTreeSet::new();
    let mut boundary_differences = BTreeMap::<usize, Vec<f32>>::new();
    for &i in component {
        let (x, y) = (i % w, i / w);
        for j in [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
        {
            let other = labels[j] as usize;
            if other != id && segmentation.regions[other].area > component.len() {
                contacts.insert(other);
                if large
                    && x >= 2
                    && x + 2 < w
                    && y >= 2
                    && y + 2 < h
                    && [i - 1, i + 1, i - w, i + w, j - 1, j + 1, j - w, j + w]
                        .iter()
                        .all(|&k| labels[k] as usize == id || labels[k] as usize == other)
                {
                    boundary_differences.entry(other).or_default().push(
                        (0..3)
                            .map(|c| (image.pixels[i][c] - image.pixels[j][c]).abs())
                            .fold(0.0_f32, f32::max),
                    );
                }
            }
        }
    }
    if contacts.is_empty() {
        return proposal;
    }
    let opacity = |i| matte.map_or(1.0, |a| a.get(i));
    let a = component.iter().map(|&i| opacity(i)).sum::<f32>() / component.len() as f32;
    if a <= 0.0
        || component
            .iter()
            .any(|&i| (opacity(i) - a).abs() > 1.5 / 255.0)
    {
        return proposal;
    }
    // A raster fringe can itself separate a tip from its material.
    // Discover source paints in the same local window, while keeping
    // direct contacts separate for ordinary same-material merging.
    // Nonincident paints are evidence only for partial coverage below.
    let incident = contacts.clone();
    if !large && neutral {
        for y in region.min_y.saturating_sub(radius)..(region.max_y + radius).min(h) {
            for x in region.min_x.saturating_sub(radius)..(region.max_x + radius).min(w) {
                let other = labels[y * w + x] as usize;
                if other != id && segmentation.regions[other].area > 128 {
                    contacts.insert(other);
                }
            }
        }
    }
    let varying_coverage = (0..3).any(|c| {
        let low = component
            .iter()
            .map(|&i| image.pixels[i][c])
            .fold(f32::INFINITY, f32::min);
        let high = component
            .iter()
            .map(|&i| image.pixels[i][c])
            .fold(f32::NEG_INFINITY, f32::max);
        high - low > 16.0 / 255.0
    });
    let resolved_interior = component.iter().any(|&i| {
        let (x, y) = (i % w, i / w);
        x + 1 < w
            && y + 1 < h
            && [i + 1, i + w, i + w + 1].iter().all(|&j| {
                labels[j] as usize == id
                    && (0..3)
                        .all(|c| (image.pixels[i][c] - image.pixels[j][c]).abs() <= 8.0 / 255.0)
            })
    });
    let mut parents = Vec::new();
    let mut edge_parents = Vec::new();
    let mut repeated = Vec::new();
    let mut same_material = Vec::new();
    // Visit the local window once, distributing its pixels to the
    // incident owners. Keep both parent order and row-major sample
    // order identical to scanning the whole window for each parent.
    let mut parent_samples = vec![(Vec::new(), Vec::new()); contacts.len()];
    for (slot, &parent) in contacts.iter().enumerate() {
        parent_slots[parent] = slot;
    }
    for y in region.min_y.saturating_sub(radius)..(region.max_y + radius).min(h) {
        for x in region.min_x.saturating_sub(radius)..(region.max_x + radius).min(w) {
            let j = y * w + x;
            let parent = labels[j] as usize;
            let slot = parent_slots[parent];
            if slot == usize::MAX || (opacity(j) - a).abs() > 1.5 / 255.0 {
                continue;
            }
            let (samples, core) = &mut parent_samples[slot];
            samples.push(j);
            if x > 0
                && x + 1 < w
                && y > 0
                && y + 1 < h
                && [j - 1, j + 1, j - w, j + w]
                    .iter()
                    .all(|&k| labels[k] as usize == parent)
            {
                core.push(j);
            }
        }
    }
    for &parent in &contacts {
        parent_slots[parent] = usize::MAX;
    }
    for (parent, (samples, core)) in contacts.into_iter().zip(parent_samples) {
        let selected = if core.len() >= 3 { &core } else { &samples };
        if selected.len() < 3 {
            continue;
        }
        let mut rgb = [0.0; 3];
        for c in 0..3 {
            let mut values: Vec<_> = selected.iter().map(|&j| image.pixels[j][c]).collect();
            values.sort_by(f32::total_cmp);
            rgb[c] = values[values.len() / 2];
        }
        if large {
            if core.len() >= 16 {
                let parent_distribution = colour_distribution(image, &core);
                let material = material_distribution.as_ref().unwrap();
                let overlapping_tails = (0..3).all(|c| {
                    material[c][0] <= parent_distribution[c][4] + 1.01 / 255.0
                        && parent_distribution[c][0] <= material[c][4] + 1.01 / 255.0
                });
                let near_black_overlap = overlapping_tails
                    && (0..3).all(|c| {
                        material[c][4].max(parent_distribution[c][4]) <= 24.0 / 255.0
                            && material[c][4] - material[c][0] <= 16.01 / 255.0
                    });
                let varying_continuation = overlapping_tails
                    && (0..3)
                        .filter(|&c| {
                            material[c][3] - material[c][1] >= 1.99 / 255.0
                                && parent_distribution[c][3] - parent_distribution[c][1]
                                    >= 0.99 / 255.0
                        })
                        .count()
                        >= 2
                    && (0..3).all(|c| {
                        (material[c][2] - parent_distribution[c][2]).abs() <= 12.01 / 255.0
                            && material[c][4] - material[c][0] <= 16.01 / 255.0
                            && parent_distribution[c][4] - parent_distribution[c][0]
                                <= 16.01 / 255.0
                            && (material[c][0] - parent_distribution[c][2]).abs() <= 16.01 / 255.0
                            && (material[c][4] - parent_distribution[c][2]).abs() <= 16.01 / 255.0
                    });
                // Quantized material can have several code values of
                // internal variation. Compare its interface with that
                // variation and two-sample quantization uncertainty.
                // Flat distinct paints cannot use the varying-field gate.
                let noise_limit = if near_black_overlap || varying_continuation {
                    let mut steps = Vec::new();
                    for &i in &interior {
                        for j in [i + 1, i + w] {
                            if labels[j] as usize == id {
                                steps.push(
                                    (0..3)
                                        .map(|c| (image.pixels[i][c] - image.pixels[j][c]).abs())
                                        .fold(0.0_f32, f32::max),
                                );
                            }
                        }
                    }
                    steps.sort_by(f32::total_cmp);
                    steps
                        .get(steps.len().saturating_sub(1) * 3 / 4)
                        .copied()
                        .map(|v| v + 2.01 / 255.0)
                        .unwrap_or(0.0)
                        .clamp(2.01 / 255.0, 4.01 / 255.0)
                } else {
                    2.01 / 255.0
                };
                let continuous = boundary_differences
                    .get_mut(&parent)
                    .is_some_and(|samples| {
                        samples.sort_by(f32::total_cmp);
                        samples.len() >= 8
                            && samples[samples.len() / 2] <= noise_limit
                            && samples[(samples.len() - 1) * 3 / 4] <= 8.01 / 255.0
                    });
                let ink_continuation = continuous && near_black_overlap;
                let median = material_distribution.as_ref().unwrap().map(|q| q[2]);
                let extra_error = interior
                    .iter()
                    .map(|&i| {
                        (0..3)
                            .map(|c| {
                                (image.pixels[i][c] - rgb[c]).powi(2)
                                    - (image.pixels[i][c] - median[c]).powi(2)
                            })
                            .sum::<f32>()
                            / 3.0
                    })
                    .sum::<f32>()
                    / interior.len() as f32;
                if (!resolved_shading || ink_continuation || (continuous && varying_continuation))
                    && ((continuous && varying_continuation)
                        || distributions_share_material(
                            material_distribution.as_ref().unwrap(),
                            &parent_distribution,
                            continuous,
                        )
                        || (extra_error <= (8.0_f32 / 255.0).powi(2)
                            && distributions_support_continuation(
                                material_distribution.as_ref().unwrap(),
                                &parent_distribution,
                                continuous,
                            )))
                {
                    let score = (0..3)
                        .map(|c| {
                            (material_distribution.as_ref().unwrap()[c][2]
                                - parent_distribution[c][2])
                                .powi(2)
                        })
                        .sum::<f32>();
                    same_material.push((score, parent, rgb));
                }
            }
            continue;
        }
        // A sampled dark outline may have a noisy local median. Two
        // matching native samples in the same incident owner prove
        // this tiny island is not a new material or an isolated dot.
        if incident.contains(&parent)
            && component.iter().all(|&i| {
                samples
                    .iter()
                    .filter(|&&j| {
                        (0..3).all(|c| {
                            (image.pixels[i][c] - image.pixels[j][c]).abs()
                                <= (if neutral { 12.01 } else { 6.01 }) / 255.0
                        })
                    })
                    .take(2)
                    .count()
                    >= 2
            })
        {
            repeated.push((parent, rgb));
        }
        let nearest = samples
            .iter()
            .min_by_key(|&&j| {
                let dx = (2 * (j % w) + 1) as isize - (region.min_x + region.max_x) as isize;
                let dy = (2 * (j / w) + 1) as isize - (region.min_y + region.max_y) as isize;
                (dx * dx + dy * dy, j)
            })
            .copied()
            .unwrap();
        // Median alone does not represent an antialiased outline:
        // local samples include both its ink and its coverage ramp.
        // Keep observed endpoints so a noisy dark fragment is not
        // forced to become a separate material below that median.
        for extreme in [
            selected.iter().min_by(|&&a, &&b| {
                rgb_to_oklab(image.pixels[a])
                    .l
                    .total_cmp(&rgb_to_oklab(image.pixels[b]).l)
            }),
            selected.iter().max_by(|&&a, &&b| {
                rgb_to_oklab(image.pixels[a])
                    .l
                    .total_cmp(&rgb_to_oklab(image.pixels[b]).l)
            }),
        ]
        .into_iter()
        .flatten()
        {
            let endpoint = image.pixels[*extreme];
            // An isolated extremum may belong to a narrow coloured
            // rim or to noise, rather than the adjacent material.
            // Only recurrent source colours are material endpoints.
            if selected
                .iter()
                .filter(|&&j| {
                    (0..3).all(|c| (image.pixels[j][c] - endpoint[c]).abs() <= 3.01 / 255.0)
                })
                .take(3)
                .count()
                == 3
            {
                edge_parents.push((parent, endpoint));
            }
        }
        edge_parents.push((parent, image.pixels[nearest]));
        if incident.contains(&parent) {
            parents.push((parent, rgb));
        } else {
            edge_parents.push((parent, rgb));
        }
    }
    if large {
        if let Some(&(_, parent, rgb)) = same_material
            .iter()
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        {
            proposal.material_union = true;
            for &i in component {
                proposal.changes.push((i, parent as u32, rgb));
            }
            proposal.merges += 1;
        }
        return proposal;
    }
    if let Some(&(parent, rgb)) = repeated
        .iter()
        .min_by_key(|&&(parent, _)| std::cmp::Reverse(segmentation.regions[parent].area))
    {
        proposal.material_union = true;
        for &i in component {
            proposal.changes.push((i, parent as u32, rgb));
        }
        proposal.merges += 1;
        return proposal;
    }
    // Border samples contain coverage of other paints. Requiring
    // every sample to equal one solid invents a new material for each
    // small piece of the same noisy outline. Compare robust material
    // colours, then bound the *additional* source error of sharing it.
    let mut material = [0.0; 3];
    for c in 0..3 {
        let mut values: Vec<_> = component.iter().map(|&i| image.pixels[i][c]).collect();
        values.sort_by(f32::total_cmp);
        material[c] = (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5;
    }
    let source_variance = component
        .iter()
        .map(|&i| {
            (0..3)
                .map(|c| (image.pixels[i][c] - material[c]).powi(2))
                .sum::<f32>()
                / 3.0
        })
        .sum::<f32>()
        / component.len() as f32;
    let shared = parents
        .iter()
        .filter_map(|&(parent, rgb)| {
            if (0..3)
                .any(|c| (material[c] - rgb[c]).abs() > (if neutral { 32.0 } else { 16.0 }) / 255.0)
            {
                return None;
            }
            let extra = component
                .iter()
                .map(|&i| {
                    (0..3)
                        .map(|c| {
                            (image.pixels[i][c] - rgb[c]).powi(2)
                                - (image.pixels[i][c] - material[c]).powi(2)
                        })
                        .sum::<f32>()
                        / 3.0
                })
                .sum::<f32>()
                / component.len() as f32;
            (extra
                <= (12.0_f32 / 255.0).powi(2)
                    + if neutral {
                        source_variance / component.len() as f32
                    } else {
                        0.0
                    }
                || (neutral && extra * component.len() as f32 <= (32.0_f32 / 255.0).powi(2)))
            .then_some((extra, parent, rgb))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    if let Some((_, parent, rgb)) = shared {
        proposal.material_union = true;
        for &i in component {
            proposal.changes.push((i, parent as u32, rgb));
        }
        proposal.merges += 1;
        return proposal;
    }
    // Preserve uniform/chromatic interiors. Noisy neutral coverage
    // shoulders can occupy a few raster rows, so require a wider
    // interior there before treating them as an independent material.
    let uniform = component
        .iter()
        .all(|&i| (0..3).all(|c| (image.pixels[i][c] - material[c]).abs() <= 6.0 / 255.0));
    if component.iter().any(|&i| {
        let (x, y) = (i % w, i / w);
        let radius = if uniform || !neutral { 1 } else { 2 };
        x >= radius
            && x + radius < w
            && y >= radius
            && y + radius < h
            && (y - radius..=y + radius)
                .all(|yy| (x - radius..=x + radius).all(|xx| labels[yy * w + xx] as usize == id))
    }) {
        return proposal;
    }
    // Source endpoints describe local shading. Nearby neutral paints
    // can explain partial coverage at a terminal, but cannot replace
    // a fully covered isolated mark or become a disconnected owner.
    parents.extend(edge_parents);
    let mut best = None::<(f32, Vec<(usize, u32, [f32; 3])>)>;
    for left in 0..parents.len() {
        for right in left + 1..parents.len() {
            let (first, c1) = parents[left];
            let (second, c2) = parents[right];
            if first == second || (!incident.contains(&first) && !incident.contains(&second)) {
                continue;
            }
            if delta_e_ok(rgb_to_oklab(c1), rgb_to_oklab(c2)) < 6.0 {
                continue;
            }
            let mut assignments = Vec::new();
            let mut error = 0.0_f32;
            for &i in component {
                let (alpha, residual) = coverage_mixture(image.pixels[i], c1, c2);
                // Independent coloured dots, highlights and extrema
                // fail the bounded two-parent colour model.
                if !(-0.02..=1.02).contains(&alpha)
                    || residual > 2.0
                    || (!incident.contains(&first) && alpha >= 0.95)
                    || (!incident.contains(&second) && alpha <= 0.05)
                {
                    break;
                }
                error = error.max(residual);
                let amount = partition_coverage_alpha(image.pixels[i], c1, c2);
                let (owner, rgb) = if amount >= 0.5 {
                    (first, c1)
                } else {
                    (second, c2)
                };
                // A nearby paint is coverage evidence, not permission
                // to create a disconnected island of its label. Keep
                // ownership in an actual incident face.
                let (owner, rgb) = if incident.contains(&owner) {
                    (owner, rgb)
                } else {
                    // Only a mixed junction without a resolved material
                    // interior may redistribute coverage to its incident
                    // paints. A narrow opaque stripe (such as a wiper)
                    // must not be replaced with the neighbouring glass.
                    if resolved_interior || (incident.len() < 2 && !varying_coverage) {
                        break;
                    }
                    parents
                        .iter()
                        .filter(|e| incident.contains(&e.0))
                        .min_by(|a, b| {
                            let distance =
                                |c: [f32; 3]| (0..3).map(|k| (c[k] - rgb[k]).powi(2)).sum::<f32>();
                            distance(a.1).total_cmp(&distance(b.1))
                        })
                        .copied()
                        .unwrap()
                };
                assignments.push((i, owner as u32, rgb));
            }
            if assignments.len() == component.len() && best.as_ref().is_none_or(|b| error < b.0) {
                best = Some((error, assignments));
            }
        }
    }
    // At a three-face junction the raster sample can mix all three
    // paints. It need not lie on any single two-colour segment.
    if best.is_none() {
        for first in 0..parents.len() {
            for second in first + 1..parents.len() {
                for third in second + 1..parents.len() {
                    let entries = [parents[first], parents[second], parents[third]];
                    if entries[0].0 == entries[1].0
                        || entries[0].0 == entries[2].0
                        || entries[1].0 == entries[2].0
                    {
                        continue;
                    }
                    if entries.iter().any(|e| !incident.contains(&e.0)) {
                        continue;
                    }
                    let mut assignments = Vec::new();
                    let mut worst = 0.0_f32;
                    for &i in component {
                        let Some((error, weights)) =
                            junction_mixture(image.pixels[i], entries.map(|p| p.1))
                        else {
                            break;
                        };
                        if error > 2.0 {
                            break;
                        }
                        worst = worst.max(error);
                        let winner = (0..3)
                            .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
                            .unwrap();
                        assignments.push((i, entries[winner].0 as u32, entries[winner].1));
                    }
                    if assignments.len() == component.len()
                        && best.as_ref().is_none_or(|b| worst < b.0)
                    {
                        best = Some((worst, assignments));
                    }
                }
            }
        }
    }
    if let Some((_, assignments)) = best {
        proposal.changes.extend(assignments);
        proposal.merges += 1;
    }
    proposal
}

// Preserve even shallow, spatially coherent shading. Overlapping colour
// histograms alone do not establish that a gradient is sampling noise.
fn has_resolved_shading(image: &Raster, pixels: &[usize]) -> bool {
    if pixels.len() < 8 {
        return false;
    }
    let n = pixels.len() as f64;
    let mx = pixels
        .iter()
        .map(|&i| (i % image.width) as f64)
        .sum::<f64>()
        / n;
    let my = pixels
        .iter()
        .map(|&i| (i / image.width) as f64)
        .sum::<f64>()
        / n;
    let mean: [f64; 3] = std::array::from_fn(|c| {
        pixels
            .iter()
            .map(|&i| image.pixels[i][c] as f64)
            .sum::<f64>()
            / n
    });
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    let (mut xc, mut yc, mut variance) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for &i in pixels {
        let x = (i % image.width) as f64 - mx;
        let y = (i / image.width) as f64 - my;
        xx += x * x;
        xy += x * y;
        yy += y * y;
        for c in 0..3 {
            let v = image.pixels[i][c] as f64 - mean[c];
            xc[c] += x * v;
            yc[c] += y * v;
            variance[c] += v * v;
        }
    }
    let det = xx * yy - xy * xy;
    (0..3).any(|c| {
        let (sx, sy) = if det > 1e-9 {
            (
                (xc[c] * yy - yc[c] * xy) / det,
                (yc[c] * xx - xc[c] * xy) / det,
            )
        } else if xx > yy && xx > 0.0 {
            (xc[c] / xx, 0.0)
        } else if yy > 0.0 {
            (0.0, yc[c] / yy)
        } else {
            return false;
        };
        if sx * xc[c] + sy * yc[c] < variance[c] * 0.75 {
            return false;
        }
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for &i in pixels {
            let v = sx * ((i % image.width) as f64 - mx) + sy * ((i / image.width) as f64 - my);
            low = low.min(v);
            high = high.max(v);
        }
        high - low > 2.0 / 255.0
    })
}

// Five source quantiles per RGB channel, measured in face interiors.
// Near medians alone cannot distinguish a real low-contrast paint boundary
// from quantizer fragments. Require overlapping central source distributions
// and bounded tails as well. Coverage reconstruction keeps its own size limit.
fn colour_distribution(image: &Raster, pixels: &[usize]) -> [[f32; 5]; 3] {
    std::array::from_fn(|c| {
        let mut values: Vec<_> = pixels.iter().map(|&i| image.pixels[i][c]).collect();
        values.sort_by(f32::total_cmp);
        [5, 25, 50, 75, 95].map(|q| values[(values.len() - 1) * q / 100])
    })
}

fn distributions_share_material(a: &[[f32; 5]; 3], b: &[[f32; 5]; 3], continuous: bool) -> bool {
    let continuous = continuous && (0..3).all(|c| a[c][4].max(b[c][4]) <= 24.0 / 255.0);
    (0..3).all(|c| {
        (a[c][2] - b[c][2]).abs() <= (if continuous { 12.0 } else { 6.0 }) / 255.0
            && (continuous
                || (a[c][1] <= b[c][3] + 1.01 / 255.0 && b[c][1] <= a[c][3] + 1.01 / 255.0))
            && (a[c][0] - b[c][2]).abs() <= (if continuous { 16.0 } else { 12.0 }) / 255.0
            && (a[c][4] - b[c][2]).abs() <= (if continuous { 16.0 } else { 12.0 }) / 255.0
            && b[c][4] - b[c][0] <= (if continuous { 24.0 } else { 16.0 }) / 255.0
    })
}

fn distributions_support_continuation(
    a: &[[f32; 5]; 3],
    b: &[[f32; 5]; 3],
    continuous: bool,
) -> bool {
    // A continuous interface can connect quantizer fragments of any colour.
    // Still require overlapping central distributions: continuity alone must
    // not erase a distinct flat face. The caller also bounds added source error.
    (0..3).all(|c| {
        (a[c][2] - b[c][2]).abs() <= (if continuous { 12.0 } else { 6.0 }) / 255.0
            && a[c][1] <= b[c][3] + 1.01 / 255.0
            && b[c][1] <= a[c][3] + 1.01 / 255.0
            && (a[c][0] - b[c][2]).abs() <= 16.0 / 255.0
            && (a[c][4] - b[c][2]).abs() <= 16.0 / 255.0
            && b[c][4] - b[c][0] <= 16.0 / 255.0
    })
}

// OKLab is sensitive to a few RGB code values near black. Coverage
// reconstruction accepts either its perceptual bound or bounded native RGB
// noise; the latter must hold in every channel, not only in luminance.
fn coverage_error(value: [f32; 3], predicted: [f32; 3]) -> f32 {
    let rgb_error = (0..3)
        .map(|c| (value[c] - predicted[c]).abs())
        .fold(0.0_f32, f32::max);
    let rms = ((0..3)
        .map(|c| (value[c] - predicted[c]).powi(2))
        .sum::<f32>()
        / 3.0)
        .sqrt();
    delta_e_ok(rgb_to_oklab(value), rgb_to_oklab(predicted))
        .min((2.0 * rgb_error / (14.0 / 255.0)).max(2.0 * rms / (8.0 / 255.0)))
}

fn coverage_mixture(value: [f32; 3], first: [f32; 3], second: [f32; 3]) -> (f32, f32) {
    [false, true]
        .into_iter()
        .map(|linear| {
            let (alpha, predicted) = mixture_prediction(value, first, second, linear);
            (alpha, coverage_error(value, predicted))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

fn junction_mixture(value: [f32; 3], parents: [[f32; 3]; 3]) -> Option<(f32, [f32; 3])> {
    let mut best = None;
    for linear in [false, true] {
        let transform = |rgb: [f32; 3]| {
            if linear {
                rgb.map(srgb_to_linear_channel)
            } else {
                rgb
            }
        };
        let p = parents.map(transform);
        let v = transform(value);
        let u: [f32; 3] = std::array::from_fn(|c| p[0][c] - p[2][c]);
        let w: [f32; 3] = std::array::from_fn(|c| p[1][c] - p[2][c]);
        let d: [f32; 3] = std::array::from_fn(|c| v[c] - p[2][c]);
        let dot = |a: [f32; 3], b: [f32; 3]| (0..3).map(|c| a[c] * b[c]).sum::<f32>();
        let (uu, ww, uw) = (dot(u, u), dot(w, w), dot(u, w));
        let determinant = uu * ww - uw * uw;
        if determinant <= 1e-9 {
            continue;
        }
        let a = (dot(d, u) * ww - dot(d, w) * uw) / determinant;
        let b = (dot(d, w) * uu - dot(d, u) * uw) / determinant;
        let weights = [a, b, 1.0 - a - b];
        if weights.iter().any(|&v| !(-0.08..=1.08).contains(&v)) {
            continue;
        }
        let predicted = std::array::from_fn(|c| {
            (0..3)
                .map(|k| weights[k] * p[k][c])
                .sum::<f32>()
                .clamp(0.0, 1.0)
        });
        let rgb = if linear {
            predicted.map(linear_to_srgb_channel)
        } else {
            predicted
        };
        let positive = weights.map(|v| v.max(0.0));
        let sum = positive.iter().sum::<f32>();
        let clipped: [f32; 3] =
            std::array::from_fn(|c| (0..3).map(|k| positive[k] * p[k][c] / sum).sum());
        let clipped = if linear {
            clipped.map(linear_to_srgb_channel)
        } else {
            clipped
        };
        if (0..3).any(|c| (clipped[c] - rgb[c]).abs() > 16.0 / 255.0) {
            continue;
        }
        let error = coverage_error(value, rgb);
        if best.as_ref().is_none_or(|&(previous, _)| error < previous) {
            best = Some((error, weights));
        }
    }
    best
}

#[cfg(test)]
include!("../tests/unit/segment_micro.rs");
