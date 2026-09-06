# Integrating gradient reconstruction

The selected research basis is Chakraborty et al., *Image Vectorization via
Gradient Reconstruction*, Eurographics 2025, DOI
[10.1111/cgf.70055](https://doi.org/10.1111/cgf.70055).
This is an integration design following the rejection of the three local
performance prototypes. The new backend is not implemented yet.

The investigation used the complete author-hosted
[paper](https://techmatt.github.io/pdfs/imageVectorizationViaGradientReconstruction.pdf)
and [supplement](https://techmatt.github.io/pdfs/imageVectorizationViaGradientReconstructionSupplemental.pdf),
in addition to the current Rust source. The author PDFs were downloaded for
reading into `/tmp/picvec-gradient-reconstruction-research/`; they are not
vendored into this repository.

## What adopting this research means

The paper's method does include an initial colour segmentation. Its principal
architectural change is to use separation constraints from discontinuities to
construct smooth domains **before** estimating their gradient fills. Therefore,
the intended replacement is the complete segmentation-and-Paint path, not just
a new direction candidate in the current fitter.

The paper combines piecewise-smooth preprocessing, local colour segmentation,
discontinuity-constrained graph cuts, solid/linear/radial reconstruction,
boundary-pixel assignment and shared-curve fitting. Its supplement specifies
local region merging and sequential cuts, plus a curve-fitting procedure.
This establishes an implementation basis, but not bit-exact reproduction:
several numerical conventions and heuristic choices remain to be specified.
([Main paper, sections 3–4](https://techmatt.github.io/pdfs/imageVectorizationViaGradientReconstruction.pdf),
[supplement, sections 1–3](https://techmatt.github.io/pdfs/imageVectorizationViaGradientReconstructionSupplemental.pdf).)

The broader [redesign review](fundamental-performance-redesign.md) described
additional ideas such as independent shading resolutions, layered correction
and residual-driven domain refinement. Those are picvec adaptations, not a
description of the original paper. Establish the research baseline before
adding them, so a failed comparison can be attributed to a particular change.

## Mapping to the existing implementation

| Component | Existing implementation | Integration decision |
| --- | --- | --- |
| Smooth image and discontinuity evidence | `edge::perceptual_smooth` uses bilateral filtering; `edge::classify` also recovers structural evidence | Implement the research preprocessing separately. Keep source RGB, alpha and protected line evidence available to the compatibility path. Do not present bilateral filtering as a reproduction of the research preprocessing. |
| Initial region partition | `segment::build_palette` assigns a global perceptual palette | Replace with the research's spatially local region partition. Preserve pixel ownership and region moments for subsequent cuts and fallback. |
| Region construction | `gradient::merge_partition` evaluates neighbouring Paint merge proposals | Replace with a compact adjacency graph and discontinuity constraints. Gradients are fitted to the resulting domains rather than to every possible union. |
| Gradient parameters | `fitted_linear_directions_from_lightness` uses spatial covariance and a lightness plane; later searches test more models | Implement direct colour-derivative-based reconstruction and one-dimensional stop fitting for the new domains. Keep fitted Paint attached to its domain. |
| Boundary-pixel ownership | Multiple regularization, thin-Paint and source-supported merge passes | Give the new backend its own final ownership pass using its reconstructed neighbouring fields. Add picvec's thin-line/alpha protection as an explicit adaptation. |
| Shared curves | `geometry.rs` already has shared topology, continuity processing, a Potrace-style polygon dynamic program, primitive and ellipse fitting | Initially reuse this geometry backend on the new **final** partition. It implements the required sharing principle but is not the exact supplementary curve optimizer. Measure this remaining cost before deciding whether to replace it. |
| SVG emission | `svg.rs` supports the current Paint vocabulary and source-alpha masks | Reuse for the compatibility path. Keep research-only field forms in an experimental representation. |

Reusing shared geometry is deliberate: the core segmentation and Paint change
can reduce its input boundary graph without discarding existing contour-quality
work. If geometry still consumes the speed budget, implement and compare the
supplement's curve optimizer on the **same** final partition. This comparison
must be labelled as a separate backend difference.

## Available reference code and CPU execution

No complete author implementation of the 2025 vectorizer was located through
the checked Adobe research page, author publication page and publisher record.
That is a search result, not a claim that no such code exists.

The cited preprocessing method does have an official implementation,
[tum-vision/fastms](https://github.com/tum-vision/fastms), with both CPU/OpenMP
and CUDA execution. Its README identifies GPLv3 terms. Use it as an optional
external numerical reference; the proposed shipped implementation is Rust
written from the mathematical method, with research attribution, rather than
copying or linking this implementation into picvec.

The preprocessing paper's reported real-time result uses a GPU. Neither that
result nor the 2025 paper's 2.05-second 1024x1024 table establishes performance
under picvec's fixed four-worker CPU setup. The CPU implementation must record
preprocessing iterations, convergence and elapsed time explicitly.
([Preprocessing research record](https://portal.fis.tum.de/en/publications/real-time-minimization-of-the-piecewise-smooth-mumford-shah-funct/),
[2025 timing table](https://techmatt.github.io/pdfs/imageVectorizationViaGradientReconstruction.pdf).)

The new backend should use the caller's existing Rayon pool. Do not create an
independent worker pool, start an unrestricted OpenMP runtime, or introduce a
GPU requirement. Single-threaded graph operations are included in timing.

## Separate research behaviour from picvec output constraints

Use one experimental implementation with two explicit evaluation profiles:

| Profile | Purpose | Output and acceptance |
| --- | --- | --- |
| Research | Check the core method on generated inputs and understand its reconstruction capacity | Allow the paper's gradient representation in a standalone experimental renderer/serializer. Report the differences from the paper, including any reuse of existing curve fitting. This output is not an Office-compatibility result. |
| Picvec-compatible | Decide whether the method solves this project's performance problem | Use the existing Paint vocabulary, at most five stops, native-source colour/structure validation, and the existing alpha contract. Time all conversion and repair. This is the adoption candidate. |

This separation does not entail executing both pipelines in normal conversion.
Run them separately during evaluation. Do not weaken `Config::validate` or the
ordinary serializer's requirements to accommodate research-only models.

Specific adaptations that need independent measurements:

- **Five-stop limit:** the paper does not impose it. For the compatible path,
  fit under the limit and validate the actual resulting Paint. If the model
  fails, subdivide the domain or retain local fallback. Never truncate a fitted
  stop sequence and assume its previous error still applies.
- **Radial vocabulary:** current `Paint::Radial` and `register_gradient` encode
  translation and axis scaling of a centred radial gradient. The paper also
  considers displaced focus and affine transforms. A research field must not
  be silently converted to this narrower representation. Constrained refitting
  and its residual must be explicit; richer fields remain experimental until
  output compatibility is established.
- **Quality metric:** the research's L1 objective and its local colour
  thresholds are not interchangeable with picvec's tonal-adjusted CIEDE2000
  thresholds. Record both during research evaluation. Adoption uses the
  existing source-based quality criteria.
- **Source alpha and ink:** preserve straight RGB and coverage separately.
  Weak highlights, authored gaps and narrow source silhouettes must survive
  preprocessing and boundary assignment. The paper's boundary-pixel treatment
  is not a sufficient specification of picvec's alpha or structural-ink rules.
- **Complex shading:** local fallback is part of the measured algorithm. Its
  time, SVG size and boundary count must be reported, particularly for photos.

## Proposed module boundary

Keep the reconstruction machinery in a separate `src/reconstruction/` module
behind an experimental feature. This path should own a compact domain graph,
discontinuity constraints, fit observations, accepted fields and an explicit
final pixel-owner map. A result contains:

```text
final partition + fitted fields + source boundary evidence + work counters
```

Adapt that result to `Segmentation` and `Vec<Paint>` once for existing geometry
and SVG serialization. The adapter must build canonical colours and region
statistics as metadata without running palette quantization again.

For domains accepted by the new backend, bypass the current sequence of
`regularize_boundaries`, `merge_partition`, `split_adaptive_paint_patches`,
`fit_all_without_topology`, and `merge_source_supported_paints`. Any retained
stage must have a named correctness purpose and a measured cost. Feeding new
domains through the entire old pipeline would defeat the intended replacement.

An initial standalone example can consume a raster and emit an SVG and work
report. Production integration should then call the same backend from
`pipeline.rs` so resolution selection, native-alpha handling and adaptive
refinement are compared under identical policies. A standalone crop result
alone does not establish an end-to-end product speedup.

## Implementation order and concrete checks

1. **Numerical specification and complete experimental path.** Implement
   preprocessing, local partition, constrained cuts, field reconstruction,
   boundary ownership and the adapter to shared geometry. Persist small
   diagnostic maps and field parameters. Keep all departures from the paper in
   one implementation note rather than hiding them in existing quality knobs.
2. **Compatible fields and detail preservation.** Add the five-stop/radial
   constraints, source alpha and protected-line handling to the same backend.
   Measure the cost and quality difference from the research profile. A broad
   fallback must be counted as a failure to accelerate that image.
3. **Whole-pipeline comparison.** Freeze a common source snapshot including
   current quality fixes. Run the existing photograph, car, transparent
   illustration, window and stress corpus at four workers. Time every stage,
   including validation, repair, probes and adaptive patches. Apply the 2x
   geometric-mean speedup and quality gates from the redesign review, then
   expand to the full clip-art sheet and contour/tonal regressions.

Tests should establish behaviour independently of implementation details:

- A separated pair stays disconnected even when another path goes around a
  material boundary; tiny-graph cut costs can be checked against exhaustive
  enumeration. Sequential cuts need not equal the global multicut optimum.
- A rotated gradient that changes colour direction along its axis is
  reconstructed without requiring a global RGB-plane fit to explain it.
- Centred radial and displaced-focus fixtures distinguish supported fields
  from fields needing explicit approximation. They must not silently render
  as one another.
- A profile requiring more than five stops exposes the added compatible-path
  error or subdivision cost; it cannot pass using the research renderer alone.
- Shared boundaries close consistently at junctions. Source-based tests cover
  faint line contrast, narrow gaps, loops and straight-RGB/alpha composition on
  white and black. The old stress baseline's missing faint lines cannot serve
  as a passing reference for line preservation.

## Numerical choices to resolve during reproduction

The papers supply an algorithm, not every implementation convention. Record
the image scale and derivative discretization used by the Mumford–Shah solver,
boundary conditions and stopping tolerance; region update order and mean
weighting; graph-weight colour units and cut-pair ordering; projection binning;
radial initialization and iteration limits; and the model acceptance threshold.
Test these choices on small analytical fixtures before tuning on the sample
corpus. Failed gradient fits and violated separation constraints need explicit
outcomes, not silent changes of tolerance.

The first deliverable is a complete, inspectable research-based SVG conversion
path with those choices documented. Adoption is determined by the compatible
path's measured quality and runtime, not by matching a published timing number.
