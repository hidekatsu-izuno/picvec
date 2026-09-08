# A line, one cubic, or two cubics for a shared contour

The man's shoulder in row 3, column 1 of `cliparts-6x6` contains a gently
curved boundary and a thin highlight. Straightening the entire shoulder would
lose its rounded transition. Previously, failed line/circle/ellipse fits kept
the existing free-curve pieces. Recursive cubic fitting could split repeatedly
at raster-scale errors without explicitly testing a fixed one- or two-cubic
model over the combined interval.

`geometry_bezier` now runs after analytic primitive regularization, before the
shared contour is sliced between incident Paint regions. It tries a line,
then one cubic, then two cubics with one interior knot. Each candidate passes
the source, baseline, endpoint-tangent and persistent-corner checks before
selection. A candidate must use fewer segments than the interval it replaces.
Existing lines and consecutive pieces of supported analytic arcs are protected.

## Fitting and validation

Observations are resampled at one-pixel spacing. Endpoints remain fixed to the
shared graph. Unconstrained cubic handles are fitted freely; constrained handles
retain the required tangent direction and have positive, bounded lengths.
Iterative closest-point reparameterization and normal-distance least squares
reduce the geometric error instead of assuming source arclength is Bezier `t`.
An ill-conditioned fit retains the previous estimate or rejects the model.

For two cubics, a bounded coarse-to-fine search tries interior positions along
the source, including the worst residual of the one-cubic model. The knot is
estimated from a small local average; both incident cubics share its position
and tangent direction (G1 continuity). The search refines promising rejected
positions too, so a coarse trial just outside the tolerance does not prevent
finding a supported nearby knot.

Candidate maximum error is capped at the smaller of the caller's corridor and
1.25 working pixels. RMS error must be at most 60% of that cap. Distances use
projections onto a sampled curve polyline, rather than nearest sample vertices.
The existing bidirectional source and baseline corridors still bound final
movement. Persistent corners cannot move farther from the model than the
previous error plus 0.125 pixels, with the existing 0.25-pixel floor. Retraced
intervals and large loops are excluded from this compact-model pass.

The consolidation search is bounded to windows of at most 12 existing pieces,
with shorter fallback windows. It is not an exhaustive minimum-node solution
for the whole image. Full shared-graph mapping remains the final authority;
local model support alone does not authorize changing a junction's topology.

## Diagnostics

Build with `--features diagnostics`, then set
`PICVEC_BEZIER_DIAGNOSTICS=/tmp/bezier.jsonl`. Records append to this file and
include interval endpoints, source point count, model, maximum/RMS error and
decision. Use a fresh path for each run. `PICVEC_CONTINUITY_DIAGNOSTICS` can be
enabled alongside it to inspect final contour adoption and ordered mapping.

Decisions distinguish source maximum/RMS failures, endpoint tangents, source
or baseline corridors, protected corners, backtracking and ill-conditioned
fits. `supported` describes a locally valid candidate; `selected` identifies
the chosen interval replacement. `final_corridor` reports an assembled-master
rollback. The inexpensive two-cubic screen reports `ordered_fit_residual`;
other measured candidate errors use `distance_to_curve_polyline`.

For example, the native-resolution shoulder interval from (314, 536) to
approximately (226.743, 574.647), containing 135 source observations, reports:

| Model | Maximum error (px) | RMS error (px) | Result |
| --- | ---: | ---: | --- |
| Line | 7.455 | 5.290 | Reject: maximum error |
| One cubic | 3.256 | 0.957 | Reject: maximum error |
| Two cubics, selected knot | 1.204 | 0.398 | Supported |

The final interval also passes validation against the original, unresampled
source. This example shows why increasing the line tolerance alone would be
the wrong fix for that portion of the shoulder.

## Tests

`src/test-data/man-shoulder-contours.json` records three continuity contours
from the original 836 × 836 source cell. Regression checks enforce fewer
pieces, exact endpoints and inter-piece connections, required tangents and
successful ordered graph mapping. Additional tests cover a single cubic with
nonuniform observations, two cubics with one smooth knot, protected corners
and retraced paths. The earlier attached-disc arc regression also remains in
place, including its supported arc spanning more than 90 degrees.

The isolated validation checkout contains the established pipeline plus the
preceding arc change and this change; unrelated concurrent reconstruction and
Paint changes are excluded. It passes 208 tests, with zero failures and three
ignored tests, and Clippy with warnings denied for diagnostic targets and the
default library.

```sh
mise exec -- cargo test --release --locked --offline --features diagnostics
mise exec -- cargo clippy --release --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo clippy --release --locked --offline --lib -- -D warnings
```

## Regenerated sample

The updated
[SVG](../sample/output/cliparts-6x6.svg) accepts all 36 native refinements with
no boundary rejections. Shared-loop fallbacks remain zero; the base geometry
retains four curve downgrades, unchanged from before.

At the original 5016 × 5016 resolution, mean CIEDE2000 error over source
foreground pixels (key-derived coverage at least 0.5) is:

| Source crop | Before | After |
| --- | ---: | ---: |
| Figure: (140, 1680)–(735, 2460) | 3.0509 | 3.0237 |
| Shoulder: (185, 2177)–(360, 2307) | 3.7353 | 3.5165 |

The shoulder error decreases by 5.9%. Across all 36 sheet cells, the average
change in cell mean error is −0.0238. The largest increase is +0.0076, in the
chart at row 1, column 4; not every local colour-error measurement improves.

The whole SVG decreases from 38,512,262 to 37,744,369 bytes (2.0%). Added
refinement data is 30,773,499 bytes, within the unchanged 32 MiB budget.
The native source-cell probe also preserves both shoulder connections while
reducing a four-piece contour to two cubics. Local snapshot tests can reduce
the unconstrained counterpart to one cubic; final output still depends on
the tangents and shared-graph mapping of each actual interval.

Fine shading subdivisions and local colour artifacts remain. The change
reduces the complexity of supported contour intervals; it does not recreate
the original illustration's authored shapes and gradients in full.

## Short-interval extension (2026-09-08)

The original search excludes intervals with fewer than 16 source observations
or an endpoint displacement below 16 working pixels. A new fallback fits two
existing cubic pieces to one cubic when their combined length is 2–32 working
pixels. It also runs on short source chains previously excluded from `compact`.
The regular one/two-cubic search retains its existing limits and runs first.

The fallback uses samples of the existing curves, fixes both endpoints and
their tangent directions, and rejects a corner or cusp at the removed node
(incident tangent dot product below 0.98). Maximum ordered fitting residual is
at most 0.20 working pixels, or the caller's tolerance if smaller; RMS is at
most 60% of that limit. Bidirectional sampled baseline corridors, original-source
corridors, persistent-corner protection and the final assembled-chain checks
remain required. These are sampled geometric bounds, not a proof of identical
rasterization. The existing graph validation still controls shared-contour
adoption. The pass does not repeatedly refit its own output, avoiding cumulative
drift. Analytic-piece protection and line protection remain in place.

Regression tests exercise a short S-shaped contour, retained endpoints and
tangents, retained change of curvature, successful shared-graph mapping, and
rejection of a corner, a small bulge and unsupported source observations.

### Sample comparison

These measurements isolate short-pair compaction, before the subsequent
[closed-outline reconstruction](closed-bezier-bands.md). The checked-in sample
outputs also include that later change.

Compared against `db159b8`, using release builds with diagnostics and
`--threads 4 --verbose --quality-metrics`, with all other CLI settings left at
their defaults. Counts below come from the completed SVG, using
`scripts.picvec_eval.svg_metrics.svg_complexity`, including final structural
corrections. Segments count lines and curves; they are not XML element counts.

| Input | SVG segments, before → after | Reduction | SVG bytes, before → after |
| --- | ---: | ---: | ---: |
| `car.png` | 36,900 → 35,918 | 2.66% | 2,135,842 → 2,091,931 |
| `boy_and_turtle.png` | 2,479 → 2,471 | 0.32% | 130,753 → 130,401 |
| `viewport1.jpg` | 161,139 → 149,267 | 7.37% | 8,462,440 → 7,974,028 |

Independent librsvg renders were compared at original input dimensions against
the source, compositing transparency onto white. CIEDE2000 uses scikit-image;
SSIM uses its default local windows with `channel_axis=2, data_range=1.0`.
These differ from the CLI's embedded resvg and whole-image luminance SSIM.

| Input | Mean ΔE00, before → after | Local-window SSIM, before → after |
| --- | ---: | ---: |
| `car.png` | 0.808058 → 0.808419 | 0.954165 → 0.954137 |
| `boy_and_turtle.png` | 0.555421 → 0.555447 | 0.970043 → 0.970039 |
| `viewport1.jpg` | 6.292128 → 6.293501 | 0.645609 → 0.645471 |

To avoid hiding local regressions in the full-image mean, pixels changing by
more than one 8-bit RGB level were dilated by three pixels and assessed
separately. Mean ΔE00 in those neighbourhoods changes from 4.6070 to 4.6517
(car), 3.3469 to 3.3903 (boy), and 8.6266 to 8.6299 (photo).
The worst 48×48 tile, searched at a 24-pixel stride, increases by 0.1909,
0.0062 and 0.1523 respectively. Those tiles were visually inspected at native
resolution and with both SVGs rendered at 4×; no conspicuous loss of contour
or detail was observed. Small numerical regressions remain; these three
samples do not guarantee perceptual equivalence for every input.

Region counts stay unchanged. XML path counts stay at 2,030 for car and 91 for
boy; the photo changes from 14,766 to 14,785 because the changed Paint preview
affects subsequent structural correction. Thus this change reduces curve/node
complexity and bytes, not necessarily path elements. Shared-loop fallbacks and
shared-curve downgrades remain zero on all three examples. Timing runs overlapped
with compilation and other checks, so they do not establish a speed improvement.

Validation for the isolated short-pair pass: all-feature/all-target release
tests pass (253 passed, 4 ignored),
as do formatting and diff-whitespace checks. Strict Clippy encounters an
existing `needless_range_loop` warning at `src/stroke_model.rs:609`; with only
that lint allowed, all-feature/all-target and default-library Clippy pass.

```sh
mise exec -- cargo test --release --locked --offline --all-features --all-targets
mise exec -- cargo fmt --all --check
mise exec -- cargo clippy --release --locked --offline --all-features --all-targets -- -D warnings -A clippy::needless_range_loop
mise exec -- cargo clippy --release --locked --offline --lib -- -D warnings -A clippy::needless_range_loop
```
