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
