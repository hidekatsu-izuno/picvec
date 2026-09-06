# Ellipses across shaded Paint regions

The later [lock and gear extension](lock-gear-refinement.md) adds fixed-endpoint
elliptical arcs and a capped size-dependent corridor for large closed loops.
The measurements below document the initial whole-contour implementation.

The window buttons in `cliparts-6x6` were fitted as several free curves even
though their complete outlines support an ellipse. The source-boundary
primitive fitter supported lines and circular arcs, while the ellipse output
variant was not a whole-contour fitting path. Closed continuity tracks were
split at corners or arbitrary storage anchors before fitting. Later fitting
of individual colour-pair chains could further change those curves.

The geometry pass now collects complete closed material contours, including
holes and contours whose opposite side changes colour. It fits a translated,
scaled quadratic system to equally spaced observations and accepts only
positive-definite ellipses. Rotation is unrestricted. Acceptance requires one
complete winding, bounded radial maximum and RMS error, preservation of
persistent corners, and the existing bidirectional raster corridor of
`sqrt(2) + 0.25` working pixels. Small authored squares, dents, open paths,
retraced loops and degenerate observations remain free curves.

The accepted model is eight cubic ellipse segments of at most 45 degrees.
Ordered projection maps all raster graph nodes onto those same master curves.
Both incident Paint faces reuse the same slices, and later local smoothing
and circular-arc fitting preserve the ellipse. The arbitrary first node of a
closed loop can move with its neighbours. Projection unwraps positions across
that storage seam before pooling raster backtracking; otherwise a one-pixel
step at the seam could reject a valid rotated ellipse.

Whole models take precedence over local colour fits. Partially overlapping
ellipse candidates are not combined, and contours touching the canvas retain
their existing geometry. No source coordinates, particular colours or sample
label IDs participate in the fitting rules. The diagnostic geometry report
adds `fitted_ellipse_contours`; this counts fitted contours, including holes,
rather than SVG ellipse elements.

Residual structural selection also respects the accepted contour. Otherwise
its source-versus-Paint comparison sees the intentional geometric correction
as missing ink and draws the old lumpy outline back on top. A narrow residual
stroke is redundant only when its complete sampled centreline follows one
accepted ellipse and its colour is already present in nearby Paint. Branches,
missing colours, wide strokes and ink explicitly transferred out of Paint
retain structural ownership. `structural.suppressed_ellipse_retraces` counts
the removed duplicates.
When Paint's colour is inaccurate, the corrective ink remains. A narrow
stroke following a single ellipse can reuse its centre, rotation and axes,
offset by half the stroke width towards the observed ink. Its source corridor
and traversal must still validate. This keeps the original stroke colour and
width while avoiding another independent raster fit. Diagnostic reports count
these as `structural.aligned_ellipse_strokes`.

The output retains the original Paint subdivisions and supported line-width
variation. A shaded ring can have elliptical inner and outer contours without
becoming a single constant-width SVG stroke. Open outlines and objects that
fail the complete-ellipse checks still use the existing line, arc and free
curve fitting.

## Validation

For the native 836 × 836 window cell (source rectangle x=836..1671,
y=836..1671), nine closed contours fit, including all three button exteriors
and the red and white button interiors. The final residual pass omits three
duplicate strokes and aligns four colour-correction strokes. This crop has
no shared-loop fallback or curve downgrade.

The table measures the final shared Paint curves against their best-fitting
ellipse, with 65 parameter samples per cubic. Units are native working pixels.
It measures consistency of the vector contour, not pixel error against the
source or the separate corrective ink layer. Colour junctions can divide the
eight master cubics into more output pieces without changing their geometry.

| Window button exterior | Previous maximum radial deviation | New maximum radial deviation |
|---|---:|---:|
| Red | 0.6832 | <0.0001 |
| Yellow | 1.2415 | <0.0001 |
| White after key removal | 0.6999 | <0.0001 |

[Source / before / after comparison](../sample/comparison/cliparts-6x6-ellipses.png)
shows the window from the complete sheet, rendered at source resolution over
white. The button row renders the SVG directly at the enlarged display scale;
the source panel enlarges the original raster.

The complete keyed sheet retains exactly the same 34 accepted refinement
regions out of 35 evaluated candidates. SVG size changes from 28,557,227 to
28,591,500 bytes (+0.12%). All 36 cells were compared at 5016 × 5016 on the
original green background, using source foreground pixels with
`1 - (G - max(R, B)) >= 0.5` in normalized RGB. The largest increase in a
cell's mean CIEDE2000 is 0.0117 (the organization chart). The window changes
from 2.1705 to 2.1716. These small colour-error increases accompany the
geometric simplification; they are not claims of improved raster fidelity.
The organization chart and disk were also inspected for changes to small
features. The base pass keeps its existing three curve downgrades and has no
shared-loop fallback.

Some narrow Paint fragments, shading transitions and partial corrective ink
still remain around the controls. This extends geometric regularization; it
does not reconstruct every outlined object as one uniform-width stroke.

Regression tests cover noisy and rotated ellipses in both directions, small
and large squares, a dent, an open contour and a repeated loop. A shared-graph
regression uses a shaded elliptical ring: its inner contour crosses multiple
Paint pairs, both contours remain within 0.25 pixels of the generating
ellipses, and no shared-loop fallback or curve downgrade occurs. It exercises
both axis-aligned and rotated rings, including projection at the storage seam.
Residual-selection tests distinguish a duplicate outline from a branch, a
different ink colour, missing Paint and transferred source ink.
All 183 regular Rust tests pass (one existing full-sheet planning test is
ignored), as do Clippy with warnings denied, formatting and diff checks.

```bash
mise exec -- cargo test --locked --offline --features diagnostics
mise exec -- cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo fmt --all -- --check
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
```
