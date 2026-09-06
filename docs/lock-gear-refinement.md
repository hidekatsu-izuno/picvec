# Lock arches and the gear's elliptical ring

The lock and gear in `cliparts-6x6` exposed three different reasons why a
geometric-looking figure could retain fragmented free curves.

## Refinement boundaries must compare vector geometry

The gear's native refinement was previously rejected at the replacement
border. That comparison bilinearly enlarged the coarse 1414 × 1414 raster
preview, spreading antialias coverage from the separator and the gear into
the narrow background gap. The actual SVG, rendered at source resolution,
has a clear gap there. The rejection measured preview interpolation rather
than the final vector join.

The base SVG is now parsed once and rendered directly into each refinement's
small source-space crop. The same 2/255 channel tolerance and two-pixel border
check remain. Scoring and candidate ordering still use the original coarse
preview; only join validation uses the native vector render. Memory scales
with the crop, rather than a second full-resolution sheet.

## A separator touching an object is not its outline

The lock's arch crosses the long pale horizontal separator near source row
3344. Binary foreground connectivity attached it to the oversized sheet grid,
so it was never proposed as an independent refinement.

For grouping, thin canvas-spanning bands are separated from object support,
including their antialias shoulders. Nearby bands still constrain padding;
only bands crossing a figure's bounds may enter its crop. The original source
and matte are retained for subsequent validation. A thin connector leading
from a figure to a large component still does not authorize cutting it.

A crossing separator cannot be independently refitted on the two sides of a
crop either: tiny colour, coverage and width differences would create a seam.
For keyed inputs, a long pale band that demonstrably passes behind opaque
foreground can instead be factored into a single source-supported vector
backdrop. Its half-coverage boundaries, median colour and peak coverage come
from the source. Only the band and its faint antialias shoulders leave the
fitting matte; foreground crossing it remains opaque and occludes the
backdrop. The backdrop is inserted beneath the complete composed SVG, so the
same line continues behind every refinement. Broad panels, isolated lines,
and weak translucent grids remain in the ordinary model. No raster is embedded.

## Partial ellipses and localization error

The primitive fitter previously supported straight lines and circular arcs,
while the ellipse fitter required a complete closed contour. An arch or
part of an elliptical rim therefore fell back to unrelated cubic pieces.

Open intervals now also try an ellipse. A similarity correction places both
ends of the fitted conic on the existing shared graph knots. Final validation
checks the corrected model, endpoint tangents, maximum and RMS source error,
and the bidirectional corridor. A short or nearly straight interval, a nearly
closed chord, or a retraced path cannot use this model. The accepted angular
span is between 90 and 270 degrees. Endpoints remain exact after serialization
roundoff is removed.

For complete ellipses, large raster-derived material boundaries can also have
several pixels of localization error. Their corridor is the larger of the
original pixel bound and 2% of the contour span, capped at 4.5 working pixels.
The RMS, winding, persistent-corner and graph-order checks still apply.
This does not make every nearly round Paint region an ellipse: the gear's
outer ring contains narrow backtracking slivers which cannot be projected
onto one ordered loop. Its supported inner ring can use one complete model.
The cap bounds geometric regularization; it is not an assertion that the
source was authored as mathematically exact ellipses.

## Validation

The regenerated [whole-sheet SVG](../sample/output/cliparts-6x6.svg)
uses the default quality settings with green-key removal. The before version
already includes the preceding connected-sphere improvements.

Both SVGs were rendered at the original 5016 × 5016 resolution over the source
green background. Mean CIEDE2000 error was measured on source foreground
pixels (key-derived coverage at least 0.5), including each complete figure:

| Figure | Source crop (x0, y0, x1, y1) | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| Lock | (980, 3310, 1525, 4020) | 4.7521 | 2.3479 | 50.6% |
| Gear | (1740, 3310, 2390, 4020) | 4.1178 | 2.3380 | 43.2% |

Native refinement now accepts all 36 proposed figures, compared with 34 of
35 before. Added refinement data is 31,510,948 bytes, within the unchanged
32 MiB budget. The complete SVG grows from 37,281,739 to 38,624,539 bytes
(3.6%). The native gear crop accepts one complete inner ellipse; the output
still contains many shading regions and is not a minimal primitive drawing.
Fine shading boundaries remain visible, especially when enlarged.

All 36 sheet cells were also compared. Improvement is not uniform: the
largest increase in mean error is the brain at row 5, column 1, from 3.0011
to 3.2054 (+0.2043). Visual inspection preserves its main outline and network,
with local shading/curve changes. The previously improved connected spheres
change from 4.0250 to 4.0500. The factored pale separator also changes appearance
slightly; its geometry and colour are estimated from raster coverage.

The diagnostic test suite passes with 203 tests, zero failures and three
ignored tests; the full-size sheet planning test also passes when explicitly
run. Clippy passes with warnings denied for all diagnostic targets and the
default library build.

Tests cover rotated elliptical arches in both directions with exact shared
endpoints, rejection of short/retraced paths, bounded large-ellipse fitting
with authored dents, a native vector join that a coarse preview incorrectly
rejects, and rejection of a genuinely damaged join. Separator tests cover
occlusion, subpixel boundaries, preservation of foreground alpha, both axis
orientations, ordinary/translucent lines, and broad panels. Rendering checks
verify that the factored line remains behind foreground and leaves the rest
of the background transparent. The full-size planning regression checks the
lock's cap above the separator as well as the gear and existing sheet figures.

```bash
mise exec -- cargo test --locked --offline --features diagnostics
mise exec -- cargo test --locked --offline --features diagnostics \
  clipart_sheet_refines_whole_figures_with_and_without_keying -- --ignored
mise exec -- cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
```
