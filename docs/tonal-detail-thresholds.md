# Shadow and highlight detail thresholds

The warehouse in row 6, column 2 of `cliparts-6x6.png` lost its pale wall
seams even in its native-resolution adaptive replacement. The smoothed
reference still contained them, but the default light-colour palette tolerance
assigned the seam and wall to the same colour. A synthetic example also showed
that a later Paint merge could flatten a retained seam because its average
error was low. Similar tolerances can flatten detail near black.

The correction adjusts error thresholds by source lightness instead of
detecting and protecting particular line shapes. `Config::tonal_detail_scale`
uses smoothstep interpolation from 1 at `dark_knee_lstar` (default L*=45)
to 0.50 at black and 0.30 at white. The scale multiplies the existing
lightness-dependent smoothing and palette tolerances, the Paint merge error
limit, and the solid-fill range and gradient-promotion thresholds. Merge
proposals use the stricter of their two child lightness scales so a midtone
average cannot conceal error in a shadow or highlight.

With default settings, the effective palette tolerance is 1.25 DeltaE00 at
black, 2.5 at L*=45, and 1.5 at white (previously 2.5, 2.5, and 5.0).
The CLI's configured tolerances remain multipliers of the same tone response.
There is no new seam mask or orientation-specific protection rule.

The default adaptive SVG budget increases from 24 to 32 MiB to accommodate
the extra tonal detail without dropping previously refined figures. When a
replacement's rendered boundary does not match, it may move inward by up to
four source pixels, but only through fully transparent source pixels. If a
faint halo prevents trimming, it may instead expand by up to four pixels
inside the already fitted child, through background below the planner's
1/16 alpha support threshold. Expansion retains the original figure and
cannot cross foreign foreground. The same 2/255 rendered-boundary tolerance
still applies. Baseline and refined quality scores are recomputed for the
resulting rectangle.

Tests check continuity and the tightening of both ends of the tone range,
including extreme knee settings, and render faint detail against both near
white and near black backgrounds while retaining an authored gap.

## Validation (2026-09-06)

The saved sheet was regenerated at default quality settings with chroma-key
removal. All 35 proposed refinements are accepted, including every previously
refined figure. SVG size changes from 28,557,227 to 34,349,390 bytes; added
refinement bytes are 27,325,356, within the new 32 MiB limit.

The following opaque wall rectangles measure mean
CIEDE2000 error at the original source resolution. Rectangles are relative to
the 710 x 660 crop beginning at source (900, 4120), with exclusive right/bottom
coordinates.

| Wall area (left, top, right, bottom) | Before | After |
|---|---:|---:|
| Roof underside (270, 160, 490, 216) | 1.2596 | 0.7386 |
| Left wall (80, 365, 190, 500) | 1.0002 | 0.7491 |
| Right wall (565, 365, 645, 500) | 0.9447 | 0.7329 |

All 36 cells were compared on the original green background using the source
foreground mask `1 - (G - max(R, B)) >= 0.5`, with RGB normalized to [0, 1].
Twenty-four cells have lower mean error. Across the entire sheet, mean error
for source L*>85 changes from 2.9598 to 2.8644, and for 25<=L*<=85 from 2.7609
to 2.6779. Shadow pixels (L*<25) change from 4.1612 to 4.1793, a slight increase;
the synthetic shadow-detail test verifies retained contrast, while the sheet
does not demonstrate an overall shadow-error improvement. Very faint shading
can still simplify, and individual contours can shift with the new partition.

Validation: 191 tests passed, one existing full-sheet test remained ignored;
the complete keyed sheet was checked separately as above. After the final
boundary adjustment, all 12 non-ignored adaptive tests passed. Clippy with
warnings denied, formatting, and whitespace checks pass.

Reproduction:

```bash
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
mise exec -- cargo test --locked --offline --features diagnostics
```
