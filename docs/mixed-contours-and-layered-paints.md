# Mixed contour intervals and layered Paint reduction

The teacher in `sample/input/cliparts-6x6.png` has a long pointer and angular
collar. Closed-ring reconstruction alone cannot simplify these connected
contours. Two independent changes address their remaining complexity.

* Mixed line/cubic intervals can participate in bounded Bezier compaction.
  Shared endpoints, endpoint tangent constraints, persistent source corners,
  and both source and incumbent curve corridors remain enforced. Analytic
  primitive runs retain their existing protection. Short connection lines
  of at most 2 working pixels remain protected from this additional search.
  Mixed intervals also require a 0.25-pixel bidirectional corridor against
  their incumbent curves, in addition to the existing raster-source checks.
* A layered Paint repeats its complete geometry for each overlay. After Paint
  merging, fit a single linear/radial field to the existing composite colors.
  Accept only when the mean composite CIEDE2000 error is at most 0.8, its
  90th percentile at most 1.6, and every owner's pixel differs by at most 3.
  Against the original paint reference, the mean error increase must be at
  most 0.25 and no individual pixel may increase by more than 1.5. Multiple
  highlights that cannot be represented by one field retain their layers.

This is perceptual approximation, not pixel identity. It reduces exported
geometry commands and actual repeated path elements; it does not merge
unrelated material owners or replace all dark contours with straight lines.

Regression coverage includes mixed intervals, preserved corners, a fragmented
rotated silhouette, redundant shaded fields, and separate local highlights.

## Controlled teacher comparison

A native crop at `(2537, 1692, 754, 766)` was converted with the same chroma
implementation on both sides, enabling only the changes above for `After`.
The crops in `sample/comparison/cliparts-6x6-teacher-detail.png` show the source,
Before, and After at 2x. This is a controlled crop conversion, not a claim
about the adaptive sheet's final path counts.

Count only paths whose entire control hull lies inside the stated ROI. These
regions include shading and adjacent small faces, not exclusively black ink.
Segments below count `C`, `L`, and `A`, excluding move/close commands.

| Native ROI | Paths before → after | Segments before → after | Mean source ΔE00 before → after |
| --- | ---: | ---: | ---: |
| Pointer `(365,345)–(487,535)` | 38 → 33 | 345 → 273 | 1.922 → 1.931 |
| Collar `(112,470)–(340,638)` | 159 → 155 | 971 → 906 | 2.601 → 2.613 |

Raster comparison uses librsvg at native resolution. The source's pure green
key background is excluded from the color measurement. The small mean error
increases are expected from bounded perceptual approximation; they are not
reported as improved pixel fidelity. Shared connections and collar corners
were also checked in the rendered comparison.

## Final sheet after highlight preservation

After completing the concurrent chroma changes and fixing the pie highlight's
antialias ownership, all seven sample SVGs were regenerated. The final embedded
teacher crop is shown in `sample/comparison/cliparts-6x6-teacher-final.png`.
Using the same fully-contained ROIs, it has 35 paths / 277 segments around the
pointer and 148 paths / 887 segments around the collar. These are final-sheet
counts, separate from the controlled crop comparison above. The pointer's
connections and the collar's corners were checked in the actual rendered SVG.
