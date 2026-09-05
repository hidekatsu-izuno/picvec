# Outline recovery and bounded-CPU performance

Whole-band recovery now requires a complete connected ink region, including
consistent width and colour at every measured section. Partial outline
networks remain filled regions whose shared contour segments are fitted to
lines and arcs. See [complete-band acceptance](#complete-band-acceptance-and-contour-connections)
for the current acceptance rules; earlier measurements below describe the
broader interval recovery used at the time.

The outline recovery pass measures both sides of a dark band before Paint
segmentation. A supported interval becomes a centre-line with one width and
ink colour; the two adjacent paints extend underneath it independently. This
avoids retaining a jagged dark Paint band beneath a cleaner SVG stroke.
The existing line fitter then fits the recovered centre-line. Duplicate
source-edge candidates and later residual strokes cannot claim the same band.

Recovery requires a narrow band with a flat ink core, two brighter incident
paints, consistent width and colour, sufficient length, and enough pixels
explained by an ink/paint mixture. A colour step or diffuse shadow fails this
test. Diagonal sample spacing is included in underpaint coverage. Unsupported
intervals keep the existing Paint representation. The pass does not infer
arbitrary variable-width brush strokes or replace the existing junction model.

CPU optimizations share Gaussian/Hessian work between bright and dark ridge
responses, visit actual palette owners when merging small components, and fit
independent final strokes in the existing Rayon pool. The default worker limit
and `--threads` behavior are unchanged. No GPU backend is introduced.

Adaptive refinement owns connected figures as whole regions. Source alpha or
reliable flat-background evidence can establish separation. This applies to
opaque images too and does not remove their backgrounds. Eight-connected
support preserves diagonal joins; overlapping padded bounds merge disconnected
details into the same fit. Source support and a rendered two-pixel border check
prevent a crop boundary from cutting a figure or introducing a colour step.
Unknown backgrounds and oversized figures retain the global base model.

There is no fixed-grid fallback and no residual-bounds trim through a figure.
The size budget is enforced by retaining oversized figures, not subdividing
them. The default candidate limit is 64 so individual figures on an icon sheet
can be evaluated. The worker limit and 24 MiB added-SVG budget are unchanged.
This is conservative figure-level refinement; it does not attempt to reconcile
two independent gradient or stroke fits after splitting an object.

## Reproduction

Runs use release diagnostics binaries, four workers, and adaptive refinement:

```bash
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
/usr/bin/time -v target/release/picvec sample/input/cliparts-6x6.png /tmp/cliparts.svg \
  --threads 4 --remove-chroma-key-background --verbose
```

Omit `--remove-chroma-key-background` to exercise opaque-background planning.
Both modes use the same no-split rule. The 5016 × 5016 source uses a
1414 × 1414 base resolution; finer regions surround intact figures instead of
bisecting them at source coordinates 1254, 2508, and 3762.

Measured on 2026-09-05 (individual runs, not statistical timing estimates):

| Measurement | Previous grid, key removed | Whole figures, key removed | Whole figures, opaque |
|---|---:|---:|---:|
| Elapsed time | 240.523 s | 154.409 s | 133.594 s |
| SVG bytes | 20,155,715 | 19,226,911 | 17,387,422 |
| Peak RSS | 2,160,872 KiB | 1,152,020 KiB | 931,152 KiB |
| Accepted regions | 16 | 26 | 20 |
| Sampled refined mean DeltaE00 | 1.344066 | 1.994104 | 1.934102 |

These speed/size results are not an equal-fidelity comparison. Conservative
separation and acceptance leave some figures at the coarser global model; the
sampled colour error increases. This avoids introducing a split through an
unseparated figure, but refinement coverage and fine shading remain limitations.

The database icon's old vertical cut is at source x=1254. On its flat top,
source rows 2595..2669, the mean DeltaE00 between columns 1253 and 1254 is
0.044 in the original, 1.565 in the grid output, and 0.035 in the new keyed
output. Measurements use SVG crops rendered at the original pixel scale.
The full outputs and database/code crops were inspected in both background
modes. White rules separating icons are present in the original input.

## Validation

Rust regression tests compare the shared ridge responses against independently
computed polarities and the indexed component merge against dense traversal,
including label maps and reassignment counters. Outline tests cover two
different incident colours, duplicate ownership, detector integration,
diagonal coverage, colour steps, and diffuse shadows. Adaptive tests cover transparent and opaque connected figures, unknown
backgrounds, oversized figures, nested details, rendered border mismatches,
full replacement, and overlapping rectangles that leave a hole.

The independent Python stroke metric now resolves parent presentation
attributes and inline styles, excludes definitions and hidden subtrees, and
reads optimized path commands. Previously, structural paths inheriting
`fill="none"` were skipped. These are anchor diagnostics, not a rendered
perceptual score; external CSS, use expansion, and mask/clip coverage are not
evaluated. In particular, do not credit a roughness improvement merely to
removing an invisible base layer from the file.

Validation commands (149 regular Rust tests plus a full-size sample regression,
9 Python tests, warning-free Clippy):

```bash
RAYON_NUM_THREADS=4 mise exec -- cargo test --offline --locked --all-features -j 2 -- --test-threads=1
RAYON_NUM_THREADS=4 mise exec -- cargo test --offline --locked --all-features -j 2 clipart_sheet_refines_whole_figures_with_and_without_keying -- --ignored --test-threads=1
mise exec -- cargo clippy --offline --locked --all-targets --all-features -j 2 -- -D warnings
PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
mise exec -- cargo fmt --check
git diff --check
```

The Python test command requires the evaluator's NumPy dependency.

## Native-alpha thin lines

`cliparts.png` contains black grid RGB with fractional alpha. The former alpha
shoulder classifier could erase a long partial-alpha component merely because
it touched an opaque junction elsewhere. Shoulder reassignment now requires
local opaque support, while isolated partial bands keep durable coverage.
Tests also preserve ordinary opaque antialias contours and intentional gaps.

The initial grid correction restored stroke colours after composited structural
selection. The Wikipedia regression below exposed limits of that approach;
structural selection now uses straight RGB throughout. Paint beneath mask-owned
thin silhouettes and partial coverage also keeps source RGB.

The regenerated 1600 × 1200 sample was rendered on transparent and white
backgrounds. At 6,730 black-grid reference positions, the count with less than
one quarter of source coverage decreased from 354 to 1; the remaining point
is the antialiased endpoint at (1200, 1165), not an interior gap. Among visible
grid samples, mean RGB channel spread decreased from 27.019 to 0 (8-bit RGB),
and maximum spread decreased from 248 to 0. Comparisons use a five-pixel strip
around each original grid axis and exclude strips containing coloured source
foreground. The generated sample SVG is updated with this result.


## Wikipedia logo: alpha/RGB separation and highlight preservation

The grayscale `wikipedia_logo_1_0.png` exposed a design error: the saturated
preview backing was also used as hidden RGB and as the source for structural
selection. It could become exported ink or leak through the interpolated mask.
Restoring stroke colours afterwards was insufficient and could weaken bright
ridges by sampling their antialiased centres.

Native-alpha conversion now extends existing foreground RGB into uncovered
samples with a deterministic nearest-frontier wavefront. Edge classification,
Paint fitting and structural residual selection all use straight RGB. The
structural Paint preview has no alpha mask; the final SVG and quality preview
apply alpha independently. A stroke without covered centreline support is
rejected. The arbitrary comparison backing no longer participates in those
colour decisions.

Source bright-ridge evidence is retained through palette construction. Colours
supported by this evidence use a maximum 1.5 DeltaE00 merge tolerance instead
of the larger global light-colour threshold. This reduces the loss of thin
highlights before geometry fitting. It does not guarantee preservation of every
highlight or remove the remaining fragmented seam geometry.

At native 1058 × 1058 resolution, rendered on neutral #555555, the number of
pixels with an RGB channel spread greater than 2 decreased from 2,577 to zero
(maximum spread: 252 to 2, in 8-bit RGB). For 39,885 opaque bright-detail sample
positions selected from source opposing-side contrast, the number losing more
than 15/255 red-channel brightness decreased from 6,263 to 4,898. This is a
heuristic brightness diagnostic, not a count of complete preserved lines.
Visual inspection still shows missing or fragmented highlights.

The native `cliparts.png` grid regression retained the previous result across
6,730 reference positions: zero RGB channel spread and one low-coverage endpoint,
with no measured interior gap. Both sample SVGs were regenerated. The full Rust
suite (all features) passes 152 tests with one manual sample test ignored;
Clippy is warning-free. Added checks cover hidden-RGB invariance, achromatic
alpha edges, source-highlight palette separation, and curved highlights beside
dark seams on both opaque and transparent backgrounds.

CPU workers remain capped at four. Single-run conversion times were 15.611 s
for Wikipedia and 34.428 s for cliparts, compared with the prior diagnostic runs
of 12.803 s and 28.097 s. These runs are not controlled speed benchmarks, but
show that this quality correction has a runtime cost. Foreground colour
extension creates additional hidden geometry; removing it from the shared
boundary fitter changed visible curves and failed the grid/highlight tests,
so that optimization was not retained. Further work is needed on both native
line topology preservation and geometry cost.


## Continuous colour rims at binary-alpha boundaries

The native Wikipedia image uses only alpha 0 and 255. Its thin bright boundary
is encoded in edge RGB, with pixel-phase-dependent intensity. Treating these
samples as ordinary Paint produced disconnected white chips next to the
transparent area, even after removing the blue backing contamination.

The converter now measures a narrow one-sided RGB band along each alpha
contour. It estimates the incident interior colour and the bright band's
coverage, rejects flat-colour silhouettes and steep interior gradients, and
smooths width measurements along the contour. Binary coverage is required;
authored partial-alpha areas keep their existing coverage model.

The alpha mask and boundary ink use the same contour fitter. The ink's exact
cubic pieces are obtained by de Casteljau subdivision, so the two boundaries
cannot drift apart through independent fitting. Compatible adjacent pieces
are joined. A narrow incident-colour stroke first covers the old boundary
chips; the bright stroke is then drawn on the shared mask curve. All incident
colour restoration precedes all bright ink, preventing overlapping span caps
from erasing the preceding white span. This pass follows Paint and structural
selection, so it does not alter the interior partition.

The regenerated Wikipedia SVG contains 301 recovered boundary-ink paths and
is 1,381,552 bytes (previous output: 1,208,811 bytes). Inspection against a dark
background confirms a continuous rim in the formerly dotted inner opening and
around the globe. Rendered pixels more than five source pixels inside the
alpha silhouette are identical before and after the correction; RGB channel
spread remains at most 2/255. Two conversion runs took 15.762 s and 17.633 s with four
workers; these are not controlled performance benchmarks.

Regression checks render a synthetic rim and verify contrast at all 360 angular
positions. They also verify that an unoutlined dark silhouette gains no white
rim and that an authored gap is retained. The all-feature Rust suite passes
154 tests (one manual large-sample test remains ignored); Clippy passes with
warnings denied.

## Fairing raster steps on the shared coverage boundary

The continuous rim above still followed pixel-scale staircase oscillations.
The coverage contour used the common colour-boundary cubic fitter but disabled
its Gaussian fairing and used a 0.3-pixel fitting tolerance. It now enables
corner-preserving fairing (sigma 1.5 contour samples, tolerance 0.65 pixels),
identically for the mask and incident rim. This is shared curve fitting, not
a replacement of alpha by an arbitrary RGB background or a unified RGBA face
partition; intentional partial opacity still uses the coverage layers.

Raw coarse corners are anchored before smoothing. Narrow reversals also anchor
their surrounding cap: the initial unprotected smoothing shortened a grid end
and failed the existing rendered-grid regression. Contours with at most 12
samples including closure retain the previous fit to avoid smoothing away
small islands and holes.

New regressions check that a rasterized circle has less than 8 radians of total
absolute turning (a circle has 2 pi), with radial error below 0.9 pixels, and
that rectangle corners and a one-pixel island survive. Existing rendered tests
cover white-rim continuity, authored gaps, and a translucent black grid. The
all-feature suite passes 156 tests with one manual test ignored; Clippy passes
with warnings denied. CPU worker limits remain unchanged.

The final Wikipedia conversion took 16.440 s with four workers and produced
1,260,807 bytes with 242 recovered boundary-ink paths. Rendering the reported
puzzle tab at 4x shows the repeated stair-step waves removed. Pixels more than
five source pixels inside the silhouette remain identical to the preceding
continuous-rim output, and channel spread remains at most 2/255. This timing
is a single diagnostic run, not a controlled speed comparison.

The regenerated `cliparts.png` grid remains neutral (zero RGB channel spread
at 6,730 reference positions). Two low-coverage endpoint/fringe samples fall
below one quarter of source alpha, compared with one previously: (1200,1165)
and (1200,1186), the latter having source alpha 42/255. These are not new
interior grid breaks; the low-alpha edge coverage is not pixel-exact.

## Fitting straight lines and circular arcs

Source-supported geometric fitting now precedes free-form cubic fitting on
corner-delimited shared boundaries and structural centre-lines. Open lines
retain both endpoints exactly. A circular arc's centre is fitted on the
perpendicular bisector of its endpoint chord; a closed circle passes through
its fixed storage anchor. Uniform arclength sampling prevents dense raster
steps from receiving extra fitting weight. Maximum and RMS residuals, ordered
angular travel, and explicit stroke tangent constraints reject unsupported
fits. Short intervals, reversals, noncircular loops and ill-conditioned shallow
arcs retain the existing fitter.

A final pass also consolidates neighbouring shared curve pieces, with a
64-piece lookahead limit. It checks the candidate against both the original
contour and the previous geometry in both directions. Shared endpoints remain fixed, and supported corners are protected from
rounding away. Both incident Paint faces reuse the resulting
chain; no face independently snaps its side of a common boundary.

Straight pieces now retain native SVG line commands. Previously, emitting
rounded cubic controls could prevent a mathematically straight line from
being recognized by the exact serializer. Circular models use cubic pieces
of at most 45 degrees, followed by the existing 0.01-pixel analytic arc
normalization. The closed-circle serializer now accepts 4 through 16 cubic
pieces, subject to its existing radial and winding checks, instead of requiring
exactly four. Alpha-mask and rim paths share the same fitter, and rim joining
handles both line and cubic commands.

The model remains conservative: it does not infer a common width for two
independently Paint-owned boundaries, join detached source components, or
replace a noncircular silhouette with a circle. On the database and source-code
icons, enlarged output still contains variable-width bands and fragmented
shading. Primitive fitting improves geometric regularity for supported spans;
it does not complete the conversion of every painted band to an authored stroke.

Regression coverage includes noisy diagonal lines, clockwise and major arcs,
closed circles with different storage origins, SVG line/arc/circle output,
noncircular shapes, reversals, graph tangent constraints, tiny support and
cumulative source drift. The existing shared-partition, corner, alpha-rim and
thin-grid rendered regressions also pass.

The keyed 5016 × 5016 sheet was regenerated with four workers and default
adaptive refinement. A rerun of the previous binary produced a byte-identical
copy of the previously committed SVG. Both versions accepted the same 26 of
28 evaluated whole-figure regions. Single-run results on 2026-09-05:

| Measurement | Before | After |
|---|---:|---:|
| SVG bytes | 21,539,248 | 21,102,874 |
| Serialized path line commands | 22,139 | 33,030 |
| Serialized path arc commands | 4,175 | 9,053 |
| Circle elements | 0 | 3 |
| Sampled refined mean DeltaE00 | 2.02090 | 1.99339 |
| Elapsed time | 194.060 s | 193.943 s |

Command counts use `picvec_eval.svg_metrics.svg_complexity`, including
paths in definitions and the retained base. They measure representation,
not visible line quality. Timings are individual diagnostic runs with other
validation work occurring during parts of the runs, not a speed benchmark.

[Enlarged source / before / after comparison](../sample/comparison/cliparts-6x6-lines.png)
shows the code-document frame, database rim and Wi-Fi arcs. These were rendered
from the full sheet SVG at original source scale, with a green backing for
comparison to the opaque source. Mean absolute RGB error (8-bit channel units)
across the respective full 836 × 836 crops changed from 4.37147 to 4.37069,
5.05316 to 5.05286, and 4.68800 to 4.67780. Visual improvement is modest; the
comparison still shows the Paint-band and shading limitations described above.
The updated SVG and comparison are stored under `sample/`.

Validation: 162 Rust tests pass (one manual large-sample regression ignored),
9 Python evaluator tests pass, Clippy passes with warnings denied, and formatting
and diff checks pass. The Python tests use the cached evaluator environment,
which includes NumPy and Pillow.

## Joint band recovery across detector fragments

The primitive fitter above cannot regularize width while both sides of a
painted band remain independent. The joint recovery pass now receives source
contours before residual overlay classification, as well as the existing dark
boundary graph. It evaluates both dark and bright bands. A provisional
1.2-pixel edge width no longer limits the search for the core of a 12-pixel
outline: core search extends at least eight pixels from the seed and the
normal profile reaches at least twenty pixels to observe both incident paints.
A weak distance tie-break and a locally supported core prevent an antialiased
edge minimum from hiding a valid interior core.

Both 20--80% transitions must be sharp, in addition to the existing contrast,
width, length, colour and ink/paint mixture checks. Mild core shading is
allowed, but the exported ink uses the median interior colour instead of the
extreme used to locate the band. Steps and diffuse dark or bright shadows
remain Paint-owned. Widths outside 1.5--16 working pixels and strongly varying
bands remain unsupported.

Nearby detector fragments are joined before rejecting short intervals. A
spatial endpoint index, up to eight joining rounds, width/colour agreement,
and tangent agreement bound the work. Every interpolated bridge cross-section
must independently recover the same band within the 2.5-pixel maximum gap;
a one-pixel authored break vetoes the join. The most complete candidates claim
Paint ownership before overlapping opposite-edge duplicates.

A recovered band is fitted once. Its exact centre-line, measured width and
interior colour survive residual selection unchanged, instead of being
independently refitted and recoloured after its Paint has already been removed.
Open intervals use butt caps and retain a 1.5-pixel original-Paint overlap at
their ends. This avoids extending a round cap into a real break or erasing
pixels the cap only partially covers. Closed bands retain their continuous
closure.

New tests cover outside-edge seeds at four raster phases, removal of old ink
under a constant-width stroke, complete circular bands, bright ink between two
different incident paints, shaded-core colour bias, joining short fragments,
and rendered one- and eight-pixel gaps. The existing step/shadow rejection is
also tested for both polarities.

The final 5016 × 5016 keyed sheet was regenerated with default adaptive
refinement and four workers. Both versions accepted 26 of 28 evaluated regions.
The base pass recovered 175 joint bands (previously 16); this counter excludes
additional bands in the adaptive regions. Its underpaint ownership increased
from 27,134 to 53,593 pixels.

[Source / before / after band comparison](../sample/comparison/cliparts-6x6-bands.png)
uses crops rendered from the full sheet SVG at source resolution, then enlarged
for inspection. The database's white separator is more continuous; the
source-code frame's straight band has less width variation. The Wi-Fi example
also shows remaining gradient and thin-outline irregularities.

For the source-code icon at sheet origin (836, 0), a scan of local rows
260..659 and columns 150..203 measures its left border. The two crossings use
half the luminance contrast between the row's ink minimum and each incident
paint. On the full-sheet render, width standard deviation decreases from
0.11015 to 0.06436 source pixels; mean width changes from 12.2550 to 12.3901
(source: 12.2463). Centre-line residual about a fitted straight line changes
from 0.04316 to 0.04709 pixels. This is a local rendered-width diagnostic,
not a claim of improved centre-line accuracy or uniform improvement everywhere.

Separate native 836 × 836 crop conversions recover 6 bands for source-code
(previously 4) and 16 for database (previously 5). Code's underpaint ownership
increases from 1,844 to 18,401 pixels and database's from 4,939 to 18,902.
Their remaining total structural stroke counts decrease from 50 to 44 and
39 to 38, respectively: recovery can replace several residual fragments with
one model even when the number of recovered bands increases.

The colour/complexity tradeoff is not uniformly favourable. On full-sheet
836 × 836 source-code, database and Wi-Fi crops, mean absolute RGB error
(8-bit channel units) changes from 4.37069 to 4.65377, 5.05286 to 5.12061,
and 4.67780 to 4.77998. Removing ink before segmentation also changes the
surrounding Paint fit; subtle shading remains a limitation. The sampled refined
mean DeltaE00 over all planned regions changes from 1.99339 to 1.98512.

The SVG grows from 21,102,874 to 21,703,335 bytes. Conversion took 258.638 s,
compared with the preceding implementation's 193.943 s diagnostic run. These
are individual runs, with some validation work overlapping, not a controlled
benchmark. The default and explicit CPU worker limits are unchanged. This
change improves supported band representation at a runtime and file-size cost.

Validation: 169 Rust tests pass (one manual large-sample regression ignored),
9 Python evaluator tests pass, and Clippy, formatting and diff checks pass.

## Car: coverage ownership and coloured contour continuity

The right body contour and the upper panel seam in `car.png` exposed three
ownership errors before and during shared-curve fitting. Removing the SVG
structural-ink layer left the defects visible in Paint. The underpaint and
smoothed RGB reference still had a continuous contour; the segmented labels
first acquired the intermittent erosion.

* Sleeve assignment used nearest-owner pixel distances as coverage. Those
  distances depend on quantisation thickness and tie on diagonal steps. In the
  reproduced input, one 42-pixel dark-red fragment was partly assigned to the
  exterior despite its source colour. Accepted sleeves now use the existing
  sRGB/linear-RGB two-parent mixture model and its half-coverage decision.
  Topology still determines which two incident parents are eligible.
* A repeated sleeve group could assign a small member to a parent belonging
  to a different member, even when that parent was not incident to it. Removed
  this group-wide owner override; each accepted member retains its local pair.
* Dark ink protection relied on near-black lightness. A coloured line can be
  clearly darker than both incident faces without being near black. A median
  source lightness more than six units below both parents now rejects the
  sleeve interpretation. This deliberately keeps ambiguous strong dark
  extrema as Paint instead of assuming they are sharpening artifacts. Light
  ringing and genuine two-parent coverage retain their absorption tests.

The shared geometry continuity classes also used contact counts before colour
consistency for coreless fragments. A change in raster phase could therefore
attach the same coloured contour alternately to the surface and exterior,
placing independent curve anchors on one visible outline. Incident colour
consistency now takes precedence, with contact counts breaking colour ties.
Paint labels and colours are not merged by this geometry classification.

Validation includes 18 combinations of slope, subpixel phase and parent label
order, coloured ink with a detector gap, light ringing beside retained ink,
and a contour whose exterior contact count exceeds its surface contact count.
The older medium-dark synthetic "ringing" fixture had no evidence distinguishing
its dark red stripe from ink; it now asserts preservation. The distant-third-
face topology fixture uses actual intermediate coverage so it still tests
parent selection independently of ink protection. All 170 regular Rust tests
pass (one manual large-sheet test remains ignored), as do Clippy and formatting.
The native code-icon crop from `cliparts-6x6.png` was also converted and inspected.

The car SVG was rendered at native size for measurements and at 5x directly
from vector geometry for inspection. On rows 405--476, the rightmost crossing
of R-G = 0.32 defines the outer coloured contour. The quadratic residual measures
local contour fluctuation, not general image quality:

| Measurement | Before | After |
| --- | ---: | ---: |
| Contour RMS about a quadratic (px) | 0.569 | 0.332 |
| Maximum quadratic residual (px) | 1.253 | 0.956 |
| Mean contour distance from source (px) | 0.544 | 0.372 |
| Whole-image mean RGB error (8-bit units) | 2.333 | 2.303 |
| SVG bytes | 940,809 | 965,998 |

The source raster's own quadratic residual is 0.091 px. Diagnostic conversions
were approximately 22.8 seconds before and 24.8 seconds after; these are single
runs, not a controlled performance benchmark. No shared-loop fallback or
whole-partition curve downgrade occurred in the final car conversion.

`sample/comparison/car-lines.png` shows source / before / after. The larger
staircase and seam-end protrusion are reduced, but coloured ink still has
uneven width and some subpixel steps at short Paint junctions. The retained
thin Paint fragments do not yet share one fitted centreline and width over the
entire contour. These measurements are evidence of improvement, not a claim
that every protrusion or line-width defect has been eliminated.

Validation scope for the car measurements: the artifacts above use the earlier
boundary-stroke recovery, and the 170-test result applies to the `segment.rs` /
`geometry.rs` changes against HEAD. Subsequent combined-workspace validation,
including the stricter stroke acceptance, is reported below.

## Complete-band acceptance and contour connections

The supplied `boy_and_turtle.svg` replaced short parts of a connected black
outline with independent butt-capped strokes. The width test accepted the
middle of a band while discarding local variations at its ends. Removing
those pixels before Paint segmentation left separate stroke/Paint owners at
the arm and shirt-hem junctions. A short retained overlap did not preserve
their connections through subsequent contour fitting.

Whole-band conversion remains automatic, with stricter acceptance:

* Every measured width must be within 0.5 working pixels of the median, and
  every ink colour within 0.08 RGB RMS distance. The former percentile rules
  could hide short tapers, bulges and colour changes.
* Before applying any Paint updates, an eight-connected traversal follows
  source pixels matching the measured ink. Ink continuing beyond the proposed
  footprint and a fixed three-pixel cap/antialias collar rejects the candidate.
  The collar accommodates the retained 1.5-pixel Paint overlap and detector
  spacing; it does not expand with the connected outline network.
* Accepted centre-lines use a 0.35-pixel fitting tolerance and 0.25 smoothing
  sigma instead of 0.75 and 0.45. Rejected regions retain both shared Paint
  contours, including their existing segment-level line and arc fitting.

No CLI or configuration switch is required. Complete isolated bands and
closed circles remain eligible. An incomplete detector observation of an
otherwise uniform band can conservatively remain in Paint.

The regenerated 800 × 744 sample has zero recovered whole-band strokes,
compared with 17 in the supplied working-tree SVG. Its Paint paths still
contain 28 line commands and 126 circular-arc commands. The file changes
from 148,860 to 135,569 bytes. The arm, cuff and hem connections were inspected
in [source / before / after crops](../sample/comparison/boy_and_turtle-lines.png).

In the source rectangle x=270..479, y=285..424, 3,983 pixels have a 3 × 3
neighbourhood with every RGB channel below 32/255. Of these dark core pixels,
10 had an output channel at least 100/255 before the change; none do afterwards.
This local native-resolution check measures visible gaps in the outline core,
not pixel-exact antialiasing or a whole-image quality score. Both SVGs were
rendered over white with `rsvg-convert`.

Regression tests cover partial intervals, branches, short width outliers,
isolated straight and diagonal bands, automatic dark and bright circles, and
authored gaps. A rendered pipeline test checks a filled outline with straight
and circular sections, a tapered branch and an optional gap. Validation passes
173 Rust tests (one existing manual sample test ignored), Clippy with warnings
denied, formatting and diff checks.
