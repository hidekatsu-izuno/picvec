# picvec x4 evaluation

`scripts/picvec_eval` is a post-hoc evaluator for completed picvec output.
It treats the native x4 output from `realesrgan-x4plus-anime` as the
reference and compares it with the SVG rendered directly at the same x4
resolution.

The evaluator is deliberately separate from the Rust vectorizer:

- it runs only after the SVG has been written;
- it never edits the SVG;
- it is not imported by the vectorizer;
- no score or x4 raster is passed to path, region, Paint, or candidate
  selection.

## Requirements

Install `rsvg-convert` for this optional evaluator's SVG rasterization step.
The Rust converter itself embeds `resvg` and does not require or invoke this
executable. Real-ESRGAN can run through either of two independent
evaluation-only backends:

- `realesrgan-ncnn-vulkan` with its `realesrgan-x4plus-anime` model; or
- SVGDeck's PyTorch/Spandrel loader with the official
  `RealESRGAN_x4plus_anime_6B.pth` file.

The scripts do not download model files. The PyTorch entry point declares its
runtime dependencies for `uv`; the model must still be supplied explicitly.

## Run

First create the SVG normally, without any x4 input:

```bash
mise exec -- cargo run --release --locked -- input.png output.svg
```

Then run the independent evaluator:

```bash
mise x -- uv run scripts/evaluate.py \
  input.png output.svg evaluation-x4 \
  --model realesrgan-x4plus-anime \
  --json
```

Use `--realesrgan PATH` when the NCNN executable is not on `PATH`, and
`--model-path DIR` when its models are stored outside the executable's default
model directory. `--tile-size`, `--tta`, and `--timeout` expose the
corresponding reproducibility/runtime settings.

To use the SVGDeck-compatible PyTorch generator directly from the evaluator:

```bash
timeout 600s nice -n 10 mise x -- uv run --with spandrel==0.4.2 scripts/evaluate.py \
  input.png output.svg evaluation-x4 \
  --realesrgan-model /path/to/RealESRGAN_x4plus_anime_6B.pth \
  --tile-size 256 \
  --json
```

The generated reference is content-addressed under
`.cache/picvec/realesrgan`. Its key includes the source pixels, model hash,
device, accelerator identity, Torch/CUDA/cuDNN/Spandrel versions, precision,
tile size, and padding. Caches made by the earlier SVGDeck-compatible v1
namespace are intentionally not reused because those keys did not capture the
complete inference environment.

The x4 PNG can also be generated without running the evaluator:

```bash
timeout 600s nice -n 10 mise x -- uv run scripts/generate_realesrgan_x4.py \
  input.png reference-x4.png \
  --model /path/to/RealESRGAN_x4plus_anime_6B.pth \
  --json
```

Use `--device cpu` or `--device cuda` to select a Torch device,
`--tile-padding` to control overlap, `--fp32` to disable CUDA fp16, and
`--no-cache` when a persistent cache is not wanted.

An existing, exact x4 reference can be evaluated without invoking
Real-ESRGAN:

```bash
mise x -- uv run scripts/evaluate.py \
  input.png output.svg evaluation-x4 \
  --reference-x4 reference.png
```

The first argument may be the original raster. If picvec resized it during
vectorization, the evaluator reads the SVG canvas and creates the corresponding
Lanczos processing raster as `source-processing.png`. This is evaluation-only
and is never read by picvec.

## Outputs

The output directory contains:

- `reference-realesrgan-x4.png`: canonical Real-ESRGAN reference;
- `source-processing.png`: input raster fitted to the SVG evaluation canvas;
- `rendered-svg-x4.png`: SVG rasterized directly at `4W x 4H`;
- `rendered-svg-native.png`: SVG rasterized at the original input dimensions;
- `report.json`: hashes, commands, settings, and metrics;
- `edges-reference.png` and `edges-svg.png`: fixed-scale L* edges;
- `missing-edges.png` and `extra-edges.png`: primary-tolerance failures;
- `overlay.png`: reference edges in magenta and SVG edges in cyan;
- `boundary-distance.png`: reference-edge distance to the SVG edge;
- `delta-e00.png`: colour-error heat map;
- `thin-lines-reference.png`, `thin-lines-svg.png`, and
  `thin-lines-missing.png`: local-contrast masks for narrow dark seams and
  contours, including strokes absent from the SVG.

The primary edge tolerance is 2 x4 pixels, or 0.5 source pixels. The report
also contains recall, precision, and F1 at 1, 2, 4, and 8 x4 pixels, symmetric
and direction-separated boundary distance (mean/p95/p99/max) in source-pixel
units, whole-image fidelity, and DeltaE00 measured inside an 8 x4 pixel
reference-boundary band. Whole-image and boundary-band colour reports include
the fraction of pixels over DeltaE00 5 and 10, which makes broad missing or
invented highlights visible even when the mean error is small.

The thin-line detector defaults to a 9 x4-pixel local window, 0.045 luminance
contrast, and a strict 1 x4-pixel matching tolerance. These can be inspected or
adjusted with `--thin-line-neighborhood`, `--thin-line-contrast`, and
`--thin-line-tolerance` when evaluating a different line scale. Each report
also scores the dark core of those strokes independently (Rec. 709 luminance
threshold 0.20, adjustable with `--dark-core-luma-threshold`), so a nearby
fill edge cannot substitute for a deleted black/dark outline.

Local colour failures are not reduced to the mean alone: the report retains
the worst tile's boundary-band DeltaE00 p90 and the tail values (p90/p99) of
the whole-image DeltaE distribution. These values are explicit terms in the
post-hoc selection report, so a narrow highlight or seam is not hidden by a
better global mean.

Edges inside the outermost 4 x4 pixels (1 source pixel) are excluded from
boundary metrics by default. This prevents a fill clipped exactly at the SVG
canvas from being reported as object geometry. Whole-image colour fidelity is
not masked. Use `--edge-frame-margin` to change or disable this exclusion.

The report additionally contains `detail_fidelity`, `coarse_fidelity`, and
`pixel_fidelity`. Detail fidelity uses source-supported high-pass regions and
the worst tile to expose erased eyes, notes, seams, and highlights. Coarse
fidelity catches a globally warped or displaced object. Pixel fidelity is
computed against the original raster (not the Real-ESRGAN image) after the
SVG x4 render is reduced to the original size; its main DeltaE comparison is
between 5x5 neighbourhood means. This suppresses harmless one-pixel variation
without any explicit boundary exemption. The same values are evaluated per
tile, and the worst tile is strongly weighted, so a displaced note or a wrong
liquid colour cannot disappear into the whole-image average.

The selection report is split into two systems. `raster.native` compares the
original raster with an SVG rendered at the original size, while `raster.x4`
compares the Real-ESRGAN x4 reference with the SVG x4 render. Both systems
apply the same narrow smoothing, colour, highlight, dark-region, and
continuous-edge filters, then compare broad-filtered difference maps. Their
worst-tile values are included and each scale has an independent floor;
`raster.valid` is false if either floor fails.

`vector` reports primitive counts in addition to path/segment counts. Lines,
rectangles, circles, and ellipses are cheaper than quadratic/cubic Bézier
segments. Vector simplicity is only used to rank raster-valid candidates; it
cannot compensate for a missing line, highlight, dark contour, or wrong fill.

To expose small local failures that whole-image quantiles can hide, the report
also includes:

- `extra_edge_components`: connected SVG-edge components farther than the
  configured tolerance from the reference edge, including the largest
  component and its bounding box;
- `missing_edge_components`: the corresponding missing-reference components;
- `local_failures`: a sliding-tile scan with the worst boundary p95, worst
  boundary-band DeltaE p90, and worst extra-edge tile.
- `micro_artifacts`: an independent score based on the total extra/missing
  edge area and their connected-component load. This prevents many short
  false lines or gaps from being hidden by whole-image averages.

The defaults are a 256 x4-pixel tile (64 source pixels) with a 128 x4-pixel
stride. They can be changed with `--worst-tile-size` and
`--worst-tile-stride`. `--extra-edge-tolerance` and
`--extra-edge-min-area` control the false-edge component report.

The report also includes legacy diagnostics under `quality` and the new
higher-is-better `selection` result. Complexity is measured from the SVG itself (not from the
rendered raster): one weighted unit per segment, three per path, and two per
gradient definition. It is normalised by the Real-ESRGAN reference edge
pixels, so a candidate made from many tiny overlays is penalised even when its
edge F1 is slightly higher. The score includes separate thin-line and dark-core
preservation terms: narrow dark strokes are detected by local contrast and
scored by distance-tolerant recall/precision, while the dark centre is scored
independently. A deleted seam or contour is therefore penalised directly
instead of being diluted by the surrounding fill. In addition, `line_integrity`
combines recall, dark-core agreement, and a nonlinear penalty for missing line
pixels. This term makes repeated short-line deletion non-compensable by a
better flat-fill score. The default nominal weights
are boundary 25%, colour 20%, local failure 10%, thin-line preservation 30%,
dark-core preservation 35%, line-integrity 85%, edge smoothness 25%, and
complexity efficiency 40% (normalised to sum to one), with additional coarse-structure and
original-raster pixel-fidelity and micro-artifact terms. The micro-artifact
term has a nominal weight of 45% before normalisation. This prevents a jagged overdrawn contour
from winning merely because it contains many dark pixels or a large region
with the wrong colour.
When SVG geometry is supplied, control-point wobble and anchor-chain
roughness are additional independent terms (default weights 25% and 20%,
respectively, before normalisation). Visible open stroke overlays (seams,
highlights, and arches) are measured separately as `svg_open_stroke_roughness`
(default weight 20%), so a smooth filled contour cannot hide a jagged line.
The open-stroke residual is length-normalised and does not reward simply
deleting short lines.
The authoritative result is `selection.valid`: raster floors are gates, and
vector simplicity is only used to rank candidates that pass both raster
scales.
Edge smoothness also reports high-frequency rendered edges that have no nearby
reference edge support; this catches staircase fragments even when the
reference itself contains more fine texture.
The vector report additionally includes `svg_anchor_roughness`, which measures
short-period oscillation of smooth closed-path anchors independently of their
Bezier handles. This catches the case where handles are collinear but the
anchors still form a raster staircase. Corners and short detail paths are
excluded from this diagnostic.
`svg_open_stroke_roughness` applies the same curvature-aware local-chord test
to visible `fill="none"` paths and is independent of element IDs.
Continuous open strokes additionally receive a native-resolution anchor
regularization pass: endpoints and detected corners remain fixed, while
interior anchors may move by at most 0.5 source pixels. Each edit is retained
only after a native render gate, and the x4/Real-ESRGAN image is never used by
the vectorizer.
The fixed clip-art profile also applies a source-only outer-silhouette trim:
the largest foreground component is converted to a smoothed signed-distance
contour at native resolution and fitted by bounded cubic Beziers. A background
even-odd face owns the complete exterior of that same curve, preventing seams
while removing external one-pixel hooks. Real-ESRGAN and the x4 image remain
evaluation-only.
The score is a ranking aid, not a vectorizer objective; use
`--complexity-target-per-edge` only when comparing datasets with a deliberately
different geometry budget.

For an honest algorithm comparison, report both the raw diagnostics and this
selection score. A higher F1 is not an improvement if it comes with a large
increase in `weighted_units_per_reference_edge_pixel`, whole-image DeltaE,
or local failures.
