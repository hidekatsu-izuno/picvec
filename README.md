# picvec

Native Rust implementation of the perceptual, structure-aware raster-to-SVG
pipeline. It creates editable SVG Paint faces, Office-compatible solid/linear/
elliptical-radial fills (at most five stops), raster-derived shared topology,
analytic primitives, and source-supported structural centre-lines.

## Build

```bash
mise exec -- cargo build --release --locked
```

Diagnostic CLI options and JSON diagnostic output are available in builds
that enable the optional `diagnostics` feature:

```bash
mise exec -- cargo build --release --locked --features diagnostics
```

The converter uses the portable [`wide`](https://crates.io/crates/wide) SIMD
library across supported CPU architectures. It contains no vendored or
hand-written assembly and requires no assembly-specific build step.

## Run

```bash
picvec input.png output.svg
```

The input processing size is selected automatically from source complexity;
`--max-dimension` sets its upper bound. The only output is the SVG file named
by the second positional argument.

Useful controls:

```text
--max-dimension <PX>
--max-input-dimension <PX>
--max-input-megapixels <MP>
--max-decode-mib <MIB>
--remove-chroma-key-background
--smoothing-radius <PX>
--segmentation-min-size <AREA>
--quantization-dark-delta-e <DE>
--quantization-light-delta-e <DE>
--gradient-merge-error <DE>
--solid-color-max-delta-e <DE>
--no-adaptive-refinement
--adaptive-tile-dimension <PX>
--adaptive-max-patches <N>
--adaptive-svg-budget-mib <MIB>
--adaptive-min-perceptual-gain <DE>
--adaptive-min-predicted-rate <RATE>
--adaptive-complexity-penalty <RATE>
--threads <N>                 # 0: min(4, detected CPUs / 2), at least one
--quality-metrics             # diagnostics feature: full-SVG DeltaE00/SSIM report
--verbose                     # diagnostics feature: JSON report on stderr
```

`--solid-color-max-delta-e` controls the within-region colour range that can
be accepted immediately as a solid fill. Lower values retain more subtle
shading as gradients; higher values favour simpler SVG output. The default is
1.5.

`--remove-chroma-key-background` detects a near-saturated red, green, blue,
cyan, magenta, or yellow backing colour in a shallow outer band. It removes
every matching region, including enclosed and disconnected regions, and
uses a soft colour-difference matte for antialiased edge pixels. White and
black are deliberately not treated as automatic key colours. Without this
option, opaque input does not use automatic chroma-key removal.

An alpha channel already present in the input is handled automatically and
does not require this option. Exact source alpha is retained as one-byte
coverage samples while the vector mask is built. A narrow run of intermediate
coverage connecting clear and opaque pixels is treated as raster antialiasing,
not as a translucent object: its half-coverage position is interpolated between
pixel centres and fitted as one fully opaque vector contour. The SVG renderer
then antialiases that curve at the display resolution, so neither an enlarged
raster staircase nor a persistent semitransparent outline is embedded in the
SVG. Broad or independent authored transparency is preserved with the uniform
two-bit levels `0`, `1/3`, `2/3`, and `1`, using nested vector regions. Visible
pixels retain their straight source RGB, so a foreground colour cannot be
removed merely because it equals the temporary colour used to normalize fully
transparent pixels. Nonzero samples are also retained as hidden underpaint,
preventing the interpolated mask edge from revealing that normalization colour.
The converter does not flatten transparent input onto white.

Constant-colour matting from a single image is inherently ambiguous when the
foreground itself contains an inferred chroma key. Such areas can be removed
with the backing; choose a key colour absent from the subject for reliable
opaque-key results. Exact source alpha does not have this ambiguity.
See [the background-removal design notes](docs/chroma-key-background-removal.md)
for the matte model, thresholds, and research basis.

Large inputs use source-resolution adaptive refinement by default. The base
SVG is rendered and compared with the original in balanced source-space
regions. A region is rerun through the same vector model at a finer scale only
when its mean/tail DeltaE00 and missing-edge improvement justify its predicted
partition cost and measured added SVG bytes. This rate-distortion rule is
content-independent: compact clip-art features can receive more detail while
expensive photographic texture is normally left at the base level. Accepted
regions are fitted from the original pixels with a halo and clipped back into
the base SVG. The byte budget and thresholds above control the quality/size
tradeoff; `--no-adaptive-refinement` restores single-resolution processing.
Independent refinement regions run concurrently. The job count is bounded by
the selected worker count and by a conservative estimate derived from the
largest crop and currently available memory; `--verbose` reports the selected
count as `adaptive_refinement.parallel_jobs`. Small source-native crops also
skip the otherwise redundant automatic-resolution probe.

Full-resolution source data waiting for adaptive refinement remains packed as
RGB8 when it came directly from the decoder and as Q0.16 RGB only when matte or
chroma processing produced fractional channels. Working crops are expanded to
`f32`, so filtering and perceptual calculations do not use fixed-point
arithmetic.

The default worker count leaves thermal and interactive headroom: it uses half
of the detected logical CPUs, capped at four workers and with a minimum of one.
`--threads N` remains an explicit override for callers that prefer a different
throughput/resource tradeoff.

Input dimensions and total area are checked from the image header before
decoding (32,768 pixels per axis and 32 megapixels by default), and decoder
allocations have a 512 MiB best-effort limit.

`--verbose` prints the in-memory timing and complexity report to stderr for
development; it still writes no sidecar files.

`--quality-metrics` explicitly enables a second, complete in-memory SVG render
and reports DeltaE00 and SSIM. It is disabled by default because those metrics
are observational and never change the generated SVG. With `--verbose` they
are included in the complete report; otherwise only the metric object is
printed to stderr.

The published Rust crate contains only the converter source and its legal
documentation; sample images and the evaluation-only Real-ESRGAN model are
kept outside the crate. See `THIRD_PARTY_NOTICES.md` and `sample/README.md`
before redistributing those repository assets.

## Examples

Each comparison shows the source raster on the left and the corresponding SVG
rendering on the right. The committed comparison assets can be regenerated by
the optional `scripts/generate_sample_comparisons.sh` maintenance script. That
script currently uses librsvg and ImageMagick, but neither is a converter
runtime dependency.

![Boy and turtle raster input and rendered SVG output](sample/comparison/boy_and_turtle.png)

![Car raster input and rendered SVG output](sample/comparison/car.png)

![Clip art raster input and rendered SVG output](sample/comparison/cliparts.png)

![Mountain raster input and rendered SVG output](sample/comparison/viewport1.png)

![Coast raster input and rendered SVG output](sample/comparison/viewport2.png)

## x4 evaluation

`scripts/evaluate.py` evaluates a completed SVG against a Real-ESRGAN x4
reference without feeding that image back into vectorization. A standalone x4
PNG can be created with SVGDeck's migrated PyTorch/Spandrel generator:

```bash
timeout 600s nice -n 10 mise x -- uv run scripts/generate_realesrgan_x4.py \
  input.png reference-x4.png \
  --model /path/to/RealESRGAN_x4plus_anime_6B.pth
```

See `scripts/picvec_eval/README.md` for NCNN and PyTorch evaluator usage,
content-addressed caching, and reproducibility controls. The evaluator's
separate SVG rasterization step currently uses `rsvg-convert`.

## Pipeline

1. Check the input size, choose a suitable base resolution, and resize the
   raster when needed.
2. Detect colour regions, boundaries, shading, and thin structural lines.
   Correct antialias pixels and merge only neighbouring regions that can share
   one fill without losing a visible boundary.
3. Fit each region with a solid colour or a linear/radial gradient. Neighbouring
   gradients are adjusted when doing so produces a smoother result.
4. Convert region boundaries into shared vector curves. Adjacent regions reuse
   the same curve, and simple regions become rectangles, circles, or ellipses
   when it is safe to do so.
5. Render the Paint layer once with embedded `resvg`, then add structural lines
   that are still missing from that preview.
6. For a downscaled input, compare the base render with source-resolution
   regions. Refit only regions that improve the common perceptual-error/SVG-rate
   objective, using original pixels and a clipped overlap halo.
7. Add a small overlap between Paint regions to hide renderer seams and write
   the final editable SVG atomically to the requested path.
