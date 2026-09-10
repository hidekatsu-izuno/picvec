# OKLab migration

All perceptual colour processing now uses OKLab: raster conversion, smoothing,
palette histogramming and representative averaging, segmentation and merging,
boundary/outline checks, gradient fitting and final quality evaluation. The
CIELAB transforms and CIE76/CIE94/CIEDE2000 distances have been removed. There is
no colour-space switch or CIELAB runtime fallback.

RGB remains the input/output and compositing representation. SVG gradients
still use the renderer's RGB interpolation; predictions are converted to OKLab
for fitting and validation. Luminance/SSIM and geometric checks remain separate
metrics rather than alternate perceptual colour spaces.

## Units and API

`color::Oklab` stores **100 times the standard OKLab coordinates** on all three
axes. White has L=100. `delta_e_ok` is the Euclidean distance between these
scaled coordinates, so a reported distance of 1 means standard DeltaEOK 0.01.
Forward and inverse sRGB transforms use
[Ottosson's 2021 matrices](https://bottosson.github.io/posts/oklab/).
The inverse clips at the final sRGB output, not in the intermediate colour space.

Native diagnostics and the Python evaluator use `delta_e_ok_mean`,
`delta_e_ok_p90`, and `delta_e_ok_p99`. These replace the old `delta_e00_*` keys;
the numbers are **not interchangeable**. The Python evaluator's derived colour
scores and coverage thresholds now also use the new units.

Public conversion entry points are `rgb_to_oklab`, `oklab_to_rgb`,
`oklab_values_to_rgb`, `edge::oklab_values`, and `edge::oklab_pixels`.
There are scalar, paired, and common-reference `delta_e_ok` helpers.
`Config::dark_knee_lstar` is replaced by `dark_knee_lightness`.
The former `--oklab-palette` opt-in flag has been removed.

## Thresholds

- Dark/midtone transition: OKLab L=52.6 (near the neutral brightness of the old
  transition). Dark-outline and chroma classification thresholds use OKLab units.
- Smoothing dark/light tolerances: 1.5 / 4.5, with a near-black noise floor
  of `1.5 * 240 / (L*L + 10)` in scaled OKLab units.
- Palette dark/light tolerances: 2.5 / 5.0, multiplier 1.0, with the existing
  smooth tonal-detail response (white endpoint 0.45; protected-detail cap 1.8).
  `--oklab-palette-threshold-scale` adjusts these.
- Palette histogram cells: 0.5 on each scaled axis. A cell's representative is
  the actual sample mean, accumulated in float64. Rounding a saturated colour
  to the cell centre would introduce an unwanted tint.
- Gradient merge tolerance: 1.8; unconditional solid-colour range: 1.2.
- Same-material boundary tolerance: 6.2, or 7.5 for a substantial attached
  shoulder. Short independent coloured marks retain their own paint.
- Geometric same-material continuity tolerance: 10.5. This joins shading
  contours for geometry; it does not merge their paint regions.
- Outline ink fitting tolerance: 3.1, or 4.0 for a near-black reference
  (median OKLab L < 20); repair tolerance: 6.2. Ink references use a local
  median along the contour. Up to 16 intervals at depth 5 carry ink and
  adjacent fill. Source-realigned bands repair both sides with 1.25-pixel
  collars and separate inner/outer underpainting (three strips per interval;
  ordinary bands keep their existing two-strip representation).
- Shared-boundary overlap: 0.3 pixels to avoid transparent cracks when the new
  colour partitions meet inside an opaque source area.

These settings are calibrated with the existing source-supported regression
suite. They are not a universal conversion from DeltaE00 to DeltaEOK.

The earlier [palette-only experiment](oklab-palette.md) is retained as a
historical record; its quality and timing numbers describe a different pipeline.

## Regression checks

The Wi-Fi rim regression now locates its reference rim after chroma-key
separation and checks colour and contrast in all seven columns from x=148 to
154, allowing at most one source pixel of boundary movement. It requires
mean rim colour error below 6 (maximum below 12) and at least 75% of the source
rim contrast in each column. The old fixed
`(x, 203)` dark-colour assertion demanded dark ink where this input is cyan
fill (for example the source at `(148, 203)` is RGB 148,253,255). Keeping that
assertion would reward a displacement from the source. The existing subpixel
inner-rim roughness and whole-foreground colour-error checks are retained.

## Historical full-pipeline sample comparison

Both the saved palette-only SVGs and the full-OKLab SVGs were rendered with the
same native resvg renderer at source resolution and composited on white. All
errors below were recomputed with the current 100-scaled OKLab metric. Lower
error is better. Four workers and normal CLI defaults were used; timings are
omitted because compilation/tests overlapped this run.

| Input | Mean error, palette-only → full | SVG bytes, palette-only → full |
| --- | ---: | ---: |
| boy_and_turtle.png | 0.6716 → 0.7949 | 130,343 → 135,914 |
| car.png | 0.6923 → 0.6827 | 1,841,977 → 2,229,139 |
| viewport1.jpg | 4.7723 → 4.6842 | 7,943,273 → 7,135,273 |

The car and photograph have lower mean error, while the illustration has higher
mean error. File size increases for the illustration and car and decreases for
the photograph. The photograph's p99 error rises despite its lower mean; this
is not a uniform quality improvement. See [the result data](oklab-results.json)
for p90/p99 and source/output hashes. These observations do not compare the old
DeltaE00 numbers with the new metric.

## Historical migration validation

- `cargo test --release --locked --offline --features diagnostics`: 284 passed,
  4 existing ignored tests, no failures.
- `cargo check --locked --offline`: passed with default features.
- `cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings`:
  passed.
- `cargo fmt --check` and `git diff --check`: passed.
- `PYTHONPATH=scripts:. python -m unittest discover -s tests -v` in the cached
  evaluator environment: 12 passed.

The four ignored tests remain opt-in full-size sample regressions. The new colour
checks cover published primary coordinates, dark-branch and gamut round trips,
distance units, tiled Python evaluation and saturated-palette sample means.

## Car calibration after the visual regression

The initial migration and subsequent spatial/geometry corrections did not
recover the pre-OKLab car. The historical tables above are not measurements
of the current output. See [the preceding calibration](car-boundaries.md)
for the reproducible CIELAB baseline and removed corrections, and
[the latest window repair](car-windows.md) for current output, measured causes,
local geometry/ink fixes, and validation.

The preceding calibration restored the pre-migration segmentation and geometry
algorithms. The subsequent window repair retains the same paint segmentation
and fixes source-supported contour and stroke reconstruction. In particular, the
edge feature channels use L/100, a/40, b/40, the neutral dark seed accepts
8-bit greys through 13 (scaled OKLab distance 16.35), and dark-region/shoulder
lightness gates use 35.3 and 56.9. These are calibrated decision thresholds,
not claims that CIEDE2000 and OKLab distances are globally proportional.

After the window repair, all nine images in `sample/input` were regenerated
with the current defaults, retaining chroma-key removal for `cliparts-6x6`.
All nine comparison images were also updated. Current input/output hashes,
source provenance, and per-image refinement results are recorded in
[the sample regeneration results](sample-regeneration-results.json).
