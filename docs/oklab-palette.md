# OKLab palette experiment

This is the historical palette-only experiment. The current implementation
uses OKLab throughout; see [the migration report](oklab.md). CIELAB and
CIEDE2000 are no longer runtime options. The DeltaE00 numbers below belong to
the historical experiment and are not comparable to current DeltaEOK numbers.

The experiment used a separate `Oklab` type and converted CIELAB directly
through D65 XYZ using [Ottosson's published matrices](https://bottosson.github.io/posts/oklab/).
It did not clip histogram colours through sRGB. Each representative's OKLab
coordinates were cached and refreshed after the CIELAB weighted representative
moved. Equal-distance candidates retained their original palette order.

The palette distance is `100 * Euclidean(OKLab)`. The acceptance radius is
`adaptive_tolerance(mean CIE L*) * oklab_palette_threshold_scale`; protected
ridge colours use `min(adaptive_tolerance, 1.5) * scale`. The multiplier is an
empirical palette setting, not a conversion between perceptual units.

The integer CIELAB histogram, representative averaging, CIE L* tonal response,
smoothing, subsequent region merging, Paint fitting and final DeltaE00 quality
metric remained unchanged. This isolated palette selection; it did not test a
complete OKLab pipeline, OKLab interpolation, or replacement of all DeltaE00
gates. A palette with fewer entries can still create more final regions because
spatial and Paint decisions follow palette assignment.

## Historical measurements and current palette sweep

Build once, then run the benchmark without concurrent builds or tests:

```bash
cargo build --release --locked --offline --features diagnostics
python3 scripts/compare_oklab.py /tmp/picvec-oklab-sweep --scales 0.5 0.75 1 1.25 1.5
python3 scripts/compare_oklab.py /tmp/picvec-oklab-repeat --scales 1 --repeats 2
```

The current script sweeps only OKLab settings; it does not recreate the removed
CIELAB baseline. The script keeps SVGs, complete diagnostic logs, input/binary/output hashes,
commands, quality metrics and wall times. Odd repeats reverse variant order.
It uses four workers and otherwise normal CLI defaults. Quality is measured by
the embedded resvg renderer against the processing-resolution source. Whole-image
SSIM is a single-window luminance metric, not local/structural validation.

## Scope of measurements

The samples are a flat illustration (`boy_and_turtle.png`, 800 × 744), a shaded
car (`car.png`, 1254 × 1254), and a mountain photograph (`viewport1.jpg`, 640 × 426). Their
processing sizes are the same between variants. Adaptive refinement is enabled,
but these samples produce no accepted refinement regions.

Threshold-sweep wall times overlapped compilation and the existing debug test
suite and must not be used to claim a speedup. Separate repeated measurements
are used for the timing comparison. These three samples are tuning data, not a
held-out validation set; the selected scale is provisional.

## Threshold sweep

Lower DeltaE00 is better. Results are from the same release binary for all
sweep variants; changing the experimental default afterwards does not change
these explicitly selected configurations. Full numerical results, SSIM, p90,
p99 and SVG hashes are in [the result data](oklab-palette-results.json).

| Input | Metric / scale | Mean DeltaE00 | SVG bytes | Palette colours | Final regions |
| --- | --- | ---: | ---: | ---: | ---: |
| boy_and_turtle.png | cielab | 0.629538 | 130,150 | 417 | 93 |
| boy_and_turtle.png | oklab-0.5 | 0.631104 | 131,215 | 866 | 93 |
| boy_and_turtle.png | oklab-0.75 | 0.628834 | 132,365 | 500 | 94 |
| boy_and_turtle.png | oklab-1.0 | 0.629339 | 130,343 | 348 | 92 |
| boy_and_turtle.png | oklab-1.25 | 0.627742 | 128,277 | 264 | 88 |
| boy_and_turtle.png | oklab-1.5 | 0.630796 | 129,700 | 209 | 90 |
| car.png | cielab | 0.829673 | 1,982,104 | 1,765 | 1,887 |
| car.png | oklab-0.5 | 0.851936 | 2,204,475 | 2,197 | 2,019 |
| car.png | oklab-0.75 | 0.832189 | 2,052,981 | 966 | 1,942 |
| car.png | oklab-1.0 | 0.823004 | 1,841,977 | 555 | 1,719 |
| car.png | oklab-1.25 | 0.834571 | 1,757,110 | 351 | 1,594 |
| car.png | oklab-1.5 | 0.835117 | 1,692,229 | 251 | 1,548 |
| viewport1.jpg | cielab | 6.322011 | 8,593,326 | 8,163 | 19,697 |
| viewport1.jpg | oklab-0.5 | 6.378575 | 8,268,998 | 9,264 | 18,858 |
| viewport1.jpg | oklab-0.75 | 6.375249 | 7,950,681 | 4,059 | 18,378 |
| viewport1.jpg | oklab-1.0 | 6.353047 | 7,943,273 | 2,258 | 19,041 |
| viewport1.jpg | oklab-1.25 | 6.360529 | 7,418,296 | 1,385 | 17,934 |
| viewport1.jpg | oklab-1.5 | 6.358884 | 7,429,483 | 934 | 18,426 |

The selected scale is **1.0**. It has the best mean DeltaE00 of the
five tested OKLab settings on the car and photograph; scale 1.25 is best on the
flat illustration. Relative to CIELAB, scale 1.0 changes mean DeltaE00 by
−0.03%, −0.80%, and +0.49% on the illustration, car and photograph respectively.
SVG size changes by +0.15%, −7.07%, and −7.56%. The photograph's SSIM improves
slightly, but its DeltaE00 mean, p90 and p99 worsen. No tested scale improves
mean DeltaE00 on all three inputs.

At scale 1.0 the unmodulated dark/light radii are 2.5/5.0 and the protected cap
is 1.5, all in **100 × OKLab distance**. The dark/light radii also use the
existing CIE L* tonal response before the protected cap is applied. Their
numerical equality to the old settings is the result
of the sweep, not a claim of equivalent units.

Rendered comparisons show almost identical flat artwork. The car changes its
internal shading partitions, without consistently eliminating thin seams.
The mountain photograph changes sky banding and rock partitions; it does not
show a clear overall visual improvement. Visual checks used librsvg previews;
numerical metrics used the pipeline's embedded resvg renderer.

OKLab was adopted for its size/speed tradeoff. These data do not justify
calling it a general quality upgrade. They also do not establish what full OKLab averaging, histogramming or
smoothing would do.

## Validation before adoption

`cargo test --release --locked --offline --features diagnostics` passes:
291 tests passed, 4 existing full-image regressions ignored. The debug suite
was interrupted because the larger image regressions were slow; the full
non-ignored suite was then run successfully in release mode. Three new tests
cover published XYZ/OKLab reference vectors and RGB primaries, threshold and
protected-ridge behavior, and configuration validation/serialization defaults.

`cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings`,
`cargo check --locked --offline`, formatting and diff checks also pass.

## Repeated timing comparison

Final diagnostics release binary, four workers, with no concurrent builds or
tests from this task. Each input runs CIELAB / OKLab / OKLab / CIELAB, with
OKLab scale 1.0. Times include SVG writing and report-only rendering/metrics.
Two runs per variant provide indicative timings, not a statistical guarantee.
Every repeated SVG is byte-identical to its corresponding threshold-sweep SVG.

| Input | CIELAB mean seconds | OKLab mean seconds | Time reduction | CIELAB palette seconds | OKLab palette seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| boy_and_turtle.png | 4.418 | 4.336 | 1.9% | 0.024 | 0.018 |
| car.png | 33.293 | 31.289 | 6.0% | 0.204 | 0.058 |
| viewport1.jpg | 19.883 | 16.296 | 18.0% | 2.831 | 0.133 |

The full-pipeline timing difference includes changes to palette size and
downstream topology/painting, so it is not a pure colour-distance microbenchmark.
The flat illustration's small timing difference should be treated as noise.
