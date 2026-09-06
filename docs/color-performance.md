# Exact colour-processing acceleration

This follows the contour-search optimization in
[geometry performance](geometry-performance.md). Worker counts, colour
thresholds, sample budgets, candidate ordering, and refinement settings are
unchanged.

## Changes

A scalar CIEDE2000 call previously dispatched a one-element batch, allocating
five preparation vectors, two angle vectors, and a result vector. It now
keeps those values on the stack and computes both hue angles in one portable
SIMD vector. It uses the same `wide` angle kernel, float32 constants, chroma
correction, hue wrapping, and final distance formula as the batch path. It
helps small-component merging and the other existing scalar colour gates.

Palette lookup already rejects candidates using the exact CIEDE2000 lightness
lower bound `abs(delta-L) / S_L`. Since
`S_L <= 1 + 0.015 * abs(L_bar - 50)`, a cheap preliminary test can reject
clearly distant lightness values before evaluating the square root in the
tighter bound. It uses an extra rounding margin; surviving candidates still
pass through the original bound and full CIEDE2000 evaluation. Representative
updates and first-minimum tie ordering are unchanged.

Within each merge-model fit, all solid, linear, radial, and additional-stop
candidates reuse one conversion of the reference samples to preprocess-Lab.
Only predicted colours are converted again. This cache is local to the fit,
so later region/source changes cannot make it stale. It deliberately does not
substitute the different Lab transform used by the final Paint fitter.

## Correctness checks

- Scalar colour distances match batch results bit for bit across a sampled
  RGB gamut, neutral colours, signed zeros, identical colours, and multiple
  pairings.
- Palette minimum selection matches exhaustive batch distances, including
  tight radii, out-of-gamut lightness, infinity, and adjacent float thresholds.
- Cached merge observations preserve both mean and 90th-percentile error
  exactly for empty, singleton, and larger sample sets.

## End-to-end measurements (2026-09-06)

Both binaries include the preceding geometry optimization and the same frozen
snapshot of the concurrent faint-seam quality changes. Only the scalar colour
path, preliminary lightness bound, and merge-observation cache differ. Thus
these are additional gains, not a comparison against the original repository
commit. In particular, earlier measurements used different segmentation code.

Release builds use Rust 1.95.0, `--locked --offline --features diagnostics`,
and the repository's release profile. Each input runs sequentially with
`--threads 4 --verbose` and otherwise default settings. Execution order is
before/after/after/before. Times below are mean wall seconds, including startup
and SVG writing; builds and tests for this task finished before measurement.
Only two runs per version were measured, so small gains are indicative rather
than a statistical guarantee.

| Input | Before this change | After | Time reduction |
|---|---:|---:|---:|
| `viewport1.jpg` | 15.34 s | 13.56 s | 11.6% |
| `car.png` | 21.21 s | 20.22 s | 4.7% |
| `boy_and_turtle.png` | 3.49 s | 3.29 s | 5.7% |

For the photograph, palette construction falls from 3.02 to 2.63 seconds,
small-component merging from 2.25 to 1.16 seconds, and Paint-aware merging
from 2.12 to 1.56 seconds. Other stages and timing variation limit the net gain.
Large adaptive-refinement sheets were not measured in this comparison.

Every SVG is byte-identical across the four runs of its input. SHA-256:

- viewport1: `f54aba6836b504b69210fa02469db90df4cd4a64c6dae146cd6f5579e4a278d8`
- car: `5b2de86b2a49901aef093dbb2187482c2643b7feeb425cb2be52425cc61dc3fe`
- boy and turtle: `dadba37e8d83d75924b769342cae3b9595d72e8f25020180b76dd5418294de78`

For each executable, use the same command and compare output bytes:

```bash
/path/to/picvec sample/input/viewport1.jpg /tmp/viewport1.svg --threads 4 --verbose
```

Final shared-workspace validation passes 190 tests, with one existing
full-sheet test ignored, plus Clippy with warnings denied, formatting, and
diff checks. The shared tonal-detail changes advanced after the benchmark
snapshot; their updated state was covered by the final test run. The timing
table remains a comparison of the frozen snapshot, not a measurement of every
subsequent quality change.
