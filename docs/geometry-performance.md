# Exact contour-search acceleration

Contour fairing previously searched every reference point for every smoothed
sample, at each smoothing scale. Its comparison also evaluated both distances
with double-precision `hypot` on every comparison. The reference samples are
now sorted by x coordinate. Each query walks outwards from its x position and
stops when horizontal distance exceeds the best Euclidean distance. The exact
`hypot` comparison is retained, and equal distances select the earliest original
reference index. This prunes work without changing the fitted model. Sorting
costs O(n log n); query cost depends on the spatial distribution, with O(n)
worst-case behavior for a vertical contour.

Bidirectional corridor checks that only need acceptance now stop at their first
supporting sample. A supporting point from the previous query is rechecked
before searching the spatial buckets. Every source and rendered sample still
has to pass the same distance threshold. Checks that need mean error retain
all nearest-distance measurements, but take the square root only after finding
the minimum squared distance.

No worker counts, resolution limits, smoothing scales, sampling densities,
error thresholds, ellipse checks, or adaptive refinement budgets change.
Tests compare the accelerated searches with exhaustive distance calculations,
including negative coordinates, threshold boundaries, empty input, repeated
x coordinates, and equal-distance tie order.

## Validation (2026-09-06)

Compared commit `eb3f39d` with this change, built with Rust 1.95.0 using
`cargo build --release --locked --offline --features diagnostics -j 2`.
Each input ran twice per binary, sequentially in before/after/after/before
order, with `--threads 4 --verbose` and all quality settings at their defaults.
No builds or test suites ran during measurement. Times are mean wall seconds,
including process startup and output writing; these small samples do not
establish throughput for every input or machine.

| Input | Before | After | Time reduction |
|---|---:|---:|---:|
| `car.png` | 29.72 s | 21.27 s | 28.4% |
| `boy_and_turtle.png` | 4.27 s | 3.37 s | 21.2% |
| `viewport1.jpg` | 16.06 s | 16.00 s | 0.4% (effectively unchanged) |

The car's shared-geometry stage fell from 11.58 s to 3.52 s on average
(69.6%). The photograph is dominated by other work and shows no meaningful
end-to-end gain. The full 5016-pixel clip-art sheet was not benchmarked here.

All four SVG files for each input have identical SHA-256 hashes, so these
examples have no output or rendering quality loss:

- car: `db397b8f9a8e9b71c64f66e1926a74174fab446b05be40b2008302a840e2c9e2`
- boy and turtle: `dadba37e8d83d75924b769342cae3b9595d72e8f25020180b76dd5418294de78`
- viewport1: `fc3b147a4ac4d239f6503bbb2c514c3c2fc17613c73ac3a83bea983524afe180`

Reproduce each invocation with the corresponding before/after executable:

```bash
/path/to/picvec sample/input/car.png /tmp/car.svg --threads 4 --verbose
```

Validation: 185 Rust tests pass, one existing full-sheet test is ignored;
Clippy with warnings denied, `cargo fmt --all -- --check`, and
`git diff --check` pass. Tests and Clippy enable `--features diagnostics`.
