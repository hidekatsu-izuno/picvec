# Clip-art detail refinement

The server rack (row 4, column 4), organization chart (row 3, column 5), and
developer portrait (row 3, column 1) in `cliparts-6x6.png` retained the coarse
1414 × 1414 base conversion of the 5016 × 5016 source. They were missing from
the saved SVG's adaptive refinement layer, so enlarging them exposed broken
ventilation holes, glasses, facial features, and highlights.

## Causes and changes

* **Server rack:** the preliminary score divided perceptual error by the full
  partition cost. Its many ventilation holes produced a cost of 4.9675 and a
  priority of 2.3683, below the default 2.5 cutoff. The actual acceptance rule
  only charges the square root of partition cost, together with measured SVG
  bytes. The preliminary score now uses that same square-root term, giving
  this candidate a priority of 5.2785. Final quality and byte-budget checks
  still decide whether to keep its result.
* **Organization chart:** the fixed 16-pixel padding reached the sheet's white
  separator at source row 1676. Source support then rejected the rectangle,
  although the figure itself was separate from the grid.
* **Developer portrait:** its old padded border at source row 1679 cleared the
  source grid, but the coarse rendered grid bled into the boundary check. A
  native crop rendered against the retained base exceeded the 2/255 channel
  tolerance across that top edge. The refined portrait was discarded.

Figure grouping still merges nearby components before fitting. Padding is now
chosen from the available background gap: scan outward up to twice the usual
16-pixel margin and place the boundary halfway to the nearest foreign support,
capped at 16 pixels. This puts the organization chart's top at 1685 and the
developer's at 1686. Both now pass the existing rendered boundary check.
Connected oversized figures still stay in the base; none are cut into tiles.
Diagnostic builds also report each evaluated rectangle's before/after scores,
boundary result, and SVG bytes.

## Measured result

Measured on 2026-09-06 using the command below, with the pre-change release
binary as the baseline and identical conversion flags. Both SVGs were rendered
by `rsvg-convert` at the original 5016 × 5016 size on the original green
background. The table measures mean CIEDE2000 error against original pixels
inside each 836 × 836 cell, restricted to the same source foreground mask
(`1 - (G - max(R, B)) >= 0.5`, normalized RGB). Lower is better.

| Figure | Before | After | Error reduction |
|---|---:|---:|---:|
| Server rack | 6.0613 | 3.2788 | 45.9% |
| Organization chart | 7.6730 | 4.2418 | 44.7% |
| Developer portrait | 6.1148 | 3.4811 | 43.1% |

All 36 cells were compared. Every previously accepted refinement remains
present; seven additional figures now refine. The largest increase in another
cell's foreground mean error was 0.0170 DeltaE00 (the conference monitor,
whose crop padding also changed). The three requested figures were inspected
at source scale, including the ventilation pattern, glasses, and code badge.
Fine shading still contains some solid-colour bands; this change restores
source-resolution fitting rather than changing the Paint model.

| Sheet measurement | Before | After |
|---|---:|---:|
| Proposed regions | 31 | 35 |
| Accepted regions | 27 | 34 |
| Prefiltered for complexity | 2 | 0 |
| Rejected after quality checks | 2 | 1 |
| Rejected for measured cost or byte budget | 0 | 0 |
| Complete SVG bytes | 22,140,216 | 28,557,227 |
| Added refinement SVG bytes | 15,770,453 | 22,109,824 |

The added refinement bytes remain within the existing 24 MiB budget. The
greater detail increases the complete SVG size by 6.4 MB.

[Source / before / after comparison](../sample/comparison/cliparts-6x6-details.png)
uses white for viewing; the source panel uses the same green-key matte.
Metrics above use the original green background, not that display composite.

## Reproduction and validation

```bash
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
rsvg-convert --width 5016 --height 5016 --background-color '#00ff00' \
  --output /tmp/cliparts-6x6-rendered.png sample/output/cliparts-6x6.svg

mise exec -- cargo test --locked --offline --features diagnostics
mise exec -- cargo test --locked --offline --features diagnostics \
  clipart_sheet_refines_whole_figures_with_and_without_keying -- --ignored
mise exec -- cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo fmt --all -- --check
```

All 179 regular Rust tests, the full-size sample regression, and 9 Python
evaluation tests passed; Clippy and formatting checks passed. New regressions
cover a separated figure near an oversized grid, a genuinely connected figure
that must stay with the grid, and dense visible detail reaching measured
evaluation. The sample regression checks that all three keyed figures have
whole-object candidates. Opaque planning retains its conservative support
rule: faint grid debris can still join nearby components in that mode.
