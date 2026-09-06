# Partial arcs on a disc attached to another contour

The blue `</>` disc in row 3, column 1 of `cliparts-6x6` touches the man's
suit. Its dark outer rim is therefore part of a material contour which also
contains the garment. Failing to fit that entire contour as an ellipse should
not prevent fitting its supported circular intervals.

The existing partial primitive pass had three limitations:

* It fitted samples of the already fitted free curves, so their local wobble
  could reject an arc supported by the original raster boundary.
* Its lookahead tried only selected piece counts (64, 32, 16, 8, 4, 2, 1),
  which could miss the transition between a rim and an attached edge.
* Source and corner checks happened after assembling all replacements. One
  unsupported replacement could discard the other useful intervals too.

The pass now associates existing curve joins with ordered source vertices,
tries every join within the same 64-piece lookahead, and fits each candidate
to the corresponding source interval first. It retains the old estimator as
a fallback for pixel staircases whose existing line already fits the raster
corridor. Both paths must pass local source, baseline and corner checks before
an interval is selected. Whole-master validation and ordered shared-graph
mapping still run afterwards.

Every existing join stays fixed. The fitting threshold for a source interval
is at most 0.85 working pixels, matching the existing whole-interval estimator;
the fallback retains its 0.5-pixel threshold. Source/baseline displacement
bounds, protected corners and endpoint tangent constraints remain in force.
There is no assumption about the icon's location, colour or semantics in the
implementation.

## Regression coverage

`src/test-data/man-disc-contour.json` records the source observations and
baseline continuity master from the native-resolution 836 × 836 source cell.
It includes both the garment and disc, rather than a pre-isolated circle.
The regression checks increased SVG arc output on the rim, fewer curve pieces,
unchanged outer endpoints, exact continuity between pieces, bounded deviation,
protected garment corners and successful ordered graph mapping. It fails with
the previous implementation. Existing diagonal-line, retracing, tangent and
shared-source-drift tests also exercise the changed pass. The recorded rim
must contain a consistent circular interval spanning more than 90 degrees,
not just additional short curved fragments.

On the native-resolution source cell, the adopted connected master changes
from 57 curve pieces to 24. The outer rim now includes a roughly 97-degree
interval with a common radius. The complete cell's serialized arc count
changes from 807 to 938, with zero shared-loop fallbacks or curve downgrades
in either version. Its SVG size changes from 989,782 to 987,910 bytes.

The diagnostic suite passes with 204 tests, zero failures and three ignored
tests. Clippy passes with warnings denied for all diagnostic targets and the
default library. Commands:

```sh
mise exec -- cargo test --locked --offline --features diagnostics
mise exec -- cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo clippy --locked --offline --lib -- -D warnings
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
```

This improves supported portions of a connected outline. It does not force
the whole disc into one circle, nor remove the separate shading regions.

## Whole-sheet result

The regenerated
[SVG](../sample/output/cliparts-6x6.svg) accepts all 36 refinements with no
boundary rejections. It changes from 38,624,539 to 38,512,262 bytes; added
refinement data remains within the existing 32 MiB budget at 31,411,744 bytes.

Mean CIEDE2000 error is measured at source resolution over the original green
background, on source pixels whose key-derived coverage is at least 0.5:

| Crop (source coordinates) | Before | After |
| --- | ---: | ---: |
| Figure: (140, 1680)–(735, 2460) | 3.0553 | 3.0509 |
| Disc: (490, 2207)–(720, 2447) | 2.4592 | 2.4413 |

The visual and colour-error changes are modest; the main improvement is
representing supported rim intervals with consistent analytic arcs. This is
not a uniform reduction of colour error across the sheet. Over all 36 cells,
the average change in cell mean error is +0.0066. The largest increase is the
certificate at row 1, column 5, from 2.4100 to 2.5413 (+0.1313); visual review
preserves the document, border, seal and ribbons. Other increases are at most
0.0247. This tradeoff is recorded rather than claiming every figure improves.
