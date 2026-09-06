# Connected sphere and line refinement

The connected blue/yellow spheres near the lower-right of `cliparts-6x6`
retained the coarse 1414 × 1414 sheet model even though the source is
5016 × 5016. Two independent issues contributed to the irregular result.

## Source-resolution acceptance

The figure was detected and evaluated correctly. Its refinement core is
`(4278, 3355, 554, 650)` in source pixels. Before this change, the measured
mean DeltaE00 improved from 4.7147 to 1.9052, the p90 improved from 14.7996
to 5.6605, and the rendered replacement boundary passed. Nevertheless, the
candidate was discarded for complexity.

The combined perceptual gain was 5.8084. The encoded child cost 1,088,437
bytes, or 3.0226 bytes per core pixel. The acceptance rule multiplied that
measured cost by the square root of the coarse partition estimate (4.3101),
charging about 6.2751 against the 5.8084 gain. Fragmented shading therefore
paid twice: once through actual SVG bytes and again through predicted
partition complexity. The byte budget was not exhausted: the original
accepted refinements used 27,196,258 of 33,554,432 available bytes.

The partition estimate now applies only before fitting, when encoded size is
unknown. Final acceptance uses `combined_gain * core_area / encoded_bytes`,
the same rate already used to order candidates against the global budget.
`adaptive_complexity_penalty` retains its documented units of DeltaE00 per
added SVG byte/source-pixel. Boundary checks, minimum gain, tail/edge checks,
and the byte budget still apply. This is not a special case for this image,
its colours, or circles; expensive detail must pay for its measured bytes.

## Shared geometric models

Continuity masters join a visible boundary across quantized Paint labels.
They were preserved by the later colour-pair curve fitter, but were not
themselves passed through the line/circular-arc regularizer. Thus a contour
could bypass primitive fitting precisely because it had been successfully
joined across shading.

The continuity pass now regularizes each complete selected master before
mapping it back to shared graph edges. Original-source and baseline corridors,
endpoint positions, supported corners, tangents, and ordered graph mapping
still constrain the result. An analytic arc is eligible even when its cubic
serialization requires more pieces than a free curve. Diagnostics report
`primitive_regularized` and the actual selected curves rather than the
intermediate fairing candidate.

On an 836 × 836 native crop containing the figure, 348 adopted continuity
masters change; their original 2,337 baseline pieces become 1,324 selected
pieces. SVG size falls from 1,137,448 to 1,103,697 bytes. Shared-loop fallback
remains zero and the existing single curve downgrade is unchanged. These
counts describe the crop, not the full sheet or SVG circle elements.

The source spheres are slightly irregular ellipses with variable outlines
and soft shading. Connected external outlines are not complete ellipses, so
the whole-ellipse fitter cannot reconstruct all seven objects as independent
circles. This change preserves supported geometry and restores source detail;
it does not infer an ideal hidden CAD model or eliminate all shading bands.

## Validation

The complete sheet was converted with the same default settings and four
workers before and after the change. Both SVGs were rendered at 5016 × 5016
over the original green backing. Per-cell colour error uses source foreground
pixels with `1 - (G - max(R, B)) >= 0.5` in normalized RGB.

| Measurement | Before | After |
|---|---:|---:|
| Connected-sphere cell mean DeltaE00 | 7.6644 | 4.0250 |
| Accepted source-resolution regions | 31 | 34 |
| Rejected for measured cost / budget | 3 | 0 |
| Added refinement bytes | 27,196,258 | 30,204,650 |
| Complete SVG bytes | 34,356,219 | 37,281,739 |
| Base shared-loop fallbacks | 0 | 0 |
| Base shared-curve downgrades | 5 | 5 |

The target's foreground mean colour error decreases by 47.5%. All 31
previously accepted rectangles remain, and three more fit inside the unchanged
32 MiB refinement budget. The whole SVG grows by 8.5%. All 36 cells were
compared; the largest mean-error increase is 0.1142 DeltaE00 in the lock cell.
The lock was also inspected visually: the geometric simplification makes
small changes to its shading and outline, without losing the shackle or
keyhole. This is a geometric regularizer, not a guarantee that every raster
colour-error score decreases.

[Source / before / after comparison](../sample/comparison/cliparts-6x6-network.png)
shows the actual full-sheet SVGs rendered directly at the display scale over
white. The source panel uses a green-key display matte. The output remains
editable vector geometry and Paint; no raster image is embedded.

Regression coverage checks that measured byte rate replaces partition cost,
still rejects an expensive candidate, and scales consistently with area and
bytes. A shared-graph test covers a diagonal boundary crossing alternating
shades and checks analytic line output without shared-loop fallback or curve
downgrade. Existing ellipse, arc, authored-corner, branch and transparency
tests also run. All 196 regular tests pass (three existing tests are ignored),
as do Clippy with warnings denied for both the default library and all
diagnostic targets, formatting, and diff checks.

```bash
mise exec -- cargo test --locked --offline --features diagnostics
mise exec -- cargo clippy --locked --offline --all-targets --features diagnostics -- -D warnings
mise exec -- cargo build --release --locked --offline --features diagnostics -j 2
target/release/picvec sample/input/cliparts-6x6.png sample/output/cliparts-6x6.svg \
  --threads 4 --remove-chroma-key-background --verbose
```
