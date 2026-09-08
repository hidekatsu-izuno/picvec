# Round button outline models

The third (green) button in row 2, column 4 of `cliparts-6x6.png` already had
an elliptical outer contour. Its fragmented inner Paint boundary did not fit
an ellipse. Band reconstruction then failed the final source-color gate, so
that jagged inner boundary remained visible while the red/yellow buttons
received reconstructed bands.

The model now preserves the accepted outer ellipse and prefers an ellipse for
the inset edge as well. Width observations around the rim fit a small inner
centre displacement (`width = base + normal dot displacement`), bounded by
1.5 working pixels and 45% of the initial width. This accommodates the original
uneven shaded rim without distorting its round boundaries. The fill repair
edge follows the same displacement; hidden-Paint/stroke removal uses the
minimum guaranteed clearance, and color checking covers the maximum affected
rim. Non-ellipse contours retain the previous centred offset model.

For ellipses, color samples use a three-sample median instead of seven, to
retain localized bevel highlights. The final acceptance limits are unchanged:
mean source ΔE00 increase at most 0.75, rim mean increase at most 3.0.

In the native `(2551,899,718,691)` document crop, the green band's mean ΔE00
was 4.837 before reconstruction. The old candidate was 6.008 and was rejected;
the revised candidate passes the unchanged gate. At 4x rendering, the top inner edge's
mean absolute second difference falls from 0.287 to 0.071 native pixels. These
are geometric smoothness measurements, not claims of identical pixel colors.
Tests cover source-derived offset bounds and the actual final rendered green
button, including both mean roughness and individual discontinuities.

Near-boundary original Paint is retained as underpaint beneath the new opaque
band. A source owner can extend slightly beyond the fitted outer ellipse;
discarding that complete owner left a small transparent triangle on the blue
backdrop. The final-render regression also checks this exterior coverage.

The regenerated full sheet was verified separately from the native crop.
`sample/comparison/cliparts-6x6-buttons.png` shows all three controls and
`sample/comparison/cliparts-6x6-green-button-detail.png` shows the green button
at 6x. Its final mean roughness is 0.07056 pixels and its maximum measured
second difference is 0.16328 (previously 1.31063). The blue exterior sample
remains painted; the prior pie-highlight regression also remains fixed.
All seven sample SVGs were regenerated, and all 272 active tests pass.

## Consistent closed-model selection

The yellow control in the adjacent window (row 2, column 5) exposed a different
problem: whole ellipses used the open-edge localization floor (1.664 pixels),
while whole cubic loops used 2.5 pixels. Its outer loop had approximately 2.36
pixels maximum radial error and 0.81 pixels RMS error, so the less constrained
cubic model won solely because it had a larger error budget.

Whole ellipses now use the same 2.5-pixel floor as whole cubic loops. Existing
bounded scale allowance, winding, persistent-corner, RMS and bidirectional
boundary checks remain. Residual ink alignment uses the accepted closed-model
budget as well. This is a model-selection rule for all closed contours; no
image names, positions, colors or button-specific predicates enter production
code. Tests cover translated/reversed source contours and retain rejection of
squares, dents, open and retraced contours under the new budget.

The expanded ellipse selection also affects the Wi-Fi dot. Its reconstructed
rim needs color intervals aligned with observed color transitions rather than
fixed loop fractions. Splits now use the largest normalized neighboring color
change within the middle half of an interval, with the same depth and eight-
interval budgets. Sparse intervals may use a constant field if it satisfies
the existing color-error gate; sufficiently sampled intervals retain gradient
fitting. The final source-render gates are unchanged. The original Wi-Fi
regression remains in force; it is not weakened to accommodate a new model.
The yellow control's outer contour is elliptical, while its interior shading
retains its existing representation when band reconstruction cannot meet the
quality/complexity budget.

A geometrically accepted ellipse retains a bounded cubic band alternative.
If its ellipse band cannot satisfy color fitting within the patch budget, the
alternative is considered under the same source-render gate. An accepted
primary band excludes overlapping alternatives. This preserves the Wi-Fi
dot's continuous dark rim; its native mean source ΔE00 improves from 4.608 to
3.913 after reconstruction. Regression samples explicitly check that bright
fill does not break through the rim, in addition to the existing wave-edge
smoothness assertions.

All seven sample SVGs were regenerated after this follow-up. The final sheet
passes the yellow-control exterior assertions and the existing green-control
checks (mean second difference 0.07043, maximum 0.16513 working pixels).
`sample/comparison/cliparts-6x6-yellow-window-button.png` and
`sample/comparison/cliparts-6x6-window-buttons.png` compare the original source,
previous SVG and final SVG. All 275 active tests pass; four remain ignored.
