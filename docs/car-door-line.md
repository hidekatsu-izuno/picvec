# Car lower-door line regression

The lower door seam was detected as one long structural ridge. It became
dotted in the final `refine_interrupted_strokes` pass, which tries to
recover genuine source dots and gaps from a stroke's cross-section widths.

`stroke_model::refine_interrupted` only measured ink mass when the local
background-to-ink luminance contrast exceeded 0.15. Otherwise it left the
mass at zero and included that value in its gap statistics. The dark red
shading along the continuous door seam repeatedly crossed this threshold.
Roughly half of its 656 cross-sections were unmeasurable, and those unknown
widths were treated as missing ink. The pass replaced the continuous
stroke with many short, disconnected `sampled-ink` segments.

The pass now retains the existing stroke if a profile lacks sufficient
contrast. A missing measurement cannot justify deleting source ink. The
existing width-distribution checks still recover genuine dotted lines
when all profiles provide measurable contrast.

`insufficient_profile_contrast_does_not_turn_a_continuous_line_into_dots`
reproduces the failure with a continuous black line on a shaded background
that crosses the contrast threshold. It fails before the fix. The existing
dotted-line and invisible-black tests cover the genuine-gap behavior.

The regenerated car has a continuous lower-door stroke, and the side-window
gap remains fixed. The library suite passes with 236 passed and 4 ignored
tests.
