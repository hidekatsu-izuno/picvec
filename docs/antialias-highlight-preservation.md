# Preserve fragmented highlights during antialias cleanup

The top-left document icon in `cliparts-6x6.png` acquired an orange notch in
its pie chart's pale highlight. The defect existed in the Paint layer before
structural strokes were added. At native sample `(271,568)`, a bright pixel
was reassigned to the orange interior during `correct_antialias_partition`;
it was not a hole in the alpha mask or a Bezier fitting artifact.

The boundary-sleeve classifier could find two nearby durable red/orange
owners around a tiny fragment of the highlight. It already excluded a dark
extremum, but did not exclude a bright extremum. A component whose median Lab
lightness exceeds both proposed parents by more than 12 is now preserved.
The allowance keeps moderate sharpening overshoot eligible for the existing
ringing-sleeve treatment. Source-supported thin highlights can retain their
own Paint even when that costs additional regions.

Regression tests cover both singleton and elongated highlights between darker
faces, and a final rendered SVG from the actual document crop. The fixture is
source rectangle `(157,106,564,710)`. Samples `(114,462)` and `(114,463)` in
that crop previously rendered orange (green channel approximately 0.38–0.40)
and now remain in the pale highlight (approximately 0.99). Existing tests
also retain the intentional handling of moderate bright ringing and dark ink.

Final full-sheet verification uses
`sample/comparison/cliparts-6x6-notch-fix.png` (10x) and
`sample/comparison/cliparts-6x6-pie-fix.png` (2x). The actual embedded SVG at
source coordinate `(271,568)` changed from RGB `(254,97,28)` to
`(223,254,216)`. All seven samples were regenerated after the fix. The final
Wi-Fi middle inner rim retained its previous measured roughness (0.06088
native pixels); the final-render highlight test and the Wi-Fi test both pass.
