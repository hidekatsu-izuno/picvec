# Car side-window stroke regression

The light gap at the lower right edge of the side window came from a
structural stroke being straightened away from its source samples.

The least-squares line ran from approximately (795.672, 499.644) to
(900.088, 484.212), with a 95th-percentile perpendicular error of 0.405 px.
It passed the 0.707 px straight-line tolerance. The writer then restored
the shared endpoint to (899.708, 481.639), a displacement of 2.602 px, but
did not recheck the line. The emitted line's 95th-percentile error was
2.498 px. Its slope pulled the stroke away from the Paint boundary,
leaving a light sliver between them. Removing structural ink isolated
this discrepancy.

`straight_graph_line` now restores shared endpoints before measuring the
candidate's error. Only the geometry actually emitted can pass the
straight-line gate. If it fails, the existing curve fitter follows the
stroke samples while preserving the shared graph endpoints.

The regression test
`straight_stroke_is_validated_after_restoring_shared_endpoints` covers
both endpoint directions and verifies that an actually straight stroke
still emits one line with both junctions preserved.

The regenerated car no longer has the light sliver. The earlier headlight
and windshield-rim fixes remain visible in their corrected form. Library
tests pass with 235 passed and 4 ignored tests.
