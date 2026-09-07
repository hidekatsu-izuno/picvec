# Car seam terminating at the wheel arch

The seam from the headlight toward the front wheel ended around
(357.9, 690.0), several pixels before the black wheel arch. The native
source seam continues into the arch. Ridge detection loses its two lighter
incident sides when it reaches the broad dark face, while the graph's
connection passes primarily join stroke endpoints to other strokes.

`extend_graph_to_dark_paint` recovers a short terminal run before fitting
the final structural geometry. It considers only free endpoints of dark
source-supported strokes, checks the source along the outgoing tangent,
and accepts an extension only when it reaches a broad dark region already
represented by Paint. Unsupported gaps and unrelated nearby dark objects
do not supply a connection. Shared nodes and independently modelled bands
retain their existing ownership.

The regression test
`terminal_ridge_reaches_dark_paint_only_with_continuous_source_ink` checks
both a continuous seam entering a dark face and an intentional source gap
before that face. Only the continuous seam is extended.

The regenerated seam reaches the wheel arch. The lower-door line remains
continuous and the side-window gap remains closed. Library tests pass with
237 passed and 4 ignored tests.
