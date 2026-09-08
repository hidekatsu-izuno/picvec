# Shaded outline junctions

The older man's shoulder in row 3, column 5 of `cliparts-6x6.png` acquired
small dark protrusions into its pale highlight. They were already present in
the Paint layer, before structural ink was overlaid. Adjacent dark shades had
different continuity classes: their CIEDE2000 separation exceeded 6.9 even
though their lightness was similar and their shared boundary with the highlight
was much stronger. Independently fitted face contours then left short offsets
at shading junctions.

Continuity classification now also recognizes a dominant shared contrast edge.
Two adjacent shades (or durable neighbours of the same coreless fragment) may
share a geometry class when their lightness difference is at most the existing
6.9 threshold and a common neighbour has at least four times their mutual
color contrast against both shades. The common-neighbour requirement is local;
there is no global dark-color bucket. Original Paint labels, colors, topology,
corner checks and curve-displacement budgets remain intact. This change does
not merge Paint regions or insert image/coordinate/color-specific branches.

Tests cover label-order independence, missing common neighbours, and a genuine
lightness boundary. A final-SVG regression uses the full native icon crop
`(3409,1676,627,802)`, including source chroma removal. At 4x rendering, the
shoulder highlight edge's mean absolute second difference changes from 0.146
to 0.087 native pixels; the maximum changes from 1.430 to 0.408. These measures
check visible discontinuities after Paint and ink serialization, rather than
merely counting successful fits.

The regenerated full sheet was checked independently of the native crop. Its
mean edge second difference falls from 0.17036 to 0.09168 working pixels, and
the maximum from 1.18816 to 0.57824. The three-panel comparison is saved as
`sample/comparison/cliparts-6x6-shoulder.png`. Final-sheet green/yellow button
checks also pass. All seven sample SVGs were regenerated; all 277 active tests
pass, with four existing ignored tests. Clippy and formatting checks pass.
