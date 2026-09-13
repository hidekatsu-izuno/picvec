# SVG transparency

- Do not generate SVG `<mask>` elements or `mask` attributes.
- Express transparency in the paint itself: `fill-opacity`, gradient `stop-opacity`, and `stroke-opacity` for strokes. Do not replace this with group opacity.
- Do not serialize faces with zero alpha, including fully transparent paint overlays.
- Fit the visible RGB silhouette itself. Do not preserve raster stair steps and hide them with a mask.
- Geometric clipping for cropped adaptive refinements or stroke outlines is allowed; it must not encode opacity.

# Visible geometry

- Do not retain lines or faces that have no effect on the final rendering. Partial overlap between useful faces is allowed.
- Prevent redundant geometry before curve fitting and SVG emission: merge identical serialized paints and keep paint-owned alpha regions out of structural line recovery.
- Verify removal against RGBA, including transparency and enlarged rendering; do not remove a thin or translucent element merely because it is small.
- Compact antialias colour islands should be absorbed into adjacent ownership before fitting, using local source evidence rather than global paint averages. Validate reported micro-object cases from the emitted SVG as well as synthetic tests.
