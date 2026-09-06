# Merging unnecessary regions in the production converter

The production converter could stop after merging disjoint pairs, leaving a
continuous colour ramp split across several final SVG faces. Its merge-template
fit also mapped `LinearPreset::Fitted` back to a horizontal preset and recomputed
`RadialOrigin::Fitted` from the bounding box, discarding already fitted direction,
centre, and eccentricity. These changes apply to the ordinary `picvec` command.

## Changes

- Preserve fitted linear direction when extending a template across a union.
  Preserve a fitted radial centre and radius ratio, extending only the overall
  radius as needed to cover the union samples.
- In the optional multi-pass mode, try a solid, either existing paint, and inexpensive fits using the two existing
  geometries before starting a full merge search. The same native-source error
  gates decide acceptance. A passing inexpensive candidate skips residual-layer
  search; it is not accepted merely for having similar mean colours.
- `--paint-merge-passes N` reconsiders merged regions, with one round by default,
  at most eight rounds, and immediate termination
  when no merge succeeds. Only the first round may invoke the full expensive
  search; subsequent rounds use the cheap proposals. Rebuild boundary evidence
  after compaction, so all contacts between the new incident regions are checked.
- Cache failed pairs whose components have not changed. A component's first pixel
  and area identify it along the monotonic sequence of unions; area changes on
  every union, invalidating its previous decisions. Disjoint-pair exclusions are
  not cached as failed fits.

The default remains one pass: simply enabling repeated merging increased latency
more than it reduced real-image region counts. Fitted-template geometry preservation
applies with the default too. Multi-pass merging is an explicit size/time tradeoff,
not a claimed speed optimization.

The accepted topology labels are replaced before shared geometry is generated.
This removes the internal contour, rather than just batching two SVG paths or
hiding their interface with an extra drawing element. Layered paints can still
emit multiple elements for a single topology region; region count and XML path
count are therefore different quantities.

## Quality conditions

Native boundary evidence, smooth-contact checks, and paint seam limits are retained.
Candidate mean/P90 CIEDE2000 regression on **each incident face** stays within
0.30/0.75, and the combined sampled mean/P90 regression stays within 0.01/0.04.
The code factors these existing checks into one helper shared by quick and full
proposals. These are sampled per-round checks, not a proof of pixelwise or globally
zero regression. Repeated native rendered comparisons and thin-line tests remain
necessary; a lower region count alone is not success.

Tests cover four ramp fragments joining into one face, preservation of a material
boundary, preservation of a faint one-pixel line, and fitted diagonal/radial
geometry preservation. The existing supported-merge tests are retained.

Use `picvec input.png output.svg --threads 4 --paint-merge-passes 8` to try the
optional mode.
