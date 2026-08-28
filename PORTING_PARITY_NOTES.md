# raster2svg Rust port parity notes

This file is the working ledger for the direct port from
`svgdeck/scripts/raster2svg` to this Rust crate.  A stage is not considered
ported merely because the final image looks similar; constants, ownership
rules, intermediate counts, and render-gated acceptance must also agree.

## Reference

- Python entry point: `scripts.raster2svg.perceptual_pipeline`
- Edge roles: `geometry_raster_edge_vectorizer.py`
- Source stroke graph: `stroke_graph.py` and `structural_ink.py`
- Paint fitting/continuity: `gradient_region_svg.py`
- Shared geometry: `bezier.py`
- Reference comparison case: `car.png`, forced to `--max-dimension 768`

## Ported and checked

- scikit-image-compatible sRGB-to-CIELAB coefficients and the Python
  pipeline's distinct float32 CIELAB-to-sRGB inverse matrix.
- Automatic complexity sizing and the 768--1600 dimension limits.
- Multiscale CIELAB structure tensor, oriented NMS, hysteresis thresholds,
  Guo--Hall thinning, and the two-pass direction-supported gap bridge.
- Local normal-profile classification of boundary, ridge,
  ridge-on-boundary, shading, and unknown samples.
- Absolute/locally-dark ridge support, Euclidean distance transform,
  medial-axis ownership, local gap interpolation, and source graph joining.
- Structural classification now uses the reference's four-connected binary
  propagation and silhouette material test.  On the 768 px car, both the
  9,846 candidate pixels and all 4,351 legacy line pixels agree exactly.
- Structural antialias unmixing transfers successfully unmixed shoulder
  pixels out of Paint without incorrectly adding those shoulders back to the
  edge-role graph.
- Paint-aware RAG merging now uses balanced child sampling, inset samples,
  solid-first fitting, warm starts, Office-compatible model enumeration, and
  per-child perceptual gates.  A long seed colour interface is no longer an
  unconditional block: the proposed Paint is accepted when it reproduces the
  measured source jump, matching `region_merge.py`.
- Paint fitting and merging use integer raster coordinates and rectangular
  diagonal-gradient endpoints, rather than the former half-pixel and
  projected-diagonal approximation.
- Shared master boundaries retain canvas corners and are sliced with exact
  de Casteljau subdivision so adjacent faces reuse one reversed curve.
- Paint geometry now uses the reference midpoint-observation straightness
  test, lexicographic Potrace optimal-polygon dynamic program, least-squares
  polygon-vertex adjustment, and Potrace corner curves.  Every material
  transition is produced by exact de Casteljau slicing of that shared master;
  it is not re-fitted per face.
- The reference continuity-class concept is also present. It traces a long
  perceptual interface through short Paint-label changes and accepts fairing
  only inside the symmetric raster corridor. The connected-endpoint subspan
  refinement remains listed below because a partial port created thin cubic
  miter overshoots and was deliberately not accepted.
- Every material transition and high-degree boundary junction is now inserted
  as a shared node before face assembly. Per-face curve fallback is no longer
  allowed.
- Local Paint coupling can promote solid bands to Office linear gradients and
  jointly solve five stop colours with source-boundary continuity constraints.
- Long source-smooth Paint faces are split into content-balanced local patches
  before the shared half-edge graph is built.

## Measured checkpoint: car at 768 px

The current Python reference (`/tmp/car-python-current-768/summary.json`) and
Rust release build were run from the same 768 x 768 RGB input.

| Measurement | Python | Rust |
| --- | ---: | ---: |
| legacy structural candidates | 9,846 | 9,846 |
| legacy structural lines | 4,351 | 4,351 |
| role-owned source lines | 4,962 | 4,962 |
| ridge underpaint ownership | 2,897 | 2,897 |
| visible-ridge graph edges | 22 | 22 |
| visible-ridge coverage pixels | 1,931 | 1,931 |
| dark-boundary graph edges | 146 | 146 |
| dark-boundary coverage pixels | 11,950 | 11,950 |
| normal-profile skeleton pixels | 12,687 | 12,687 |
| corrected quantized regions | 1,669 | 1,669 |
| boundary-regularization moves | 1,123 | 1,123 |
| Paint-aware merges | 91 | 91 |
| thin-Paint examined/refined/removed | 578 / 9 / 7 | 578 / 9 / 7 |
| adaptive faces/splits/patches | 86 / 17 / 29 | 86 / 17 / 29 |
| final SVG faces | 1,600 | 1,600 |
| structural strokes | 87 | 87 |

The corrected label partition, regularized label partition, and Paint-merged
label partition are pixel-for-pixel equivalent (a bijection of label IDs).
All normal-profile role masks and both rasterized source graphs are also
pixel-for-pixel equal.

Native librsvg comparison against the source gives mean / p90 / p99 DeltaE00
of 1.080 / 2.001 / 12.859 for Python and 0.946 / 1.810 / 11.656 for Rust.
The remaining exporter differences therefore do not hide a quality-relaxing
shortcut: the native Rust render has lower source error at all three reported
percentiles.

## Five-sample output gate

The checked-in Python `evaluation-v2/*/current` renders and the Rust SVGs were
rendered with the same librsvg build at identical raster dimensions. Dog and
viewport2 were forced to the reference run's 768 px ceiling; the other three
reference runs already use the same automatically selected dimensions. Values
below are source-to-render DeltaE00 mean / p90 and RGB SSIM.

| Sample | Python DeltaE00 | Rust DeltaE00 | Python SSIM | Rust SSIM |
| --- | ---: | ---: | ---: | ---: |
| car | 0.924 / 1.692 | 0.775 / 1.479 | 0.9551 | 0.9703 |
| cliparts | 0.829 / 0.784 | 0.779 / 0.526 | 0.9434 | 0.9683 |
| dog | 0.501 / 0.749 | 0.297 / 0.000 | 0.9824 | 0.9814 |
| viewport1 | 6.123 / 12.858 | 5.416 / 11.302 | 0.6774 | 0.6967 |
| viewport2 | 4.089 / 9.464 | 3.653 / 8.267 | 0.7797 | 0.8046 |

Rust improves both reported perceptual-error statistics on all five cases.
The only lower aggregate entry is dog SSIM by 0.0010; its mean DeltaE00 falls
by 41 percent and the visual thin-feature check passes, so matching that one
metric by relaxing edge fidelity would be a regression. Native-size renders
of all five SVGs were also checked for transparent cracks, missing thin ink,
and unsupported dark regions.

## Parity defects found and corrected

- Two-point skeleton chains were being classified even though the reference
  minimum is three points.
- Gap repair evaluated coordinates and even-length medians in float32 and
  selected one extra bridge pixel. It now uses the reference float64 sampling
  and median convention.
- Non-ridge profile centres and widths were stored as zero instead of NaN.
  That disabled the reference interpolation across a short unsupported medial
  interval and shifted a dark centre-line by about 0.34 px.
- Source-graph width history was initialized before overlap slicing. It is now
  initialized only on an actual join, so weighted medians use the surviving
  point counts.
- The antialias graph cut returned the minimal source side of an equal-energy
  cut. NetworkX returns the maximal source side by complementing residual
  sink-reachability; Rust now makes the same deterministic ownership choice.
- Paint inset samples used a fixed 2 px disc. The reference scales the inset
  from 2 px at 1024, so it is 1.5 px at 768. The Rust stencil now implements
  the same arbitrary-resolution Euclidean condition.
- The minimum coherent structural-colour boundary length was fixed at eight
  pixels. It now follows the same spatial scaling (six pixels at 768), which
  prevents three unsupported Paint merges on the car case.
- The merge estimator expanded every shortlisted gradient geometry. The
  reference expands the best perceptual finalist first and opens the rescue
  pool only if that result would fall back to Solid; Rust now follows that
  model-selection order.
- Structural paths stopped after RDP simplification, so long shallow curves
  retained raster steps as alternating anchors around the car roof and door.
  They now pass through bounded Gaussian fairing before SVG emission.  Every
  accepted curve must remain within one raster-cell diagonal of both the raw
  detector chain and its fitted baseline in both directions, while persistent
  corners and endpoints are retained.  Applying the same displacement to
  Paint's shared half-edge graph was tested and rejected: it increased both
  runtime and source-render error, so Paint topology remains unchanged.
- Paint shared boundaries used a greedy grid simplifier and local Catmull
  controls instead of `bezier.py`'s global optimal polygon.  More importantly,
  a Rust-only endpoint-area check treated a valid shallow cubic as if only its
  endpoints described its area and downgraded hundreds of masters to the
  original pixel walk.  That check has been removed.  Both incident faces now
  reuse the accepted corridor-bounded master, as in Python.  On the 768 px car
  the shared-curve downgrade count changes from 443 to zero; the background
  path changes from 95 line commands to a curve-only path, with zero shared
  loop discontinuities or fallbacks.

## Required review / known deviations

1. **Dark-boundary selection.**  Python performs a rendered normal-profile
   residual gate and source-valley interval selection before emitting these
   paths. Rust offers the same 146 supported graph edges to the general
   residual selector. The final stroke count agrees on the car reference, but
   role-by-role emission should still be compared on the other samples.

2. **Structural curve fitting.**  Python retains source graphs, applies
   bounded fairing, role-specific continuation, intersection/junction snaps,
   and mask-constrained fallback curves. Rust now applies the same bounded
   fairing principle, including symmetric source support and persistent-corner
   protection. Its source graphs avoid re-skeletonising expanded ridge
   coverage, but role-specific continuation and the internal fallback/profile
   split are not yet identical.

3. **Paint fitting and coupling.** The face partition is exact, while the car
   exporter currently chooses 184 linear and 50 radial gradients versus
   Python's 189 and 48. Rust's rendered DeltaE is lower, so thresholds must
   not be altered merely to force these counts. Compare individual Paint
   model decisions and port the remaining profile/coupling details only when
   they preserve or improve the render gate.

4. **Geometry fitting parity.**  The midpoint-based optimal polygon, Potrace
   corner construction, exact shared slicing, and bounded fairing are ported.
   A ternary exhaustive 3x3 topology test and the five sample images produce
   zero discontinuous or per-face fallback loops. Two conservative reference
   refinements still need their complete acceptance rules: persistent-corner
   subspan splitting with connected master ownership, and the short-corner-
   excursion regularizer. The latter must retain its support-ray,
   material-pair, bounded-area, and protected-junction gates. Do not replace
   either with morphology or a generic corner filter.

5. **Final optimizer.**  Rust writes rectangles and shared gradients, but the
   Python optimizer's render-gated batching and analytic circle/arc
   replacement are not fully ported.  Optimization must remain downstream of
   fidelity work.

6. **Performance.**  Structural stroke joining now uses deterministic endpoint
    buckets while retaining the reference's global best-candidate ordering;
    the viewport1 structural stage fell from 32.6 s to 0.44 s without changing
    its SVG bytes. Boundary-protection queries are indexed by incident label,
    and independent initial Paint fits run in parallel with indexed collection,
    which preserves region order. A later parity sync had accidentally changed
    this loop back to sequential iteration; restoring the indexed parallel
    iterator reduced car 768 px Paint fitting from 63.991 s to 39.620 s and
    total time from 74.807 s to 50.198 s, with byte-identical car and dog SVGs.
    Each harmonization pass now also evaluates its independent adjacent-owner
    proposals in parallel, then restores the reference decision order with the
    existing complete score/owner sort before applying any proposal. On car at
    768 px this reduced the harmonization substage from 34.939 s to 0.557 s and
    total time from 52.074 s to 16.829 s. Car and dog SVGs remained
    byte-identical, including all internal DeltaE00 and SSIM values.
    Local Paint coupling now likewise fits independent member geometries and
    solves the three fixed RGB systems with indexed parallel iterators. Car
    coupling fell from 4.408 s to 3.506 s and total time from 16.829 s to
    15.799 s; the small dog coupling case also stayed neutral-to-faster at
    0.289 s to 0.284 s. Both SVGs remained byte-identical.
    Disjoint coupling groups are now evaluated against one immutable Paint
    snapshot in parallel and their accepted assignments are applied in label
    order. This further reduced car coupling from 3.506 s to 1.239 s and total
    time from 15.799 s to 13.737 s; dog coupling fell from 0.284 s to 0.263 s.
    The car and dog SVG hashes were again unchanged.
    Final-canonical ridge evidence and Lab values are now computed once and
    shared by Paint-sample adjustment and strong-branch selection; the former
    implementation repeated the complete detector immediately in Paint
    fitting. Car Paint setup fell from 1.195 s to 0.031 s and total time from
    13.737 s to 12.504 s. Dog Paint setup fell from 6.968 s to 0.099 s and
    total time from 35.451 s to 28.287 s. Both SVGs remained byte-identical.
    Normal-profile overlap removal now keeps its accepted-owner samples in a
    one-pixel spatial index instead of rebuilding and scanning the complete
    sample list for every candidate. The exact 1.5 px distance and 20 degree
    tangent gates, candidate ordering, and incremental ownership semantics are
    unchanged and covered by a direct-scan equivalence test. On a dense,
    spatially separated profile workload the release build fell from 511.3 ms
    to 4.1 ms (124x); the car 768 SVG remained byte-identical at 845,706 bytes
    with unchanged DeltaE00 and SSIM. A native automatic run of viewport2
    (1600 x 1067, 78,618 regions) still takes 208 s and emits 22 MB.

7. **Illumination/ridge Paint ownership.**  Python computes broad
    shade/light masks, encodes `paint_key = parent * 3 + tone`, partitions dark
    and chromatic/bright ridge fills, and reconstructs unsupported one-pixel
    Paint before disconnected-face splitting.  Rust currently represents most
    of this information through segmentation barriers and a later structural
    layer.  Port these ownership stages directly before claiming full parity.

8. **Backend-independent numerical kernels.** Exact parity currently requires
   NumPy 2.4.4's CPU-dispatched float32 `pow`, `cbrt`, `atan2`, `exp`, and
   reduction order. After parity is established, define and test a canonical
   colour-math backend so results do not depend on the host NumPy/SVML dispatch.
   This is intentionally not being changed during the parity pass.

9. **Batched merge colour conversion.** The reference evaluates Paint errors
   over contiguous arrays. The direct Rust port now preserves that operation
   shape, but repeatedly allocates temporary RGB/Lab buffers per proposal.
   Reusable per-worker buffers or a fused exact kernel could remove this cost;
   defer it until SVG byte parity tests prove that allocation/order changes do
   not alter any fitted Paint or merge decision.

## Geometry defect found during the port

The former Rust implementation fitted a long master Bezier and then cut it at
a material transition using `raw_edge_index / span_length` as the Bezier
parameter.  Raw index is not Bezier arclength, so two incident strands computed
different coordinates for the same nominal junction.  On car at 768 px this
made 1529 loops fail continuity and silently fall back to independently fitted
face paths, which created the transparent wedges that looked like black
triangles.  Mandatory pre-fit junction nodes reduce discontinuities to zero;
the whole-partition downgrade gate prevents this class of crack from recurring
on a different image.

## Verification commands

Run short checks first, then a sample conversion; all expensive commands use
the repository-required timeout and reduced priority.

```sh
timeout 180s nice -n 10 cargo test
timeout 180s nice -n 10 cargo clippy --all-targets -- -D warnings
timeout 240s nice -n 10 cargo build --release
timeout 180s nice -n 10 target/release/picvec \
  sample/input/car.png /tmp/car.svg --max-dimension 768 --verbose
```

For every accepted change, record both the intermediate-count movement and a
native-size render comparison.  A lower global mean error is not sufficient
if a source-supported thin feature becomes disconnected.

## Deferred improvements found during exact-parity work

- **Canonical graph-cut saturation.** `networkx.minimum_cut` removes a
  residual edge only when `flow == capacity`. In the car AA partition, the
  value-only preflow leaves one edge at `0.15000000000000002` for a capacity
  of `0.15`; NetworkX therefore treats that over-capacity edge as traversable
  and selects a different member of an equal-cut plateau. A tolerance-based
  saturation test or a canonical feasible-max-flow partition would be more
  numerically robust, but it would change the Python reference result. The
  Rust port intentionally preserves the reference behaviour; make any such
  change in both implementations only after parity is complete and add an
  explicit plateau regression test.

- **Stable Lab histogram cells.** The current one-unit Lab histogram uses
  raw `rint` at half-integer boundaries. Sub-ulp differences between NumPy's
  vector colour math and scalar Rust colour math can move a handful of pixels
  to another cell and cascade through palette representative updates. A
  tolerance-aware or fixed-point cell definition would make the algorithm
  portable across math backends, but changing that definition is a quality
  improvement rather than a faithful port and is deliberately deferred.

- **Math-backend-independent RAG priority.** Paint-merge proposals with an
  error around `1e-5` can become exact zero under scalar float32 colour math.
  The reference resolves equal heap scores by scikit-image RAG insertion
  order, so a numerically insignificant difference may change later merge
  availability. Quantizing the priority below a documented epsilon and using
  an explicit topology-derived tie key would be more portable, but it would
  change Python's present queue order. The port therefore preserves the RAG
  order and leaves priority stabilization for a coordinated post-parity
  change.

- **High-region-count segmentation cost.** The current exact native run of
  `viewport2.jpg` selects 1600 x 1067, produces 77,884 regions, and takes
  566.182 s; the 300 s bounded run stopped while still in segmentation. The
  earlier approximate implementation completed the same sample much faster,
  so allocation reuse, indexed neighbourhood updates, and exact-result
  batching should be profiled specifically on this case. Do not change merge
  order or reduce the automatically selected input size merely to improve the
  timing; guard any optimization with label, Paint, and rendered-SVG parity.

- **Decision stability after millipixel formatting.** On the final car
  comparison, sub-millipixel numerical differences leave the same 1,787 paths,
  67 lines, 252 linear gradients, and 68 radial gradients, but can change a
  few equivalent optimizer choices (straight cubic versus `L`, cubic versus
  analytic arc) and retain one extra Office gradient stop. Canonicalizing
  decision inputs at the SVG formatting precision would make the output more
  backend-stable, but it would change the current Python reference thresholds.
  Apply such stabilization to both implementations only after parity and add
  command/stop-count regression coverage.
