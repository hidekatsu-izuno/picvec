# Chroma-key and source-alpha background removal

## Behaviour

Source alpha and inferred chroma keys are both matte-aware, but have
intentionally different activation and representation rules:

- A source alpha channel is honoured automatically. No option is required.
- An opaque red, green, blue, cyan, magenta, or yellow backing is considered
  only with `--remove-chroma-key-background` (or the corresponding `Config`
  field).
- White and black are never selected as automatic chroma keys.

The pipeline does not flatten either form onto white. Exact source alpha first
remains at its decoded one-byte coverage precision. The mask builder
distinguishes raster edge coverage from durable translucent regions. A
connected intermediate-alpha component which merely forms a thin shoulder
between clear and opaque regions is raster antialiasing. It is used to locate
the `alpha = 0.5` crossing between pixel centres, then discarded as an output
opacity. Cubic fitting produces one fully opaque contour at that subpixel
position, and the SVG renderer supplies fresh antialiasing at the current
display resolution. This avoids both enlarged pixel steps and a
semitransparent vector halo.

Intermediate alpha which occupies an independent region or has a broad,
erosion-resistant core is treated as authored transparency and quantized
uniformly to two bits:

```text
level       = round(clamp(alpha, 0, 1) * 3)
alpha2bit   = level / 3
```

The four levels `0`, `1/3`, `2/3`, and `1` are fitted as nested binary
superlevel regions and applied as one SVG alpha mask to Paint faces and
structural strokes. Adjacent levels therefore meet at a shared vector boundary
rather than through a source-resolution opacity ramp. Exact decoder alpha remains compact at one byte per source sample. A
temporary grayscale raster is materialized only when fitting continuous
coverage fields.
Broad continuously varying alpha is an exception to the four-level model.
Smooth interior samples distinguish authored coverage ramps from silhouette
antialiasing. Connected coverage bands are fitted with linear or elliptical
fields against the original decoder alpha, including the high-opacity band.
The fields form a grayscale SVG luminance mask. Opaque silhouettes and flat
transparency retain the compact mask above.

A field must keep mean absolute alpha error within 2/255 and p90 within 4/255.
Unresolved components use local six-bit coverage contours; the fallback does
not repeat whole-canvas contours at every level. Small overlaps between partial
coverage paths prevent antialias gaps at artificial band boundaries. Opaque
mask paths are not expanded. The reported alpha precision is 8 for a field mask
and 2 for the flat mask; it reports the maximum precision, not a guarantee of
exact per-pixel reconstruction. Adaptive child masks propagate that precision.
Quality evaluation composites with original decoder alpha, not its quantized
approximation.

Visible pixels retain straight source RGB. Pixels quantized to zero are
not automatically discarded: every sample with nonzero source coverage is
kept as underpaint beneath the interpolated mask. Only exactly zero-alpha RGB
is normalized to the saturated RGB cube corner farthest on average from
covered source pixels. That colour is never used to decide source-alpha
visibility. A foreground area equal to the normalization colour therefore
remains visible through the independent mask, while a subpixel mask crossing
cannot expose a raster-aligned normalization fringe.

SVG previews and rate-distortion comparisons composite the resulting mask over
the same normalization colour. Inferred chroma keys continue to use the
half-coverage ownership model described below because they do not provide an
independent authored alpha channel.

## Automatic chroma-key model

The detector examines a shallow band around the image (roughly the outer
1/64, clamped to 2--64 pixels) and considers the six RGB cube corners other
than black and white. A candidate must cover at least 20% of that band with
RGB Euclidean distance at most `56/255`. The component-wise median of the
matching band samples becomes the actual backing colour; this tolerates a
narrow neutral frame as well as small quantisation, JPEG, or capture
variation.

Detection selects the colour, but does not authorize removing every matching
pixel. Removal starts at four-connected patches within `56/255` of that
sampled colour whose area is at least `clamp(source_pixels / 1024, 64, 256)`
pixels. The upper cap prevents a larger canvas from disqualifying useful
enclosed background patches.
Each qualifying patch seeds a flood through matching colour and connected
darker faces with key-channel excess above `32/255` and 35% of their weakest
high channel. Opaque foreground anchors stop the flood. This works for broad
enclosed backgrounds as well as outer backgrounds; there is no border-touch
requirement. Small isolated matching patches, such as the server rack lamps,
do not seed removal. Key-hued samples outside the connected background retain
full opacity and their original RGB, even beside a one-pixel separating
outline. Within three pixels of the connected background, weakly contaminated
silhouette samples can still use local antialias fitting to remove colour
spill. Their fallback coverage is above 0.5 and the fit cannot lower it, so
this exception cannot create a new background-owned patch across an outline.

Opaque anchor detection uses an RGB distance of `64/255` from the sampled
backing. Pixels beyond that distance whose neighbours within a two-pixel radius
are also distinct are opaque foreground anchors. Their source colour is retained,
even when it has the same hue as the key. For example, a green object with
RGB `(50,213,86)` is distinct from a `(0,255,0)` backing.
Smaller backing variations, such as `(0,239,0)`, do not become opaque anchors
even when they occupy a broad patch. Shaded backing is also excluded from
anchor evidence when at least one of the key's low channels remains within
`24/255` of zero and the key's high channels differ by at most `24/255`.
This allows brightness variation and contamination in one low channel while
retaining distinct same-hue objects with raised low channels. Darker
foreground-side fringes are unmixed, preventing shaded green from becoming
opaque green ink along the hole. Local fitting against genuine foreground
anchors remains enabled to preserve same-hue object antialiasing. This
keeps the dark green gap under the woman's arm in `cliparts-6x6` on the
background side without weakening distinct-colour foreground anchors.
The anchor threshold is separate from
the tighter edge-fitting tolerance so that suppressing backing variation does
not loosen the fit of antialiased edges.

Residual shaded patches require spatial evidence as well as colour. Among
foreground-side non-anchor samples, a candidate must retain key-channel
excess greater than `32/255` and greater than 35% of its weakest high channel.
An opaque anchor with key-channel excess above `24/255` within three pixels
protects nearby same-hue foreground. A four-connected candidate component
must belong to the seeded connected background, touch an already
background-owned sample, and contain a core surviving
one-pixel erosion. Only samples within three graph steps of its cores are
cleared. Thus a broad connected backing shadow can be removed without
following a thin antialias fringe around the rest of the silhouette or
clearing isolated dark marks. This is a spatial heuristic; genuine foreground
indistinguishable from shaded backing remains ambiguous.

The shaded-gap regression uses a native crop from `(1140, 2260, 130, 150)`
of `cliparts-6x6.png`, checking clear gap ownership and opaque blue clothing.
Synthetic regressions cover all six key colours, dark fringe unmixing,
bounded shadow propagation, isolated marks, and same-hue foreground
antialiasing. The regenerated full sheet accepts all 36
native refinements. [Before/after at 4×](../sample/comparison/cliparts-6x6-armpit.png)
shows the same source-space crop rendered over white, including the residual
upper and lower patches removed by the connected-shadow pass. The native
server-rack fixture checks all four lamps at full and half resolution.
Synthetic tests also cover a one-pixel separating outline and the capped
seed area on large canvases. [Server-rack before/after](../sample/comparison/cliparts-6x6-server-rack.png)
shows the full-sheet rendering after applying the broad-seed requirement.
Validation: 268 tests passed, 4 existing tests ignored; the changed file's
formatting and Clippy passed (with the
existing `needless_range_loop` lint allowance).

At the fringe, anchors within three pixels provide candidate foreground
colours. Projecting `C - K` onto `F - K` estimates coverage; candidates must
reconstruct the observed colour within `12/255`. Exact backing pixels in
qualified connected backgrounds remain clear. A small isolated exact-key
patch is preserved as foreground.

For unsupported details, let `H` be the selected corner's channels equal to
`FF` and `L` its channels equal to `00`. The fallback colour-difference matte is

```text
d(C)     = min(C[h] for h in H) - max(C[l] for l in L)
alpha(C) = clamp(1 - d(C) / d(Ckey), 0, 1)
```

This is a symmetric six-corner extension of the classic colour-difference
keyer. Accepted local estimates can raise this fallback coverage, while opaque
anchors have coverage one. The matte is not converted to a white-backed raster. At the 0.5 matte
crossing, background-side pixels are normalized to the key colour. On the
foreground side, key contamination is removed by solving the compositing
equation for `F`; ordinary segmentation then sees a clean foreground-to-key
boundary instead of green/blue/etc. antialias colours.

After final paint merging, every region is classified independently from its
matte samples. The 0.5 matte crossing is the foreground/background ownership
boundary: a region is omitted when more than half its pixels are below 0.5
and its mean alpha is also below 0.5. The mean guard makes the decision
conservative when a region contains a small number of high-coverage foreground
samples. Using the ownership crossing, rather than only nearly clear pixels,
also assigns key-contaminated antialiasing to the omitted side of the vector
boundary. This is deliberately not restricted to a border flood fill: an
enclosed background can qualify by containing its own broad key patch. Full region
geometry is built before omission so the surrounding foreground retains the
corresponding hole. Structural strokes are filtered with the same ownership
test so key-coloured antialias fragments cannot survive in a separate SVG
line layer.

## Research basis and limitation

Smith and Blinn formulate constant-colour matting as
`C = alpha * F + (1 - alpha) * K`, review the Vlahos colour-difference method,
and show that recovering both foreground colour and alpha from one keyed image
is underdetermined in general: [Blue Screen Matting, SIGGRAPH 1996](https://www.microsoft.com/en-us/research/publication/blue-screen-matting-2/),
[DOI 10.1145/237170.237263](https://doi.org/10.1145/237170.237263).
Porter and Duff establish the alpha/coverage compositing model and the need to
preserve fractional coverage at antialiased boundaries:
[Compositing Digital Images, SIGGRAPH 1984](https://keithp.com/~keithp/porterduff/p253-porter.pdf).

Foreground indistinguishable from the sampled key can still be removed, and
thin details without nearby opaque anchors retain the colour-difference
assumption. Broad translucent areas can be interpreted as opaque under the
distinct-interior model. Exact source alpha does not
have that inference ambiguity: its visibility is represented by the
independent vector mask rather than by colour classification. Smooth authored
alpha uses the continuous luminance-mask path described above; flat coverage
retains the two-bit mask.
