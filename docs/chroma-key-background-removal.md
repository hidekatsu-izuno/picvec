# Chroma-key and source-alpha background removal

## Behaviour

Source alpha and inferred chroma keys use one common matte-aware vectorization
path, but have intentionally different activation rules:

- A source alpha channel is honoured automatically. No option is required.
- An opaque red, green, blue, cyan, magenta, or yellow backing is considered
  only with `--remove-chroma-key-background` (or the corresponding `Config`
  field).
- White and black are never selected as automatic chroma keys.

The pipeline does not flatten either form onto white. For source alpha, it
chooses the saturated RGB cube corner farthest on average from covered source
pixels and temporarily evaluates

```text
Ckeyed = alpha * Cforeground + (1 - alpha) * Ckey
```

For both input forms, the inferred or exact matte is converted into vector
ownership at its half-coverage crossing. SVG previews and rate-distortion
comparisons are rendered over the same temporary or detected key used by the
vector model.

## Automatic chroma-key model

The detector examines a shallow band around the image (roughly the outer
1/64, clamped to 2--64 pixels) and considers the six RGB cube corners other
than black and white. A candidate must cover at least 20% of that band with
RGB Euclidean distance at most `56/255`. The component-wise median of the
matching band samples becomes the actual backing colour; this tolerates a
narrow neutral frame as well as small quantisation, JPEG, or capture
variation.

For a selected corner, let `H` be its channels equal to `FF` and `L` its
channels equal to `00`. The soft colour-difference matte is

```text
d(C)     = min(C[h] for h in H) - max(C[l] for l in L)
alpha(C) = clamp(1 - d(C) / d(Ckey), 0, 1)
```

This is a symmetric six-corner extension of the classic colour-difference
keyer. The matte is not converted to a white-backed raster. At the 0.5 matte
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
boundary. This is deliberately not a border flood fill: an enclosed or
otherwise disconnected region of the same backing is removed too. Full region
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

The unavoidable consequence is that opaque foreground content which itself
looks like the selected key can be removed. Exact source alpha does not have
that inference ambiguity; its temporary key is chosen from the subject colours
to reduce the chance that the RGB segmentation merges subject and backing.
