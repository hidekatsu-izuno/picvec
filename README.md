# picvec

Native Rust implementation of the perceptual, structure-aware raster-to-SVG
pipeline. It creates editable SVG Paint faces, Office-compatible solid/linear/
elliptical-radial fills (at most five stops), raster-derived shared topology,
analytic primitives, and source-supported structural centre-lines.

The converter does not invoke the former Python implementation.

## Build

```bash
mise exec -- cargo build --release --locked
```

`rsvg-convert` (librsvg) is required at runtime for the native-resolution,
in-memory Paint/structural ownership check. Its temporary SVGs are removed
before the command returns.

## Run

```bash
picvec input.png output.svg
```

The input processing size is selected automatically from source complexity;
`--max-dimension` sets its upper bound. The only output is the SVG file named
by the second positional argument. No source copy, rendered PNG, or JSON
sidecar is created.

Useful controls:

```text
--max-dimension <PX>
--max-input-dimension <PX>
--max-input-megapixels <MP>
--max-decode-mib <MIB>
--smoothing-radius <PX>
--segmentation-min-size <AREA>
--quantization-dark-delta-e <DE>
--quantization-light-delta-e <DE>
--gradient-merge-error <DE>
--threads <N>                 # 0: detected physical cores
--rsvg-convert <PATH>
--verbose
```

Input dimensions and total area are checked from the image header before
decoding (32,768 pixels per axis and 32 megapixels by default), and decoder
allocations have a 512 MiB best-effort limit. The verbose report records the
exact `rsvg-convert --version` output.

`--verbose` prints the in-memory timing and complexity report to stderr for
development; it still writes no sidecar files.

The published Rust crate contains only the converter source and its legal
documentation; sample images and the evaluation-only Real-ESRGAN model are
kept outside the crate. See `THIRD_PARTY_NOTICES.md` and `sample/README.md`
before redistributing those repository assets.

## Examples

Each comparison shows the source raster on the left and the corresponding SVG
rendered by librsvg on the right. The comparison assets can be regenerated with
`scripts/generate_sample_comparisons.sh`.

![Boy and turtle raster input and rendered SVG output](sample/comparison/boy_and_turtle.png)

![Car raster input and rendered SVG output](sample/comparison/car.png)

![Clip art raster input and rendered SVG output](sample/comparison/cliparts.png)

![Mountain raster input and rendered SVG output](sample/comparison/viewport1.png)

![Coast raster input and rendered SVG output](sample/comparison/viewport2.png)

## x4 evaluation

`scripts/evaluate.py` evaluates a completed SVG against a Real-ESRGAN x4
reference without feeding that image back into vectorization. A standalone x4
PNG can be created with SVGDeck's migrated PyTorch/Spandrel generator:

```bash
timeout 600s nice -n 10 mise x -- uv run scripts/generate_realesrgan_x4.py \
  input.png reference-x4.png \
  --model /path/to/RealESRGAN_x4plus_anime_6B.pth
```

See `scripts/picvec_eval/README.md` for NCNN and PyTorch evaluator usage,
content-addressed caching, and reproducibility controls.

## Pipeline

1. Select a processing dimension from a bounded source-complexity probe.
2. Build a multiscale structure-tensor direction field and classify normal
   profiles as material boundaries, medial ridges, ridge-on-boundary, or
   shading in CIELAB.
3. Give thin structural ink one nearest incident Paint owner, unmix only
   source-supported antialias shoulders, and keep dark filled faces in Paint.
4. Quantize global Lab histogram cells without transitive spatial chaining;
   only four-connected equal-palette samples become one geometry region.
5. Preserve locally visible/independent small materials and return unsupported
   one-pixel antialias shoulders to an adjacent Paint owner.
6. Resolve ambiguous two-parent antialias pixels with a locally regularized
   graph cut, then regularize only quantization boundaries unsupported by a
   dilated source barrier; structural pixels and topology-changing moves stay
   locked.
7. Compress the exact dense ownership partition into a non-uniform quadtree.
   Uniform interiors retain one rectangular cell while mixed cells split down
   to source pixels; expanding the leaves reproduces the raster labels exactly.
8. Fit each physical raster interface from the same hierarchy as one master
   Bezier chain, inserting
   material transitions and high-degree intersections as exact nodes before
   fitting, and reuse every resulting boundary in reverse for its neighbour.
   Validate the complete partition together; if a curve collapses an incident
   face, downgrade that canonical curve for all neighbours rather than fitting
   either face independently. Closed faces must preserve orientation and
   source-supported area before exact rectangles/circles/ellipses are
   substituted.
9. Validate each Paint owner against the same hierarchy, then fit solid,
   arbitrarily oriented axial linear, or elliptical radial
   Office-compatible Paint with at most five stops. Adjacent linear gradients
   are coupled only when the combined fitted error remains acceptable.
10. Render the Paint base at native resolution in memory, transfer only missing
   structural intervals, and retain the lower-DeltaE ownership candidate.
11. Overlap Paint boundaries by 0.2 source pixels to suppress renderer seams.
