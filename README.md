# picvec

Native Rust implementation of the perceptual, structure-aware raster-to-SVG
pipeline. It creates editable SVG Paint faces, Office-compatible solid/linear/
elliptical-radial fills (at most five stops), raster-derived shared topology,
analytic primitives, and source-supported structural centre-lines.

The converter does not invoke the former Python implementation.

## Build

```bash
mise exec -- cargo build --release
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
--smoothing-radius <PX>
--segmentation-min-size <AREA>
--quantization-dark-delta-e <DE>
--quantization-light-delta-e <DE>
--gradient-merge-error <DE>
--verbose
```

`--verbose` prints the in-memory timing and complexity report to stderr for
development; it still writes no sidecar files.

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
7. Fit each physical raster interface as one master Bezier chain, inserting
   material transitions and high-degree intersections as exact nodes before
   fitting, and reuse every resulting boundary in reverse for its neighbour.
   Validate the complete partition together; if a curve collapses an incident
   face, downgrade that canonical curve for all neighbours rather than fitting
   either face independently. Closed faces must preserve orientation and
   source-supported area before exact rectangles/circles/ellipses are
   substituted.
8. Fit solid, arbitrarily oriented axial linear, or elliptical radial
   Office-compatible Paint with at most five stops. Adjacent linear gradients
   are coupled only when the combined fitted error remains acceptable.
9. Render the Paint base at native resolution in memory, transfer only missing
   structural intervals, and retain the lower-DeltaE ownership candidate.
10. Overlap Paint boundaries by 0.2 source pixels to suppress renderer seams.
