# picvec

picvec converts raster images into editable SVG using Rust.

## Build and convert

### Build

```sh
mise exec -- cargo build --release --locked
```

```sh
picvec input.png output.svg
```

The second argument is the exact SVG file to write. 

Completion reports the dimensions, **final SVG object count**, **path contour
count** (`subpaths`) and elapsed time. Groups and definitions are excluded from
the object count. A compound path is one object even when it contains many
`M`/`m` contours; the contour count makes that distinction visible. These counts
come from the final SVG, including accepted source refinements.

| Option | Purpose and default |
| --- | --- |
| `--max-dimension <PX>` | Maximum automatic **base** processing dimension (1600). Source refinements may be larger. |
| `--no-adaptive-refinement` | Disable source-resolution refinement. |
| `--adaptive-svg-budget-mib <MIB>` | Additional SVG byte budget for refinement (0: unlimited). Quality and efficiency checks still apply. |
| `--remove-chroma-key-background` | Detect and remove a saturated red, green, blue, cyan, magenta or yellow backing. |
| `--paint-merge-passes <N>` | Paint merge passes, 1–8 (1). |
| `--oklab-palette-threshold-scale <FACTOR>` | Palette tolerance multiplier in 100-scaled OKLab units (1.0). |
| `--threads <N>` | Worker count (0: half the detected CPUs, at least 1 and capped at 10). |
| `--max-input-dimension <PX>` | Maximum source width or height (32768). |
| `--max-input-megapixels <MP>` | Maximum source area (32). |
| `--max-decode-mib <MIB>` | Best-effort decoder allocation limit (512 MiB). |

Enable the optional diagnostic build for `--verbose` (stage timings and JSON
summary) or `--quality-metrics` (completed-SVG OKLab/SSIM measurements):

```sh
mise exec -- cargo build --release --locked --features diagnostics
./target/release/picvec input.png output.svg --verbose --quality-metrics
```

Quality diagnostics report the evaluated width, height and composite background.
They compare the completed SVG at the processing resolution with the processing
reference. `global_ssim` covers the complete image; `local_ssim` averages uniform
7×7 windows on linear luminance (smaller odd windows for tiny images). The
`worst_tiles` list locates up to eight 64×64 tiles with the largest mean OKLab
error, including their maximum error and local SSIM. Coordinates refer to the
reported evaluation resolution. These RGB diagnostics use the reported backing.

Diagnostic output goes to stderr. Segmentation and geometry diagnostics describe
the base processing stages; the final object/contour counts describe the complete
emitted document.

## WebAssembly and browser converter

An English, entirely client-side image-to-SVG page lives in [`docs`](docs/README.md).
Build its Wasm module and serve it locally:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked
./scripts/build-wasm.sh
python3 -m http.server 8080 --directory docs
```

Open <http://localhost:8080>. PNG/JPEG conversion runs in a cancellable Web Worker
with previews, transparency support, optional chroma key removal, and SVG download.
See [the web documentation](docs/README.md) for browser limits, API usage, and deployment.

## Processing order

picvec rebuilds an image as editable shapes and lines, using the original as a
visual reference. It starts with the overall picture, then checks whether a
closer look can improve the details.

1. **Read the image.** Check that it can be processed and choose a working size.
   Large images may be reduced for the first pass.
2. **Find the parts of the picture.** Identify coloured areas, shading and thin
   lines. Fold tiny colour patches caused by softened pixel edges into nearby
   areas, while preserving details such as dots and highlights.
3. **Choose how to colour each part.** Represent its colour with a single fill
   or a gradual colour transition. Combine areas that can share the same fill
   without losing visible differences.
4. **Turn pixel edges into shapes.** Follow the visible outlines with smooth
   curves or simple shapes. Neighbouring shapes share an edge so they fit together.
5. **Arrange the layers.** Decide which shapes appear in front and let them
   overlap where needed to avoid gaps. Compare the result with the original
   before removing hidden shapes or lines.
6. **Check finer details.** If the first pass used a smaller image, revisit the
   original at a larger size using the same steps. Keep changes only when they
   improve the match, preserve edges and justify the added SVG size.
7. **Save the SVG.** Combine the accepted improvements, remove shapes they have
   replaced, and count and write the final drawing elements.

The detail check may revisit a small area or, when the image supports it, the
whole picture. Checking the whole picture helps keep connected lines and gradual
colour transitions consistent, but can take much longer. If no suitable
improvement is found, picvec keeps the first result. You can limit the extra file
size with `--adaptive-svg-budget-mib`.

### Transparency and visible geometry

- Transparency belongs to the colours of shapes and lines, including colours
  that gradually become more transparent.
- Soft pixel edges guide the shape's outline, so the SVG follows the visible
  form rather than tracing each pixel step.
- When replacing a small area with a more detailed version, picvec trims it to
  fit and checks that its edges blend with the surrounding picture.
- Shapes that make no visible difference are removed. Small or faint details
  are kept when they affect the picture, with checks at both the original size
  and an enlarged view that also account for transparency.

## Samples

`cliparts-6x6.png` uses the sample's explicit
`--remove-chroma-key-background` setting. Other samples use the normal defaults.

| Sample | Original | SVG |
| --- | --- | --- |
| Boy and turtle | [<img src="sample/input/boy_and_turtle.png" alt="Boy and turtle original" width="280">](sample/input/boy_and_turtle.png) | [<img src="sample/output/boy_and_turtle.svg" alt="Boy and turtle SVG" width="280">](sample/output/boy_and_turtle.svg) |
| Car | [<img src="sample/input/car.png" alt="Car original" width="280">](sample/input/car.png) | [<img src="sample/output/car.svg" alt="Car SVG" width="280">](sample/output/car.svg) |
| Viewport 1 | [<img src="sample/input/viewport1.jpg" alt="Viewport 1 original" width="280">](sample/input/viewport1.jpg) | [<img src="sample/output/viewport1.svg" alt="Viewport 1 SVG" width="280">](sample/output/viewport1.svg) |

| Input | Editable output | Comparison | Objects | Path contours |
| --- | --- | --- | ---: | ---: |
| [Booster layout](sample/input/booster-layout.jpg) | [SVG](sample/output/booster-layout.svg) | [PNG](sample/comparison/booster-layout.png) | 16,445 | 24,614 |
| [Boy and turtle](sample/input/boy_and_turtle.png) | [SVG](sample/output/boy_and_turtle.svg) | [PNG](sample/comparison/boy_and_turtle.png) | 80 | 128 |
| [Car](sample/input/car.png) | [SVG](sample/output/car.svg) | [PNG](sample/comparison/car.png) | 853 | 952 |
| [Cliparts](sample/input/cliparts.png) | [SVG](sample/output/cliparts.svg) | [PNG](sample/comparison/cliparts.png) | 4,692 | 14,060 |
| [Cliparts 6×6](sample/input/cliparts-6x6.png) | [SVG](sample/output/cliparts-6x6.svg) | [PNG](sample/comparison/cliparts-6x6.png) | 29,282 | 30,722 |
| [Still life](sample/input/vectorization-stress-still-life.png) | [SVG](sample/output/vectorization-stress-still-life.svg) | [PNG](sample/comparison/vectorization-stress-still-life.png) | 13,897 | 18,877 |
| [Viewport 1](sample/input/viewport1.jpg) | [SVG](sample/output/viewport1.svg) | [PNG](sample/comparison/viewport1.png) | 5,652 | 6,258 |
| [Viewport 2](sample/input/viewport2.jpg) | [SVG](sample/output/viewport2.svg) | [PNG](sample/comparison/viewport2.png) | 30,500 | 35,501 |
| [Wikipedia logo](sample/input/wikipedia_logo_1_0.png) | [SVG](sample/output/wikipedia_logo_1_0.svg) | [PNG](sample/comparison/wikipedia_logo_1_0.png) | 1,917 | 2,623 |
