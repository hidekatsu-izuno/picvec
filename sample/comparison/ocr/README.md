# Small-detail evaluation

The subsequent [all-sample regression check](../regression/README.md) found
local quality regressions, including non-text details. The improvements below
must not be interpreted as a guarantee that other regions are unchanged.

Baseline: original pipeline at `c659d51`, using the committed sample SVGs.
The new outputs are [catwhale.svg](../../output/catwhale.svg) and
[booster-layout.svg](../../output/booster-layout.svg).
OCR is restricted to the standalone [evaluation tool](../../../scripts/ocr_eval/README.md);
picvec preserves source-resolution thin silhouettes without recognizing text.

| Sample | Source OCR disagreement ↓ | Ink IoU ↑ | RGB MAE ↓ |
| --- | ---: | ---: | ---: |
| catwhale | 32.5% → 17.2% | 0.746 → 0.847 | 9.79 → 6.84 |
| booster-layout | 48.0% → 34.3% | 0.673 → 0.741 | 31.32 → 25.33 |

The detector runs only on the source. Complete SVGs are rendered at source
dimensions before evaluating the same fixed regions in both candidates.
Catwhale has 41 detected regions, including 35 nonempty source transcriptions
(203 characters); booster-layout has 54 regions, including 33 nonempty
transcriptions (306 characters). Pixel metrics include all detections.
Source OCR disagreement compares against imperfect source OCR, not human
ground truth. False detections and individual regressions remain in the report.

- [Catwhale region comparison](catwhale.html) / [metrics and bounds](catwhale-metrics.json)
- [Booster-layout region comparison](booster-layout.html) / [metrics and bounds](booster-layout-metrics.json)
- Actual SVG renders at 4×: [catwhale](catwhale-4x.png), [booster-layout](booster-layout-4x.png).
  Each figure shows source raster enlarged, baseline SVG, then new SVG.

Small holes and thin strokes improve, but some letters on complex backgrounds
and distorted engineering annotations still collapse. The aggregate improvement
does not mean every region improves. The HTML previews enlarge native raster
renders; the separate 4× figures render the vector curves at higher resolution.

Catwhale SVG size changes from 2,367,164 to 2,211,240 bytes; booster-layout grows
from 7,170,075 to 8,667,557 bytes (+20.9%). Wikipedia-logo and car control outputs
are byte-identical to their baseline SVGs, with no extracted compact silhouettes.

Validation: 473 Rust library tests and 2 integration tests pass (10 existing
tests ignored), plus WASM smoke tests. The two emitted sample SVGs contain no
masks, embedded rasters, text elements, group opacity, or zero-alpha faces.
Regression tests cover actual small Japanese lettering, engineering annotations,
both ink polarities, counters, adjacent colored marks, and rejection of broad
objects or nonuniform backgrounds.
