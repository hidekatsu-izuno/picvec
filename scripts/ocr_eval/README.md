# Fixed-region OCR evaluation

This tool uses [ocrs-cjk](https://github.com/kent-tokyo/ocrs-cjk) **only for
evaluation**. Its standalone Cargo package does not add OCR, models, downloads
or character recognition to picvec, its CLI, or its WASM build.

The detector runs on overlapping source-image tiles at native resolution.
Recognition uses each detected text run independently, avoiding accidental
joining of distant diagram labels. The exact same rotated source rectangles
are recognized in every candidate. Missing candidate text counts as deletion;
it cannot disappear from the evaluation through failed redetection.

## Run

Supply PP-OCRv5 detection and recognition ONNX models and the matching alphabet
explicitly. The evaluator performs no downloads. Build the pinned OCR library:

```sh
mise exec -- cargo build --release --locked --manifest-path scripts/ocr_eval/Cargo.toml
```

Render **complete** old and new SVGs at the source dimensions, then create a
configuration file (paths are relative to the working directory):

```json
{
  "source": "sample/input/catwhale.png",
  "detection_model": "/path/to/detection.onnx",
  "recognition_model": "/path/to/recognition.onnx",
  "alphabet": "/path/to/alphabet.txt",
  "candidates": {
    "before": "/tmp/before.png",
    "after": "/tmp/after.png"
  },
  "tile_size": 1024,
  "max_height": 64,
  "output": "/tmp/ocr-regions.json"
}
```

```sh
rsvg-convert --width 1370 --height 1148 before.svg --output /tmp/before.png
rsvg-convert --width 1370 --height 1148 after.svg --output /tmp/after.png
RAYON_NUM_THREADS=4 scripts/ocr_eval/target/release/picvec-ocr-eval config.json
mise exec -- uv run scripts/ocr_eval/report.py /tmp/ocr-regions.json /tmp/ocr-report
```

The report includes every source detection in RGB and silhouette metrics,
including detections whose source transcription is empty. Character metrics
use nonempty source transcriptions, normalize Unicode width and whitespace,
and retain empty candidate results. `source_ocr_disagreement` is edit distance
divided by source OCR character count, **not accuracy against human ground
truth**. The source itself contains recognition errors. `ink_iou` compares
thresholded silhouettes using a threshold and polarity determined only from
the source crop. `rgb_mae` is the mean absolute RGB error on the 0–255 scale.
Review individual rows as well as aggregates; drawings can be false positives.

The self-contained HTML shows native-resolution crops enlarged for inspection.
Also render the SVG itself at enlarged resolution to check its actual curves;
enlarging a raster preview cannot do that. Test non-text and RGBA examples
separately. No OCR bounds or strings should be used to tune individual image
locations inside picvec.

## Model provenance for the sample evaluation

- ocrs-cjk revision: `2f72f76ac2152ee1112ee6b6e36048bab069f391`.
- Models: `PP-OCRv5_server_det_infer.onnx` and
  `PP-OCRv5_server_rec_infer.onnx` from
  [marsena/paddleocr-onnx-models](https://huggingface.co/marsena/paddleocr-onnx-models).
- Alphabet: concatenate the first character of each `character_dict` entry in
  the matching `PP-OCRv5_server_rec_infer.yml`, then append one space, without
  an extra newline (18,384 characters).
- SHA-256, detection: `127edf0182bb3d218ad59476377b02ca90296cfb4cc85df55042d671a3e53aeb`.
- SHA-256, recognition: `13d0dda27d63dc0f4938af48df2c55b33f3c989a0bd5eacb8410e30f1735f644`.
- SHA-256, alphabet: `383f570e2ac46418c2150c9c77b7a6f5b650616080167f9e936da0b87821b4ee`.

Models are not redistributed in this repository.
