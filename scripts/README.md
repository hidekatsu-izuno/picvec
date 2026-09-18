# Development tools

Keep evaluation tools, generators and manual benchmarks in this directory.
Automated Rust regression tests and their fixtures live in `tests/`.

- `build-wasm.sh` and `test-wasm.mjs`: Build and smoke-test the browser converter.
- `generate_sample_svgs.sh` and `generate_sample_comparisons.sh`: Regenerate samples.
- `benchmark_exact.py`: Compare two executables for exact output and timing.
- `benchmark_committed_svgs.py`: Compare generated samples with a Git revision.
- `compare_oklab.py`: Sweep palette thresholds.
- `evaluate.py`, `generate_realesrgan_x4.py` and `picvec_eval/`: Optional image
  evaluation tools; see [the evaluator documentation](picvec_eval/README.md).
- `check_cliparts_browser.mjs` and `check_viewport_browser.mjs`: Browser rendering
  checks for generated SVGs.
- `color_batch_performance.rs`: Manual colour conversion scheduling benchmark.
  Run alone for meaningful timings:

  ```sh
  mise exec -- cargo run --release --locked --example color_batch_performance
  ```

- `ocr_eval/`: Optional [ocrs-cjk evaluation](ocr_eval/README.md) of fixed
  source text regions. OCR and its models are not dependencies of picvec.

Python and `uv` remain available for the optional Python tools. They are not
required to run the Rust test suite.
