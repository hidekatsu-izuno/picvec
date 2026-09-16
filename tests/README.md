# Tests

- `unit/`: Rust unit tests and test-only helpers, grouped by source module.
  Each source module includes its corresponding file only under `cfg(test)`.
  This preserves access to private implementation details without exposing
  additional production APIs.
- `reference/`: Independent reference implementations used by regression tests.
- `data/`: Images, contours and labels used by regression tests and benchmarks.
- Top-level Rust files: Integration tests.

Run the Rust tests with:

```sh
mise exec -- cargo test --locked --all-features
```

Keep new test bodies, helpers and fixtures here rather than under `src/`.
The Rust package includes the Rust test files and fixtures so the same tests
remain available after packaging.

Development tools and manual benchmarks live under `scripts/`. They may use
Python; the Rust test suite does not require Python or those tools.
