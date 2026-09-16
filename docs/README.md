# Browser converter

The English site in this directory converts PNG and JPEG files to SVG locally in
an ES module Web Worker. No upload service, bundler, CDN, or cross-origin isolation
headers are required. Serve it over HTTP or HTTPS, not `file://`.

## Build and serve

From the repository root, with the Rust version in `mise.toml` active:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked
./scripts/build-wasm.sh
python3 -m http.server 8080 --directory docs
```

Open <http://localhost:8080>. The build writes `docs/pkg/picvec.js`,
`picvec_bg.wasm`, and TypeScript declarations. These generated files are committed
to Git so the checked-in `docs` directory is ready to serve. Rebuild them after
changing Rust code. The CLI version must match the pinned
`wasm-bindgen` dependency in `Cargo.toml`.

Run the built-module smoke test with Node.js 22 or later:

```sh
node scripts/test-wasm.mjs
```

There are two repository workflows:

- **Update web Wasm** (manual, default branch only): builds and smoke-tests Wasm,
  commits changes under `docs/pkg`, and requests a GitHub Pages rebuild. Configure
  **Settings → Pages → Build and deployment → Source → Deploy from a branch**,
  select the default branch (usually `main`), and choose **/docs**. The workflow
  needs permission to push to that branch; it does not bypass branch protection.
- **Release binaries** (manual): builds the six native packages and a
  `picvec-v<VERSION>-wasm32-unknown-unknown.tar.gz` archive from the same release
  tag, then publishes all seven archives with checksums. The Wasm archive contains
  the browser page, JavaScript bindings, Wasm module, and TypeScript declarations.
  Extract it and serve the extracted directory with any static HTTP server.

The release workflow publishes downloadable assets; use **Update web Wasm** to
update the hosted page. The explicit rebuild request is needed because commits
made with `GITHUB_TOKEN` do not automatically trigger branch-based Pages builds
([GitHub documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)).
GitHub's built-in **pages build and deployment** may still appear in Actions;
there is no separate Pages deployment workflow in this repository.

## Browser limits

- PNG/JPEG, up to 20 MiB, 16 million source pixels, and 8,192 pixels per side.
- Processing sizes of 256, 512 (default), 768, or 1,024 pixels on the longest side.
- Single-threaded conversion in a dedicated worker; Cancel terminates the worker.
- Source-resolution adaptive refinement is disabled to bound browser work.
- Source alpha is preserved; saturated chroma key removal is optional.
- Complex images can still take minutes or exceed a device's available memory.
  Try a smaller processing size if conversion fails.

The original preview is displayed after the Rust decoder validates and converts
the image, so selecting a file does not trigger an unbounded browser image decode.

## API

```js
import init, { convert_image } from './pkg/picvec.js';
await init();
const svg = convert_image(new Uint8Array(encodedImage), 512, false);
```

Call this synchronous API inside a worker, as `worker.js` does. Arguments are
encoded PNG/JPEG bytes, maximum processing dimension (64–1024), and whether to
remove a chroma key background. It returns SVG text or throws an error.

Rust callers can use `picvec::vectorize_bytes(&bytes, &config)` for in-memory
conversion on either target. It returns `(String, Summary)`; `Summary::output` is
empty. The file-based `picvec::vectorize` and CLI remain available on native targets.
