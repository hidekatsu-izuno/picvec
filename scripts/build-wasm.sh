#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
version=0.2.127
if ! command -v wasm-bindgen >/dev/null || [[ "$(wasm-bindgen --version)" != "wasm-bindgen $version" ]]; then
  echo "Install the matching bindings generator: cargo install wasm-bindgen-cli --version $version --locked" >&2
  exit 1
fi
cargo build --release --locked --lib --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir docs/pkg --out-name picvec "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/release/picvec.wasm"
cp LICENSE docs/pkg/LICENSE
# Keep the bindings and binary on the same cache version after a Pages update.
if command -v sha256sum >/dev/null; then
  version_hash=$(cat docs/pkg/picvec.js docs/pkg/picvec_bg.wasm | sha256sum)
else
  version_hash=$(cat docs/pkg/picvec.js docs/pkg/picvec_bg.wasm | shasum -a 256)
fi
printf '{"version":"%s"}\n' "${version_hash%% *}" > docs/pkg/version.json
