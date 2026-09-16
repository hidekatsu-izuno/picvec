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
