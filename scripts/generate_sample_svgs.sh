#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repository_root"
inputs=("$@")
if (( ${#inputs[@]} == 0 )); then
    for path in sample/input/*; do
        [[ -f "$path" ]] && inputs+=("${path##*/}")
    done
fi
for name in "${inputs[@]}"; do
    if [[ "$name" != "${name##*/}" || ! -f "sample/input/$name" ]]; then
        echo "Expected a file name from sample/input: $name" >&2
        exit 1
    fi
done
cargo build --release --locked --features diagnostics
mkdir -p sample/output
working_directory=$(mktemp -d "$repository_root/sample/output/.regenerate-XXXXXX")
trap 'rm -rf -- "$working_directory"' EXIT
for name in "${inputs[@]}"; do
    options=(--threads 4 --verbose)
    # This sheet historically removes its green backing. Keying also separates
    # the grid from touching figures before source-resolution refinement.
    if [[ "$name" == cliparts-6x6.png ]]; then
        options+=(--remove-chroma-key-background)
    fi
    output_name="${name%.*}.svg"
    target/release/picvec "sample/input/$name" "$working_directory/$output_name" "${options[@]}"
    mv -- "$working_directory/$output_name" "sample/output/$output_name"
done
