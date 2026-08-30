#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_directory=${1:-"$repository_root/sample/comparison"}

if command -v magick >/dev/null 2>&1; then
    image_command=(magick)
elif command -v convert >/dev/null 2>&1; then
    image_command=(convert)
else
    echo "ImageMagick (magick or convert) is required" >&2
    exit 1
fi
if ! command -v rsvg-convert >/dev/null 2>&1; then
    echo "rsvg-convert is required by this optional comparison generator (not by picvec)" >&2
    exit 1
fi

mkdir -p -- "$output_directory"
working_directory=$(mktemp -d)
trap 'rm -rf -- "$working_directory"' EXIT
export XDG_CACHE_HOME="$working_directory/cache"
mkdir -p -- "$XDG_CACHE_HOME"

pairs=(
    "boy_and_turtle|Boy and turtle|boy_and_turtle.png|boy_and_turtle.svg"
    "car|Car|car.png|car.svg"
    "cliparts|Clip art|cliparts.png|cliparts.svg"
    "viewport1|Viewport 1|viewport1.jpg|viewport1.svg"
    "viewport2|Viewport 2|viewport2.jpg|viewport2.svg"
)

for specification in "${pairs[@]}"; do
    IFS='|' read -r stem title input_name output_name <<<"$specification"
    input_path="$repository_root/sample/input/$input_name"
    svg_path="$repository_root/sample/output/$output_name"
    rendered_path="$working_directory/$stem-rendered.png"
    input_panel="$working_directory/$stem-input.png"
    output_panel="$working_directory/$stem-output.png"
    comparison_path="$output_directory/$stem.png"

    rsvg-convert \
        --background-color white \
        --output "$rendered_path" \
        "$svg_path"

    "${image_command[@]}" "$input_path" \
        -auto-orient \
        -background white \
        -alpha remove \
        -alpha off \
        -resize '578x528>' \
        -gravity center \
        -extent 598x548 \
        -bordercolor '#d0d7de' \
        -border 1x1 \
        +repage \
        "$input_panel"

    "${image_command[@]}" "$rendered_path" \
        -background white \
        -alpha remove \
        -alpha off \
        -resize '578x528>' \
        -gravity center \
        -extent 598x548 \
        -bordercolor '#d0d7de' \
        -border 1x1 \
        +repage \
        "$output_panel"

    "${image_command[@]}" \
        -size 1280x720 \
        'xc:#f6f8fa' \
        -font DejaVu-Sans \
        -fill '#24292f' \
        -gravity North \
        -pointsize 30 \
        -annotate +0+24 "$title" \
        -gravity NorthWest \
        -pointsize 22 \
        -annotate +24+92 'Raster input' \
        -annotate +656+92 'Rendered SVG output' \
        "$input_panel" -geometry +24+145 -composite \
        "$output_panel" -geometry +656+145 -composite \
        -strip \
        -define png:compression-level=9 \
        "PNG24:$comparison_path"
done
