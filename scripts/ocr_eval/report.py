# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pillow>=10"]
# ///
"""Measure fixed source OCR rectangles and write an HTML visual comparison.

Usage: uv run scripts/ocr_eval/report.py evaluation.json output_directory
The JSON is produced by the separate ocr_eval Rust executable. Candidate PNGs
must be renders of complete SVG documents at the original raster dimensions.
"""
import base64
import html
import io
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np
from PIL import Image


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = next_row
    return row[-1]


def threshold(source):
    """Otsu threshold fitted once on the source, reused by every candidate."""
    values = source.mean(axis=2).astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    weight = hist.cumsum()
    weighted_sum = (hist * np.arange(256)).cumsum()
    other = weight[-1] - weight
    valid = (weight > 0) & (other > 0)
    variance = np.zeros(256)
    variance[valid] = (weighted_sum[-1] * weight[valid] - weighted_sum[valid] * weight[-1]) ** 2 / (weight[valid] * other[valid])
    return int(variance.argmax())


def inline_image(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    config = data["config"]
    source = Image.open(config["source"]).convert("RGB")
    candidates = {name: Image.open(path).convert("RGB") for name, path in config["candidates"].items()}
    for im in candidates.values():
        if im.size != source.size:
            raise ValueError("Candidate dimensions must equal source dimensions")
    totals = {name: dict(error=0, characters=0, exact=0, recognized_regions=0, ink_intersection=0,
                        ink_union=0, absolute_rgb_error=0, samples=0) for name in candidates}
    rows = []
    html_rows = []
    for idx, region in enumerate(data["regions"]):
        l, t, r, b = region["bbox"]
        rect = (max(0, int(np.floor(l)) - 2), max(0, int(np.floor(t)) - 2),
                min(source.width, int(np.ceil(r)) + 2), min(source.height, int(np.ceil(b)) + 2))
        crop = source.crop(rect)
        rgb = np.asarray(crop).astype(float)
        cut = threshold(rgb)
        ink = rgb.mean(axis=2) <= cut
        invert = ink.mean() > 0.5
        if invert:
            ink = ~ink
        reference = normalize(region["reference"])
        row = dict(id=idx, bbox=rect, reference=region["reference"], variants={})
        cells = [f'<td>{idx}: {html.escape(region["reference"])}<br>{inline_image(crop)}</td>']
        for name, image in candidates.items():
            rendered = image.crop(rect)
            pred = np.asarray(rendered).astype(float)
            foreground = pred.mean(axis=2) <= cut
            if invert:
                foreground = ~foreground
            intersection = int((foreground & ink).sum())
            union = int((foreground | ink).sum())
            error = float(np.abs(rgb - pred).sum())
            text = region["texts"][name]
            edits = distance(reference, normalize(text)) if reference else None
            value = dict(text=text, edit_distance=edits, ink_iou=intersection / max(union, 1), rgb_mae=error / rgb.size)
            row["variants"][name] = value
            total = totals[name]
            if reference:
                total["error"] += edits
                total["characters"] += len(reference)
                total["exact"] += int(edits == 0)
                total["recognized_regions"] += 1
            total["ink_intersection"] += intersection
            total["ink_union"] += union
            total["absolute_rgb_error"] += error
            total["samples"] += rgb.size
            cells.append(f'<td>{html.escape(text)}<br>{inline_image(rendered)}<br>IoU {value["ink_iou"]:.3f}, MAE {value["rgb_mae"]:.2f}</td>')
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
        rows.append(row)
    for total in totals.values():
        total["source_ocr_disagreement"] = total["error"] / max(1, total["characters"])
        total["ink_iou"] = total["ink_intersection"] / max(1, total["ink_union"])
        total["rgb_mae"] = total["absolute_rgb_error"] / max(1, total["samples"])
    report = dict(source=config["source"], image_size=list(source.size),
                  ocrs_cjk_revision=data["ocrs_cjk_revision"], regions=len(rows), totals=totals, details=rows)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    headings = "<th>source</th>" + "".join(f"<th>{html.escape(n)}</th>" for n in candidates)
    page = ('<!doctype html><meta charset="utf-8"><title>Fixed source OCR regions</title>'
            '<style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #aaa;vertical-align:top}'
            'img{display:block;image-rendering:pixelated;zoom:3}pre{white-space:pre-wrap}</style>'
            '<h1>Fixed source OCR regions</h1><p>OCR is an imperfect reference, not ground truth. '
            'All detected source regions remain in image metrics; empty source transcriptions have no character score. '
            'Images show native-resolution renders enlarged 3×; inspect enlarged SVG rendering separately.</p>'
            + '<pre>' + html.escape(json.dumps(totals, indent=2)) + '</pre><table><tr>' + headings + '</tr>'
            + "".join(html_rows) + '</table>')
    (out / "comparison.html").write_text(page)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
