# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pillow>=10",
#   "spandrel==0.4.2",
# ]
# ///
"""Generate a standalone Real-ESRGAN x4 PNG for picvec evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from picvec_eval.upscale import generate_realesrgan_x4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source RGB PNG/JPEG")
    parser.add_argument("output", type=Path, help="exact x4 PNG to create")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="official RealESRGAN_x4plus_anime_6B .pth model",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-padding", type=int, default=16)
    parser.add_argument("--device", help="torch device, for example cpu or cuda")
    parser.add_argument("--fp32", action="store_true", help="disable CUDA fp16")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/picvec/realesrgan"),
        help="content-addressed PNG cache",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"source image not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Real-ESRGAN model not found: {args.model}")
    source = np.asarray(Image.open(args.input).convert("RGB"))
    report = generate_realesrgan_x4(
        source,
        args.output,
        model_path=args.model,
        tile_size=max(32, args.tile_size),
        tile_padding=max(0, args.tile_padding),
        use_half=not args.fp32,
        device=args.device,
        cache_dir=None if args.no_cache else args.cache_dir,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        cache = "hit" if report["cache_hit"] else "generated"
        print(f"wrote {args.output} ({report['width']}x{report['height']}, {cache})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
