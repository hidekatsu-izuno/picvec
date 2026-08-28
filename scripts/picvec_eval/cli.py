"""Standalone Real-ESRGAN x4 evaluation CLI for completed SVG files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from xml.etree import ElementTree as ET

from .svg_metrics import (
    svg_anchor_roughness,
    svg_complexity,
    svg_geometry_wobble,
    svg_open_stroke_roughness,
)

from .evaluation import EvaluationConfig, evaluate_x4_images, load_rgb, save_rgb
from .support import resize_image
from .upscale import generate_realesrgan_x4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable(value: str, purpose: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value)
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(f"{purpose} executable not found: {value}")
    return resolved


def _run(command: list[str], *, timeout: float, purpose: str) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"{purpose} failed: {detail}")


def _generate_reference(
    input_path: Path,
    output_path: Path,
    *,
    executable: str,
    model: str,
    model_path: Path | None,
    tile_size: int,
    tta: bool,
    timeout: float,
) -> list[str]:
    runner = _executable(executable, "Real-ESRGAN")
    with tempfile.TemporaryDirectory(
        prefix=".picvec-eval-realesrgan-",
        dir=output_path.parent,
    ) as directory:
        temporary = Path(directory) / "reference.png"
        command = [
            runner,
            "-i",
            str(input_path),
            "-o",
            str(temporary),
            "-n",
            model,
            "-s",
            "4",
            "-f",
            "png",
            "-t",
            str(max(0, int(tile_size))),
        ]
        if model_path is not None:
            command.extend(("-m", str(model_path)))
        if tta:
            command.append("-x")
        _run(command, timeout=timeout, purpose="Real-ESRGAN x4 inference")
        if not temporary.is_file():
            raise RuntimeError("Real-ESRGAN did not produce the requested PNG")
        temporary.replace(output_path)
    return command


def _render_svg(
    svg_path: Path,
    output_path: Path,
    width: int,
    height: int,
    *,
    executable: str,
    background: str,
    timeout: float,
) -> list[str]:
    renderer = _executable(executable, "librsvg")
    with tempfile.TemporaryDirectory(
        prefix=".picvec-eval-render-",
        dir=output_path.parent,
    ) as directory:
        temporary = Path(directory) / "rendered.png"
        command = [
            renderer,
            "--format=png",
            f"--width={width}",
            f"--height={height}",
            f"--background-color={background}",
            f"--output={temporary}",
            str(svg_path),
        ]
        _run(command, timeout=timeout, purpose="x4 SVG rendering")
        if not temporary.is_file():
            raise RuntimeError("rsvg-convert did not produce the requested PNG")
        temporary.replace(output_path)
    return command


def _svg_canvas_dimensions(svg_path: Path) -> tuple[int, int]:
    """Return the integral raster canvas represented by an SVG document."""

    root = ET.fromstring(svg_path.read_bytes())

    def numeric_length(name: str) -> float | None:
        raw = root.get(name)
        if raw is None:
            return None
        value = raw.strip()
        if value.endswith("px"):
            value = value[:-2].strip()
        try:
            return float(value)
        except ValueError:
            return None

    width = numeric_length("width")
    height = numeric_length("height")
    if width is None or height is None or width <= 0 or height <= 0:
        view_box = root.get("viewBox")
        if view_box is None:
            raise ValueError("SVG must have numeric width/height or a viewBox")
        values = view_box.replace(",", " ").split()
        if len(values) != 4:
            raise ValueError("SVG viewBox must contain four numbers")
        try:
            width = float(values[2])
            height = float(values[3])
        except ValueError as exc:
            raise ValueError("SVG viewBox must contain four numbers") from exc
    resolved_width = int(round(width))
    resolved_height = int(round(height))
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("SVG canvas dimensions must be positive")
    return resolved_width, resolved_height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "compare a Real-ESRGAN x4 reference with a completed SVG rendered at x4; "
            "the result never feeds back into vectorization"
        )
    )
    parser.add_argument("input", type=Path, help="raster source used for vectorization")
    parser.add_argument("svg", type=Path, help="completed SVG to evaluate")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--reference-x4",
        type=Path,
        help="use an existing x4 reference instead of invoking Real-ESRGAN",
    )
    parser.add_argument("--realesrgan", default="realesrgan-ncnn-vulkan")
    parser.add_argument("--model", default="realesrgan-x4plus-anime")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument(
        "--realesrgan-model",
        type=Path,
        help=(
            "official RealESRGAN_x4plus_anime_6B .pth; selects the "
            "SVGDeck-compatible PyTorch/Spandrel generator"
        ),
    )
    parser.add_argument("--tile-padding", type=int, default=16)
    parser.add_argument("--device", help="torch device for the PyTorch backend")
    parser.add_argument(
        "--realesrgan-fp32",
        action="store_true",
        help="disable CUDA fp16 in the PyTorch backend",
    )
    parser.add_argument(
        "--realesrgan-cache-dir",
        type=Path,
        default=Path(".cache/picvec/realesrgan"),
        help="content-addressed PyTorch x4 cache",
    )
    parser.add_argument(
        "--no-realesrgan-cache",
        action="store_true",
        help="disable the content-addressed PyTorch x4 cache",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--rsvg-convert", default="rsvg-convert")
    parser.add_argument("--background", default="#ffffff")
    parser.add_argument("--edge-sigma", type=float, default=1.2)
    parser.add_argument("--edge-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--extra-edge-tolerance",
        type=float,
        default=None,
        help="x4-pixel tolerance for false/extra edge components (default: primary tolerance)",
    )
    parser.add_argument(
        "--extra-edge-min-area",
        type=int,
        default=4,
        help="minimum x4-pixel area reported as a significant edge component",
    )
    parser.add_argument(
        "--edge-frame-margin",
        type=int,
        default=4,
        help="ignore this many x4 pixels at the clipped canvas frame",
    )
    parser.add_argument("--boundary-band-radius", type=float, default=8.0)
    parser.add_argument(
        "--thin-line-neighborhood",
        type=int,
        default=9,
        help="odd local window used to detect narrow dark strokes",
    )
    parser.add_argument(
        "--thin-line-contrast",
        type=float,
        default=0.045,
        help="minimum local luminance contrast for a thin-line pixel",
    )
    parser.add_argument(
        "--thin-line-tolerance",
        type=float,
        default=1.0,
        help="x4-pixel distance tolerance for thin-line recall/precision (default: 1)",
    )
    parser.add_argument(
        "--dark-core-luma-threshold",
        type=float,
        default=0.20,
        help="maximum Rec.709 luminance for an outline core (default: 0.20)",
    )
    parser.add_argument(
        "--structural-match-tolerance",
        type=float,
        default=1.25,
        help="native-pixel tolerance for one-to-one stroke matching",
    )
    parser.add_argument(
        "--structural-min-component-pixels",
        type=int,
        default=2,
        help="minimum component size retained by the topology metric",
    )
    parser.add_argument(
        "--structural-max-analysis-size",
        type=int,
        default=1024,
        help="maximum analysis dimension for native/x4 topology matching",
    )
    parser.add_argument(
        "--structural-score-floor",
        type=float,
        default=0.12,
        help="minimum geometric-mean structural topology score for acceptance",
    )
    parser.add_argument(
        "--structural-max-line-missing-fraction",
        type=float,
        default=0.45,
        help="maximum missing dark-line component fraction at either scale",
    )
    parser.add_argument(
        "--structural-max-line-duplicate-fraction",
        type=float,
        default=0.16,
        help="maximum duplicate dark-line component fraction at either scale",
    )
    parser.add_argument(
        "--structural-max-edge-duplicate-fraction",
        type=float,
        default=0.21,
        help="maximum duplicate luminance-edge component fraction at either scale",
    )
    parser.add_argument(
        "--worst-tile-size",
        type=int,
        default=256,
        help="local failure tile size in x4 pixels",
    )
    parser.add_argument(
        "--worst-tile-stride",
        type=int,
        default=128,
        help="local failure tile stride in x4 pixels",
    )
    parser.add_argument(
        "--complexity-target-per-edge",
        type=float,
        default=0.05,
        help=(
            "weighted SVG geometry units allowed per reference edge pixel "
            "for the higher-is-better quality score"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true", help="also print report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"source image not found: {args.input}")
    if not args.svg.is_file():
        raise FileNotFoundError(f"SVG not found: {args.svg}")
    if args.reference_x4 is not None and not args.reference_x4.is_file():
        raise FileNotFoundError(f"x4 reference not found: {args.reference_x4}")
    if args.model_path is not None and not args.model_path.exists():
        raise FileNotFoundError(f"Real-ESRGAN model path not found: {args.model_path}")
    if args.realesrgan_model is not None and not args.realesrgan_model.is_file():
        raise FileNotFoundError(
            f"Real-ESRGAN PyTorch model not found: {args.realesrgan_model}"
        )
    if args.realesrgan_model is not None and args.tta:
        raise ValueError("--tta is supported only by the NCNN Real-ESRGAN backend")

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_path = output_directory / "reference-realesrgan-x4.png"
    rendered_path = output_directory / "rendered-svg-x4.png"
    native_rendered_path = output_directory / "rendered-svg-native.png"
    original_source = load_rgb(args.input, background=args.background)
    canvas_width, canvas_height = _svg_canvas_dimensions(args.svg)
    source_was_resized = original_source.shape[:2] != (canvas_height, canvas_width)
    source = (
        resize_image(original_source, (canvas_height, canvas_width))
        if source_was_resized
        else original_source
    )
    processing_source_path = output_directory / "source-processing.png"
    save_rgb(source, processing_source_path)
    expected_shape = (source.shape[0] * 4, source.shape[1] * 4)

    realesrgan_command: list[str] | None = None
    realesrgan_backend = "provided-reference"
    realesrgan_generation: dict[str, Any] | None = None
    if args.reference_x4 is not None:
        reference = load_rgb(args.reference_x4, background=args.background)
        save_rgb(reference, reference_path)
    elif args.realesrgan_model is not None:
        pytorch_tile_size = max(32, args.tile_size if args.tile_size > 0 else 256)
        realesrgan_generation = generate_realesrgan_x4(
            source,
            reference_path,
            model_path=args.realesrgan_model,
            tile_size=pytorch_tile_size,
            tile_padding=max(0, args.tile_padding),
            use_half=not args.realesrgan_fp32,
            device=args.device,
            cache_dir=(
                None if args.no_realesrgan_cache else args.realesrgan_cache_dir
            ),
        )
        realesrgan_backend = "pytorch-spandrel"
        reference = load_rgb(reference_path, background=args.background)
    else:
        realesrgan_backend = "ncnn-vulkan"
        realesrgan_command = _generate_reference(
            processing_source_path,
            reference_path,
            executable=args.realesrgan,
            model=args.model,
            model_path=args.model_path,
            tile_size=args.tile_size,
            tta=args.tta,
            timeout=max(1.0, args.timeout),
        )
        reference = load_rgb(reference_path, background=args.background)
    if reference.shape[:2] != expected_shape:
        raise ValueError(
            "Real-ESRGAN reference must be exactly x4: "
            f"got {reference.shape[1]}x{reference.shape[0]}, "
            f"expected {expected_shape[1]}x{expected_shape[0]}"
        )

    render_command = _render_svg(
        args.svg,
        rendered_path,
        reference.shape[1],
        reference.shape[0],
        executable=args.rsvg_convert,
        background=args.background,
        timeout=max(1.0, args.timeout),
    )
    rendered = load_rgb(rendered_path, background=args.background)
    native_render_command = _render_svg(
        args.svg,
        native_rendered_path,
        source.shape[1],
        source.shape[0],
        executable=args.rsvg_convert,
        background=args.background,
        timeout=max(1.0, args.timeout),
    )
    native_rendered = load_rgb(native_rendered_path, background=args.background)
    config = EvaluationConfig(
        scale=4,
        edge_sigma=max(0.1, args.edge_sigma),
        primary_edge_tolerance=max(0.0, args.edge_tolerance),
        boundary_band_radius=max(0.0, args.boundary_band_radius),
        extra_edge_tolerance=(
            None
            if args.extra_edge_tolerance is None
            else max(0.0, args.extra_edge_tolerance)
        ),
        extra_edge_min_component_area=max(1, args.extra_edge_min_area),
        edge_frame_margin=max(0, args.edge_frame_margin),
        thin_line_neighborhood=max(3, args.thin_line_neighborhood),
        thin_line_contrast=max(0.0, args.thin_line_contrast),
        thin_line_tolerance=max(0.0, args.thin_line_tolerance),
        dark_core_luma_threshold=min(1.0, max(0.0, args.dark_core_luma_threshold)),
        structural_match_tolerance=max(0.0, args.structural_match_tolerance),
        structural_min_component_pixels=max(1, args.structural_min_component_pixels),
        structural_max_analysis_size=max(64, args.structural_max_analysis_size),
        structural_score_floor=min(1.0, max(0.0, args.structural_score_floor)),
        structural_max_line_missing_fraction=min(
            1.0, max(0.0, args.structural_max_line_missing_fraction)
        ),
        structural_max_line_duplicate_fraction=min(
            1.0, max(0.0, args.structural_max_line_duplicate_fraction)
        ),
        structural_max_edge_duplicate_fraction=min(
            1.0, max(0.0, args.structural_max_edge_duplicate_fraction)
        ),
        worst_tile_size=max(8, args.worst_tile_size),
        worst_tile_stride=max(1, args.worst_tile_stride),
        complexity_target_units_per_edge=max(1e-9, args.complexity_target_per_edge),
    )
    complexity = svg_complexity(args.svg.read_bytes()).as_dict()
    geometry_wobble = svg_geometry_wobble(args.svg.read_bytes())
    anchor_roughness = svg_anchor_roughness(args.svg.read_bytes())
    open_stroke_roughness = svg_open_stroke_roughness(args.svg.read_bytes())
    metrics_report = evaluate_x4_images(
        reference,
        rendered,
        output_directory,
        config=config,
        svg_complexity=complexity,
        geometry_wobble=geometry_wobble,
        anchor_roughness=anchor_roughness,
        open_stroke_roughness=open_stroke_roughness,
        pixel_reference=source,
        native_rendered=native_rendered,
    )
    report: dict[str, Any] = {
        "evaluation_only": True,
        "input": str(args.input),
        "svg": str(args.svg),
        "input_sha256": _sha256(args.input),
        "evaluation_source": str(processing_source_path),
        "evaluation_source_sha256": _sha256(processing_source_path),
        "source_was_resized_to_svg_canvas": source_was_resized,
        "svg_sha256": _sha256(args.svg),
        "reference_x4": str(reference_path),
        "reference_x4_sha256": _sha256(reference_path),
        "rendered_x4": str(rendered_path),
        "rendered_x4_sha256": _sha256(rendered_path),
        "rendered_native": str(native_rendered_path),
        "rendered_native_sha256": _sha256(native_rendered_path),
        "realesrgan": {
            "backend": realesrgan_backend,
            "model": (
                "RealESRGAN_x4plus_anime_6B"
                if realesrgan_backend == "pytorch-spandrel"
                else args.model
            ),
            "model_path": str(args.model_path) if args.model_path is not None else None,
            "pytorch_model_path": (
                str(args.realesrgan_model)
                if realesrgan_backend == "pytorch-spandrel"
                else None
            ),
            "source": (
                str(args.reference_x4)
                if args.reference_x4 is not None
                else "generated"
            ),
            "command": realesrgan_command,
            "generation": realesrgan_generation,
            "tile_size": (
                max(32, args.tile_size if args.tile_size > 0 else 256)
                if realesrgan_backend == "pytorch-spandrel"
                else max(0, int(args.tile_size))
            ),
            "tile_padding": (
                max(0, int(args.tile_padding))
                if realesrgan_backend == "pytorch-spandrel"
                else None
            ),
            "tta": bool(args.tta),
        },
        "renderer": {
            "background": args.background,
            "command": render_command,
            "native_command": native_render_command,
        },
        "metrics": metrics_report,
    }
    report_path = output_directory / "report.json"
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    report_path.write_text(serialized, encoding="utf-8")
    if args.json:
        print(serialized, end="")
    else:
        primary = metrics_report["boundary"]["primary"]
        distance = metrics_report["boundary"]["symmetric_distance"]
        extra = metrics_report["boundary"]["extra_edge_components"]
        worst_tile = metrics_report["local_failures"]["worst_tile"]
        worst_tile_p95 = (
            worst_tile["boundary_p95_source_pixels"] if worst_tile else None
        )
        selection = metrics_report.get("selection") or {}
        print(
            f"wrote {report_path} "
            f"(edge F1={primary['f1']:.4f} at "
            f"{primary['tolerance_source_pixels']:.2f} source px; "
            f"p95 distance={distance['p95_source_pixels']}; "
            f"extra-largest={extra['largest_source_pixels']:.2f} source px; "
            f"worst-tile-p95={worst_tile_p95}; "
            f"raster={selection.get('raster_score', metrics_report['quality']['score']):.4f}; "
            f"valid={selection.get('valid', True)})"
        )
    return 0
