from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from generate_realesrgan_x4 import build_parser as build_generator_parser
from picvec_eval.cli import (
    _write_text_atomic,
    build_parser as build_evaluator_parser,
    main as evaluate_main,
)
from picvec_eval.upscale import (
    CachedImageUpscaler,
    RealESRGANUpscaler,
    generate_realesrgan_x4,
)


class RepeatX4Upscaler:
    scale = 4

    def __init__(self) -> None:
        self.calls = 0

    def upscale(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)


class RealESRGANGenerationTests(unittest.TestCase):
    def test_upscaler_import_does_not_load_evaluator_dependencies(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import picvec_eval.upscale; "
                    "assert 'scipy' not in sys.modules; "
                    "assert 'skimage' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_content_addressed_cache_reuses_identical_source(self) -> None:
        inner = RepeatX4Upscaler()
        source = np.full((8, 10, 3), 0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            cached = CachedImageUpscaler(inner, directory, "test-model")
            first = cached.upscale(source)
            first_path = cached.last_cache_path
            second = cached.upscale(source.copy())
            changed = cached.upscale(np.full_like(source, 0.5))

        self.assertEqual(inner.calls, 2)
        self.assertIsNotNone(first_path)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, changed))

    def test_standalone_generator_writes_exact_x4_and_hits_cache(self) -> None:
        source = np.linspace(0.0, 1.0, 7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
        calls = 0

        def repeat(upscaler: RealESRGANUpscaler, image: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pth"
            model.write_bytes(b"deterministic-test-model")
            cache = root / "cache"
            first_path = root / "first.png"
            second_path = root / "second.png"
            with mock.patch.object(RealESRGANUpscaler, "upscale", repeat):
                first = generate_realesrgan_x4(
                    source,
                    first_path,
                    model_path=model,
                    device="cpu",
                    cache_dir=cache,
                )
                second = generate_realesrgan_x4(
                    source.copy(),
                    second_path,
                    model_path=model,
                    device="cpu",
                    cache_dir=cache,
                )

            self.assertEqual(calls, 1)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            with Image.open(first_path) as generated:
                self.assertEqual(generated.size, (36, 28))

    def test_both_clis_expose_pytorch_reproducibility_controls(self) -> None:
        generator = build_generator_parser().parse_args(
            ["input.png", "x4.png", "--model", "model.pth", "--device", "cpu"]
        )
        evaluator = build_evaluator_parser().parse_args(
            [
                "input.png",
                "output.svg",
                "evaluation",
                "--realesrgan-model",
                "model.pth",
                "--tile-padding",
                "24",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(generator.model, Path("model.pth"))
        self.assertEqual(evaluator.realesrgan_model, Path("model.pth"))
        self.assertEqual(evaluator.tile_padding, 24)
        self.assertEqual(evaluator.device, "cpu")

    def test_atomic_text_writer_replaces_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("old", encoding="utf-8")
            _write_text_atomic(output, "new\n")

            self.assertEqual(output.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(list(output.parent.glob(".report-*")), [])

    def test_evaluator_uses_original_dimensions_for_native_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            svg = root / "output.svg"
            reference = root / "reference.png"
            output = root / "evaluation"
            Image.new("RGB", (8, 6), "white").save(source)
            Image.new("RGB", (16, 12), "white").save(reference)
            svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="3" '
                'viewBox="0 0 4 3"><rect width="4" height="3" fill="white"/></svg>',
                encoding="utf-8",
            )

            def render(
                _svg_path: Path,
                output_path: Path,
                width: int,
                height: int,
                **_kwargs: object,
            ) -> list[str]:
                Image.new("RGB", (width, height), "white").save(output_path)
                return ["fake-rsvg-convert", str(width), str(height)]

            def evaluate(
                _reference: np.ndarray,
                _rendered: np.ndarray,
                _output_directory: Path,
                **kwargs: object,
            ) -> dict[str, bool]:
                self.assertEqual(kwargs["pixel_reference"].shape[:2], (6, 8))
                self.assertEqual(kwargs["native_rendered"].shape[:2], (6, 8))
                return {"valid": True}

            with (
                mock.patch("picvec_eval.cli._render_svg", side_effect=render),
                mock.patch("picvec_eval.cli.evaluate_x4_images", side_effect=evaluate),
                mock.patch("picvec_eval.cli._command_version", return_value="test"),
                mock.patch("builtins.print"),
            ):
                result = evaluate_main(
                    [
                        str(source),
                        str(svg),
                        str(output),
                        "--reference-x4",
                        str(reference),
                        "--json",
                    ]
                )

            self.assertEqual(result, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["original_source_width"], 8)
            self.assertEqual(report["original_source_height"], 6)
            self.assertEqual(report["processing_source_width"], 4)
            self.assertEqual(report["processing_source_height"], 3)


if __name__ == "__main__":
    unittest.main()
