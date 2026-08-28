from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from generate_realesrgan_x4 import build_parser as build_generator_parser
from picvec_eval.cli import build_parser as build_evaluator_parser
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


if __name__ == "__main__":
    unittest.main()
