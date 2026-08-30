"""PyTorch/Spandrel Real-ESRGAN x4 generation with deterministic caching.

This is the evaluation-only upscaler formerly used by SVGDeck.  Nothing in
the Rust vectorizer imports this module or observes the generated x4 image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

import numpy as np

from .support import FloatImage, normalize_image


class ImageUpscaler(Protocol):
    """A replaceable image-to-image model with an integer output scale."""

    @property
    def scale(self) -> int: ...

    def upscale(self, image: FloatImage) -> FloatImage: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rgb_png(image: FloatImage, output_path: Path) -> None:
    from PIL import Image

    raster = np.rint(normalize_image(image) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}-",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        Image.fromarray(raster, mode="RGB").save(temporary, format="PNG")
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_atomic(source: Path, output_path: Path) -> None:
    if source.resolve() == output_path.resolve():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}-",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class CachedImageUpscaler:
    """Cache deterministic x4 evidence as a content-addressed lossless PNG."""

    inner: ImageUpscaler
    cache_dir: str | Path
    namespace: str
    cache_hit: bool = field(default=False, init=False)
    last_cache_path: Path | None = field(default=None, init=False)

    @property
    def scale(self) -> int:
        return self.inner.scale

    def cache_path(self, image: FloatImage) -> Path:
        source = normalize_image(image)
        digest = hashlib.sha256()
        digest.update(b"picvec-realesrgan-cache-v2\0")
        digest.update(self.namespace.encode("utf-8"))
        digest.update(np.asarray(source.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(source, dtype=np.float32).tobytes())
        return Path(self.cache_dir) / f"{digest.hexdigest()}.png"

    def upscale(self, image: FloatImage) -> FloatImage:
        from PIL import Image

        source = normalize_image(image)
        cache_file = self.cache_path(source)
        self.last_cache_path = cache_file
        self.cache_hit = False
        expected_shape = (
            source.shape[0] * self.scale,
            source.shape[1] * self.scale,
            3,
        )
        if cache_file.is_file():
            with Image.open(cache_file) as handle:
                cached = normalize_image(np.asarray(handle.convert("RGB")))
            if cached.shape == expected_shape:
                self.cache_hit = True
                return cached

        output = normalize_image(self.inner.upscale(source))
        if output.shape != expected_shape:
            raise ValueError(
                f"upscaler returned {output.shape}, expected {expected_shape}"
            )
        # The first run and cache hits must expose exactly the same 8-bit
        # evidence. This also keeps the 4x cache practical for repeated evals.
        raster = np.rint(output * 255.0).astype(np.uint8)
        cached_output = normalize_image(raster)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _save_rgb_png(cached_output, cache_file)
        return cached_output


@dataclass(slots=True)
class RealESRGANUpscaler:
    """Run the official x4 anime RRDB model through Spandrel."""

    model_path: str | Path
    tile_size: int = 256
    tile_padding: int = 16
    use_half: bool = True
    device: str | None = None
    _scale: int = 4

    @property
    def scale(self) -> int:
        return self._scale

    def runtime_device(self) -> str:
        try:
            import torch

            return self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return self.device or "unavailable"

    def runtime_fingerprint(self) -> dict[str, Any]:
        """Describe libraries and accelerator state that can alter inference."""

        try:
            spandrel_version = importlib.metadata.version("spandrel")
        except importlib.metadata.PackageNotFoundError:
            spandrel_version = "unavailable"
        result: dict[str, Any] = {
            "spandrel": spandrel_version,
            "device": self.runtime_device(),
        }
        try:
            import torch
        except ImportError:
            result["torch"] = "unavailable"
            return result
        result.update(
            {
                "torch": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda or "unavailable"),
                "cudnn": (
                    str(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else "unavailable"
                ),
            }
        )
        device = result["device"]
        if isinstance(device, str) and device.startswith("cuda"):
            result["accelerator"] = torch.cuda.get_device_name(torch.device(device))
        return result

    def cache_namespace(self) -> str:
        """Fingerprint model bytes and every inference-affecting setting."""

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Real-ESRGAN model not found: {model_file}")
        settings = json.dumps(
            {
                "model_sha256": _file_sha256(model_file),
                "tile_size": int(self.tile_size),
                "tile_padding": int(self.tile_padding),
                "use_half": bool(self.use_half),
                "runtime": self.runtime_fingerprint(),
                "scale": self.scale,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return settings

    def upscale(self, image: FloatImage) -> FloatImage:
        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch Real-ESRGAN generation requires torch and spandrel; "
                "run scripts/generate_realesrgan_x4.py with uv, or add "
                "`uv run --with spandrel==0.4.2` when using evaluate.py"
            ) from exc

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Real-ESRGAN model not found: {model_file}")
        descriptor = ModelLoader().load_from_file(str(model_file))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"model is not an image-to-image network: {model_file}")
        if int(descriptor.scale) != 4:
            raise ValueError(
                "RealESRGAN_x4plus_anime_6B must report scale 4, "
                f"got {descriptor.scale}"
            )
        self._scale = int(descriptor.scale)

        device = torch.device(self.runtime_device())
        half = bool(self.use_half and device.type == "cuda" and descriptor.supports_half)
        dtype = torch.float16 if half else torch.float32
        descriptor = descriptor.to(device=device, dtype=dtype).eval()

        source = normalize_image(image)
        height, width = source.shape[:2]
        scale = self.scale
        output = np.empty((height * scale, width * scale, 3), dtype=np.float32)
        tile = max(32, int(self.tile_size))
        padding = max(0, int(self.tile_padding))

        with torch.inference_mode():
            for y0 in range(0, height, tile):
                y1 = min(height, y0 + tile)
                input_y0 = max(0, y0 - padding)
                input_y1 = min(height, y1 + padding)
                for x0 in range(0, width, tile):
                    x1 = min(width, x0 + tile)
                    input_x0 = max(0, x0 - padding)
                    input_x1 = min(width, x1 + padding)
                    patch = np.ascontiguousarray(
                        source[input_y0:input_y1, input_x0:input_x1].transpose(2, 0, 1)
                    )
                    tensor = torch.from_numpy(patch).unsqueeze(0).to(
                        device=device, dtype=dtype
                    )
                    predicted = descriptor(tensor).clamp_(0.0, 1.0)
                    predicted_np = (
                        predicted[0]
                        .to(dtype=torch.float32)
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                    )
                    crop_y0 = (y0 - input_y0) * scale
                    crop_x0 = (x0 - input_x0) * scale
                    crop_y1 = crop_y0 + (y1 - y0) * scale
                    crop_x1 = crop_x0 + (x1 - x0) * scale
                    output[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = (
                        predicted_np[crop_y0:crop_y1, crop_x0:crop_x1]
                    )
        return np.clip(output, 0.0, 1.0).astype(np.float32)


def generate_realesrgan_x4(
    image: FloatImage,
    output_path: Path,
    *,
    model_path: Path,
    tile_size: int = 256,
    tile_padding: int = 16,
    use_half: bool = True,
    device: str | None = None,
    cache_dir: Path | None = Path(".cache/picvec/realesrgan"),
) -> dict[str, Any]:
    """Generate one exact x4 PNG and return reproducibility metadata."""

    source = normalize_image(image)
    inner = RealESRGANUpscaler(
        model_path=model_path,
        tile_size=max(32, int(tile_size)),
        tile_padding=max(0, int(tile_padding)),
        use_half=bool(use_half),
        device=device,
    )
    cached: CachedImageUpscaler | None = None
    upscaler: ImageUpscaler = inner
    if cache_dir is not None:
        cached = CachedImageUpscaler(inner, cache_dir, inner.cache_namespace())
        upscaler = cached
    output = upscaler.upscale(source)
    cache_path = cached.last_cache_path if cached is not None else None
    if cache_path is not None and cache_path.is_file():
        _copy_atomic(cache_path, output_path)
    else:
        _save_rgb_png(output, output_path)
    expected_shape = (source.shape[0] * 4, source.shape[1] * 4, 3)
    if output.shape != expected_shape:
        raise ValueError(f"generated x4 image has {output.shape}, expected {expected_shape}")
    return {
        "backend": "pytorch-spandrel",
        "model_path": str(model_path),
        "model_sha256": _file_sha256(model_path),
        "device": inner.runtime_device(),
        "runtime": inner.runtime_fingerprint(),
        "tile_size": inner.tile_size,
        "tile_padding": inner.tile_padding,
        "half_requested": inner.use_half,
        "cache_enabled": cached is not None,
        "cache_hit": cached.cache_hit if cached is not None else False,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "output": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "width": int(output.shape[1]),
        "height": int(output.shape[0]),
    }
