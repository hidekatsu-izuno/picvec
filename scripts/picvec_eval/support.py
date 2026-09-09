"""Small, self-contained image helpers used by the post-hoc evaluator.

Colour conversion and distances use the same 100-scaled OKLab convention as
the native vectorizer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatImage = NDArray[np.float32]


def normalize_image(image: NDArray[np.generic]) -> FloatImage:
    """Return RGB data in the display-sRGB range [0, 1]."""

    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in (3, 4):
        raise ValueError("expected a HxWx3 or HxWx4 image")
    if value.shape[2] == 4:
        value = value[:, :, :3]
    value = value.astype(np.float32, copy=False)
    if (
        np.issubdtype(np.asarray(image).dtype, np.integer)
        or float(np.nanmax(value, initial=0.0)) > 1.0
    ):
        value = value / 255.0
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def resize_image(image: FloatImage, shape: tuple[int, int]) -> FloatImage:
    """Resize float RGB data with a deterministic Lanczos filter."""

    from PIL import Image

    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    value = normalize_image(image)
    if value.shape[:2] == shape:
        return value.copy()
    raster = Image.fromarray(np.rint(value * 255.0).astype(np.uint8), mode="RGB")
    resized = raster.resize((width, height), Image.Resampling.LANCZOS)
    return normalize_image(np.asarray(resized))


def srgb_to_oklab(image: FloatImage) -> FloatImage:
    """Display sRGB to OKLab with all coordinates scaled by 100, as in Rust."""
    rgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    cone_matrix = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], dtype=np.float32)
    opponent_matrix = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ], dtype=np.float32)
    return (100.0 * (np.cbrt(linear @ cone_matrix.T) @ opponent_matrix.T)).astype(np.float32)


def delta_e_ok(first: FloatImage, second: FloatImage) -> NDArray[np.float32]:
    """Euclidean distance in 100-scaled OKLab, one value per pixel."""
    difference = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
    return np.sqrt(np.sum(difference * difference, axis=-1)).astype(np.float32)


def luminance_edges(
    luminance: NDArray[np.floating],
    *,
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    dark_luminance: float = 40.0,
    dark_log_gain: float = 15.0,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return ordinary edges and dark-logarithmic recovered edges."""

    from scipy import ndimage
    from skimage.feature import canny

    value = np.clip(np.asarray(luminance, dtype=np.float32), 0.0, 1.0)
    canny_kwargs = {
        "sigma": max(0.1, float(sigma)),
        "low_threshold": float(np.clip(low_threshold, 0.0, 1.0)),
        "high_threshold": float(np.clip(high_threshold, 0.0, 1.0)),
    }
    ordinary = canny(value, **canny_kwargs)
    gain = max(0.0, float(dark_log_gain))
    if gain <= 0.0:
        return ordinary, np.zeros(ordinary.shape, dtype=bool)
    logarithmic_value = np.log1p(gain * value) / np.log1p(gain)
    logarithmic = canny(logarithmic_value, **canny_kwargs)
    dark_pixels = value <= float(np.clip(dark_luminance / 100.0, 0.0, 1.0))
    dark_support = ndimage.binary_dilation(dark_pixels, iterations=2)
    dark_edges = logarithmic & dark_support
    return ordinary | dark_edges, dark_edges


def load_rgb(path: str | Path, *, background: str = "#ffffff") -> FloatImage:
    """Load an image and composite transparency onto the evaluation background."""

    from PIL import Image

    color = _parse_color(background)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*color, 255))
        rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _parse_color(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(channel * 2 for channel in text)
    if len(text) != 6:
        raise ValueError(f"background must be #RGB or #RRGGBB, got {value!r}")
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"invalid background colour: {value!r}") from exc
