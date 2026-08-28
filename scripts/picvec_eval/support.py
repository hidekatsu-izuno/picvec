"""Small, self-contained image helpers used by the post-hoc evaluator.

These functions preserve the numerical behaviour of the SVGDeck reference
evaluator without importing any part of its vectorization pipeline.
"""

from __future__ import annotations

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


def srgb_to_lab(image: FloatImage) -> FloatImage:
    rgb = np.asarray(image, dtype=np.float32)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f = np.where(
        xyz > delta**3,
        np.cbrt(np.maximum(xyz, 0.0)),
        xyz / (3 * delta * delta) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ),
        axis=-1,
    ).astype(np.float32)


def delta_e2000(lab_a: FloatImage, lab_b: FloatImage) -> NDArray[np.float32]:
    """Vectorised CIEDE2000, returning one value per pixel."""

    a = np.asarray(lab_a, dtype=np.float32)
    b = np.asarray(lab_b, dtype=np.float32)
    l1, a1, b1 = np.moveaxis(a, -1, 0)
    l2, a2, b2 = np.moveaxis(b, -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(np.maximum(c_bar**7 / (c_bar**7 + 25**7), 0)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360)
    dl = l2 - l1
    dc = cp2 - cp1
    dh = hp2 - hp1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where((cp1 * cp2) == 0, 0, dh)
    d_h = 2 * np.sqrt(np.maximum(cp1 * cp2, 0)) * np.sin(np.radians(dh / 2))
    l_bar, c_bar_p = (l1 + l2) / 2, (cp1 + cp2) / 2
    h_bar = np.where(
        cp1 * cp2 == 0,
        hp1 + hp2,
        np.where(
            np.abs(hp1 - hp2) <= 180,
            (hp1 + hp2) / 2,
            np.where(
                hp1 + hp2 < 360,
                (hp1 + hp2 + 360) / 2,
                (hp1 + hp2 - 360) / 2,
            ),
        ),
    )
    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    rt = -2 * np.sqrt(
        np.maximum(c_bar_p**7 / (c_bar_p**7 + 25**7), 0)
    ) * np.sin(np.radians(60 * np.exp(-((h_bar - 275) / 25) ** 2)))
    return np.sqrt(
        np.maximum(
            (dl / sl) ** 2
            + (dc / sc) ** 2
            + (d_h / sh) ** 2
            + rt * (dc / sc) * (d_h / sh),
            0,
        )
    ).astype(np.float32)


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
