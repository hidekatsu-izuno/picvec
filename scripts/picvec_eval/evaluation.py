"""Metrics and diagnostics for an x4 raster reference and x4 SVG render."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from skimage import color, metrics
from skimage.morphology import skeletonize

from .support import delta_e2000, luminance_edges, resize_image, srgb_to_lab


FloatImage = NDArray[np.float32]
BoolImage = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Pixel-domain settings expressed at the x4 evaluation resolution."""

    scale: int = 4
    edge_sigma: float = 1.2
    edge_low_threshold: float = 0.025
    edge_high_threshold: float = 0.07
    edge_tolerances: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    primary_edge_tolerance: float = 2.0
    boundary_band_radius: float = 8.0
    extra_edge_tolerance: float | None = None
    extra_edge_min_component_area: int = 4
    edge_frame_margin: int = 4
    # Thin dark seams/contours are evaluated separately from the ordinary
    # luminance edge map.  The latter is intentionally tolerant and can
    # therefore hide a one-pixel line that disappeared inside a similarly
    # coloured region.
    thin_line_neighborhood: int = 9
    thin_line_contrast: float = 0.045
    # A one-x4-pixel tolerance is intentionally stricter than the ordinary
    # boundary tolerance: a missing stroke must not be forgiven merely
    # because a nearby fill edge remains within half a source pixel.
    thin_line_tolerance: float = 1.0
    # Luminance threshold for the dark centre of an outline.  This is kept
    # separate from the local-contrast detector because a nearby fill can
    # otherwise masquerade as a surviving line.
    dark_core_luma_threshold: float = 0.20
    roughness_sigma: float = 1.5
    roughness_dilation: int = 2
    roughness_reference_tolerance: float = 1.5
    worst_tile_size: int = 256
    worst_tile_stride: int = 128
    # A rendered edge can score well while the SVG is made of thousands of
    # tiny, unstable pieces.  The selection score therefore treats weighted
    # SVG geometry per reference edge pixel as an explicit cost.  This is a
    # reporting/selection metric only; it never feeds the vectorizer.
    complexity_target_units_per_edge: float = 0.05
    quality_weight_boundary: float = 0.25
    quality_weight_colour: float = 0.20
    quality_weight_local: float = 0.10
    quality_weight_thin_line: float = 0.30
    quality_weight_dark_core: float = 0.35
    # A missing dark seam must not be compensated by a better flat-fill
    # score.  This term combines recall, dark-core agreement, and the amount
    # of disconnected missing line evidence into one integrity value.
    quality_weight_line_integrity: float = 0.85
    quality_weight_roughness: float = 0.25
    quality_weight_complexity: float = 0.40
    quality_weight_detail: float = 0.45
    # Low-resolution structure catches a globally displaced/distorted object
    # even when its local edges and average colour still look acceptable.
    quality_weight_coarse: float = 0.35
    # Direct rendered-pixel colour agreement.  This prevents a large flat
    # fill (for example liquid) from being hidden by edge/detail scores.
    quality_weight_pixel: float = 0.55
    quality_weight_geometry_wobble: float = 0.25
    quality_weight_anchor_roughness: float = 0.20
    quality_weight_open_stroke_roughness: float = 0.20
    # Connected micro-artifacts are visible in the x4 inspection image even
    # when their total area is small.  Keep this separate from ordinary edge
    # F1 so a large smooth region cannot compensate for many short false or
    # missing boundaries.
    quality_weight_micro_artifact: float = 0.45
    # Topology-aware line matching is intentionally independent from pixel
    # averages: a deleted short stroke and two SVG strokes claiming the same
    # source stroke must both lower the score even when their covered area is
    # small.
    quality_weight_structural: float = 0.95
    # Structural topology is a hard acceptance criterion as well as a quality
    # term.  A candidate with a good mean colour score must not be accepted if
    # it loses a thin stroke or creates a parallel duplicate contour.
    structural_score_floor: float = 0.12
    structural_max_line_missing_fraction: float = 0.50
    structural_max_line_duplicate_fraction: float = 0.16
    # Edge topology is broader than the dark-core mask and is therefore given
    # a slightly wider duplicate allowance for antialiased fill boundaries.
    structural_max_edge_duplicate_fraction: float = 0.21
    structural_match_tolerance: float = 1.25
    structural_min_component_pixels: int = 2
    structural_max_analysis_size: int = 1024
    structural_duplicate_support: float = 0.20
    structural_missing_recall: float = 0.50
    pixel_neighborhood_size: int = 5
    feature_narrow_size: int = 3
    feature_broad_size: int = 15
    feature_analysis_max_size: int = 768
    # Floors are deliberately independent: the x4 view must retain detailed
    # contours, while the native view must retain the original composition.
    raster_native_floor: float = 0.60
    raster_x4_floor: float = 0.65
    detail_tile_size: int = 64
    detail_tile_min_pixels: int = 100


def load_rgb(path: str | Path, *, background: str = "#ffffff") -> FloatImage:
    """Load an image and composite transparency onto the evaluation background."""

    color = _parse_color(background)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*color, 255))
        rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def save_rgb(image: NDArray[np.floating], path: str | Path) -> None:
    value = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(value, mode="RGB").save(path)


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


def _edge_map(image: FloatImage, config: EvaluationConfig) -> BoolImage:
    lightness = srgb_to_lab(image)[..., 0] / 100.0
    edges, _ = luminance_edges(
        lightness,
        sigma=max(0.1, float(config.edge_sigma)),
        low_threshold=config.edge_low_threshold,
        high_threshold=config.edge_high_threshold,
    )
    margin = min(
        max(0, int(config.edge_frame_margin)),
        max(0, (min(edges.shape) - 1) // 2),
    )
    if margin:
        edges[:margin] = False
        edges[-margin:] = False
        edges[:, :margin] = False
        edges[:, -margin:] = False
    return edges


def _thin_line_mask(
    image: FloatImage,
    config: EvaluationConfig | None = None,
    *,
    frame_margin: int | None = None,
) -> BoolImage:
    """Detect narrow dark strokes by local contrast, not by global colour.

    A normal Canny/edge score treats a missing interior seam as a small
    perturbation of a large red panel.  A local dark-line mask instead marks
    the stroke core (the pixel is darker than its neighbourhood), so its
    disappearance remains visible even when the surrounding fill is close.
    This is evaluation-only and is never passed to the vectorizer.
    """

    resolved = config or EvaluationConfig()
    neighbourhood = max(3, int(resolved.thin_line_neighborhood))
    if neighbourhood % 2 == 0:
        neighbourhood += 1
    # Rec. 709 luminance is sufficient here; the detector is deliberately
    # colour-agnostic so dark red lines are handled like black lines.
    luminance = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32, copy=False)
    local_mean = ndimage.uniform_filter(
        luminance,
        size=neighbourhood,
        mode="nearest",
    )
    dark_contrast = local_mean - luminance
    mask = dark_contrast >= float(resolved.thin_line_contrast)
    # Keep the mask to stroke-like pixels.  Removing the interior of a large
    # dark region prevents a missing broad fill from being misreported as a
    # missing line, while retaining antialiased contour pixels.
    dark_region = luminance <= local_mean - float(resolved.thin_line_contrast) * 0.5
    distance_from_background = ndimage.distance_transform_cdt(
        dark_region, metric="chessboard"
    )
    mask &= distance_from_background <= max(2.0, neighbourhood / 3.0)
    margin = (
        int(resolved.edge_frame_margin)
        if frame_margin is None
        else max(0, int(frame_margin))
    )
    margin = min(margin, max(0, (min(mask.shape) - 1) // 2))
    if margin:
        mask[:margin] = False
        mask[-margin:] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    return mask


def thin_line_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    *,
    tolerance: float = 1.0,
    frame_margin: int = 0,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Measure recall/precision of narrow dark strokes in two equal images."""

    reference_value = np.asarray(reference, dtype=np.float32)
    rendered_value = np.asarray(rendered, dtype=np.float32)
    if reference_value.shape != rendered_value.shape:
        raise ValueError(
            "thin-line images must have equal dimensions: "
            f"{reference_value.shape} != {rendered_value.shape}"
        )
    resolved = config or EvaluationConfig()
    # The ordinary edge metric remains x4.  Thin-line matching is reduced to a
    # capped source-scale grid so its distance fields cannot compete for
    # memory with the x4 boundary and colour fields.  The tolerance is scaled
    # accordingly, preserving the same quarter-source-pixel strictness.
    source_shape = (
        max(64, min(512, reference_value.shape[0] // max(1, int(resolved.scale)))),
        max(64, min(512, reference_value.shape[1] // max(1, int(resolved.scale)))),
    )
    reference_source = resize_image(reference_value, source_shape)
    rendered_source = resize_image(rendered_value, source_shape)
    source_frame_margin = max(0, int(frame_margin) // max(1, int(resolved.scale)))
    reference_mask = _thin_line_mask(
        reference_source, resolved, frame_margin=source_frame_margin
    )
    rendered_mask = _thin_line_mask(
        rendered_source, resolved, frame_margin=source_frame_margin
    )
    reference_count = int(np.count_nonzero(reference_mask))
    rendered_count = int(np.count_nonzero(rendered_mask))
    distance_to_rendered = (
        ndimage.distance_transform_cdt(~rendered_mask, metric="chessboard")
        if rendered_count
        else np.full(reference_mask.shape, np.inf, dtype=np.float32)
    )
    distance_to_reference = (
        ndimage.distance_transform_cdt(~reference_mask, metric="chessboard")
        if reference_count
        else np.full(reference_mask.shape, np.inf, dtype=np.float32)
    )
    resolved_tolerance = max(0.0, float(tolerance)) / max(1, int(resolved.scale))
    recall = (
        float(np.mean(distance_to_rendered[reference_mask] <= resolved_tolerance))
        if reference_count
        else 1.0
    )
    precision = (
        float(np.mean(distance_to_reference[rendered_mask] <= resolved_tolerance))
        if rendered_count
        else float(reference_count == 0)
    )
    missing = reference_mask & (distance_to_rendered > resolved_tolerance)
    extra = rendered_mask & (distance_to_reference > resolved_tolerance)
    divisor = 1.0
    result: dict[str, Any] = {
        "reference_line_pixels": reference_count,
        "rendered_line_pixels": rendered_count,
        "tolerance_x4_pixels": resolved_tolerance,
        "tolerance_source_pixels": resolved_tolerance,
        "recall": recall,
        "precision": precision,
        "f1": _f1(recall, precision),
        # Recall is intentionally weighted more heavily: inventing a short
        # dark mark is bad, but deleting a source stroke is the severe error
        # this metric is designed to expose.
        "line_score": float(0.75 * recall + 0.25 * precision),
        "missing_pixels_x4": int(np.count_nonzero(missing) * divisor),
        "missing_source_pixels": float(np.count_nonzero(missing)),
        "extra_pixels_x4": int(np.count_nonzero(extra) * divisor),
        "extra_source_pixels": float(np.count_nonzero(extra)),
        "missing_components": _component_summary(
            missing,
            scale=1,
            minimum_area=1,
        ),
        "extra_components": _component_summary(
            extra,
            scale=1,
            minimum_area=1,
        ),
    }
    # A nearby red fill can satisfy local-contrast recall even when the actual
    # black/dark outline is gone. Track the dark core independently.
    # Dark-core matching is intentionally performed at source scale.  The
    # ordinary thin-line metric already owns the x4 distance fields; creating
    # two more 25M-pixel fields here would make evaluation needlessly exceed
    # the memory budget on the car fixture.
    source_shape = (
        max(64, min(512, reference_value.shape[0] // max(1, int(resolved.scale)))),
        max(64, min(512, reference_value.shape[1] // max(1, int(resolved.scale)))),
    )
    reference_source = resize_image(reference_value, source_shape)
    rendered_source = resize_image(rendered_value, source_shape)
    source_config = resolved
    source_frame_margin = max(0, int(frame_margin) // max(1, int(resolved.scale)))
    reference_source_mask = _thin_line_mask(
        reference_source, source_config, frame_margin=source_frame_margin
    )
    rendered_source_mask = _thin_line_mask(
        rendered_source, source_config, frame_margin=source_frame_margin
    )
    reference_luma = (
        0.2126 * reference_source[..., 0]
        + 0.7152 * reference_source[..., 1]
        + 0.0722 * reference_source[..., 2]
    )
    rendered_luma = (
        0.2126 * rendered_source[..., 0]
        + 0.7152 * rendered_source[..., 1]
        + 0.0722 * rendered_source[..., 2]
    )
    dark_threshold = float(np.clip(resolved.dark_core_luma_threshold, 0.0, 1.0))
    dark_reference = reference_source_mask & (reference_luma <= dark_threshold)
    dark_rendered = rendered_source_mask & (rendered_luma <= dark_threshold)
    dark_reference_count = int(np.count_nonzero(dark_reference))
    dark_rendered_count = int(np.count_nonzero(dark_rendered))
    dark_to_rendered = (
        ndimage.distance_transform_edt(~dark_rendered).astype(np.float32, copy=False)
        if dark_rendered_count
        else np.full(dark_reference.shape, np.inf, dtype=np.float32)
    )
    dark_to_reference = (
        ndimage.distance_transform_edt(~dark_reference).astype(np.float32, copy=False)
        if dark_reference_count
        else np.full(dark_reference.shape, np.inf, dtype=np.float32)
    )
    source_tolerance = resolved_tolerance / max(1, int(resolved.scale))
    dark_recall = (
        float(np.mean(dark_to_rendered[dark_reference] <= source_tolerance))
        if dark_reference_count
        else 1.0
    )
    dark_precision = (
        float(np.mean(dark_to_reference[dark_rendered] <= source_tolerance))
        if dark_rendered_count
        else float(dark_reference_count == 0)
    )
    result["dark_core"] = {
        "reference_pixels": dark_reference_count,
        "rendered_pixels": dark_rendered_count,
        "recall": dark_recall,
        "precision": dark_precision,
        "score": float(0.80 * dark_recall + 0.20 * dark_precision),
        "missing_pixels": int(
            np.count_nonzero(dark_reference & (dark_to_rendered > source_tolerance))
        ),
    }
    del (
        reference_source,
        rendered_source,
        reference_source_mask,
        rendered_source_mask,
        reference_luma,
        rendered_luma,
        dark_reference,
        dark_rendered,
        dark_to_rendered,
        dark_to_reference,
    )
    return result


def _structure_component_match(
    reference_mask: BoolImage,
    rendered_mask: BoolImage,
    *,
    tolerance: float,
    minimum_area: int,
    duplicate_support: float,
    missing_recall: float,
) -> dict[str, Any]:
    """Match disconnected source strokes to SVG strokes one-to-one.

    Distance-tolerant pixel F1 is deliberately not used here.  It allows one
    long rendered stroke to explain several source strokes, and it does not
    distinguish one correct stroke from two parallel strokes.  The component
    report instead performs a local, greedy one-to-one assignment and exposes
    both unmatched source components and secondary candidate components.
    """

    structure = np.ones((3, 3), dtype=np.uint8)
    reference_labels, reference_count = ndimage.label(reference_mask, structure=structure)
    rendered_labels, rendered_count = ndimage.label(rendered_mask, structure=structure)
    reference_areas = np.bincount(reference_labels.ravel())[1:]
    rendered_areas = np.bincount(rendered_labels.ravel())[1:]
    reference_ids = [
        index + 1
        for index, area in enumerate(reference_areas)
        if int(area) >= max(1, int(minimum_area))
    ]
    rendered_ids = [
        index + 1
        for index, area in enumerate(rendered_areas)
        if int(area) >= max(1, int(minimum_area))
    ]
    tolerance = max(0.0, float(tolerance))
    rendered_distance = (
        ndimage.distance_transform_edt(~rendered_mask)
        if rendered_ids
        else np.full(reference_mask.shape, np.inf, dtype=np.float32)
    )
    reference_distance = (
        ndimage.distance_transform_edt(~reference_mask)
        if reference_ids
        else np.full(reference_mask.shape, np.inf, dtype=np.float32)
    )

    source_coverages: list[float] = []
    support_by_reference: dict[int, list[tuple[float, int]]] = {}
    for reference_id in reference_ids:
        component = reference_labels == reference_id
        area = max(1, int(reference_areas[reference_id - 1]))
        coverage = float(np.mean(rendered_distance[component] <= tolerance))
        source_coverages.append(coverage)
        dilation = ndimage.binary_dilation(
            component,
            iterations=max(1, int(np.ceil(tolerance))),
            structure=structure,
        )
        candidate_ids: list[tuple[float, int]] = []
        for rendered_id in rendered_ids:
            candidate = rendered_labels == rendered_id
            overlap = int(np.count_nonzero(candidate & dilation))
            if overlap == 0:
                continue
            # Require a meaningful fraction of the source component to be
            # explained.  This keeps a long neighbouring contour from making
            # every tiny source mark appear matched.
            candidate_support = float(
                np.count_nonzero(component & ndimage.binary_dilation(
                    candidate,
                    iterations=max(1, int(np.ceil(tolerance))),
                    structure=structure,
                )) / area
            )
            if candidate_support >= float(duplicate_support):
                candidate_ids.append((candidate_support, rendered_id))
        candidate_ids.sort(reverse=True)
        support_by_reference[reference_id] = candidate_ids

    # Greedy assignment is deterministic and prevents one candidate component
    # from hiding several missing source components.
    pairs = sorted(
        (
            support,
            reference_id,
            rendered_id,
        )
        for reference_id, candidates in support_by_reference.items()
        for support, rendered_id in candidates
    )
    pairs.reverse()
    assigned_reference: set[int] = set()
    assigned_rendered: set[int] = set()
    assignments: list[tuple[int, int, float]] = []
    for support, reference_id, rendered_id in pairs:
        if reference_id in assigned_reference or rendered_id in assigned_rendered:
            continue
        assigned_reference.add(reference_id)
        assigned_rendered.add(rendered_id)
        assignments.append((reference_id, rendered_id, support))

    supported_rendered: set[int] = {
        rendered_id
        for candidates in support_by_reference.values()
        for _support, rendered_id in candidates
    }
    duplicate_components = sum(
        max(0, len(candidates) - 1)
        for candidates in support_by_reference.values()
    )
    missing_components = sum(
        1
        for coverage in source_coverages
        if coverage < float(missing_recall)
    )
    extra_components = sum(
        1 for rendered_id in rendered_ids if rendered_id not in supported_rendered
    )
    source_count = len(reference_ids)
    rendered_count = len(rendered_ids)
    macro_recall = float(np.mean(source_coverages)) if source_coverages else 1.0
    # Equal component weighting is intentional: a short musical note stroke
    # must not disappear behind a large car silhouette.
    weighted_recall = (
        float(
            np.average(
                source_coverages,
                weights=np.sqrt(
                    np.asarray(
                        [reference_areas[index - 1] for index in reference_ids],
                        dtype=np.float32,
                    )
                ),
            )
        )
        if source_coverages
        else 1.0
    )
    missing_fraction = missing_components / max(1, source_count)
    duplicate_fraction = duplicate_components / max(1, source_count)
    extra_fraction = extra_components / max(1, rendered_count)
    score = float(
        np.clip(
            (0.70 * macro_recall + 0.30 * weighted_recall)
            * np.exp(
                -1.8 * missing_fraction
                -1.35 * duplicate_fraction
                -0.90 * extra_fraction
            ),
            0.0,
            1.0,
        )
    )
    return {
        "reference_components": source_count,
        "rendered_components": rendered_count,
        "matched_components": len(assignments),
        "missing_components": int(missing_components),
        "duplicate_components": int(duplicate_components),
        "extra_components": int(extra_components),
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "missing_component_fraction": float(missing_fraction),
        "duplicate_component_fraction": float(duplicate_fraction),
        "extra_component_fraction": float(extra_fraction),
        "score": score,
        "assignments": [
            {
                "reference_id": int(reference_id),
                "rendered_id": int(rendered_id),
                "support": float(support),
            }
            for reference_id, rendered_id, support in assignments
        ],
    }


def structural_line_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    *,
    scale: int,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate stroke topology at native-pixel scale.

    The reference is only used to derive masks for evaluation.  No mask or
    score is fed into vectorization.  Both the dark stroke core and a
    skeletonized luminance edge map are reported: dark-core matching catches
    deleted outlines, while the edge topology catches coloured silhouettes
    and unsupported protrusions.
    """

    if reference.shape != rendered.shape:
        raise ValueError("structural images must have equal dimensions")
    resolved_scale = max(1, int(scale))
    max_size = max(64, int(config.structural_max_analysis_size))
    analysis_shape = (
        max(64, min(max_size, reference.shape[0] // resolved_scale)),
        max(64, min(max_size, reference.shape[1] // resolved_scale)),
    )
    reference_analysis = resize_image(reference, analysis_shape)
    rendered_analysis = resize_image(rendered, analysis_shape)
    local_config = replace(
        config,
        scale=1,
        edge_sigma=max(0.25, float(config.edge_sigma) / resolved_scale),
        edge_frame_margin=max(0, int(np.ceil(config.edge_frame_margin / resolved_scale))),
    )
    reference_lines = _thin_line_mask(reference_analysis, local_config)
    rendered_lines = _thin_line_mask(rendered_analysis, local_config)
    reference_edges = skeletonize(_edge_map(reference_analysis, local_config))
    rendered_edges = skeletonize(_edge_map(rendered_analysis, local_config))
    line_report = _structure_component_match(
        reference_lines,
        rendered_lines,
        tolerance=max(0.25, float(config.structural_match_tolerance) / resolved_scale),
        minimum_area=config.structural_min_component_pixels,
        duplicate_support=config.structural_duplicate_support,
        missing_recall=config.structural_missing_recall,
    )
    edge_report = _structure_component_match(
        reference_edges,
        rendered_edges,
        tolerance=max(0.5, float(config.structural_match_tolerance) / resolved_scale),
        minimum_area=config.structural_min_component_pixels,
        duplicate_support=config.structural_duplicate_support,
        missing_recall=config.structural_missing_recall,
    )
    # Dark strokes carry the highest perceptual cost; edge topology remains a
    # secondary term so ordinary antialiased fill boundaries do not dominate.
    score = float(0.70 * line_report["score"] + 0.30 * edge_report["score"])
    return {
        "analysis_width": int(analysis_shape[1]),
        "analysis_height": int(analysis_shape[0]),
        "analysis_scale": resolved_scale,
        "line_components": line_report,
        "edge_components": edge_report,
        "score": score,
        "interpretation": (
            "Components are matched one-to-one. Missing short strokes, "
            "parallel duplicate strokes claiming one source component, and "
            "unsupported extra components are penalized independently of "
            "area-based pixel similarity."
        ),
    }


def _edge_roughness_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    reference_edges: BoolImage,
    rendered_edges: BoolImage,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Measure excess high-frequency edge energy in the rendered image.

    Thin-line recall alone rewards a jagged stroke if it happens to overlap the
    reference.  Comparing fine edges with edges that survive a small blur
    exposes staircase/noise contours while treating missing detail separately
    through ``thin_lines``.
    """

    # Evaluate roughness at source scale.  This both matches the visual unit
    # of a staircase and avoids allocating another full x4 distance field.
    source_shape = (
        max(64, min(512, reference.shape[0] // max(1, int(config.scale)))),
        max(64, min(512, reference.shape[1] // max(1, int(config.scale)))),
    )

    def ratio(image: FloatImage) -> float:
        # Strided sampling is sufficient for the roughness statistic and
        # avoids a second full-image PIL conversion while x4 edge masks are
        # resident.
        step = max(1, int(config.scale))
        image = np.asarray(image[::step, ::step], dtype=np.float32)
        if image.shape[:2] != source_shape:
            image = resize_image(image, source_shape)
        fine = _edge_map(image, config)
        blurred = ndimage.gaussian_filter(
            image,
            sigma=(max(0.5, float(config.roughness_sigma)),) * 2 + (0.0,),
        )
        coarse = _edge_map(blurred, config)
        covered = ndimage.binary_dilation(
            coarse,
            iterations=max(1, int(config.roughness_dilation)),
        )
        high_frequency = fine & ~covered
        value = float(np.count_nonzero(high_frequency)) / max(
            1, int(np.count_nonzero(fine))
        )
        del image, fine, blurred, coarse, covered, high_frequency
        return value

    reference_ratio = ratio(reference)
    rendered_ratio = ratio(rendered)
    step = max(1, int(config.scale))
    reference_support = ndimage.binary_dilation(
        reference_edges[::step, ::step],
        iterations=max(1, int(round(config.roughness_reference_tolerance))),
    )
    rendered_source = np.asarray(rendered[::step, ::step], dtype=np.float32)
    rendered_fine = _edge_map(rendered_source, config)
    rendered_blurred = ndimage.gaussian_filter(
        rendered_source,
        sigma=(max(0.5, float(config.roughness_sigma)),) * 2 + (0.0,),
    )
    rendered_coarse = _edge_map(rendered_blurred, config)
    rendered_high = rendered_fine & ~ndimage.binary_dilation(
        rendered_coarse, iterations=max(1, int(config.roughness_dilation))
    )
    if reference_support.shape != rendered_high.shape:
        reference_support = resize_image(
            reference_support.astype(np.float32)[..., None], rendered_high.shape
        )[..., 0] > 0.5
    unsupported_ratio = float(np.count_nonzero(rendered_high & ~reference_support)) / max(
        1, int(np.count_nonzero(rendered_fine))
    )
    del (
        reference_support,
        rendered_source,
        rendered_fine,
        rendered_blurred,
        rendered_coarse,
        rendered_high,
    )
    excess = max(0.0, rendered_ratio - reference_ratio)
    return {
        "reference_high_frequency_ratio": reference_ratio,
        "rendered_high_frequency_ratio": rendered_ratio,
        "excess_high_frequency_ratio": excess,
        "unsupported_high_frequency_ratio": unsupported_ratio,
        "score": float(1.0 / (1.0 + 12.0 * excess)),
    }


def _detail_fidelity_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Measure preservation of source-supported fine colour/luminance detail.

    Global colour and boundary averages can hide a small face, eye, highlight,
    or hand detail.  This metric selects high-pass energy from the reference
    only, then checks whether the rendered image retains comparable local
    variation and colour there.  It is evaluation-only and is never passed to
    vectorization.
    """

    step = max(1, int(config.scale))
    source_shape = (
        max(64, min(512, reference.shape[0] // step)),
        max(64, min(512, reference.shape[1] // step)),
    )
    ref = resize_image(reference[::step, ::step], source_shape)
    out = resize_image(rendered[::step, ::step], source_shape)
    ref_luma = (
        0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    )
    out_luma = (
        0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    )
    ref_blur = ndimage.gaussian_filter(ref_luma, sigma=1.0)
    out_blur = ndimage.gaussian_filter(out_luma, sigma=1.0)
    ref_high = np.abs(ref_luma - ref_blur)
    out_high = np.abs(out_luma - out_blur)
    threshold = float(np.percentile(ref_high, 82.0))
    support = ref_high >= max(0.012, threshold)
    if not np.any(support):
        return {"reference_pixels": 0.0, "recall": 1.0, "contrast_ratio": 1.0, "score": 1.0}
    # A detail survives when at least half of its reference local contrast is
    # still present.  This catches smoothed-away eyes/highlights without
    # rewarding unrelated noise elsewhere.
    retained = out_high >= np.maximum(0.008, ref_high * 0.50)
    recall = float(np.mean(retained[support]))
    contrast_ratio = float(
        np.mean(np.minimum(out_high[support] / np.maximum(ref_high[support], 1e-6), 1.0))
    )
    colour_error = delta_e2000(
        np.asarray(color.rgb2lab(ref), dtype=np.float32),
        np.asarray(color.rgb2lab(out), dtype=np.float32),
    )
    colour_score = float(1.0 / (1.0 + float(np.mean(colour_error[support])) / 8.0))
    ref_edge = np.hypot(ndimage.sobel(ref_luma, axis=0), ndimage.sobel(ref_luma, axis=1))
    out_edge = np.hypot(ndimage.sobel(out_luma, axis=0), ndimage.sobel(out_luma, axis=1))
    edge_support = ref_edge >= max(0.12, float(np.percentile(ref_edge, 88.0)))
    edge_recall = float(
        np.mean(out_edge[edge_support] >= np.maximum(0.08, ref_edge[edge_support] * 0.45))
    ) if np.any(edge_support) else 1.0
    tile_scores: list[tuple[float, int, int, int]] = []
    tile_size = max(8, int(config.detail_tile_size))
    for y in range(0, source_shape[0], tile_size):
        for x in range(0, source_shape[1], tile_size):
            tile_support = support[y : y + tile_size, x : x + tile_size]
            count = int(np.count_nonzero(tile_support))
            if count < max(1, int(config.detail_tile_min_pixels)):
                continue
            tile_recall = float(np.mean(retained[y : y + tile_size, x : x + tile_size][tile_support]))
            tile_scores.append((tile_recall, count, x, y))
    tile_scores.sort(key=lambda item: item[0])
    tile_p10 = float(np.percentile([item[0] for item in tile_scores], 10.0)) if tile_scores else recall
    score = float(0.30 * recall + 0.18 * contrast_ratio + 0.15 * colour_score + 0.22 * tile_p10 + 0.15 * edge_recall)
    return {
        "reference_pixels": float(np.count_nonzero(support)),
        "recall": recall,
        "contrast_ratio": contrast_ratio,
        "colour_score": colour_score,
        "edge_recall": edge_recall,
        "tile_p10_recall": tile_p10,
        "worst_tile": (
            {"recall": tile_scores[0][0], "reference_pixels": tile_scores[0][1], "x": tile_scores[0][2], "y": tile_scores[0][3]}
            if tile_scores
            else None
        ),
        "score": score,
    }


def _coarse_fidelity_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Measure low-frequency shape and colour agreement.

    This is deliberately computed on a small source-scale image.  It is not
    used by the vectorizer: its purpose is to reject candidates whose object
    is globally warped or shifted while fine edge metrics still happen to
    match nearby contours.
    """

    step = max(1, int(config.scale))
    source_shape = (
        max(32, min(128, reference.shape[0] // step)),
        max(32, min(128, reference.shape[1] // step)),
    )
    ref = resize_image(reference[::step, ::step], source_shape)
    out = resize_image(rendered[::step, ::step], source_shape)
    ref_lab = srgb_to_lab(ref)
    out_lab = srgb_to_lab(out)
    coarse_delta = delta_e2000(ref_lab, out_lab)
    delta_mean = float(np.mean(coarse_delta))
    delta_p90 = float(np.percentile(coarse_delta, 90.0))
    ref_luma = (
        0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    )
    out_luma = (
        0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    )
    luma_mae = float(np.mean(np.abs(ref_luma - out_luma)))
    # Colour and luminance terms are bounded and remain interpretable in the
    # report.  The p90 term prevents a large local displacement from being
    # hidden by a mostly correct background.
    colour_score = 1.0 / (1.0 + delta_mean / 7.0 + delta_p90 / 35.0)
    luma_score = 1.0 / (1.0 + luma_mae / 0.12)
    return {
        "source_width": float(source_shape[1]),
        "source_height": float(source_shape[0]),
        "delta_e00_mean": delta_mean,
        "delta_e00_p90": delta_p90,
        "luma_mae": luma_mae,
        "colour_score": float(colour_score),
        "luma_score": float(luma_score),
        "score": float(0.65 * colour_score + 0.35 * luma_score),
    }


def _pixel_fidelity_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Score the rendered image directly in pixel colour space.

    The primary score compares neighbourhood means at the original
    raster resolution, so a one-pixel antialiasing wobble at an outline is
    not treated like a flat colour error.  A lower tile percentile keeps
    small but important objects from disappearing in the global mean.
    """

    source_shape = reference.shape[:2]
    ref_source = np.asarray(reference, dtype=np.float32)
    out_source = resize_image(np.asarray(rendered, dtype=np.float32), source_shape)
    neighborhood = max(3, int(config.pixel_neighborhood_size))
    if neighborhood % 2 == 0:
        neighborhood += 1
    ref_mean = ndimage.uniform_filter(
        ref_source, size=(neighborhood, neighborhood, 1), mode="nearest"
    )
    out_mean = ndimage.uniform_filter(
        out_source, size=(neighborhood, neighborhood, 1), mode="nearest"
    )
    local_delta = delta_e2000(srgb_to_lab(ref_mean), srgb_to_lab(out_mean))
    values = np.asarray(local_delta, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "mean_delta_e00": 0.0,
            "p90_delta_e00": 0.0,
            "interior_raw_delta_e00_mean": 0.0,
            "within_delta_e00_5": 1.0,
            "pixel_score": 1.0,
            "tile_p10_score": 1.0,
            "score": 1.0,
        }
    mean_delta = float(np.mean(finite))
    p90_delta = float(np.percentile(finite, 90.0))
    within_five = float(np.mean(finite <= 5.0))
    # Keep a raw-pixel diagnostic for inspection. The score itself does not
    # branch on edge/interior status: every source pixel uses the same mean.
    raw_lab = delta_e2000(srgb_to_lab(ref_source), srgb_to_lab(out_source))
    interior_raw_mean = float(np.mean(raw_lab))
    # Delta-E 20 is visibly wrong; Delta-E 5 is a useful boundary for a
    # near-match.  The reciprocal form remains stable for antialiasing noise.
    mean_score = 1.0 / (1.0 + mean_delta / 5.0)
    p90_score = 1.0 / (1.0 + p90_delta / 20.0)
    tile_size = max(8, int(config.worst_tile_size) // max(1, int(config.scale)))
    tile_scores: list[float] = []
    for y in range(0, values.shape[0], tile_size):
        for x in range(0, values.shape[1], tile_size):
            tile = values[y : y + tile_size, x : x + tile_size]
            if tile.size:
                tile_mean = float(np.mean(tile))
                tile_scores.append(1.0 / (1.0 + tile_mean / 5.0))
    tile_p05 = float(np.percentile(tile_scores, 5.0)) if tile_scores else mean_score
    tile_min = float(min(tile_scores)) if tile_scores else mean_score
    score = float(
        0.30 * mean_score
        + 0.15 * p90_score
        + 0.15 * within_five
        + 0.20 * tile_p05
        + 0.20 * tile_min
    )
    return {
        "mean_delta_e00": mean_delta,
        "p90_delta_e00": p90_delta,
        "interior_raw_delta_e00_mean": interior_raw_mean,
        "within_delta_e00_5": within_five,
        "mean_score": float(mean_score),
        "p90_score": float(p90_score),
        "tile_p05_score": tile_p05,
        "worst_tile_score": tile_min,
        "score": score,
    }


def _feature_maps(image: FloatImage, *, analysis_shape: tuple[int, int], scale: int, config: EvaluationConfig) -> dict[str, NDArray[np.float32]]:
    """Build continuous colour/highlight/dark/edge features for raster scoring."""
    value = resize_image(np.asarray(image, dtype=np.float32), analysis_shape)
    shrink = max(1.0, float(max(image.shape[:2]) / max(analysis_shape)))
    narrow = max(1, int(round(max(1, config.feature_narrow_size) * max(1, scale) / shrink)))
    if narrow % 2 == 0:
        narrow += 1
    smooth = ndimage.uniform_filter(value, size=(narrow, narrow, 1), mode="nearest")
    lab = srgb_to_lab(smooth)
    luma = (0.2126 * smooth[..., 0] + 0.7152 * smooth[..., 1] + 0.0722 * smooth[..., 2]).astype(np.float32)
    local = ndimage.gaussian_filter(luma, sigma=max(1.0, narrow / 2.0))
    highlight = np.maximum(luma - local, 0.0)
    dark = np.maximum(local - luma, 0.0)
    edge = np.hypot(ndimage.sobel(luma, axis=0), ndimage.sobel(luma, axis=1)).astype(np.float32)
    return {
        "colour": lab.astype(np.float32),
        "highlight": highlight,
        "dark": dark,
        "edge": edge,
    }


def _filtered_raster_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    *,
    scale: int,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Compare filtered feature maps; no boundary pixels are exempted."""
    max_size = max(128, int(config.feature_analysis_max_size))
    factor = max(reference.shape[:2]) / max(1, min(max(reference.shape[:2]), max_size))
    shape = (
        max(32, int(round(reference.shape[0] / factor))),
        max(32, int(round(reference.shape[1] / factor))),
    )
    ref = _feature_maps(reference, analysis_shape=shape, scale=scale, config=config)
    out = _feature_maps(rendered, analysis_shape=shape, scale=scale, config=config)
    broad = max(3, int(round(config.feature_broad_size / max(1.0, factor))))
    if broad % 2 == 0:
        broad += 1
    diffs: dict[str, NDArray[np.float32]] = {
        "colour": np.linalg.norm(ref["colour"] - out["colour"], axis=2).astype(np.float32),
        "highlight": np.abs(ref["highlight"] - out["highlight"]),
        "dark": np.abs(ref["dark"] - out["dark"]),
        "edge": np.abs(ref["edge"] - out["edge"]),
    }
    thresholds = {"colour": 5.0, "highlight": 0.06, "dark": 0.06, "edge": 0.10}
    feature_report: dict[str, Any] = {}
    scores: list[float] = []
    for name, diff in diffs.items():
        filtered = ndimage.uniform_filter(diff, size=broad, mode="nearest")
        mean = float(np.mean(filtered))
        p90 = float(np.percentile(filtered, 90.0))
        score = 1.0 / (1.0 + mean / max(thresholds[name], 1e-6) + p90 / max(4.0 * thresholds[name], 1e-6))
        support = ref[name] >= np.percentile(ref[name], 85.0)
        if name == "colour":
            support = np.linalg.norm(ref[name] - ndimage.uniform_filter(ref[name], size=broad, mode="nearest"), axis=2) >= 2.0
        recall = float(np.mean(filtered[support] <= thresholds[name])) if np.any(support) else 1.0
        feature_report[name] = {
            "mean_filtered_difference": mean,
            "p90_filtered_difference": p90,
            "support_recall": recall,
            "score": score,
        }
        scores.append(score)
    tile_size = max(8, int(round(64 / max(1.0, factor))))
    tile_scores: list[float] = []
    colour_diff = ndimage.uniform_filter(diffs["colour"], size=broad, mode="nearest")
    for y in range(0, shape[0], tile_size):
        for x in range(0, shape[1], tile_size):
            tile_scores.append(1.0 / (1.0 + float(np.mean(colour_diff[y:y+tile_size, x:x+tile_size])) / 5.0))
    worst_tile = float(min(tile_scores)) if tile_scores else min(scores)
    score = float(0.30 * scores[0] + 0.20 * scores[1] + 0.25 * scores[2] + 0.25 * scores[3])
    return {
        "analysis_width": int(shape[1]),
        "analysis_height": int(shape[0]),
        "narrow_filter_pixels": int(max(1, config.feature_narrow_size) * max(1, scale)),
        "broad_filter_pixels": int(max(3, config.feature_broad_size) * max(1, scale)),
        "features": feature_report,
        "worst_tile_score": worst_tile,
        "score": float(0.75 * score + 0.25 * worst_tile),
    }


def _raster_selection_metrics(
    native: Mapping[str, Any],
    x4: Mapping[str, Any],
    config: EvaluationConfig,
    structural_score: float | None = None,
    structural_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    native_score = float(native.get("score", 0.0))
    x4_score = float(x4.get("score", 0.0))
    raster_score = float(0.5 * native_score + 0.5 * x4_score)
    resolved_structural = (
        None
        if structural_score is None
        else float(np.clip(structural_score, 0.0, 1.0))
    )
    # Keep the historical raster score visible, but use a modest structural
    # contribution for candidate ranking.  This is not a rollback/gate: it
    # only makes duplicate or missing strokes affect the reported selection
    # score instead of being hidden by broad colour averages.
    selection_score = (
        float(0.75 * raster_score + 0.25 * resolved_structural)
        if resolved_structural is not None
        else raster_score
    )
    failure_reasons: list[str] = []
    if native_score < float(config.raster_native_floor):
        failure_reasons.append("native_below_floor")
    if x4_score < float(config.raster_x4_floor):
        failure_reasons.append("x4_below_floor")
    if resolved_structural is not None:
        if resolved_structural < float(config.structural_score_floor):
            failure_reasons.append("structural_below_floor")
        # The line component report is intentionally checked at both scales.
        # The x4 report catches defects that disappear after downsampling,
        # while native catches source-resolution seams and short notes.
        for scale_name in ("native", "x4"):
            scale_report = (structural_report or {}).get(scale_name) or {}
            line_report = scale_report.get("line_components") or {}
            missing = float(line_report.get("missing_component_fraction") or 0.0)
            duplicate = float(line_report.get("duplicate_component_fraction") or 0.0)
            if missing > float(config.structural_max_line_missing_fraction):
                failure_reasons.append(f"{scale_name}_line_missing_above_limit")
            if duplicate > float(config.structural_max_line_duplicate_fraction):
                failure_reasons.append(f"{scale_name}_line_duplicate_above_limit")
            edge_report = scale_report.get("edge_components") or {}
            edge_duplicate = float(
                edge_report.get("duplicate_component_fraction") or 0.0
            )
            if edge_duplicate > float(config.structural_max_edge_duplicate_fraction):
                failure_reasons.append(f"{scale_name}_edge_duplicate_above_limit")
    valid = not failure_reasons
    return {
        "native": native,
        "x4": x4,
        "native_floor": float(config.raster_native_floor),
        "x4_floor": float(config.raster_x4_floor),
        "native_score": native_score,
        "x4_score": x4_score,
        "raster_score": raster_score,
        "structural_score": resolved_structural,
        "score": selection_score,
        "valid": valid,
        "structural_score_floor": float(config.structural_score_floor),
        "structural_max_line_missing_fraction": float(config.structural_max_line_missing_fraction),
        "structural_max_line_duplicate_fraction": float(config.structural_max_line_duplicate_fraction),
        "structural_max_edge_duplicate_fraction": float(
            config.structural_max_edge_duplicate_fraction
        ),
        "failure_reasons": failure_reasons,
    }


def _f1(recall: float, precision: float) -> float:
    denominator = recall + precision
    return 2.0 * recall * precision / denominator if denominator > 0.0 else 0.0


def _nullable(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _distance_summary(
    reference_distances: NDArray[np.floating],
    rendered_distances: NDArray[np.floating],
    *,
    scale: int,
) -> dict[str, Any]:
    divisor = float(scale)

    def stats(values: NDArray[np.floating]) -> dict[str, float | None]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return {
                "mean_source_pixels": None,
                "p95_source_pixels": None,
                "p99_source_pixels": None,
                "max_source_pixels": None,
            }
        return {
            "mean_source_pixels": float(np.mean(finite) / divisor),
            "p95_source_pixels": float(np.percentile(finite, 95.0) / divisor),
            "p99_source_pixels": float(np.percentile(finite, 99.0) / divisor),
            "max_source_pixels": float(np.max(finite) / divisor),
        }

    reference_stats = stats(reference_distances)
    rendered_stats = stats(rendered_distances)
    combined = np.concatenate(
        (
            reference_distances[np.isfinite(reference_distances)],
            rendered_distances[np.isfinite(rendered_distances)],
        )
    )
    symmetric = stats(combined)
    return {
        **symmetric,
        "reference_to_svg": reference_stats,
        "svg_to_reference": rendered_stats,
    }


def _component_summary(
    mask: BoolImage,
    *,
    scale: int,
    minimum_area: int,
) -> dict[str, Any]:
    """Summarise disconnected error components, including the worst one."""

    labelled, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return {
            "count": 0,
            "count_over_minimum_area": 0,
            "total_pixels_x4": 0,
            "total_source_pixels": 0.0,
            "largest_pixels_x4": 0,
            "largest_source_pixels": 0.0,
            "largest_bbox_x4": None,
        }
    objects = ndimage.find_objects(labelled)
    areas = np.bincount(labelled.ravel())[1:]
    significant = areas >= max(1, int(minimum_area))
    largest_index = int(np.argmax(areas))
    largest_slice = objects[largest_index]
    bbox = None
    if largest_slice is not None:
        y_slice, x_slice = largest_slice
        bbox = {
            "x": int(x_slice.start),
            "y": int(y_slice.start),
            "width": int(x_slice.stop - x_slice.start),
            "height": int(y_slice.stop - y_slice.start),
        }
    divisor = float(scale * scale)
    largest = int(areas[largest_index])
    return {
        "count": int(count),
        "count_over_minimum_area": int(np.count_nonzero(significant)),
        "total_pixels_x4": int(np.sum(areas)),
        "total_source_pixels": float(np.sum(areas) / divisor),
        "largest_pixels_x4": largest,
        "largest_source_pixels": float(largest / divisor),
        "largest_bbox_x4": bbox,
    }


def _tile_boundary_metrics(
    reference_edges: BoolImage,
    rendered_edges: BoolImage,
    distance_to_rendered: NDArray[np.float64],
    distance_to_reference: NDArray[np.float64],
    delta_e: NDArray[np.float32],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Return worst local boundary/error tiles instead of only global quantiles."""

    height, width = reference_edges.shape
    size = max(8, int(config.worst_tile_size))
    stride = max(1, int(config.worst_tile_stride))
    radius = max(0.0, float(config.boundary_band_radius))
    frame_margin = min(
        max(0, int(config.edge_frame_margin)),
        max(0, (min(reference_edges.shape) - 1) // 2),
    )
    tolerance = (
        float(config.primary_edge_tolerance)
        if config.extra_edge_tolerance is None
        else max(0.0, float(config.extra_edge_tolerance))
    )
    tiles: list[dict[str, Any]] = []
    y_positions = list(range(0, max(1, height - size + 1), stride))
    x_positions = list(range(0, max(1, width - size + 1), stride))
    if not y_positions or y_positions[-1] + size < height:
        y_positions.append(max(0, height - size))
    if not x_positions or x_positions[-1] + size < width:
        x_positions.append(max(0, width - size))
    for y0 in sorted(set(y_positions)):
        y1 = min(height, y0 + size)
        for x0 in sorted(set(x_positions)):
            x1 = min(width, x0 + size)
            ref = reference_edges[y0:y1, x0:x1]
            rendered = rendered_edges[y0:y1, x0:x1]
            ref_dist = distance_to_rendered[y0:y1, x0:x1][ref]
            svg_dist = distance_to_reference[y0:y1, x0:x1][rendered]
            distances = np.concatenate(
                (
                    ref_dist[np.isfinite(ref_dist)],
                    svg_dist[np.isfinite(svg_dist)],
                )
            )
            boundary_band = (
                distance_to_reference[y0:y1, x0:x1] <= radius
            ) | (distance_to_rendered[y0:y1, x0:x1] <= radius)
            if frame_margin:
                valid = np.ones(ref.shape, dtype=bool)
                top = max(0, frame_margin - y0)
                left = max(0, frame_margin - x0)
                bottom = min(y1 - y0, height - frame_margin - y0)
                right = min(x1 - x0, width - frame_margin - x0)
                valid[:top] = False
                valid[bottom:] = False
                valid[:, :left] = False
                valid[:, right:] = False
                boundary_band &= valid
            selected_delta = delta_e[y0:y1, x0:x1][boundary_band]
            extra = rendered & (
                distance_to_reference[y0:y1, x0:x1] > tolerance
            )
            missing = ref & (distance_to_rendered[y0:y1, x0:x1] > tolerance)
            entry = {
                "x_x4": int(x0),
                "y_x4": int(y0),
                "width_x4": int(x1 - x0),
                "height_x4": int(y1 - y0),
                "boundary_p95_source_pixels": (
                    float(np.percentile(distances, 95.0) / config.scale)
                    if distances.size
                    else None
                ),
                "boundary_delta_e_p90": (
                    float(np.percentile(selected_delta, 90.0))
                    if selected_delta.size
                    else None
                ),
                "extra_edge_pixels_x4": int(np.count_nonzero(extra)),
                "missing_edge_pixels_x4": int(np.count_nonzero(missing)),
            }
            entry["score"] = max(
                float(entry["boundary_delta_e_p90"] or 0.0),
                4.0 * float(entry["boundary_p95_source_pixels"] or 0.0),
            )
            tiles.append(entry)
    worst = max(tiles, key=lambda item: item["score"]) if tiles else None
    worst_delta = max(
        tiles,
        key=lambda item: float(item["boundary_delta_e_p90"] or -1.0),
    ) if tiles else None
    worst_extra = max(tiles, key=lambda item: item["extra_edge_pixels_x4"]) if tiles else None
    return {
        "tile_size_x4": size,
        "tile_stride_x4": stride,
        "tile_size_source_pixels": float(size / config.scale),
        "tile_count": len(tiles),
        "worst_tile": worst,
        "worst_boundary_delta_e_tile": worst_delta,
        "worst_extra_edge_tile": worst_extra,
    }


def _boundary_metrics(
    reference_edges: BoolImage,
    rendered_edges: BoolImage,
    config: EvaluationConfig,
) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    reference_count = int(np.count_nonzero(reference_edges))
    rendered_count = int(np.count_nonzero(rendered_edges))

    distance_to_rendered = (
        ndimage.distance_transform_edt(~rendered_edges).astype(
            np.float32, copy=False
        )
        if rendered_count
        else np.full(reference_edges.shape, np.inf, dtype=np.float32)
    )
    distance_to_reference = (
        ndimage.distance_transform_edt(~reference_edges).astype(
            np.float32, copy=False
        )
        if reference_count
        else np.full(reference_edges.shape, np.inf, dtype=np.float32)
    )
    reference_distances = distance_to_rendered[reference_edges]
    rendered_distances = distance_to_reference[rendered_edges]

    tolerance_metrics: list[dict[str, float]] = []
    requested = set(float(value) for value in config.edge_tolerances)
    requested.add(float(config.primary_edge_tolerance))
    for tolerance in sorted(requested):
        recall = (
            float(np.mean(reference_distances <= tolerance))
            if reference_count
            else 1.0
        )
        precision = (
            float(np.mean(rendered_distances <= tolerance))
            if rendered_count
            else float(reference_count == 0)
        )
        tolerance_metrics.append(
            {
                "tolerance_x4_pixels": tolerance,
                "tolerance_source_pixels": tolerance / float(config.scale),
                "recall": recall,
                "precision": precision,
                "f1": _f1(recall, precision),
            }
        )
    primary = next(
        item
        for item in tolerance_metrics
        if item["tolerance_x4_pixels"] == float(config.primary_edge_tolerance)
    )
    result: dict[str, Any] = {
        "reference_edge_pixels": reference_count,
        "rendered_edge_pixels": rendered_count,
        "primary": primary,
        "by_tolerance": tolerance_metrics,
        "symmetric_distance": _distance_summary(
            reference_distances,
            rendered_distances,
            scale=config.scale,
        ),
        "reference_to_svg_mean_source_pixels": (
            _nullable(float(np.mean(reference_distances)) / config.scale)
            if reference_distances.size
            else None
        ),
        "svg_to_reference_mean_source_pixels": (
            _nullable(float(np.mean(rendered_distances)) / config.scale)
            if rendered_distances.size
            else None
        ),
    }
    tolerance = (
        float(config.primary_edge_tolerance)
        if config.extra_edge_tolerance is None
        else max(0.0, float(config.extra_edge_tolerance))
    )
    extra = rendered_edges & (distance_to_reference > tolerance)
    missing = reference_edges & (distance_to_rendered > tolerance)
    result["extra_edge_tolerance_x4_pixels"] = tolerance
    result["extra_edge_tolerance_source_pixels"] = tolerance / float(config.scale)
    result["extra_edge_components"] = _component_summary(
        extra,
        scale=config.scale,
        minimum_area=config.extra_edge_min_component_area,
    )
    result["missing_edge_components"] = _component_summary(
        missing,
        scale=config.scale,
        minimum_area=config.extra_edge_min_component_area,
    )
    return result, distance_to_rendered, distance_to_reference


def _micro_artifact_metrics(boundary: Mapping[str, Any]) -> dict[str, float | int]:
    """Score connected extra/missing edge fragments independently.

    The ordinary boundary score is distance tolerant and therefore treats a
    cloud of short fragments similarly to one correctly placed contour.  The
    x4 review image shows that these fragments are a first-class failure, so
    this metric combines their area density with their connected-component
    count.  It is derived only from the reference/render edge maps.
    """

    reference_pixels = max(1, int(boundary.get("reference_edge_pixels") or 0))
    extra = boundary.get("extra_edge_components") or {}
    missing = boundary.get("missing_edge_components") or {}
    extra_pixels = int(extra.get("total_pixels_x4") or 0)
    missing_pixels = int(missing.get("total_pixels_x4") or 0)
    extra_count = int(extra.get("count_over_minimum_area") or extra.get("count") or 0)
    missing_count = int(
        missing.get("count_over_minimum_area") or missing.get("count") or 0
    )
    area_density = (extra_pixels + missing_pixels) / float(reference_pixels)
    component_load = (extra_count + missing_count) / 1000.0
    score = float(np.exp(-0.55 * area_density - 0.18 * component_load))
    return {
        "score": score,
        "extra_pixels_x4": extra_pixels,
        "missing_pixels_x4": missing_pixels,
        "extra_components": extra_count,
        "missing_components": missing_count,
        "area_density": float(area_density),
        "component_load_per_1000": float(component_load),
    }


def _fidelity_metrics(
    reference: FloatImage,
    rendered: FloatImage,
    delta_e: NDArray[np.float32],
    mask: BoolImage | None = None,
) -> dict[str, float | int | None]:
    selected = delta_e[mask] if mask is not None else delta_e.reshape(-1)
    if not selected.size:
        return {
            "pixel_count": 0,
            "delta_e00_mean": None,
            "delta_e00_p90": None,
            "delta_e00_p99": None,
            "within_delta_e00_2_3": None,
        }
    report: dict[str, float | int | None] = {
        "pixel_count": int(selected.size),
        "delta_e00_mean": float(np.mean(selected)),
        "delta_e00_p90": float(np.percentile(selected, 90.0)),
        "delta_e00_p99": float(np.percentile(selected, 99.0)),
        "within_delta_e00_2_3": float(np.mean(selected <= 2.3)),
        "over_delta_e00_5": float(np.mean(selected > 5.0)),
        "over_delta_e00_10": float(np.mean(selected > 10.0)),
    }
    if mask is None:
        squared_sum = 0.0
        pixel_count = 0
        for top in range(0, reference.shape[0], 256):
            bottom = min(reference.shape[0], top + 256)
            difference = reference[top:bottom].astype(np.float32) - rendered[
                top:bottom
            ].astype(np.float32)
            squared_sum += float(np.sum(np.square(difference), dtype=np.float64))
            pixel_count += int(difference.size)
        mean_squared_error = squared_sum / max(1, pixel_count)
        report["psnr"] = (
            None
            if mean_squared_error == 0.0
            else float(10.0 * np.log10(1.0 / mean_squared_error))
        )
        reference_ssim = reference[::4, ::4]
        rendered_ssim = rendered[::4, ::4]
        minimum_dimension = min(reference_ssim.shape[:2])
        window = min(7, minimum_dimension if minimum_dimension % 2 else minimum_dimension - 1)
        report["ssim"] = (
            _nullable(
                float(
                    metrics.structural_similarity(
                        reference_ssim,
                        rendered_ssim,
                        channel_axis=2,
                        data_range=1.0,
                        win_size=window,
                    )
                )
            )
            if window >= 3
            else None
        )
    return report


def _delta_e2000_tiled(
    reference: FloatImage,
    rendered: FloatImage,
    *,
    tile_height: int = 32,
    storage_path: Path | None = None,
) -> NDArray[np.float32]:
    """Compute DeltaE00 without holding both full Lab conversion graphs.

    A 5016x5016 car render is large enough that the vectorised colour
    conversion's temporary arrays can exceed the evaluator's memory limit.
    CIEDE2000 is pixel-local, so vertical tiling is numerically equivalent and
    keeps peak memory bounded.
    """

    if reference.shape != rendered.shape:
        raise ValueError("reference and rendered images must have equal shapes")
    height = int(reference.shape[0])
    # Half precision is sufficient for the report's DeltaE tails (the score
    # is not used for colour reconstruction) and halves the persistent field.
    shape = (height, int(reference.shape[1]))
    result: NDArray[np.float32]
    if storage_path is None:
        result = np.empty(shape, dtype=np.float16)
    else:
        result = np.memmap(storage_path, mode="w+", dtype=np.float16, shape=shape)
    step = max(1, int(tile_height))
    for top in range(0, height, step):
        bottom = min(height, top + step)
        reference_lab = srgb_to_lab(reference[top:bottom])
        rendered_lab = srgb_to_lab(rendered[top:bottom])
        result[top:bottom] = delta_e2000(reference_lab, rendered_lab)
        del reference_lab, rendered_lab
    if isinstance(result, np.memmap):
        result.flush()
    return result


def _complexity_metrics(
    complexity: Mapping[str, int | float],
    *,
    reference_edge_pixels: int,
    width: int,
    height: int,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Normalise SVG structure so micro-path overfitting is visible.

    ``segment_count`` alone misses fragmentation: splitting one contour into
    many paths is also costly.  The weighted unit count consequently includes
    three units per path and two per gradient definition.  The resulting
    efficiency score is monotonic and bounded, making it safe to combine with
    pixel metrics while retaining all raw counts for inspection.
    """

    def count(name: str) -> int:
        value = complexity.get(name, 0)
        return max(0, int(value))

    path_count = count("path_count")
    segment_count = count("segment_count")
    gradient_count = count("linear_gradient_count") + count("radial_gradient_count")
    weighted_units = segment_count + 3 * path_count + 2 * gradient_count
    edge_pixels = max(1, int(reference_edge_pixels))
    units_per_edge = float(weighted_units / edge_pixels)
    source_megapixels = (float(width) * float(height)) / (4.0 * 4.0 * 1_000_000.0)
    units_per_megapixel = (
        float(weighted_units / source_megapixels)
        if source_megapixels > 0.0
        else None
    )
    target = max(float(config.complexity_target_units_per_edge), 1e-9)
    efficiency = 1.0 / (1.0 + units_per_edge / target)
    primitive_units = (
        count("line_element_count")
        + count("rect_count")
        + count("circle_count")
        + int(1.2 * count("ellipse_count"))
        + int(1.5 * count("polygon_count"))
        + int(1.5 * count("polyline_count"))
        + 2 * count("quadratic_count")
        + 3 * count("cubic_count")
        + 2 * count("arc_count")
    )
    vector_score = 1.0 / (1.0 + primitive_units / max(1.0, float(edge_pixels) * 0.08))
    return {
        "counts": {str(key): int(value) for key, value in complexity.items()},
        "weighted_geometry_units": int(weighted_units),
        "source_megapixels": source_megapixels,
        "weighted_units_per_source_megapixel": units_per_megapixel,
        "weighted_units_per_reference_edge_pixel": units_per_edge,
        "path_to_segment_ratio": (
            float(path_count / segment_count) if segment_count else None
        ),
        "target_units_per_reference_edge_pixel": target,
        "complexity_efficiency": float(efficiency),
        "primitive_units": int(primitive_units),
        "vector_score": float(vector_score),
    }


def _quality_metrics(
    boundary: Mapping[str, Any],
    global_fidelity: Mapping[str, Any],
    boundary_band: Mapping[str, Any],
    local_failures: Mapping[str, Any],
    thin_lines: Mapping[str, Any],
    roughness: Mapping[str, Any],
    complexity: Mapping[str, Any],
    detail: Mapping[str, Any],
    coarse: Mapping[str, Any],
    pixel: Mapping[str, Any],
    geometry_wobble: Mapping[str, Any] | None,
    anchor_roughness: Mapping[str, Any] | None,
    open_stroke_roughness: Mapping[str, Any] | None,
    micro_artifacts: Mapping[str, Any],
    structural: Mapping[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Return a transparent, higher-is-better selection score.

    Boundary F1 remains useful, but a missing thin stroke is a first-class
    failure.  Colour error, worst local failures, line preservation, and SVG
    complexity are independent terms, so a large smooth fill cannot hide a
    deleted seam or contour.
    """

    primary_f1 = float(boundary["primary"]["f1"])
    mean_delta = float(global_fidelity.get("delta_e00_mean") or 0.0)
    p90_delta = float(global_fidelity.get("delta_e00_p90") or 0.0)
    within = float(global_fidelity.get("within_delta_e00_2_3") or 0.0)
    colour_score = (
        0.45 / (1.0 + mean_delta / 2.3)
        + 0.30 / (1.0 + p90_delta / 10.0)
        + 0.25 * within
    )
    worst_boundary = local_failures.get("worst_tile") or {}
    worst_colour = local_failures.get("worst_boundary_delta_e_tile") or {}
    local_distance = max(
        float(worst_boundary.get("boundary_p95_source_pixels") or 0.0),
        float(worst_colour.get("boundary_p95_source_pixels") or 0.0),
    )
    local_delta = max(
        float(worst_boundary.get("boundary_delta_e_p90") or 0.0),
        float(worst_colour.get("boundary_delta_e_p90") or 0.0),
    )
    local_score = 1.0 / (1.0 + max(local_distance / 10.0, local_delta / 50.0))
    thin_line_score = float(thin_lines.get("line_score", 0.0))
    dark_core_score = float(
        (thin_lines.get("dark_core") or {}).get("score", 0.0)
    )
    reference_line_pixels = max(
        1.0, float(thin_lines.get("reference_line_pixels") or 0.0)
    )
    missing_line_pixels = float(thin_lines.get("missing_source_pixels") or 0.0)
    missing_line_ratio = min(1.0, missing_line_pixels / reference_line_pixels)
    # The exponential term is intentionally non-linear: losing a handful of
    # antialiased pixels is cheap, while repeatedly deleting whole short
    # strokes quickly becomes a first-class failure.  Multiplication by the
    # dark core score prevents a neighbouring red/blue fill from satisfying
    # the line-recall term after the black line itself has disappeared.
    line_integrity_score = float(
        np.clip(
            (0.90 * float(thin_lines.get("recall", 0.0))
             + 0.10 * float(thin_lines.get("precision", 0.0)))
            * max(0.0, dark_core_score)
            * np.exp(-1.5 * missing_line_ratio),
            0.0,
            1.0,
        )
    )
    roughness_score = float(roughness.get("score", 0.0))
    complexity_score = float(complexity["complexity_efficiency"])
    detail_score = float(detail.get("score", 0.0))
    coarse_score = float(coarse.get("score", 0.0))
    pixel_score = float(pixel.get("score", 0.0))
    normalized_wobble = float(
        (geometry_wobble or {}).get("normalized_control_wobble", 0.0)
    )
    geometry_wobble_score = 1.0 / (1.0 + 200.0 * max(0.0, normalized_wobble))
    normalized_anchor_wobble = float(
        (anchor_roughness or {}).get("normalized_anchor_wobble", 0.0)
    )
    anchor_roughness_score = 1.0 / (
        1.0 + 0.5 * max(0.0, normalized_anchor_wobble)
    )
    normalized_open_wobble = float(
        (open_stroke_roughness or {}).get("normalized_anchor_wobble", 0.0)
    )
    open_stroke_roughness_score = 1.0 / (
        1.0 + 0.5 * max(0.0, normalized_open_wobble)
    )
    structural_score = float(np.clip(structural.get("score", 0.0), 0.0, 1.0))
    weights = {
        "boundary": max(0.0, float(config.quality_weight_boundary)),
        "colour": max(0.0, float(config.quality_weight_colour)),
        "local": max(0.0, float(config.quality_weight_local)),
        # The defaults deliberately give a deleted line roughly the same
        # impact as a substantial colour/complexity regression.
        "thin_line": max(0.0, float(config.quality_weight_thin_line)),
        "dark_core": max(0.0, float(config.quality_weight_dark_core)),
        "line_integrity": max(0.0, float(config.quality_weight_line_integrity)),
        "roughness": max(0.0, float(config.quality_weight_roughness)),
        "complexity": max(0.0, float(config.quality_weight_complexity)),
        "detail": max(0.0, float(config.quality_weight_detail)),
        "coarse": max(0.0, float(config.quality_weight_coarse)),
        "pixel": max(0.0, float(config.quality_weight_pixel)),
        "geometry_wobble": max(0.0, float(config.quality_weight_geometry_wobble)),
        "anchor_roughness": (
            max(0.0, float(config.quality_weight_anchor_roughness))
            if anchor_roughness is not None
            else 0.0
        ),
        "open_stroke_roughness": (
            max(0.0, float(config.quality_weight_open_stroke_roughness))
            if open_stroke_roughness is not None
            else 0.0
        ),
        "micro_artifact": max(0.0, float(config.quality_weight_micro_artifact)),
        "structural": max(0.0, float(config.quality_weight_structural)),
    }
    weight_total = sum(weights.values()) or 1.0
    score = (
        weights["boundary"] * primary_f1
        + weights["colour"] * colour_score
        + weights["local"] * local_score
        + weights["thin_line"] * thin_line_score
        + weights["dark_core"] * dark_core_score
        + weights["line_integrity"] * line_integrity_score
        + weights["roughness"] * roughness_score
        + weights["complexity"] * complexity_score
        + weights["detail"] * detail_score
        + weights["coarse"] * coarse_score
        + weights["pixel"] * pixel_score
        + weights["geometry_wobble"] * geometry_wobble_score
        + weights["anchor_roughness"] * anchor_roughness_score
        + weights["open_stroke_roughness"] * open_stroke_roughness_score
        + weights["micro_artifact"] * float(micro_artifacts.get("score", 0.0))
        + weights["structural"] * structural_score
    ) / weight_total
    return {
        "higher_is_better": True,
        "score": float(score),
        "components": {
            "boundary_f1": primary_f1,
            "colour_fidelity": float(colour_score),
            "local_fidelity": float(local_score),
            "thin_line_preservation": thin_line_score,
            "dark_core_preservation": dark_core_score,
            "line_integrity": line_integrity_score,
            "edge_smoothness": roughness_score,
            "complexity_efficiency": complexity_score,
            "detail_fidelity": detail_score,
            "coarse_fidelity": coarse_score,
            "pixel_fidelity": pixel_score,
            "geometry_wobble": float(geometry_wobble_score),
            "anchor_roughness": float(anchor_roughness_score),
            "open_stroke_roughness": float(open_stroke_roughness_score),
            "micro_artifact_fidelity": float(micro_artifacts.get("score", 0.0)),
            "structural_line_topology": structural_score,
        },
        "local_worst_boundary_p95_source_pixels": local_distance,
        "local_worst_boundary_delta_e_p90": local_delta,
        "weights": weights,
        "interpretation": (
            "Thin-line preservation is recall-weighted; dark outline cores "
            "are scored independently and excess high-frequency edge energy "
            "is penalised separately; "
            "boundary, colour, local failure, coarse structure, original-raster "
            "pixel fidelity, SVG complexity, vector control-point wobble, "
            "and anchor-chain roughness remain independent terms."
            " Open stroke-chain roughness is also measured separately so that"
            " seams and highlights cannot be hidden by filled-contour metrics."
            " Connected extra/missing edge fragments are penalized separately"
            " so local x4-visible artifacts cannot be hidden by global means."
            " Stroke topology is matched one-to-one at both native and x4 "
            "analysis scales, penalizing missing, duplicate, and unsupported "
            "line components."
        ),
    }


def _write_diagnostics(
    output_directory: Path,
    reference_edges: BoolImage,
    rendered_edges: BoolImage,
    distance_to_rendered: NDArray[np.float64],
    distance_to_reference: NDArray[np.float64],
    delta_e: NDArray[np.float32],
    thin_line_reference: BoolImage,
    thin_line_rendered: BoolImage,
    thin_line_missing: BoolImage,
    config: EvaluationConfig,
) -> None:
    reference_rgb = np.repeat(reference_edges[..., None], 3, axis=2).astype(np.float32)
    rendered_rgb = np.repeat(rendered_edges[..., None], 3, axis=2).astype(np.float32)
    save_rgb(reference_rgb, output_directory / "edges-reference.png")
    save_rgb(rendered_rgb, output_directory / "edges-svg.png")

    tolerance = float(config.primary_edge_tolerance)
    missing = reference_edges & (distance_to_rendered > tolerance)
    extra = rendered_edges & (distance_to_reference > tolerance)
    missing_rgb = np.ones((*missing.shape, 3), dtype=np.float32)
    missing_rgb[missing] = (0.90, 0.05, 0.05)
    extra_rgb = np.ones((*extra.shape, 3), dtype=np.float32)
    extra_rgb[extra] = (0.05, 0.25, 0.95)
    save_rgb(missing_rgb, output_directory / "missing-edges.png")
    save_rgb(extra_rgb, output_directory / "extra-edges.png")

    overlay = np.zeros((*reference_edges.shape, 3), dtype=np.float32)
    overlay[reference_edges] = (1.0, 0.1, 0.45)
    overlay[rendered_edges] += (0.0, 0.85, 1.0)
    save_rgb(np.clip(overlay, 0.0, 1.0), output_directory / "overlay.png")

    cap = max(tolerance * 4.0, 1.0)
    distances = np.clip(distance_to_rendered / cap, 0.0, 1.0)
    distance_rgb = np.zeros((*reference_edges.shape, 3), dtype=np.float32)
    distance_rgb[..., 0] = distances.astype(np.float32)
    distance_rgb[..., 1] = (1.0 - distances).astype(np.float32)
    distance_rgb[~reference_edges] = 0.0
    save_rgb(distance_rgb, output_directory / "boundary-distance.png")

    delta_scale = np.clip(delta_e / 20.0, 0.0, 1.0)
    delta_rgb = np.zeros((*delta_e.shape, 3), dtype=np.float32)
    delta_rgb[..., 0] = delta_scale
    delta_rgb[..., 1] = np.clip(1.0 - delta_scale * 1.5, 0.0, 1.0)
    save_rgb(delta_rgb, output_directory / "delta-e00.png")

    reference_lines = np.repeat(thin_line_reference[..., None], 3, axis=2).astype(
        np.float32
    )
    rendered_lines = np.repeat(thin_line_rendered[..., None], 3, axis=2).astype(
        np.float32
    )
    missing_lines = np.ones((*thin_line_missing.shape, 3), dtype=np.float32)
    missing_lines[thin_line_missing] = (0.95, 0.05, 0.05)
    save_rgb(reference_lines, output_directory / "thin-lines-reference.png")
    save_rgb(rendered_lines, output_directory / "thin-lines-svg.png")
    save_rgb(missing_lines, output_directory / "thin-lines-missing.png")


def evaluate_x4_images(
    reference: FloatImage,
    rendered: FloatImage,
    output_directory: str | Path,
    *,
    config: EvaluationConfig | None = None,
    svg_complexity: Mapping[str, int | float] | None = None,
    geometry_wobble: Mapping[str, int | float] | None = None,
    anchor_roughness: Mapping[str, int | float] | None = None,
    open_stroke_roughness: Mapping[str, int | float] | None = None,
    pixel_reference: FloatImage | None = None,
    native_rendered: FloatImage | None = None,
) -> dict[str, Any]:
    """Compare equal-sized x4 images and write diagnostic rasters."""

    resolved = config or EvaluationConfig()
    reference_value = np.asarray(reference, dtype=np.float32)
    rendered_value = np.asarray(rendered, dtype=np.float32)
    if reference_value.shape != rendered_value.shape:
        raise ValueError(
            "x4 reference and SVG render must have equal dimensions: "
            f"{reference_value.shape} != {rendered_value.shape}"
        )
    if reference_value.ndim != 3 or reference_value.shape[2] != 3:
        raise ValueError("evaluation images must be HxWx3 RGB arrays")
    pixel_reference_value = (
        np.asarray(pixel_reference, dtype=np.float32)
        if pixel_reference is not None
        else reference_value
    )
    if pixel_reference_value.ndim != 3 or pixel_reference_value.shape[2] != 3:
        raise ValueError("pixel reference must be an HxWx3 RGB array")
    native_reference = pixel_reference_value
    native_output = (
        np.asarray(native_rendered, dtype=np.float32)
        if native_rendered is not None
        else resize_image(rendered_value, native_reference.shape[:2])
    )
    if native_output.shape != native_reference.shape:
        raise ValueError(
            "native reference and SVG render must have equal dimensions: "
            f"{native_reference.shape} != {native_output.shape}"
        )

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference_edges = _edge_map(reference_value, resolved)
    rendered_edges = _edge_map(rendered_value, resolved)
    # Compute the small source-scale roughness statistic before allocating the
    # large x4 distance/DeltaE fields below.
    roughness = _edge_roughness_metrics(
        reference_value,
        rendered_value,
        reference_edges,
        rendered_edges,
        resolved,
    )
    detail = _detail_fidelity_metrics(reference_value, rendered_value, resolved)
    coarse = _coarse_fidelity_metrics(reference_value, rendered_value, resolved)
    boundary, distance_to_rendered, distance_to_reference = _boundary_metrics(
        reference_edges,
        rendered_edges,
        resolved,
    )
    delta_storage = directory / ".delta-e00.dat"
    delta_e = _delta_e2000_tiled(
        reference_value,
        rendered_value,
        storage_path=delta_storage,
    )
    boundary_band = distance_to_reference <= float(resolved.boundary_band_radius)
    local_boundary = _tile_boundary_metrics(
        reference_edges,
        rendered_edges,
        distance_to_rendered,
        distance_to_reference,
        delta_e,
        resolved,
    )
    thin_lines = thin_line_metrics(
        reference_value,
        rendered_value,
        tolerance=resolved.thin_line_tolerance,
        frame_margin=resolved.edge_frame_margin,
        config=resolved,
    )
    thin_line_reference = _thin_line_mask(
        reference_value, resolved, frame_margin=resolved.edge_frame_margin
    )
    thin_line_rendered = _thin_line_mask(
        rendered_value, resolved, frame_margin=resolved.edge_frame_margin
    )
    thin_line_distance = (
        ndimage.distance_transform_cdt(~thin_line_rendered, metric="chessboard")
        if np.any(thin_line_rendered)
        else np.full(thin_line_reference.shape, np.inf)
    )
    thin_line_missing = thin_line_reference & (
        thin_line_distance > float(resolved.thin_line_tolerance)
    )
    global_fidelity = _fidelity_metrics(
        reference_value,
        rendered_value,
        delta_e,
    )
    pixel = _pixel_fidelity_metrics(
        pixel_reference_value,
        rendered_value,
        resolved,
    )
    native_filtered = _filtered_raster_metrics(
        native_reference, native_output, scale=1, config=resolved
    )
    x4_filtered = _filtered_raster_metrics(
        reference_value, rendered_value, scale=resolved.scale, config=resolved
    )
    native_structural = structural_line_metrics(
        native_reference,
        native_output,
        scale=1,
        config=resolved,
    )
    x4_structural = structural_line_metrics(
        reference_value,
        rendered_value,
        scale=resolved.scale,
        config=resolved,
    )
    structural = {
        "native": native_structural,
        "x4": x4_structural,
        # The native image preserves the actual source-pixel topology; x4 is
        # equally important for defects that only become visible after
        # enlargement.  The geometric mean prevents one view from hiding a
        # near-total failure in the other.
        "score": float(
            np.sqrt(
                max(0.0, float(native_structural["score"]))
                * max(0.0, float(x4_structural["score"]))
            )
        ),
    }
    raster_selection = _raster_selection_metrics(
        native_filtered,
        x4_filtered,
        resolved,
        structural_score=float(structural["score"]),
        structural_report=structural,
    )
    micro_artifacts = _micro_artifact_metrics(boundary)
    boundary_band_report = {
        "radius_x4_pixels": float(resolved.boundary_band_radius),
        "radius_source_pixels": float(resolved.boundary_band_radius) / resolved.scale,
        **_fidelity_metrics(
            reference_value,
            rendered_value,
            delta_e,
            boundary_band,
        ),
    }
    report: dict[str, Any] = {
        "width": int(reference_value.shape[1]),
        "height": int(reference_value.shape[0]),
        "scale": int(resolved.scale),
        "edge_parameters": {
            "sigma_x4_pixels": float(resolved.edge_sigma),
            "low_threshold": float(resolved.edge_low_threshold),
            "high_threshold": float(resolved.edge_high_threshold),
            "primary_tolerance_x4_pixels": float(resolved.primary_edge_tolerance),
            "primary_tolerance_source_pixels": (
                float(resolved.primary_edge_tolerance) / resolved.scale
            ),
            "frame_margin_x4_pixels": int(resolved.edge_frame_margin),
            "frame_margin_source_pixels": (
                float(resolved.edge_frame_margin) / resolved.scale
            ),
        },
        "detail_parameters": {
            "source_tile_size": int(resolved.detail_tile_size),
            "minimum_reference_detail_pixels": int(resolved.detail_tile_min_pixels),
        },
        "pixel_parameters": {
            "reference": "original_raster" if pixel_reference is not None else "x4_reference_downsampled",
            "neighbourhood": f"{max(3, int(resolved.pixel_neighborhood_size) | 1)}x{max(3, int(resolved.pixel_neighborhood_size) | 1)}_mean_at_source_resolution",
        },
        "structural_parameters": {
            "reference": "native_source_and_realesrgan_x4",
            "match_tolerance_native_pixels": float(resolved.structural_match_tolerance),
            "minimum_component_pixels": int(resolved.structural_min_component_pixels),
            "duplicate_support": float(resolved.structural_duplicate_support),
            "missing_recall_threshold": float(resolved.structural_missing_recall),
            "score_floor": float(resolved.structural_score_floor),
            "max_line_missing_fraction": float(
                resolved.structural_max_line_missing_fraction
            ),
            "max_line_duplicate_fraction": float(
                resolved.structural_max_line_duplicate_fraction
            ),
            "analysis_max_dimension": int(resolved.structural_max_analysis_size),
            "matching": "greedy_one_to_one_component_assignment",
        },
        "boundary": boundary,
        "micro_artifacts": micro_artifacts,
        "global_fidelity": global_fidelity,
        "boundary_band_fidelity": boundary_band_report,
        "local_failures": local_boundary,
        "thin_lines": thin_lines,
        "edge_roughness": roughness,
        "detail_fidelity": detail,
        "coarse_fidelity": coarse,
        "pixel_fidelity": pixel,
        "raster": raster_selection,
        "structural": structural,
    }
    if svg_complexity is not None:
        complexity_report = _complexity_metrics(
            svg_complexity,
            reference_edge_pixels=boundary["reference_edge_pixels"],
            width=int(reference_value.shape[1]),
            height=int(reference_value.shape[0]),
            config=resolved,
        )
        report["svg_complexity"] = complexity_report
        if geometry_wobble is not None:
            report["svg_geometry_wobble"] = dict(geometry_wobble)
        if anchor_roughness is not None:
            report["svg_anchor_roughness"] = dict(anchor_roughness)
        if open_stroke_roughness is not None:
            report["svg_open_stroke_roughness"] = dict(open_stroke_roughness)
        report["quality"] = _quality_metrics(
            boundary,
            global_fidelity,
            boundary_band_report,
            local_boundary,
            thin_lines,
            roughness,
            complexity_report,
            detail,
            coarse,
            pixel,
            geometry_wobble,
            anchor_roughness,
            open_stroke_roughness,
            micro_artifacts,
            structural,
            resolved,
        )
        report["vector"] = {
            "complexity": complexity_report,
            "valid": bool(raster_selection["valid"]),
            "selection_score": float(complexity_report.get("vector_score", 0.0)),
        }
        report["selection"] = {
            "valid": bool(raster_selection["valid"]),
            "score": float(raster_selection["score"]),
            "failure_reasons": raster_selection["failure_reasons"],
            "raster_score": float(raster_selection["raster_score"]),
            "structural_score": float(raster_selection["structural_score"] or 0.0),
            "vector_score": float(complexity_report.get("vector_score", 0.0)),
            "interpretation": (
                "Native/x4 raster floors and structural topology gates are "
                "required; vector simplicity is compared only among candidates "
                "that preserve line ownership."
            ),
        }
    _write_diagnostics(
        directory,
        reference_edges,
        rendered_edges,
        distance_to_rendered,
        distance_to_reference,
        delta_e,
        thin_line_reference,
        thin_line_rendered,
        thin_line_missing,
        resolved,
    )
    # The car fixture is roughly 25M pixels at x4.  Release the large working
    # arrays before the CLI serialises the report so repeated candidate runs do
    # not hit the process memory limit after diagnostics have been written.
    del (
        reference_edges,
        rendered_edges,
        distance_to_rendered,
        distance_to_reference,
        delta_e,
        thin_line_reference,
        thin_line_rendered,
        thin_line_missing,
        thin_line_distance,
    )
    if delta_storage.exists():
        delta_storage.unlink()
    gc.collect()
    return report
