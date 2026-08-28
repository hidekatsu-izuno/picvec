"""Canonical, command-aware complexity metrics for SVG path geometry.

The vectorizer emits a mixture of absolute and relative commands, and some
paths contain more than one subpath.  Counting textual ``C`` tokens therefore
does not describe the geometry reliably.  This module tokenises the SVG path
grammar and counts geometric commands without changing the path itself.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from xml.etree import ElementTree as ET

import numpy as np


_TOKEN = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_PARAMS = {
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "S": 4,
    "Q": 4,
    "T": 2,
    "A": 7,
    "Z": 0,
}


@dataclass(frozen=True, slots=True)
class SvgPathComplexity:
    """Command-normalised counts for one SVG document."""

    path_count: int = 0
    subpath_count: int = 0
    segment_count: int = 0
    close_count: int = 0
    line_count: int = 0
    cubic_count: int = 0
    quadratic_count: int = 0
    arc_count: int = 0
    endpoint_count: int = 0
    linear_gradient_count: int = 0
    radial_gradient_count: int = 0
    line_element_count: int = 0
    rect_count: int = 0
    circle_count: int = 0
    ellipse_count: int = 0
    polygon_count: int = 0
    polyline_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _path_counts(data: str) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Return subpaths, segments, closes, lines, cubics, quadratics, arcs,
    endpoints, and a validity flag for one ``d`` attribute.

    A malformed path is ignored rather than partially counted.  This is safer
    for complexity reporting than presenting a plausible but incomplete count.
    """

    tokens = _TOKEN.findall(data)
    if not tokens or "".join(tokens) != re.sub(r"[\s,]+", "", data):
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    index = 0
    command: str | None = None
    first_moveto = False
    subpaths = segments = closes = lines = cubics = quadratics = arcs = endpoints = 0
    valid = True
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command.upper() == "Z":
                closes += 1
                segments += 1
                command = None
                continue
            first_moveto = command.upper() == "M"
        if command is None:
            valid = False
            break
        upper = command.upper()
        parameter_count = _PARAMS.get(upper)
        if parameter_count is None or parameter_count == 0:
            valid = False
            break
        remaining = len(tokens) - index
        if remaining < parameter_count:
            valid = False
            break
        if any(token.isalpha() for token in tokens[index : index + parameter_count]):
            valid = False
            break
        index += parameter_count
        if upper == "M":
            if first_moveto:
                subpaths += 1
                endpoints += 1
                first_moveto = False
            else:
                segments += 1
                lines += 1
                endpoints += 1
            command = "l" if command.islower() else "L"
        elif upper in {"L", "H", "V"}:
            segments += 1
            lines += 1
            endpoints += 1
        elif upper in {"C", "S"}:
            segments += 1
            cubics += 1
            endpoints += 1
        elif upper in {"Q", "T"}:
            segments += 1
            quadratics += 1
            endpoints += 1
        elif upper == "A":
            segments += 1
            arcs += 1
            endpoints += 1

    if not valid:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    return (subpaths, segments, closes, lines, cubics, quadratics, arcs, endpoints, 1)


def svg_complexity_root(root: ET.Element) -> SvgPathComplexity:
    """Measure a parsed SVG tree without serializing and parsing it again."""

    values = {
        "path_count": 0,
        "subpath_count": 0,
        "segment_count": 0,
        "close_count": 0,
        "line_count": 0,
        "cubic_count": 0,
        "quadratic_count": 0,
        "arc_count": 0,
        "endpoint_count": 0,
        "linear_gradient_count": 0,
        "radial_gradient_count": 0,
        "line_element_count": 0,
        "rect_count": 0,
        "circle_count": 0,
        "ellipse_count": 0,
        "polygon_count": 0,
        "polyline_count": 0,
    }
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "path":
            values["path_count"] += 1
            data = element.get("d")
            if not data:
                continue
            counts = _path_counts(data)
            keys = (
                "subpath_count",
                "segment_count",
                "close_count",
                "line_count",
                "cubic_count",
                "quadratic_count",
                "arc_count",
                "endpoint_count",
            )
            for key, count in zip(keys, counts[:-1]):
                values[key] += count
        elif name == "linearGradient":
            values["linear_gradient_count"] += 1
        elif name == "radialGradient":
            values["radial_gradient_count"] += 1
        elif name == "line":
            values["line_element_count"] += 1
        elif name == "rect":
            values["rect_count"] += 1
        elif name == "circle":
            values["circle_count"] += 1
        elif name == "ellipse":
            values["ellipse_count"] += 1
        elif name == "polygon":
            values["polygon_count"] += 1
        elif name == "polyline":
            values["polyline_count"] += 1
    return SvgPathComplexity(**values)


def svg_complexity(svg: str | bytes) -> SvgPathComplexity:
    """Parse an SVG document and return command-normalised complexity."""

    return svg_complexity_root(ET.fromstring(svg))


def svg_geometry_wobble(svg: str | bytes) -> dict[str, float | int]:
    """Measure control-point oscillation in editable cubic geometry.

    This is intentionally a vector-domain diagnostic.  It does not judge
    whether a source edge is important; it reports how much a path's cubic
    controls deviate from their endpoint chords.  Near-linear spans with
    unnecessary handles are the characteristic source of rippled fills and
    are therefore reported separately from raster fidelity.
    """

    root = ET.fromstring(svg)
    total_length = 0.0
    control_energy = 0.0
    near_linear_segments = 0
    cubic_segments = 0
    for element in root.iter():
        if _local_name(element.tag) != "path":
            continue
        data = element.get("d") or ""
        tokens = _TOKEN.findall(data)
        if not tokens or "".join(tokens) != re.sub(r"[\s,]+", "", data):
            continue
        index = 0
        current: np.ndarray | None = None
        subpath_start: np.ndarray | None = None
        while index < len(tokens):
            command = tokens[index]
            index += 1
            if command == "M" and index + 1 < len(tokens):
                current = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])), dtype=np.float64
                )
                subpath_start = current.copy()
                index += 2
            elif command == "L" and current is not None and index + 1 < len(tokens):
                endpoint = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])), dtype=np.float64
                )
                total_length += float(np.linalg.norm(endpoint - current))
                current = endpoint
                index += 2
            elif command == "C" and current is not None and index + 5 < len(tokens):
                first = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])), dtype=np.float64
                )
                second = np.asarray(
                    (float(tokens[index + 2]), float(tokens[index + 3])), dtype=np.float64
                )
                endpoint = np.asarray(
                    (float(tokens[index + 4]), float(tokens[index + 5])), dtype=np.float64
                )
                chord = endpoint - current
                length = float(np.linalg.norm(chord))
                if length > 1.0:
                    first_offset = first - current
                    second_offset = second - current
                    deviation = max(
                        abs(float(chord[0] * first_offset[1] - chord[1] * first_offset[0]))
                        / length,
                        abs(float(chord[0] * second_offset[1] - chord[1] * second_offset[0]))
                        / length,
                    )
                    total_length += length
                    cubic_segments += 1
                    if deviation <= 2.0:
                        near_linear_segments += 1
                        control_energy += (deviation * deviation) / length
                current = endpoint
                index += 6
            elif command == "Z":
                if current is not None and subpath_start is not None:
                    total_length += float(np.linalg.norm(subpath_start - current))
                current = subpath_start.copy() if subpath_start is not None else current
            else:
                # Relative and shorthand commands are already represented in
                # the complexity report; skip them here rather than inventing
                # a partial geometry interpretation.
                break
    normalized = control_energy / max(total_length, 1.0)
    return {
        "cubic_segments": int(cubic_segments),
        "near_linear_control_segments": int(near_linear_segments),
        "control_wobble_energy": float(control_energy),
        "normalized_control_wobble": float(normalized),
    }


def svg_anchor_roughness(svg: str | bytes) -> dict[str, float | int]:
    """Measure short-period anchor oscillation on smooth closed contours.

    Control handles can be perfectly collinear while the anchors themselves
    still follow a raster staircase.  This metric samples the anchor chain
    directly and measures each smooth anchor's distance from a wider local
    chord.  High-curvature corners are excluded, so the metric does not ask a
    genuine corner to become round.  It is a diagnostic only and never feeds
    back into vectorization.
    """

    root = ET.fromstring(svg)
    energy = 0.0
    smooth_anchors = 0
    closed_subpaths = 0
    for element in root.iter():
        if _local_name(element.tag) != "path":
            continue
        tokens = _TOKEN.findall(element.get("d") or "")
        if not tokens or "".join(tokens) != re.sub(r"[\s,]+", "", element.get("d") or ""):
            continue
        index = 0
        command: str | None = None
        start: np.ndarray | None = None
        anchors: list[np.ndarray] = []
        closed = False

        def flush(is_closed: bool) -> None:
            nonlocal energy, smooth_anchors, closed_subpaths, anchors
            if not is_closed or len(anchors) < 7 or start is None:
                anchors = []
                return
            closed_subpaths += 1
            values = np.asarray(anchors, dtype=np.float64)
            count = len(values)
            for position, point in enumerate(values):
                previous = values[(position - 1) % count]
                following = values[(position + 1) % count]
                incoming = previous - point
                outgoing = following - point
                incoming_length = float(np.linalg.norm(incoming))
                outgoing_length = float(np.linalg.norm(outgoing))
                if incoming_length < 1.0 or outgoing_length < 1.0:
                    continue
                continuation = float(
                    np.dot(-incoming, outgoing)
                    / (incoming_length * outgoing_length)
                )
                if continuation < 0.70:
                    continue
                chord = values[(position + 2) % count] - values[(position - 2) % count]
                chord_squared = float(np.dot(chord, chord))
                if chord_squared < 4.0:
                    continue
                fraction = float(
                    np.dot(point - values[(position - 2) % count], chord)
                    / chord_squared
                )
                projection = values[(position - 2) % count] + np.clip(
                    fraction, 0.0, 1.0
                ) * chord
                scale = max((incoming_length + outgoing_length) * 0.5, 1.0)
                residual = float(np.linalg.norm(point - projection)) / scale
                energy += residual * residual
                smooth_anchors += 1
            anchors = []

        while index < len(tokens):
            token = tokens[index]
            if token.isalpha():
                index += 1
                command = token
                if command.upper() == "Z":
                    closed = True
                    flush(True)
                    command = None
                    continue
            if command is None:
                break
            upper = command.upper()
            if upper == "M" and index + 1 < len(tokens):
                flush(closed)
                closed = False
                point = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])),
                    dtype=np.float64,
                )
                index += 2
                start = point.copy()
                anchors = [point]
                command = "L" if command.isupper() else "l"
            elif upper == "L" and index + 1 < len(tokens):
                point = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])),
                    dtype=np.float64,
                )
                index += 2
                anchors.append(point)
            elif upper == "C" and index + 5 < len(tokens):
                point = np.asarray(
                    (float(tokens[index + 4]), float(tokens[index + 5])),
                    dtype=np.float64,
                )
                index += 6
                anchors.append(point)
            else:
                break
    normalized = energy / max(float(smooth_anchors), 1.0)
    return {
        "smooth_anchor_count": int(smooth_anchors),
        "closed_subpaths": int(closed_subpaths),
        "anchor_wobble_energy": float(energy),
        "normalized_anchor_wobble": float(normalized),
    }


def svg_open_stroke_roughness(svg: str | bytes) -> dict[str, float | int]:
    """Measure short-period oscillation on visible, open stroke paths.

    Open contour overlays are common in raster-to-SVG output (seams,
    highlights, wheel arches).  ``svg_anchor_roughness`` intentionally only
    considers closed filled contours, so those overlays could be simplified
    without changing the reported geometry quality.  This companion metric
    uses the same curvature-aware local-chord residual for stroke-only paths;
    corners and very short runs are excluded.  It is diagnostic/selection
    metadata only and does not depend on element IDs or image-specific
    coordinates.
    """

    root = ET.fromstring(svg)
    energy = 0.0
    smooth_anchors = 0
    open_subpaths = 0
    total_length = 0.0

    def visible_stroke(element: ET.Element) -> bool:
        fill = (element.get("fill") or "").strip().lower()
        stroke = (element.get("stroke") or "").strip().lower()
        style = (element.get("style") or "").lower().replace(" ", "")
        if "fill:none" not in style and fill not in {"none", "transparent"}:
            return False
        if "stroke:none" in style or stroke in {"", "none", "transparent"}:
            return False
        return True

    for element in root.iter():
        if _local_name(element.tag) != "path" or not visible_stroke(element):
            continue
        data = element.get("d") or ""
        tokens = _TOKEN.findall(data)
        if not tokens or "".join(tokens) != re.sub(r"[\s,]+", "", data):
            continue
        anchors: list[np.ndarray] = []
        closed = False

        def flush() -> None:
            nonlocal energy, smooth_anchors, open_subpaths, total_length, anchors, closed
            if closed or len(anchors) < 5:
                anchors = []
                closed = False
                return
            open_subpaths += 1
            values = np.asarray(anchors, dtype=np.float64)
            if len(values) > 1:
                total_length += float(
                    np.linalg.norm(np.diff(values, axis=0), axis=1).sum()
                )
            for position in range(2, len(values) - 2):
                point = values[position]
                previous = values[position - 1]
                following = values[position + 1]
                incoming = previous - point
                outgoing = following - point
                incoming_length = float(np.linalg.norm(incoming))
                outgoing_length = float(np.linalg.norm(outgoing))
                if incoming_length < 1.0 or outgoing_length < 1.0:
                    continue
                continuation = float(
                    np.dot(-incoming, outgoing)
                    / (incoming_length * outgoing_length)
                )
                if continuation < 0.70:
                    continue
                left = values[position - 2]
                right = values[position + 2]
                chord = right - left
                chord_squared = float(np.dot(chord, chord))
                if chord_squared < 4.0:
                    continue
                fraction = float(np.dot(point - left, chord) / chord_squared)
                projection = left + np.clip(fraction, 0.0, 1.0) * chord
                scale = max((incoming_length + outgoing_length) * 0.5, 1.0)
                residual = float(np.linalg.norm(point - projection)) / scale
                energy += residual * residual
                smooth_anchors += 1
            anchors = []
            closed = False

        index = 0
        command: str | None = None
        current = np.zeros(2, dtype=np.float64)
        start = current.copy()
        while index < len(tokens):
            token = tokens[index]
            if token.isalpha():
                index += 1
                command = token
                if command.upper() == "Z":
                    closed = True
                    flush()
                    command = None
                    continue
            if command is None:
                break
            upper = command.upper()
            relative = command.islower()
            if upper == "M" and index + 1 < len(tokens):
                flush()
                point = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])),
                    dtype=np.float64,
                )
                index += 2
                if relative:
                    point += current
                current = point
                start = point.copy()
                anchors = [point]
                command = "l" if relative else "L"
            elif upper == "L" and index + 1 < len(tokens):
                point = np.asarray(
                    (float(tokens[index]), float(tokens[index + 1])),
                    dtype=np.float64,
                )
                index += 2
                if relative:
                    point += current
                current = point
                anchors.append(point)
            elif upper == "C" and index + 5 < len(tokens):
                point = np.asarray(
                    (float(tokens[index + 4]), float(tokens[index + 5])),
                    dtype=np.float64,
                )
                index += 6
                if relative:
                    point += current
                current = point
                anchors.append(point)
            else:
                break
        if anchors:
            flush()

    # A square-root length normalisation keeps the statistic comparable across
    # images without making a simplifier look worse merely because it removed
    # short oscillatory spans (the residual energy itself is accumulated once
    # per visible contour, rather than divided by the reduced anchor count).
    normalized = energy / max(total_length, 1.0) ** 0.5
    return {
        "smooth_anchor_count": int(smooth_anchors),
        "open_subpaths": int(open_subpaths),
        "anchor_wobble_energy": float(energy),
        "stroke_length": float(total_length),
        "normalized_anchor_wobble": float(normalized),
    }
