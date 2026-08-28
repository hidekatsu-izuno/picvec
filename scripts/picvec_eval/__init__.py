"""Post-hoc x4 evaluation for picvec outputs.

Nothing in the Rust vectorization pipeline imports it or observes its results.
Evaluation symbols are loaded lazily so the standalone x4 generator does not
need SciPy or scikit-image merely to import ``picvec_eval.upscale``.
"""

from __future__ import annotations

from typing import Any


__all__ = ["EvaluationConfig", "evaluate_x4_images", "structural_line_metrics"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import evaluation

        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
