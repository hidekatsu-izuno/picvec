"""Post-hoc x4 evaluation for picvec outputs.

Nothing in the Rust vectorization pipeline imports it or observes its results.
"""

from .evaluation import EvaluationConfig, evaluate_x4_images, structural_line_metrics

__all__ = ["EvaluationConfig", "evaluate_x4_images", "structural_line_metrics"]
