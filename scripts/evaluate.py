# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pillow>=10",
#   "scikit-image>=0.26",
#   "scipy>=1.17",
# ]
# ///
"""Run picvec's independent raster-to-SVG quality evaluator."""

from picvec_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
