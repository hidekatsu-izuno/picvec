# Car headlight highlight regression

The brown shadow below the headlight in `sample/input/car.png` was caused
by bright-ridge sample filtering in `fit_all_internal`, not structural ink
or the overlap used to hide SVG seams.

Before the fix, face 1727 in the final pre-fit partition contained 512
pixels. Its bright-ridge classification reduced the fit and evaluation
set to 234 pixels. A five-stop linear gradient fitted this subset well
(mean DeltaE00 1.69 with its residual overlay), but extended the dark
endpoint colour `#913937` into the unsampled curved tip. Across the whole
face, its mean DeltaE00 was 8.85 versus 4.30 for the canonical solid.
The gradient was already present in the initial Paint fit, before
harmonization, Paint merging, geometry generation, or structural ink.

Highlight fitting now retains all valid Paint samples. Bright-ridge
membership describes the crest, not the complete spatial support of a
highlight face, and must not exclude its remaining native colour evidence.
Dark-ridge directional fitting retains its existing sampling policy.

The regression test `curved_car_highlight_keeps_colour_beyond_the_bright_ridge`
uses a 122 × 47 crop at source coordinate (192, 643):

- `src/test-data/car-highlight-source.png` contains the native underpaint.
- `src/test-data/car-highlight-mask.png` encodes face membership in red,
  valid Paint samples in green, and effective bright-ridge membership in
  blue. Blue reconstructs the original 234 selected samples plus the 32
  excluded Paint sites, preserving the majority classification and the
  exact effective fitting subset without depending on ridge detection.
- The surrounding pixels retain the fitting halo; a fixed surrounding
  Paint isolates the highlight fit from unrelated image regions.
- The test measures colour error in the curved tip (source x >= 270),
  requiring mean DeltaE00 below 5. The original implementation fails at
  approximately 22.

The normal CLI output and the README car comparison were regenerated.
