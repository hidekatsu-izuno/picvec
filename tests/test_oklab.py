"""Reference and tiled-evaluation checks for the shared 100-scaled OKLab units."""
import unittest
import numpy as np

from scripts.picvec_eval.support import srgb_to_oklab, delta_e_ok
from scripts.picvec_eval.evaluation import _delta_e_ok_tiled


class OklabTests(unittest.TestCase):
    def test_published_primary_coordinates(self):
        rgb = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        expected = [[0, 0, 0], [100, 0, 0], [62.7955, 22.4863, 12.5846],
                    [86.6440, -23.3888, 17.9498], [45.2014, -3.2457, -31.1528]]
        np.testing.assert_allclose(srgb_to_oklab(rgb), expected, atol=0.002)
        self.assertEqual(float(delta_e_ok(np.array([50, 0, 0]), np.array([50, 3, 4]))), 5.0)

    def test_tiled_distance_matches_direct_evaluation(self):
        reference = np.random.default_rng(42).random((35, 11, 3), dtype=np.float32)
        rendered = np.clip(reference * 0.9 + 0.02, 0, 1)
        direct = delta_e_ok(srgb_to_oklab(reference), srgb_to_oklab(rendered))
        for height in [1, 8, 32, 64]:
            np.testing.assert_allclose(_delta_e_ok_tiled(reference, rendered, tile_height=height),
                                       direct, atol=0.004, rtol=0.001)
        self.assertTrue(np.all(_delta_e_ok_tiled(reference, reference) == 0))


if __name__ == '__main__':
    unittest.main()
