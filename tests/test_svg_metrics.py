import unittest
from xml.etree import ElementTree as ET

from picvec_eval.svg_metrics import _visible_stroke_paths, svg_open_stroke_roughness


class StrokeMetricTests(unittest.TestCase):
    path = 'M 0 0 L 5 1 L 10 -1 L 15 1 L 20 -1 L 25 0'

    def test_inherited_stroke_matches_explicit_presentation(self):
        explicit = f'<svg><path fill="none" stroke="black" d="{self.path}"/></svg>'
        inherited = f'<svg fill="none"><g stroke="black"><g><path d="{self.path}"/></g></g></svg>'
        expected = svg_open_stroke_roughness(explicit)
        self.assertGreater(expected['anchor_wobble_energy'], 0)
        self.assertEqual(svg_open_stroke_roughness(inherited), expected)

    def test_overrides_and_hidden_definitions(self):
        svg = '''<svg fill="none" stroke="black">
          <defs><path id="definition"/></defs>
          <g display="none"><path id="hidden"/></g>
          <g opacity="0"><path id="transparent"/></g>
          <path id="filled" fill="red"/>
          <path id="unstroked" style="stroke:none"/>
          <path id="zero-width" stroke-width="0"/>
          <path id="zero-opacity" stroke-opacity="0"/>
          <path id="inline" fill="red" style="fill: none; stroke: blue"/>
          <g visibility="hidden"><path id="visible" visibility="visible"/></g>
        </svg>'''
        self.assertEqual([p.get('id') for p in _visible_stroke_paths(ET.fromstring(svg))], ['inline', 'visible'])

    def test_optimized_commands_do_not_truncate_path(self):
        a = '<svg fill="none" stroke="black"><path d="M0 0 H5 V2 h5 v-2 H15 V2 H20 V0"/></svg>'
        b = '<svg fill="none" stroke="black"><path d="M0 0 L5 0 L5 2 L10 2 L10 0 L15 0 L15 2 L20 2 L20 0"/></svg>'
        self.assertEqual(svg_open_stroke_roughness(a), svg_open_stroke_roughness(b))
        a = f'<svg fill="none" stroke="black"><path d="M0 0 A5 5 0 0 0 5 1 Q7 3 10 -1 S12 2 15 1 T20 -1 L25 0"/></svg>'
        b = f'<svg fill="none" stroke="black"><path d="{self.path}"/></svg>'
        self.assertEqual(svg_open_stroke_roughness(a), svg_open_stroke_roughness(b))


if __name__ == '__main__':
    unittest.main()
