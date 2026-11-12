import unittest
from home_explore.perception_schema import decode_detection_response


class SchemaTests(unittest.TestCase):
    def raw(self, fmt, box):
        return dict(bbox_format=fmt, detections=[dict(label='bed', confidence=0.9, bbox=box)])

    def test_norm1000(self):
        result = decode_detection_response(self.raw('norm1000', [100, 200, 800, 900]), 512, 424)
        self.assertEqual(result['detections'][0]['bbox'], [0.1, 0.2, 0.8, 0.9])

    def test_pixels(self):
        result = decode_detection_response(self.raw('pixels', [100, 50, 400, 200]), 500, 250)
        self.assertEqual(result['detections'][0]['bbox'], [0.2, 0.2, 0.8, 0.8])

    def test_undeclared_coordinates_are_not_guessed(self):
        raw = self.raw('unknown', [100, 200, 800, 900])
        with self.assertRaises(ValueError):
            decode_detection_response(raw, 512, 424)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            decode_detection_response(self.raw('norm1000', [100, 200, 1100, 900]), 512, 424)
