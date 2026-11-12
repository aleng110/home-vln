import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from home_explore.arrival_policy import (
    arrival_geometry,
    geometries_agree,
    arrival_distance,
    within_arrival_range,
    surface_distance,
    _bbox_depth_region,
)


class ArrivalPolicyTests(unittest.TestCase):
    def camera(self):
        q = Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).as_quat()
        return [120, 0, 60, 0, 120, 60, 0, 0, 1], [0.0, 0.0, 1.3], q

    def geometry(self, depth, box=(0, 0, 1, 1), mask=None):
        k, t, q = self.camera()
        return arrival_geometry(box, depth, k, t, q, mask=mask)

    def test_instance_mask_excludes_foreground_and_is_intersected_with_confirmed_box(self):
        depth = np.ones((120, 120), np.float32)
        depth[30:90, 30:90] = 2.0
        mask = np.ones(depth.shape, bool)
        geometry = self.geometry(depth, box=(0.25, 0.25, 0.75, 0.75), mask=mask)
        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry['point'][0], 2.0)
        self.assertEqual(geometry['method'], 'mask_arrival_front_depth_band')
        self.assertIsNone(geometry['reference_depth_delta'])
        self.assertTrue(geometry['independent_range'])

    def test_upper_central_fallback_excludes_lower_foreground_coffee_table(self):
        depth = np.full((120, 120), 2.0, np.float32)
        depth[75:, :] = 1.0
        geometry = self.geometry(depth)
        self.assertIsNotNone(geometry)
        self.assertAlmostEqual(geometry['point'][0], 2.0)
        self.assertEqual(geometry['method'], 'bbox_depth_region_arrival_front_depth_band')

    def test_connected_sloping_seat_extends_beyond_upper_seed_core(self):
        depth = np.full((120, 120), 2.0, np.float32)
        depth[45:100] = np.linspace(2.0, 1.0, 55)[:, None]
        geometry = self.geometry(depth)
        self.assertIsNotNone(geometry)
        self.assertLess(geometry['point'][0], 1.4)
        self.assertGreater(geometry['depth_region']['seed_depth'], 1.5)

    def test_depth_gap_stops_foreground_even_inside_grown_bbox(self):
        depth = np.full((120, 120), 2.0, np.float32)
        depth[72:] = 1.3
        k, t, q = self.camera()
        mask, meta = _bbox_depth_region((0, 0, 1, 1), depth, k, t, q)
        self.assertIsNotNone(mask)
        self.assertFalse(mask[75:, :].any())
        self.assertAlmostEqual(self.geometry(depth)['point'][0], 2.0)

    def test_disconnected_wall_background_is_not_grown(self):
        depth = np.full((120, 120), 3.5, np.float32)
        depth[20:105, 25:95] = 2.0
        k, t, q = self.camera()
        mask, meta = _bbox_depth_region((0, 0, 1, 1), depth, k, t, q)
        self.assertTrue(mask[50, 60])
        self.assertFalse(mask[:, 100:].any())
        self.assertAlmostEqual(self.geometry(depth)['point'][0], 2.0)

    def test_empty_instance_mask_never_falls_back_to_bbox(self):
        depth = np.full((120, 120), 2.0, np.float32)
        self.assertIsNone(self.geometry(depth, mask=np.zeros(depth.shape, bool)))

    def test_sparse_or_invalid_core_does_not_fall_back_to_lower_bbox(self):
        depth = np.full((120, 120), np.nan, np.float32)
        depth[80:, :] = 1.0
        self.assertIsNone(self.geometry(depth))
        depth[30, 60] = 2.0
        self.assertIsNone(self.geometry(depth))
        self.assertIsNone(self.geometry(np.zeros((120, 120), np.float32)))

    def test_mask_alignment_error_fails_explicitly(self):
        with self.assertRaises(ValueError):
            self.geometry(np.ones((120, 120), np.float32), mask=np.ones((60, 60), bool))

    def test_two_geometries_must_agree_in_xy_and_depth_quality(self):
        first = self.geometry(np.full((120, 120), 2.0, np.float32))
        second = copy.deepcopy(first)
        second['point'][0] += 0.19
        self.assertTrue(geometries_agree(first, second))
        second['point'][0] += 0.02
        self.assertFalse(geometries_agree(first, second))
        second = copy.deepcopy(first)
        second['depth_spread'] = 0.13
        self.assertFalse(geometries_agree(first, second))
        second = copy.deepcopy(first)
        second['valid_points'] = 39
        self.assertFalse(geometries_agree(first, second))

    def test_far_fresh_surface_is_not_completed_as_near(self):
        geometry = self.geometry(np.full((120, 120), 2.0, np.float32))
        self.assertGreater(arrival_distance(geometry['point'], [0, 0, 0]), 1.9)
        self.assertFalse(within_arrival_range(geometry['point'], [0, 0, 0]))
        self.assertTrue(within_arrival_range(geometry['point'], [1.0, 0, 0]))

    def test_invalid_ranges_and_geometries_fail_closed(self):
        self.assertFalse(geometries_agree(None, {}))
        self.assertFalse(within_arrival_range([math.nan, 0, 1], [0, 0, 0]))
        self.assertFalse(within_arrival_range([math.inf, 0, 1], [0, 0, 0]))
        self.assertFalse(within_arrival_range(None, [0, 0, 0]))

    def test_surface_distance_uses_quantile_not_minimum_outlier(self):
        points = (
            [[1.0, 0.0, 0.5] for _ in range(30)]
            + [[2.0, 0.0, 0.5] for _ in range(65)]
            + [[0.1, 0.0, 0.5] for _ in range(5)]
        )
        geometry = {'points': points, 'point': [2.0, 0.0, 0.5]}
        self.assertAlmostEqual(surface_distance(geometry, [0, 0, 0]), 1.0)
        self.assertAlmostEqual(arrival_distance(geometry['point'], [0, 0, 0]), 2.0)

    def test_surface_distance_requires_enough_finite_samples(self):
        self.assertTrue(math.isinf(surface_distance({'points': [[1, 0, 1]] * 19}, [0, 0, 0])))
        self.assertTrue(math.isinf(surface_distance({'points': [[math.nan, 0, 1]] * 25}, [0, 0, 0])))
        self.assertTrue(math.isinf(surface_distance(None, [0, 0, 0])))


if __name__ == '__main__':
    unittest.main()
