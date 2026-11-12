import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from home_explore.view_policy import ranked_search, visual_yaw_delta, visible_target, framing
from home_explore.semantic import surface_geometry, Memory
from home_explore.grid import Grid


class ViewPolicyTests(unittest.TestCase):
    def optical(self):
        return Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).as_quat()

    def test_startup_prefers_forward_safe_candidate_over_cheaper_side_goal(self):
        candidates = [
            dict(x=0, y=1, yaw=0, score=-3, kind='frontier'),
            dict(x=2, y=0, yaw=0, score=0, kind='frontier'),
        ]
        result = ranked_search(candidates, (0, 0, 0), 0, startup=True)
        self.assertEqual((result[0]['x'], result[0]['y']), (2, 0))
        self.assertTrue(result[0]['forward_preferred'])

    def test_heading_preference_does_not_remove_only_available_side_route(self):
        result = ranked_search([dict(x=0, y=1, yaw=0, score=0, kind='frontier')], (0, 0, 0), 0, True)
        self.assertEqual(len(result), 1)

    def test_target_on_image_right_requires_right_turn(self):
        delta = visual_yaw_delta(
            [0.6, 0.2, 0.9, 0.8], [400, 0, 256, 0, 400, 212, 0, 0, 1], (424, 512), self.optical()
        )
        self.assertLess(delta, 0)
        centred = visual_yaw_delta(
            [0.3, 0.2, 0.7, 0.8], [400, 0, 256, 0, 400, 212, 0, 0, 1], (424, 512), self.optical()
        )
        self.assertAlmostEqual(centred, 0, places=7)

    def test_choose_instance_near_projected_reference_not_highest_score(self):
        detections = [
            dict(label='chair', bbox=[0.1, 0.1, 0.3, 0.8], confidence=0.99),
            dict(label='chair', bbox=[0.6, 0.1, 0.9, 0.8], confidence=0.7),
        ]
        self.assertEqual(visible_target(detections, 'chair', [0.75, 0.5])['confidence'], 0.7)

    def test_frame_boundary_is_recorded(self):
        self.assertTrue(framing([0, 0.2, 0.7, 0.9])['horizontal_clipped'])
        self.assertFalse(framing([0.2, 0.2, 0.8, 0.8])['clipped'])

    def test_reference_depth_rejects_nearer_bbox_background(self):
        depth = np.ones((80, 80), np.float32)
        depth[20:60, 20:60] = 2
        q = [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)]
        g = surface_geometry(
            [0, 0, 1, 1], depth, [80, 0, 40, 0, 80, 40, 0, 0, 1], [0, 0, 1], q, reference_point=[2, 0, 1]
        )
        self.assertAlmostEqual(g['point'][0], 2)
        self.assertEqual(g['method'], 'bbox_reference_depth_band')

    def test_wide_surface_is_not_bad_depth_consistency(self):
        depth = np.full((160, 320), 3.0, np.float32)
        g = surface_geometry(
            [0, 0, 1, 1],
            depth,
            [160, 0, 160, 0, 160, 80, 0, 0, 1],
            [0, 0, 1],
            self.optical(),
            reference_point=[3, 0, 1],
        )
        self.assertGreater(g['spread'], 0.35)
        self.assertAlmostEqual(g['depth_spread'], 0)
        self.assertAlmostEqual(g['reference_depth_delta'], 0)

    def test_docking_yaw_uses_observed_look_point(self):
        grid = Grid(np.zeros((100, 100), np.int16), 0.1, (0, 0))
        obj = dict(point=[5, 5, 0.5], look_at=[5.8, 5.3, 0.7], extent=[0.4, 0.4])
        goals = grid.docking(obj, (2, 2, 0))
        self.assertTrue(goals)
        for g in goals:
            self.assertAlmostEqual(g['yaw'], math.atan2(5.3 - g['y'], 5.8 - g['x']))


if __name__ == '__main__':
    unittest.main()
