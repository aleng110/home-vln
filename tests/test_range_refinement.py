import math
import unittest
import numpy as np
from home_explore.arrival_policy import range_refinement_goal


class RangeRefinementTests(unittest.TestCase):
    def geometry(self, distance):
        return {'points': [[distance, 0.0, 0.5]] * 30}

    def test_boundary_range_requires_meaningful_point_three_five_meter_step(self):
        result = range_refinement_goal(self.geometry(1.27), [0, 0, 0], 1.27)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['step'], 0.35)
        self.assertGreater(result['step'], 0.22)
        self.assertAlmostEqual(result['predicted_surface_gap'], 0.92)

    def test_far_local_range_is_capped_at_point_six_meter_step(self):
        result = range_refinement_goal(self.geometry(1.6), [0, 0, 0], 1.6)
        self.assertAlmostEqual(result['step'], 0.6)
        self.assertAlmostEqual(result['predicted_surface_gap'], 1.0)

    def test_slight_boundary_changes_do_not_make_repeated_sub_tolerance_goals(self):
        for distance in (1.33, 1.295, 1.278, 1.264, 1.251):
            result = range_refinement_goal(self.geometry(distance), [0, 0, 0], distance)
            self.assertGreaterEqual(result['step'], 0.35)
            self.assertGreaterEqual(result['predicted_surface_gap'], 0.65)

    def test_selects_an_actual_near_surface_sample_not_a_centroid(self):
        points = [[0.0, 1.4, 0.6]] * 6 + [[2.0, 0.0, 0.5]] * 24
        result = range_refinement_goal({'points': points}, [0, 0, 0], 1.4)
        self.assertEqual(result['look_at_odom'], [0.0, 1.4, 0.6])
        self.assertAlmostEqual(result['point_odom'][0], 0.0)
        self.assertGreater(result['point_odom'][1], 0.35)
        self.assertEqual(result['point_odom'][2], 0.0)

    def test_odom_translation_preserves_direction_and_ground_goal(self):
        points = [[3.0, -0.4, 0.8]] * 30
        result = range_refinement_goal({'points': points}, [3.0, -2.0, 0.0], 1.6)
        self.assertAlmostEqual(result['point_odom'][0], 3.0)
        self.assertAlmostEqual(result['point_odom'][1], -1.4)
        self.assertEqual(result['look_at_odom'], [3.0, -0.4, 0.8])

    def test_already_near_or_nonlocal_range_produces_no_motion(self):
        for distance in (1.2, 1.25, 3.0, math.inf, math.nan):
            self.assertIsNone(range_refinement_goal(self.geometry(1.6), [0, 0, 0], distance))

    def test_missing_sparse_nonfinite_support_fails_closed(self):
        self.assertIsNone(range_refinement_goal(None, [0, 0, 0], 1.6))
        self.assertIsNone(range_refinement_goal({}, [0, 0, 0], 1.6))
        self.assertIsNone(range_refinement_goal({'points': [[1.6, 0, 0.5]] * 19}, [0, 0, 0], 1.6))
        self.assertIsNone(range_refinement_goal({'points': [[math.nan, 0, 0.5]] * 30}, [0, 0, 0], 1.6))
        self.assertIsNone(range_refinement_goal(self.geometry(1.6), [math.nan, 0, 0], 1.6))

    def test_contradictory_range_never_overrides_actual_support_clearance(self):
        self.assertIsNone(range_refinement_goal(self.geometry(0.8), [0, 0, 0], 1.6))


if __name__ == '__main__':
    unittest.main()
