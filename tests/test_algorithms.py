import unittest
import math
import numpy as np
from home_explore.grid import Grid
from home_explore.semantic import Memory, surface_geometry, transform_points


class GeometryTests(unittest.TestCase):
    def test_unknown_is_not_free(self):
        a = np.zeros((60, 60), dtype=np.int16)
        a[:, 30:] = -1
        g = Grid(a, 0.1, (0, 0))
        self.assertFalse(g.safe((3.5, 2.0)))
        self.assertFalse(g.safe((2.95, 2.0)))
        self.assertTrue(g.safe((2.0, 2.0)))

    def test_frontier_endpoint_steps_back_into_free_space(self):
        a = np.full((80, 80), -1, dtype=np.int16)
        a[10:60, 10:60] = 0
        g = Grid(a, 0.1, (0, 0))
        frontiers = g.frontiers((3.0, 3.0, 0.0))
        self.assertGreater(len(frontiers), 0)
        for q in frontiers:
            self.assertTrue(g.safe((q['x'], q['y'])))

    def test_complete_known_map_has_no_frontiers(self):
        g = Grid(np.zeros((50, 50), dtype=np.int16), 0.1, (0, 0))
        self.assertEqual(g.frontiers((2.0, 2.0, 0.0)), [])

    def test_closed_frontier_ring_generates_multiple_boundary_goals(self):
        a = np.full((100, 100), -1, dtype=np.int16)
        a[20:80, 20:80] = 0
        g = Grid(a, 0.1, (0, 0))
        goals = g.frontiers((5.0, 5.0, 0.0))
        self.assertGreaterEqual(len(goals), 4)
        self.assertTrue(all(math.dist((v['x'], v['y']), (5.0, 5.0)) > 1.5 for v in goals))

    def test_map_boundary_does_not_grant_clearance(self):
        g = Grid(np.zeros((50, 50), dtype=np.int16), 0.1, (0, 0))
        self.assertFalse(g.safe((0.05, 2.0)))

    def test_docking_rejects_wall_and_unknown(self):
        a = np.zeros((100, 100), dtype=np.int16)
        a[:, 55:] = 100
        g = Grid(a, 0.1, (0, 0))
        obj = dict(point=[5.0, 5.0, 0.6], extent=[0.3, 0.3])
        choices = g.docking(obj, (2.0, 2.0, 0.0))
        self.assertTrue(choices)
        self.assertTrue(all(q['x'] < 5.5 and g.safe((q['x'], q['y'])) for q in choices))

    def test_visibility_does_not_see_through_wall(self):
        a = np.zeros((60, 60), dtype=np.int16)
        a[:, 30] = 100
        self.assertFalse(Grid(a, 0.1, (0, 0)).visible((1.0, 2.0), (4.0, 2.0)))

    def test_generated_points_move_with_object(self):
        g = Grid(np.zeros((100, 100), dtype=np.int16), 0.1, (0, 0))
        a = g.docking(dict(point=[4.0, 4.0, 0.5], extent=[0.2, 0.3]), (2.0, 2.0, 0.0))
        b = g.docking(dict(point=[6.0, 6.0, 0.5], extent=[0.2, 0.3]), (2.0, 2.0, 0.0))
        self.assertTrue(a and b)
        self.assertNotEqual((a[0]['x'], a[0]['y']), (b[0]['x'], b[0]['y']))

    def test_dock_gap_uses_observed_boundary_not_centroid_radius(self):
        g = Grid(np.zeros((100, 100), dtype=np.int16), 0.1, (0, 0))
        candidates = g.docking(dict(point=[5.0, 5.0, 0.5], extent=[2.0, 1.0]), (2.0, 2.0, 0.0))
        self.assertTrue(candidates)
        self.assertTrue(
            all(
                0.75 - 1e-6 <= math.dist((q['x'], q['y']), q['boundary_point'][:2]) <= 1.0 + 1e-6
                for q in candidates
            )
        )
        self.assertTrue(any(math.dist((q['x'], q['y']), (5.0, 5.0)) > 1.35 for q in candidates))


class SemanticTests(unittest.TestCase):
    def geometry(self, x=2.0, spread=0.15):
        return dict(
            point=[x, 1.0, 0.7],
            extent=[0.4, 0.3, 0.2],
            points=[],
            valid_points=80,
            spread=spread,
            partial=True,
        )

    def test_single_observation_requires_supplement(self):
        m = Memory()
        o = m.add('bed', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a.jpg')
        self.assertFalse(m.quality(o, 10.0)['reliable'])

    def test_two_consistent_observations_are_sufficient_without_full_shape(self):
        m = Memory()
        m.add('bed', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a.jpg')
        o = m.add('bed', 0.9, self.geometry(2.1), [0.2, 0, 1], 12.0, 'b.jpg')
        self.assertTrue(m.quality(o, 12.0)['reliable'])
        self.assertTrue(o['partial'])

    def test_tracker_cannot_confirm_semantics(self):
        m = Memory()
        m.add('bed', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a.jpg')
        o = m.add('bed', 0.99, self.geometry(), [0, 0, 1], 12.0, 'b.jpg', source='tracker')
        self.assertFalse(m.quality(o, 12.0)['reliable'])

    def test_stale_observations_do_not_complete(self):
        m = Memory()
        m.add('bed', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a')
        o = m.add('bed', 0.9, self.geometry(), [0, 0, 1], 12.0, 'b')
        self.assertFalse(m.quality(o, 200.0)['reliable'])

    def test_distant_beds_remain_separate(self):
        m = Memory()
        m.add('bed', 0.9, self.geometry(), [0, 0, 1], 10.0, 'a')
        m.add('bed', 0.9, self.geometry(5.0), [0, 0, 1], 12.0, 'b')
        self.assertEqual(len(m.objects), 2)

    def test_sparse_depth_is_rejected(self):
        depth = np.full((50, 50), np.nan)
        self.assertIsNone(
            surface_geometry([0, 0, 1, 1], depth, [50, 0, 25, 0, 50, 25, 0, 0, 1], [0, 0, 0], [0, 0, 0, 1])
        )

    def test_rigid_transform_has_correct_direction(self):
        p = transform_points([[1, 0, 0]], [2, 3, 0], [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        np.testing.assert_allclose(p, [[2, 4, 0]], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
