import unittest
import math
import numpy as np
from home_explore.grid import Grid
from home_explore.coverage import camera_view_mask, coverage_candidates, observed_mask


class CoverageTests(unittest.TestCase):
    def room(self):
        a = np.zeros((100, 120), np.int16)
        a[[0, -1], :] = 100
        a[:, [0, -1]] = 100
        a[:, 60] = 100
        a[43:57, 60] = 0
        return Grid(a, 0.1, (0, 0))

    def test_laser_known_room_still_requires_camera_search(self):
        g = self.room()
        self.assertEqual(g.frontiers((2.0, 5.0, 0.0)), [])
        goals, covered = coverage_candidates(g, (2.0, 5.0, 0.0))
        self.assertTrue(goals)
        self.assertFalse(covered.any())

    def test_camera_mask_does_not_see_through_wall(self):
        g = self.room()
        mask = camera_view_mask(g, (3.0, 2.0), 0.0, distance=5.0)
        self.assertFalse(mask[20, 80])

    def test_camera_seen_does_not_mark_unknown_free(self):
        a = np.zeros((80, 80), np.int16)
        a[:, 40:] = -1
        g = Grid(a, 0.1, (0, 0))
        mask = camera_view_mask(g, (2.0, 4.0), 0.0, distance=4.0)
        self.assertFalse(mask[:, 40:].any())

    def test_view_direction_changes_coverage(self):
        g = self.room()
        a = camera_view_mask(g, (3.0, 5.0), 0.0)
        b = camera_view_mask(g, (3.0, 5.0), math.pi)
        self.assertGreater(np.sum(a & ~b), 100)

    def test_already_searched_room_does_not_hide_unseen_room(self):
        g = self.room()
        views = [
            dict(xy=(x, y), yaw=a, range=2.8)
            for x in [1.5, 3.0, 4.8]
            for y in [2.0, 5.0, 8.0]
            for a in [0.0, math.pi / 2, math.pi, -math.pi / 2]
        ]
        goals, covered = coverage_candidates(g, (4.0, 5.0, 0.0), views)
        self.assertTrue(goals)
        self.assertTrue(any(v['x'] > 6.0 for v in goals[:6]))

    def test_frontier_tabu_prevents_short_back_and_forth(self):
        a = np.full((100, 100), -1, np.int16)
        a[10:80, 10:80] = 0
        g = Grid(a, 0.1, (0, 0))
        old = g.frontiers((4.0, 4.0, 0.0))
        visited = [(v['x'], v['y']) for v in old]
        new = g.frontiers((4.0, 4.0, 0.0), visited)
        self.assertTrue(all(math.dist((v['x'], v['y']), p) >= 0.8 for v in new for p in visited))

    def test_visual_candidates_must_leave_already_visited_region(self):
        g = self.room()
        visited = [(3.0, 5.0, 0.0)]
        goals, _ = coverage_candidates(g, (3.0, 5.0, 0.0), visited=visited)
        self.assertTrue(goals)
        self.assertTrue(all(math.dist((v['x'], v['y']), (3.0, 5.0)) >= 1.0 for v in goals))


if __name__ == '__main__':
    unittest.main()
