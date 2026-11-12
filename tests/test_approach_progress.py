import math
import unittest
import numpy as np
from home_explore.grid import Grid


class ApproachProgressTests(unittest.TestCase):
    def object(self, point=(6.0, 4.0, 0.7)):
        return dict(
            id='sofa_001',
            point=list(point),
            look_at=list(point),
            point_odom=list(point),
            look_at_odom=list(point),
            extent=[0.5, 0.4, 0.5],
        )

    def test_short_free_progress_exists_when_final_docking_is_unknown(self):
        data = np.full((100, 100), -1, np.int16)
        data[10:70, 10:45] = 0
        grid = Grid(data, 0.1, (0.0, 0.0))
        obj = self.object()
        robot = (2.0, 4.0, 0.0)
        self.assertEqual(grid.docking(obj, robot), [])
        goals = grid.approach_progress(obj, robot)
        self.assertTrue(goals)
        self.assertTrue(all(grid.safe((g['x'], g['y'])) for g in goals))
        self.assertTrue(all(grid.visible(robot, (g['x'], g['y']), ignore_last=0.0) for g in goals))
        self.assertTrue(all(g['kind'] == 'approach_progress' and g['object_id'] == obj['id'] for g in goals))

    def test_endpoint_proposals_do_not_claim_a_path_through_a_wall(self):
        data = np.zeros((100, 100), np.int16)
        data[:, 40] = 100
        grid = Grid(data, 0.1, (0.0, 0.0))
        robot = (3.3, 4.0, 0.0)
        obj = self.object()
        goals = grid.approach_progress(obj, robot)
        self.assertTrue(goals)
        self.assertTrue(all(g['requires_path_validation'] for g in goals))
        self.assertTrue(all(grid.safe((g['x'], g['y'])) for g in goals))

    def test_connected_room_behind_door_can_use_nonstraight_nav2_route(self):
        data = np.zeros((100, 100), np.int16)
        data[:, 40] = 100
        data[45:65, 40] = 0
        grid = Grid(data, 0.1, (0.0, 0.0))
        robot = (3.3, 4.0, 0.0)
        obj = self.object()
        goals = grid.approach_progress(obj, robot)
        self.assertTrue(any(g['x'] > 4.0 for g in goals))
        self.assertTrue(any(not grid.visible(robot, (g['x'], g['y']), ignore_last=0.0) for g in goals))
        self.assertTrue(all(grid.safe((g['x'], g['y']), radius=0.31) for g in goals))

    def test_blocked_and_visited_endpoints_are_excluded(self):
        grid = Grid(np.zeros((100, 100), np.int16), 0.1, (0.0, 0.0))
        obj = self.object()
        robot = (2.0, 4.0, 0.0)
        old = grid.approach_progress(obj, robot)
        blocked = [(g['x'], g['y']) for g in old[:3]]
        visited = [(g['x'], g['y'], g['yaw']) for g in old[3:]]
        new = grid.approach_progress(obj, robot, visited=visited, blocked=blocked)
        self.assertTrue(all(math.dist((g['x'], g['y']), p) >= 0.5 for g in new for p in blocked))
        self.assertTrue(all(math.dist((g['x'], g['y']), p[:2]) >= 0.45 for g in new for p in visited))

    def test_all_proposals_reduce_range_and_preserve_look_reference(self):
        grid = Grid(np.zeros((100, 100), np.int16), 0.1, (0.0, 0.0))
        obj = self.object()
        robot = (2.0, 4.0, 0.0)
        goals = grid.approach_progress(obj, robot, limit=4)
        self.assertEqual(len(goals), 4)
        for goal in goals:
            self.assertLess(math.dist((goal['x'], goal['y']), obj['point'][:2]), 3.9)
            self.assertGreater(goal['progress_m'], 0.1)
            self.assertEqual(goal['look_at_odom'], obj['look_at_odom'])
            self.assertAlmostEqual(goal['yaw'], math.atan2(4.0 - goal['y'], 6.0 - goal['x']))
            self.assertGreaterEqual(math.dist((goal['x'], goal['y']), robot[:2]), 0.55)
            self.assertLessEqual(math.dist((goal['x'], goal['y']), robot[:2]), 3.00001)

    def test_unknown_origin_or_unknown_endpoints_do_not_grant_reachability(self):
        grid = Grid(np.full((100, 100), -1, np.int16), 0.1, (0.0, 0.0))
        self.assertEqual(grid.approach_progress(self.object(), (2.0, 4.0, 0.0)), [])

    def test_progress_does_not_overshoot_a_near_surface(self):
        grid = Grid(np.zeros((100, 100), np.int16), 0.1, (0.0, 0.0))
        robot = (2.0, 4.0, 0.0)
        self.assertEqual(grid.approach_progress(self.object((2.8, 4.0, 0.7)), robot), [])
        self.assertEqual(grid.approach_progress(self.object(), robot, limit=0), [])


if __name__ == '__main__':
    unittest.main()
