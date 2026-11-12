import math
import unittest
import numpy as np
from home_explore.grid import Grid
from home_explore.coverage import local_observation_candidates, camera_view_mask


class LocalSearchTests(unittest.TestCase):
    def room(self):
        data = np.zeros((100, 100), np.int16)
        data[[0, -1], :] = 100
        data[:, [0, -1]] = 100
        return Grid(data, 0.1, (0.0, 0.0))

    def difference(self, a, b):
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def test_same_xy_visited_in_one_direction_keeps_other_directions(self):
        grid = self.room()
        robot = (5.0, 5.0, 0.0)
        views = [dict(xy=robot[:2], yaw=0.0, range=2.8)]
        goals = local_observation_candidates(grid, robot, views, attempted=[robot])
        self.assertTrue(goals)
        self.assertTrue(all((g['x'], g['y']) == robot[:2] for g in goals))
        self.assertTrue(all(g['kind'] == 'search_look' for g in goals))
        self.assertTrue(
            any(
                self.difference(g['yaw'], math.pi / 2) < 0.1 or self.difference(g['yaw'], -math.pi / 2) < 0.1
                for g in goals
            )
        )

    def test_processed_directions_are_not_proposed_again(self):
        grid = self.room()
        robot = (5.0, 5.0, 0.0)
        views = [dict(xy=robot[:2], yaw=yaw, range=2.8) for yaw in np.arange(-math.pi, math.pi, math.pi / 4)]
        self.assertEqual(local_observation_candidates(grid, robot, views), [])

    def test_attempted_direction_is_excluded_without_detection(self):
        grid = self.room()
        robot = (5.0, 5.0, 0.0)
        attempted = [(5.0, 5.0, math.pi / 2), dict(x=5.0, y=5.0, yaw=-math.pi / 2)]
        goals = local_observation_candidates(grid, robot, attempted=attempted, limit=8)
        self.assertTrue(goals)
        self.assertTrue(all(self.difference(g['yaw'], math.pi / 2) >= math.radians(35) for g in goals))
        self.assertTrue(all(self.difference(g['yaw'], -math.pi / 2) >= math.radians(35) for g in goals))

    def test_far_away_attempt_does_not_ban_new_location(self):
        grid = self.room()
        robot = (5.0, 5.0, 0.0)
        before = local_observation_candidates(grid, robot, limit=8)
        after = local_observation_candidates(
            grid, robot, attempted=[(1.0, 1.0, g['yaw']) for g in before], limit=8
        )
        self.assertEqual([g['yaw'] for g in before], [g['yaw'] for g in after])

    def test_wall_blocks_gain_from_room_behind_it(self):
        data = np.zeros((100, 100), np.int16)
        data[:, 40] = 100
        grid = Grid(data, 0.1, (0.0, 0.0))
        robot = (3.0, 5.0, math.pi / 2)
        goals = local_observation_candidates(grid, robot, limit=8, min_gain=0.0)
        east = next(g for g in goals if self.difference(g['yaw'], 0) < 0.01)
        mask = camera_view_mask(grid, robot[:2], east['yaw'])
        self.assertFalse(mask[:, 41:].any())
        self.assertEqual(east['information_cells'], int(mask.sum()))

    def test_unknown_space_never_adds_visual_gain(self):
        data = np.zeros((100, 100), np.int16)
        data[:, 40:] = -1
        grid = Grid(data, 0.1, (0.0, 0.0))
        robot = (3.0, 5.0, math.pi / 2)
        goals = local_observation_candidates(grid, robot, limit=8, min_gain=0.0)
        east = next(g for g in goals if self.difference(g['yaw'], 0) < 0.01)
        mask = camera_view_mask(grid, robot[:2], east['yaw'])
        self.assertFalse(mask[:, 40:].any())
        self.assertAlmostEqual(east['unseen_free_area'], float(mask.sum()) * 0.01)
        self.assertEqual(
            local_observation_candidates(
                Grid(np.full((50, 50), -1, np.int16), 0.1, (0.0, 0.0)), (2.0, 2.0, 0.0)
            ),
            [],
        )

    def test_proposals_are_bounded_and_not_small_recentring(self):
        goals = local_observation_candidates(self.room(), (5.0, 5.0, 0.37), limit=2)
        self.assertEqual(len(goals), 2)
        self.assertTrue(all(g['turn_angle'] >= math.radians(35) for g in goals))
        self.assertTrue(all(g['visual_gain'] >= 0.35 and g['new_visual_fraction'] >= 0.3 for g in goals))
        self.assertEqual(local_observation_candidates(self.room(), (5.0, 5.0, 0.0), limit=0), [])

    def test_unsafe_rotation_origin_is_not_used(self):
        data = np.zeros((80, 80), np.int16)
        data[40, 40] = 100
        self.assertEqual(local_observation_candidates(Grid(data, 0.1, (0.0, 0.0)), (4.05, 4.05, 0.0)), [])


if __name__ == '__main__':
    unittest.main()
