import math
import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
from geometry_msgs.msg import PoseStamped
from home_explore.robot_io import RobotIO
import numpy as np
from home_explore.grid import Grid


def observed_object():
    # 仅使用合成支持点，不读取场景家具真值。
    xs = np.linspace(3.0, 7.0, 45)
    ys = np.linspace(4.5, 5.5, 15)
    points = np.array(
        [[x, y, 0.7] for x in xs for y in (4.5, 5.5)] + [[x, y, 0.7] for x in (3.0, 7.0) for y in ys]
    )
    raw = points - np.array([10.0, 20.0, 0.0])
    return dict(
        id='sofa_1',
        point=[5.0, 5.0, 0.7],
        look_at=[5.0, 5.0, 0.8],
        extent=[4.0, 1.0, 0.2],
        support_points_map=points.tolist(),
        support_points_odom=raw.tolist(),
        look_at_odom=[-5.0, -15.0, 0.8],
    )


class SurfaceDockingTests(unittest.TestCase):
    def test_wide_object_goals_are_outside_observed_bounds_with_real_gap(self):
        grid = Grid(np.zeros((200, 200), np.int16), 0.05, (0, 0))
        obj = observed_object()
        goals = grid.docking(obj, (1.0, 5.0, 0.0))
        self.assertTrue(goals)
        for goal in goals:
            x, y = goal['x'], goal['y']
            self.assertFalse(3.0 <= x <= 7.0 and 4.5 <= y <= 5.5)
            self.assertGreaterEqual(math.dist((x, y), goal['boundary_point'][:2]), 0.75 - 1e-8)
            self.assertLessEqual(math.dist((x, y), goal['boundary_point'][:2]), 1.0 + 1e-8)
        self.assertTrue(any(math.dist((g['x'], g['y']), obj['point'][:2]) > 2.0 for g in goals))

    def test_boundary_sets_position_but_observed_middle_sets_facing(self):
        grid = Grid(np.zeros((200, 200), np.int16), 0.05, (0, 0))
        obj = observed_object()
        points = np.array(obj['support_points_map'])
        raw = np.array(obj['support_points_odom'])
        for goal in grid.docking(obj, (1.0, 5.0, 0.0)):
            matches = np.flatnonzero(np.linalg.norm(points - np.array(goal['boundary_point']), axis=1) < 1e-9)
            self.assertTrue(len(matches))
            np.testing.assert_allclose(goal['look_at_odom'], obj['look_at_odom'])
            expected = math.atan2(obj['look_at'][1] - goal['y'], obj['look_at'][0] - goal['x'])
            self.assertAlmostEqual(goal['yaw'], expected)

    def test_walls_unknown_and_live_clearance_still_reject_endpoints(self):
        cells = np.zeros((200, 200), np.int16)
        cells[:, 140:] = 100
        cells[:60, :] = -1
        grid = Grid(cells, 0.05, (0, 0))
        goals = grid.docking(observed_object(), (1.0, 5.0, 0.0))
        self.assertTrue(goals)
        for goal in goals:
            self.assertTrue(grid.safe((goal['x'], goal['y'])))
            self.assertLess(goal['x'], 7.0)
            self.assertGreater(goal['y'], 3.0)

    def test_no_support_falls_back_to_observed_extent_and_original_reference(self):
        grid = Grid(np.zeros((200, 200), np.int16), 0.05, (0, 0))
        obj = dict(
            point=[5.0, 5.0, 0.7], look_at=[5.3, 5.1, 0.8], extent=[2.0, 1.0], look_at_odom=[-5.0, -15.0, 0.8]
        )
        goals = grid.docking(obj, (1.0, 5.0, 0.0))
        self.assertTrue(goals)
        for goal in goals:
            self.assertEqual(goal['boundary_source'], 'observed_extent_fallback')
            self.assertEqual(goal['look_at_odom'], obj['look_at_odom'])
            self.assertAlmostEqual(goal['yaw'], math.atan2(5.1 - goal['y'], 5.3 - goal['x']))

    def test_observation_requires_real_novel_view_and_preserves_larger_gaps(self):
        grid = Grid(np.zeros((200, 200), np.int16), 0.05, (0, 0))
        obj = observed_object()
        goals = grid.docking(obj, (1.0, 5.0, 0.0), mode='observe', attempted=[(1.0, 5.0)])
        self.assertTrue(goals)
        for goal in goals:
            self.assertIn(round(goal['boundary_gap'], 1), (1.2, 1.5, 1.8))
            self.assertGreaterEqual(math.dist((goal['x'], goal['y']), (1.0, 5.0)), 0.55)

    def test_rejected_endpoint_does_not_erase_its_close_safe_alternative(self):
        a = RobotIO.__new__(RobotIO)
        a.check = Mock()
        a.record = Mock()
        a.pose = lambda: (0.0, 0.0, 0.0)
        a.candidate_is_clear = lambda c, **kw: c['x'] != 0.0

        def pose(c):
            p = PoseStamped()
            p.pose.position.x = float(c['x'])
            p.pose.position.y = float(c['y'])
            return p

        a.goal_pose = pose
        a.wait_future = lambda value, timeout: value

        def plan(goal):
            result = S(status=4, result=S(path=S(poses=[pose(dict(x=0, y=0)), goal.goal])))
            return S(accepted=True, get_result_async=lambda: result)

        a.planner = S(wait_for_server=lambda **kw: True, send_goal_async=plan)
        values = [dict(x=x, y=0.0, yaw=0.0, score=x, kind='dock') for x in (0.0, 0.1, 5.0)]
        selected = a.path_candidate(values)
        self.assertAlmostEqual(selected['x'], 0.1)


if __name__ == '__main__':
    unittest.main()
