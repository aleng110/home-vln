import math
import time
import threading
import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from home_explore.view_policy import forward_entry_candidates, forward_entry_path
from home_explore.agent import Explorer
from home_explore.robot_io import RobotIO, NavigationUnavailable


class ForwardEntryTests(unittest.TestCase):
    def robot(self):
        a = Explorer.__new__(Explorer)
        a.check = Mock()
        a.pose = Mock(return_value=(0.0, 0.0, 0.0))
        a.lock = threading.RLock()
        a.grid = S(frontiers=Mock())
        a.map_received = time.monotonic()
        a.entry_origin_odom = [0.0, 0.0, 0.0]
        a.entry_heading_odom = 0.0
        a.entry_complete = False
        a.entry_attempts = 0
        a.entry_blocked_rounds = 0
        a.last_xy = (0.0, 0.0)
        a.opt = dict(entry_min_progress=1.1, entry_forward_distance=1.5)
        a.map_geometry = lambda obj: obj
        a.publish_objects = Mock()
        a.publish = Mock()
        a.record = Mock()
        a.path_candidate = Mock(return_value=None)
        a.get_target = Mock(side_effect=AssertionError('entry must precede target pursuit'))
        return a

    def test_relative_forward_candidates_do_not_encode_room_coordinates(self):
        values = forward_entry_candidates((7.0, -3.0, 0.0), math.pi / 2)
        self.assertEqual(values[0]['entry_distance'], 1.5)
        for goal in values:
            self.assertAlmostEqual(goal['x'], 7.0)
            self.assertGreater(goal['y'], -3.0)
            self.assertTrue(goal['requires_path_validation'])

    def test_navfn_raster_jitter_is_allowed(self):
        self.assertTrue(forward_entry_path([(0, 0), (0.3, -0.02), (0.7, -0.06), (1.5, 0)], (0, 0), 0))

    def test_straight_endpoint_does_not_allow_side_loop(self):
        self.assertFalse(forward_entry_path([(0, 0), (0.3, -0.45), (1.5, 0)], (0, 0), 0))

    def test_backtracking_and_invalid_paths_are_rejected(self):
        self.assertFalse(forward_entry_path([(0, 0), (0.7, 0), (0.3, 0), (1.5, 0)], (0, 0), 0))
        self.assertFalse(forward_entry_path([(0, 0), (float('nan'), 0)], (0, 0), 0))
        self.assertFalse(forward_entry_path([], (0, 0), 0))

    def test_forward_entry_precedes_known_target_pursuit(self):
        a = self.robot()
        a.path_candidate.side_effect = lambda values: dict(values[0], path_length=1.5, departure_yaw=0.0)
        goal, obj, quality = a.decide()
        self.assertEqual(goal['kind'], 'entry_forward')
        self.assertIsNone(obj)
        a.get_target.assert_not_called()
        a.grid.frontiers.assert_not_called()

    def test_blocked_forward_route_never_falls_back_to_outdoor_side(self):
        a = self.robot()
        for _ in range(2):
            self.assertIsNone(a.decide()[0])
        with self.assertRaisesRegex(NavigationUnavailable, '初始正前方没有安全可达'):
            a.decide()
        a.get_target.assert_not_called()
        a.grid.frontiers.assert_not_called()

    def test_forward_entry_can_be_disabled_for_different_start_conditions(self):
        a = self.robot()
        a.opt['entry_forward_distance'] = 0.0
        self.assertIsNone(a.entry_goal((0.0, 0.0, 0.0)))
        self.assertTrue(a.entry_complete)
        a.path_candidate.assert_not_called()

    def test_successful_progress_marks_session_entry_complete(self):
        a = self.robot()
        a.last_xy = (1.25, 0.02)
        self.assertIsNone(a.entry_goal((1.25, 0.02, 0.0)))
        self.assertTrue(a.entry_complete)
        a.path_candidate.assert_not_called()

    def test_second_leg_only_finishes_remaining_entry_progress(self):
        a = self.robot()
        a.entry_attempts = 1
        a.last_xy = (0.7, 0.0)
        a.path_candidate.side_effect = lambda values: values[0]
        goal = a.entry_goal((0.7, 0.0, 0.0))
        self.assertLessEqual(goal['entry_distance'], 0.71)
        self.assertGreaterEqual(goal['entry_distance'], 0.65)

    def test_slam_rotation_is_applied_to_odom_heading(self):
        a = self.robot()
        a.map_geometry = lambda obj: dict(point=[10.0, 2.0, 0.0], observer_positions=[[10.0, 3.0, 0.0]])
        a.path_candidate.side_effect = lambda values: values[0]
        goal = a.entry_goal((10.0, 2.0, math.pi / 2))
        self.assertAlmostEqual(goal['x'], 10.0)
        self.assertAlmostEqual(goal['y'], 3.5)

    def test_anchor_is_captured_once_from_live_odometry(self):
        a = self.robot()
        a.entry_origin_odom = None
        a.entry_heading_odom = None
        a.busy = False
        a.last_xy = None
        msg = Odometry()
        msg.pose.pose.position.x = 10.0
        msg.pose.pose.position.y = 2.0
        msg.pose.pose.orientation.z = math.sin(math.pi / 4)
        msg.pose.pose.orientation.w = math.cos(math.pi / 4)
        a.on_odom(msg)
        msg.pose.pose.position.y = 3.0
        a.on_odom(msg)
        self.assertEqual(a.entry_origin_odom, [10.0, 2.0, 0.0])
        self.assertAlmostEqual(a.entry_heading_odom, math.pi / 2)

    def motion_robot(self, xy):
        a = self.robot()
        a.last_xy = xy
        a.motion = True
        a.action_kind = 'navigate'
        a.nav_kind = 'entry_forward'
        a.entry_peak_progress = 0.0
        a.fault_reason = None
        a.busy = True
        a.stop_event = threading.Event()
        a.lease_pub = Mock()
        a.status_pub = Mock()
        a.goal_handle = Mock()
        a.state = {}
        a.obstacles_ready = Mock(return_value=True)
        return a

    def test_replanned_side_departure_revokes_actual_motion_lease(self):
        a = self.motion_robot((0.5, 0.3))
        a.heartbeat()
        self.assertFalse(a.lease_pub.publish.call_args.args[0].data)
        self.assertIn('初始前行', a.fault_reason)
        a.goal_handle.cancel_goal_async.assert_called_once()

    def test_straight_motion_passes_but_backtracking_is_stopped(self):
        a = self.motion_robot((0.6, 0.03))
        a.heartbeat()
        self.assertTrue(a.lease_pub.publish.call_args.args[0].data)
        self.assertIsNone(a.fault_reason)
        a.last_xy = (0.3, 0.03)
        a.heartbeat()
        self.assertFalse(a.lease_pub.publish.call_args.args[0].data)

    def test_live_path_validation_rejects_side_detour(self):
        a = RobotIO.__new__(RobotIO)
        a.check = Mock()
        a.record = Mock()
        a.goal_pose = Mock(return_value=PoseStamped())
        a.planner = S(wait_for_server=Mock(return_value=True), send_goal_async=Mock(return_value=object()))
        points = []
        for x, y in [(0.0, 0.0), (0.5, -0.45), (1.5, 0.0)]:
            p = PoseStamped()
            p.pose.position.x = x
            p.pose.position.y = y
            points.append(p)
        handle = S(accepted=True, get_result_async=Mock(return_value=object()))
        result = S(status=GoalStatus.STATUS_SUCCEEDED, result=S(path=S(poses=points)))
        a.wait_future = Mock(side_effect=[handle, result])
        self.assertIsNone(a.path_candidate(forward_entry_candidates((0, 0, 0), 0)[:1]))
        self.assertEqual(a.record.call_args.args[0], 'entry_path_rejected')


if __name__ == '__main__':
    unittest.main()
