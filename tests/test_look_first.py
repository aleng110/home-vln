"""验证到附近与先观察后平移的行为边界。"""

import math
import time
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation
from home_explore.view_policy import useful_view, departure_heading
from home_explore.robot_io import RobotIO
from home_explore.agent import Explorer
from home_explore.guard import Guard
from geometry_msgs.msg import Twist


class LookFirstTests(unittest.TestCase):
    def test_visible_off_centre_target_is_acceptable(self):
        self.assertTrue(useful_view([0.15, 0.2, 0.6, 0.9], math.radians(18)))
        self.assertTrue(useful_view([0.1, 0.2, 0.4, 0.8], math.radians(24)))
        self.assertFalse(useful_view([0, 0.2, 0.3, 0.8], math.radians(24)))
        self.assertFalse(useful_view([0.05, 0.2, 0.25, 0.8], math.radians(30)))

    def test_clipping_does_not_alone_require_repeated_yaw_correction(self):
        self.assertTrue(useful_view([0, 0.1, 0.95, 0.99], 0.03))

    def test_path_heading_follows_first_leg_around_wall(self):
        points = [(0, 0), (0, 0.3), (0, 0.7), (1, 1), (3, 0)]
        self.assertAlmostEqual(departure_heading(points, (0, 0, 0)), math.pi / 2)

    def test_memory_target_behind_is_looked_at_before_path_alignment(self):
        robot = RobotIO.__new__(RobotIO)
        trace = []
        robot.path_candidate = Mock(return_value={'departure_yaw': math.pi / 2})
        robot.should_interrupt = Mock(return_value=False)
        robot.arrival_yaw = Mock(return_value=math.pi)
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.publish = Mock()
        robot.record = Mock()
        robot.turn_to = lambda yaw, reason, **kw: trace.append(('turn', reason, yaw))
        robot.departure_observe = lambda goal: trace.append(('observed', goal['kind']))
        RobotIO.prepare_departure(robot, dict(x=-2, y=0, yaw=0, kind='dock', look_at_odom=[-3, 0, 1]))
        self.assertEqual(
            trace,
            [
                ('turn', 'look_before_move', math.pi),
                ('observed', 'dock'),
                ('turn', 'path_departure', math.pi / 2),
            ],
        )

    def test_frontier_looks_along_reachable_path_not_goal_final_yaw(self):
        robot = RobotIO.__new__(RobotIO)
        trace = []
        robot.path_candidate = Mock(return_value={'departure_yaw': 0.7})
        robot.should_interrupt = Mock(return_value=False)
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.publish = Mock()
        robot.record = Mock()
        robot.turn_to = lambda yaw, reason, **kw: trace.append(yaw)
        robot.departure_observe = Mock()
        RobotIO.prepare_departure(robot, dict(x=2, y=1, yaw=-2, kind='frontier'))
        self.assertEqual(trace, [0.7, 0.7])
        robot.departure_observe.assert_called_once()

    def test_no_translation_preparation_if_current_path_is_invalid(self):
        robot = RobotIO.__new__(RobotIO)
        robot.path_candidate = Mock(return_value=None)
        robot.turn_to = Mock()
        with self.assertRaises(ValueError):
            RobotIO.prepare_departure(robot, dict(x=1, y=1, yaw=0))
        robot.turn_to.assert_not_called()

    def test_missing_yolo_detection_goes_to_vlm_without_blind_scan(self):
        robot = Explorer.__new__(Explorer)
        frame = (None, None, None, 3, None, {'detections': []})
        robot.local_frame = Mock(return_value=frame)
        robot.reference_pixel = Mock(return_value=(0.5, 0.5))
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.target = 'sofa'
        robot.record = Mock()
        robot.turn_to = Mock()
        self.assertIs(Explorer.repair_target_view(robot, {'id': 'sofa_1'}, {}), frame)
        robot.turn_to.assert_not_called()
        robot.local_frame.assert_called_once()

    def test_local_frame_requires_post_motion_image(self):
        robot = Explorer.__new__(Explorer)
        robot.last_motion_stamp = 11.0
        robot.capture = Mock(return_value=(np.zeros((4, 4, 3), np.uint8), None, None, 12, None))
        robot.local_detector = SimpleNamespace(infer=Mock(return_value={}))
        robot.check = Mock()
        Explorer.local_frame(robot)
        robot.capture.assert_called_once_with(after=11.0)

    def test_repeated_docking_cannot_repeat_coarse_arrival_turn(self):
        robot = Explorer.__new__(Explorer)
        robot.opt = {'view_accept_angle': math.radians(20)}
        robot.arrival_yaw = Mock(return_value=math.pi)
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.local_detector = None
        robot.arrival_facing_done = False
        robot.record = Mock()
        robot.step_index = 1
        robot.target = 'sofa'
        robot.turn_to = Mock()
        for _ in range(4):
            Explorer.finish_navigation_view(robot, {'kind': 'dock'})
        robot.turn_to.assert_called_once()

    def test_useful_arrival_bearing_does_not_trigger_facing_turn(self):
        robot = Explorer.__new__(Explorer)
        robot.opt = {'view_accept_angle': math.radians(20)}
        robot.arrival_yaw = Mock(return_value=math.radians(18))
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.record = Mock()
        robot.turn_to = Mock()
        Explorer.finish_navigation_view(robot, {'kind': 'view'})
        robot.turn_to.assert_not_called()

    def test_actual_relocation_allows_one_new_necessary_facing_turn(self):
        robot = Explorer.__new__(Explorer)
        robot.opt = {'view_accept_angle': math.radians(20)}
        robot.arrival_yaw = Mock(return_value=math.pi)
        robot.pose = Mock(return_value=(0, 0, 0))
        robot.local_detector = None
        robot.arrival_facing_done = False
        robot.record = Mock()
        robot.step_index = 1
        robot.target = 'sofa'
        robot.turn_to = Mock()
        robot.last_xy = (0, 0)
        Explorer.finish_navigation_view(robot, {'kind': 'dock'})
        robot.last_xy = (0.5, 0)
        Explorer.finish_navigation_view(robot, {'kind': 'dock'})
        Explorer.finish_navigation_view(robot, {'kind': 'dock'})
        self.assertEqual(robot.turn_to.call_count, 2)

    def test_guard_does_not_pass_reverse_velocity(self):
        guard = Guard.__new__(Guard)
        guard.enabled_at = time.monotonic()
        guard.command_at = guard.enabled_at
        guard.command = Twist()
        guard.command.linear.x = -0.1
        guard.command.angular.z = 0.2
        guard.pub = Mock()
        Guard.tick(guard)
        output = guard.pub.publish.call_args.args[0]
        self.assertEqual(output.linear.x, 0.0)
        self.assertAlmostEqual(output.angular.z, 0.2)

    def test_local_arrival_verification_does_not_wait_for_background_semantics(self):
        a = Explorer.__new__(Explorer)
        a.set_motion = Mock()
        a.publish = Mock()
        a.target = 'sofa'
        a.local_detector = object()
        a.verify_local_target = Mock(return_value=True)
        a.fresh_observation = Mock(side_effect=AssertionError('Redundant background barrier'))
        self.assertTrue(a.verify({'id': 'sofa_1'}))
        a.fresh_observation.assert_not_called()

    def test_semantic_mapping_lifetime_does_not_depend_on_task_cancel_or_activity(self):
        import threading
        from home_explore.robot_io import Cancelled

        a = Explorer.__new__(Explorer)
        a.semantic_active = False
        a.stop_event = threading.Event()
        a.stop_event.set()
        a.perception_shutdown = threading.Event()
        with patch('home_explore.agent.rclpy.ok', return_value=True):
            a.mapping_check()
            a.perception_shutdown.set()
            with self.assertRaises(Cancelled):
                a.mapping_check()

    def test_trusted_local_arrival_uses_no_cloud_and_still_requires_two_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Explorer.__new__(Explorer)
            a.target = 'tea_table'
            a.run_dir = Path(folder)
            a.observation_seq = 1
            a.opt = dict(
                near_distance=1.25, view_accept_angle=0.35, view_limit_angle=0.44, docking_standoff=0.9
            )
            geom = dict(
                point=[1.0, 0.0, 0.4],
                extent=[0.4, 0.4, 0.1],
                support_points=[[1.0, 0.0, 0.4]] * 40,
                valid_points=100,
                depth_spread=0.02,
            )
            original = dict(
                id='tea_table_1',
                label='tea_table',
                point=[1.0, 0.0, 0.4],
                look_at=[1.0, 0.0, 0.4],
                history=[dict(geom, confidence=0.9)],
                last_stamp=10.0,
                support_points=geom['support_points'],
            )
            a.memory = SimpleNamespace(
                yolo_thresholds={'tea_table': 0.4},
                quality=Mock(return_value={'reliable': True}),
                evidence_ok=Mock(return_value=True),
                get_by_id=Mock(return_value=original),
                snapshot=lambda: {},
            )
            a.check = Mock()
            a.record = Mock()
            a.lock = __import__('threading').RLock()
            a.semantic_pub = Mock()
            a.budget_ask = Mock(side_effect=AssertionError('A matched local detection must not call cloud'))
            a.reference_pixel = Mock(return_value=(0.5, 0.5))
            a.pose = lambda: (0.0, 0.0, 0.0)
            a.last_xy = (0.0, 0.0)
            a.map_geometry = lambda obj: obj
            tf = SimpleNamespace(
                transform=SimpleNamespace(
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            )
            info = SimpleNamespace(k=[40, 0, 20, 0, 40, 20, 0, 0, 1])
            image = np.zeros((40, 40, 3), np.uint8)
            detection = dict(label='tea_table', confidence=0.7, bbox=[0.0, 0.2, 0.615, 0.8])
            local = dict(detections=[detection], masks=[np.ones((40, 40), np.uint8)])
            first = (image, np.ones((40, 40)), info, 11.0, tf, local)
            second = (*first[:3], 12.0, tf, local)
            a.repair_target_view = Mock(return_value=first)
            a.local_frame = Mock(return_value=second)
            with patch('home_explore.agent.arrival_geometry', return_value=geom) as ranges, patch(
                'home_explore.agent.geometries_agree', return_value=True
            ):
                self.assertTrue(a.verify_local_target(original))
                self.assertEqual(ranges.call_count, 2)
            a.budget_ask.assert_not_called()
            a.local_frame.assert_called_once_with(after=11.0)
            self.assertEqual(a.pinned_object_id, 'tea_table_1')
            self.assertNotIn('approach_point', original)
            verification = next(c for c in a.record.call_args_list if c.args[0] == 'verification')
            self.assertEqual(verification.kwargs['source'], 'arrival_yolo_rgbd')
            self.assertFalse(verification.kwargs['facing'], 'Off-centre does not invalidate a near arrival')
            self.assertEqual(verification.kwargs['range_stamps'], [11.0, 12.0])


if __name__ == '__main__':
    unittest.main()
