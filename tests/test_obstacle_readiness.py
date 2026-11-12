import threading
import time
import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
from home_explore.robot_io import RobotIO, NavigationUnavailable, Cancelled


class ObstacleReadinessTests(unittest.TestCase):
    def robot(self):
        robot = RobotIO.__new__(RobotIO)
        robot.obstacle_status = dict(ready=True, published=3, last_cloud_stamp=100.0, depth_received=6)
        robot.obstacle_status_received = time.monotonic()
        robot.get_clock = lambda: S(now=lambda: S(nanoseconds=100_200_000_000))
        robot.check = Mock()
        robot.set_motion = Mock()
        robot.publish = Mock()
        robot.record = Mock()
        return robot

    def test_fresh_status_and_sensor_stamp_are_both_required(self):
        robot = self.robot()
        self.assertTrue(robot.obstacles_ready())
        robot.obstacle_status_received -= 3.0
        self.assertFalse(robot.obstacles_ready())
        robot.obstacle_status_received = time.monotonic()
        robot.obstacle_status['last_cloud_stamp'] = 90.0
        self.assertFalse(robot.obstacles_ready())

    def test_fresh_producer_status_survives_bounded_local_clock_callback_lag(self):
        robot = self.robot()
        robot.obstacle_status.update(last_cloud_stamp=100.7, ros_time=100.8)
        self.assertTrue(robot.obstacles_ready())

    def test_source_timestamp_cannot_be_future_relative_to_its_own_clock(self):
        robot = self.robot()
        robot.obstacle_status.update(last_cloud_stamp=102.0, ros_time=100.8)
        self.assertFalse(robot.obstacles_ready())

    def test_unrelated_producer_clock_cannot_enable_motion(self):
        robot = self.robot()
        robot.obstacle_status.update(last_cloud_stamp=200.0, ros_time=200.0)
        self.assertFalse(robot.obstacles_ready())

    def test_source_reports_stale_cloud_even_with_recent_status(self):
        robot = self.robot()
        robot.obstacle_status.update(last_cloud_stamp=98.0, ros_time=100.8)
        self.assertFalse(robot.obstacles_ready())

    def test_never_received_depth_cannot_enable_motion(self):
        robot = self.robot()
        robot.obstacle_status = dict(ready=False, published=0, depth_received=0, reason='waiting_depth')
        with self.assertRaisesRegex(NavigationUnavailable, '尚未收到原始深度图'):
            robot.wait_for_obstacle_cloud(timeout=0.0)
        robot.set_motion.assert_called_once_with(False)
        robot.record.assert_not_called()

    def test_wait_is_cancelable(self):
        robot = self.robot()
        robot.obstacle_status = None
        robot.check.side_effect = Cancelled('cancel')
        with self.assertRaises(Cancelled):
            robot.wait_for_obstacle_cloud(timeout=0.1)

    def test_healthy_input_completes_startup_without_motion(self):
        robot = self.robot()
        robot.wait_for_obstacle_cloud(timeout=0.1)
        robot.set_motion.assert_called_once_with(False)
        self.assertEqual(robot.record.call_args.args[0], 'obstacle_cloud_ready')

    def test_lost_input_revokes_lease_and_cancels_active_action(self):
        robot = self.robot()
        robot.obstacle_status['ready'] = False
        robot.motion = True
        robot.busy = True
        robot.fault_reason = None
        robot.stop_event = threading.Event()
        robot.lease_pub = Mock()
        robot.status_pub = Mock()
        robot.goal_handle = Mock()
        robot.state = {}
        robot.heartbeat()
        self.assertFalse(robot.lease_pub.publish.call_args.args[0].data)
        self.assertIn('避障点云', robot.fault_reason)
        robot.goal_handle.cancel_goal_async.assert_called_once()


if __name__ == '__main__':
    unittest.main()
