import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock, patch
from tf2_ros import TransformException
from home_explore.robot_io import RobotIO, Cancelled


class PoseStartupTests(unittest.TestCase):
    def robot(self):
        robot = RobotIO.__new__(RobotIO)
        robot.check = Mock()
        robot.get_clock = Mock(return_value=S(now=lambda: S(nanoseconds=8_000_000_000)))
        transform = S(
            header=S(stamp=S(sec=8, nanosec=0)),
            transform=S(translation=S(x=1.0, y=2.0), rotation=S(x=0.0, y=0.0, z=0.0, w=1.0)),
        )
        robot.tf = S(
            lookup_transform=Mock(side_effect=[TransformException('transient startup interval'), transform])
        )
        return robot

    def test_transient_initial_tf_window_is_retried(self):
        robot = self.robot()
        with patch('home_explore.robot_io.time.sleep'):
            self.assertEqual(RobotIO.pose(robot), (1.0, 2.0, 0.0))
        self.assertEqual(robot.tf.lookup_transform.call_count, 2)

    def test_cancel_interrupts_tf_retry(self):
        robot = self.robot()
        robot.check = Mock(side_effect=[None, Cancelled('cancel')])
        with patch('home_explore.robot_io.time.sleep'):
            with self.assertRaises(Cancelled):
                RobotIO.pose(robot)
        self.assertEqual(robot.tf.lookup_transform.call_count, 1)


if __name__ == '__main__':
    unittest.main()
