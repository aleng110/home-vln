import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
from home_explore.robot_io import RobotIO, NavigationUnavailable
from geometry_msgs.msg import PoseStamped
from home_explore.task_result import build_result


class PlannerTimeoutTests(unittest.TestCase):
    def robot(self):
        robot = RobotIO.__new__(RobotIO)
        robot.check = Mock()
        robot.goal_pose = Mock(return_value=PoseStamped())
        robot.planner = S(
            wait_for_server=Mock(return_value=True), send_goal_async=Mock(return_value=object())
        )
        return robot

    def test_no_goal_acknowledgement_has_specific_navigation_error(self):
        robot = self.robot()
        robot.wait_future = Mock(side_effect=TimeoutError('等待 ROS action 超时'))
        with self.assertRaisesRegex(NavigationUnavailable, '全局规划服务5秒内未确认请求'):
            RobotIO.path_candidate(robot, [dict(x=0, y=0, yaw=0, score=0)])

    def test_no_path_result_cancels_goal_and_identifies_the_server(self):
        robot = self.robot()
        handle = S(accepted=True, get_result_async=Mock(return_value=object()), cancel_goal_async=Mock())
        robot.wait_future = Mock(side_effect=[handle, TimeoutError('等待 ROS action 超时')])
        with self.assertRaisesRegex(NavigationUnavailable, '全局规划服务8秒内未返回路径'):
            RobotIO.path_candidate(robot, [dict(x=0, y=0, yaw=0, score=0)])
        handle.cancel_goal_async.assert_called_once()

    def test_planner_failure_is_not_reported_as_object_or_unknown_failure(self):
        result = build_result(
            dict(
                outcome='failed',
                completed_targets=[],
                current_target='sofa',
                inspection_question='桌上有什么',
                error='全局规划服务8秒内未返回路径；请检查代价地图点云/TF更新及启动日志',
            )
        )
        self.assertIn('导航服务暂不可用', result['user_message'])
        self.assertNotIn('任务遇到异常', result['user_message'])
        self.assertFalse(result['inspection_performed'])


if __name__ == '__main__':
    unittest.main()
