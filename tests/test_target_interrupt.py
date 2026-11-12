import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from home_explore.agent import Explorer


class TargetInterruptTests(unittest.TestCase):
    def robot(self, pose, goal):
        robot = Explorer.__new__(Explorer)
        robot.semantic_active = True
        robot.nav_kind = 'frontier'
        robot.interrupted_objects = set()
        robot.pose = Mock(return_value=pose)
        robot.current_navigation_goal = goal
        robot.candidate_is_clear = Mock(return_value=True)
        robot.record = Mock()
        robot.get_target = Mock(
            return_value=(
                {'id': 'sofa_1', 'label': 'sofa', 'point': [5, 0, 0.7], 'history': [{'stamp': 20}]},
                {},
            )
        )
        robot.memory = SimpleNamespace(evidence_ok=lambda *a: True)
        robot.nav_started_stamp = 10
        robot.observation_seq = 3
        robot.nav_observation_start = 1
        return robot

    def test_small_semantic_boundary_updates_do_not_cancel_docking(self):
        robot = self.robot((0, 0, 0), {'x': 2, 'y': 0, 'kind': 'dock'})
        robot.nav_kind = 'dock'
        robot.candidate_is_clear.return_value = False
        self.assertFalse(robot.should_interrupt())
        robot.candidate_is_clear.assert_not_called()

    def test_distant_goal_on_existing_safe_approach_does_not_cancel_entry(self):
        self.assertFalse(Explorer.should_interrupt(self.robot((0, 0, 0), {'x': 2, 'y': 0})))

    def test_near_target_interrupts_search_for_final_approach(self):
        self.assertTrue(Explorer.should_interrupt(self.robot((4, 0, 0), {'x': 2, 'y': 0})))

    def test_distant_target_cancels_existing_goal_that_goes_away(self):
        self.assertTrue(Explorer.should_interrupt(self.robot((0, 0, 0), {'x': -2, 'y': 0})))


if __name__ == '__main__':
    unittest.main()
