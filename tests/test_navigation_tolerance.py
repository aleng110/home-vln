"""规划器容差必须与到达验收范围一致。"""

import math
import unittest
from pathlib import Path
import yaml


class NavigationToleranceTests(unittest.TestCase):
    def test_goal_checker_margin_exceeds_planner_tolerance_and_half_grid_diagonal(self):
        config = (
            yaml.safe_load(
                (Path(__file__).resolve().parents[1] / 'src/home_explore/config/nav2.yaml').read_text()
            )
            if (Path(__file__).resolve().parents[1] / 'src').exists()
            else yaml.safe_load(
                (Path(__file__).resolve().parents[1] / 'home_explore/config/nav2.yaml').read_text()
            )
        )
        c = config['controller_server']['ros__parameters']
        p = config['planner_server']['ros__parameters']['GridBased']
        resolution = config['global_costmap']['global_costmap']['ros__parameters']['resolution']
        self.assertGreater(
            c['general_goal_checker']['xy_goal_tolerance'], p['tolerance'] + resolution / math.sqrt(2)
        )
        self.assertEqual(c['general_goal_checker']['xy_goal_tolerance'], c['FollowPath']['xy_goal_tolerance'])


if __name__ == '__main__':
    unittest.main()
