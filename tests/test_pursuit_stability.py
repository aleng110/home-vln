"""验证目标保持和局部失败后的停止行为。"""

import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from home_explore.agent import Explorer
from home_explore.semantic import Memory


def geometry(x):
    return dict(
        point=[x, 0, 0.7],
        look_at=[x, 0, 0.8],
        extent=[0.5, 0.5, 0.3],
        points=[],
        valid_points=100,
        spread=0.6,
        depth_spread=0.02,
        method='mask_front_depth_band',
        partial=True,
    )


class PursuitStabilityTests(unittest.TestCase):
    def robot(self):
        robot = Explorer.__new__(Explorer)
        robot.check = Mock()
        robot.pose = Mock(return_value=(0.0, 0.0, 0.0))
        robot.lock = threading.RLock()
        robot.map_received = time.monotonic()
        robot.target = 'sofa'
        robot.pinned_object_id = None
        robot.local_view_attempts = set()
        robot.pursuit_rechecks = set()
        robot.observation_views = []
        robot.visited = []
        robot.blocked = []
        robot.visited_views = []
        robot.search_look_attempts = []
        robot.travel_heading = 0.0
        robot.search_moves = 0
        robot.opt = {'near_distance': 1.25, 'docking_standoff': 0.95, 'search_looks_per_location': 2}
        robot.grid = SimpleNamespace(
            docking=Mock(return_value=[]),
            frontiers=Mock(return_value=[]),
            approach_progress=Mock(return_value=[]),
            free=np.ones((2, 2), bool),
            data=np.zeros((2, 2), np.int16),
            resolution=0.05,
            origin=(0, 0),
        )
        robot.memory = Mock()
        robot.memory.quality.return_value = {'reliable': True, 'reasons': []}
        robot.publish = Mock()
        robot.publish_objects = Mock()
        robot.record = Mock()
        robot.path_candidate = Mock(return_value=None)
        robot.mapped_search_views = Mock(return_value=[])
        robot.map_geometry = lambda obj: dict(
            obj, point_odom=list(obj['point']), look_at_odom=list(obj.get('look_at', obj['point']))
        )
        robot.get_clock = Mock(
            return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=20_000_000_000))
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        robot.run_dir = Path(self.temp.name)
        return robot

    def obj(self, x=3.0):
        return {
            'id': 'sofa_001',
            'label': 'sofa',
            'point': [x, 0, 0.7],
            'look_at_odom': [x, 0, 0.8],
            'last_stamp': 12.0,
            'history': [],
        }

    def test_verified_range_request_precedes_repeating_the_same_dock(self):
        robot = self.robot()
        obj = self.obj(1.33)
        robot.get_target = Mock(return_value=(obj, {'reliable': True, 'reasons': []}))
        robot.grid.safe = Mock(return_value=True)
        robot.range_request = {
            'object_id': obj['id'],
            'point_odom': [0.35, 0, 0],
            'look_at_odom': [1.33, 0, 0.7],
        }
        robot.path_candidate.side_effect = lambda values: dict(values[0], path_length=0.35)
        goal, _, _ = Explorer.decide(robot)
        self.assertTrue(goal['range_refinement'])
        self.assertEqual(goal['x'], 0.35)
        robot.grid.docking.assert_not_called()
        robot.grid.frontiers.assert_not_called()
        self.assertEqual(robot.range_refinement_count, 1)
        self.assertIsNone(robot.range_request)

    def test_short_range_refinement_rejects_a_long_detour(self):
        robot = self.robot()
        obj = self.obj(1.33)
        robot.get_target = Mock(return_value=(obj, {'reliable': True, 'reasons': []}))
        robot.grid.safe = Mock(return_value=True)
        robot.range_request = {
            'object_id': obj['id'],
            'point_odom': [0.35, 0, 0],
            'look_at_odom': [1.33, 0, 0.7],
        }
        robot.path_candidate.side_effect = lambda values: dict(values[0], path_length=5.0) if values else None
        goal, _, _ = Explorer.decide(robot)
        self.assertEqual(goal['kind'], 'view')
        self.assertNotIn('range_refinement', goal)
        robot.grid.frontiers.assert_not_called()

    def test_get_target_keeps_pinned_instance_despite_newer_reliable_instance(self):
        robot = self.robot()
        memory = Memory({'sofa': 0.35})
        old = memory.add('sofa', 0.9, geometry(2.0), [0, 0, 1], 10.0, 'old', source='yolo')
        for stamp in (12.0, 14.0, 16.0):
            memory.add('sofa', 0.9, geometry(5.0), [0, 0, 1], stamp, 'new', source='yolo')
        robot.memory = memory
        robot.pinned_object_id = old['id']
        selected, _ = Explorer.get_target(robot)
        self.assertEqual(selected['id'], old['id'])
        self.assertEqual(robot.pinned_object_id, old['id'])
        robot.record.assert_not_called()

    def test_weak_one_frame_anchor_can_yield_to_reliable_observed_target(self):
        robot = self.robot()
        memory = Memory({'sofa': 0.35})
        weak = geometry(2.0)
        weak['valid_points'] = 37
        old = memory.add('sofa', 0.9, weak, [0, 0, 1], 10.0, 'wall_contaminated', source='yolo')
        for stamp in (12.0, 14.0):
            new = memory.add('sofa', 0.9, geometry(5.0), [0, 0, 1], stamp, 'actual_furniture', source='yolo')
        robot.memory = memory
        robot.pinned_object_id = old['id']
        selected, _ = Explorer.get_target(robot)
        self.assertEqual(selected['id'], new['id'])
        self.assertFalse(old.get('rejected', False))

    def test_weak_anchor_is_not_replaced_by_another_weak_glimpse(self):
        robot = self.robot()
        memory = Memory({'sofa': 0.35})
        weak = geometry(2.0)
        weak['valid_points'] = 37
        old = memory.add('sofa', 0.9, weak, [0, 0, 1], 10.0, 'old', source='yolo')
        other = geometry(5.0)
        other['valid_points'] = 37
        memory.add('sofa', 0.9, other, [0, 0, 1], 14.0, 'other', source='yolo')
        robot.memory = memory
        robot.pinned_object_id = old['id']
        selected, _ = Explorer.get_target(robot)
        self.assertEqual(selected['id'], old['id'])

    def test_pinned_temporarily_deferred_goal_cannot_silently_become_unknown(self):
        robot = self.robot()
        memory = Memory({'sofa': 0.35})
        obj = memory.add('sofa', 0.9, geometry(2.0), [0, 0, 1], 10.0, 'old', source='yolo')
        memory.defer(obj['id'], 'temporary_occlusion', 12.0)
        robot.memory = memory
        robot.pinned_object_id = obj['id']
        try:
            selected, _ = Explorer.get_target(robot)
        except RuntimeError:
            return  # 有上限的明确失败也符合停止要求。
        self.assertIsNotNone(
            selected, 'A temporary visual miss must not turn a committed target into global search'
        )
        self.assertEqual(selected['id'], obj['id'])

    def test_known_far_goal_without_approach_gets_one_local_check_then_stops(self):
        robot = self.robot()
        obj = self.obj(3.0)
        robot.get_target = Mock(return_value=(obj, {'reliable': True, 'reasons': []}))
        with patch('home_explore.agent.coverage_candidates') as coverage:
            goal, selected, _ = Explorer.decide(robot)
            self.assertEqual(goal['kind'], 'view')
            self.assertEqual((goal['x'], goal['y']), (0, 0))
            self.assertEqual(selected['id'], obj['id'])
            with self.assertRaises(RuntimeError):
                Explorer.decide(robot)
        coverage.assert_not_called()
        robot.grid.frontiers.assert_not_called()

    def test_known_goal_can_make_local_progress_without_global_room_search(self):
        robot = self.robot()
        obj = self.obj(4.0)
        robot.get_target = Mock(return_value=(obj, {'reliable': True, 'reasons': []}))
        progress = dict(x=0.7, y=0.0, yaw=0.0, score=0.7, kind='approach_progress', object_id=obj['id'])
        robot.grid.approach_progress.return_value = [progress]
        robot.path_candidate.side_effect = lambda candidates: (
            dict(candidates[0], path_score=0.7) if candidates else None
        )
        with patch('home_explore.agent.coverage_candidates') as coverage:
            goal, selected, _ = Explorer.decide(robot)
        self.assertEqual(goal['kind'], 'approach_progress')
        self.assertEqual(selected['id'], obj['id'])
        coverage.assert_not_called()
        robot.grid.frontiers.assert_not_called()

    def test_known_near_goal_does_not_get_two_independent_stationary_rechecks(self):
        robot = self.robot()
        obj = self.obj(0.9)
        robot.get_target = Mock(return_value=(obj, {'reliable': True, 'reasons': []}))
        with patch('home_explore.agent.coverage_candidates') as coverage:
            first, _, _ = Explorer.decide(robot)
            self.assertEqual(first['kind'], 'view')
            with self.assertRaises(RuntimeError):
                Explorer.decide(robot)
        coverage.assert_not_called()
        robot.grid.frontiers.assert_not_called()

    def test_unknown_target_checks_informative_current_view_before_translation(self):
        robot = self.robot()
        robot.get_target = Mock(return_value=(None, None))
        view = dict(x=0.0, y=0.0, yaw=1.0, score=-1.0, kind='search_look', information_cells=100)
        with patch('home_explore.agent.local_observation_candidates', return_value=[view]), patch(
            'home_explore.agent.coverage_candidates'
        ) as coverage:
            goal, obj, _ = Explorer.decide(robot)
        self.assertEqual(goal['kind'], 'search_look')
        self.assertIsNone(obj)
        self.assertEqual((goal['x'], goal['y']), (0, 0))
        coverage.assert_not_called()
        robot.grid.frontiers.assert_not_called()
        robot.path_candidate.assert_not_called()

    def test_two_local_observations_exhaust_current_position_budget(self):
        robot = self.robot()
        robot.get_target = Mock(return_value=(None, None))
        robot.search_look_attempts = [(0.0, 0.0, 1.0), (0.05, 0.05, -1.0)]
        frontier = dict(x=1.0, y=0.0, yaw=0.0, score=0.0, kind='frontier', information_cells=10)
        robot.grid.frontiers.return_value = [frontier]
        robot.path_candidate.side_effect = lambda values: dict(values[0], path_score=values[0]['score'] + 1.0)
        with patch('home_explore.agent.local_observation_candidates') as local, patch(
            'home_explore.agent.coverage_candidates', return_value=([], np.zeros((2, 2), bool))
        ), patch('home_explore.agent.camera_view_mask', return_value=np.zeros((2, 2), bool)):
            goal, _, _ = Explorer.decide(robot)
        local.assert_not_called()
        self.assertEqual(goal['kind'], 'frontier')

    def test_old_observations_elsewhere_do_not_consume_current_position_budget(self):
        robot = self.robot()
        robot.get_target = Mock(return_value=(None, None))
        robot.search_look_attempts = [(3.0, 3.0, 1.0), (4.0, 4.0, -1.0)]
        view = dict(x=0.0, y=0.0, yaw=1.0, score=-1.0, kind='search_look', information_cells=100)
        with patch('home_explore.agent.local_observation_candidates', return_value=[view]) as local:
            goal, _, _ = Explorer.decide(robot)
        local.assert_called_once()
        self.assertEqual(goal['kind'], 'search_look')

    def test_historically_valid_anchor_uses_approach_then_fresh_verification(self):
        robot = self.robot()
        obj = self.obj(3.0)
        robot.get_target = Mock(return_value=(obj, {'reliable': False, 'reasons': ['stale']}))
        robot.memory.quality.return_value = {'reliable': True, 'reasons': []}
        dock = dict(x=2.0, y=0.0, yaw=0.0, score=1.0, kind='dock')
        robot.grid.docking.return_value = [dock]
        robot.path_candidate.return_value = dock
        goal, _, _ = Explorer.decide(robot)
        self.assertEqual(goal['kind'], 'dock')
        self.assertEqual(robot.grid.docking.call_args.args[2], 'dock')


if __name__ == '__main__':
    unittest.main()
