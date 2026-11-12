import unittest
import math
import numpy as np
from home_explore.grid import Grid
from home_explore.semantic import Memory
from home_explore.task_result import build_result


class ObservationRecoveryTests(unittest.TestCase):
    def geometry(self, x=4.0):
        return dict(
            point=[x, 4.0, 0.5],
            extent=[0.3, 0.3, 0.2],
            points=[],
            valid_points=80,
            spread=0.15,
            method='mask_front_depth_band',
            partial=True,
        )

    def test_observation_goals_leave_nav2_arrival_tolerance(self):
        grid = Grid(np.zeros((120, 120), np.int16), 0.1, (0, 0))
        obj = dict(point=[5.0, 5.0, 0.5], extent=[0.4, 0.4], observer_positions=[])
        goals = grid.docking(obj, (3.7, 5.0, 0.0), 'observe')
        self.assertTrue(goals)
        self.assertTrue(all(math.dist((v['x'], v['y']), (3.7, 5.0)) >= 0.55 for v in goals))
        tried = [(goals[0]['x'], goals[0]['y'])]
        next_goals = grid.docking(obj, (3.7, 5.0, 0.0), 'observe', attempted=tried)
        self.assertTrue(all(math.dist((v['x'], v['y']), tried[0]) >= 0.45 for v in next_goals))

    def test_new_weak_candidate_cannot_replace_reliable_instance(self):
        m = Memory({'tea_table': 0.4})
        m.add('tea_table', 0.7, self.geometry(), [1, 1, 1], 10, 'a', source='yolo')
        strong = m.add('tea_table', 0.7, self.geometry(), [1, 1, 1], 12, 'b', source='yolo')
        m.add('tea_table', 0.7, self.geometry(7), [1, 1, 1], 13, 'c', source='yolo')
        self.assertEqual(m.target('tea_table', now=13)['id'], strong['id'])

    def test_deferred_instance_requires_new_consistent_evidence(self):
        m = Memory({'tea_table': 0.4})
        obj = m.add('tea_table', 0.7, self.geometry(), [1, 1, 1], 10, 'a', source='yolo')
        m.defer(obj['id'], 'insufficient', 11)
        self.assertIsNone(m.target('tea_table', now=11))
        m.add('tea_table', 0.7, self.geometry(), [2, 1, 1], 12, 'b', source='yolo')
        self.assertIsNone(m.target('tea_table', now=12))
        m.add('tea_table', 0.7, self.geometry(), [2, 1, 1], 14, 'c', source='yolo')
        self.assertEqual(m.target('tea_table', now=14)['id'], obj['id'])

    def test_stale_quality_and_historical_quality_are_distinct(self):
        m = Memory({'tea_table': 0.4})
        m.add('tea_table', 0.7, self.geometry(), [1, 1, 1], 10, 'a', source='yolo')
        obj = m.add('tea_table', 0.7, self.geometry(), [1, 1, 1], 12, 'b', source='yolo')
        self.assertFalse(m.quality(obj, 110)['reliable'])
        self.assertTrue(m.quality(obj, obj['last_stamp'])['reliable'])


class ResultTests(unittest.TestCase):
    def test_partial_failure_has_honest_user_reply(self):
        result = build_result(
            dict(
                outcome='failed',
                completed_targets=['dining_table'],
                current_target='tea_table',
                inspection_question='茶几上有什么',
                error='补充观察4次仍不能建立可靠目标几何',
            )
        )
        self.assertIn('已到达餐桌', result['user_message'])
        self.assertIn('茶几', result['user_message'])
        self.assertIn('最终检查尚未完成', result['user_message'])
        self.assertNotIn('没有物品', result['user_message'])
        self.assertFalse(result['inspection_performed'])
        self.assertTrue(result['answer'])

    def test_cancel_has_reply_without_inspection_claim(self):
        result = build_result(
            dict(outcome='cancelled', completed_targets=[], inspection_question='桌上有什么')
        )
        self.assertIn('停止移动', result['user_message'])
        self.assertIsNone(result['inspection_answer'])

    def test_success_keeps_actual_inspection_answer(self):
        result = build_result(
            dict(outcome='succeeded', completed_targets=['tea_table'], inspection_answer='桌上有杯子。')
        )
        self.assertEqual(result['answer'], '桌上有杯子。')
        self.assertTrue(result['inspection_performed'])

    def test_reply_preserves_execution_order_and_final_inspection(self):
        result = build_result(
            dict(
                outcome='succeeded',
                completed_targets=['dining_table', 'tea_table'],
                inspection_answer='茶几上没有物品。',
            )
        )
        self.assertIn('先到达了餐桌', result['user_message'])
        self.assertIn('随后到达茶几', result['user_message'])
        self.assertLess(result['user_message'].index('餐桌'), result['user_message'].index('茶几'))
        self.assertIn('完成检查', result['user_message'])

    def test_repeated_target_steps_are_not_deduplicated_in_reply(self):
        result = build_result(dict(outcome='succeeded', completed_targets=['sofa', 'tea_table', 'sofa']))
        self.assertEqual(result['user_message'].count('沙发'), 2)

    def test_unsupported_task_does_not_claim_navigation(self):
        result = build_result(dict(outcome='failed', error='当前任务不支持：床', completed_targets=[]))
        self.assertIn('不支持', result['user_message'])
        self.assertNotIn('已到达', result['user_message'])


if __name__ == '__main__':
    unittest.main()
