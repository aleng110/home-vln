import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock, patch
import numpy as np
from home_explore.inspection import assess_inspection, run_inspection, InspectionIncomplete, inspection_prompt


class InspectionTests(unittest.TestCase):
    def valid(self):
        return dict(
            observed_label='dining_table',
            confidence=0.9,
            bbox_format='norm1000',
            target_in_frame=True,
            target_bbox=[100, 100, 900, 970],
            region_bbox=[200, 250, 800, 700],
            region_polygon=[[200, 250], [800, 250], [800, 700], [200, 700]],
            full_region_visible=True,
            occluded=False,
            answer='桌面上没有发现物品。',
            reason='四条边与四个角点均完整可见',
        )

    def test_prompt_uses_actual_coordinates_and_distinguishes_cast_shadows(self):
        prompt = inspection_prompt('dining_table', '桌上有什么？', '桌面')
        self.assertNotIn('[100,100,900,950]', prompt)
        self.assertIn('投影阴影', prompt)
        self.assertIn('坐标必须从实际图像测得', prompt)
        self.assertIn('待报告的桌上物品本身不是外部遮挡', prompt)

    def test_full_tabletop_is_accepted_without_requiring_all_legs(self):
        self.assertTrue(assess_inspection(self.valid(), 'dining_table')['complete'])

    def test_clipped_region_cannot_claim_complete_even_if_model_says_true(self):
        raw = self.valid()
        raw['region_bbox'] = [200, 250, 1000, 700]
        raw['region_polygon'][1][0] = 1000
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_complete_perspective_seat_does_not_require_ten_percent_image_height(self):
        raw = self.valid()
        raw['region_bbox'] = [352, 426, 822, 512]
        raw['region_polygon'] = [[352, 426], [822, 426], [822, 512], [352, 512]]
        self.assertTrue(assess_inspection(raw, 'dining_table')['complete'])

    def test_region_from_another_object_cannot_claim_complete(self):
        raw = self.valid()
        raw['target_bbox'] = [50, 50, 150, 950]
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_occluded_region_is_incomplete(self):
        raw = self.valid()
        raw['occluded'] = True
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_cropped_furniture_does_not_complete_an_inspection(self):
        raw = self.valid()
        raw['target_bbox'][2] = 1000
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_uncertain_answer_cannot_be_published_as_a_completed_inspection(self):
        raw = self.valid()
        raw['answer'] = '部分区域不可见，无法确认是否有物品。'
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_wrong_object_or_low_resolution_is_not_complete(self):
        raw = self.valid()
        self.assertFalse(assess_inspection(raw, 'sofa')['complete'])
        raw['region_bbox'] = [200, 250, 800, 300]
        self.assertFalse(assess_inspection(raw, 'dining_table')['complete'])

    def test_sofa_arms_and_back_are_explicit_occluders(self):
        self.assertIn('扶手、靠背挡住座面', inspection_prompt('sofa', '沙发上有什么', '座面'))

    def robot(self, folder, raw):
        obj = dict(point=[2, 0, 0], look_at_odom=[2, 0, 0], support_points_odom=[[2, 0, 0]] * 30)
        image = np.zeros((424, 512, 3), np.uint8)
        info = S(k=[400, 0, 256, 0, 400, 212, 0, 0, 1])
        tf = S(
            transform=S(rotation=S(x=0.0, y=2**-0.5, z=0.0, w=2**-0.5), translation=S(x=0.0, y=0.0, z=0.0))
        )
        return S(
            get_target=lambda: (obj, {}),
            last_xy=(0.0, 0.0),
            pose=lambda: (0.0, 0.0, 0.0),
            check=lambda: None,
            set_motion=lambda value: None,
            publish=Mock(),
            record=Mock(),
            local_detector=None,
            capture=lambda: (image, None, info, 1.0, tf),
            run_dir=Path(folder),
            budget_ask=lambda *a: copy.deepcopy(raw),
            target='dining_table',
            task_token='test',
            navigate=Mock(),
        )

    def test_incomplete_region_budget_ends_in_explicit_failure(self):
        raw = self.valid()
        raw['full_region_visible'] = False
        raw['answer'] = ''
        with tempfile.TemporaryDirectory() as folder:
            robot = self.robot(folder, raw)
            with self.assertRaisesRegex(InspectionIncomplete, '检查未完成'):
                run_inspection(robot, '桌上有什么？', max_views=1)
            self.assertFalse(robot.inspection_result['complete'])
            robot.navigate.assert_not_called()

    def test_only_complete_assessment_saves_final_inspection_image_and_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            robot = self.robot(folder, self.valid())
            self.assertEqual(run_inspection(robot, '桌上有什么？', max_views=1), '桌面上没有发现物品。')
            self.assertTrue(robot.inspection_result['complete'])
            self.assertTrue((Path(folder) / 'inspection.jpg').exists())

    def test_complete_inspection_after_move_finishes_without_return_or_background_wait(self):
        first = self.valid()
        first['full_region_visible'] = False
        with tempfile.TemporaryDirectory() as folder:
            a = self.robot(folder, first)
            a.publish_objects = Mock()
            a.path_candidate = lambda choices: choices[0]
            a.budget_ask = Mock(side_effect=[first, self.valid()])
            a.verify = Mock(
                side_effect=AssertionError(
                    'A completed inspection must not start another arrival verification'
                )
            )
            a.fresh_observation = Mock(
                side_effect=AssertionError('Inspection has paused background perception')
            )
            a.navigate = Mock(side_effect=lambda goal: setattr(a, 'last_xy', (1.0, 0.0)))
            with patch(
                'home_explore.inspection.inspection_candidates',
                return_value=[dict(x=1.0, y=0.0, yaw=0.0, kind='inspection')],
            ):
                answer = run_inspection(a, '桌上有什么？')
            self.assertTrue(answer)
            a.navigate.assert_called_once()
            a.verify.assert_not_called()
            a.fresh_observation.assert_not_called()
            self.assertEqual(len(a.inspection_result['attempts']), 2)

    def test_complete_current_image_is_not_rejected_for_imperfect_geometric_bearing(self):
        raw = self.valid()
        raw['observed_label'] = 'sofa'
        with tempfile.TemporaryDirectory() as folder:
            a = self.robot(folder, raw)
            a.target = 'sofa'
            a.last_xy = (3.0, 0.0)
            obj, _ = a.get_target()
            self.assertTrue(run_inspection(a, '沙发上有什么？', max_views=1))
            a.navigate.assert_not_called()

    def test_explicit_resource_budget_stops_without_claiming_inspection_success(self):
        first = self.valid()
        first['full_region_visible'] = False
        with tempfile.TemporaryDirectory() as folder:
            a = self.robot(folder, first)
            a.publish_objects = Mock()
            a.path_candidate = lambda choices: choices[0]
            with patch(
                'home_explore.inspection.inspection_candidates',
                return_value=[dict(x=1.0, y=0.0, yaw=0.0, kind='inspection')],
            ):
                with self.assertRaises(InspectionIncomplete):
                    run_inspection(a, '桌上有什么？', max_views=2)
            a.navigate.assert_called_once()
            self.assertFalse(a.inspection_result['complete'])


if __name__ == '__main__':
    unittest.main()
