import threading
import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
from home_explore.object_space import furniture_regions, occupied_by, signed_distance
from home_explore.semantic import Memory
from home_explore.agent import Explorer


class ObjectSpaceTests(unittest.TestCase):
    def memory(self):
        m = Memory({'tea_table': 0.4})
        x, y = np.meshgrid(np.linspace(2, 3, 10), np.linspace(1, 2, 10))
        p = np.column_stack((x.ravel(), y.ravel(), np.full(100, 0.4))).tolist()
        g = dict(
            point=[2.5, 1.5, 0.4],
            look_at=[2.5, 1.5, 0.4],
            extent=[1, 1, 0],
            points=p,
            support_points=p,
            valid_points=100,
            depth_spread=0.02,
            spread=0.4,
            partial=True,
            method='mask_front_depth_band',
        )
        for stamp in (10.0, 12.0):
            m.add('tea_table', 0.9, g, [0, 0, 1], stamp, 'frame.jpg', source='yolo')
        return m

    def test_furniture_interior_is_occupied_even_without_points_in_the_middle(self):
        m = self.memory()
        o = m.objects[0]
        for e in o['history']:
            e['support_points'] = [
                [x, y, 0.4] for x, y in [(2, 1), (2, 2), (3, 1), (3, 2)] for _ in range(10)
            ]
        regions = furniture_regions(m.objects, m.evidence_ok)
        self.assertEqual(occupied_by(regions, [(2.5, 1.5)], 0.39)['label'], 'tea_table')

    def test_robot_body_buffer_excludes_a_near_edge_but_allows_a_clear_goal(self):
        m = self.memory()
        r = furniture_regions(m.objects, m.evidence_ok)
        self.assertIsNotNone(occupied_by(r, [(1.8, 1.5)], 0.39))
        self.assertIsNone(occupied_by(r, [(1.5, 1.5)], 0.39))

    def test_unconfirmed_map_object_cannot_create_a_hard_exclusion_zone(self):
        m = self.memory()
        m.objects[0].update(confirmed=False, state='tentative')
        self.assertEqual(furniture_regions(m.objects, m.evidence_ok), [])

    def test_extreme_isolated_support_does_not_fill_an_unseen_room(self):
        m = self.memory()
        m.objects[0]['history'][-1]['support_points'].append([100, 100, 0.4])
        r = furniture_regions(m.objects, m.evidence_ok)
        self.assertIsNone(occupied_by(r, [(50, 50)], 0.39))

    def test_thin_observed_surface_uses_a_segment_buffer(self):
        self.assertAlmostEqual(signed_distance([[0, 0], [2, 0]], [1, 0.3]), -0.3)
        self.assertAlmostEqual(signed_distance([[0, 0]], [0, 0.3]), -0.3)

    def robot(self):
        a = Explorer.__new__(Explorer)
        a.memory = self.memory()
        a.lock = threading.RLock()
        a.record = Mock()
        a.opt = dict(furniture_clearance=0.32)
        transform = S(translation=S(x=0.0, y=0.0, z=0.0), rotation=S(x=0.0, y=0.0, z=0.0, w=1.0))
        a.tf = S(lookup_transform=Mock(return_value=S(transform=transform)))
        return a

    def test_sofa_goal_cannot_be_inside_an_observed_coffee_table(self):
        a = self.robot()
        a.target = 'sofa'
        self.assertFalse(a.candidate_is_clear(dict(x=2.5, y=1.5, kind='dock')))
        self.assertEqual(a.record.call_args.kwargs['object_id'], 'tea_table_001')

    def test_free_endpoint_does_not_allow_a_path_through_furniture(self):
        a = self.robot()
        g = dict(x=4.0, y=1.5, kind='dock')
        self.assertTrue(a.candidate_is_clear(g))
        self.assertFalse(a.candidate_is_clear(g, path=[(0, 1.5), (2.5, 1.5), (4, 1.5)]))

    def test_map_to_odom_rotation_and_translation_are_applied(self):
        a = self.robot()
        a.tf.lookup_transform.return_value.transform = S(
            translation=S(x=10.0, y=20.0, z=0.0), rotation=S(x=0.0, y=0.0, z=2**-0.5, w=2**-0.5)
        )
        self.assertFalse(a.candidate_is_clear(dict(x=-18.5, y=7.5, kind='dock')))
        np.testing.assert_allclose(a.record.call_args.kwargs['point_odom'], [2.5, 1.5], atol=1e-6)

    def test_stationary_visual_checks_do_not_require_new_translation_clearance(self):
        a = self.robot()
        self.assertTrue(a.candidate_is_clear(dict(x=2.5, y=1.5, kind='view')))
        a.tf.lookup_transform.assert_not_called()


if __name__ == '__main__':
    unittest.main()
