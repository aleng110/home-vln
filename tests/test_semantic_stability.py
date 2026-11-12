"""验证多次追踪中的对象身份和几何稳定性。"""

import unittest
import numpy as np
from home_explore.semantic import Memory, geometry_reliable


def geometry(x=2.0, spread=0.7, depth_spread=0.02, points=None):
    value = dict(
        point=[x, 1.0, 0.7],
        look_at=[x, 1.0, 0.8],
        extent=[1.5, 0.4, 0.5],
        points=[] if points is None else points,
        valid_points=100,
        spread=spread,
        partial=True,
        method='mask_front_depth_band',
    )
    if depth_spread is not None:
        value['depth_spread'] = depth_spread
    return value


class SemanticStabilityTests(unittest.TestCase):
    def memory(self):
        return Memory({'sofa': 0.35, 'tea_table': 0.4, 'dining_table': 0.45, 'chair': 0.5})

    def add(self, m, x=2.0, stamp=10, label='sofa', **kw):
        return m.add(label, 0.9, geometry(x, **kw), [0, 0, 1], stamp, 'image.jpg', source='yolo')

    def test_repeated_closeups_do_not_erase_earlier_observed_surface_bounds(self):
        m = self.memory()
        first = geometry(points=[[x, 1.0, 0.7] for x in np.linspace(1.0, 3.0, 40)])
        obj = m.add('sofa', 0.9, first, [0, 0, 1], 1.0, 'wide', source='yolo')
        for stamp in range(2, 9):
            close = geometry(points=[[3.0, 1.0, 0.7]] * 40)
            current = m.add('sofa', 0.9, close, [0, 0, 1], float(stamp), 'closeup', source='yolo')
        self.assertEqual(current['id'], obj['id'])
        xs = np.array(current['support_points'])[:, 0]
        self.assertLessEqual(xs.min(), 1.1)
        self.assertGreaterEqual(xs.max(), 2.9)

    def test_wide_consistent_surface_does_not_require_endless_observation(self):
        m = self.memory()
        self.add(m, stamp=10)
        obj = self.add(m, stamp=12)
        self.assertTrue(m.quality(obj, 12)['reliable'])
        self.assertGreater(obj['history'][0]['spread'], 0.35)

    def test_small_surface_with_noisy_depth_is_not_reliable(self):
        m = self.memory()
        self.add(m, stamp=10, spread=0.02, depth_spread=0.3)
        obj = self.add(m, stamp=12, spread=0.02, depth_spread=0.3)
        self.assertFalse(m.quality(obj, 12)['reliable'])

    def test_invalid_depth_statistic_does_not_fall_back_to_tight_shape(self):
        value = geometry(spread=0.01)
        value['depth_spread'] = float('nan')
        self.assertFalse(geometry_reliable(value))
        value['depth_spread'] = None
        self.assertFalse(geometry_reliable(value))

    def test_legacy_geometry_keeps_legacy_quality_gate(self):
        self.assertTrue(geometry_reliable(geometry(spread=0.1, depth_spread=None)))
        self.assertFalse(geometry_reliable(geometry(spread=0.7, depth_spread=None)))

    def test_deferred_wide_object_recovers_from_new_consistent_depth(self):
        m = self.memory()
        obj = self.add(m, stamp=10)
        m.defer(obj['id'], 'need_evidence', 11)
        self.add(m, stamp=12)
        self.assertIsNone(m.target('sofa', now=12))
        self.add(m, stamp=14)
        self.assertEqual(m.target('sofa', now=14)['id'], obj['id'])

    def test_confirmed_established_instance_beats_newer_reliable_instance(self):
        m = self.memory()
        self.add(m, stamp=10)
        old = self.add(m, stamp=12)
        old['semantic_confirmation'] = {'stamp': 12, 'decision': {'confirmed': True}}
        for stamp in (13, 15, 17, 19, 21):
            self.add(m, x=5, stamp=stamp)
        self.assertEqual(m.target('sofa', now=21)['id'], old['id'])

    def test_one_newer_glimpse_does_not_replace_established_target(self):
        m = self.memory()
        self.add(m, stamp=10)
        old = self.add(m, stamp=12)
        self.add(m, x=5, stamp=13)
        self.assertEqual(m.target('sofa', now=13)['id'], old['id'])

    def test_preferred_id_never_revives_rejected_or_deferred_object(self):
        m = self.memory()
        old = self.add(m, stamp=10)
        new = self.add(m, x=5, stamp=12)
        self.assertEqual(m.target('sofa', now=12, preferred_id=new['id'])['id'], new['id'])
        m.reject(new['id'], 'not_sofa', 13)
        self.assertEqual(m.target('sofa', now=13, preferred_id=new['id'])['id'], old['id'])
        self.assertIn('rejected', m.get_by_id(new['id']))
        m.defer(old['id'], 'not_ready', 14)
        self.assertIsNone(m.target('sofa', now=14, preferred_id=old['id']))
        self.assertIn('deferred', m.get_by_id(old['id']))

    def test_adjacent_chairs_are_not_merged_by_old_point75_radius(self):
        m = self.memory()
        one = self.add(m, x=2.0, stamp=10, label='chair')
        two = self.add(m, x=2.55, stamp=12, label='chair')
        again = self.add(m, x=2.6, stamp=14, label='chair')
        self.assertNotEqual(one['id'], two['id'])
        self.assertEqual(two['id'], again['id'])

    def test_overlapping_actual_furniture_support_can_link_different_visible_strips(self):
        m = self.memory()
        support = np.column_stack((np.linspace(2.0, 3.0, 40), np.ones(40), np.full(40, 0.7))).tolist()
        one = self.add(m, x=2.0, stamp=10, points=support)
        two = self.add(m, x=2.9, stamp=12, points=support)
        self.assertEqual(one['id'], two['id'])

    def test_extent_alone_cannot_merge_separate_broad_objects(self):
        m = self.memory()
        one = self.add(m, x=2.0, stamp=10)
        two = self.add(m, x=2.9, stamp=12)
        self.assertNotEqual(one['id'], two['id'])

    def test_track_does_not_walk_arbitrarily_far_by_small_updates(self):
        m = self.memory()
        first = self.add(m, x=2.0, stamp=10)
        seen = []
        for n, x in enumerate(np.arange(2.2, 5.0, 0.2), 11):
            obj = self.add(m, x=float(x), stamp=n)
            if obj['id'] == first['id']:
                seen.append(float(x))
        self.assertTrue(seen)
        self.assertLess(max(seen) - 2.0, 1.25)


if __name__ == '__main__':
    unittest.main()
