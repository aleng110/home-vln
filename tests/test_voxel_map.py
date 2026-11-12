import json
import unittest
import numpy as np
from home_explore.voxel_fusion import VoxelFusion
from home_explore.semantic import Memory


def cloud(x=0.0):
    return np.array([[x + a, b, 1.0] for a in np.linspace(0, 0.8, 12) for b in np.linspace(0, 0.8, 12)])


def geometry(points):
    p = np.asarray(points)
    return dict(
        point=np.mean(p, axis=0).tolist(),
        look_at=np.mean(p, axis=0).tolist(),
        extent=np.ptp(p, axis=0).tolist(),
        points=p.tolist(),
        support_points=p.tolist(),
        valid_points=len(p),
        depth_spread=0.02,
        spread=0.4,
        method='mask_front_depth_band',
        partial=True,
    )


class VoxelMapTests(unittest.TestCase):
    def test_pixels_and_replayed_same_frame_do_not_fake_multi_frame_confirmation(self):
        f = VoxelFusion()
        p = np.repeat(cloud(), 4, axis=0)
        f.integrate('a', p, 1.0)
        f.integrate('a', p, 1.0)
        _, total, stable = f.summary('a')
        self.assertGreater(total, 20)
        self.assertEqual(stable, 0)
        f.integrate('a', p, 2.0)
        self.assertGreater(f.summary('a')[2], 20)

    def test_repeated_closeup_does_not_drag_fused_centre_to_one_corner(self):
        f = VoxelFusion()
        p = cloud()
        f.integrate('a', p, 1)
        f.integrate('a', p, 2)
        centre = f.summary('a')[0].mean(axis=0)
        for stamp in range(3, 12):
            f.integrate('a', p[p[:, 0] < 0.25], stamp)
        np.testing.assert_allclose(f.summary('a')[0].mean(axis=0), centre, atol=f.resolution * 0.1)

    def test_one_frame_outlier_does_not_expand_confirmed_shape(self):
        f = VoxelFusion()
        f.integrate('a', cloud(), 1)
        f.integrate('a', cloud(), 2)
        f.integrate('a', np.vstack((cloud(), [100, 100, 100])), 3)
        self.assertLess(f.summary('a')[0].max(), 2.0)

    def test_visible_free_space_clears_only_after_repeated_contradictions(self):
        f = VoxelFusion()
        p = cloud(-0.4)
        f.integrate('a', p, 1)
        f.integrate('a', p, 2)
        k = [20, 0, 20, 0, 20, 20, 0, 0, 1]
        depth = np.full((50, 50), 3.0)
        for stamp in (3, 4):
            self.assertEqual(f.clear_observed_free_space(depth, k, [0, 0, 0], [0, 0, 0, 1], stamp), 0)
        self.assertGreater(f.clear_observed_free_space(depth, k, [0, 0, 0], [0, 0, 0, 1], 5), 0)
        self.assertEqual(f.summary('a')[1], 0)

    def test_occlusion_invalid_depth_and_out_of_view_preserve_semantics(self):
        f = VoxelFusion()
        f.integrate('a', cloud(), 1)
        f.integrate('a', cloud(), 2)
        k = [20, 0, 20, 0, 20, 20, 0, 0, 1]
        before = f.summary('a')[1]
        for stamp in range(3, 9):
            f.clear_observed_free_space(np.full((50, 50), 0.6), k, [0, 0, 0], [0, 0, 0, 1], stamp)
        self.assertEqual(f.summary('a')[1], before)
        for stamp in range(9, 15):
            f.clear_observed_free_space(np.full((50, 50), np.nan), k, [0, 0, 0], [0, 0, 0, 1], stamp)
        self.assertEqual(f.summary('a')[1], before)
        for stamp in range(15, 20):
            f.clear_observed_free_space(np.full((50, 50), 3.0), k, [50, 0, 0], [0, 0, 0, 1], stamp)
        self.assertEqual(f.summary('a')[1], before)

    def test_memory_tracks_spatial_overlap_and_publishes_confirmed_voxels(self):
        m = Memory({'sofa': 0.4})
        g = geometry(cloud())
        first = m.add('sofa', 0.9, g, [0, 0, 0], 1, 'a', source='yolo')
        self.assertFalse(first['confirmed'])
        second = m.add('sofa', 0.9, g, [0, 0, 0], 2, 'b', source='yolo')
        self.assertEqual(first['id'], second['id'])
        self.assertTrue(second['confirmed'])
        self.assertGreater(second['stable_voxel_count'], 20)
        self.assertEqual(m.snapshot(False)['observations'], [])
        self.assertNotIn('history', m.snapshot(False)['objects'][0])
        json.dumps(m.snapshot())

    def test_one_frame_per_instance_assignment_keeps_adjacent_objects_separate(self):
        m = Memory({'chair': 0.4})
        g = geometry(cloud())
        a = m.add('chair', 0.9, g, [0, 0, 0], 1, 'a', source='yolo')
        b = m.add('chair', 0.9, g, [0, 0, 0], 1, 'a', source='yolo', exclude_ids=[a['id']])
        self.assertNotEqual(a['id'], b['id'])

    def test_low_confidence_repetitions_never_create_confirmed_furniture(self):
        m = Memory({'sofa': 0.8})
        g = geometry(cloud())
        for stamp in range(5):
            a = m.add('sofa', 0.2, g, [0, 0, 0], stamp, 'a', source='yolo')
        self.assertFalse(a['confirmed'])
        self.assertEqual(a['stable_voxel_count'], 0)

    def test_cleared_object_is_not_a_navigation_target(self):
        m = Memory({'sofa': 0.4})
        g = geometry(cloud(-0.4))
        for stamp in (1, 2):
            a = m.add('sofa', 0.9, g, [0, 0, 0], stamp, 'a', source='yolo')
        k = [20, 0, 20, 0, 20, 20, 0, 0, 1]
        for stamp in (3, 4, 5):
            m.observe_depth(np.full((50, 50), 3.0), k, [0, 0, 0], [0, 0, 0, 1], stamp)
        self.assertEqual(a['state'], 'cleared')
        self.assertIsNone(m.target('sofa', now=5))


if __name__ == '__main__':
    unittest.main()
