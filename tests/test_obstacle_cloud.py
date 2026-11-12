import unittest
import numpy as np
from home_explore.obstacle_cloud import depth_xyz


class ObstacleCloudTests(unittest.TestCase):
    def test_known_pixel_coordinates_and_depth_are_preserved(self):
        depth = np.full((3, 3), 2.0, np.float32)
        xyz = depth_xyz(depth, [2, 0, 1, 0, 2, 1, 0, 0, 1], stride=2)
        np.testing.assert_allclose(xyz, [[-1, -1, 2], [1, -1, 2], [-1, 1, 2], [1, 1, 2]])
        self.assertEqual(xyz.dtype, np.dtype('float32'))

    def test_invalid_depth_is_removed_without_inventing_obstacles(self):
        z = np.array([[np.nan, np.inf, 0.1, 8.0, 2.0]], np.float32)
        xyz = depth_xyz(z, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1)
        np.testing.assert_allclose(xyz, [[8, 0, 2]])

    def test_runtime_cloud_is_at_most_one_sixteenth_native_pixels(self):
        xyz = depth_xyz(np.ones((424, 512), np.float32), [400, 0, 255.5, 0, 400, 211.5, 0, 0, 1])
        self.assertEqual(len(xyz), 106 * 128)
        self.assertLess(xyz.nbytes, 170000)

    def test_invalid_intrinsics_fail_closed(self):
        with self.assertRaises(ValueError):
            depth_xyz(np.ones((2, 2)), [0] * 9)


if __name__ == '__main__':
    unittest.main()
