import math
from types import SimpleNamespace as Message
import unittest

from home_explore.map_adapter import pad_grid


def grid(yaw=0.0):
    return Message(
        header=Message(stamp=Message(sec=12, nanosec=345), frame_id='map'),
        info=Message(
            width=3,
            height=2,
            resolution=0.05,
            map_load_time=Message(sec=8, nanosec=9),
            origin=Message(
                position=Message(x=-6.0, y=2.0, z=0.2),
                orientation=Message(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
            ),
        ),
        data=[-1, 0, 100, 15, 60, -1],
    )


class MapAdapterTests(unittest.TestCase):
    def test_dimensions_and_row_major_values_preserved(self):
        original = grid()
        result = pad_grid(original, 2)
        self.assertEqual((result.info.width, result.info.height), (7, 6))
        for row in range(2):
            start = (row + 2) * 7 + 2
            self.assertEqual(result.data[start : start + 3], original.data[row * 3 : (row + 1) * 3])

    def test_entire_added_border_is_unknown_no_free_space_invented(self):
        result = pad_grid(grid(), 2)
        for row in range(result.info.height):
            for column in range(result.info.width):
                if not (2 <= row < 4 and 2 <= column < 5):
                    self.assertEqual(result.data[row * result.info.width + column], -1)
        self.assertEqual(result.data.count(0), 1)
        self.assertEqual(result.data.count(100), 1)

    def test_origin_moves_by_margin_at_zero_yaw(self):
        result = pad_grid(grid(), 20)
        self.assertAlmostEqual(result.info.origin.position.x, -7.0)
        self.assertAlmostEqual(result.info.origin.position.y, 1.0)
        self.assertAlmostEqual(result.info.origin.position.z, 0.2)

    def test_rotated_origin_and_cell_world_coordinates(self):
        for yaw in (math.pi / 2, -0.7, math.pi):
            original = grid(yaw)
            result = pad_grid(original, 4)
            offset = -0.2
            self.assertAlmostEqual(
                result.info.origin.position.x, -6 + offset * (math.cos(yaw) - math.sin(yaw))
            )
            self.assertAlmostEqual(
                result.info.origin.position.y, 2 + offset * (math.sin(yaw) + math.cos(yaw))
            )

            # 原始格子中心的世界坐标保持不变。
            def cell_world(message, column, row):
                dx = (column + 0.5) * message.info.resolution
                dy = (row + 0.5) * message.info.resolution
                return (
                    message.info.origin.position.x + math.cos(yaw) * dx - math.sin(yaw) * dy,
                    message.info.origin.position.y + math.sin(yaw) * dx + math.cos(yaw) * dy,
                )

            before, after = cell_world(original, 1, 0), cell_world(result, 5, 4)
            self.assertAlmostEqual(before[0], after[0])
            self.assertAlmostEqual(before[1], after[1])

    def test_metadata_and_orientation_preserved(self):
        original = grid(0.4)
        result = pad_grid(original)
        self.assertEqual(result.header, original.header)
        self.assertEqual(result.info.map_load_time, original.info.map_load_time)
        self.assertEqual(result.info.resolution, original.info.resolution)
        self.assertEqual(result.info.origin.orientation, original.info.origin.orientation)

    def test_no_mutation_and_no_shared_mutable_children(self):
        original = grid()
        result = pad_grid(original, 1)
        result.header.frame_id = 'changed'
        result.info.origin.position.x = 999.0
        result.info.origin.orientation.w = 0.0
        result.data[0] = 100
        self.assertEqual(original.header.frame_id, 'map')
        self.assertEqual(original.info.origin.position.x, -6.0)
        self.assertEqual(original.info.origin.orientation.w, 1.0)
        self.assertEqual(original.data, [-1, 0, 100, 15, 60, -1])

    def test_zero_margin_still_returns_independent_copy(self):
        original = grid()
        result = pad_grid(original, 0)
        self.assertEqual(result, original)
        self.assertIsNot(result.data, original.data)

    def test_invalid_inputs_rejected(self):
        for margin in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                pad_grid(grid(), margin)
        original = grid()
        original.data.pop()
        with self.assertRaises(ValueError):
            pad_grid(original)
        original = grid()
        original.info.resolution = float('nan')
        with self.assertRaises(ValueError):
            pad_grid(original)


if __name__ == '__main__':
    unittest.main()
