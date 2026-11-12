"""为在线占用图增加未知边框，保留真实观测值。"""

from copy import deepcopy
import math


def pad_grid(message, margin_cells=20):
    """扩展未知边框并补偿地图原点，使原有格子的世界坐标保持一致。"""
    if isinstance(margin_cells, bool) or not isinstance(margin_cells, int) or margin_cells < 0:
        raise ValueError('margin_cells must be a nonnegative integer')
    width, height = message.info.width, message.info.height
    resolution = message.info.resolution
    if width <= 0 or height <= 0 or len(message.data) != width * height:
        raise ValueError('Invalid occupancy-grid dimensions or data length')
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('Occupancy-grid resolution must be positive and finite')
    result = deepcopy(message)
    if margin_cells == 0:
        return result
    q = message.info.origin.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid occupancy-grid origin quaternion')
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    offset = -margin_cells * resolution
    # 按地图姿态旋转原点偏移，同时兼容带倾斜的栅格。
    shift_x = offset * (1 - 2 * (y * y + z * z) + 2 * (x * y - z * w))
    shift_y = offset * (2 * (x * y + z * w) + 1 - 2 * (x * x + z * z))
    shift_z = offset * (2 * (x * z - y * w) + 2 * (y * z + x * w))
    result.info.origin.position.x += shift_x
    result.info.origin.position.y += shift_y
    result.info.origin.position.z += shift_z
    result.info.width = new_width = width + 2 * margin_cells
    result.info.height = new_height = height + 2 * margin_cells
    values = [-1] * (new_width * new_height)
    for row in range(height):
        start = (row + margin_cells) * new_width + margin_cells
        values[start : start + width] = message.data[row * width : (row + 1) * width]
    result.data = values
    return result


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid

    class OnlineMapPadding(Node):
        def __init__(self):
            super().__init__('online_map_padding')
            self.declare_parameter('margin_cells', 20)
            self.margin = self.get_parameter('margin_cells').value
            if isinstance(self.margin, bool) or not isinstance(self.margin, int) or self.margin < 0:
                raise ValueError('margin_cells must be a nonnegative integer')
            qos = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
            )
            self.publisher = self.create_publisher(OccupancyGrid, '/map', qos)
            self.create_subscription(OccupancyGrid, '/map_raw', self.on_map, qos)

        def on_map(self, message):
            try:
                self.publisher.publish(pad_grid(message, self.margin))
            except ValueError as error:
                self.get_logger().warning('拒绝无效在线地图：' + str(error))

    rclpy.init(args=args)
    node = OnlineMapPadding()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
