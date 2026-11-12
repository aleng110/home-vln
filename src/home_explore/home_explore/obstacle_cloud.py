"""将对齐深度转为小规模、带时间戳的避障点云。"""

import time
import json
import numpy as np
from .image_utils import decode_ros_image


def depth_xyz(depth, k, stride=4):
    depth = np.asarray(depth)
    if depth.ndim != 2 or len(k) != 9 or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0:
        raise ValueError('Invalid aligned depth/intrinsics')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('Invalid depth stride')
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0.5) & (z < 5.8)
    return np.column_stack(
        ((u[valid] - k[2]) * z[valid] / k[0], (v[valid] - k[5]) * z[valid] / k[4], z[valid])
    ).astype('<f4')


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.clock import Clock, ClockType
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField

    class ObstacleCloud(Node):
        def __init__(self):
            super().__init__('aligned_depth_obstacle_cloud')
            self.info = None
            self.last_stamp = -1.0
            self.last_warning = 0.0
            self.depth_received = 0
            self.info_received = 0
            self.published = 0
            self.last_depth_wall = None
            self.last_info_wall = None
            self.last_publish_wall = None
            self.last_depth_stamp = None
            self.reason = 'waiting_depth'
            self.declare_parameter('reliable_depth', True)
            # 大体积深度帧使用可靠传输和短队列；小点云使用 BEST_EFFORT。
            depth_qos = QoSProfile(
                depth=1,
                reliability=(
                    ReliabilityPolicy.RELIABLE
                    if self.get_parameter('reliable_depth').value
                    else ReliabilityPolicy.BEST_EFFORT
                ),
            )
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(PointCloud2, '/camera/obstacle_points', qos)
            self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info, qos)
            self.create_subscription(Image, '/camera/depth/image_raw', self.depth, depth_qos)
            self.status_pub = self.create_publisher(
                String,
                '/vln/obstacle_cloud/status',
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
            self.create_timer(1.0, self.report, clock=Clock(clock_type=ClockType.STEADY_TIME))

        def camera_info(self, message):
            self.info = message
            self.info_received += 1
            self.last_info_wall = time.monotonic()

        def report(self):
            wall = time.monotonic()
            age = lambda received: None if received is None else wall - received
            state = dict(
                ready=self.last_publish_wall is not None and wall - self.last_publish_wall < 1.0,
                reason=self.reason,
                depth_received=self.depth_received,
                info_received=self.info_received,
                published=self.published,
                depth_age_wall=age(self.last_depth_wall),
                info_age_wall=age(self.last_info_wall),
                cloud_age_wall=age(self.last_publish_wall),
                last_depth_stamp=self.last_depth_stamp,
                last_cloud_stamp=self.last_stamp,
                ros_time=self.get_clock().now().nanoseconds / 1e9,
                reliable_depth=self.get_parameter('reliable_depth').value,
                width=self.info.width if self.info else None,
                height=self.info.height if self.info else None,
                frame=self.info.header.frame_id if self.info else None,
            )
            self.status_pub.publish(String(data=json.dumps(state)))

        def depth(self, message):
            self.depth_received += 1
            self.last_depth_wall = time.monotonic()
            stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            self.last_depth_stamp = stamp
            if self.info is None:
                self.reason = 'waiting_camera_info'
                return
            if stamp - self.last_stamp < 0.15:
                return
            now = self.get_clock().now().nanoseconds / 1e9
            if now - stamp > 1.0 or stamp - now > 0.3:
                self.reason = 'depth_clock_mismatch'
                return
            try:
                if (message.height, message.width, message.header.frame_id) != (
                    self.info.height,
                    self.info.width,
                    self.info.header.frame_id,
                ):
                    raise ValueError('Depth and camera calibration must be aligned')
                xyz = depth_xyz(decode_ros_image(message), self.info.k)
                cloud = PointCloud2()
                cloud.header = message.header
                cloud.height = 1
                cloud.width = len(xyz)
                cloud.fields = [
                    PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                    for index, name in enumerate(('x', 'y', 'z'))
                ]
                cloud.is_bigendian = False
                cloud.point_step = 12
                cloud.row_step = len(xyz) * 12
                cloud.is_dense = True
                cloud.data = xyz.tobytes()
                self.pub.publish(cloud)
                self.last_stamp = stamp
                self.published += 1
                self.last_publish_wall = time.monotonic()
                self.reason = 'publishing'
            except ValueError as error:
                self.reason = 'invalid_depth_or_calibration: ' + str(error)
                if time.monotonic() - self.last_warning > 3.0:
                    self.get_logger().warning(str(error))
                    self.last_warning = time.monotonic()

    rclpy.init(args=args)
    node = ObstacleCloud()
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
