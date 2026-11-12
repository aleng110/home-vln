"""将无回波激光置为 range_max，配合较小的 SLAM 有效距离清除自由空间。"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class Adapter(Node):
    def __init__(self):
        super().__init__('scan_adapter')
        self.pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan_raw', self.receive, qos_profile_sensor_data)

    def receive(self, msg):
        msg.ranges = [msg.range_max if math.isinf(v) and v > 0 else v for v in msg.ranges]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Adapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
