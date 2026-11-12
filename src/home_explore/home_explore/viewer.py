"""查看实时或检查图像，不发布运动指令。"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from .image_utils import decode_ros_image


def main(args=None):
    rclpy.init(args=args)
    node = Node('vln_camera_viewer')
    node.declare_parameter('image_topic', '/camera/image_raw')
    node.declare_parameter('transient_local', False)
    topic = node.get_parameter('image_topic').value
    qos = (
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        if node.get_parameter('transient_local').value
        else qos_profile_sensor_data
    )
    latest = [None]
    node.create_subscription(Image, topic, lambda m: latest.__setitem__(0, m), qos)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if latest[0] is not None:
                cv2.imshow('WPB Home VLN camera - Q to close', decode_ros_image(latest[0]))
                if cv2.waitKey(1) & 255 in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
