"""统一速度出口：墙钟租约、输入看门狗与速度限幅。"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class Guard(Node):
    def __init__(self):
        super().__init__('vln_velocity_guard')
        self.enabled_at = 0.0
        self.command_at = 0.0
        self.command = Twist()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Twist, '/vln/cmd_vel_raw', self.on_command, 10)
        self.create_subscription(Bool, '/vln/motion_enabled', self.on_lease, 10)
        # 仿真时钟暂停或任务进程退出后也须停车。
        self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_lease(self, msg):
        self.enabled_at = time.monotonic() if msg.data else 0.0
        if not msg.data:
            self.command_at = 0.0
            self.pub.publish(Twist())

    def on_command(self, msg):
        values = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(v) for v in values):
            self.command_at = 0.0
            return
        self.command = msg
        self.command_at = time.monotonic()

    def tick(self):
        now = time.monotonic()
        result = Twist()
        if now - self.enabled_at < 1.0 and now - self.command_at < 0.5:
            result.linear.x = max(0.0, min(0.25, self.command.linear.x))
            result.angular.z = max(-0.5, min(0.5, self.command.angular.z))
        self.pub.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = Guard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
