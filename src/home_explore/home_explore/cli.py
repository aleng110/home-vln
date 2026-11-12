import argparse
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ['status', 'result', 'cancel', 'explore']:
        sub.add_parser(cmd)
    run = sub.add_parser('run')
    run.add_argument('instruction')
    args = p.parse_args()
    rclpy.init(args=[])
    node = Node('home_explore_cli')
    status = {}
    code = 1

    def receive(msg):
        status.update(json.loads(msg.data))

    node.create_subscription(
        String, '/vln/status', receive, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    )
    try:
        if args.command == 'cancel':
            client = node.create_client(Trigger, '/vln/cancel')
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('任务节点未运行')
            f = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, f, timeout_sec=5.0)
            if not f.done():
                raise TimeoutError('取消服务超时')
            print(f.result().message)
            code = 0
        elif args.command == 'result':
            result = {}
            node.create_subscription(
                String,
                '/vln/result',
                lambda m: result.update(json.loads(m.data)),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
            end = time.monotonic() + 8.0
            while not result and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            if not result:
                raise RuntimeError('尚无任务结果；请先执行任务')
            print(result['user_message'])
            code = 0
        else:
            end = time.monotonic() + 8.0
            while not status and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            if not status:
                raise RuntimeError('未收到状态；检查是否启动，以及是否source同一env.sh')
            if args.command == 'status':
                print(json.dumps(status, ensure_ascii=False, indent=2))
                code = 0
            else:
                if status['state'] not in ['READY', 'SUCCEEDED', 'FAILED', 'CANCELLED']:
                    raise RuntimeError('任务进行中，请先取消并等待终态')
                previous = status.get('run_id')
                pub = node.create_publisher(String, '/vln/instruction', 10)
                end = time.monotonic() + 5.0
                while not pub.get_subscription_count() and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.1)
                if not pub.get_subscription_count():
                    raise RuntimeError('无任务订阅者')
                pub.publish(String(data='__explore__' if args.command == 'explore' else args.instruction))
                end = time.monotonic() + 1900.0
                accept_end = time.monotonic() + 10.0
                last = None
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.2)
                    if status.get('run_id') != previous:
                        pair = (status['state'], status['message'])
                        if pair != last:
                            print('[' + pair[0] + '] ' + pair[1], flush=True)
                            last = pair
                        if pair[0] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                            print('任务回复：' + status.get('user_message', pair[1]), flush=True)
                            if pair[0] == 'FAILED' and status.get('error'):
                                print('诊断原因：' + str(status['error']), flush=True)
                            code = 0 if pair[0] == 'SUCCEEDED' else 1
                            break
                    elif time.monotonic() > accept_end:
                        raise RuntimeError('任务未被接受')
                else:
                    raise TimeoutError('客户端超时，请检查status或cancel')
    except KeyboardInterrupt:
        print('客户端退出；使用 scripts/vln.sh cancel 停止任务')
        code = 130
    except Exception as exc:
        print(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)
