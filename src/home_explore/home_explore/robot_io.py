"""ROS 传感器、TF、Nav2 动作及任务运动保护。"""

from __future__ import annotations
import json
import math
import os
import threading
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from lifecycle_msgs.srv import GetState
from lifecycle_msgs.msg import State as LifecycleState
from nav2_msgs.action import NavigateToPose, Spin, ComputePathToPose
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from .core import angle_error
from .image_utils import decode_ros_image
from .view_policy import departure_heading, forward_entry_path


class Cancelled(Exception):
    pass


class NavigationUnavailable(RuntimeError):
    pass


class Replan(Exception):
    pass


class RobotIO(Node):
    def __init__(self):
        super().__init__('home_explore_agent')
        backend = os.environ.get('EXPLORE_PERCEPTION_BACKEND', 'yolo')
        self.declare_parameter('perception_backend', backend)
        backend = self.get_parameter('perception_backend').value
        defaults = dict(
            results_dir='results',
            model=os.environ.get('VLN_MODEL', 'qwen3.8-flash'),
            project_root=os.environ.get('EXPLORE_ROOT', str(Path(__file__).resolve().parents[3])),
            cloud_timeout=40.0,
            task_timeout=1800.0,
            inspection_timeout=120.0,
            navigation_timeout=150.0,
            navigation_startup_timeout=90.0,
            image_max_age=3.0,
            perception_interval=0.5,
            max_cloud_calls=32,
            near_distance=1.25,
            max_decisions=60,
            docking_standoff=0.9,
            furniture_clearance=0.32,
            semantic_voxel_size=0.10,
            semantic_confirm_hits=2,
            semantic_min_stable_voxels=20,
            entry_forward_distance=1.5,
            entry_min_progress=1.1,
            search_looks_per_location=2,
            search_cloud_checks=2,
            view_accept_angle=math.radians(20),
            view_limit_angle=math.radians(25),
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.opt = {k: self.get_parameter(k).value for k in defaults}
        self.opt['perception_backend'] = backend
        self.root = Path(self.opt['results_dir']).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.group = ReentrantCallbackGroup()
        self.worker = None
        self.busy = False
        self.goal_handle = None
        self.navigation_pending = False
        self.motion = False
        self.fault_reason = None
        self.rgb = self.depth = self.info = None
        self.received = {}
        self.obstacle_status = None
        self.obstacle_status_received = 0.0
        self.events = None
        self.run_dir = None
        self.last_xy = None
        self.path_length = 0.0
        self.started = 0.0
        self.state = dict(state='READY', message='等待任务；不加载先验地图和观察点', run_id=None)
        self.nav_kind = None
        self.nav_result_future = None
        self.action_kind = None
        self.last_motion_stamp = None
        self.tf = Buffer(cache_time=Duration(seconds=120.0))
        self.listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose', callback_group=self.group)
        self.spin_action = ActionClient(self, Spin, '/spin', callback_group=self.group)
        self.planner = ActionClient(
            self, ComputePathToPose, '/compute_path_to_pose', callback_group=self.group
        )
        self.navigation_clients = {
            name: self.create_client(GetState, '/' + name + '/get_state', callback_group=self.group)
            for name in [
                'controller_server',
                'smoother_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother',
            ]
        }
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/vln/status', latched)
        self.result_pub = self.create_publisher(String, '/vln/result', latched)
        self.create_subscription(
            String, '/vln/obstacle_cloud/status', self.on_obstacle_status, latched, callback_group=self.group
        )
        self.lease_pub = self.create_publisher(Bool, '/vln/motion_enabled', 10)
        for key, topic, typ, qos in [
            ('rgb', '/camera/image_raw', Image, QoSProfile(depth=1)),
            ('depth', '/camera/depth/image_raw', Image, QoSProfile(depth=1)),
            ('info', '/camera/camera_info', CameraInfo, qos_profile_sensor_data),
        ]:
            self.create_subscription(
                typ, topic, lambda m, k=key: self.sensor(k, m), qos, callback_group=self.group
            )
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched, callback_group=self.group)
        self.create_subscription(String, '/vln/instruction', self.on_instruction, 10)
        self.create_service(Trigger, '/vln/cancel', self.cancel, callback_group=self.group)
        self.create_timer(0.2, self.heartbeat, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def heartbeat(self):
        if self.motion and not self.obstacles_ready():
            self.fault_reason = '避障点云中断或状态过期，已停止运动'
            self.motion = False
            self.record(
                'obstacle_cloud_fault',
                status=self.obstacle_status,
                status_age_wall=time.monotonic() - self.obstacle_status_received,
                clock_now=self.get_clock().now().nanoseconds / 1e9,
            )
        self.lease_pub.publish(Bool(data=self.motion and self.busy and not self.stop_event.is_set()))
        if (self.stop_event.is_set() or self.fault_reason) and self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.status_pub.publish(String(data=json.dumps(self.state, ensure_ascii=False)))

    def on_obstacle_status(self, message):
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                return
        except (ValueError, TypeError):
            return
        self.obstacle_status = value
        self.obstacle_status_received = time.monotonic()

    def obstacles_ready(self):
        state = self.obstacle_status
        if not state or time.monotonic() - self.obstacle_status_received > 2.5:
            return False
        try:
            stamp = float(state['last_cloud_stamp'])
            now = self.get_clock().now().nanoseconds / 1e9
            source_now = float(state.get('ros_time', now))
            # 仿真时钟回调可能滞后，结合生产端时间及节点间允许偏差检查点云新鲜度。
            return (
                state.get('ready') is True
                and state.get('published', 0) >= 2
                and math.isfinite(stamp)
                and math.isfinite(source_now)
                and -0.3 <= source_now - stamp <= 1.5
                and abs(now - source_now) <= 1.5
                and max(now, source_now) - stamp <= 1.5
            )
        except (KeyError, TypeError, ValueError):
            return False

    def obstacle_diagnosis(self):
        state = self.obstacle_status or {}
        reason = {
            'waiting_depth': '尚未收到原始深度图',
            'waiting_camera_info': '尚未收到相机标定',
            'depth_clock_mismatch': '深度与仿真时钟不同步',
            'publishing': '点云状态已过期',
        }.get(state.get('reason'), '尚无有效点云状态')
        return (
            reason
            + '；深度接收='
            + str(state.get('depth_received', 0))
            + '，点云发布='
            + str(state.get('published', 0))
        )

    def wait_for_obstacle_cloud(self, timeout=15.0):
        self.set_motion(False)
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() < deadline:
            self.check()
            if self.obstacles_ready():
                self.record('obstacle_cloud_ready', status=self.obstacle_status)
                return
            reason = self.obstacle_diagnosis()
            if reason != previous:
                self.publish('WAITING_OBSTACLES', '等待避障点云就绪：' + reason)
                previous = reason
            time.sleep(0.1)
        raise NavigationUnavailable('避障点云未就绪：' + self.obstacle_diagnosis() + '；未发送运动请求')

    def goal_pose(self, v):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(v['x'])
        msg.pose.position.y = float(v['y'])
        msg.pose.orientation.z = math.sin(v['yaw'] / 2)
        msg.pose.orientation.w = math.cos(v['yaw'] / 2)
        return msg

    def navigate(self, v):
        self.check()
        self.wait_for_navigation(timeout=10.0)
        self.nav_kind = v.get('kind', 'frontier')
        self.current_navigation_goal = dict(v)
        self.goal_safety_seq = -1
        self.replan_reason = None
        if not self.candidate_is_clear(v):
            raise Replan('目标点被已观测家具占用，重新选点')
        x, y, yaw = self.pose()
        self.publish(
            (
                'ENTERING'
                if self.nav_kind == 'entry_forward'
                else (
                    'SEARCHING_VIEW'
                    if self.nav_kind == 'search_look'
                    else 'SUPPLEMENTARY_OBSERVATION' if self.nav_kind == 'observe' else 'NAVIGATING'
                )
            ),
            (
                '先沿初始朝向前进'
                if self.nav_kind == 'entry_forward'
                else (
                    '目标尚未发现，先在当前位置查看新的方向'
                    if self.nav_kind == 'search_look'
                    else '前往在线生成的位置'
                )
            ),
            goal=v,
        )
        if math.hypot(x - v['x'], y - v['y']) < 0.12:
            self.finish_navigation_view(v)
            return
        self.prepare_departure(v)
        if not self.nav.wait_for_server(timeout_sec=3.0):
            raise NavigationUnavailable('导航action不可用')
        goal = NavigateToPose.Goal()
        goal.pose = self.goal_pose(v)
        self.record('navigation_translation_start', goal=v, pose=list(self.pose()))
        self.execute_action(self.nav, goal, self.opt['navigation_timeout'], action_kind='navigate')
        # 底盘到位与相机朝向分别处理，避免 DWB 为对准视角反复调整。
        self.finish_navigation_view(v)
        self.record('navigation_reached', goal=v, pose=list(self.pose()))

    def arrival_yaw(self, goal):
        return goal['yaw']

    def finish_navigation_view(self, goal):
        self.turn_to(self.arrival_yaw(goal), 'generated_facing', tolerance=self.opt['view_accept_angle'])

    def departure_observe(self, goal):
        frame = self.capture(after=self.last_motion_stamp)
        self.record('departure_observation', stamp=frame[3], goal=goal, observed=True)

    def prepare_departure(self, goal):
        # 出发前重新验证路径，避免使用 SLAM 修正或中断前的旧路线。
        fresh = self.path_candidate([dict(goal, score=0.0)])
        if fresh is None:
            raise ValueError('出发前路径已不可达，重新选择目标')
        path_yaw = fresh['departure_yaw']
        look_yaw = self.arrival_yaw(goal) if goal.get('look_at_odom') is not None else path_yaw
        self.publish('LOOKING_BEFORE_MOVE', '先转向目标或行进方向，用新画面观察后再前进', goal=goal)
        self.turn_to(look_yaw, 'look_before_move', tolerance=0.28)
        self.record('departure_look', goal=goal, pose=list(self.pose()))
        self.departure_observe(goal)
        if self.should_interrupt():
            raise Replan(getattr(self, 'replan_reason', None) or '出发前新画面已发现目标，先重新决策')
        # 目标可能位于墙后，实际出发方向取自安全路径首段。
        self.turn_to(path_yaw, 'path_departure', tolerance=0.28)
        self.record('departure_ready', goal=goal, pose=list(self.pose()), path_heading=path_yaw)

    def candidate_is_clear(self, candidate, path=None, record_rejection=True):
        return True

    def path_candidate(self, candidates):
        if not self.planner.wait_for_server(timeout_sec=5.0):
            raise NavigationUnavailable('规划服务未就绪')
        feasible = []
        # 先过滤家具占用，再对相邻候选去重，保留安全的近侧替代点。
        clear = [c for c in candidates if self.candidate_is_clear(c)]
        candidates = []
        for c in clear:
            if c.get('kind') in ('dock', 'observe', 'inspection') and any(
                math.hypot(c['x'] - p['x'], c['y'] - p['y']) < 0.18 for p in candidates
            ):
                continue
            candidates.append(c)
        limit = 8 if candidates and candidates[0].get('kind') in ('dock', 'observe', 'inspection') else 6
        for candidate in candidates[:limit]:
            self.check()
            if not self.candidate_is_clear(candidate):
                continue
            goal = ComputePathToPose.Goal()
            goal.goal = self.goal_pose(candidate)
            try:
                handle = self.wait_future(self.planner.send_goal_async(goal), 5.0)
            except TimeoutError as exc:
                raise NavigationUnavailable('全局规划服务5秒内未确认请求；请检查Nav2及代价地图更新') from exc
            if not handle.accepted:
                continue
            future = handle.get_result_async()
            try:
                result = self.wait_future(future, 8.0)
            except TimeoutError as exc:
                handle.cancel_goal_async()
                raise NavigationUnavailable(
                    '全局规划服务8秒内未返回路径；请检查代价地图点云/TF更新及启动日志'
                ) from exc
            except Exception:
                handle.cancel_goal_async()
                raise
            points = result.result.path.poses
            if result.status != GoalStatus.STATUS_SUCCEEDED or not points:
                continue
            if (
                math.hypot(
                    points[-1].pose.position.x - candidate['x'], points[-1].pose.position.y - candidate['y']
                )
                > 0.25
            ):
                continue
            length = sum(
                math.hypot(a.pose.position.x - b.pose.position.x, a.pose.position.y - b.pose.position.y)
                for a, b in zip(points, points[1:])
            )
            xy = [(p.pose.position.x, p.pose.position.y) for p in points]
            if not self.candidate_is_clear(candidate, path=xy):
                continue
            if candidate.get('kind') == 'entry_forward' and not forward_entry_path(
                xy, candidate['entry_origin'], candidate['entry_heading']
            ):
                self.record('entry_path_rejected', goal=candidate, reason='side_or_back_detour')
                continue
            score = (
                length + 0.1 * abs(candidate.get('boundary_gap', 0.9) - 0.9)
                if candidate.get('kind') == 'dock'
                else candidate['score'] + length
            )
            feasible.append(
                dict(
                    candidate,
                    path_length=length,
                    path_score=score,
                    departure_yaw=departure_heading(xy, self.pose()),
                )
            )
        return min(feasible, key=lambda c: c['path_score'], default=None)

    def sensor(self, name, msg):
        with self.lock:
            setattr(self, name, msg)
            self.received[name] = time.monotonic()

    def publish(self, state, message, **extra):
        with self.lock:
            self.state.update(state=state, message=message, **extra)
            payload = dict(self.state)
        if rclpy.ok():
            self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.get_logger().info(f'[{state}] {message}')
        self.record('state', **payload)

    def record(self, event, **fields):
        with self.log_lock:
            if self.events is not None:
                self.events.write(
                    json.dumps(
                        {
                            'event': event,
                            'wall_time': time.time(),
                            'sim_time': self.get_clock().now().nanoseconds / 1e9,
                            **fields,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                self.events.flush()

    def on_odom(self, msg):
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        with self.lock:
            active = self.busy
            if active and self.last_xy:
                delta = math.dist(xy, self.last_xy)
                if delta < 1.0:
                    self.path_length += delta
            self.last_xy = xy
        if active:
            self.record('odom', x=xy[0], y=xy[1])

    def set_motion(self, enabled):
        self.motion = enabled
        if rclpy.ok():
            self.lease_pub.publish(Bool(data=enabled and not self.stop_event.is_set()))

    def cancel(self, request, response):
        self.stop_event.set()
        self.set_motion(False)
        if self.goal_handle:
            self.goal_handle.cancel_goal_async()
        response.success = True
        response.message = '已请求取消并关闭运动许可；正在等待中的云请求不会恢复运动'
        self.publish('CANCELLING', response.message)
        return response

    def on_instruction(self, msg):
        self.start(msg.data.strip())

    def start(self, instruction):
        if not instruction or len(instruction) > 2000:
            self.publish('REJECTED', '指令为空或超过 2000 字符')
            return
        with self.lock:
            if self.busy or self.navigation_pending:
                self.get_logger().warning('已有任务或未结束的导航请求，请先取消并等待终态')
                return
            self.stop_event.clear()
            self.fault_reason = None
            self.busy = True
            self.worker = threading.Thread(target=self.run, args=(instruction,), daemon=True)
            self.worker.start()

    def check(self):
        if self.fault_reason:
            raise ValueError(self.fault_reason)
        if self.stop_event.is_set() or not rclpy.ok():
            raise Cancelled('用户取消或节点退出')
        if self.started and time.monotonic() - self.started > self.opt['task_timeout']:
            raise TimeoutError('任务超过总时间预算')

    def wait_future(self, future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            self.check()
            if (
                future is self.nav_result_future
                and self.action_kind == 'navigate'
                and self.should_interrupt()
            ):
                raise Replan(getattr(self, "replan_reason", None) or "新目标证据到达，取消探索并重新决策")
            if time.monotonic() > deadline:
                raise TimeoutError('等待 ROS action 超时')
            time.sleep(0.05)
        if future.exception():
            raise RuntimeError(str(future.exception()))
        return future.result()

    def pose(self):
        deadline = time.monotonic() + 2.0
        while True:
            self.check()
            try:
                t = self.tf.lookup_transform('map', 'base_footprint', Time(), timeout=Duration(seconds=0.3))
                break
            except TransformException as exc:
                if time.monotonic() >= deadline:
                    raise NavigationUnavailable('定位 TF 暂未同步，已停止：' + str(exc)) from exc
                time.sleep(0.05)
        stamp = t.header.stamp.sec + t.header.stamp.nanosec / 1e9
        if self.get_clock().now().nanoseconds / 1e9 - stamp > 3.0:
            raise ValueError('定位 TF 已过期')
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def wait_for_navigation(self, timeout=None):
        """等待导航生命周期激活；所有服务等待使用有上限的墙钟超时。"""
        self.set_motion(False)
        deadline = time.monotonic() + (timeout or self.opt['navigation_startup_timeout'])
        previous = None
        pending = list(self.navigation_clients)
        while time.monotonic() < deadline:
            self.check()
            states, futures = {}, {}
            for name, client in self.navigation_clients.items():
                if client.service_is_ready():
                    futures[name] = client.call_async(GetState.Request())
                else:
                    states[name] = 'service_missing'
            round_deadline = min(deadline, time.monotonic() + 1.0)
            while any(not f.done() for f in futures.values()) and time.monotonic() < round_deadline:
                try:
                    self.check()
                except Exception:
                    for name, future in futures.items():
                        if not future.done():
                            self.navigation_clients[name].remove_pending_request(future)
                            future.cancel()
                    raise
                time.sleep(0.05)
            for name, future in futures.items():
                if not future.done():
                    self.navigation_clients[name].remove_pending_request(future)
                    future.cancel()
                    states[name] = 'response_timeout'
                elif future.exception():
                    states[name] = 'service_error'
                else:
                    state = future.result().current_state
                    states[name] = (
                        'active' if state.id == LifecycleState.PRIMARY_STATE_ACTIVE else state.label
                    )
            pending = [
                name + '=' + states.get(name, 'unknown')
                for name in self.navigation_clients
                if states.get(name) != 'active'
            ]
            if not pending:
                self.record('navigation_ready', nodes=list(self.navigation_clients))
                return
            if pending != previous:
                self.publish('WAITING_NAVIGATION', '等待导航节点激活：' + ', '.join(pending))
                previous = pending
            time.sleep(min(0.3, max(0, deadline - time.monotonic())))
        raise NavigationUnavailable(
            '导航未就绪，已停止任务且关闭运动许可：'
            + ', '.join(pending)
            + '；检查 Nav2 启动日志，并确保所有终端使用 scripts/env.sh'
        )

    def turn_to(self, yaw, waypoint, tolerance=0.28):
        self.check()
        self.wait_for_navigation(timeout=10.0)
        x, y, current_yaw = self.pose()
        delta = math.atan2(math.sin(yaw - current_yaw), math.cos(yaw - current_yaw))
        if abs(delta) <= tolerance:
            self.record('navigation_skipped', waypoint=waypoint, reason='already_at_pose')
            return False
        self.publish('TURNING', '在当前位置转向观察目标', waypoint=waypoint)
        if not self.spin_action.wait_for_server(timeout_sec=5.0):
            raise NavigationUnavailable('Nav2 Spin 服务未就绪')
        goal = Spin.Goal()
        goal.target_yaw = float(delta)
        goal.time_allowance.sec = 25
        self.execute_action(
            self.spin_action, goal, min(120.0, self.opt['navigation_timeout']), action_kind='spin'
        )
        actual = self.pose()
        if angle_error(actual[2], yaw) > max(0.14, tolerance) or math.dist(actual[:2], (x, y)) > 0.35:
            raise ValueError('原地观察转向未达到要求，拒绝使用错误朝向')
        self.record('turn_reached', waypoint=waypoint, x=actual[0], y=actual[1], yaw=actual[2])
        return True

    def capture(self, after=None, check=None):
        validate = check or self.check
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            validate()
            with self.lock:
                rgb, depth, info = self.rgb, self.depth, self.info
                fresh = all(
                    time.monotonic() - self.received.get(k, 0) < self.opt['image_max_age']
                    for k in ('rgb', 'depth', 'info')
                )
            if rgb and depth and info and fresh:
                if (
                    rgb.header.frame_id != depth.header.frame_id
                    or rgb.header.frame_id != info.header.frame_id
                ):
                    raise ValueError('RGB、深度和CameraInfo必须来自同一已对齐光学坐标系')
                if len(info.k) != 9 or not np.isfinite(info.k).all() or info.k[0] <= 0 or info.k[4] <= 0:
                    raise ValueError('相机内参无效')
                stamp = rgb.header.stamp.sec + rgb.header.stamp.nanosec / 1e9
                dstamp = depth.header.stamp.sec + depth.header.stamp.nanosec / 1e9
                if abs(stamp - dstamp) < 0.15 and (after is None or stamp > after + 0.1):
                    try:
                        transform = self.tf.lookup_transform(
                            'odom',
                            rgb.header.frame_id,
                            Time.from_msg(rgb.header.stamp),
                            timeout=Duration(seconds=0.5),
                        )
                        return (decode_ros_image(rgb), decode_ros_image(depth), info, stamp, transform)
                    except TransformException:
                        # 低实时率时图像可能暂时领先里程计 TF，允许有上限的重试。
                        pass
            time.sleep(0.08)
        raise ValueError('没有新鲜且同步的 RGB-D/内参数据；检查相机、TF 和仿真是否暂停')

    def execute_action(self, client, goal, timeout, action_kind=None):
        self.wait_for_obstacle_cloud(timeout=5.0)
        self.action_kind = action_kind
        self.navigation_pending = True
        future = client.send_goal_async(goal)

        def accepted(f):
            try:
                handle = f.result()
                self.goal_handle = handle if handle.accepted else None
                self.navigation_pending = False
                if (self.stop_event.is_set() or self.fault_reason) and handle.accepted:
                    handle.cancel_goal_async()
            except Exception:
                self.navigation_pending = False

        future.add_done_callback(accepted)
        self.set_motion(True)
        try:
            handle = self.wait_future(future, 15.0)
            if not handle.accepted:
                raise NavigationUnavailable('Nav2 拒绝了目标；导航可能尚未激活，已停止而不是重复拍照')
            self.goal_handle = handle
            finished = handle.get_result_async()
            self.nav_result_future = finished
            result = self.wait_future(finished, timeout)
            self.check()
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                raise ValueError('Nav2 action 未成功，status=' + str(result.status))
        except Exception:
            self.set_motion(False)
            active_handle = self.goal_handle or locals().get('handle')
            if active_handle is not None and active_handle.accepted:
                cancel = active_handle.cancel_goal_async()
                # 先确认旧动作取消，再允许发送新目标。
                deadline = time.monotonic() + 3.0
                while not cancel.done() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not cancel.done():
                    self.stop_event.set()
                elif 'finished' in locals():
                    while not finished.done() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if not finished.done():
                        self.stop_event.set()
            if not future.done():
                self.stop_event.set()
            raise
        finally:
            self.set_motion(False)
            self.last_motion_stamp = self.get_clock().now().nanoseconds / 1e9
            self.goal_handle = None
            self.nav_result_future = None
            self.action_kind = None
