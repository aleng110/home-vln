"""在线探索、对象语义建图与动态停靠的任务调度。"""

from datetime import datetime
import copy
import json
import math
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, DurabilityPolicy
from .robot_io import RobotIO, Cancelled, NavigationUnavailable, Replan
from .grid import Grid
from .semantic import Memory, surface_geometry, transform_points, project_observed_point, geometry_reliable
from .cloud import Cloud
from .core import validate_plan, angle_error
from .perception_schema import decode_detection_response
from .coverage import coverage_candidates, camera_view_mask, local_observation_candidates
from .arrival_policy import (
    arrival_geometry,
    geometries_agree,
    arrival_distance,
    surface_distance,
    range_refinement_goal,
)
from .yolo_client import YoloClient
from .yolo_settings import load_settings
from .task_result import build_result
from .object_space import furniture_regions, occupied_by
from .inspection import run_inspection
from .view_policy import (
    ranked_search,
    forward_entry_candidates,
    visual_yaw_delta,
    framing,
    visible_target,
    useful_view,
)


class Explorer(RobotIO):
    def __init__(self):
        self.grid = None
        self.map_received = 0.0
        self.map_version = 0
        self.target = None
        self.semantic_active = False
        self.observation_seq = 0
        self.observation_at = 0.0
        self.force_observe = threading.Event()
        self.cloud_mutex = threading.Lock()
        self.visited = []
        self.blocked = []
        self.task_token = None
        self.initial_scan_done = False
        self.entry_origin_odom = None
        self.entry_heading_odom = None
        self.entry_complete = False
        self.entry_attempts = 0
        self.entry_blocked_rounds = 0
        self.entry_peak_progress = 0.0
        self.search_views = []
        self.visited_views = []
        self.observation_min_stamp = None
        self.interrupted_objects = set()
        self.observation_views = []
        self.local_view_attempts = set()
        self.travel_heading = None
        self.search_moves = 0
        self.last_target_framing = None
        self.step_index = 0
        self.arrival_facing_done = False
        self.pinned_object_id = None
        self.search_look_attempts = []
        self.search_cloud_checks = 0
        self.pursuit_rechecks = set()
        self.last_verify_reason = None
        self.approach_progress_count = 0
        self.range_request = None
        self.range_refinement_count = 0
        self.cloud = None
        self.perception_shutdown = threading.Event()
        super().__init__()
        if self.opt['perception_backend'] != 'yolo':
            raise ValueError('持续语义建图使用本地yolo；云模型用于任务理解与检查')
        settings = load_settings(Path(self.opt['project_root']) / 'config/yolo_scene_classes.yaml')
        self.supported_labels = tuple(settings['labels'])
        self.memory = Memory(
            settings['thresholds'],
            self.opt['semantic_voxel_size'],
            self.opt['semantic_confirm_hits'],
            self.opt['semantic_min_stable_voxels'],
        )
        self.mapping_folder = self.root / ('semantic_session_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.mapping_folder.mkdir(exist_ok=True)
        self.semantic_updated_wall = 0.0
        self.semantic_stamp = None
        self.semantic_error = 'waiting_frames'
        log = self.root / ('yolo_worker_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
        self.local_detector = YoloClient(self.opt['project_root'], log)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.semantic_pub = self.create_publisher(String, '/vln/semantic_map', latched)
        self.semantic_status_pub = self.create_publisher(String, '/vln/semantic_map/status', latched)
        self.create_timer(1.0, self.publish_semantic_status, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.perception_image_pub = self.create_publisher(Image, '/vln/perception/image', 1)
        self.inspection_pub = self.create_publisher(
            Image, '/vln/inspection/image', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.markers = self.create_publisher(MarkerArray, '/vln/markers', 1)
        self.perception_thread = threading.Thread(target=self.perception_loop, daemon=True)
        self.perception_thread.start()
        self.publish(
            'READY',
            '四类家居导航：' + self.opt['perception_backend'] + '感知；等待语言任务',
            perception_backend=self.opt['perception_backend'],
            supported_targets=self.supported_labels,
        )

    def on_map(self, msg):
        try:
            g = Grid.from_msg(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        with self.lock:
            self.grid = g
            self.map_received = time.monotonic()
            self.map_version += 1

    def heartbeat(self):
        # Nav2 执行中也会重规划，初始前行必须持续保持在已验证的通路内。
        if (
            self.motion
            and self.action_kind == 'navigate'
            and self.nav_kind == 'entry_forward'
            and self.entry_origin_odom is not None
            and self.last_xy is not None
        ):
            dx = self.last_xy[0] - self.entry_origin_odom[0]
            dy = self.last_xy[1] - self.entry_origin_odom[1]
            c = math.cos(self.entry_heading_odom)
            s = math.sin(self.entry_heading_odom)
            progress = dx * c + dy * s
            lateral = abs(-dx * s + dy * c)
            self.entry_peak_progress = max(self.entry_peak_progress, progress)
            if lateral > 0.25 or progress < self.entry_peak_progress - 0.12:
                self.fault_reason = '初始前行偏离已验证通路，已停止'
                self.motion = False
        super().heartbeat()

    def on_odom(self, message):
        super().on_odom(message)
        if self.entry_origin_odom is None:
            p = message.pose.pose.position
            q = message.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            if all(math.isfinite(v) for v in (p.x, p.y, yaw)):
                self.entry_origin_odom = [p.x, p.y, 0.0]
                self.entry_heading_odom = yaw

    def entry_goal(self, robot):
        if self.opt['entry_forward_distance'] <= 0:
            self.entry_complete = True
            return None
        origin = self.entry_origin_odom
        heading = self.entry_heading_odom
        progress = float(
            np.dot(np.asarray(self.last_xy) - np.asarray(origin[:2]), [math.cos(heading), math.sin(heading)])
        )
        if progress >= self.opt['entry_min_progress']:
            self.entry_complete = True
            self.record('entry_complete', progress_m=progress, origin_odom=origin)
            return None
        if self.entry_attempts >= 3:
            raise NavigationUnavailable('初始向前移动未取得足够进展，已停止')
        ahead = [origin[0] + math.cos(heading), origin[1] + math.sin(heading), 0.0]
        reference = self.map_geometry(dict(point=origin, observer_positions=[ahead]))
        delta = np.asarray(reference['observer_positions'][0]) - np.asarray(reference['point'])
        map_heading = math.atan2(delta[1], delta[0])
        maximum = self.opt['entry_forward_distance']
        if self.entry_attempts:
            maximum = min(maximum, max(0.65, self.opt['entry_min_progress'] - progress + 0.3))
        proposals = forward_entry_candidates(robot, map_heading, maximum)
        self.publish_objects(proposals)
        for proposal in proposals:
            candidate = self.path_candidate([proposal])
            if candidate:
                self.entry_attempts += 1
                self.entry_blocked_rounds = 0
                self.record('entry_forward_selected', goal=candidate, progress_m=progress, origin_odom=origin)
                return candidate
        self.entry_blocked_rounds += 1
        if self.entry_blocked_rounds >= 3:
            raise NavigationUnavailable('初始正前方没有安全可达的通路，已停留原地')
        self.publish('WAITING_ENTRY', '等待在线地图形成安全的正前方通路')
        return None

    def map_geometry(self, obj):
        """用当前 map←odom 变换重投影观测；仅处理整体刚体修正，未做位姿图重积分。"""
        t = self.tf.lookup_transform('map', 'odom', Time(), timeout=Duration(seconds=0.5)).transform
        q = t.rotation
        p = t.translation
        tr = [p.x, p.y, p.z]
        rot = [q.x, q.y, q.z, q.w]
        result = copy.deepcopy(obj)
        result['point_odom'] = list(obj['point'])
        result['look_at_odom'] = list(obj.get('look_at', obj['point']))
        result['look_at'] = transform_points([result['look_at_odom']], tr, rot)[0].tolist()
        result['point'] = transform_points([obj['point']], tr, rot)[0].tolist()
        support = obj.get('support_points', [])
        if support:
            result['support_points_odom'] = copy.deepcopy(support)
            result['support_points'] = transform_points(support, tr, rot).tolist()
        if 'extent' in result:
            from scipy.spatial.transform import Rotation

            result['extent'] = (
                np.abs(Rotation.from_quat(rot).as_matrix()) @ np.asarray(obj['extent'])
            ).tolist()
        result['observer_positions'] = transform_points(
            obj.get('observer_positions', []) or [[0, 0, 0]], tr, rot
        ).tolist()
        return result

    def budget_ask(self, prompt, image=None, token=None):
        with self.cloud_mutex:
            self.check()
            if token is not None and token != self.task_token:
                raise Cancelled('旧任务请求已失效')
            cloud = self.cloud
            if cloud.calls >= self.opt['max_cloud_calls']:
                raise ValueError('达到云端调用次数预算')
            result = cloud.ask(prompt, image)
            if token is not None and token != self.task_token:
                raise Cancelled('丢弃旧任务模型结果')
            self.check()
            self.record(
                'cloud_usage',
                calls=cloud.calls,
                reported_model=cloud.last_reported_model,
                usage=cloud.last_usage,
            )
            return result

    def plan(self, instruction):
        prompt = (
            '解析家庭机器人任务，仅支持沙发sofa、餐桌dining_table、茶几tea_table、椅子chair。'
            '保留顺序，最多5步，每步到对象附近，可到达后检查。客厅可解释为寻找沙发，餐厅可解释为寻找餐桌，但不能声称已识别房间。'
            '系统明确支持到达后的静态视觉问答：观察、看看、检查、桌上有什么、描述画面。把问题填入inspection_question，不能将它判为拿取任务。'
            '例如“先到沙发，再到茶几，看看桌上有什么”应为sofa和tea_table两个导航步骤，加一个检查问题，unsupported为空。'
            '床、书柜、橱柜、卧室目前不支持，在unsupported明确说明，不得替换成其他家具。'
            '精确空间关系(靠窗、左侧、背后)、拿取、人物、楼层、指定几何路线目前不支持，必须在unsupported明确写出，不能忽略。'
            '这些限制针对用户明确要求执行的导航或操作动作；用户没有要求空间关系时不得以缺少空间关系为理由拒绝。检查问题中的“桌上”不要求执行导航空间关系。'
            '输出{"steps":[{"target":"sofa","relation":"near","description":"找到沙发"}],"unsupported":"",'
            '"inspection_question":"到达最后一个对象后需要回答的视觉问题；没有则为空"}。用户：'
            + json.dumps(instruction, ensure_ascii=False)
        )
        raw = self.budget_ask(prompt, token=self.task_token)
        self.record('language_plan', response=raw)
        return (
            validate_plan(raw, allowed_targets=self.supported_labels),
            str(raw.get('inspection_question', ''))[:400],
        )

    def mapping_check(self):
        if self.perception_shutdown.is_set() or not rclpy.ok():
            raise Cancelled('语义建图关闭')

    def publish_semantic_status(self):
        with self.lock:
            state = dict(
                ready=bool(
                    self.semantic_updated_wall and time.monotonic() - self.semantic_updated_wall < 3.0
                ),
                processed_frames=self.observation_seq,
                map_version=self.memory.version,
                confirmed_objects=sum(bool(o.get('confirmed')) for o in self.memory.objects),
                tentative_objects=sum(o.get('state') == 'tentative' for o in self.memory.objects),
                last_stamp=self.semantic_stamp,
                age_wall=(
                    time.monotonic() - self.semantic_updated_wall if self.semantic_updated_wall else None
                ),
                last_error=self.semantic_error,
                continuous=True,
            )
        self.semantic_status_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    def perception_loop(self):
        # 感知线程独立运行，任务完成或取消后继续维护语义图。
        while not self.perception_shutdown.wait(0.05):
            if time.monotonic() - self.observation_at < self.opt['perception_interval']:
                continue
            with self.lock:
                forced = self.force_observe.is_set()
                minimum = self.observation_min_stamp
                self.force_observe.clear()
            try:
                image, depth, info, stamp, tf = self.capture(
                    after=minimum if forced else self.semantic_stamp, check=self.mapping_check
                )
                q = tf.transform.rotation
                t = tf.transform.translation
                trans = [t.x, t.y, t.z]
                quat = [q.x, q.y, q.z, q.w]
                result = self.local_detector.infer(image, self.mapping_check)
                if self.get_clock().now().nanoseconds / 1e9 - stamp > self.opt['image_max_age']:
                    raise ValueError('语义建图收到过期推理帧')
                masks = result.pop('masks')
                overlay = result.pop('overlay')
                number = self.observation_seq
                # 活动任务保存详细证据；空闲时只保存最新图像和语义图。
                folder = self.run_dir if self.busy and self.run_dir is not None else None
                name = f'image_{number:06d}.jpg' if folder is not None else ''
                if folder is not None:
                    cv2.imwrite(str(folder / name), image)
                    np.savez_compressed(folder / f'depth_{number:06d}.npz', depth=depth)
                    cv2.imwrite(str(folder / f'perception_{number:06d}.jpg'), overlay)
                cv2.imwrite(str(self.mapping_folder / 'latest.jpg'), image)
                display = Image()
                display.header.frame_id = tf.child_frame_id
                display.header.stamp = Time(seconds=stamp).to_msg()
                display.height, display.width = image.shape[:2]
                display.encoding = 'bgr8'
                display.step = display.width * 3
                display.data = overlay.tobytes()
                self.perception_image_pub.publish(display)
                from scipy.spatial.transform import Rotation

                forward = Rotation.from_quat(quat).apply([0, 0, 1])
                camera_yaw = math.atan2(forward[1], forward[0])
                with self.lock:
                    cleared = self.memory.observe_depth(depth, info.k, trans, quat, stamp)
                    assigned = []
                    for i, (detection, mask) in enumerate(zip(result['detections'], masks)):
                        geometry = surface_geometry(detection['bbox'], depth, info.k, trans, quat, mask=mask)
                        obj = self.memory.add(
                            detection['label'],
                            detection['confidence'],
                            geometry,
                            trans,
                            stamp,
                            name,
                            source='yolo',
                            exclude_ids=assigned,
                        )
                        if obj is not None:
                            assigned.append(obj['id'])
                        detection.update(geometry=geometry, object_id=obj['id'] if obj else None)
                        if folder is not None:
                            mask_name = f'mask_{number:06d}_{i:02d}.png'
                            cv2.imwrite(str(folder / mask_name), mask.astype(np.uint8) * 255)
                            detection['mask_image'] = mask_name
                    observation = dict(
                        stamp=stamp,
                        image=name,
                        frame='odom',
                        observer=trans,
                        camera=dict(
                            frame=tf.child_frame_id, k=list(info.k), translation=trans, quaternion=quat
                        ),
                        **result,
                    )
                    self.memory.observations.append(observation)
                    self.memory.observations = self.memory.observations[-100:]
                    last = self.search_views[-1] if self.search_views else None
                    valid = depth[np.isfinite(depth) & (depth > 0.5) & (depth < 5.8)]
                    view_range = (
                        float(np.clip(np.quantile(valid, 0.85) * 0.9, 1.2, 2.8)) if valid.size else 1.2
                    )
                    if (
                        not last
                        or math.dist(trans[:2], last['point'][:2]) > 0.4
                        or angle_error(camera_yaw, last['yaw']) > 0.3
                    ):
                        self.search_views.append(
                            dict(
                                point=trans,
                                yaw=camera_yaw,
                                stamp=stamp,
                                fov=2 * math.atan(image.shape[1] / (2 * info.k[0])),
                                range=view_range,
                            )
                        )
                    self.search_views = self.search_views[-300:]
                    self.observation_seq += 1
                    self.semantic_updated_wall = time.monotonic()
                    self.semantic_stamp = stamp
                    self.semantic_error = None
                    snapshot = copy.deepcopy(self.memory.snapshot(include_history=False))
                temporary = self.mapping_folder / 'semantic_map.tmp'
                temporary.write_text(json.dumps(snapshot, ensure_ascii=False))
                temporary.replace(self.mapping_folder / 'semantic_map.json')
                self.semantic_pub.publish(String(data=json.dumps(snapshot, ensure_ascii=False)))
                self.record('semantic_observation', **observation)
                if cleared:
                    self.record('semantic_free_space_update', removed_voxels=cleared, stamp=stamp)
                self.publish_objects()
            except Cancelled:
                pass
            except Exception as exc:
                self.semantic_error = str(exc)
                self.record('perception_error', reason=str(exc))
                self.get_logger().warning('语义建图暂不可用：' + str(exc))
            finally:
                self.observation_at = time.monotonic()

    def candidate_is_clear(self, candidate, path=None, record_rejection=True):
        if candidate.get('kind') in ('view', 'search_look'):
            return True
        with self.lock:
            regions = furniture_regions(self.memory.objects, self.memory.evidence_ok)
        if not regions:
            return True
        try:
            t = self.tf.lookup_transform('odom', 'map', Time(), timeout=Duration(seconds=0.5)).transform
        except Exception as exc:
            raise NavigationUnavailable('家具占用区域坐标TF未就绪') from exc
        q = t.rotation
        p = t.translation
        translation = [p.x, p.y, p.z]
        quaternion = [q.x, q.y, q.z, q.w]
        goal_odom = transform_points([[candidate['x'], candidate['y'], 0.0]], translation, quaternion)[:, :2]
        blocked = occupied_by(regions, goal_odom, self.opt['furniture_clearance'])
        if blocked is None and path:
            points_odom = transform_points([[x, y, 0.0] for x, y in path], translation, quaternion)[:, :2]
            blocked = occupied_by(regions, points_odom, self.opt['furniture_clearance'])
        if blocked is not None:
            if record_rejection:
                self.record(
                    'furniture_space_rejected', goal=candidate, stage='path' if path else 'goal', **blocked
                )
            return False
        return True

    def publish_objects(self, candidates=()):
        candidates = [c for c in candidates if self.candidate_is_clear(c, record_rejection=False)]
        markers = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        with self.lock:
            objects = copy.deepcopy(self.memory.objects)
        for i, obj in enumerate(objects):
            if obj.get('state') == 'cleared':
                continue
            try:
                o = self.map_geometry(obj)
            except Exception:
                continue
            m = Marker()
            m.header.frame_id = 'map'
            m.ns = 'objects'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = o['point'][0]
            m.pose.position.y = o['point'][1]
            m.pose.position.z = o['point'][2]
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.g = 1.0
            m.color.a = 0.9
            markers.markers.append(m)
            cloud = copy.deepcopy(m)
            cloud.ns = 'semantic_voxels'
            cloud.id = i
            cloud.type = Marker.CUBE_LIST
            cloud.pose.position.x = cloud.pose.position.y = cloud.pose.position.z = 0.0
            cloud.scale.x = cloud.scale.y = cloud.scale.z = self.opt['semantic_voxel_size']
            cloud.color.a = 0.65 if obj.get('confirmed') else 0.18
            cloud.points = [
                Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in o.get('support_points', [])
            ]
            markers.markers.append(cloud)
            text = copy.deepcopy(m)
            text.ns = 'names'
            text.type = Marker.TEXT_VIEW_FACING
            text.text = obj['id'] + ' ' + obj.get('state', 'hypothesis')
            text.pose.position.z += 0.3
            text.scale.z = 0.18
            markers.markers.append(text)
        for i, v in enumerate(candidates):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns = 'generated_goals'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = self.goal_pose(v).pose
            m.scale.x = 0.35
            m.scale.y = 0.08
            m.scale.z = 0.08
            m.color.b = 1.0
            m.color.r = 0.3
            m.color.a = 0.85
            markers.markers.append(m)
        self.markers.publish(markers)

    def get_target(self):
        with self.lock:
            now = self.get_clock().now().nanoseconds / 1e9
            pinned = self.memory.get_by_id(self.pinned_object_id) if self.pinned_object_id else None
            # 单帧弱线索可让位于可靠候选；已有可信深度或语义确认时保持目标实例。
            if (
                pinned
                and not pinned.get('semantic_confirmation')
                and not any(
                    self.memory.evidence_ok(e, pinned['label']) and geometry_reliable(e)
                    for e in pinned['history']
                )
            ):
                replacement = self.memory.target(self.target, now=now)
                if (
                    replacement
                    and replacement['id'] != pinned['id']
                    and self.memory.quality(replacement, now)['reliable']
                ):
                    self.record(
                        'weak_target_promoted',
                        previous=pinned['id'],
                        object_id=replacement['id'],
                        target=self.target,
                    )
                    self.pinned_object_id = replacement['id']
                    pinned = replacement
            obj = (
                copy.deepcopy(
                    pinned
                    if pinned and pinned.get('deferred') and not pinned.get('rejected')
                    else self.memory.target(self.target, now=now, preferred_id=self.pinned_object_id)
                )
                if self.target
                else None
            )
            quality = self.memory.quality(obj, self.get_clock().now().nanoseconds / 1e9) if obj else None
            if obj and obj.get('deferred'):
                quality = dict(
                    quality, reliable=False, reasons=quality['reasons'] + ['当前视角暂未确认，保留局部目标']
                )
            selected = obj['id'] if obj else None
            if selected != self.pinned_object_id:
                self.record(
                    'target_commitment',
                    previous=self.pinned_object_id,
                    object_id=selected,
                    target=self.target,
                )
                self.pinned_object_id = selected if obj is None or obj.get('confirmed', True) else None
        return (self.map_geometry(obj), quality) if obj else (None, None)

    def arrival_yaw(self, goal):
        reference = goal.get('look_at_odom')
        if reference is None:
            return goal['yaw']
        point = self.map_geometry(dict(point=reference, observer_positions=[]))['point']
        x, y, _ = self.pose()
        return math.atan2(point[1] - y, point[0] - x)

    def departure_observe(self, goal):
        if self.semantic_active:
            # 平移前处理新画面并更新语义记忆。
            self.fresh_observation()
            with self.lock:
                stamp = self.memory.observations[-1]['stamp']
            self.record('departure_observation', stamp=stamp, goal=goal, observed=True)
        else:
            super().departure_observe(goal)

    def finish_navigation_view(self, goal):
        if goal.get('kind') in ('approach_progress', 'entry_forward'):
            return
        if goal.get('kind') == 'inspection':
            self.turn_to(self.arrival_yaw(goal), 'inspection_surface_face', tolerance=0.12)
            return
        if goal.get('kind') not in ('dock', 'view', 'observe'):
            return super().finish_navigation_view(goal)
        desired = self.arrival_yaw(goal)
        if angle_error(self.pose()[2], desired) <= self.opt['view_accept_angle']:
            self.record('arrival_view_accepted', reason='bearing_within_useful_view', goal=goal)
            return
        # 当前画面已能辨认目标时接受现有朝向，避免在同一停靠位置反复转向。
        if self.local_detector is not None:
            frame = self.local_frame()
            target = visible_target(frame[5]['detections'], self.target, None)
            if target is not None:
                image, _, info, _, tf = frame[:5]
                q = tf.transform.rotation
                delta = visual_yaw_delta(target['bbox'], info.k, image.shape[:2], [q.x, q.y, q.z, q.w])
                if useful_view(
                    target['bbox'], delta, self.opt['view_accept_angle'], self.opt['view_limit_angle']
                ):
                    self.record('arrival_view_accepted', reason='fresh_visual_view', goal=goal)
                    return
        if self.arrival_facing_done:
            position = tuple(getattr(self, 'last_xy', None) or self.pose()[:2])
            previous = getattr(self, 'arrival_facing_position', None)
            if previous is None or math.dist(position, previous) < 0.25:
                self.record('arrival_facing_limited', goal=goal, step_index=self.step_index)
                return
        self.arrival_facing_done = True
        self.arrival_facing_position = tuple(getattr(self, 'last_xy', None) or self.pose()[:2])
        self.arrival_episode = getattr(self, 'arrival_episode', 0) + 1
        self.record(
            'arrival_facing_adjustment',
            goal=goal,
            step_index=self.step_index,
            target=self.target,
            arrival_episode=self.arrival_episode,
            position_odom=self.arrival_facing_position,
        )
        self.turn_to(desired, 'arrival_target_face', tolerance=self.opt['view_accept_angle'])

    def should_interrupt(self):
        if not self.semantic_active or self.nav_kind not in ('frontier', 'coverage'):
            return False
        obj, quality = self.get_target()
        if obj is None:
            return False
        if obj['id'] in self.interrupted_objects:
            return False
        current_goal = getattr(self, 'current_navigation_goal', None)
        if current_goal is not None:
            robot = self.pose()
            distance = math.dist(robot[:2], obj['point'][:2])
            remaining = math.dist((current_goal['x'], current_goal['y']), obj['point'][:2])
            # 若当前安全路线已在接近门后的目标，就继续这段路线。
            if distance > 2.0 and remaining < distance - 0.2:
                return False
        last = obj['history'][-1]
        interrupt = (
            self.memory.evidence_ok(last, obj['label'])
            and last['stamp'] >= self.nav_started_stamp
            and self.observation_seq > self.nav_observation_start
        )
        if interrupt:
            self.interrupted_objects.add(obj['id'])
        return interrupt

    def mapped_search_views(self):
        from scipy.spatial.transform import Rotation

        t = self.tf.lookup_transform('map', 'odom', Time(), timeout=Duration(seconds=0.5)).transform
        q = t.rotation
        p = t.translation
        rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        yaw = math.atan2(rotation.as_matrix()[1, 0], rotation.as_matrix()[0, 0])
        with self.lock:
            views = copy.deepcopy(self.search_views)
        return [
            dict(
                xy=(rotation.apply(v['point']) + np.array([p.x, p.y, p.z]))[:2].tolist(),
                yaw=v['yaw'] + yaw,
                fov=v['fov'],
                range=v['range'],
            )
            for v in views
        ]

    def fresh_observation(self, timeout=None):
        timeout = (15.0 if self.local_detector is not None else 60.0) if timeout is None else timeout
        before = self.observation_seq
        minimum_stamp = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.observation_min_stamp = minimum_stamp
            self.force_observe.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            if self.observation_seq > before:
                with self.lock:
                    stamp = self.memory.observations[-1]['stamp']
                if stamp >= minimum_stamp:
                    return
                self.force_observe.set()
            time.sleep(0.15)
        raise TimeoutError('等待新语义观测超时')

    def decide(self, explore_only=False):
        self.check()
        robot = self.pose()
        with self.lock:
            grid = self.grid
        if grid is None or time.monotonic() - self.map_received > 8.0:
            raise ValueError('在线地图缺失或过期')
        if getattr(self, 'entry_origin_odom', None) is not None and not self.entry_complete:
            entry = self.entry_goal(robot)
            if not self.entry_complete:
                return entry, None, None
        obj, quality = (None, None) if explore_only else self.get_target()
        if obj and 'confirmed' in obj and not obj['confirmed']:
            self.publish('CONFIRMING_MAP', '目标刚被发现，等待空间重合的后续观测')
            self.fresh_observation()
            obj, quality = self.get_target()
            if obj and not obj.get('confirmed', False):
                obj = None
        if obj:
            distance = surface_distance(obj, robot)
            if not math.isfinite(distance):
                distance = math.dist(robot[:2], obj['point'][:2])
            request = getattr(self, 'range_request', None)
            if request and request['object_id'] == obj['id']:
                self.range_request = None
                target_point = self.map_geometry(dict(point=request['point_odom'], observer_positions=[]))[
                    'point'
                ]
                goal = dict(
                    x=target_point[0],
                    y=target_point[1],
                    yaw=robot[2],
                    score=0.0,
                    kind='dock',
                    object_id=obj['id'],
                    look_at_odom=obj['look_at_odom'],
                    range_refinement=True,
                )
                if grid.safe((goal['x'], goal['y'])):
                    candidate = self.path_candidate([goal])
                    if candidate and candidate.get('path_length', 0.0) <= 3 * request.get('step', 0.6):
                        self.range_refinement_count = getattr(self, 'range_refinement_count', 0) + 1
                        self.publish('APPROACHING', '距离仍偏远，沿已检查路线再靠近一步')
                        self.record(
                            'range_refinement',
                            goal=candidate,
                            request=request,
                            attempt=self.range_refinement_count,
                        )
                        return candidate, obj, quality
                self.record('range_refinement_unavailable', request=request)
            if obj.get('deferred'):
                if obj['id'] in self.pursuit_rechecks:
                    raise RuntimeError('目标候选局部复核后仍不能确认，已停止，未改为全屋漫游')
                self.pursuit_rechecks.add(obj['id'])
                self.local_view_attempts.add(obj['id'])
                obj['local_view'] = True
                return (
                    dict(
                        x=robot[0],
                        y=robot[1],
                        yaw=robot[2],
                        score=0.0,
                        kind='view',
                        object_id=obj['id'],
                        look_at_odom=obj['look_at_odom'],
                    ),
                    obj,
                    quality,
                )
            if 0.6 < distance < self.opt['near_distance'] and obj['id'] not in self.local_view_attempts:
                self.local_view_attempts.add(obj['id'])
                self.pursuit_rechecks.add(obj['id'])
                obj['local_view'] = True
                goal = dict(
                    x=robot[0],
                    y=robot[1],
                    yaw=robot[2],
                    score=0.0,
                    kind='view',
                    object_id=obj['id'],
                    look_at_odom=obj['look_at_odom'],
                )
                self.record('local_view_first', object_id=obj['id'], distance=distance)
                return goal, obj, quality
            # 历史位置用于引导，是否到达由当前画面和双帧深度决定。
            mode = 'dock'
            candidates = grid.docking(
                obj, robot, mode, attempted=self.observation_views, standoff=self.opt['docking_standoff']
            )
            candidates = [
                v for v in candidates if not any(math.dist((v['x'], v['y']), b) < 0.5 for b in self.blocked)
            ]
            self.publish_objects(candidates)
            candidate = self.path_candidate(candidates)
            if candidate:
                return candidate, obj, quality
            progress = grid.approach_progress(obj, robot, self.visited, self.blocked)
            candidate = self.path_candidate(progress) if progress else None
            if candidate:
                self.approach_progress_count = getattr(self, 'approach_progress_count', 0) + 1
                self.record(
                    'target_approach_progress',
                    object_id=obj['id'],
                    goal=candidate,
                    distance_before=distance,
                    attempt=self.approach_progress_count,
                )
                return candidate, obj, quality
            # 目标已锁定时先原地复核；无安全接近路线则停止。
            self.record('no_target_view', object_id=obj['id'], reason='目标已锁定，先就地核实而非全局漫游')
            if obj['id'] not in self.pursuit_rechecks:
                self.pursuit_rechecks.add(obj['id'])
                obj['local_view'] = True
                return (
                    dict(
                        x=robot[0],
                        y=robot[1],
                        yaw=robot[2],
                        score=0.0,
                        kind='view',
                        object_id=obj['id'],
                        look_at_odom=obj['look_at_odom'],
                    ),
                    obj,
                    quality,
                )
            raise RuntimeError('已发现目标，但局部核实后仍无安全可达的接近位置，已停止，未切换到其他区域漫游')
        if not explore_only:
            local_count = sum(math.dist(robot[:2], v[:2]) < 0.65 for v in self.search_look_attempts)
            if local_count < self.opt['search_looks_per_location']:
                looks = local_observation_candidates(
                    grid, robot, self.mapped_search_views(), self.search_look_attempts
                )
                if looks:
                    self.publish_objects(looks)
                    self.record('local_search_before_travel', goal=looks[0], used=local_count)
                    return looks[0], None, None
        frontiers = grid.frontiers(robot, self.visited, self.blocked)
        coverage = []
        if not explore_only:
            coverage, covered = coverage_candidates(
                grid, robot, self.mapped_search_views(), self.visited_views, self.blocked
            )
            np.savez_compressed(
                self.run_dir / 'search_state.npz',
                data=grid.data,
                covered=covered,
                resolution=grid.resolution,
                origin=grid.origin,
            )
            for candidate in frontiers:
                mask = camera_view_mask(grid, (candidate['x'], candidate['y']), candidate['yaw'])
                gain = float(np.sum(mask & ~covered & grid.free)) * grid.resolution**2
                candidate['visual_gain'] = gain
                candidate['score'] -= min(2.0, math.sqrt(gain))
        startup = self.search_moves == 0
        heading = self.travel_heading if self.travel_heading is not None else robot[2]
        ranked = ranked_search(frontiers + coverage, robot, heading, startup)
        if startup:
            forward = [c for c in ranked if c['forward_preferred']]
            other = [c for c in ranked if not c['forward_preferred']]
            candidates = forward[:6] + other[:4]
        else:
            candidates = ranked[:10]
        self.record(
            'search_candidates',
            frontiers=len(frontiers),
            visual_views=len(coverage),
            visited=len(self.visited),
            goals=candidates[:10],
        )
        self.publish_objects(candidates)
        feasible = []
        # 使用 Nav2 验证完整路径，欧氏距离仅用于候选初筛。
        for v in candidates[:10]:
            c = self.path_candidate([v])
            if c:
                feasible.append(c)
            else:
                self.blocked.append((v['x'], v['y']))
        if not feasible:
            return None, obj, quality
        if startup and any(c['forward_preferred'] for c in feasible):
            feasible = [c for c in feasible if c['forward_preferred']]
        self.record(
            'search_heading_policy',
            heading=heading,
            startup=startup,
            feasible=[dict(x=c['x'], y=c['y'], forward=c['forward_preferred']) for c in feasible],
        )
        feasible.sort(key=lambda v: v['path_score'])
        return feasible[0], obj, quality

    def wait_map(self):
        deadline = time.monotonic() + 90.0
        while self.grid is None and time.monotonic() < deadline:
            self.check()
            time.sleep(0.2)
        if self.grid is None:
            raise RuntimeError('SLAM没有生成在线地图')
        self.wait_for_navigation()
        self.pose()
        self.wait_for_obstacle_cloud()

    def verify(self, obj):
        self.last_verify_reason = None
        self.set_motion(False)
        self.publish('VERIFYING', '新画面、深度和目标类别复核', target=self.target, object_id=obj['id'])
        return self.verify_local_target(obj)

    def local_frame(self, after=None):
        floor = (
            max(x for x in (self.last_motion_stamp, after) if x is not None)
            if any(x is not None for x in (self.last_motion_stamp, after))
            else None
        )
        frame = self.capture(after=floor)
        result = self.local_detector.infer(frame[0], self.check)
        return (*frame, result)

    def reference_pixel(self, original, frame):
        image, depth, info, stamp, tf = frame[:5]
        q = tf.transform.rotation
        t = tf.transform.translation
        return project_observed_point(
            original.get('look_at', original['point']),
            info.k,
            [t.x, t.y, t.z],
            [q.x, q.y, q.z, q.w],
            image.shape[:2],
        )

    def repair_target_view(self, planned, original):
        # 先检查当前画面；本地漏检时由 VLM 复核。
        frame = self.local_frame()
        target = visible_target(frame[5]['detections'], self.target, self.reference_pixel(original, frame))
        self.record(
            'local_view_sample',
            object_id=planned['id'],
            found=target is not None,
            stamp=frame[3],
            robot_pose=list(self.pose()),
        )
        return frame

    def verify_local_target(self, planned):
        original = copy.deepcopy(planned)
        original['point'] = original.get('point_odom', original['point'])
        original['look_at'] = original.get('look_at_odom', original.get('look_at', original['point']))
        original['support_points'] = original.get('support_points_odom', original.get('support_points', []))
        frame = self.repair_target_view(planned, original)
        image, depth, info, stamp, tf, local = frame
        q = tf.transform.rotation
        t = tf.transform.translation
        translation = [t.x, t.y, t.z]
        quaternion = [q.x, q.y, q.z, q.w]
        projected = self.reference_pixel(original, frame)
        number = self.observation_seq
        image_name = f'confirmation_{number:03d}_0.jpg'
        cv2.imwrite(str(self.run_dir / image_name), image)
        detection = visible_target(local['detections'], planned['label'], projected)
        threshold = self.memory.yolo_thresholds.get(planned['label'], 0.8)
        if detection is not None and detection['confidence'] >= threshold:
            box = detection['bbox']
            source = 'yolo'
            decision = dict(
                confirmed=True,
                observed_label=detection['label'],
                confidence=detection['confidence'],
                source='yolo',
                bbox_format='normalized',
                bbox=box,
                reason='本地分割；另需实例关联和双帧深度确认',
            )
        else:
            source = 'cloud'
            threshold = 0.8
            prompt = (
                '只根据当前图像核对' + planned['label'] + '。tea_table是茶几/coffee table，不要求摆茶具；'
                'sofa是沙发，dining_table是餐桌，chair是椅子。候选标签可能错误，不可根据任务猜测。'
                '历史支持投影约为' + json.dumps(projected) + '，只用于关联实例，不是类别真值。'
                '到附近只需类别可辨，允许家具部分裁剪。输出JSON：confirmed布尔、observed_label实际类别或unknown、'
                'confidence分数、bbox_format="norm1000"、bbox=[左,上,右,下]、reason证据；坐标0到1000。'
            )
            decision = self.budget_ask(prompt, image, self.task_token)
            semantic_ok = (
                decision.get('confirmed') is True and decision.get('observed_label') == planned['label']
            )
            if not semantic_ok:
                self.last_verify_reason = 'class_unconfirmed'
                self.record(
                    'target_semantic_confirmation',
                    object_id=planned['id'],
                    image=image_name,
                    decision=decision,
                    source=source,
                    success=False,
                )
                self.publish('TARGET_UNCONFIRMED', '当前画面未确认目标；不继续盲目原地旋转')
                return False
            decoded = decode_detection_response(
                dict(
                    bbox_format=decision.get('bbox_format'),
                    detections=[
                        dict(
                            label=decision['observed_label'],
                            confidence=decision.get('confidence'),
                            bbox=decision.get('bbox'),
                        )
                    ],
                ),
                image.shape[1],
                image.shape[0],
            )
            detection = decoded['detections'][0]
            box = detection['bbox']
        (self.run_dir / f'confirmation_{number:03d}_0.json').write_text(
            json.dumps(decision, ensure_ascii=False, indent=2)
        )
        self.record(
            'target_semantic_confirmation',
            object_id=planned['id'],
            image=image_name,
            decision=decision,
            source=source,
            success=True,
        )
        references = [projected] if projected is not None else []
        support = original.get('support_points_odom', original.get('support_points', []))
        for point in support[:: max(1, len(support) // 32)]:
            value = project_observed_point(point, info.k, translation, quaternion, image.shape[:2])
            if value is not None:
                references.append(value)
        matched = any(
            box[0] - 0.05 <= value[0] <= box[2] + 0.05 and box[1] - 0.05 <= value[1] <= box[3] + 0.05
            for value in references
        )
        local_target = visible_target(local['detections'], planned['label'], projected)
        mask = None
        if local_target is not None:
            a = local_target['bbox']
            overlap = max(0, min(a[2], box[2]) - max(a[0], box[0])) * max(
                0, min(a[3], box[3]) - max(a[1], box[1])
            )
            if overlap > 0.2 * max(1e-6, (a[2] - a[0]) * (a[3] - a[1])):
                mask = local['masks'][local['detections'].index(local_target)]
        # 当前深度独立测距，避免用历史距离把前景茶几误作沙发。
        geometry = arrival_geometry(box, depth, info.k, translation, quaternion, mask=mask)
        second = self.local_frame(after=stamp)
        image2, depth2, info2, stamp2, tf2, local2 = second
        q2 = tf2.transform.rotation
        t2 = tf2.transform.translation
        translation2 = [t2.x, t2.y, t2.z]
        quaternion2 = [q2.x, q2.y, q2.z, q2.w]
        projected2 = self.reference_pixel(original, second)
        target2 = visible_target(local2['detections'], planned['label'], projected2)
        mask2 = local2['masks'][local2['detections'].index(target2)] if target2 is not None else None
        # 两帧使用一致的取点方式，避免实例掩膜和框内区域混用。
        if mask is None:
            mask2 = None
        elif mask2 is None:
            geometry = arrival_geometry(box, depth, info.k, translation, quaternion)
        geometry2 = arrival_geometry(box, depth2, info2.k, translation2, quaternion2, mask=mask2)
        stationary = math.dist(translation, translation2) < 0.035 and abs(
            float(np.dot(quaternion, quaternion2))
        ) > math.cos(0.06 / 2)
        range_image = f'range_check_{number:03d}_0.jpg'
        cv2.imwrite(str(self.run_dir / range_image), image2)
        distance = None
        representative_distance = None
        facing = None
        success = False
        yaw_error = visual_yaw_delta(box, info.k, image.shape[:2], quaternion)
        depth_ok = stationary and stamp2 > stamp and geometries_agree(geometry, geometry2)
        if depth_ok:
            mapped = self.map_geometry({**original, 'point': geometry['point'], 'extent': geometry['extent']})
            mapped2 = self.map_geometry(
                {**original, 'point': geometry2['point'], 'extent': geometry2['extent']}
            )
            x, y, yaw = self.pose()
            representative_distance = max(
                arrival_distance(mapped['point'], (x, y)), arrival_distance(mapped2['point'], (x, y))
            )
            base_odom = getattr(self, 'last_xy', None)
            distance = (
                max(surface_distance(geometry, base_odom), surface_distance(geometry2, base_odom))
                if base_odom is not None
                else math.inf
            )
            facing = useful_view(box, yaw_error, self.opt['view_accept_angle'], self.opt['view_limit_angle'])
            success = bool(
                matched
                and detection['confidence'] >= threshold
                and depth_ok
                and 0.5 < distance <= self.opt['near_distance']
            )
        reason = (
            'confirmed_near'
            if success
            else (
                'too_far'
                if distance is not None and distance > self.opt['near_distance']
                else 'invalid_fresh_depth' if not depth_ok else 'reference_or_view_unconfirmed'
            )
        )
        self.last_verify_reason = reason
        self.last_target_framing = framing(box)
        self.record(
            'verification',
            success=success,
            object_id=planned['id'],
            image=image_name,
            source='arrival_yolo_rgbd' if source == 'yolo' else 'arrival_vlm_rgbd',
            projected_reference=projected,
            matched_reference=matched,
            distance_to_observed_surface=distance,
            distance_to_representative_point=representative_distance,
            distance_metric='verified_surface_xy_q10',
            facing=facing,
            visual_yaw_error=yaw_error,
            depth_ok=depth_ok,
            framing=self.last_target_framing,
            geometry=geometry,
            range_geometry=geometry2,
            independent_range=True,
            range_stamps=[stamp, stamp2],
            range_image=range_image,
            stationary=stationary,
            reason=reason,
        )
        if matched and detection['confidence'] >= threshold and depth_ok:
            with self.lock:
                # 到达测距只更新确认状态，局部近景不能替换整个历史对象。
                updated = self.memory.get_by_id(planned['id'])
                if updated is not None:
                    updated['semantic_confirmation'] = dict(stamp=stamp2, decision=decision, source=source)
                    updated.pop('deferred', None)
                self.pinned_object_id = planned['id']
                if reason == 'too_far':
                    proposal = range_refinement_goal(geometry2, self.last_xy, distance)
                    self.range_request = dict(proposal, object_id=planned['id']) if proposal else None
                elif success:
                    self.range_request = None
                self.record(
                    'target_range_verified', object_id=planned['id'], distance=distance, source=source
                )
        self.save_task_map()
        return success

    def search_visual_fallback(self):
        """本地未发现目标时，按配置预算调用 VLM 复查当前画面。"""
        obj, _ = self.get_target()
        if obj is not None or self.search_cloud_checks >= self.opt['search_cloud_checks']:
            return
        self.search_cloud_checks += 1
        image, depth, info, stamp, tf = self.capture()
        self.publish('SEARCH_VERIFYING', '本地识别尚未发现目标，核对当前搜索画面')
        raw = self.budget_ask(
            '只检查当前图像是否清楚可见' + self.target + '。不可根据任务猜测；没有就返回空数组。'
            'sofa=沙发，tea_table=低矮茶几，dining_table=餐桌，chair=椅子。'
            '输出{"bbox_format":"norm1000","detections":[{"label":"'
            + self.target
            + '","confidence":0.9,"bbox":[100,200,800,900]}],"description":"实际可见证据"}。',
            image,
            self.task_token,
        )
        decoded = decode_detection_response(raw, image.shape[1], image.shape[0])
        q = tf.transform.rotation
        t = tf.transform.translation
        name = f'search_check_{self.observation_seq:03d}.jpg'
        cv2.imwrite(str(self.run_dir / name), image)
        for detection in decoded['detections']:
            if detection['label'] != self.target or detection['confidence'] < 0.8:
                continue
            geometry = arrival_geometry(
                detection['bbox'], depth, info.k, [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
            )
            with self.lock:
                self.memory.add(
                    detection['label'],
                    detection['confidence'],
                    geometry,
                    [t.x, t.y, t.z],
                    stamp,
                    str(self.run_dir / name),
                    source='cloud',
                )
        self.save_task_map()
        self.record('search_visual_fallback', image=name, response=raw)

    def save_task_map(self):
        """保存当前任务的对象语义图快照。"""
        with self.lock:
            snapshot = copy.deepcopy(self.memory.snapshot())
        (self.run_dir / 'semantic_map.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))

    def inspect(self, question):
        return run_inspection(self, question)

    def run(self, instruction):
        self.task_observation_start = self.observation_seq
        self.started = time.monotonic()
        self.path_length = 0.0
        self.task_token = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        token = self.task_token
        self.run_dir = self.root / token
        self.run_dir.mkdir()
        self.events = (self.run_dir / 'events.jsonl').open('w', buffering=1)
        self.visited = []
        self.visited_views = []
        self.blocked = []
        self.semantic_active = False
        self.target = None
        self.inspection_result = None
        completed = []
        with self.lock:
            self.memory.clear_rejections()
        self.cloud = Cloud(self.opt['model'], self.opt['cloud_timeout'])
        self.cloud.max_calls = self.opt['max_cloud_calls']
        summary = dict(
            instruction=instruction,
            run_id=token,
            mode='unknown_map_online_exploration',
            prior_map=False,
            fixed_views=False,
            robot='WPB Home',
            model=self.cloud.model,
            outcome='failed',
            search_policy='continuous_dualmap_v6',
            perception_backend=self.opt['perception_backend'],
            supported_targets=self.supported_labels,
            detector_model=self.local_detector.settings['model'] if self.local_detector else None,
        )
        self.publish(
            'STARTING',
            '开始任务；等待在线地图和导航就绪',
            run_id=token,
            target=None,
            object_id=None,
            goal=None,
            waypoint=None,
            user_message=None,
            answer=None,
            outcome='running',
            completed_targets=[],
        )
        explore_only = instruction.strip() == '__explore__'
        try:
            self.wait_map()
            self.travel_heading = self.pose()[2]
            self.search_moves = 0
            if self.local_detector is not None and not explore_only:
                self.publish('WAITING_PERCEPTION', '等待本地YOLO分割模型就绪')
                ready = self.local_detector.wait_ready(self.check)
                self.record('local_perception_ready', **ready)
            if explore_only:
                steps = [dict(target=None)]
                question = ''
            else:
                steps, question = self.plan(instruction)
                summary.update(planned_steps=steps, inspection_question=question)
                self.target = steps[0]['target']
                self.semantic_active = True
            if not self.initial_scan_done:
                self.publish('INITIAL_OBSERVATION', '先观察当前前方；其他方向随在线探索逐步补充')
                if not explore_only:
                    self.fresh_observation()
                self.initial_scan_done = True
            for step_index, step in enumerate(steps, 1):
                self.target = step['target']
                self.pinned_object_id = None
                self.pursuit_rechecks = set()
                self.approach_progress_count = 0
                self.range_request = None
                self.range_refinement_count = 0
                self.search_look_attempts = []
                self.search_cloud_checks = 0
                self.step_index = step_index
                self.arrival_facing_done = False
                self.arrival_facing_position = None
                self.arrival_episode = 0
                self.interrupted_objects = set()
                self.observation_views = []
                self.local_view_attempts = set()
                self.last_target_framing = None
                self.publish(
                    'TARGET_STEP',
                    '几何探索' if explore_only else '当前目标：' + self.target,
                    target=self.target,
                    step_index=step_index,
                    object_id=None,
                    goal=None,
                    waypoint=None,
                )
                if not explore_only:
                    self.fresh_observation()
                # 起点已观察，记录后避免立即重复同一位置的搜索。
                origin = self.pose()
                self.visited.append(tuple(origin[:2]))
                self.visited_views.append(tuple(origin))
                empty_rounds = 0
                for number in range(self.opt['max_decisions']):
                    self.check()
                    candidate, obj, quality = self.decide(explore_only)
                    if candidate is None:
                        empty_rounds += 1
                        if empty_rounds >= 3:
                            if explore_only and self.visited:
                                break
                            raise RuntimeError('没有可达探索前沿或目标观察位置，尚未确认目标')
                        self.publish('WAITING_MAP', '暂时没有可达候选，等待地图更新')
                        time.sleep(2.0)
                        continue
                    empty_rounds = 0
                    self.record('decision', number=number, goal=candidate, target=self.target)
                    self.nav_observation_start = self.observation_seq
                    self.nav_started_stamp = self.get_clock().now().nanoseconds / 1e9
                    before_navigation = self.pose()
                    try:
                        self.navigate(candidate)
                    except Replan as exc:
                        self.record('navigation_interrupted', reason=str(exc))
                        continue
                    except (Cancelled, NavigationUnavailable):
                        raise
                    except Exception as exc:
                        if candidate['kind'] not in ('view', 'search_look'):
                            self.blocked.append((candidate['x'], candidate['y']))
                        elif candidate['kind'] == 'search_look':
                            self.search_look_attempts.append(
                                (candidate['x'], candidate['y'], candidate['yaw'])
                            )
                        self.record('navigation_failed', reason=str(exc), goal=candidate)
                        continue
                    self.visited.append((candidate['x'], candidate['y']))
                    self.visited_views.append(tuple(self.pose()))
                    if candidate['kind'] in ('frontier', 'coverage', 'entry_forward'):
                        after_navigation = self.pose()
                        if math.dist(before_navigation[:2], after_navigation[:2]) > 0.4:
                            self.travel_heading = math.atan2(
                                after_navigation[1] - before_navigation[1],
                                after_navigation[0] - before_navigation[0],
                            )
                            self.search_moves += 1
                    if candidate['kind'] in ('dock', 'view'):
                        if candidate.get('look_at_odom') is not None:
                            obj['look_at_odom'] = candidate['look_at_odom']
                        if self.verify(obj):
                            break
                        if candidate['kind'] == 'view':
                            self.record('local_view_unconfirmed', object_id=obj['id'])
                            continue
                        if self.last_verify_reason != 'too_far':
                            self.blocked.append((candidate['x'], candidate['y']))
                    if candidate['kind'] == 'search_look':
                        self.search_look_attempts.append(tuple(self.pose()))
                        self.publish('SEARCH_OBSERVATION', '在当前位置补看未观察方向，不先走向远处')
                        self.fresh_observation()
                        self.search_visual_fallback()
                    elif candidate['kind'] == 'observe':
                        # 空检测也记录为已尝试的视角，但不能证明目标不存在。
                        self.observation_views.extend(
                            [(candidate['x'], candidate['y']), tuple(self.pose()[:2])]
                        )
                        self.fresh_observation()
                    elif (
                        candidate['kind'] in ('frontier', 'coverage', 'approach_progress', 'entry_forward')
                        and not explore_only
                    ):
                        self.publish('SEARCH_OBSERVATION', '在自动搜索视角取得新图像，记录实际视觉覆盖')
                        self.fresh_observation()
                    else:
                        # 保留几何探索进展记录。
                        self.record(
                            'exploration_progress', map_version=self.map_version, visited=len(self.visited)
                        )
                else:
                    raise RuntimeError('达到决策次数预算')
                if not explore_only:
                    completed.append(self.target)
            self.semantic_active = False
            answer = self.inspect(question) if question else ''
            summary.update(outcome='succeeded', answer=answer, inspection_answer=answer if question else None)
        except Cancelled:
            summary['outcome'] = 'cancelled'
        except Exception as exc:
            summary['error'] = str(exc)
        finally:
            self.semantic_active = False
            self.set_motion(False)
            summary.update(
                path_length_m=self.path_length,
                wall_seconds=time.monotonic() - self.started,
                cloud_calls=self.cloud.calls,
                observations=self.observation_seq - self.task_observation_start,
                mapping_total_frames=self.observation_seq,
                semantic_map_version=self.memory.version,
                visited_positions=len(self.visited),
                objects=len(self.memory.objects),
            )
            summary.update(
                completed_targets=completed, current_target=self.target, localization_accuracy_measured=False
            )
            if self.inspection_result is not None:
                summary['inspection_complete'] = self.inspection_result.get('complete', False)
                summary['inspection_evidence'] = self.inspection_result.get('image')
                if self.inspection_result.get('complete'):
                    summary['inspection_answer'] = self.inspection_result.get('answer')
            self.save_task_map()
            result = build_result(summary)
            summary.update(user_message=result['user_message'], answer=result['answer'])
            (self.run_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            (self.run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            if self.grid is not None:
                np.savez_compressed(
                    self.run_dir / 'online_map.npz',
                    data=self.grid.data,
                    resolution=self.grid.resolution,
                    origin=self.grid.origin,
                )
            self.result_pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
            self.publish(
                {'succeeded': 'SUCCEEDED', 'failed': 'FAILED', 'cancelled': 'CANCELLED'}[summary['outcome']],
                result['user_message'],
                user_message=result['user_message'],
                answer=result['answer'],
                outcome=summary['outcome'],
                completed_targets=completed,
                error=summary.get('error'),
            )
            self.record('task_result', result=result)
            with self.log_lock:
                self.events.close()
                self.events = None
            self.busy = False
            self.started = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Explorer()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        node.semantic_active = False
        node.perception_shutdown.set()
        node.set_motion(False)
        node.perception_thread.join(timeout=2.0)
        if node.local_detector is not None:
            node.local_detector.close()
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
