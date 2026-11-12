"""完整表面检查：依据当前画面验收，必要时补充取景。"""

import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from rclpy.duration import Duration
from .semantic import project_observed_point
from .view_policy import visual_yaw_delta
from .arrival_policy import surface_distance


class InspectionIncomplete(RuntimeError):
    pass


def _box(value):
    try:
        a = np.asarray(value, dtype=float) / 1000.0
        if a.shape != (4,) or not np.isfinite(a).all():
            return None
        if not (0 <= a[0] < a[2] <= 1 and 0 <= a[1] < a[3] <= 1):
            return None
        return a.tolist()
    except (TypeError, ValueError):
        return None


def assess_inspection(raw, label):
    result = dict(complete=False, reason='检查区域未完整可见', region_bbox=None, answer='')
    if not isinstance(raw, dict):
        return result
    region = _box(raw.get('region_bbox'))
    target = _box(raw.get('target_bbox'))
    result.update(
        region_bbox=region,
        target_bbox=target,
        occluded=raw.get('occluded'),
        reason=str(raw.get('reason', '检查区域不完整')),
    )
    if (
        raw.get('observed_label') != label
        or raw.get('target_in_frame') is not True
        or raw.get('full_region_visible') is not True
        or raw.get('occluded') is not False
        or raw.get('bbox_format') != 'norm1000'
        or region is None
        or target is None
    ):
        return result
    try:
        confidence = float(raw.get('confidence', 0))
        polygon = np.asarray(raw.get('region_polygon'), dtype=float) / 1000.0
    except (TypeError, ValueError):
        return result
    if (
        not math.isfinite(confidence)
        or confidence < 0.8
        or polygon.ndim != 2
        or polygon.shape[1:] != (2,)
        or len(polygon) < 4
    ):
        return result
    if not np.isfinite(polygon).all() or (polygon < 0.025).any() or (polygon > 0.975).any():
        return result
    # 检查的是指定桌面或座面，不能只把桌腿、靠背等部分拍全。
    if min(region[:2]) < 0.025 or max(region[2:]) > 0.975:
        return result
    if min(target[:2]) < 0.015 or max(target[2:]) > 0.985:
        result['reason'] = '家具整体取景仍被图像边界裁断'
        return result
    if any(region[i] < target[i] - 0.03 for i in (0, 1)) or any(region[i] > target[i] + 0.03 for i in (2, 3)):
        result['reason'] = '检查表面区域与目标外框不一致'
        return result
    if (
        (polygon[:, 0] < region[0] - 0.03).any()
        or (polygon[:, 0] > region[2] + 0.03).any()
        or (polygon[:, 1] < region[1] - 0.03).any()
        or (polygon[:, 1] > region[3] + 0.03).any()
    ):
        return result
    area = (
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
        )
        / 2
    )
    if area < 0.015:
        return result
    answer = str(raw.get('answer', '')).strip()
    if not answer or any(
        word in answer
        for word in ('看不清', '无法判断', '无法确认', '不能确定', '不确定', '未入镜', '不可见')
    ):
        return result
    result.update(
        complete=True,
        answer=answer,
        region_polygon=polygon.tolist(),
        reason='指定表面边界完整、无遮挡且分辨率足够',
    )
    return result


def inspection_prompt(label, question, region):
    return (
        '检查当前实际相机图像，目标是' + label + '，任务：' + question + '。'
        '先确认家具整体外轮廓在画面内，再检查整个' + region + '是否可见、是否有物品。'
        'target_in_frame只指整体取景未被画面边缘裁断，不要求看到家具背面或天然被桌板挡住的桌腿。'
        'full_region_visible要求整个指定检查表面可见，不能将局部空白当作整面为空。'
        '扶手、靠背挡住座面也算遮挡；座面是水平可坐区域，不包括靠背立面、坐垫下方或家具接缝内部。'
        '投影阴影和深色纹理不算遮挡，待报告的桌上物品本身不是外部遮挡。'
        '只根据图像给出坐标，不根据任务或历史位置猜测；坐标必须从实际图像测得。'
        '坐标采用norm1000，左上原点；region_polygon沿真实表面透视轮廓，不能直接复制外框角点。'
        '只有整体取景和完整表面都合格才填写answer，使用中文说明实际物品；否则answer为空，reason说明缺少哪一边或哪片区域。'
        '只输出JSON：observed_label、confidence、bbox_format="norm1000"、target_bbox=[左,上,右,下]、'
        'region_bbox、region_polygon=[[x,y],...]、target_in_frame布尔、full_region_visible布尔、occluded布尔、answer、reason。'
    )


def inspection_candidates(agent, obj, frame, attempted):
    grid = agent.grid
    robot = agent.pose()
    image, _, info, _, tf = frame[:5]
    support = np.asarray(obj.get('support_points', []), dtype=float)
    raw_support = np.asarray(obj.get('support_points_odom', []), dtype=float)
    if support.ndim != 2 or support.shape[1:] != (3,) or len(support) < 20:
        return []
    centre = (support.min(axis=0) + support.max(axis=0)) * 0.5
    raw_centre = (
        ((raw_support.min(axis=0) + raw_support.max(axis=0)) * 0.5).tolist()
        if raw_support.shape == support.shape
        else obj['look_at_odom']
    )
    current = agent.tf.lookup_transform('map', 'odom', Time(), timeout=Duration(seconds=0.5)).transform
    q = current.rotation
    t = current.translation
    map_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
    camera_rotation = map_rotation * Rotation.from_quat(
        [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
    )
    camera_position = map_rotation.apply(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
    ) + [t.x, t.y, t.z]
    relative = camera_position - np.array([robot[0], robot[1], 0.0])
    values = grid.docking(obj, robot, 'inspect')
    out = []
    height, width = image.shape[:2]
    for proposal in values:
        if any(math.dist((proposal['x'], proposal['y']), p[:2]) < 0.45 for p in attempted):
            continue
        yaw = math.atan2(centre[1] - proposal['y'], centre[0] - proposal['x'])
        delta = Rotation.from_euler('z', yaw - robot[2])
        rotation = delta * camera_rotation
        position = np.array([proposal['x'], proposal['y'], 0.0]) + delta.apply(relative)
        cam = rotation.inv().apply(support - position)
        if np.mean(cam[:, 2] > 0.2) < 0.95:
            continue
        z = np.maximum(cam[:, 2], 0.05)
        u = (info.k[0] * cam[:, 0] / z + info.k[2]) / width
        v = (info.k[4] * cam[:, 1] / z + info.k[5]) / height
        fraction = float(np.mean((u > 0.12) & (u < 0.88) & (v > 0.08) & (v < 0.92) & (cam[:, 2] > 0.2)))
        near = surface_distance(obj, (proposal['x'], proposal['y']))
        old_bearing = math.atan2(robot[1] - centre[1], robot[0] - centre[0])
        new_bearing = math.atan2(proposal['y'] - centre[1], proposal['x'] - centre[0])
        novelty = abs(math.atan2(math.sin(new_bearing - old_bearing), math.cos(new_bearing - old_bearing)))
        score = math.dist(robot[:2], (proposal['x'], proposal['y'])) + 6 * (1 - fraction) - 0.5 * novelty
        out.append(
            dict(
                proposal,
                yaw=yaw,
                look_at_odom=raw_centre,
                kind='inspection',
                score=score,
                predicted_fit=fraction,
                predicted_surface_distance=near,
            )
        )
    return sorted(out, key=lambda c: c['score'])


def run_inspection(agent, question, max_views=None):
    obj, _ = agent.get_target()
    if obj is None:
        raise InspectionIncomplete('检查未完成：目标记忆不可用')
    if agent.last_xy is None:
        raise InspectionIncomplete('检查未完成：缺少有效里程计')
    import time

    attempted = [tuple(agent.pose())]
    tried_orientations = []
    assessment = None
    deadline = time.monotonic() + getattr(agent, 'opt', {}).get('inspection_timeout', 120.0)
    limit = max_views if max_views is not None else getattr(agent, 'opt', {}).get('max_decisions', 60)
    agent.inspection_result = dict(complete=False, question=question, attempts=[])
    for index in range(limit):
        if time.monotonic() > deadline:
            break
        agent.check()
        agent.set_motion(False)
        agent.publish('INSPECTION_VERIFYING', '确认整个指定检查表面完整入镜且无遮挡')
        frame = agent.local_frame() if agent.local_detector is not None else agent.capture()
        image, depth, info, stamp, tf = frame[:5]
        tried_orientations.append(tuple(agent.pose()))
        name = f'inspection_{index:02d}.jpg'
        cv2.imwrite(str(agent.run_dir / name), image)
        region = '桌面' if agent.target in ('tea_table', 'dining_table') else '座面'
        prompt = inspection_prompt(agent.target, question, region)
        raw = agent.budget_ask(prompt, image, agent.task_token)
        assessment = assess_inspection(raw, agent.target)
        # 当前画面优先验收；历史几何用于关联目标和生成补充视点。
        projected = []
        q = tf.transform.rotation
        t = tf.transform.translation
        for point in obj.get('support_points_odom', [])[::5]:
            uv = project_observed_point(point, info.k, [t.x, t.y, t.z], [q.x, q.y, q.z, q.w], image.shape[:2])
            if uv is not None:
                projected.append(uv)
        b = assessment.get('target_bbox')
        if assessment['complete'] and (
            not b or not any(b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3] for p in projected)
        ):
            assessment.update(complete=False, reason='完整画面未能与已确认目标的三维支持点关联')
        record = dict(image=name, stamp=stamp, raw=raw, assessment=assessment)
        agent.inspection_result['attempts'].append(record)
        (agent.run_dir / f'inspection_{index:02d}.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
        agent.record('inspection_assessment', **record)
        if assessment['complete']:
            agent.inspection_result.update(
                complete=True, answer=assessment['answer'], image=name, stamp=stamp
            )
            cv2.imwrite(str(agent.run_dir / 'inspection.jpg'), image)
            (agent.run_dir / 'inspection.json').write_text(
                json.dumps(agent.inspection_result, ensure_ascii=False, indent=2)
            )
            if getattr(agent, 'inspection_pub', None) is not None:
                from sensor_msgs.msg import Image

                message = Image()
                message.header.stamp = Time(seconds=stamp).to_msg()
                message.header.frame_id = tf.child_frame_id
                message.height, message.width = image.shape[:2]
                message.encoding = 'bgr8'
                message.step = message.width * 3
                message.data = image.tobytes()
                agent.inspection_pub.publish(message)
            agent.publish('INSPECTION_COMPLETE', '指定检查区域完整可见，检查已完成')
            return assessment['answer']
        if index == limit - 1:
            break
        region_box = assessment.get('target_bbox') or assessment.get('region_bbox')
        if region_box is not None and assessment.get('occluded') is not True:
            q = tf.transform.rotation
            delta = visual_yaw_delta(region_box, info.k, image.shape[:2], [q.x, q.y, q.z, q.w])
            pose = agent.pose()
            desired = pose[2] + delta
            tried = any(
                math.dist(pose[:2], v[:2]) < 0.2
                and abs(math.atan2(math.sin(desired - v[2]), math.cos(desired - v[2]))) < 0.1
                for v in tried_orientations
            )
            if abs(delta) > 0.05 and not tried:
                agent.publish('INSPECTION_ALIGNING', '当前图像取景不足，原地转向后重新检查')
                agent.turn_to(desired, 'inspection_region', tolerance=0.06)
                continue
        latest, _ = agent.get_target()
        if latest is not None:
            obj = latest
        choices = inspection_candidates(agent, obj, frame, attempted)
        agent.publish_objects(choices)
        candidate = agent.path_candidate(choices)
        if candidate is None:
            break
        attempted.append((candidate['x'], candidate['y'], candidate['yaw']))
        agent.publish('INSPECTION_VIEWPOINT', '当前位置看不全检查表面，前往经过路径检查的新观察位')
        agent.navigate(candidate)
    (agent.run_dir / 'inspection.json').write_text(
        json.dumps(agent.inspection_result, ensure_ascii=False, indent=2)
    )
    raise InspectionIncomplete(
        '检查未完成：可达新视点或检查预算已用尽，指定表面仍未完整清晰可见；未给出无物品结论'
    )
