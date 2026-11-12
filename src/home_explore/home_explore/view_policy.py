"""搜索方向、路径朝向与相机取景策略。"""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def useful_view(box, delta, accept_angle=math.radians(20), limit_angle=math.radians(25)):
    """附近任务允许偏心或局部裁剪；完整检查另行验收。"""
    centre = (box[0] + box[2]) * 0.5
    return bool(
        abs(delta) <= accept_angle
        or (abs(delta) <= limit_angle and not framing(box)['horizontal_clipped'] and 0.15 <= centre <= 0.85)
    )


def departure_heading(points, robot, lookahead=0.65):
    """从已验证路径提取初始前进方向。"""
    if not points:
        return robot[2]
    distance = 0.0
    previous = robot[:2]
    for point in points:
        distance += math.dist(previous, point)
        previous = point
        if distance >= lookahead and math.dist(robot[:2], point) > 0.15:
            return math.atan2(point[1] - robot[1], point[0] - robot[0])
    for point in reversed(points):
        if math.dist(robot[:2], point) > 0.15:
            return math.atan2(point[1] - robot[1], point[0] - robot[0])
    return robot[2]


def forward_entry_candidates(robot, heading, maximum=1.5):
    """沿初始相对朝向生成候选；是否可走必须交给 Nav2 完整路径验证。"""
    if not all(math.isfinite(x) for x in (*robot[:2], heading, maximum)) or maximum < 0.6:
        return []
    distances = sorted(
        {min(maximum, d) for d in (1.5, 1.2, 0.9, 0.65) if min(maximum, d) >= 0.6}, reverse=True
    )
    return [
        dict(
            x=robot[0] + d * math.cos(heading),
            y=robot[1] + d * math.sin(heading),
            yaw=heading,
            score=0.0,
            kind='entry_forward',
            entry_origin=list(robot[:2]),
            entry_heading=heading,
            entry_distance=d,
            requires_path_validation=True,
        )
        for d in distances
    ]


def forward_entry_path(points, origin, heading, max_lateral=0.2, max_heading=math.radians(15)):
    """检查整条初始路径，避免终点在前方而路线绕向侧面或后方。"""
    if not points:
        return False
    a = np.asarray(points, dtype=float)
    o = np.asarray(origin[:2], dtype=float)
    if (
        o.shape != (2,)
        or a.ndim != 2
        or a.shape[1] != 2
        or not np.isfinite(a).all()
        or not np.isfinite(o).all()
        or not math.isfinite(heading)
    ):
        return False
    forward = np.array([math.cos(heading), math.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    relative = a - o
    progress = relative @ forward
    lateral = np.abs(relative @ side)
    if progress.min() < -0.1 or lateral.max() > max_lateral or progress[-1] < 0.4:
        return False
    if np.maximum(0.0, -np.diff(progress)).sum() > 0.08:
        return False
    yaw = departure_heading(points, (*origin[:2], heading))
    return abs(math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))) <= max_heading


def ranked_search(candidates, robot, heading, startup=False):
    output = []
    for candidate in candidates:
        value = dict(candidate)
        bearing = math.atan2(value['y'] - robot[1], value['x'] - robot[0])
        error = abs(math.atan2(math.sin(bearing - heading), math.cos(bearing - heading)))
        penalty = (1.8 if startup else 0.6) * (1 - math.cos(error)) + 0.35 * math.sin(error)
        value.update(
            score=value['score'] + penalty,
            search_heading_error=error,
            forward_preferred=error <= math.radians(40),
        )
        output.append(value)
    return sorted(output, key=lambda c: (startup and not c['forward_preferred'], c['score']))


def visual_yaw_delta(box, k, shape, quaternion):
    height, width = shape
    u = (box[0] + box[2]) * 0.5 * width
    v = (box[1] + box[3]) * 0.5 * height
    rotation = Rotation.from_quat(quaternion)
    ray = rotation.apply([(u - k[2]) / k[0], (v - k[5]) / k[4], 1.0])
    forward = rotation.apply([0, 0, 1.0])
    delta = math.atan2(ray[1], ray[0]) - math.atan2(forward[1], forward[0])
    return math.atan2(math.sin(delta), math.cos(delta))


def framing(box):
    return dict(
        clipped=box[0] < 0.025 or box[2] > 0.975 or box[1] < 0.025 or box[3] > 0.975,
        horizontal_clipped=box[0] < 0.025 or box[2] > 0.975,
        width=box[2] - box[0],
        height=box[3] - box[1],
    )


def visible_target(detections, label, reference):
    choices = [d for d in detections if d['label'] == label]
    if not choices:
        return None

    def score(d):
        b = d['bbox']
        centre = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
        match = (
            reference is not None
            and b[0] - 0.1 <= reference[0] <= b[2] + 0.1
            and b[1] - 0.1 <= reference[1] <= b[3] + 0.1
        )
        distance = math.dist(centre, reference) if reference is not None else 0.0
        return (match, -distance, d['confidence'])

    return max(choices, key=score)
