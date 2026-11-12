"""用新观测表面独立测距。优先使用实例掩膜；仅有框时提取深度连通区域。

结果只描述可见表面，不能保证恢复完整家具。"""

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from .semantic import surface_geometry


def _bounds(box, shape):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    height, width = shape
    return int(x0 * width), int(y0 * height), min(width, int(x1 * width)), min(height, int(y1 * height))


def geometry_is_consistent(geometry, min_points=40, max_depth_spread=0.12):
    if not geometry:
        return False
    try:
        point = np.asarray(geometry['point'], dtype=float)
        spread = float(geometry['depth_spread'])
        count = int(geometry['valid_points'])
    except (KeyError, ValueError, TypeError):
        return False
    return bool(
        point.shape == (3,)
        and np.isfinite(point).all()
        and count >= min_points
        and math.isfinite(spread)
        and 0 <= spread <= max_depth_spread
    )


def _bbox_depth_region(box, depth, k, translation, quaternion):
    """在收缩的检测框内按相邻深度生长区域；接触且深度相近的物体仍可能混合。"""
    height, width = depth.shape
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    region_box = [x0 + 0.04 * bw, y0 + 0.04 * bh, x1 - 0.04 * bw, y1 - 0.04 * bh]
    bounds = _bounds(region_box, depth.shape)
    if bounds is None or len(k) != 9 or k[0] <= 0 or k[4] <= 0:
        return None, None
    left, top, right, bottom = bounds
    valid = np.zeros(depth.shape, bool)
    z = np.asarray(depth[top:bottom, left:right], np.float32)
    v, u = np.mgrid[top:bottom, left:right]
    rotation = Rotation.from_quat(quaternion).as_matrix()
    world_z = translation[2] + z * (
        rotation[2, 0] * (u - k[2]) / k[0] + rotation[2, 1] * (v - k[5]) / k[4] + rotation[2, 2]
    )
    valid[top:bottom, left:right] = (
        np.isfinite(z) & (z > 0.5) & (z < 5.8) & (world_z > 0.12) & (world_z < 2.2)
    )
    core_box = [x0 + 0.2 * bw, y0 + 0.12 * bh, x0 + 0.8 * bw, y0 + 0.55 * bh]
    l, t, r, b = _bounds(core_box, depth.shape)
    core = np.zeros(depth.shape, bool)
    core[t:b, l:r] = valid[t:b, l:r]
    if np.sum(core) < 30:
        return None, None
    front = float(np.quantile(depth[core], 0.15))
    band = core & (np.abs(depth - front) <= max(0.06, 0.03 * front))
    neighbors = cv2.boxFilter(band.astype(np.uint8), -1, (3, 3), normalize=False)
    eligible = band & (neighbors >= 5)
    rr, cc = np.where(eligible)
    if not len(rr):
        return None, None
    centre_x = (x0 + x1) * 0.5 * width
    centre_y = (y0 + 0.32 * bh) * height
    central = ((cc - centre_x) / max(1.0, bw * width)) ** 2 + ((rr - centre_y) / max(1.0, bh * height)) ** 2
    index = int(np.argmin(central))
    seed = (int(cc[index]), int(rr[index]))
    seed_depth = float(depth[seed[1], seed[0]])
    difference = max(0.05, min(0.15, 0.03 * seed_depth))
    # 相邻深度渐变可连通，桌子、墙面等明显深度跳变处停止生长。
    fill_mask = np.ones((height + 2, width + 2), np.uint8)
    fill_mask[1:-1, 1:-1] = np.where(valid, 0, 255).astype(np.uint8)
    values = np.nan_to_num(np.asarray(depth, np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    cv2.floodFill(
        values,
        fill_mask,
        seed,
        0.0,
        loDiff=difference,
        upDiff=difference,
        flags=4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8),
    )
    grown = fill_mask[1:-1, 1:-1] == 2
    metadata = dict(
        seed_pixel=list(seed),
        seed_depth=seed_depth,
        adjacency_depth_limit=difference,
        grown_pixel_count=int(np.sum(grown)),
        seed_core=core_box,
    )
    return grown, metadata


def arrival_geometry(box, depth, k, translation, quaternion, mask=None, min_points=40, max_depth_spread=0.12):
    """返回当前帧独立几何；提供的掩膜无效时直接失败。

    调用方负责同帧类别关联。尺寸错配抛异常，深度稀疏返回 None；不使用历史目标距离筛点。"""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError('到达测距需要二维对齐深度图')
    bounds = _bounds(box, depth.shape)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    region = np.zeros(depth.shape, np.uint8)
    if mask is not None:
        if np.shape(mask) != depth.shape:
            raise ValueError('到达掩膜和对齐深度尺寸不一致')
        region[top:bottom, left:right] = np.asarray(mask, dtype=bool)[top:bottom, left:right]
        method = 'mask_arrival_front_depth_band'
        region_box = list(box)
        region_metadata = None
    else:
        region, region_metadata = _bbox_depth_region(box, depth, k, translation, quaternion)
        if region is None:
            return None
        method = 'bbox_depth_region_arrival_front_depth_band'
        region_box = list(box)
    if not region.any():
        return None
    geometry = surface_geometry(box, depth, k, translation, quaternion, mask=region)
    if not geometry_is_consistent(geometry, min_points, max_depth_spread):
        return None
    geometry['method'] = method
    geometry['arrival_region'] = region_box
    if region_metadata is not None:
        geometry['depth_region'] = region_metadata
    geometry['independent_range'] = True
    return geometry


def geometries_agree(first, second, max_xy_delta=0.2, max_depth_spread=0.12, min_points=40):
    """比较两帧表面是否一致；调用方另行检查时间戳递增及拍摄时机器人静止。"""
    if not all(geometry_is_consistent(g, min_points, max_depth_spread) for g in (first, second)):
        return False
    return math.dist(first['point'][:2], second['point'][:2]) <= max_xy_delta


def arrival_distance(point, robot):
    try:
        target = np.asarray(point, dtype=float)
        pose = np.asarray(robot, dtype=float)
        if target.ndim != 1 or pose.ndim != 1 or min(len(target), len(pose)) < 2:
            return math.inf
        if not np.isfinite(target[:2]).all() or not np.isfinite(pose[:2]).all():
            return math.inf
        return float(np.linalg.norm(target[:2] - pose[:2]))
    except (TypeError, ValueError):
        return math.inf


def surface_distance(geometry, robot, min_samples=20, quantile=0.1):
    """计算已验证表面点到机器人 XY 的距离分位数；两者必须使用同一坐标系。"""
    if not geometry or not 0 < quantile < 0.5:
        return math.inf
    try:
        points = np.asarray(geometry.get('support_points', geometry.get('points', [])), dtype=float)
        pose = np.asarray(robot, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] < 2
            or pose.ndim != 1
            or len(pose) < 2
            or not np.isfinite(pose[:2]).all()
        ):
            return math.inf
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < min_samples:
            return math.inf
        distances = np.linalg.norm(points[:, :2] - pose[:2], axis=1)
        return float(np.quantile(distances, quantile))
    except (TypeError, ValueError):
        return math.inf


def within_arrival_range(point, robot, max_distance=1.25, min_distance=0.5):
    return min_distance < arrival_distance(point, robot) <= max_distance


def range_refinement_goal(
    geometry, base_odom, range_m, desired=0.95, min_step=0.35, max_step=0.6, max_range=2.0
):
    try:
        distance = float(range_m)
        pose = np.asarray(base_odom, dtype=float)
        points = np.asarray(geometry.get('support_points', geometry.get('points', [])), dtype=float)
        parameters = (distance, desired, min_step, max_step, max_range)
        if not all(math.isfinite(x) for x in parameters):
            return None
        if not (1.25 < distance <= max_range and 0.22 < min_step <= max_step and desired > 0.65):
            return None
        if pose.ndim != 1 or len(pose) < 2 or not np.isfinite(pose[:2]).all():
            return None
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 20:
            return None
        distances = np.linalg.norm(points[:, :2] - pose[:2], axis=1)
        robust_distance = float(np.quantile(distances, 0.1))
        index = int(np.argmin(np.abs(distances - robust_distance)))
        support = points[index]
        support_distance = float(distances[index])
        if support_distance <= 0.65:
            return None
        # 兼顾双帧保守测距和当前实际支持点，确保前移后仍保留安全间隙。
        step = min(max_step, max(min_step, distance - desired), distance - 0.65, support_distance - 0.65)
        if step < min_step:
            return None
        direction = (support[:2] - pose[:2]) / support_distance
        goal = pose[:2] + direction * step
        return dict(
            point_odom=[float(goal[0]), float(goal[1]), 0.0],
            look_at_odom=support.tolist(),
            step=float(step),
            observed_range=distance,
            support_distance=support_distance,
            predicted_surface_gap=support_distance - step,
            source='verified_surface_q10',
        )
    except (AttributeError, TypeError, ValueError):
        return None
