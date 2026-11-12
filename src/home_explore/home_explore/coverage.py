"""基于已处理画面的二维覆盖估计。历史位姿保存在 odom，再投到当前地图。

覆盖仅表示可能进入过视野，不能证明家具已被识别。"""

import math
import numpy as np


def camera_view_mask(grid, xy, yaw, fov=math.radians(70), distance=2.8):
    angles = yaw + np.linspace(-fov / 2, fov / 2, max(9, int(math.degrees(fov) / 0.7) + 1))
    ranges = np.arange(0.1, distance + grid.resolution, grid.resolution * 0.8)
    x = xy[0] + np.cos(angles[:, None]) * ranges
    y = xy[1] + np.sin(angles[:, None]) * ranges
    rows = np.floor((y - grid.origin[1]) / grid.resolution).astype(int)
    cols = np.floor((x - grid.origin[0]) / grid.resolution).astype(int)
    inside = (rows >= 0) & (cols >= 0) & (rows < grid.data.shape[0]) & (cols < grid.data.shape[1])
    rr = np.clip(rows, 0, grid.data.shape[0] - 1)
    cc = np.clip(cols, 0, grid.data.shape[1] - 1)
    free = inside & grid.free[rr, cc]
    # 首个占用格可见，其后的格子被遮挡。
    clear_before = np.concatenate(
        (np.ones((len(angles), 1), bool), np.logical_and.accumulate(free, axis=1)[:, :-1]), axis=1
    )
    visible = inside & clear_before & (grid.data[rr, cc] >= 0)
    mask = np.zeros(grid.data.shape, bool)
    mask[rr[visible], cc[visible]] = True
    return mask


def observed_mask(grid, views):
    mask = np.zeros(grid.data.shape, bool)
    for view in views:
        mask |= camera_view_mask(
            grid, view['xy'], view['yaw'], view.get('fov', math.radians(70)), view.get('range', 2.8)
        )
    return mask


def local_observation_candidates(
    grid, robot, views=(), attempted=(), limit=3, min_gain=0.35, min_yaw_delta=math.radians(35), distance=2.8
):
    """在当前位置提出补看方向；依据实际历史视角和尝试记录去重，次数由调用方控制。"""
    if limit <= 0 or not grid.safe(robot[:2]):
        return []
    covered = observed_mask(grid, views)
    xy = tuple(robot[:2])
    result = []

    def angle_error(a, b):
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def tried(pose, yaw):
        if isinstance(pose, dict):
            position = pose.get('xy', (pose.get('x'), pose.get('y')))
            old_yaw = pose['yaw']
        else:
            position = pose[:2]
            old_yaw = pose[2]
        return math.dist(xy, position) < 0.75 and angle_error(yaw, old_yaw) < min_yaw_delta

    # 相对于当前朝向采样，不预设房间观察方向。
    for offset in np.arange(-math.pi, math.pi, math.pi / 4):
        if abs(offset) < min_yaw_delta:
            continue
        yaw = math.atan2(math.sin(robot[2] + offset), math.cos(robot[2] + offset))
        if any(tried(v, yaw) for v in views) or any(tried(v, yaw) for v in attempted):
            continue
        visible = camera_view_mask(grid, xy, yaw, distance=distance)
        unseen = visible & ~covered
        visible_count = int(np.sum(visible))
        new_count = int(np.sum(unseen))
        # 过滤几乎重复的视野，防止栅格边缘误差造成虚假的新增收益。
        if not visible_count or new_count < 0.3 * visible_count:
            continue
        free_area = float(np.sum(unseen & grid.free)) * grid.resolution**2
        boundary_length = float(np.sum(unseen & (grid.data >= 65))) * grid.resolution
        gain = 0.3 * free_area + 0.8 * boundary_length
        if gain < min_gain:
            continue
        turn = angle_error(yaw, robot[2])
        result.append(
            dict(
                x=float(xy[0]),
                y=float(xy[1]),
                yaw=float(yaw),
                kind='search_look',
                score=-2.0 * math.sqrt(gain) + 0.15 * turn,
                visual_gain=gain,
                information_cells=new_count,
                unseen_free_area=free_area,
                new_visual_fraction=new_count / visible_count,
                turn_angle=turn,
            )
        )
    result.sort(key=lambda goal: goal['score'])
    # 去除近似朝向；每次拍完新画面后重新计算候选。
    distinct = []
    for goal in result:
        if not any(angle_error(goal['yaw'], old['yaw']) < math.radians(44) for old in distinct):
            distinct.append(goal)
        if len(distinct) >= limit:
            break
    return distinct


def coverage_candidates(grid, robot, views=(), visited=(), blocked=(), limit=16):
    """从已知安全区域生成新增视觉覆盖较多的观察位。"""
    covered = observed_mask(grid, views)
    stride = max(1, round(0.9 / grid.resolution))
    rr, cc = np.where(grid.clearance >= 0.33)
    pick = (rr % stride == 0) & (cc % stride == 0)
    positions = [grid.xy(rc) for rc in zip(rr[pick], cc[pick])]
    positions.append(tuple(robot[:2]))
    result = []
    for xy in positions:
        if any(math.dist(xy, b) < 0.5 for b in blocked):
            continue
        # 平移候选避开已访问位置；原地补看由独立候选函数处理。
        if any(math.dist(xy, v[:2]) < 1.0 for v in visited):
            continue
        d = math.dist(xy, robot[:2])
        if d > 14.0:
            continue
        best = None
        for yaw in np.arange(-math.pi, math.pi, math.pi / 4):
            # 按位置和朝向共同去重。
            if any(
                math.dist(xy, v[:2]) < 0.65
                and abs(math.atan2(math.sin(yaw - v[2]), math.cos(yaw - v[2]))) < 0.65
                for v in visited
            ):
                continue
            mask = camera_view_mask(grid, xy, yaw)
            unseen = mask & ~covered
            free_area = float(np.sum(unseen & grid.free)) * grid.resolution**2
            boundary_length = float(np.sum(unseen & (grid.data >= 65))) * grid.resolution
            # 同时奖励新障碍表面，避免大片空地仅凭面积压过室内家具区域。
            gain = 0.3 * free_area + 0.8 * boundary_length
            if gain < 0.35:
                continue
            turn = abs(math.atan2(math.sin(yaw - robot[2]), math.cos(yaw - robot[2])))
            candidate = dict(
                x=xy[0],
                y=xy[1],
                yaw=float(yaw),
                kind='coverage',
                score=0.3 * d - 2.0 * math.sqrt(gain) + 0.1 * turn,
                visual_gain=gain,
                information_cells=int(np.sum(unseen)),
                unseen_free_area=free_area,
            )
            if best is None or candidate['score'] < best['score']:
                best = candidate
        if best:
            result.append(best)
    result.sort(key=lambda v: v['score'])
    distinct = []
    for v in result:
        if not any(math.dist((v['x'], v['y']), (o['x'], o['y'])) < 0.75 for o in distinct):
            distinct.append(v)
    return distinct[:limit], covered
