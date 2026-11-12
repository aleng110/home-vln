"""在线几何地图查询；未知区域不作为安全终点。"""

from dataclasses import dataclass
import math
import cv2
import numpy as np
from scipy.ndimage import label, binary_dilation


@dataclass
class Grid:
    data: np.ndarray
    resolution: float
    origin: tuple

    def __post_init__(self):
        self.free = (self.data >= 0) & (self.data < 25)
        # 地图外部按未知处理，避免边缘产生虚假净空。
        padded = np.pad(self.free.astype(np.uint8), 1)
        self.clearance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * self.resolution

    @classmethod
    def from_msg(cls, msg):
        q = msg.info.origin.orientation
        if abs(q.z) > 0.001 or msg.info.resolution <= 0:
            raise ValueError('Only axis-aligned positive-resolution maps supported')
        return cls(
            np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width),
            msg.info.resolution,
            (msg.info.origin.position.x, msg.info.origin.position.y),
        )

    def cell(self, xy):
        return int(math.floor((xy[1] - self.origin[1]) / self.resolution)), int(
            math.floor((xy[0] - self.origin[0]) / self.resolution)
        )

    def xy(self, rc):
        return (
            self.origin[0] + (rc[1] + 0.5) * self.resolution,
            self.origin[1] + (rc[0] + 0.5) * self.resolution,
        )

    def safe(self, xy, radius=0.30):
        r, c = self.cell(xy)
        return 0 <= r < self.data.shape[0] and 0 <= c < self.data.shape[1] and self.clearance[r, c] >= radius

    def visible(self, start, target, ignore_last=0.25):
        d = math.dist(start[:2], target[:2])
        for f in np.linspace(0, max(0, 1 - ignore_last / max(d, 0.001)), max(2, int(d / self.resolution))):
            r, c = self.cell((start[0] + f * (target[0] - start[0]), start[1] + f * (target[1] - start[1])))
            if not (0 <= r < self.data.shape[0] and 0 <= c < self.data.shape[1]) or not self.free[r, c]:
                return False
        return True

    def frontiers(self, robot, visited=(), blocked=(), limit=12):
        edge = self.free & binary_dilation(self.data < 0)
        regions, n = label(edge, np.ones((3, 3)))
        result = []
        groups = []
        for i in range(1, n + 1):
            cells = np.argwhere(regions == i)
            # 将连通前沿分块，避免环形边界的整体均值落到房间内部。
            buckets = np.floor(cells * self.resolution / 0.9).astype(int)
            for key in np.unique(buckets, axis=0):
                groups.append(cells[np.all(buckets == key, axis=1)])
        for cells in groups:
            if len(cells) < 4:
                continue
            centre = np.mean(cells, axis=0)
            # 从未知边界退回已知自由区选终点。
            rr, cc = np.where(self.clearance >= 0.31)
            near = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2
            valid = np.where(near < (1.4 / self.resolution) ** 2)[0]
            if not len(valid):
                continue
            j = valid[np.argmin(near[valid])]
            xy = self.xy((rr[j], cc[j]))
            if math.dist(xy, robot[:2]) < 0.45 or any(math.dist(xy, b) < 0.6 for b in blocked):
                continue
            # 跳过已到访的相邻位置，避免重复选择同一前沿。
            if any(math.dist(xy, v[:2]) < 0.8 for v in visited):
                continue
            travel = math.dist(xy, robot[:2])
            target = self.xy(centre)
            yaw = math.atan2(target[1] - xy[1], target[0] - xy[0])
            # 工程启发式几何评分，未实现 VLFM 语言价值地图。
            score = travel - 0.025 * min(len(cells), 80)
            result.append(
                dict(x=xy[0], y=xy[1], yaw=yaw, score=score, information_cells=len(cells), kind='frontier')
            )
        result.sort(key=lambda a: a['score'])
        distinct = []
        for v in result:
            if not any(math.dist((v['x'], v['y']), (p['x'], p['y'])) < 0.7 for p in distinct):
                distinct.append(v)
        return distinct[:limit]

    def approach_progress(self, obj, robot, visited=(), blocked=(), limit=12):
        """在已知自由区内生成接近目标的中间点。

        Nav2 负责验证绕门等完整路线；中间点不算到达目标。"""
        if limit <= 0:
            return []
        target = obj['point'][:2]
        origin = robot[:2]
        current = math.dist(origin, target)
        if current <= 0.85:
            return []
        row, column = self.cell(origin)
        if not (0 <= row < self.data.shape[0] and 0 <= column < self.data.shape[1]):
            return []
        bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
        look = obj.get('look_at', obj['point'])
        candidates = []
        positions = []
        for radius in (0.65, 1.0, 1.5, 2.0):
            # 中间点须位于已见表面之前，再检查实际障碍净空。
            if radius > current - 0.3:
                continue
            for offset in (
                0.0,
                math.radians(20),
                -math.radians(20),
                math.radians(40),
                -math.radians(40),
                math.radians(60),
                -math.radians(60),
            ):
                angle = bearing + offset
                q = (origin[0] + radius * math.cos(angle), origin[1] + radius * math.sin(angle))
                positions.append(q)
        # 提出已知安全终点，再让 Nav2 检查可绕行的完整路线。
        stride = max(1, round(0.35 / self.resolution))
        rr, cc = np.where(self.clearance >= 0.31)
        pick = (rr % stride == 0) & (cc % stride == 0)
        positions.extend(self.xy((r, c)) for r, c in zip(rr[pick], cc[pick]))
        for q in positions:
            travel = math.dist(origin, q)
            remaining = math.dist(q, target)
            if not (0.55 <= travel <= 3.0) or remaining >= current - 0.1 or remaining < 0.7:
                continue
            if not self.safe(q, radius=0.31):
                continue
            if any(math.dist(q, p[:2]) < 0.45 for p in visited):
                continue
            if any(math.dist(q, p[:2]) < 0.5 for p in blocked):
                continue
            yaw = math.atan2(look[1] - q[1], look[0] - q[0])
            direction = math.atan2(q[1] - origin[1], q[0] - origin[0])
            offset = abs(math.atan2(math.sin(direction - bearing), math.cos(direction - bearing)))
            score = 0.35 * travel + 1.6 * remaining + 0.15 * offset
            candidates.append(
                dict(
                    x=q[0],
                    y=q[1],
                    yaw=yaw,
                    kind='approach_progress',
                    score=score,
                    object_id=obj.get('id'),
                    look_at_odom=obj.get('look_at_odom', obj.get('point_odom')),
                    remaining_distance=remaining,
                    progress_m=current - remaining,
                    requires_path_validation=True,
                )
            )
        candidates.sort(key=lambda goal: goal['score'])
        distinct = []
        for goal in candidates:
            if not any(math.dist((goal['x'], goal['y']), (old['x'], old['y'])) < 0.35 for old in distinct):
                distinct.append(goal)
            if len(distinct) >= limit:
                break
        return distinct

    def docking(self, obj, robot, mode='dock', attempted=(), standoff=0.9):
        """沿已见表面边界向外采样停靠位，安全性仍需实时地图与 Nav2 验证。"""
        centre = np.asarray(obj['point'], dtype=float)
        points = np.asarray(obj.get('support_points_map', obj.get('support_points', [])), dtype=float)
        raw = np.asarray(obj.get('support_points_odom', []), dtype=float)
        supported = (
            points.ndim == 2 and points.shape[1:] == (3,) and len(points) >= 4 and np.isfinite(points).all()
        )
        aligned = supported and raw.shape == points.shape and np.isfinite(raw).all()
        if not supported:
            extent = np.asarray(obj.get('extent', [0.2, 0.2]), dtype=float)
            if extent.ndim != 1 or len(extent) < 2 or not np.isfinite(extent[:2]).all():
                extent = np.array([0.2, 0.2])
            sx, sy = np.maximum(extent[:2], 0.0) * 0.5
            points = np.array(
                [[centre[0] + dx, centre[1] + dy, centre[2]] for dx in (-sx, sx) for dy in (-sy, sy)]
            )
        default_reference = obj.get('look_at_odom', obj.get('point_odom'))
        default_look = np.asarray(obj.get('look_at', obj['point']), dtype=float)
        gaps = (
            (max(0.55, standoff - 0.15), standoff, standoff + 0.1)
            if mode == 'dock'
            else (1.0, 1.2, 1.6, 2.0, 2.4) if mode == 'inspect' else (1.2, 1.5, 1.8)
        )
        candidates = []
        for angle in np.arange(0, 2 * math.pi, math.pi / 18):
            direction = np.array([math.cos(angle), math.sin(angle)])
            projection = points[:, :2] @ direction
            extreme = np.flatnonzero(projection >= np.quantile(projection, 0.85) - 1e-9)
            median = np.median(points[extreme], axis=0)
            index = int(extreme[np.argmin(np.linalg.norm(points[extreme] - median, axis=1))])
            boundary = points[index]
            # 表面边界决定间隙，已见对象中部决定朝向；参考点保留 odom 坐标。
            reference = default_reference
            look = default_look
            for gap in gaps:
                q = boundary[:2] + gap * direction
                if not self.safe(q) or not self.visible(q, boundary):
                    continue
                travel = math.dist(q, robot[:2])
                if mode == 'observe' and travel < 0.55:
                    continue
                if mode == 'observe' and any(math.dist(q, v[:2]) < 0.45 for v in attempted):
                    continue
                prior = obj.get('observer_positions', [])
                novelty = min([math.dist(q, p[:2]) for p in prior] or [1.0])
                if mode == 'observe' and novelty < 0.35:
                    continue
                yaw = math.atan2(look[1] - q[1], look[0] - q[0])
                turn = abs(math.atan2(math.sin(yaw - robot[2]), math.cos(yaw - robot[2])))
                score = travel + 0.3 * turn
                if mode == 'dock':
                    score += 0.5 * abs(gap - standoff)
                else:
                    score -= 0.6 * min(novelty, 1.5)
                candidates.append(
                    dict(
                        x=float(q[0]),
                        y=float(q[1]),
                        yaw=yaw,
                        score=score,
                        kind=mode,
                        object_id=obj.get('id'),
                        look_at_odom=reference,
                        boundary_point=boundary.tolist(),
                        boundary_gap=float(gap),
                        boundary_source='observed_support' if supported else 'observed_extent_fallback',
                    )
                )
        candidates.sort(key=lambda value: value['score'])
        # 家具占用过滤与候选间距去重由 RobotIO 统一处理。
        return candidates
