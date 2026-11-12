"""RGB-D 部分表面几何与对象记忆；模型分数不是校准概率。"""

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from .voxel_fusion import VoxelFusion


def transform_points(points, translation, quaternion):
    return Rotation.from_quat(quaternion).apply(points) + np.array(translation)


def project_observed_point(point, k, translation, quaternion, shape):
    """将 odom 中的历史表面点投影到当前相机图像。"""
    camera = Rotation.from_quat(quaternion).inv().apply(np.asarray(point) - np.asarray(translation))
    if not np.isfinite(camera).all() or camera[2] <= 0.1:
        return None
    height, width = shape
    return [
        float((k[0] * camera[0] / camera[2] + k[2]) / width),
        float((k[4] * camera[1] / camera[2] + k[5]) / height),
    ]


def surface_geometry(bbox, depth, k, translation, quaternion, mask=None, reference_point=None):
    h, w = depth.shape
    if mask is not None:
        if np.shape(mask) != (h, w):
            raise ValueError('实例掩膜和对齐深度尺寸不匹配')
        # 腐蚀掩膜边缘，减少背景深度混入。
        mask = cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    x0, y0, x1, y1 = bbox
    l, r = max(0, int(x0 * w)), min(w, int(x1 * w))
    t, b = max(0, int(y0 * h)), min(h, int(y1 * h))
    if r - l < 4 or b - t < 4 or k[0] <= 0 or k[4] <= 0:
        return None
    v, u = np.mgrid[t:b:3, l:r:3]
    z = depth[t:b:3, l:r:3]
    valid = np.isfinite(z) & (z > 0.5) & (z < 5.8)
    if mask is not None:
        valid &= mask[t:b:3, l:r:3]
    if valid.sum() < 30:
        return None
    cam = np.column_stack(
        ((u[valid] - k[2]) * z[valid] / k[0], (v[valid] - k[5]) * z[valid] / k[4], z[valid])
    )
    odom = transform_points(cam, translation, quaternion)
    furniture = (odom[:, 2] > 0.12) & (odom[:, 2] < 2.2)
    odom, cam = odom[furniture], cam[furniture]
    if len(odom) < 30:
        return None
    # 从实例内筛选前表面深度带，输出仍是部分可见表面。
    look_at = np.median(odom, axis=0) if mask is not None else None
    if reference_point is None:
        front = np.quantile(cam[:, 2], 0.15)
        pick = np.abs(cam[:, 2] - front) < max(0.18, front * 0.06)
    else:
        reference = (
            Rotation.from_quat(quaternion).inv().apply(np.asarray(reference_point) - np.asarray(translation))
        )
        if reference[2] <= 0.1:
            return None
        front = float(reference[2])
        pick = np.abs(cam[:, 2] - front) < max(0.2, front * 0.08)
    points = odom[pick]
    if len(points) < 25:
        return None
    median = np.median(points, axis=0)
    mad = float(np.median(np.linalg.norm(points - median, axis=1)))
    depths = cam[pick, 2]
    depth_median = float(np.median(depths))
    depth_spread = float(np.median(np.abs(depths - depth_median)))
    # 限制支持点数量，用于后续重投影与显示。
    points = points[:: max(1, len(points) // 150)]
    support = odom[np.linspace(0, len(odom) - 1, min(200, len(odom)), dtype=int)]
    return dict(
        point=median.tolist(),
        points=points.tolist(),
        support_points=support.tolist(),
        valid_points=int(pick.sum()),
        spread=mad,
        extent=(np.quantile(points, 0.9, axis=0) - np.quantile(points, 0.1, axis=0)).tolist(),
        look_at=(look_at if look_at is not None else median).tolist(),
        depth_spread=depth_spread,
        reference_depth_delta=abs(depth_median - front) if reference_point is not None else None,
        partial=True,
        method=(
            ('mask_reference_depth_band' if mask is not None else 'bbox_reference_depth_band')
            if reference_point is not None
            else ('mask_front_depth_band' if mask is not None else 'bbox_front_depth_band')
        ),
    )


def geometry_reliable(event):
    """用深度 MAD 判断当前测量质量，避免把家具宽度当作测距噪声。

    旧记录缺少该字段时沿用 spread 校验；当前无效深度统计不能通过。"""
    if event.get('valid_points', 0) < 40:
        return False
    key = 'depth_spread' if 'depth_spread' in event else 'spread'
    value = event.get(key)
    return (
        value is not None and np.isfinite(value) and 0 <= value <= (0.12 if key == 'depth_spread' else 0.35)
    )


class Memory:
    def __init__(self, yolo_thresholds=None, voxel_size=0.10, confirm_hits=2, min_stable_voxels=20):
        self.fusion = VoxelFusion(voxel_size, confirm_hits, min_stable_voxels)
        self.version = 0
        self.objects = []
        self.observations = []
        self.next_id = 1
        self.yolo_thresholds = dict(yolo_thresholds or {})

    def evidence_ok(self, event, label):
        source = event.get('source', 'cloud')
        if source == 'cloud':
            return event['confidence'] >= 0.8
        return (
            source == 'yolo'
            and label in self.yolo_thresholds
            and event.get('method') in ('mask_front_depth_band', 'mask_arrival_front_depth_band')
            and event['confidence'] >= self.yolo_thresholds[label]
        )

    def clear_rejections(self):
        for obj in self.objects:
            obj.pop('rejected', None)
            obj.pop('deferred', None)

    def get_by_id(self, object_id):
        """按 ID 取得对象，保留拒绝或待复核状态。"""
        return next((obj for obj in self.objects if obj['id'] == object_id), None)

    def association_distance(self, obj, geometry, label):
        if obj.get('state') == 'cleared':
            return None
        new = np.asarray(geometry.get('support_points', geometry.get('points', [])), dtype=float)
        old = np.asarray(obj.get('support_points', []), dtype=float)
        distance = math.dist(obj['point'][:2], geometry['point'][:2])
        if new.ndim == 2 and old.ndim == 2 and len(new) >= 3 and len(old) >= 3:
            radius = 2 * self.fusion.resolution
            a = cKDTree(old).query(new)[0]
            b = cKDTree(new).query(old)[0]
            overlap = max(float(np.mean(a < radius)), float(np.mean(b < radius)))
            if overlap >= 0.2:
                return (1.0 - overlap) + 0.05 * distance
            return None
        # 仅有位置的假设不足以确认对象表面。
        bound = 3 * self.fusion.resolution
        anchored = math.dist(obj.get('anchor_point', obj['point'])[:2], geometry['point'][:2])
        return distance if distance < bound and anchored < 2 * bound else None

    def _refresh_surface(self, obj):
        cloud, total, stable = self.fusion.summary(obj['id'])
        had_surface = obj.get('has_surface', False) or total > 0
        obj.update(
            voxel_count=total,
            stable_voxel_count=stable,
            confirmed=stable >= self.fusion.min_stable_voxels,
            has_surface=had_surface,
            state=(
                'confirmed'
                if stable >= self.fusion.min_stable_voxels
                else 'tentative' if total or not had_surface else 'cleared'
            ),
        )
        if len(cloud):
            obj['point'] = np.mean(cloud, axis=0).tolist()
            obj['look_at'] = ((cloud.min(axis=0) + cloud.max(axis=0)) * 0.5).tolist()
            obj['extent'] = (cloud.max(axis=0) - cloud.min(axis=0)).tolist()
            obj['support_points'] = cloud[
                np.linspace(0, len(cloud) - 1, min(600, len(cloud)), dtype=int)
            ].tolist()
        else:
            obj['support_points'] = []

    def observe_depth(self, depth, k, translation, quaternion, stamp):
        count = self.fusion.clear_observed_free_space(depth, k, translation, quaternion, stamp)
        if count:
            for obj in self.objects:
                self._refresh_surface(obj)
            self.version += 1
        return count

    def defer(self, object_id, reason, stamp):
        for obj in self.objects:
            if obj['id'] == object_id:
                obj['deferred'] = dict(reason=reason, stamp=float(stamp))
                break

    def reject(self, object_id, reason, stamp):
        for obj in self.objects:
            if obj['id'] == object_id:
                obj['rejected'] = dict(reason=reason, stamp=stamp)
                break

    def add(self, label, confidence, geometry, observer, stamp, image, source='cloud', exclude_ids=()):
        if geometry is None:
            return None
        p = np.array(geometry['point'])
        matches = [
            o
            for o in self.objects
            if o['label'] == label
            and o['id'] not in exclude_ids
            and self.association_distance(o, geometry, label) is not None
        ]
        obj = min(matches, key=lambda o: self.association_distance(o, geometry, label)) if matches else None
        if obj is None:
            obj = dict(
                id=f'{label}_{self.next_id:03d}',
                label=label,
                history=[],
                point=p.tolist(),
                anchor_point=p.tolist(),
                first_stamp=float(stamp),
            )
            self.next_id += 1
            self.objects.append(obj)
        event = dict(
            **geometry,
            confidence=float(confidence),
            observer=list(observer),
            stamp=float(stamp),
            image=image,
            source=source,
        )
        obj['history'].append(event)
        if self.evidence_ok(event, label):
            self.fusion.integrate(
                obj['id'], geometry.get('support_points', geometry.get('points', [])), stamp
            )
        self.version += 1
        obj['history'] = obj['history'][-12:]
        obj['observer_positions'] = [e['observer'] for e in obj['history'][-12:]]
        obj['last_stamp'] = stamp
        obj['partial'] = True
        frames = {e['stamp']: e['confidence'] for e in obj['history'] if self.evidence_ok(e, label)}
        obj['observation_count'] = len(frames)
        obj['class_evidence'] = sum(frames.values())
        # 体素等权融合，避免反复近拍一个角落拖动对象中心或覆盖旧表面。
        self._refresh_surface(obj)
        if not obj['voxel_count']:
            obj.update(
                point=p.tolist(),
                look_at=geometry.get('look_at', p.tolist()),
                extent=geometry.get('extent', [0, 0, 0]),
            )
        if obj.get('deferred'):
            after = obj['deferred']['stamp']
            fresh = [
                e
                for e in obj['history']
                if e['stamp'] > after
                and self.evidence_ok(e, label)
                and geometry_reliable(e)
                and math.dist(e['point'][:2], obj['point'][:2]) <= 0.35
            ]
            if any(abs(a['stamp'] - b['stamp']) >= 1.0 for a in fresh for b in fresh):
                obj.pop('deferred')
        return obj

    def quality(self, obj, now, max_age=90.0):
        recent = [
            e
            for e in obj['history']
            if 0 <= now - e['stamp'] <= max_age and self.evidence_ok(e, obj['label'])
        ]
        reliable = [e for e in recent if geometry_reliable(e)]
        consistent = []
        for e in reliable:
            if math.dist(e['point'][:2], obj['point'][:2]) <= 0.35:
                if not any(abs(e['stamp'] - old['stamp']) < 1.0 for old in consistent):
                    consistent.append(e)
        reasons = []
        if obj.get('rejected'):
            reasons.append('目标类别未通过到达时的VLM复核')
        if not recent:
            reasons.append('缺少新鲜且明确的语义识别')
        if not reliable:
            reasons.append('有效深度点少于40或深度一致性不足')
        if len(consistent) < 2:
            reasons.append('不足两次时间分离且位置一致的视觉观测')
        if obj.get('voxel_count', 0):
            reasons = []
            if not obj.get('confirmed'):
                reasons.append('尚无足够多帧重合的语义体素')
            if not recent:
                reasons.append('目标暂未出现在最近观测中，保留历史地图')
            if obj.get('rejected'):
                reasons.append('目标类别待核对')
        return dict(
            reliable=not reasons,
            reasons=reasons,
            consistent_observations=len(consistent),
            position_tolerance_m=0.35,
            full_shape_required=False,
            source_thresholds=dict(cloud=0.8, yolo=self.yolo_thresholds),
        )

    def target(self, label, now=None, preferred_id=None):
        found = [
            o
            for o in self.objects
            if o['label'] == label
            and o.get('state') != 'cleared'
            and not o.get('rejected')
            and not o.get('deferred')
        ]
        preferred = next((o for o in found if o['id'] == preferred_id), None)
        if preferred is not None:
            return preferred
        if now is None:
            now = max((o['last_stamp'] for o in found), default=0.0)

        def score(obj):
            quality = self.quality(obj, now)
            strong = sum(self.evidence_ok(e, label) and geometry_reliable(e) for e in obj['history'])
            confirmed = bool(obj.get('semantic_confirmation'))
            first = obj.get('first_stamp', obj['history'][0]['stamp'])
            # 优先稳定实例；任务可用 preferred_id 持续锁定同一个对象。
            return (
                bool(obj.get('confirmed')),
                quality['reliable'],
                confirmed,
                min(quality['consistent_observations'], 3),
                min(strong, 5),
                -first,
            )

        return max(found, key=score, default=None)

    def snapshot(self, include_history=True):
        objects = (
            self.objects
            if include_history
            else [{k: v for k, v in o.items() if k != 'history'} for o in self.objects]
        )
        return dict(
            frame='odom',
            version=self.version,
            objects=objects,
            observations=self.observations[-100:] if include_history else [],
        )
