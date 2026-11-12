"""由 odom 中的家具支持点生成保守 XY 占用凸包，避免停到桌面下方。

凸包只来自已见表面，不代表完整物体重建。"""

import cv2
import numpy as np
from .semantic import geometry_reliable


def furniture_regions(objects, evidence_ok):
    regions = []
    for obj in objects:
        good = [e for e in obj.get('history', []) if evidence_ok(e, obj['label']) and geometry_reliable(e)]
        if 'confirmed' in obj:
            if not obj['confirmed'] or obj.get('state') == 'cleared':
                continue
            points = obj.get('support_points', [])
        else:
            points = [p for e in good[-5:] for p in e.get('support_points', e.get('points', []))]
        try:
            xyz = np.asarray(points, dtype=float)
        except (ValueError, TypeError):
            continue
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 20:
            continue
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.15) & (xyz[:, 2] < 1.8)]
        if len(xyz) < 20:
            continue
        # 裁去少量孤立极端点；机器人本体外扩由后续参数单独处理。
        low, high = np.quantile(xyz[:, :2], [0.03, 0.97], axis=0)
        xy = xyz[np.all((xyz[:, :2] >= low - 0.04) & (xyz[:, :2] <= high + 0.04), axis=1), :2]
        if len(xy) < 20:
            continue
        hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
        regions.append(
            dict(
                object_id=obj['id'],
                label=obj['label'],
                vertices=hull,
                observations=len(good),
                source_points=len(xy),
                last_stamp=obj.get('last_stamp', max([e['stamp'] for e in good] or [0.0])),
            )
        )
    return regions


def signed_distance(vertices, point):
    hull = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    if len(hull) == 0:
        return -float('inf')
    if len(hull) == 1:
        return -float(np.linalg.norm(np.asarray(point) - hull[0]))
    if len(hull) == 2:
        a, b = hull
        edge = b - a
        scale = np.dot(edge, edge)
        f = np.clip(np.dot(np.asarray(point) - a, edge) / scale, 0, 1) if scale > 1e-10 else 0.0
        return -float(np.linalg.norm(np.asarray(point) - (a + f * edge)))
    return float(cv2.pointPolygonTest(hull.reshape(-1, 1, 2), (float(point[0]), float(point[1])), True))


def occupied_by(regions, points, clearance):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        return dict(reason='nonfinite_position')
    for region in regions:
        for index, point in enumerate(points):
            distance = signed_distance(region['vertices'], point)
            if distance >= -clearance:
                return dict(
                    object_id=region['object_id'],
                    label=region['label'],
                    point_index=index,
                    point_odom=point.tolist(),
                    signed_distance_m=distance,
                    required_clearance_m=clearance,
                )
    return None
