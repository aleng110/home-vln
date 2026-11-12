"""多帧体素投票融合可见表面。"""

import numpy as np
from scipy.spatial.transform import Rotation


class VoxelFusion:
    def __init__(self, resolution=0.10, confirm_hits=2, min_stable_voxels=20):
        self.resolution = float(resolution)
        self.confirm_hits = int(confirm_hits)
        self.min_stable_voxels = int(min_stable_voxels)
        self.tracks = {}
        if self.resolution <= 0 or self.confirm_hits < 2 or self.min_stable_voxels < 3:
            raise ValueError('Invalid semantic fusion parameters')

    def integrate(self, object_id, points, stamp):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1:] != (3,):
            return
        p = p[np.isfinite(p).all(axis=1)]
        if not len(p):
            return
        keys = np.floor(p / self.resolution).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        cells = self.tracks.setdefault(object_id, {})
        for i, key in enumerate(unique):
            key = tuple(key.tolist())
            point = np.mean(p[inverse == i], axis=0)
            old = cells.get(key)
            if old is not None and stamp <= old['stamp']:
                continue
            if old is None:
                cells[key] = dict(point=point, hits=1, misses=0, stamp=float(stamp), negative_stamp=-1.0)
            else:
                weight = min(old['hits'], 5)
                old.update(
                    point=(old['point'] * weight + point) / (weight + 1),
                    hits=min(old['hits'] + 1, 8),
                    misses=0,
                    stamp=float(stamp),
                )

    def summary(self, object_id):
        cells = list(self.tracks.get(object_id, {}).values())
        stable = [v for v in cells if v['hits'] >= self.confirm_hits]
        points = np.asarray(
            [v['point'] for v in (stable if len(stable) >= self.min_stable_voxels else cells)], dtype=float
        ).reshape(-1, 3)
        return points, len(cells), len(stable)

    def clear_observed_free_space(self, depth, k, translation, quaternion, stamp):
        """累计三次有效空闲反证后清除体素；相近深度重置计数，遮挡不计反证。"""
        removed = 0
        rotation = Rotation.from_quat(quaternion)
        origin = np.asarray(translation)
        h, w = depth.shape
        uncertainty = 3 * self.resolution
        for cells in self.tracks.values():
            keys = list(cells)
            if not keys:
                continue
            camera = rotation.inv().apply(np.asarray([cells[q]['point'] for q in keys]) - origin)
            z = camera[:, 2]
            front = z > 0.5
            u = np.rint(k[0] * camera[:, 0] / np.maximum(z, 0.001) + k[2]).astype(int)
            v = np.rint(k[4] * camera[:, 1] / np.maximum(z, 0.001) + k[5]).astype(int)
            valid = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            for i in np.flatnonzero(valid):
                measured = float(depth[v[i], u[i]])
                cell = cells[keys[i]]
                if stamp <= max(cell['stamp'], cell['negative_stamp']):
                    continue
                if not np.isfinite(measured) or not 0.5 < measured < 5.8:
                    continue
                if measured > z[i] + uncertainty:
                    cell['misses'] += 1
                    cell['negative_stamp'] = float(stamp)
                    if cell['misses'] >= 3:
                        del cells[keys[i]]
                        removed += 1
                elif abs(measured - z[i]) <= uncertainty:
                    cell['misses'] = 0
        return removed
