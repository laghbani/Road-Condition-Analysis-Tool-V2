from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
try:
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    PointCloud2 = None  # type: ignore


def _pc_to_xyz(pc_msg: PointCloud2, step: int = 1) -> np.ndarray:
    rec = np.fromiter(
        pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        count=-1 if step == 1 else None,
    )
    if step > 1:
        rec = rec[::step]
    pts = np.column_stack((rec["x"], rec["y"], rec["z"])).astype(np.float32, copy=False)
    return pts


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    uu, ss, vv = np.linalg.svd(points - centroid)
    normal = vv[2]
    d = -float(normal.dot(centroid))
    return normal, d


def _rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999:
        return -np.eye(3)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R


class MapBuilder(QThread):
    result = pyqtSignal(object, object)
    progress = pyqtSignal(int, int)

    def __init__(self, pc_arrays: list, times: list[float] | None,
                 pre: float, post: float, current_time: float, voxel: float = 0.05) -> None:
        super().__init__()
        self.pc_arrays = pc_arrays
        self.times = times or []
        self.pre = pre
        self.post = post
        self.cur_time = current_time
        self.voxel = voxel

    def _iter_points(self, raw) -> np.ndarray:
        if PointCloud2 is not None and isinstance(raw, PointCloud2):
            return _pc_to_xyz(raw)
        return np.asarray(raw, dtype=np.float32)

    def run(self) -> None:
        if not self.pc_arrays:
            return
        if self.times:
            idxs = [i for i, t in enumerate(self.times)
                    if self.cur_time - self.pre <= t <= self.cur_time + self.post]
        else:
            idxs = list(range(len(self.pc_arrays)))
        n_scans = len(idxs)
        if n_scans == 0:
            return
        voxels: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        for step, i in enumerate(idxs, 1):
            pts = self._iter_points(self.pc_arrays[i])
            if pts.size == 0:
                self.progress.emit(step, n_scans)
                continue
            if pts.shape[0] > 50000:
                sel = np.random.choice(pts.shape[0], 50000, replace=False)
                pts = pts[sel]
            z = pts[:, 2]
            mask = z <= np.quantile(z, 0.3)
            if mask.sum() >= 3:
                normal, d = _fit_plane_svd(pts[mask])
                R = _rotation_matrix(normal, np.array([0.0, 0.0, 1.0]))
                pts = (R @ pts.T).T
                height = -d / np.linalg.norm(normal)
                pts[:, 2] += height
            idx_vox = np.floor(pts / self.voxel).astype(np.int32)
            for idx_tuple, p in zip(map(tuple, idx_vox), pts):
                if idx_tuple in voxels:
                    s, c = voxels[idx_tuple]
                    voxels[idx_tuple] = (s + p, c + 1)
                else:
                    voxels[idx_tuple] = (p.copy(), 1)
            self.progress.emit(step, n_scans)
        min_hits = max(1, int(0.6 * n_scans))
        out_pts = []
        out_cols = []
        for s, c in voxels.values():
            if c >= min_hits:
                pt = s / c
                out_pts.append(pt)
                frac = c / n_scans
                out_cols.append([1 - frac, frac, 0.0, 1.0])
        if out_pts:
            pts = np.vstack(out_pts).astype(np.float32)
            cols = np.vstack(out_cols).astype(np.float32)
            self.result.emit(pts, cols)

