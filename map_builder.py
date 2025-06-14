"""Utilities for building a combined LIDAR map from point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sensor_msgs.msg import PointCloud2

from videopc_widget import _pc_to_xyz
from imu_csv_export_v2 import _save_pcd


def filter_points(
    pts: np.ndarray,
    *,
    min_z: float | None = None,
    max_z: float | None = None,
    max_dist: float | None = None,
) -> np.ndarray:
    """Filter points by z-range and distance."""
    if pts.size == 0:
        return pts
    mask = np.ones(len(pts), dtype=bool)
    if min_z is not None:
        mask &= pts[:, 2] >= min_z
    if max_z is not None:
        mask &= pts[:, 2] <= max_z
    if max_dist is not None:
        dist = np.linalg.norm(pts, axis=1)
        mask &= dist <= max_dist
    return pts[mask]


def build_map(
    pcs: Sequence[PointCloud2 | np.ndarray],
    *,
    step: int = 1,
    min_z: float | None = None,
    max_z: float | None = None,
    max_dist: float | None = None,
) -> np.ndarray:
    """Return a single point cloud containing filtered points from *pcs*."""
    out: list[np.ndarray] = []
    for pc in pcs:
        if isinstance(pc, PointCloud2):
            pts = _pc_to_xyz(pc, step)
        else:
            pts = np.asarray(pc, dtype=np.float32)
        pts = filter_points(pts, min_z=min_z, max_z=max_z, max_dist=max_dist)
        if pts.size:
            out.append(pts)
    if out:
        return np.vstack(out)
    return np.empty((0, 3), np.float32)


def save_map(path: Path, pts: np.ndarray) -> None:
    """Save *pts* as a PCD file."""
    _save_pcd(path, pts)
