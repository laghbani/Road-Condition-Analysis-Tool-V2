import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from imu_csv_export_v2 import (
    gravity_from_quat,
    find_stationary_bias,
    lowpass,
    rot_from_quat_absolute,
    rot_from_quat_dynamic,
    rot_from_quat_static,
    rot_from_gps,
)


def estimate_vehicle_rot(df: pd.DataFrame, ori_quat: np.ndarray, gps_df: pd.DataFrame | None):
    """Estimate sensor→vehicle rotation matrix.

    This mirrors the logic used during CSV export but omits any
    file-writing side effects. ``df`` is modified in-place to add the
    ``ax_corr``, ``ay_corr`` and ``az_corr`` columns required by the
    rotation helpers.
    """
    if len(df) == 0 or len(ori_quat) != len(df):
        return None

    norm_ok = np.abs(np.linalg.norm(ori_quat, axis=1) - 1.0) < 0.05
    has_quat = bool(norm_ok.any())
    if has_quat:
        rotvec = R.from_quat(ori_quat[norm_ok]).as_rotvec()
        var_ok = (np.ptp(rotvec, axis=0) > 0.005).all()
    else:
        var_ok = False

    if has_quat:
        g_vec = gravity_from_quat(
            pd.DataFrame(ori_quat, columns=["ox", "oy", "oz", "ow"])
        )
        acc_corr = df[["ax", "ay", "az"]].to_numpy() - g_vec
        bias_vec = None
        comp_type = "quaternion"
    else:
        bias_vec = find_stationary_bias(df)
        if bias_vec is None:
            bias_vec = np.zeros(3)
        acc_corr = df[["ax", "ay", "az"]].to_numpy() - bias_vec
        comp_type = "static_bias"

    if len(df["time"]) > 1:
        fs = 1.0 / np.median(np.diff(df["time"]))
    else:
        fs = 0.0
    smooth = lowpass(acc_corr, fs) if fs > 0 else acc_corr
    df[["ax_corr", "ay_corr", "az_corr"]] = smooth

    if comp_type == "quaternion":
        rot_mat = rot_from_quat_absolute(ori_quat[norm_ok]) if norm_ok.any() else None
        if rot_mat is None and var_ok:
            rot_mat = rot_from_quat_dynamic(ori_quat[norm_ok])
        if rot_mat is None:
            rot_mat = rot_from_quat_static(ori_quat[norm_ok], gps_df)
    else:
        rot_mat = rot_from_gps(df, gps_df)

    return rot_mat
