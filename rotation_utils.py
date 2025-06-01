"""
rotation_utils.py – v0.7  (robuste Rotationsschätzung + Qualitätsscore)
========================================================================
· API bleibt identisch zu früher:
      rot_mat, method, g_comp = estimate_vehicle_rot(df, quat, gps_df)
· Neu:
      score, az_mean, rms_xy  stecken in einem dict «meta»,
      damit GUI & Export beide dieselben Daten sehen.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

G_STD = 9.80665

# ---------------------------------------------------------------- Butter-Low-Pass
def _lowpass(x: np.ndarray, fs: float, fc: float = 2.0):
    from math import pi
    b, a = butter(2, fc / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)

# ---------------------------------------------------------------- Hilfsfunktionen
def _quat_ok(q: np.ndarray) -> bool:
    return len(q) >= 5 and (abs(np.linalg.norm(q, axis=1) - 1) < .05).any()

# ⇢ deine bisherigen rot_between, rot_from_quat_* und rot_from_gps
#    einfach **unverändert** hier einfügen  ----------------------------


def rot_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return minimal rotation matrix mapping ``v1`` to ``v2``.

    Both vectors must be non-zero and are normalized internally.
    """
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def rot_from_gps(df: pd.DataFrame, gps_df: pd.DataFrame | None) -> list | None:
    """Calculate 3x3 rotation matrix sensor→vehicle or return ``None``.

    This GPS-based approach is used as fallback when no reliable orientation
    quaternion is present.
    """

    # --- Z-Achse (Schwerkraft) -----------------------------------------
    if not {"ax_corr", "ay_corr", "az_corr"}.issubset(df.columns):
        return None

    a = df[["ax_corr", "ay_corr", "az_corr"]].to_numpy()
    mag = np.linalg.norm(a, axis=1)
    # Kandidaten: Fenster mit kleiner Varianz UND Betrag nahe g
    w = 200                         # ~2 s bei 100 Hz
    var = pd.Series(mag).rolling(w, center=True).var().to_numpy()
    cand = (var < 0.02) & (np.abs(mag - 9.81) < 0.2)
    if not cand.any():
        return None                 # kein Ruhesegment → lieber abbrechen
    g_sens = -a[cand].mean(axis=0)
    if np.linalg.norm(g_sens) < 1:
        return None
    z_sens = g_sens / np.linalg.norm(g_sens)
    R_z = rot_between(z_sens, np.array([0, 0, 1]))

    # --- X-axis via GPS track ------------------------------------------
    if gps_df is None or len(gps_df) < 2:
        return None
    lat = np.deg2rad(gps_df["lat"].to_numpy())
    lon = np.deg2rad(gps_df["lon"].to_numpy())
    dx = np.diff(lon) * np.cos(lat[:-1])
    dy = np.diff(lat)
    # 5-s-Fenster mitteln → weniger Zick-Zack
    win = max(3, int(5 / np.median(np.diff(gps_df["time"]))))
    vx = pd.Series(dx).rolling(win).mean().dropna().to_numpy()
    vy = pd.Series(dy).rolling(win).mean().dropna().to_numpy()
    if len(vx) < 20:
        return None
    # PCA – Hauptachse wählen
    M = np.cov(np.vstack((vx, vy)))
    eigval, eigvec = np.linalg.eig(M)
    x_world = eigvec[:, np.argmax(eigval)]

    # candidates after R_z rotation
    cands = [
        R_z @ np.array([1, 0, 0]),
        R_z @ np.array([0, 1, 0]),
        R_z @ np.array([-1, 0, 0]),
        R_z @ np.array([0, -1, 0]),
    ]
    dots = [np.dot(c[:2], x_world) for c in cands]
    x_sens = cands[int(np.argmax(np.abs(dots)))]
    if dots[np.argmax(np.abs(dots))] < 0:
        x_sens = -x_sens
    y_sens = np.cross(np.array([0, 0, 1]), x_sens)
    y_sens /= np.linalg.norm(y_sens)
    R_sv = np.stack([x_sens, y_sens, np.array([0, 0, 1])]).T @ R_z
    return [[round(c, 3) for c in row] for row in R_sv]


def z_axis_from_quat(q: np.ndarray) -> np.ndarray:
    """Return average ``-g`` vector in sensor frame from quaternion series."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return -R.from_quat(q).apply([0, 0, G_STD]).mean(axis=0)


def rot_from_quat_absolute(q: np.ndarray) -> list | None:
    """Sensor→Vehicle-Rotation nur aus dem Quaternion (keine Bewegung, kein GPS nötig).

    Annahmen:
      • z_vehicle = +z_world  (Up)
      • x_vehicle zeigt nach vorne ≈ x_world, so gut es geht
    """
    if len(q) < 5:
        return None
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    Rw_s = R.from_quat(q)

    z_s = -Rw_s.apply([0, 0, G_STD]).mean(axis=0)
    z_s /= np.linalg.norm(z_s)
    R_z = rot_between(z_s, [0, 0, 1])

    x_w = Rw_s.apply([1, 0, 0]).mean(axis=0)
    x_w[:2] /= np.linalg.norm(x_w[:2])

    cands = [R_z @ np.array(v) for v in ([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0])]
    dots = [np.dot(c[:2], x_w[:2]) for c in cands]
    idx = int(np.argmax(np.abs(dots)))
    x_s = cands[idx]
    if dots[idx] < 0:
        x_s = -x_s
    y_s = np.cross([0, 0, 1], x_s)
    y_s /= np.linalg.norm(y_s)
    return np.round(np.stack([x_s, y_s, [0, 0, 1]]).T @ R_z, 3).tolist()


def rot_from_quat_static(q: np.ndarray,
                         gps_df: pd.DataFrame | None) -> list | None:
    z_sens = z_axis_from_quat(q)
    if np.linalg.norm(z_sens) < 1:
        return None
    R_z = rot_between(z_sens / np.linalg.norm(z_sens), [0, 0, 1])

    if gps_df is None or len(gps_df) < 3:
        return None
    lat = np.deg2rad(gps_df["lat"].to_numpy())
    lon = np.deg2rad(gps_df["lon"].to_numpy())
    dx = np.diff(lon) * np.cos(lat[:-1])
    dy = np.diff(lat)
    win = max(3, int(5 / np.median(np.diff(gps_df["time"]))))
    vx = pd.Series(dx).rolling(win).mean().dropna().to_numpy()
    vy = pd.Series(dy).rolling(win).mean().dropna().to_numpy()
    if len(vx) < 20:
        return None
    M = np.cov(np.vstack((vx, vy)))
    eigval, eigvec = np.linalg.eig(M)
    x_world = eigvec[:, np.argmax(eigval)]
    cands = [
        R_z @ np.array([1, 0, 0]),
        R_z @ np.array([0, 1, 0]),
        R_z @ np.array([-1, 0, 0]),
        R_z @ np.array([0, -1, 0]),
    ]
    dots = [np.dot(c[:2], x_world) for c in cands]
    x_sens = cands[int(np.argmax(np.abs(dots)))]
    if dots[np.argmax(np.abs(dots))] < 0:
        x_sens = -x_sens
    y_sens = np.cross([0, 0, 1], x_sens)
    y_sens /= np.linalg.norm(y_sens)
    R_sv = np.stack([x_sens, y_sens, [0, 0, 1]]).T @ R_z
    return [[round(c, 3) for c in row] for row in R_sv]


def rot_from_quat_dynamic(q: np.ndarray) -> list | None:
    z_sens = z_axis_from_quat(q)
    if np.linalg.norm(z_sens) < 1:
        return None
    R_z = rot_between(z_sens / np.linalg.norm(z_sens), [0, 0, 1])

    x_w = R.from_quat(q).apply([1, 0, 0]).mean(axis=0)
    x_w[:2] /= np.linalg.norm(x_w[:2])
    cands = [
        R_z @ np.array([1, 0, 0]),
        R_z @ np.array([0, 1, 0]),
        R_z @ np.array([-1, 0, 0]),
        R_z @ np.array([0, -1, 0]),
    ]
    dots = [np.dot(c[:2], x_w[:2]) for c in cands]
    x_sens = cands[int(np.argmax(np.abs(dots)))]
    if dots[np.argmax(np.abs(dots))] < 0:
        x_sens = -x_sens
    y_sens = np.cross([0, 0, 1], x_sens)
    y_sens /= np.linalg.norm(y_sens)
    R_sv = np.stack([x_sens, y_sens, [0, 0, 1]]).T @ R_z
    return [[round(c, 3) for c in row] for row in R_sv]


# ---------------------------------------------------------------- Hauptroutine
def estimate_vehicle_rot(df: pd.DataFrame,
                         ori_q: np.ndarray,
                         gps_df: pd.DataFrame | None):
    """
    Liefert (R_sv | None, method:str, g_comp:str, meta:dict)
    meta = {"score":0-100, "az_mean":…, "rms_xy":…}
    """
    if len(df) < 3:
        return None, "", "", {"score": 0, "az_mean": np.nan, "rms_xy": np.nan}

    # 1) g-Kompensation ---------------------------------------------------
    if _quat_ok(ori_q):
        g_vec   = R.from_quat(ori_q).apply([0, 0, -G_STD])
        acc_corr = df[["ax", "ay", "az"]].to_numpy() - g_vec
        g_comp = "quaternion"
    else:
        bias    = df[["ax", "ay", "az"]].rolling(200, center=True).median()
        bias    = bias[(bias.abs().sum(1) < 1).fillna(False)].mean().to_numpy()
        acc_corr = df[["ax", "ay", "az"]].to_numpy() - bias
        g_comp  = "static_bias"

    fs = 1 / np.median(np.diff(df["time"])) if len(df) > 1 else 0
    df[["ax_corr", "ay_corr", "az_corr"]] = _lowpass(acc_corr, fs) if fs else acc_corr

    # 2) Rotation ---------------------------------------------------------
    method = ""
    R_sv   = None
    if g_comp == "quaternion":
        R_sv = rot_from_quat_absolute(ori_q);    method = "quat_abs"
        if R_sv is None and rot_from_quat_dynamic is not None:
            R_sv = rot_from_quat_dynamic(ori_q); method = "quat_dyn"
        if R_sv is None:
            R_sv = rot_from_quat_static(ori_q, gps_df); method = "quat_gps"
    if R_sv is None:
        R_sv = rot_from_gps(df, gps_df);         method = "gps_only"

    # 3) Score ------------------------------------------------------------
    if R_sv is not None:
        R_arr   = np.array(R_sv)
        veh     = df[["ax","ay","az"]].to_numpy() @ R_arr.T
        az_m    = veh[:,2].mean()
        rms_xy  = np.sqrt((veh[:,0]**2 + veh[:,1]**2).mean())
        score   = 100*np.exp(-((az_m-G_STD)**2)/.5)*np.exp(-rms_xy/.5)
    else:
        az_m = rms_xy = np.nan; score = 0

    meta = {"score": round(float(score),1),
            "az_mean": round(float(az_m),2),
            "rms_xy":  round(float(rms_xy),2)}
    return R_sv, method, g_comp, meta
