import hashlib
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from typing import Tuple


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


def add_speed(work: pd.DataFrame, gps_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``speed_mps`` column to *work* derived from ``gps_df``."""
    if gps_df is None or len(gps_df) < 3:
        work["speed_mps"] = np.nan
        return work
    R_earth = 6_378_137.0
    lat0 = np.deg2rad(gps_df["lat"].iat[0])
    dx = (
        R_earth
        * np.deg2rad(gps_df["lon"] - gps_df["lon"].iat[0])
        * np.cos(lat0)
    )
    dy = R_earth * np.deg2rad(gps_df["lat"] - gps_df["lat"].iat[0])
    dt = np.diff(gps_df["time"])
    v = np.hypot(np.diff(dx), np.diff(dy)) / dt
    g2 = gps_df.iloc[1:].copy()
    g2["speed_mps"] = v
    out = pd.merge_asof(
        work.sort_values("time_abs"),
        g2[["time", "speed_mps"]].rename(columns={"time": "time_abs"}),
        on="time_abs",
        direction="nearest",
    )
    out["speed_mps"] = out["speed_mps"].interpolate(limit_direction="both")
    return out


G_STD = 9.80665
FSR_2G = 2 * G_STD
FSR_8G = 8 * G_STD
CAND_G = np.array([2, 4, 8, 16]) * G_STD        #  19.6, 39.2 … 156.9 m/s²


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_fsr(abs_a: np.ndarray) -> float:
    """
    Versucht die Full-Scale-Range ausschließlich aus den Daten abzuleiten.
    Idee:  Bei einem saturierten Sensor häufen sich Samples in der Nähe des
           Maximalwerts.  Wir zählen, wie viele Messpunkte ≥ 98 % jedes
           Kandidaten liegen, und wählen den kleinsten Kandidaten, der
           »signifikant« getroffen wird.
    """
    hits = [(c, np.sum(abs_a >= 0.98 * c)) for c in CAND_G]
    # 0.02 % der Gesamtlänge als Faustregel
    thr = 0.0002 * len(abs_a)
    for c, cnt in hits:
        if cnt > thr:
            return c
    # Fallback –  heuristisch wie bisher
    vmax = np.nanpercentile(abs_a, 99.9)
    return CAND_G[0] if vmax < 22 else CAND_G[2]   # ±2 g oder ±8 g


def find_stationary_bias(df: pd.DataFrame,
                         win_s: float = 2.0,
                         thr_sigma: float = 0.05) -> np.ndarray | None:
    if len(df) < 2:
        return None
    dt = np.median(np.diff(df["time"]))
    fs = 1.0 / dt if dt > 0 else 1.0
    win = max(1, int(win_s * fs))
    mag = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(), axis=1)
    ser = pd.Series(mag)
    std = ser.rolling(win, center=True).std()
    mean = ser.rolling(win, center=True).mean()
    mask = (std < thr_sigma) & (np.abs(mean - G_STD) < 0.2)
    if mask.any():
        idx = mask.idxmax()
        j0 = max(0, idx - win // 2)
        j1 = min(len(df), idx + win // 2 + 1)
        return df.iloc[j0:j1][["ax", "ay", "az"]].mean().to_numpy()
    return None


def remove_gravity_lowpass(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate gravity via low-pass filter and remove it from accelerations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``ax``, ``ay`` and ``az`` columns.

    Returns
    -------
    tuple
        ``(acc_corr, g_est, bias_vec)`` where ``acc_corr`` is the gravity
        compensated acceleration array (``N×3``), ``g_est`` is the estimated
        gravity vector per sample and ``bias_vec`` is the stationary bias
        estimate used for compensation.
    """
    bias_vec = find_stationary_bias(df)
    if bias_vec is None:
        bias_vec = np.zeros(3)
    acc_bias = df[["ax", "ay", "az"]].to_numpy() - bias_vec

    # --- LOW-PASS → Gravitation schätzen -------------------------------
    dt = np.median(np.diff(df["time"]))
    fs = 1.0 / dt if dt > 0 else 100.0
    fc = 0.3  # Cut-off 0.3 Hz ≈ 5 s
    b, a = butter(2, fc / (0.5 * fs), btype="low")
    g_est = filtfilt(b, a, acc_bias, axis=0)

    # --- Norm-Scaling auf 1 g -----------------------------------------
    scale = 9.80665 / np.median(np.linalg.norm(g_est, axis=1))
    g_est *= scale

    acc_corr = acc_bias - g_est
    return acc_corr, g_est, bias_vec


def gravity_from_quat(df: pd.DataFrame) -> np.ndarray:
    q = df[["ox", "oy", "oz", "ow"]].to_numpy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rots = R.from_quat(q)
    g_world = np.array([0.0, 0.0, G_STD])
    return rots.inv().apply(g_world)


# -------- Hilfs-Wrapper ----------------------------------------------------
def compensate_gravity(df: pd.DataFrame, ori: np.ndarray | None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Gravity compensation via quaternion or low-pass.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``ax``, ``ay`` and ``az`` columns.
    ori : np.ndarray | None
        ``N×4`` array of quaternions (x, y, z, w). If ``None`` the low-pass
        approach is used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, str]
        ``(acc_corr, g_vec, bias_vec, comp_type)`` where ``acc_corr`` is the
        gravity compensated acceleration, ``g_vec`` is the estimated gravity per
        sample, ``bias_vec`` is the bias vector used (or ``None``) and
        ``comp_type`` denotes the method used.
    """
    if ori is not None:
        g_vec = gravity_from_quat(
            pd.DataFrame(ori, columns=["ox", "oy", "oz", "ow"]))
        return df[["ax", "ay", "az"]].to_numpy() - g_vec, g_vec, None, "quaternion"

    acc_corr, g_est, bias_vec = remove_gravity_lowpass(df)
    return acc_corr, g_est, bias_vec, "lowpass_bias"


def get_sensor_model(bag_root: Path, topic: str) -> str:
    meta_path = bag_root / "metadata.yaml"
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text())
            for k in ("device_type", "product"):
                if k in meta:
                    return str(meta[k])
        except Exception:
            pass
    tl = topic.lower()
    if "ouster" in tl:
        return "Ouster OS-Series"
    if "zed" in tl:
        return "StereoLabs ZED 2/2i"
    return "unknown"


def first_frame_id(samples: list) -> str:
    for s in samples:
        fid = getattr(getattr(s.msg, "header", None), "frame_id", "")
        if fid:
            return fid
    return "sensor_native"


def auto_vehicle_frame(df: pd.DataFrame, gps_df: pd.DataFrame | None) -> list | None:
    """Calculate 3x3 rotation matrix sensor→vehicle or return ``None``."""

    # ---------- Z-Achse -------------------------------------------------
    if {"g_x", "g_y", "g_z"}.issubset(df.columns):
        z_sens = df[["g_x", "g_y", "g_z"]].mean().to_numpy()
    elif {"ax", "ay", "az"}.issubset(df.columns):
        z_sens = df[["ax", "ay", "az"]].mean().to_numpy()
    else:
        return None
    if np.linalg.norm(z_sens) < 3:
        return None
    z_sens /= np.linalg.norm(z_sens)
    R_z = rot_between(z_sens, np.array([0, 0, 1]))

    # --- X-axis via GPS track ------------------------------------------
    if gps_df is None or len(gps_df) < 2:
        return None
    lat = np.deg2rad(gps_df["lat"].to_numpy())
    lon = np.deg2rad(gps_df["lon"].to_numpy())
    dx = np.diff(lon) * np.cos(lat[:-1])
    dy = np.diff(lat)
    track = np.stack([dx, dy]).mean(axis=1)
    if np.allclose(track, 0):
        return None
    x_world = track / np.linalg.norm(track)

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


def export_csv_smart_v2(self, gps_df: pd.DataFrame | None = None) -> None:
    bag_root = Path(self.bag_path)
    from PyQt5.QtWidgets import QFileDialog

    folder = QFileDialog.getExistingDirectory(
        self,
        "Ziel-Ordner für CSV/JSON wählen",
        str(Path(self.bag_path).parent),
    )
    if not folder:          # Dialog abgebrochen
        return
    dest = Path(folder)
    self.last_export_dir = str(dest)   # für den GUI-Check
    exporter_sha = sha1_of_file(Path(__file__))
    for topic, df in self.dfs.items():
        if not (df["label_id"] != 99).any():
            continue  # nichts gelabelt → kein Export
        try:
            samps = self.samples[topic]

            ori = np.array(
                [
                    [s.msg.orientation.x, s.msg.orientation.y,
                     s.msg.orientation.z, s.msg.orientation.w]
                    for s in samps
                ]
            )
            gyro = np.array(
                [
                    [s.msg.angular_velocity.x, s.msg.angular_velocity.y,
                     s.msg.angular_velocity.z]
                    for s in samps
                ]
            )
            norm_ok  = np.abs(np.linalg.norm(ori, axis=1) - 1.0) < 0.05
            var_ok   = np.ptp(ori, axis=0).max() > 1e-3           # bewegt sich etwas?
            has_quat = bool(norm_ok.any() and var_ok)
            has_gyro = not np.allclose(gyro, 0.0)

            work = df.copy()
            if has_gyro:
                work["gx"] = gyro[:, 0]
                work["gy"] = gyro[:, 1]
                work["gz"] = gyro[:, 2]

            abs_a = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(), axis=1)
            fsr = detect_fsr(abs_a)
            clipped = (df[["ax", "ay", "az"]].abs() >= 0.98 * fsr).any(axis=1)

            ori_for_g = ori if has_quat else None
            acc_corr, g_vec, bias_vec, comp_type = compensate_gravity(df, ori_for_g)

            work[["ax_corr", "ay_corr", "az_corr"]] = acc_corr
            work[["g_x", "g_y", "g_z"]] = g_vec

            # Rotation & Beschleunigung transformieren
            rot_mat = auto_vehicle_frame(work, gps_df)
            rot_avail = bool(rot_mat)
            if rot_avail:
                R = np.array(rot_mat)
                veh = acc_corr @ R.T
                work["ax_veh"], work["ay_veh"], work["az_veh"] = veh.T

            if has_gyro:
                abs_w = np.linalg.norm(gyro, axis=1)
            else:
                abs_w = np.full(len(work), np.nan)

            work["|a|_raw"] = abs_a
            work["|ω|_raw"] = abs_w
            work["clipped_flag"] = clipped.astype(int)

            if len(work["time"]) > 1:
                fs = 1.0 / np.median(np.diff(work["time"]))
            else:
                fs = 0.0

            work = add_speed(work, gps_df)

            header = {
                "file_format": "imu_v1",
                "sensor_topic": topic,
                "sensor_model": get_sensor_model(bag_root, topic),
                "coordinate_frame": first_frame_id(samps),
                "sensor_fs_accel_g": round(fsr / G_STD, 3),
                "bias_vector_mps2": [round(x, 3) for x in bias_vec] if bias_vec is not None else None,
                "g_compensation": comp_type,
                "vehicle_rot_mat": rot_mat,
                "rotation_available": rot_avail,
                "sampling_rate_hz": round(fs, 2),
                "exporter_sha1": exporter_sha,
            }

            cols = ["time", "time_abs", "ax", "ay", "az", "speed_mps"]
            if has_gyro:
                cols += ["gx", "gy", "gz"]
            cols += ["ax_corr", "ay_corr", "az_corr"]
            if rot_mat is not None:
                cols += ["ax_veh", "ay_veh", "az_veh"]
            cols += ["|a|_raw", "|ω|_raw", "clipped_flag", "label_id", "label_name"]

            out_df = work[cols]

            stem = f"{topic.strip('/').replace('/', '__')}_{bag_root.stem}__imu_v1"
            csv_path = dest / f"{stem}.csv"
            meta_path = dest / f"{stem}.json"

            # --- DEBUG: prüfen, ob g sauber entfernt wurde -------------------
            abs_a_corr = np.linalg.norm(acc_corr, axis=1)
            print(
                f"[{topic}]  |a|_raw μ={abs_a.mean():.2f}  σ={abs_a.std():.2f}  "
                f"|a|_corr μ={abs_a_corr.mean():.2f}  σ={abs_a_corr.std():.2f}"
            )

            meta_path.write_text(json.dumps(header, indent=2))
            out_df.to_csv(csv_path, index=False)

        except Exception as exc:
            QMessageBox = getattr(__import__("PySide6.QtWidgets", fromlist=["QMessageBox"]), "QMessageBox", None)
            if QMessageBox is None:
                from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export-Fehler", f"{topic}: {exc}", QMessageBox.Ok)
            continue

    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(f"Export abgeschlossen → {dest}")

    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.information(self, "Export fertig",
                            f"CSV + JSON liegen in:\n{dest}")
