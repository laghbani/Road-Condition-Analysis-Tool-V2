"""
Smart-CSV v3 –  trennt Metadaten in JSON und exportiert nur gelabelte Topics.
"""
import json
import yaml
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# Konstanten -----------------------------------------------------------------
G = 9.80665
FSR_2G, FSR_8G = 2 * G, 8 * G

# ----------------------------------------------------------------------------

def _sha1(p: Path) -> str:
    h = hashlib.sha1(); h.update(p.read_bytes()); return h.hexdigest()

def _detect_fsr(a_abs: np.ndarray) -> float:
    return FSR_2G if np.percentile(a_abs, 99.9) < 0.9 * FSR_8G else FSR_8G


def _find_bias(df: pd.DataFrame) -> np.ndarray | None:
    mag = np.linalg.norm(df[["ax", "ay", "az"]], axis=1)
    win = max(1, int(2 / np.median(np.diff(df["time"]))))
    s = pd.Series(mag).rolling(win, center=True)
    msk = (s.std() < 0.05) & (abs(s.mean() - G) < 0.2)
    if not msk.any():
        return None
    idx = msk.idxmax(); j0 = max(0, idx - win // 2); j1 = min(len(df), idx + win // 2)
    return df.iloc[j0:j1][["ax", "ay", "az"]].mean().values


def _gravity_from_quat(Q: np.ndarray) -> np.ndarray:
    r = R.from_quat(Q)
    return r.inv().apply([0, 0, G])

# ---------------------- Fahrzeug-Frame --------------------------------------

def _vehicle_rotation(df: pd.DataFrame, gps: pd.DataFrame | None) -> list | None:
    if gps is None or len(gps) < 5:
        return None
    # Heading-Vektor
    lat, lon = np.deg2rad(gps["lat"]), np.deg2rad(gps["lon"])
    dx = np.diff(lon) * np.cos(lat[:-1])
    dy = np.diff(lat)
    h = np.array([dx.sum(), dy.sum()])
    n = np.linalg.norm(h)
    if n < 1e-6:
        return None
    h /= n
    # g-Vektor (Up)
    gvec = -(df[["ax", "ay", "az"]] - df[["ax_corr", "ay_corr", "az_corr"]]).mean().values
    gvec /= np.linalg.norm(gvec)
    # Right = h × up
    rvec = np.cross(h.tolist() + [0], gvec)
    n = np.linalg.norm(rvec)
    if n < 1e-3:
        return None
    rvec /= n
    fvec = np.cross(gvec, rvec)
    Rvs = np.vstack([fvec, rvec, gvec])  # Body→Veh
    df[["ax_veh", "ay_veh", "az_veh"]] = (Rvs @ df[["ax_corr", "ay_corr", "az_corr"]].T).T
    return Rvs.round(4).tolist()

# ---------------------------- Fitness-Analyse (GUI) -------------------------

def analyse_topic_fitness(dfs: dict, gps_df: pd.DataFrame | None):
    out = []
    for t, df in dfs.items():
        has_lbl = (df["label_id"] != 99).any()
        rot_ok = {"ax_veh", "ay_veh", "az_veh"}.issubset(df.columns)
        out.append(dict(topic=t, has_label=has_lbl, rot_ok=rot_ok))
    return out

# ---------------------------- Haupt-Export-Funktion -------------------------

def export_csv_smart_v3(self, gps_df: pd.DataFrame | None = None):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    start_dir = str(Path(self.bag_path).parent)
    dir_selected = QFileDialog.getExistingDirectory(self, "CSV/JSON speichern nach", start_dir)
    if not dir_selected:
        return
    dest = Path(dir_selected)
    bag = Path(self.bag_path).stem
    sha = _sha1(Path(__file__))

    for topic, df0 in self.dfs.items():
        if not (df0["label_id"] != 99).any():
            continue
        try:
            # 1 Rohdaten + evtl. Gyro/Quat sammeln -------------------------
            samps = self.samples[topic]
            Q = np.array([[s.msg.orientation.x, s.msg.orientation.y,
                           s.msg.orientation.z, s.msg.orientation.w] for s in samps])
            GYR = np.array([[s.msg.angular_velocity.x, s.msg.angular_velocity.y,
                             s.msg.angular_velocity.z] for s in samps])
            has_q = not np.allclose(Q, 0)
            has_g = not np.allclose(GYR, 0)

            df = df0.copy()
            if has_g:
                df[["gx", "gy", "gz"]] = GYR

            # 2 FSR + Clipping --------------------------------------------
            a_abs = np.linalg.norm(df[["ax", "ay", "az"]], axis=1)
            fsr = _detect_fsr(a_abs)
            df["clipped_flag"] = (df[["ax", "ay", "az"]].abs() >= 0.98 * fsr).any(1).astype("int8")

            # 3 g-Kompensation -------------------------------------------
            if has_q:
                g_b = _gravity_from_quat(Q)
                acc_c = df[["ax", "ay", "az"]].values - g_b
                bias_vec = None
                comp_type = "quaternion"
            else:
                bias_vec = _find_bias(df) or np.zeros(3)
                acc_c = df[["ax", "ay", "az"]].values - bias_vec
                comp_type = "static_bias"
            df[["ax_corr", "ay_corr", "az_corr"]] = acc_c

            # 4 Vehicle-Frame --------------------------------------------
            rot_mat = _vehicle_rotation(df, gps_df)

            # 5 Zusatzfeatures -------------------------------------------
            df["|a|_raw"] = a_abs
            df["|ω|_raw"] = np.linalg.norm(GYR, axis=1) if has_g else np.nan
            fs = 1 / np.median(np.diff(df["time"])) if len(df) > 1 else 0.0

            # 6 Spaltenreihenfolge ---------------------------------------
            cols = ["time", "time_abs", "ax", "ay", "az"]
            if has_g:
                cols += ["gx", "gy", "gz"]
            cols += ["ax_corr", "ay_corr", "az_corr"]
            if rot_mat:
                cols += ["ax_veh", "ay_veh", "az_veh"]
            cols += ["|a|_raw", "|ω|_raw", "clipped_flag", "label_id", "label_name"]
            df_out = df[cols]

            # 7 JSON-Metadatei -------------------------------------------
            meta = dict(
                sensor_topic=topic,
                sensor_model="Ouster" if "ouster" in topic else "ZED",
                coordinate_frame=getattr(samps[0].msg.header, "frame_id", "sensor_native"),
                file_format="imu_v1",
                sensor_fs_accel_g=round(fsr / G, 2),
                g_compensation=comp_type,
                bias_vector_mps2=None if bias_vec is None else [round(x, 3) for x in bias_vec],
                vehicle_rot_mat=rot_mat,
                sampling_rate_hz=round(fs, 2),
                exporter_sha1=sha,
            )
            stem = topic.strip("/").replace("/", "__") + f"_{bag}__imu_v1"
            (dest / f"{stem}.json").write_text(json.dumps(meta, indent=2))

            # 8 CSV -------------------------------------------------------
            df_out.to_csv(dest / f"{stem}.csv", index=False, float_format="%.6f")

        except Exception as e:
            QMessageBox.critical(self, "Export-Fehler", f"{topic}: {e}")

    QMessageBox.information(self, "Export fertig", f"CSV+JSON in: {dest}")
